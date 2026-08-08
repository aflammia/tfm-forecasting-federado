#!/usr/bin/env bash
# 01_provision.sh — crea TODA la infraestructura del simulacro en Azure (idempotente en lo posible).
#   RG + VNet + NSG + 4 VMs (cloud-init) + workspace Azure ML + managed identity + auto-shutdown.
#
# Prerrequisitos: az CLI instalado, `az login` hecho, AZURE_SUBSCRIPTION_ID exportado.
# Uso:  bash infra/01_provision.sh
# Coste: a partir de aquí las 4 VMs están ENCENDIDAS. Recuerda infra/deallocate.sh al terminar.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/config.sh"

az account set --subscription "$AZURE_SUBSCRIPTION_ID"
az extension add -n ml -y 2>/dev/null || az extension update -n ml

# spaincentral (region nueva) tarda a veces varios segundos en hacer consultable un recurso recien
# creado -> ResourceNotFound transitorio en el comando siguiente. `reintentar` repite solo ese tipo
# de fallo (no los errores reales, que se propagan) hasta 8 veces con 8s de espera.
reintentar() {
  local i
  for ((i=1; i<=8; i++)); do
    if "$@" 2>/tmp/reintentar_err; then return 0; fi
    if grep -qiE "ResourceNotFound|was not found|does not exist|NotFound" /tmp/reintentar_err; then
      echo "  (reintento $i: recurso aun no consultable, espero 8s...)"; sleep 8
    else
      cat /tmp/reintentar_err >&2; return 1   # error real, no transitorio -> abortar
    fi
  done
  cat /tmp/reintentar_err >&2; return 1
}

echo "== 1/7 Resource group =="
az group create -n "$RG" -l "$LOCATION" -o none

echo "== 2/7 Red (VNet + subred) =="
az network vnet create -g "$RG" -n "$VNET" --address-prefix 10.0.0.0/16 \
  --subnet-name "$SUBNET" --subnet-prefix 10.0.0.0/24 -o none
reintentar az network vnet show -g "$RG" -n "$VNET" -o none   # esperar a que sea consultable

echo "== 3/7 NSG (SSH desde tu IP; 9092/9093 dentro de la subred) =="
az network nsg create -g "$RG" -n "$NSG" -o none
reintentar az network nsg show -g "$RG" -n "$NSG" -o none     # esperar a que sea consultable
MI_IP="$(curl -s https://api.ipify.org)"
reintentar az network nsg rule create -g "$RG" --nsg-name "$NSG" -n allow-ssh \
  --priority 100 --destination-port-ranges 22 --source-address-prefixes "$MI_IP" \
  --access Allow --protocol Tcp -o none
reintentar az network nsg rule create -g "$RG" --nsg-name "$NSG" -n allow-flower \
  --priority 110 --destination-port-ranges "$PUERTO_FLEET" "$PUERTO_EXEC" \
  --source-address-prefixes 10.0.0.0/24 --access Allow --protocol Tcp -o none
# Permitir tambien que el autor lance `flwr run` al Exec API desde su IP (paso 05_run):
reintentar az network nsg rule create -g "$RG" --nsg-name "$NSG" -n allow-exec-desde-mi-ip \
  --priority 120 --destination-port-ranges "$PUERTO_EXEC" --source-address-prefixes "$MI_IP" \
  --access Allow --protocol Tcp -o none
reintentar az network vnet subnet update -g "$RG" --vnet-name "$VNET" -n "$SUBNET" \
  --network-security-group "$NSG" -o none

echo "== 4/7 Workspace de Azure ML =="
# Crear solo si no existe: `az ml workspace create` sobre uno ya existente re-valida sus recursos
# dependientes (keyvault, storage) y puede fallar por el desfase de consistencia de la region.
if az ml workspace show -g "$RG" -n "$WORKSPACE" -o none 2>/dev/null; then
  echo "  (ya existe, se reutiliza)"
else
  az ml workspace create -g "$RG" -n "$WORKSPACE" -l "$LOCATION" -o none
fi

echo "== 5/7 VMs (3× $VM_SIZE, cloud-init; agregador=$VM_AGREGADOR) =="
for VM in "${TODAS_LAS_VMS[@]}"; do
  IDENTITY_ARGS=()
  # Solo el agregador (Grande) loguea a Azure ML -> necesita managed identity.
  [ "$VM" = "$VM_AGREGADOR" ] && IDENTITY_ARGS=(--assign-identity)
  echo "  creando $VM ($VM_SIZE) ..."
  reintentar az vm create -g "$RG" -n "$VM" --image "$VM_IMAGE" --size "$VM_SIZE" \
    --admin-username "$ADMIN_USER" --generate-ssh-keys \
    --vnet-name "$VNET" --subnet "$SUBNET" --nsg "" \
    --custom-data "$DIR/cloud-init.yaml" "${IDENTITY_ARGS[@]}" -o none
  # auto-shutdown diario (red de seguridad de coste). Best-effort: spaincentral NO soporta el
  # recurso DevTestLab/schedules, asi que si falla se avisa y se sigue -- el control de coste
  # recae entonces en infra/deallocate.sh (pararlas al terminar cada sesion).
  az vm auto-shutdown -g "$RG" -n "$VM" --time "$AUTO_SHUTDOWN_HORA" -o none 2>/dev/null \
    || echo "  (auto-shutdown no disponible en $LOCATION -- usa infra/deallocate.sh al terminar)"
done

echo "== 6/7 Rol del agregador sobre el workspace (para loguear metricas y registrar modelo) =="
PRINCIPAL_ID="$(az vm show -g "$RG" -n "$VM_AGREGADOR" --query identity.principalId -o tsv)"
WS_ID="$(az ml workspace show -g "$RG" -n "$WORKSPACE" --query id -o tsv)"
# `az role assignment create` falla con MissingSubscription en az 2.89.0 (bug del comando) ->
# se crea la asignacion directamente por la API REST de ARM, que si funciona. Rol "AzureML Data
# Scientist" (guid f6c7c914-...). GUID aleatorio para el nombre de la asignacion.
ROLE_DEF_ID="/subscriptions/$AZURE_SUBSCRIPTION_ID/providers/Microsoft.Authorization/roleDefinitions/f6c7c914-8db3-469d-8ca1-694a8f32e121"
ASSIGN_GUID="$(python -c 'import uuid;print(uuid.uuid4())' 2>/dev/null || cat /proc/sys/kernel/random/uuid)"
if az role assignment list --scope "$WS_ID" --assignee "$PRINCIPAL_ID" --query "[0]" -o tsv 2>/dev/null | grep -q .; then
  echo "  (el agregador ya tiene el rol, se omite)"
else
  reintentar az rest --method put \
    --url "https://management.azure.com${WS_ID}/providers/Microsoft.Authorization/roleAssignments/${ASSIGN_GUID}?api-version=2022-04-01" \
    --body "{\"properties\":{\"roleDefinitionId\":\"${ROLE_DEF_ID}\",\"principalId\":\"${PRINCIPAL_ID}\",\"principalType\":\"ServicePrincipal\"}}" \
    -o none
fi

echo "== 7/7 Resumen de IPs =="
for VM in "${TODAS_LAS_VMS[@]}"; do
  IP_PUB="$(az vm show -d -g "$RG" -n "$VM" --query publicIps -o tsv)"
  IP_PRIV="$(az vm show -d -g "$RG" -n "$VM" --query privateIps -o tsv)"
  echo "  $VM  público=$IP_PUB  privado=$IP_PRIV"
done

echo
echo "OK — infraestructura creada. Siguiente: bash infra/03_deploy.sh"
echo "RECUERDA: al terminar cada sesión, bash infra/deallocate.sh (para el gasto)."

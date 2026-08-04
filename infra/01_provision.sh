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

echo "== 1/7 Resource group =="
az group create -n "$RG" -l "$LOCATION" -o none

echo "== 2/7 Red (VNet + subred) =="
az network vnet create -g "$RG" -n "$VNET" --address-prefix 10.0.0.0/16 \
  --subnet-name "$SUBNET" --subnet-prefix 10.0.0.0/24 -o none

echo "== 3/7 NSG (SSH desde tu IP; 9092/9093 dentro de la subred) =="
az network nsg create -g "$RG" -n "$NSG" -o none
MI_IP="$(curl -s https://api.ipify.org)"
az network nsg rule create -g "$RG" --nsg-name "$NSG" -n allow-ssh \
  --priority 100 --destination-port-ranges 22 --source-address-prefixes "$MI_IP" \
  --access Allow --protocol Tcp -o none
az network nsg rule create -g "$RG" --nsg-name "$NSG" -n allow-flower \
  --priority 110 --destination-port-ranges "$PUERTO_FLEET" "$PUERTO_EXEC" \
  --source-address-prefixes 10.0.0.0/24 --access Allow --protocol Tcp -o none
# Permitir también que el autor lance `flwr run` al Exec API desde su IP (paso 05_run):
az network nsg rule create -g "$RG" --nsg-name "$NSG" -n allow-exec-desde-mi-ip \
  --priority 120 --destination-port-ranges "$PUERTO_EXEC" --source-address-prefixes "$MI_IP" \
  --access Allow --protocol Tcp -o none
az network vnet subnet update -g "$RG" --vnet-name "$VNET" -n "$SUBNET" --network-security-group "$NSG" -o none

echo "== 4/7 Workspace de Azure ML =="
az ml workspace create -g "$RG" -n "$WORKSPACE" -l "$LOCATION" -o none

echo "== 5/7 VMs (4× $VM_SIZE, cloud-init) =="
for VM in "${TODAS_LAS_VMS[@]}"; do
  echo "  creando $VM ..."
  IDENTITY_ARGS=()
  [ "$VM" = "$VM_COORD" ] && IDENTITY_ARGS=(--assign-identity)   # solo el coordinador loguea a Azure ML
  az vm create -g "$RG" -n "$VM" --image "$VM_IMAGE" --size "$VM_SIZE" \
    --admin-username "$ADMIN_USER" --generate-ssh-keys \
    --vnet-name "$VNET" --subnet "$SUBNET" --nsg "" \
    --custom-data "$DIR/cloud-init.yaml" "${IDENTITY_ARGS[@]}" -o none
  # auto-shutdown diario (red de seguridad de coste)
  az vm auto-shutdown -g "$RG" -n "$VM" --time "$AUTO_SHUTDOWN_HORA" -o none
done

echo "== 6/7 Rol del coordinador sobre el workspace (para loguear métricas y registrar modelo) =="
PRINCIPAL_ID="$(az vm show -g "$RG" -n "$VM_COORD" --query identity.principalId -o tsv)"
WS_ID="$(az ml workspace show -g "$RG" -n "$WORKSPACE" --query id -o tsv)"
az role assignment create --assignee "$PRINCIPAL_ID" --role "AzureML Data Scientist" \
  --scope "$WS_ID" -o none

echo "== 7/7 Resumen de IPs =="
for VM in "${TODAS_LAS_VMS[@]}"; do
  IP_PUB="$(az vm show -d -g "$RG" -n "$VM" --query publicIps -o tsv)"
  IP_PRIV="$(az vm show -d -g "$RG" -n "$VM" --query privateIps -o tsv)"
  echo "  $VM  público=$IP_PUB  privado=$IP_PRIV"
done

echo
echo "OK — infraestructura creada. Siguiente: bash infra/03_deploy.sh"
echo "RECUERDA: al terminar cada sesión, bash infra/deallocate.sh (para el gasto)."

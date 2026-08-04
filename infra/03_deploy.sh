#!/usr/bin/env bash
# 03_deploy.sh — sube el código y los datos a las VMs e instala las dependencias.
#   - Código (src, configs, flower_app) -> las 4 VMs.
#   - Datos: cada silo recibe SOLO su parquet; val_global solo al coordinador. (Aislamiento real.)
#   - pip install (torch CPU + flower_app/requirements.txt) en el venv de cada VM.
#   - Escribe /home/azureuser/tfm_env.sh en cada VM con las env vars que necesitan los procesos.
#
# Prerrequisito: infra/01_provision.sh hecho, e infra/02_split_datos.py corrido (datos_silos/).
# Uso:  bash infra/03_deploy.sh

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$DIR/.." && pwd)"
source "$DIR/config.sh"

if [ ! -f "$REPO/infra/datos_silos/val_global.parquet" ]; then
  echo "Faltan los parquets. Ejecuta primero: ./.venv/Scripts/python.exe infra/02_split_datos.py"; exit 1
fi

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
ip_de() { az vm show -d -g "$RG" -n "$1" --query publicIps -o tsv; }
priv_de() { az vm show -d -g "$RG" -n "$1" --query privateIps -o tsv; }

MLFLOW_URI="$(az ml workspace show -g "$RG" -n "$WORKSPACE" --query mlflow_tracking_uri -o tsv)"

deploy_codigo() {
  local vm="$1" ip="$2"
  echo "  [$vm] subiendo código..."
  ssh $SSH_OPTS "$ADMIN_USER@$ip" "mkdir -p $REPO_REMOTO $DATOS_REMOTO"
  # Solo lo que las VMs necesitan (no data/, no notebooks, no .venv).
  scp $SSH_OPTS -r -q "$REPO/src" "$REPO/configs" "$REPO/flower_app" "$ADMIN_USER@$ip:$REPO_REMOTO/"
  echo "  [$vm] instalando dependencias (torch CPU + requirements)..."
  ssh $SSH_OPTS "$ADMIN_USER@$ip" "source ~/venv/bin/activate && \
    pip install -q --upgrade pip && \
    pip install -q torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -q -r $REPO_REMOTO/flower_app/requirements.txt"
}

escribir_env() {
  local vm="$1" ip="$2" extra="$3"
  ssh $SSH_OPTS "$ADMIN_USER@$ip" "cat > /home/$ADMIN_USER/tfm_env.sh <<EOF
export TFM_REPO_ROOT=$REPO_REMOTO
export TFM_DATA_DIR=$DATOS_REMOTO
$extra
EOF"
}

echo "== Coordinador =="
COORD_IP="$(ip_de "$VM_COORD")"
deploy_codigo "$VM_COORD" "$COORD_IP"
scp $SSH_OPTS -q "$REPO/infra/datos_silos/val_global.parquet" "$ADMIN_USER@$COORD_IP:$DATOS_REMOTO/"
# azure-ai-ml/identity solo en el coordinador (loguea a MLflow y registra el modelo).
ssh $SSH_OPTS "$ADMIN_USER@$COORD_IP" "source ~/venv/bin/activate && \
  pip install -q azure-ai-ml azure-identity azureml-mlflow"
# El coordinador loguea a Azure ML (MLflow) y registra modelos con la managed identity
# (DefaultAzureCredential). AZ_* los necesita infra/06_registrar_modelo.py para el MLClient.
escribir_env "$VM_COORD" "$COORD_IP" "export MLFLOW_TRACKING_URI=$MLFLOW_URI
export AZ_SUBSCRIPTION=$AZURE_SUBSCRIPTION_ID
export AZ_RG=$RG
export AZ_WORKSPACE=$WORKSPACE"

echo "== Silos =="
for VM in "${VMS_SILO[@]}"; do
  SILO="${SILO_DE_VM[$VM]}"
  IP="$(ip_de "$VM")"
  deploy_codigo "$VM" "$IP"
  # SOLO el parquet de este silo. Verificable: en esta VM no existe el de ningún otro.
  scp $SSH_OPTS -q "$REPO/infra/datos_silos/${SILO,,}.parquet" "$ADMIN_USER@$IP:$DATOS_REMOTO/"
  escribir_env "$VM" "$IP" "export TFM_SILO=$SILO"
done

echo
echo "OK — código y datos desplegados. Aislamiento: cada VM de silo tiene solo su parquet."
echo "Coordinador privado=$(priv_de "$VM_COORD")  (los silos se conectarán a esta IP:$PUERTO_FLEET)"
echo "Siguiente: bash infra/04_start_federation.sh"

#!/usr/bin/env bash
# 03_deploy.sh — sube el codigo y los datos a las 3 VMs e instala las dependencias.
#   - Codigo (src, configs, flower_app) -> las 3 VMs.
#   - Datos: cada silo recibe SOLO su parquet; val_global solo al agregador (Grande). Aislamiento real.
#   - pip install (torch CPU + flower_app/requirements.txt) en el venv de cada VM.
#   - Escribe /home/azureuser/tfm_env.sh en cada VM con las env vars que necesitan los procesos.
#   - El agregador (vm-silo-grande) tiene rol DOBLE: silo Grande + coordinador -> su env combina
#     TFM_SILO con MLFLOW_TRACKING_URI y AZ_* (para loguear/registrar en Azure ML).
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
  echo "  [$vm] subiendo codigo..."
  ssh $SSH_OPTS "$ADMIN_USER@$ip" "mkdir -p $REPO_REMOTO $DATOS_REMOTO"
  # Solo lo que las VMs necesitan (no data/, no notebooks, no .venv).
  scp $SSH_OPTS -r -q "$REPO/src" "$REPO/configs" "$REPO/flower_app" "$ADMIN_USER@$ip:$REPO_REMOTO/"
  # Ubuntu 22.04 trae Python 3.10, pero flwr 1.32 (el del PC que lanza `flwr run`) exige >=3.11.
  # La version de flwr debe coincidir entre PC y VMs o el protocolo gRPC no casa -> instalar 3.11
  # (deadsnakes) y recrear el venv con 3.11 si aun no lo es. Idempotente.
  echo "  [$vm] asegurando Python 3.11..."
  ssh $SSH_OPTS "$ADMIN_USER@$ip" "set -e; \
    if ! ~/venv/bin/python --version 2>/dev/null | grep -q '3.11'; then \
      sudo add-apt-repository -y ppa:deadsnakes/ppa >/dev/null 2>&1; \
      sudo apt-get update -q >/dev/null 2>&1; \
      sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q python3.11 python3.11-venv >/dev/null 2>&1; \
      rm -rf ~/venv && python3.11 -m venv ~/venv; \
    fi; ~/venv/bin/python --version"
  echo "  [$vm] instalando dependencias (torch CPU + requirements)..."
  ssh $SSH_OPTS "$ADMIN_USER@$ip" "source ~/venv/bin/activate && \
    pip install -q --upgrade pip && \
    pip install -q torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -q -r $REPO_REMOTO/flower_app/requirements.txt"
}

for VM in "${VMS_SILO[@]}"; do
  SILO="${SILO_DE_VM[$VM]}"
  IP="$(ip_de "$VM")"
  echo "== $VM (silo $SILO$([ "$VM" = "$VM_AGREGADOR" ] && echo " + agregador")) =="
  deploy_codigo "$VM" "$IP"
  # SOLO el parquet de este silo. Verificable: en esta VM no existe el de ningun otro.
  scp $SSH_OPTS -q "$REPO/infra/datos_silos/${SILO,,}.parquet" "$ADMIN_USER@$IP:$DATOS_REMOTO/"

  # env base (comun a todos los silos)
  ENV_EXTRA="export TFM_SILO=$SILO"

  if [ "$VM" = "$VM_AGREGADOR" ]; then
    # Rol doble: ademas del silo, hospeda el SuperLink/ServerApp -> loguea y registra en Azure ML.
    scp $SSH_OPTS -q "$REPO/infra/datos_silos/val_global.parquet" "$ADMIN_USER@$IP:$DATOS_REMOTO/"
    ssh $SSH_OPTS "$ADMIN_USER@$IP" "source ~/venv/bin/activate && \
      pip install -q azure-ai-ml azure-identity azureml-mlflow"
    ENV_EXTRA="$ENV_EXTRA
export MLFLOW_TRACKING_URI=$MLFLOW_URI
export AZ_SUBSCRIPTION=$AZURE_SUBSCRIPTION_ID
export AZ_RG=$RG
export AZ_WORKSPACE=$WORKSPACE"
  fi

  ssh $SSH_OPTS "$ADMIN_USER@$IP" "cat > /home/$ADMIN_USER/tfm_env.sh <<EOF
export TFM_REPO_ROOT=$REPO_REMOTO
export TFM_DATA_DIR=$DATOS_REMOTO
$ENV_EXTRA
EOF"
done

echo
echo "OK — codigo y datos desplegados. Aislamiento: cada VM de silo tiene solo su parquet"
echo "(el agregador $VM_AGREGADOR tiene ademas val_global, que no es train de nadie)."
echo "Agregador privado=$(priv_de "$VM_AGREGADOR")  (los silos se conectaran a esta IP:$PUERTO_FLEET)"
echo "Siguiente: bash infra/04_start_federation.sh"

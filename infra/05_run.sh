#!/usr/bin/env bash
# 05_run.sh — lanza UNA corrida federada contra la federación ya arrancada, y la ve en el dashboard.
#   Uso:  bash infra/05_run.sh [fedavg|fedprox]   (por defecto fedavg)
#
# Escribe ~/.flwr/config.toml (formato de conexión de flwr 1.32.1) con la IP del Exec API del
# coordinador y lanza `flwr run . remote-federation`. El coordinador, mientras agrega las rondas,
# loguea val_loss/wmape_val a Azure ML -> se ve EN VIVO en Azure ML Studio (experimento
# "tfm-federado-simulacro"). Se recomienda correr fedavg y luego fedprox para comparar ambas curvas.
#
# Prerrequisito: infra/04_start_federation.sh hecho.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$DIR/.." && pwd)"
source "$DIR/config.sh"
ESTRATEGIA="${1:-fedavg}"

AGR_IP="$(az vm show -d -g "$RG" -n "$VM_AGREGADOR" --query publicIps -o tsv)"
# URI de MLflow del workspace -> se pasa al ServerApp por run-config (el superexec no propaga env).
MLFLOW_URI="$(az ml workspace show -g "$RG" -n "$WORKSPACE" --query mlflow_tracking_uri -o tsv)"

# Config de conexion al SuperLink (Exec API) para esta version de flwr.
mkdir -p "$HOME/.flwr"
cat > "$HOME/.flwr/config.toml" <<EOF
[superlink]
default = "remote-federation"

[superlink.remote-federation]
address = "${AGR_IP}:${PUERTO_EXEC}"
insecure = true
EOF

echo "== Lanzando corrida '$ESTRATEGIA' contra $AGR_IP:$PUERTO_EXEC =="
cd "$REPO/flower_app"
# PYTHONIOENCODING/PYTHONUTF8: la consola de Windows (cp1252) no codifica el emoji 🌸 que imprime
# flwr y revienta -> forzar UTF-8.
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
"$REPO/.venv/Scripts/flwr.exe" run . remote-federation \
  --run-config "estrategia='${ESTRATEGIA}' mlflow-tracking-uri='${MLFLOW_URI}'" || \
  flwr run . remote-federation \
  --run-config "estrategia='${ESTRATEGIA}' mlflow-tracking-uri='${MLFLOW_URI}'"

echo
echo "OK — corrida enviada. Abre Azure ML Studio -> Jobs/Experiments -> 'tfm-federado-simulacro'"
echo "y observa val_loss/wmape_val bajar ronda a ronda. Repite con: bash infra/05_run.sh fedprox"

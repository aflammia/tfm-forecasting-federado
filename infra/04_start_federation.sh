#!/usr/bin/env bash
# 04_start_federation.sh — arranca la federacion de Flower (Deployment Engine):
#   - SuperLink (agregador) en vm-silo-grande, en modo inseguro dentro de la VNet privada.
#   - Un SuperNode por silo (incluida Grande, que corre en la misma VM que el SuperLink),
#     conectandose al SuperLink por la IP PRIVADA del agregador, cada uno declarando su silo por
#     --node-config (asi client_app carga solo los datos de ESE silo).
# Los procesos quedan de fondo (nohup); sus logs, en ~/superlink.log y ~/supernode.log de cada VM.
#
# Prerrequisito: infra/03_deploy.sh hecho.
# Uso:  bash infra/04_start_federation.sh

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/config.sh"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
ip_de() { az vm show -d -g "$RG" -n "$1" --query publicIps -o tsv; }
priv_de() { az vm show -d -g "$RG" -n "$1" --query privateIps -o tsv; }

AGR_IP="$(ip_de "$VM_AGREGADOR")"
AGR_PRIV="$(priv_de "$VM_AGREGADOR")"

# Arranca un demonio en la VM desacoplado del canal SSH (setsid + </dev/null), via `ssh bash -s`
# con heredoc -- el metodo inline (comillas anidadas) colgaba SSH y no persistia el proceso.
arrancar_remoto() {  # arrancar_remoto <ip> <nombre_proc> <comando> <logname>
  local ip="$1" proc="$2" cmd="$3" logname="$4"
  ssh $SSH_OPTS "$ADMIN_USER@$ip" bash -s <<REMOTE 2>&1 | grep -v "Warning: Permanently"
source ~/venv/bin/activate
source ~/tfm_env.sh
pkill -f '$proc' 2>/dev/null; sleep 1
setsid $cmd </dev/null >/home/$ADMIN_USER/$logname 2>&1 &
sleep 2
echo "  lanzado (log: ~/$logname)"
REMOTE
}

echo "== SuperLink en el agregador $VM_AGREGADOR ($AGR_PRIV) =="
# --disable-runtime-dependency-installation: que el ServerApp corra en el venv BASE del agregador
# (que ya tiene torch/flwr/pandas + mlflow + azure), en vez de un venv aislado con solo las deps
# del FAB (que NO incluye mlflow -> el dashboard no loguearia). Los SuperNodes ya usan el venv base
# por defecto (no llevan --allow-runtime-dependency-installation).
arrancar_remoto "$AGR_IP" "flower-superlink" \
  "flower-superlink --insecure --disable-runtime-dependency-installation" "superlink.log"
sleep 5   # dar tiempo al SuperLink a abrir el Fleet API antes de conectar los SuperNodes

echo "== SuperNodes (uno por silo, incluido Grande en la propia VM del agregador) =="
for VM in "${VMS_SILO[@]}"; do
  SILO="${SILO_DE_VM[$VM]}"
  IP="$(ip_de "$VM")"
  echo "  [$VM] silo=$SILO -> SuperLink $AGR_PRIV:$PUERTO_FLEET"
  arrancar_remoto "$IP" "flower-supernode" \
    "flower-supernode --insecure --superlink ${AGR_PRIV}:${PUERTO_FLEET} --node-config \"silo='${SILO}'\"" \
    "supernode.log"
done

echo
echo "OK — federacion lanzada. Verifica los logs con:"
echo "  ssh azureuser@$AGR_IP 'tail ~/superlink.log'   (los 3 SuperNodes deben registrarse)"
echo "Siguiente: bash infra/05_run.sh fedavg   (y luego fedprox)"

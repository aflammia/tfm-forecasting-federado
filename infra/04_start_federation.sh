#!/usr/bin/env bash
# 04_start_federation.sh — arranca la federación de Flower (Deployment Engine):
#   - SuperLink (agregador) en el coordinador, en modo inseguro dentro de la VNet privada.
#   - Un SuperNode por silo, conectándose al SuperLink por la IP PRIVADA del coordinador, cada uno
#     declarando su silo por --node-config (así client_app carga solo los datos de ESE silo).
# Los procesos quedan de fondo (nohup); sus logs, en ~/superlink.log y ~/supernode.log de cada VM.
#
# Prerrequisito: infra/03_deploy.sh hecho.
# Uso:  bash infra/04_start_federation.sh

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/config.sh"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
ip_de() { az vm show -d -g "$RG" -n "$1" --query publicIps -o tsv; }
priv_de() { az vm show -d -g "$RG" -n "$1" --query privateIps -o tsv; }

COORD_IP="$(ip_de "$VM_COORD")"
COORD_PRIV="$(priv_de "$VM_COORD")"

echo "== SuperLink en el coordinador ($COORD_PRIV) =="
ssh $SSH_OPTS "$ADMIN_USER@$COORD_IP" \
  "source ~/venv/bin/activate && source ~/tfm_env.sh && \
   pkill -f flower-superlink || true; sleep 1; \
   nohup flower-superlink --insecure > ~/superlink.log 2>&1 & \
   sleep 3; echo 'superlink arrancado'; tail -n 5 ~/superlink.log"

echo "== SuperNodes (uno por silo) =="
for VM in "${VMS_SILO[@]}"; do
  SILO="${SILO_DE_VM[$VM]}"
  IP="$(ip_de "$VM")"
  echo "  [$VM] silo=$SILO -> SuperLink $COORD_PRIV:$PUERTO_FLEET"
  ssh $SSH_OPTS "$ADMIN_USER@$IP" \
    "source ~/venv/bin/activate && source ~/tfm_env.sh && \
     pkill -f flower-supernode || true; sleep 1; \
     nohup flower-supernode --insecure --superlink ${COORD_PRIV}:${PUERTO_FLEET} \
       --node-config \"silo='${SILO}'\" > ~/supernode.log 2>&1 & \
     sleep 3; echo 'supernode arrancado'; tail -n 5 ~/supernode.log"
done

echo
echo "OK — federación arriba. Verifica en ~/superlink.log del coordinador que los 3 SuperNodes"
echo "se han registrado. Siguiente: bash infra/05_run.sh fedavg   (y luego fedprox)"

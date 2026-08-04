#!/usr/bin/env bash
# deallocate.sh — PARA (deallocate) las 4 VMs para que dejen de consumir crédito de cómputo.
# Una VM deallocated cuesta ~0 (solo el disco, céntimos/día). El código y los datos se conservan;
# se vuelven a encender con `az vm start` (o infra/start.sh). CÓRRELO AL TERMINAR CADA SESIÓN.
#   Uso:  bash infra/deallocate.sh

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/config.sh"

for VM in "${TODAS_LAS_VMS[@]}"; do
  echo "  deallocating $VM ..."
  az vm deallocate -g "$RG" -n "$VM" --no-wait
done
echo
echo "OK — las 4 VMs se están apagando (deallocate). El gasto de cómputo baja a ~0."
echo "Para retomar: az vm start -g $RG --ids \$(az vm list -g $RG --query '[].id' -o tsv)"

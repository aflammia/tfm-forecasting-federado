#!/usr/bin/env bash
# destroy.sh — BORRA todo el resource group (las 4 VMs, red, discos, workspace de Azure ML).
# Gasto -> 0 absoluto. IRREVERSIBLE: el workspace y los modelos registrados se pierden. Ejecútalo
# solo cuando el simulacro esté TERMINADO y DOCUMENTADO (capturas del dashboard, RESEARCH_LOG).
#   Uso:  bash infra/destroy.sh

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/config.sh"

echo "Vas a BORRAR el resource group '$RG' y TODO lo que contiene (VMs, workspace, modelos)."
read -r -p "Escribe el nombre del RG para confirmar: " CONFIRMA
if [ "$CONFIRMA" != "$RG" ]; then
  echo "Cancelado (no coincide)."; exit 1
fi
az group delete -n "$RG" --yes --no-wait
echo "OK — borrado en marcha. El gasto de este simulacro queda en 0."

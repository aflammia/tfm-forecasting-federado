#!/usr/bin/env bash
# config.sh — variables compartidas por todos los scripts del simulacro federado en Azure.
# Se sourcea al principio de cada script (`source "$(dirname "$0")/config.sh"`).
#
# NO pongas aquí secretos. AZURE_SUBSCRIPTION_ID se lee de la env var del mismo nombre
# (exportala en tu shell o en un azure_env.sh gitignored). Ver infra/README.md.

set -euo pipefail

# --- Suscripción (obligatorio: export AZURE_SUBSCRIPTION_ID=... antes de ejecutar) ---
: "${AZURE_SUBSCRIPTION_ID:?Falta AZURE_SUBSCRIPTION_ID. Exporta la env var (portal.azure.com -> Subscriptions).}"

# --- Nombres y ubicación (confirmados con el autor: rg-tfm-federado / westeurope) ---
export LOCATION="westeurope"
export RG="rg-tfm-federado"
export VNET="vnet-tfm"
export SUBNET="snet-tfm"
export NSG="nsg-tfm"
export WORKSPACE="ws-tfm-federado"

# --- VMs: 1 coordinador (neutral) + 3 silos. B2s (2 vCPU, 4 GB) -- CPU basta para un MLP de 5k
# parámetros; una GPU sería absurda. Ubuntu 22.04 LTS. ---
export VM_SIZE="Standard_B2s"
export VM_IMAGE="Ubuntu2204"
export ADMIN_USER="azureuser"

export VM_COORD="vm-coordinador"
export VMS_SILO=("vm-silo-grande" "vm-silo-mediano" "vm-silo-pequeno")
# Mapeo VM -> nombre de silo (lo que espera flower_app: node-config "silo='Grande'").
declare -gA SILO_DE_VM=(
  ["vm-silo-grande"]="Grande"
  ["vm-silo-mediano"]="Mediano"
  ["vm-silo-pequeno"]="Pequeno"
)
export TODAS_LAS_VMS=("$VM_COORD" "${VMS_SILO[@]}")

# --- Puertos de Flower (Deployment Engine) ---
export PUERTO_FLEET="9092"   # SuperNodes -> SuperLink (los silos se conectan aquí)
export PUERTO_EXEC="9093"    # `flwr run` -> SuperLink (el autor lanza la corrida aquí)

# --- Rutas remotas dentro de cada VM ---
export REPO_REMOTO="/home/${ADMIN_USER}/tfm"          # código (src, configs, flower_app)
export DATOS_REMOTO="/home/${ADMIN_USER}/datos"        # datos del silo (SOLO el suyo)

# --- Apagado automático (red de seguridad de coste): hora local del RG, formato HHMM ---
export AUTO_SHUTDOWN_HORA="2200"

echo "[config] RG=$RG  LOCATION=$LOCATION  VM_SIZE=$VM_SIZE  (suscripción ...${AZURE_SUBSCRIPTION_ID: -4})"

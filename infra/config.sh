#!/usr/bin/env bash
# config.sh — variables compartidas por todos los scripts del simulacro federado en Azure.
# Se sourcea al principio de cada script (`source "$(dirname "$0")/config.sh"`).
#
# NO pongas aquí secretos. AZURE_SUBSCRIPTION_ID se lee de la env var del mismo nombre
# (exportala en tu shell o en un azure_env.sh gitignored). Ver infra/README.md.

set -euo pipefail

# --- Suscripción (obligatorio: export AZURE_SUBSCRIPTION_ID=... antes de ejecutar) ---
: "${AZURE_SUBSCRIPTION_ID:?Falta AZURE_SUBSCRIPTION_ID. Exporta la env var (portal.azure.com -> Subscriptions).}"

# --- Nombres y ubicación. spaincentral: la suscripción de estudiante bloquea westeurope por
# política ("region not accepting new customers"); spaincentral está permitida y es la más cercana
# a España (menor latencia). Verificado sondeando regiones (Sesión 30). ---
export LOCATION="spaincentral"
export RG="rg-tfm-federado"
export VNET="vnet-tfm"
export SUBNET="snet-tfm"
export NSG="nsg-tfm"
export WORKSPACE="ws-tfm-federado"

# --- VMs: 3 silos, y el AGREGADOR (SuperLink) se aloja en uno de ellos (vm-silo-grande). CPU basta
# para un MLP de 5k parámetros; una GPU sería absurda. Ubuntu 22.04 LTS.
# DISEÑO DE 3 NODOS (forzado por la cuota + capacidad de Azure for Students en spaincentral,
# verificado con `az vm create`, Sesión 30): total regional = 6 vCPUs, y las tallas de 1 vCPU
# (B1s/B1ms) NO tienen capacidad ahora mismo en spaincentral (SkuNotAvailable). Solo B2s_v2
# (2 vCPU) tiene hueco. 4 nodos × 2 vCPU = 8 > 6 no cabe -> se pasa a 3 nodos × B2s_v2 = 6 vCPUs
# (justo la cuota), con el agregador co-alojado en el silo Grande. Es un patrón legítimo de FL
# cross-silo (un operador hospeda la coordinación); el aislamiento de datos se mantiene: cada silo
# tiene SOLO su train; Grande además ve val_global y los pesos agregados, nunca el train ajeno.
# Coste ~$0,12/h las 3 encendidas. ---
export VM_SIZE="Standard_B2s_v2"
export VM_IMAGE="Ubuntu2204"
export ADMIN_USER="azureuser"

# El AGREGADOR (SuperLink + ServerApp) corre en esta VM, que es a la vez el silo Grande.
export VM_AGREGADOR="vm-silo-grande"
export VMS_SILO=("vm-silo-grande" "vm-silo-mediano" "vm-silo-pequeno")
# Mapeo VM -> nombre de silo (lo que espera flower_app: node-config "silo='Grande'").
declare -gA SILO_DE_VM=(
  ["vm-silo-grande"]="Grande"
  ["vm-silo-mediano"]="Mediano"
  ["vm-silo-pequeno"]="Pequeno"
)
export TODAS_LAS_VMS=("${VMS_SILO[@]}")   # 3 VMs; el agregador es una de ellas (Grande)

# --- Puertos de Flower (Deployment Engine) ---
export PUERTO_FLEET="9092"   # SuperNodes -> SuperLink (los silos se conectan aquí)
export PUERTO_EXEC="9093"    # `flwr run` -> SuperLink (el autor lanza la corrida aquí)

# --- Rutas remotas dentro de cada VM ---
export REPO_REMOTO="/home/${ADMIN_USER}/tfm"          # código (src, configs, flower_app)
export DATOS_REMOTO="/home/${ADMIN_USER}/datos"        # datos del silo (SOLO el suyo)

# --- Apagado automático (red de seguridad de coste): hora local del RG, formato HHMM.
# NOTA: spaincentral NO soporta auto-shutdown (recurso DevTestLab/schedules); 01_provision lo
# intenta best-effort y, si falla, el control de coste recae en infra/deallocate.sh. ---
export AUTO_SHUTDOWN_HORA="2200"

echo "[config] RG=$RG  LOCATION=$LOCATION  VMs=3×$VM_SIZE  agregador=$VM_AGREGADOR  (suscripción ...${AZURE_SUBSCRIPTION_ID: -4})"

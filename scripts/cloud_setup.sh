#!/usr/bin/env bash
# cloud_setup.sh — Setup script para el entorno de Claude Code on the web (VM Linux).
#
# Pega el contenido de este archivo (o la línea de abajo) en el campo "Setup script"
# de la configuración del entorno cloud en claude.ai/code. Su salida se cachea, así que
# las dependencias quedan instaladas al inicio de cada sesión sin reinstalar.
#
# NOTA: las variables de entorno del panel (p.ej. KAGGLE_API_TOKEN) NO están disponibles
# durante el setup script — solo cuando la sesión ya corre. Por eso aquí solo instalamos
# dependencias; la descarga de datos se hace en sesión con `python src/00_download_data.py`.

set -euo pipefail

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Entorno cloud preparado: dependencias instaladas."

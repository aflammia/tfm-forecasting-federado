#!/usr/bin/env bash
# compilar.sh — genera memoria/main.pdf.
#
# Se invoca pdflatex/biber directamente en lugar de latexmk por dos peculiaridades de este
# equipo, ambas detectadas y resueltas al montar la memoria:
#   1. latexmk (envoltorio de MiKTeX) aborta al analizar una entrada rota del PATH de Windows
#      (el stub de Python de la Microsoft Store), así que se usa un PATH reducido.
#   2. pdflatex se queda colgado indefinidamente si hereda una entrada estándar abierta; con
#      `< /dev/null` termina con normalidad. Un pdflatex colgado bloquea además los ficheros
#      auxiliares y hace fallar a biber con "Cannot find control file".
#
# Uso:  bash compilar.sh        (desde memoria/)

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

MIKTEX="/c/Users/alefl/AppData/Local/Programs/MiKTeX/miktex/bin/x64"
export PATH="$MIKTEX:/usr/bin:/bin"

pasada() {  # pasada <n>
  echo "  [$1] pdflatex..."
  pdflatex -interaction=batchmode main.tex < /dev/null > /dev/null 2>&1 || true
}

echo "== Compilando la memoria =="
pasada 1
echo "  [2] biber (bibliografia)..."
biber main < /dev/null 2>&1 | grep -aiE "ERROR|WARN" || true
pasada 3
pasada 4   # segunda pasada: fija indices, referencias cruzadas y citas

echo
if [ -f main.pdf ]; then
  echo "OK — main.pdf generado ($(stat -c%s main.pdf) bytes)"
else
  echo "FALLO — no se generó main.pdf"; exit 1
fi

echo "== Avisos que conviene revisar =="
grep -aE "Warning: (Citation|Reference|There were undefined)" main.log | sort -u | head -20 || echo "  (ninguno)"

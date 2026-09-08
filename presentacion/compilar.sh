#!/usr/bin/env bash
# compilar.sh — genera presentacion/presentacion.pdf.
#
# Mismas dos peculiaridades de este equipo que en memoria/compilar.sh:
#   1. latexmk aborta al analizar una entrada rota del PATH de Windows, así que
#      se invoca pdflatex directamente con un PATH reducido.
#   2. pdflatex se cuelga si hereda una entrada estándar abierta; `< /dev/null`
#      lo evita.
# No hay bibliografía, así que no interviene biber. Las figuras se toman de
# ../memoria/figuras/ (vía \graphicspath) para no duplicarlas.
#
# Uso:  bash compilar.sh        (desde presentacion/)

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

MIKTEX="/c/Users/alefl/AppData/Local/Programs/MiKTeX/miktex/bin/x64"
export PATH="$MIKTEX:/usr/bin:/bin"

pasada() {
  echo "  [$1] pdflatex..."
  pdflatex -interaction=batchmode presentacion.tex < /dev/null > /dev/null 2>&1 || true
}

echo "== Compilando la presentacion =="
pasada 1
pasada 2   # segunda pasada: fija las referencias de posicion de TikZ

echo
if [ -f presentacion.pdf ]; then
  echo "OK — presentacion.pdf generado ($(stat -c%s presentacion.pdf) bytes)"
else
  echo "FALLO — no se genero presentacion.pdf"; exit 1
fi

echo "== Avisos que conviene revisar =="
grep -aE "Overfull|Underfull|Warning" presentacion.log | sort -u | head -20 || echo "  (ninguno)"

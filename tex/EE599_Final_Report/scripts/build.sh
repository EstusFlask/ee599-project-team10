#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
if command -v bibtex >/dev/null 2>&1; then
  bibtex main >/dev/null
elif command -v bibtex.original >/dev/null 2>&1; then
  bibtex.original main >/dev/null
elif command -v bibtex8 >/dev/null 2>&1; then
  bibtex8 main >/dev/null
else
  echo 'BibTeX engine missing; install bibtex or use Overleaf.' >&2
  exit 1
fi
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
printf 'Compiled: %s/main.pdf\n' "$PWD"

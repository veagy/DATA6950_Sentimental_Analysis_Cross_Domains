#!/usr/bin/env bash
# Export model documentation catalog → output/models/
#
# From TEMP (repo) root:
#   bash scripts/run_export_model_documentation.sh
#
# Environment (optional):
#   THESIS_PYTHON — default python3
#   THESIS_OUTPUT_DIR — passed as --output-dir
#
# Flags after -- are forwarded to export_model_documentation.py, e.g.:
#   bash scripts/run_export_model_documentation.sh -- --mirror-docs

set -euo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

PY="${THESIS_PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

OUT_DEFAULT="${ROOT}/output/models"
OUTPUT_DIR="${THESIS_OUTPUT_DIR:-${OUT_DEFAULT}}"

EXTRA=()
if [[ $# -gt 0 && "$1" == "--" ]]; then
  shift
  EXTRA=("$@")
fi

exec "$PY" Code/thesis/tools/export_model_documentation.py \
  --repo-root "$ROOT" \
  --output-dir "$OUTPUT_DIR" \
  "${EXTRA[@]}"

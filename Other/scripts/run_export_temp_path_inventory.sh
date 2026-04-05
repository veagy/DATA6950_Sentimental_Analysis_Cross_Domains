#!/usr/bin/env bash
# Export recursive path inventory → output/path/
#
# From TEMP (repo) root:
#   bash scripts/run_export_temp_path_inventory.sh
#
# Environment (optional):
#   THESIS_PYTHON — default python3
#   THESIS_PATH_OUTPUT_DIR — passed as --output-dir
#
# Flags after -- are forwarded to export_temp_path_inventory.py, e.g.:
#   bash scripts/run_export_temp_path_inventory.sh -- --no-default-excludes

set -euo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

PY="${THESIS_PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

OUT_DEFAULT="${ROOT}/output/path"
OUTPUT_DIR="${THESIS_PATH_OUTPUT_DIR:-${OUT_DEFAULT}}"

EXTRA=()
if [[ $# -gt 0 && "$1" == "--" ]]; then
  shift
  EXTRA=("$@")
fi

exec "$PY" Code/thesis/tools/export_temp_path_inventory.py \
  --repo-root "$ROOT" \
  --output-dir "$OUTPUT_DIR" \
  "${EXTRA[@]}"

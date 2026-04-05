#!/usr/bin/env bash
# Dataset EDA for Parquet files under data/ → output/dataset analysis/
#
# From TEMP (repo) root:
#   bash scripts/run_dataset_analysis.sh
#
# Environment (optional):
#   THESIS_PYTHON — default python3
#   THESIS_DATA_ROOT, THESIS_OUTPUT_DIR — passed as --data-root / --output-dir
#
# Flags after -- are forwarded to analyze_thesis_datasets.py, e.g.:
#   bash scripts/run_dataset_analysis.sh -- --max-rows-per-file 10000 --plots

set -euo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

PY="${THESIS_PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

OUT_DEFAULT="${ROOT}/output/dataset analysis"
DATA_ROOT="${THESIS_DATA_ROOT:-${ROOT}/data}"
OUTPUT_DIR="${THESIS_OUTPUT_DIR:-${OUT_DEFAULT}}"

EXTRA=()
if [[ $# -gt 0 && "$1" == "--" ]]; then
  shift
  EXTRA=("$@")
fi

exec "$PY" Code/thesis/tools/analyze_thesis_datasets.py \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  "${EXTRA[@]}"

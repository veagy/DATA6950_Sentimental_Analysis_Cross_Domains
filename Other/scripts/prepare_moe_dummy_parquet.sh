#!/usr/bin/env bash
# Slice aligned rows into TEMP/DUMMY/data/{processed,transformed} for MoE smoke.
# Usage:
#   bash scripts/prepare_moe_dummy_parquet.sh
#   DATA_ROOT=/path/to/data SOURCE_STEM=all-data N_ROWS=1000 bash scripts/prepare_moe_dummy_parquet.sh
#   SYNTHETIC=1 bash scripts/prepare_moe_dummy_parquet.sh   # no source parquets
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-}:${ROOT}"
ARGS=(--repo-root "$ROOT" --n-rows "${N_ROWS:-1000}" --out-stem "${OUT_STEM:-moe_dummy_1k}")
if [[ -n "${SOURCE_STEM:-}" ]]; then
  ARGS+=(--source-stem "$SOURCE_STEM")
fi
if [[ -n "${DATA_ROOT:-}" ]]; then
  ARGS+=(--data-root "$DATA_ROOT")
fi
if [[ "${SYNTHETIC:-0}" == "1" ]]; then
  ARGS+=(--synthetic)
fi
python3 "$ROOT/scripts/prepare_moe_dummy_data.py" "${ARGS[@]}"

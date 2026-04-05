#!/usr/bin/env bash
# Per-source_stem split + checkpoint evaluation → output/metrics/
#
# From TEMP (repo) root:
#   bash scripts/run_per_source_stem_metrics.sh
#
# Environment (optional):
#   THESIS_PYTHON   — default python3
#   THESIS_DATA_ROOT, THESIS_CHECKPOINT_ROOT — passed as --data-root / --checkpoint-root
#
# Full split on large all-data can take a long time and needs RAM. For smoke tests:
#   THESIS_SPLIT_MAX_ROWS=5000 THESIS_MAX_SAMPLES=32 bash scripts/run_per_source_stem_metrics.sh
#
# After a run, flat tables are written (and refreshed from all metrics.json):
#   output/metrics/2label_metrics_table.csv
#   output/metrics/3label_metrics_table.csv
# Rebuild only those CSVs from existing JSON:
#   PYTHONPATH=. python3 Code/thesis/tools/eval_per_source_stem_metrics.py --export-metrics-csv-only
#
# Flags after -- are forwarded to eval_per_source_stem_metrics.py (e.g. --only-stems a,b)

set -euo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

PY="${THESIS_PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(
  --output-dir "${THESIS_METRICS_OUT:-$ROOT/output/metrics}"
  --data-root "${THESIS_DATA_ROOT:-$ROOT/data}"
  --checkpoint-root "${THESIS_CHECKPOINT_ROOT:-$ROOT/checkpoints}"
)

if [[ -n "${THESIS_SPLIT_MAX_ROWS:-}" ]]; then
  ARGS+=(--split-max-rows "$THESIS_SPLIT_MAX_ROWS")
fi
if [[ -n "${THESIS_MAX_SAMPLES:-}" ]]; then
  ARGS+=(--max-samples "$THESIS_MAX_SAMPLES")
fi
if [[ -n "${THESIS_ONLY_STEMS:-}" ]]; then
  ARGS+=(--only-stems "$THESIS_ONLY_STEMS")
fi
if [[ "${THESIS_SKIP_SPLIT:-0}" == "1" ]]; then
  ARGS+=(--skip-split)
fi
if [[ "${THESIS_NO_RUN_META:-0}" == "1" ]]; then
  ARGS+=(--no-run-meta)
fi
if [[ "${THESIS_SKIP_EXISTING:-0}" == "1" ]]; then
  ARGS+=(--skip-existing)
fi
if [[ -n "${THESIS_MOE_MANIFESTS:-}" ]]; then
  ARGS+=(--moe-manifests "$THESIS_MOE_MANIFESTS")
fi

exec "$PY" "$ROOT/Code/thesis/tools/eval_per_source_stem_metrics.py" "${ARGS[@]}" "$@"

#!/usr/bin/env bash
# Run docs/ml Track B or C (train_ml_processed_embed_meta.py) before MoE gate training.
# Writes under checkpoints/moe/ml_stack/{2,3}-labels/{stem}/ per TRAINING_PIPELINES.md §4–5.
#
# Usage:
#   TRACK=b N_LABELS=2 STEM=all-data bash scripts/run_ml_bc_before_moe.sh
#   TRACK=c N_LABELS=3 STEM=all-data DATA_ROOT=/path/to/data bash scripts/run_ml_bc_before_moe.sh
#
# Env: ROOT (default: TEMP repo), TRACK (b|c), N_LABELS (2|3), STEM, DATA_ROOT, CHECKPOINT_ROOT, EXTRA_ARGS
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-}:${ROOT}"

TRACK="${TRACK:-b}"
N_LABELS="${N_LABELS:-2}"
STEM="${STEM:-all-data}"
QUEUE="$ROOT/Code/thesis/config/ml_queue/track_${TRACK}_${N_LABELS}_labels.json"
if [[ ! -f "$QUEUE" ]]; then
  echo "Missing queue config: $QUEUE" >&2
  exit 1
fi

ARGS=(
  --queue_config "$QUEUE"
  --dataset_stem "$STEM"
)
if [[ -n "${DATA_ROOT:-}" ]]; then
  ARGS+=(--data_root "$DATA_ROOT")
fi
if [[ -n "${CHECKPOINT_ROOT:-}" ]]; then
  ARGS+=(--checkpoint_root "$CHECKPOINT_ROOT")
fi
# shellcheck disable=2206
EXTRA=( ${EXTRA_ARGS:-} )

exec python3 "$ROOT/Code/thesis/train/train_ml_processed_embed_meta.py" "${ARGS[@]}" "${EXTRA[@]}"

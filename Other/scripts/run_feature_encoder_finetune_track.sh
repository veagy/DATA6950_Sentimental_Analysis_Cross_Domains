#!/usr/bin/env bash
# FeatureEncoderClassifier on UMAP 100-D features: CNN, LSTM, GRU, RNN, FFNN (100→100 latent + K-way head).
# Configs: Code/thesis/config/feature_encoder/{2_labels,3_labels}/FeatEnc_*.json
# Data: data/transformed/{dataset_stem}.parquet column features_100d (+ sentiment_value).
#
# Docs: TEMP/docs/plans/feature_models_all-data_pretrain_e9470305.plan.md
#       TEMP/docs/thesis_parameter_counts.md (FeatEnc_* rows)
#       TEMP/docs/ml/TRAINING_PIPELINES.md §3 (same parquet as Track A tabular)
#
# Invokes Code/thesis/train/train_queue.py --feature-encoder-only so cnn/B9_CNN_Text and
# rnn/B7_BiLSTM-style configs are NOT queued (those are separate text-sequence models).
#
# Usage (cwd = repo root, usually TEMP):
#   bash scripts/run_feature_encoder_finetune_track.sh
#
# Env:
#   THESIS_QUEUE_INCLUDE_ALL_DATA=1   — default on; required if only all-data.parquet exists under transformed/
#   THESIS_FEATENC_SKIP_PRETRAIN=1    — skip five FeaturePretrainAutoencoder runs; finetune encoders from init
#   THESIS_FEATENC_EPOCHS=2           — --epochs_finetune (default 2; was 8 historically)
#   THESIS_BATCH_SIZE                 — per-process batch for torchrun (default 24 in train_queue)
#   THESIS_NPROC_PER_NODE             — default 2 in train_queue (dual-GPU torchrun)
#   THESIS_PYTHON, THESIS_DATA_ROOT, THESIS_CHECKPOINT_ROOT, THESIS_QUEUE_RUN_ID

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
PY="${THESIS_PYTHON:-python3}"
[[ -x "$ROOT/.venv/bin/python" ]] && PY="$ROOT/.venv/bin/python"

export PYTHONPATH="${PYTHONPATH:-}:$ROOT"
export THESIS_QUEUE_INCLUDE_ALL_DATA="${THESIS_QUEUE_INCLUDE_ALL_DATA:-1}"

EPOCHS="${THESIS_FEATENC_EPOCHS:-2}"
EXTRA=(--feature-encoder-only --skip-wait)
if [[ "${THESIS_FEATENC_SKIP_PRETRAIN:-0}" =~ ^(1|true|TRUE|yes|YES)$ ]]; then
  EXTRA+=(--skip-feature-encoder-pretrain)
fi

exec "$PY" "$ROOT/Code/thesis/train/train_queue.py" \
  "${EXTRA[@]}" \
  --epochs_finetune "$EPOCHS" \
  --data_root "${THESIS_DATA_ROOT:-$ROOT/data}" \
  --checkpoint_root "${THESIS_CHECKPOINT_ROOT:-$ROOT/checkpoints}" \
  --log_dir "${THESIS_LOG_DIR:-$ROOT/logs}"

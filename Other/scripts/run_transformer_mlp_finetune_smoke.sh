#!/usr/bin/env bash
# Quick sanity run: few rows, 1 epoch, 1 GPU, small batch. For local/CI checks only.
#
# Defaults (override via env):
#   THESIS_MAX_SAMPLES=128
#   THESIS_EPOCHS_FINETUNE=1
#   THESIS_TRANSFORMER_FINETUNE_BATCH=8
#   CUDA_VISIBLE_DEVICES=0, THESIS_NPROC_PER_NODE=1
#   THESIS_TORCH_MASTER_PORT=29521  (avoid clashing with a full DDP job on 29500)
#   THESIS_DETACH=none
#
# Usage (from TEMP/):
#   bash scripts/run_transformer_mlp_finetune_smoke.sh 3
#   bash scripts/run_transformer_mlp_finetune_smoke.sh 2 B4_E_DL3_BERT_mlp768_1024

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ -z "${THESIS_PYTHON:-}" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    export THESIS_PYTHON="$ROOT/.venv/bin/python"
  else
    export THESIS_PYTHON=python3
  fi
fi

export THESIS_MAX_SAMPLES="${THESIS_MAX_SAMPLES:-128}"
export THESIS_EPOCHS_FINETUNE="${THESIS_EPOCHS_FINETUNE:-1}"
export THESIS_TRANSFORMER_FINETUNE_BATCH="${THESIS_TRANSFORMER_FINETUNE_BATCH:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export THESIS_NPROC_PER_NODE="${THESIS_NPROC_PER_NODE:-1}"
export THESIS_TORCH_MASTER_PORT="${THESIS_TORCH_MASTER_PORT:-29521}"
export THESIS_DETACH="${THESIS_DETACH:-none}"
export THESIS_ENABLE_GPU_MONITOR="${THESIS_ENABLE_GPU_MONITOR:-0}"
export THESIS_NO_RESUME="${THESIS_NO_RESUME:-1}"

exec bash "$SCRIPT_DIR/run_transformer_mlp_finetune.sh" "$@"

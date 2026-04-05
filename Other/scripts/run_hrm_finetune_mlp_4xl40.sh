#!/usr/bin/env bash
# 4× NVIDIA L40 / L40S (~48–50GB): pin GPUs 0–3, nproc=4, default per-GPU batch 32 for MLP finetune.
# Drops any THESIS_MAX_SAMPLES so the full all-data parquet is used.
#
# Dataset (default stem all-data): data/processed/all-data.parquet — columns text, sentiment_value,
# source_stem; sentiment_value is canonical 3-way only {0,1,2} on disk (rewritten). Override stem:
#   THESIS_HRM_PRETRAIN_STEM=my-stem bash scripts/run_hrm_finetune_mlp_4xl40.sh 3
# If you re-merge all-data from heterogeneous sources, re-run:
#   python Code/thesis/data/rewrite_all_data_sentiment_three_class.py
#
# Usage: bash scripts/run_hrm_finetune_mlp_4xl40.sh [2|3]
# On OOM, lower batch: THESIS_HRM_FINETUNE_BATCH=24 bash scripts/run_hrm_finetune_mlp_4xl40.sh 3
# Sequential 3-label then 2-label: see run_hrm_finetune_mlp_4xl40_sequential.sh
# If a parent shell exported a low smoke-test THESIS_HRM_FINETUNE_BATCH, use:
#   env -u THESIS_HRM_FINETUNE_BATCH bash scripts/run_hrm_finetune_mlp_4xl40.sh 3

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export THESIS_NPROC_PER_NODE=4
unset THESIS_HRM_BATCH 2>/dev/null || true
export THESIS_HRM_FINETUNE_BATCH="${THESIS_HRM_FINETUNE_BATCH:-32}"
unset THESIS_MAX_SAMPLES
export THESIS_CHECKPOINT_ROOT="${THESIS_CHECKPOINT_ROOT:-$ROOT/checkpoints/hrm}"
export THESIS_HRM_PRETRAIN_STEM="${THESIS_HRM_PRETRAIN_STEM:-all-data}"

exec "$SCRIPT_DIR/run_hrm_finetune_mlp.sh" "${1:-3}"

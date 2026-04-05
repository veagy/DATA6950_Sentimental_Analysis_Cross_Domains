#!/usr/bin/env bash
# HRM MLP finetune: **2 classes (negative / positive only)** from processed all-data parquet.
#
# Data:  $ROOT/data/processed/all-data.parquet  (override with THESIS_DATA_ROOT)
# Rows:  sentiment_value 0 = negative, 1 = positive, 2 = neutral → **neutral rows are dropped**
#        (train_single ParquetTextDataset + default 2-class HRM path; do NOT pass --no_hrm_exclude_neutral).
#
# Encoder: checkpoints/hrm/pretrain/all-data/E_HRM1_4Level.safetensors (override THESIS_HRM_ENCODER_CKPT)
# Output:  checkpoints/hrm/fine-tune/all-data/2-labels/E_HRM1_4Level_ft_mlp_2label.safetensors (on full success)
# Live:    logs/resume/E_HRM1_4Level_ft_mlp_2label__all-data__2l/
# Log:     logs/hrm_finetune_mlp_2label.log
#
# 4× L40S (batch 32, 2 ep, GC): scripts/run_hrm_finetune_mlp_2label_posneg_4xl40s.sh
#
# Usage:
#   cd /path/to/TEMP && bash scripts/run_hrm_finetune_mlp_2label_posneg_all_data.sh
# Foreground:  THESIS_DETACH=none bash scripts/run_hrm_finetune_mlp_2label_posneg_all_data.sh
# GPUs:        CUDA_VISIBLE_DEVICES=0,1,2,3  (must match nproc)
# Epochs:      THESIS_EPOCHS_FINETUNE=3  (default in run_hrm_finetune_mlp.sh if unset)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export THESIS_HRM_PRETRAIN_STEM="${THESIS_HRM_PRETRAIN_STEM:-all-data}"
export THESIS_DATA_ROOT="${THESIS_DATA_ROOT:-$ROOT/data}"
export THESIS_HRM_FINETUNE_LOG="${THESIS_HRM_FINETUNE_LOG:-$ROOT/logs/hrm_finetune_mlp_2label.log}"
export THESIS_SESSION="${THESIS_SESSION:-hrm_finetune_mlp_2label}"
# Single-file parquet path: $THESIS_DATA_ROOT/processed/${THESIS_HRM_PRETRAIN_STEM}.parquet
export THESIS_HRM_FINETUNE_SHARDED="${THESIS_HRM_FINETUNE_SHARDED:-0}"

exec bash "$SCRIPT_DIR/run_hrm_finetune_mlp.sh" 2

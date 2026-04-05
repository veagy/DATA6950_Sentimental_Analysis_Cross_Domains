#!/usr/bin/env bash
# Production: 3-label MLP finetune first, then 2-label, sequentially on the same 4×L40 box.
# Both phases use the same frozen encoder; outputs go to fine-tune/all-data/{3,2}-labels/ separately.
# Epochs per phase: THESIS_EPOCHS_FINETUNE (default 3; set to 2 for shorter runs).
#
# Data: default THESIS_HRM_PRETRAIN_STEM=all-data → data/processed/all-data.parquet with
# sentiment_value ∈ {0,1,2} only (see run_hrm_finetune_mlp_4xl40.sh header). 2-label phase drops
# neutral (2) rows per train_single / ParquetTextDataset.
#
# Detach both phases together with one outer tmux (avoid nested tmux per run):
#   tmux new-session -d -s hrm_mlp_finetune_seq \
#     "cd /path/to/TEMP && export THESIS_PYTHON=... THESIS_EPOCHS_FINETUNE=3 && bash scripts/run_hrm_finetune_mlp_4xl40_sequential.sh"
#   tmux attach -t hrm_mlp_finetune_seq
# Tip: if your environment exports a stale THESIS_HRM_FINETUNE_BATCH, prefix with env -u THESIS_HRM_FINETUNE_BATCH

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export THESIS_EPOCHS_FINETUNE="${THESIS_EPOCHS_FINETUNE:-3}"
export THESIS_HRM_PRETRAIN_STEM="${THESIS_HRM_PRETRAIN_STEM:-all-data}"

echo "[hrm-finetune-mlp-4xl40-seq] phase 1/2: 3-label (epochs=${THESIS_EPOCHS_FINETUNE})"
THESIS_DETACH=none bash "$SCRIPT_DIR/run_hrm_finetune_mlp_4xl40.sh" 3

echo "[hrm-finetune-mlp-4xl40-seq] phase 2/2: 2-label (epochs=${THESIS_EPOCHS_FINETUNE})"
THESIS_DETACH=none bash "$SCRIPT_DIR/run_hrm_finetune_mlp_4xl40.sh" 2

echo "[hrm-finetune-mlp-4xl40-seq] done (3-label then 2-label)."

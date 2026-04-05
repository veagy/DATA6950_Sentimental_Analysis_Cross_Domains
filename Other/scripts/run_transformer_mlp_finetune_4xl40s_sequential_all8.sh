#!/usr/bin/env bash
# 4× L40/L40S: run all eight transformer MLP finetunes in one foreground pipeline —
# B3–B6 configs × 3-label, then the same four × 2-label.
#
# Same resource defaults as run_transformer_mlp_finetune_4xl40s.sh:
#   CUDA 0–3, nproc=4, per-GPU batch 128, 2 finetune epochs (override via env).
# Uses THESIS_TRANSFORMER_RUN_ALL=1 twice (labels 3 then 2); THESIS_DETACH=none so
# this script blocks (wrap in tmux/screen at the caller for SSH-safe long runs).
#
# Resume: inherits periodic saves from run_transformer_mlp_finetune.sh unless THESIS_NO_RESUME=1.
#
# Usage (repo root = TEMP, directory containing Code/):
#   bash scripts/run_transformer_mlp_finetune_4xl40s_sequential_all8.sh
# After HRM 2-label finetune (GPU idle): scripts/queue_transformer_mlp_all8_after_hrm_2label.sh
#
# Env: same as scripts/run_transformer_mlp_finetune.sh (HF_TOKEN, THESIS_PYTHON,
#   THESIS_DATA_ROOT, THESIS_CHECKPOINT_ROOT, THESIS_SAVE_EVERY_*, etc.)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export THESIS_NPROC_PER_NODE="${THESIS_NPROC_PER_NODE:-4}"
export THESIS_TRANSFORMER_FINETUNE_BATCH="${THESIS_TRANSFORMER_FINETUNE_BATCH:-128}"
export THESIS_EPOCHS_FINETUNE="${THESIS_EPOCHS_FINETUNE:-2}"
# Default torchrun rendezvous port (29500 often taken if another DDP job is running).
export THESIS_TORCH_MASTER_PORT="${THESIS_TORCH_MASTER_PORT:-29601}"

export THESIS_DETACH=none
export THESIS_TRANSFORMER_RUN_ALL=1

echo "[transformer-mlp-4xl40s-all8] phase 1/2: RUN_ALL 3-label (B3–B6), batch=${THESIS_TRANSFORMER_FINETUNE_BATCH} epochs=${THESIS_EPOCHS_FINETUNE}"
bash "$SCRIPT_DIR/run_transformer_mlp_finetune.sh" 3

echo "[transformer-mlp-4xl40s-all8] phase 2/2: RUN_ALL 2-label (B3–B6)"
bash "$SCRIPT_DIR/run_transformer_mlp_finetune.sh" 2

echo "[transformer-mlp-4xl40s-all8] done (8 finetunes complete)."

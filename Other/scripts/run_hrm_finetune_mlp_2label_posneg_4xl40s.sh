#!/usr/bin/env bash
# 4× L40 / L40S (~50GB): 2-label HRM MLP finetune on **negative + positive only** (neutral dropped)
# from data/processed/all-data.parquet. Distributed DDP, per-GPU batch 256 (override THESIS_HRM_FINETUNE_BATCH), 2 epochs, periodic GC.
#
#   bash scripts/run_hrm_finetune_mlp_2label_posneg_4xl40s.sh
# Foreground: THESIS_DETACH=none bash scripts/run_hrm_finetune_mlp_2label_posneg_4xl40s.sh
# Fresh run (ignore live resume): THESIS_NO_RESUME=1 bash scripts/run_hrm_finetune_mlp_2label_posneg_4xl40s.sh
#
# THESIS_DETACH defaults to **none** so 4-GPU / batch / epoch env is not lost when run_hrm_finetune_mlp.sh
# would otherwise spawn a nested tmux with a bare __inner (no exports). Wrap this script in your own tmux.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Do not use ${VAR:-default}: a parent shell often exports CUDA_VISIBLE_DEVICES=0 and would hide all other GPUs.
export CUDA_VISIBLE_DEVICES="${CUDA_FOUR_GPUS:-0,1,2,3}"
export THESIS_NPROC_PER_NODE="${THESIS_NPROC_PER_NODE:-4}"
unset THESIS_MAX_SAMPLES 2>/dev/null || true
unset THESIS_HRM_BATCH 2>/dev/null || true

export THESIS_HRM_FINETUNE_BATCH="${THESIS_HRM_FINETUNE_BATCH:-256}"
export THESIS_EPOCHS_FINETUNE="${THESIS_EPOCHS_FINETUNE:-2}"
export THESIS_GC_EVERY="${THESIS_GC_EVERY:-50}"
export THESIS_NUM_WORKERS="${THESIS_NUM_WORKERS:-8}"
export THESIS_SESSION="${THESIS_SESSION:-hrm_ft_2l_4xl40s}"
export THESIS_HRM_FINETUNE_LOG="${THESIS_HRM_FINETUNE_LOG:-$ROOT/logs/hrm_finetune_mlp_2label.log}"
export THESIS_DETACH="${THESIS_DETACH:-none}"

exec bash "$SCRIPT_DIR/run_hrm_finetune_mlp_2label_posneg_all_data.sh"

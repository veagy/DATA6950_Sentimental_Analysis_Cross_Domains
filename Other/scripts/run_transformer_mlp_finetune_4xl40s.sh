#!/usr/bin/env bash
# 4× GPU defaults for L40/L40S: per-GPU batch 128, 2 epochs (override THESIS_TRANSFORMER_FINETUNE_BATCH).
# Chain all eight finetunes: run_transformer_mlp_finetune_4xl40s_sequential_all8.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export THESIS_NPROC_PER_NODE="${THESIS_NPROC_PER_NODE:-4}"
export THESIS_TRANSFORMER_FINETUNE_BATCH="${THESIS_TRANSFORMER_FINETUNE_BATCH:-128}"
export THESIS_EPOCHS_FINETUNE="${THESIS_EPOCHS_FINETUNE:-2}"
export THESIS_TORCH_MASTER_PORT="${THESIS_TORCH_MASTER_PORT:-29601}"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_transformer_mlp_finetune.sh" "$@"

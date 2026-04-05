#!/usr/bin/env bash
# Fine-tune GeLU head on GPUs 1, 2, 3 (physical indices after CUDA_VISIBLE_DEVICES remap to 0,1,2).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}"
N_CLASSES="${1:?usage: $0 <2|3> [extra torchrun/train args...]}"
shift
PORT="${MASTER_PORT:-29531}"
exec torchrun --nproc_per_node=3 --master_port="${PORT}" \
  "${ROOT}/Code/thesis/train/train_frozen_pretrain_mlp_head_ddp.py" \
  --data_parquet "${ROOT}/data/transformed/all-data.parquet" \
  --checkpoint_root "${ROOT}/checkpoints" \
  --log_dir "${ROOT}/logs" \
  --pretrain_arch ffnn \
  --n_classes "${N_CLASSES}" \
  --epochs 2 \
  --batch_size 512 \
  --num_workers 4 \
  "$@"

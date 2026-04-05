#!/usr/bin/env bash
# B11: frozen CNN→LSTM pretrain stack + GeLU head (DDP on 3 GPUs).
#
# WAIT until these finish first (disk + GPU contention):
#   - logs/mlp_geLU_head_ddp_ALL_CONFIGS_*.log
#   - logs/ml_bc_queue_rerun_*.log
# Then run manually, e.g.:
#   CUDA_VISIBLE_DEVICES=1,2,3 /path/to/run_b11_cnn_lstm_stack_gelu_3gpu.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}"
PORT="${MASTER_PORT:-29620}"

run_one() {
  local CFG="$1"
  echo "=== B11 stack: $CFG (port $PORT) ==="
  "${ROOT}/.venv/bin/torchrun" --nproc_per_node=3 --master_port="${PORT}" \
    "${ROOT}/Code/thesis/train/train_b11_cnn_lstm_stack_gelu_ddp.py" \
    --config "${CFG}" \
    --checkpoint_root "${ROOT}/checkpoints" \
    --log_dir "${ROOT}/logs"
  PORT=$((PORT + 1))
}

run_one "${ROOT}/Code/thesis/config/b11_cnn_lstm_stack/2_labels/B11_CNN_LSTM_stack.json"
run_one "${ROOT}/Code/thesis/config/b11_cnn_lstm_stack/3_labels/B11_CNN_LSTM_stack.json"
echo "B11 2-label + 3-label runs finished."

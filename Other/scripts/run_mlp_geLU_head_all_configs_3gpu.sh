#!/usr/bin/env bash
# Run every thesis config under config/mlp_gelu_head_ddp/{2_labels,3_labels}/ (FFN, CNN, LSTM, GRU, RNN).
# Uses physical GPUs 1,2,3 only (see CUDA_VISIBLE_DEVICES).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}"
PORT="${MASTER_PORT_BASE:-29550}"
CFG_ROOT="${ROOT}/Code/thesis/config/mlp_gelu_head_ddp"
AGG="${ROOT}/logs/mlp_geLU_head_ddp_ALL_CONFIGS_$(date -u +%Y%m%d_%H%M%S).log"
echo "Aggregate log: $AGG"
{
  echo "======== ALL mlp_gelu_head_ddp configs ========"
  for LABEL_DIR in "${CFG_ROOT}/2_labels" "${CFG_ROOT}/3_labels"; do
    [[ -d "$LABEL_DIR" ]] || continue
    for cfg in "${LABEL_DIR}"/*.json; do
      [[ -f "$cfg" ]] || continue
      echo ""
      echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
      echo "$(date -u -Iseconds) START $cfg port=$PORT"
      echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
      "${ROOT}/.venv/bin/torchrun" --nproc_per_node=3 --master_port="${PORT}" \
        "${ROOT}/Code/thesis/train/train_frozen_pretrain_mlp_head_ddp.py" \
        --config "${cfg}" \
        --checkpoint_root "${ROOT}/checkpoints" \
        --log_dir "${ROOT}/logs" \
        "$@" || exit $?
      echo "$(date -u -Iseconds) END $cfg"
      PORT=$((PORT + 1))
    done
  done
  echo "EXIT:0"
} 2>&1 | tee "$AGG"

#!/usr/bin/env bash
# HRM MLM on data/processed/all-data.parquet (DDP), then CNN/RNN/ML queue.
# Repo root:  bash scripts/run_thesis_pretrain.sh
#
# Env (high level):
#   THESIS_PYTHON, THESIS_DETACH=none|tmux|screen|nohup, THESIS_SESSION
#   CUDA_VISIBLE_DEVICES — e.g. 0,1 for 2 GPUs or 0,1,2,3 for 4 (must match nproc)
#   THESIS_NPROC_PER_NODE — defaults to comma-count of CUDA_VISIBLE_DEVICES (after default 0,1)
#   THESIS_HRM_BATCH, THESIS_FEAT_BATCH — override per-GPU batch sizes
#   HRM E-HRM1 uses seq_len 512 (long-context attention): if you OOM, set THESIS_HRM_BATCH lower than the VRAM-tier defaults below.
#   THESIS_NUM_WORKERS (default 8), THESIS_GC_EVERY (default 0)
#   THESIS_SAVE_EVERY_MINUTES (default 5), THESIS_SAVE_EVERY_STEPS (default 1000)
#   THESIS_MIN_SAVE_INTERVAL_SEC (default 45)
#   (Removed) THESIS_HRM_PRETRAIN_3LABEL — redundant for encoder-only MLM; finetune 2/3-way uses --n_classes separately.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
PY="${THESIS_PYTHON:-python3}"

export THESIS_RESUME_TEMP="${THESIS_RESUME_TEMP:-$ROOT/logs/resume}"
export NVIDIA_TF32_OVERRIDE="${NVIDIA_TF32_OVERRIDE:-1}"
mkdir -p "$ROOT/logs" "$THESIS_RESUME_TEMP"

# Default visible devices first so nproc matches what PyTorch sees (override with CUDA_VISIBLE_DEVICES=0,1,2,3 for 4 GPUs).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

detect_vram() {
  if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' \r' || echo "24000"
  else
    echo "24000"
  fi
}

VRAM_MB="$(detect_vram)"
if [[ -z "${THESIS_NPROC_PER_NODE:-}" ]]; then
  NPROC="$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')"
  if ! [[ "${NPROC}" =~ ^[1-9][0-9]*$ ]]; then
    NPROC=2
  fi
else
  NPROC="${THESIS_NPROC_PER_NODE}"
fi

if [[ "${VRAM_MB:-0}" =~ ^[0-9]+$ ]] && [[ "${VRAM_MB:-0}" -lt 8000 ]]; then
  HRM_BS="${THESIS_HRM_BATCH:-2}"
  FEAT_BS="${THESIS_FEAT_BATCH:-8}"
elif [[ "${VRAM_MB:-0}" =~ ^[0-9]+$ ]] && [[ "${VRAM_MB:-0}" -lt 24000 ]]; then
  HRM_BS="${THESIS_HRM_BATCH:-8}"
  FEAT_BS="${THESIS_FEAT_BATCH:-24}"
elif [[ "${VRAM_MB:-0}" =~ ^[0-9]+$ ]] && [[ "${VRAM_MB:-0}" -lt 80000 ]]; then
  HRM_BS="${THESIS_HRM_BATCH:-24}"
  FEAT_BS="${THESIS_FEAT_BATCH:-48}"
else
  HRM_BS="${THESIS_HRM_BATCH:-64}"
  FEAT_BS="${THESIS_FEAT_BATCH:-96}"
fi

NUM_WORKERS="${THESIS_NUM_WORKERS:-8}"
GC_EVERY="${THESIS_GC_EVERY:-0}"
SAVE_MIN="${THESIS_SAVE_EVERY_MINUTES:-5}"
SAVE_STEPS="${THESIS_SAVE_EVERY_STEPS:-1000}"
MIN_SAVE_INT="${THESIS_MIN_SAVE_INTERVAL_SEC:-45}"

HRM_STEM="${THESIS_HRM_PRETRAIN_STEM:-all-data}"

# shellcheck disable=SC2034
TRAIN_SINGLE_EXTRA=(
  --num_workers "$NUM_WORKERS"
  --gc_every "$GC_EVERY"
  --save_every_minutes "$SAVE_MIN"
  --save_every_steps "$SAVE_STEPS"
  --min_save_interval_sec "$MIN_SAVE_INT"
)

run_inner() {
  echo "[pretrain] nproc=$NPROC CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES HRM_BS=$HRM_BS workers=$NUM_WORKERS save every ${SAVE_MIN}m or ${SAVE_STEPS} steps"
  echo "[pretrain] HRM MLM encoder-only (checkpoints/pretrain/...), all-data.parquet stem=$HRM_STEM"
  "$PY" -m torch.distributed.run --nproc_per_node="$NPROC" \
    "$ROOT/Code/thesis/train/train_single.py" \
    --config "$ROOT/Code/thesis/config/hrm/E_HRM1_4Level.json" \
    --dataset_stem "$HRM_STEM" \
    --data_root "$ROOT/data" \
    --checkpoint_root "$ROOT/checkpoints" \
    --log_dir "$ROOT/logs" \
    --phase pretrain \
    --pretrain_text_source all_data_parquet \
    --epochs_pretrain "${THESIS_EPOCHS_PRETRAIN:-1}" \
    --epochs_finetune 0 \
    --batch_size "$HRM_BS" \
    "${TRAIN_SINGLE_EXTRA[@]}"

  echo "[pretrain] Single HRM MLM pass (K-agnostic checkpoint); no duplicate 3-label MLM."

  echo "[pretrain] CNN/RNN/feature-encoder/ML queue (5× FeaturePretrainAutoencoder on transformed/all-data if present, then queue)"
  export THESIS_BATCH_SIZE="$FEAT_BS"
  # Include merged transformed stem so feature-encoder finetune jobs run when only all-data.parquet exists.
  export THESIS_QUEUE_INCLUDE_ALL_DATA="${THESIS_QUEUE_INCLUDE_ALL_DATA:-1}"
  "$PY" "$ROOT/Code/thesis/train/train_queue.py" \
    --data_root "$ROOT/data" \
    --checkpoint_root "$ROOT/checkpoints" \
    --log_dir "$ROOT/logs" \
    --epochs_finetune "${THESIS_EPOCHS_FEAT:-8}" \
    --phase finetune \
    --include-ml \
    --num_workers "${THESIS_NUM_WORKERS:-8}"
}

if [[ "${1:-}" == "__inner" ]]; then
  run_inner
  exit 0
fi

DETACH="${THESIS_DETACH:-none}"
SESS="${THESIS_SESSION:-thesis_pretrain}"
LOG="$ROOT/logs/pretrain_pipeline.log"

case "$DETACH" in
  tmux)
    tmux new-session -d -s "$SESS" "bash \"$SCRIPT_DIR/run_thesis_pretrain.sh\" __inner"
    echo "tmux session $SESS — attach: tmux attach -t $SESS"
    ;;
  screen)
    screen -dmS "$SESS" bash "$SCRIPT_DIR/run_thesis_pretrain.sh" __inner
    echo "screen $SESS — reattach: screen -r $SESS"
    ;;
  nohup)
    nohup bash "$SCRIPT_DIR/run_thesis_pretrain.sh" __inner >>"$LOG" 2>&1 &
    echo "nohup PID $! — tail -f $LOG"
    ;;
  *)
    run_inner
    ;;
esac

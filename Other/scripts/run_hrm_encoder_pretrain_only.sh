#!/usr/bin/env bash
# HRM encoder-only MLM pretrain ONLY (no CNN/RNN/feature queue, no finetune).
# Writes: {THESIS_CHECKPOINT_ROOT:-checkpoints}/pretrain/{stem}/E_HRM1_4Level.safetensors
#         + tokenizer at .../tokenizer/ (bert-base-uncased via save_pretrained)
# Data:    data/processed/all-data.parquet (text column) unless you override stem/source.
#   If missing, build it: python Code/thesis/data/merge_all_data_parquet.py
#   See data/processed/README.md
#
# Repo root:  bash scripts/run_hrm_encoder_pretrain_only.sh
#
# Env (same knobs as the HRM block in run_thesis_pretrain.sh):
#   THESIS_PYTHON, THESIS_DETACH=none|tmux|screen|nohup, THESIS_SESSION
#   CUDA_VISIBLE_DEVICES  — e.g. 0 (laptop) or 0,1 (2 GPUs); must match nproc count
#   THESIS_NPROC_PER_NODE — default: number of commas in CUDA_VISIBLE_DEVICES (min 1)
#   THESIS_HRM_BATCH — per-GPU batch; unset uses VRAM tiers (2 / 8 / 24 / 64 MB thresholds).
#     Cloud / 96GB: default tier is already 64; for a fixed batch regardless of tier use:
#       export THESIS_HRM_BATCH=64
#     Laptop / 8GB: use export THESIS_HRM_BATCH=1 (or 2) to avoid OOM; do not force 64.
#   THESIS_NUM_WORKERS, THESIS_GC_EVERY
#   THESIS_SAVE_EVERY_MINUTES, THESIS_SAVE_EVERY_STEPS, THESIS_MIN_SAVE_INTERVAL_SEC
#   THESIS_EPOCHS_PRETRAIN — default 2 epochs when unset (see below)
#   THESIS_MAX_SAMPLES — optional cap for smoke tests (passes --max_samples to train_single)
#   THESIS_CHECKPOINT_ROOT — default \$ROOT/checkpoints (use \$ROOT/checkpoints/hrm for hrm/ layout)
#   THESIS_AMP_BF16 — 1|true|yes → --amp_bf16 (bf16 autocast, Blackwell-friendly)
#   THESIS_DATALOADER_PERSISTENT — 1|true|yes → --dataloader_persistent_workers
#   THESIS_RESUME_TEMP, THESIS_HRM_PRETRAIN_STEM (default all-data)
#   THESIS_PRETRAIN_TEXT_SOURCE — default all_data_parquet; set to all_processed or dataset if needed
#
# Example (dual GPU, 2 epochs, batch 64, tmux):
#   cd /path/to/TEMP && export CUDA_VISIBLE_DEVICES=0,1 THESIS_EPOCHS_PRETRAIN=2 THESIS_HRM_BATCH=64 THESIS_DETACH=tmux
#   bash scripts/run_hrm_encoder_pretrain_only.sh
#
# Logs (under repo logs/):
#   logs/hrm_encoder_pretrain_only.log — appended by every run (foreground, tmux, screen, nohup via __inner).
#   logs/resume/ — live resume bundles (THESIS_RESUME_TEMP).
#   Optional GPU usage (only __inner / actual training): THESIS_GPU_MONITOR_LOG (see below).
#   THESIS_GPU_MONITOR_LOG — append nvidia-smi samples (per-GPU util, memory, temp); default logs/hrm_gpu_monitor.log
#   THESIS_GPU_MONITOR_INTERVAL_SEC — sample period (default 5)
#   THESIS_ENABLE_GPU_MONITOR — set 0|false|no to disable

set -euo pipefail
export PYTHONUNBUFFERED=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
PY="${THESIS_PYTHON:-python3}"

export THESIS_RESUME_TEMP="${THESIS_RESUME_TEMP:-$ROOT/logs/resume}"
export NVIDIA_TF32_OVERRIDE="${NVIDIA_TF32_OVERRIDE:-1}"
mkdir -p "$ROOT/logs" "$THESIS_RESUME_TEMP"
RUN_LOG="${THESIS_HRM_PRETRAIN_LOG:-$ROOT/logs/hrm_encoder_pretrain_only.log}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

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
    NPROC=1
  fi
else
  NPROC="${THESIS_NPROC_PER_NODE}"
fi

if [[ "${VRAM_MB:-0}" =~ ^[0-9]+$ ]] && [[ "${VRAM_MB:-0}" -lt 8000 ]]; then
  HRM_BS="${THESIS_HRM_BATCH:-2}"
elif [[ "${VRAM_MB:-0}" =~ ^[0-9]+$ ]] && [[ "${VRAM_MB:-0}" -lt 24000 ]]; then
  HRM_BS="${THESIS_HRM_BATCH:-8}"
elif [[ "${VRAM_MB:-0}" =~ ^[0-9]+$ ]] && [[ "${VRAM_MB:-0}" -lt 80000 ]]; then
  HRM_BS="${THESIS_HRM_BATCH:-24}"
else
  HRM_BS="${THESIS_HRM_BATCH:-64}"
fi

NUM_WORKERS="${THESIS_NUM_WORKERS:-8}"
GC_EVERY="${THESIS_GC_EVERY:-0}"
SAVE_MIN="${THESIS_SAVE_EVERY_MINUTES:-5}"
SAVE_STEPS="${THESIS_SAVE_EVERY_STEPS:-1000}"
MIN_SAVE_INT="${THESIS_MIN_SAVE_INTERVAL_SEC:-45}"

HRM_STEM="${THESIS_HRM_PRETRAIN_STEM:-all-data}"
TEXT_SRC="${THESIS_PRETRAIN_TEXT_SOURCE:-all_data_parquet}"
CHECKPOINT_ROOT="${THESIS_CHECKPOINT_ROOT:-$ROOT/checkpoints}"

TRAIN_SINGLE_EXTRA=(
  --num_workers "$NUM_WORKERS"
  --gc_every "$GC_EVERY"
  --save_every_minutes "$SAVE_MIN"
  --save_every_steps "$SAVE_STEPS"
  --min_save_interval_sec "$MIN_SAVE_INT"
)
if [[ -n "${THESIS_MAX_SAMPLES:-}" ]]; then
  TRAIN_SINGLE_EXTRA+=(--max_samples "${THESIS_MAX_SAMPLES}")
fi
case "${THESIS_AMP_BF16:-0}" in
  1|true|TRUE|yes|YES) TRAIN_SINGLE_EXTRA+=(--amp_bf16) ;;
esac
case "${THESIS_DATALOADER_PERSISTENT:-0}" in
  1|true|TRUE|yes|YES) TRAIN_SINGLE_EXTRA+=(--dataloader_persistent_workers) ;;
esac

run_hrm_only() {
  echo "[hrm-encoder-pretrain-only] nproc=$NPROC CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES batch=$HRM_BS workers=$NUM_WORKERS"
  echo "[hrm-encoder-pretrain-only] stem=$HRM_STEM pretrain_text_source=$TEXT_SRC -> ${CHECKPOINT_ROOT}/pretrain/${HRM_STEM}/"
  "$PY" -m torch.distributed.run --nproc_per_node="$NPROC" \
    "$ROOT/Code/thesis/train/train_single.py" \
    --config "$ROOT/Code/thesis/config/hrm/E_HRM1_4Level.json" \
    --dataset_stem "$HRM_STEM" \
    --data_root "$ROOT/data" \
    --checkpoint_root "$CHECKPOINT_ROOT" \
    --log_dir "$ROOT/logs" \
    --phase pretrain \
    --pretrain_text_source "$TEXT_SRC" \
    --epochs_pretrain "${THESIS_EPOCHS_PRETRAIN:-2}" \
    --epochs_finetune 0 \
    --batch_size "$HRM_BS" \
    "${TRAIN_SINGLE_EXTRA[@]}"
  echo "[hrm-encoder-pretrain-only] done (encoder-only MLM; no other jobs run)."
}

GPU_MONITOR_PID=""

stop_gpu_monitor() {
  local had=0
  if [[ -n "${GPU_MONITOR_PID}" ]]; then
    had=1
    kill "${GPU_MONITOR_PID}" 2>/dev/null || true
    wait "${GPU_MONITOR_PID}" 2>/dev/null || true
    GPU_MONITOR_PID=""
  fi
  if [[ "$had" == "1" ]] && [[ -n "${GPU_MON_LOG:-}" ]]; then
    echo "==== GPU monitor stop $(date -Iseconds 2>/dev/null || date) ====" >>"${GPU_MON_LOG}" 2>/dev/null || true
  fi
}

start_gpu_monitor() {
  GPU_MON_LOG="${THESIS_GPU_MONITOR_LOG:-$ROOT/logs/hrm_gpu_monitor.log}"
  GPU_MON_INTERVAL="${THESIS_GPU_MONITOR_INTERVAL_SEC:-5}"
  case "${THESIS_ENABLE_GPU_MONITOR:-1}" in
    0|false|FALSE|no|NO) return 0 ;;
  esac
  if ! command -v nvidia-smi &>/dev/null; then
    return 0
  fi
  mkdir -p "$(dirname "$GPU_MON_LOG")"
  {
    echo "==== GPU monitor start $(date -Iseconds 2>/dev/null || date) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} nproc=${NPROC} ===="
    nvidia-smi -L 2>/dev/null || true
    echo "# Each line: ISO_time index name gpu_util% mem_used_mib mem_total_mib temp_C"
  } >>"$GPU_MON_LOG"
  (
    while true; do
      ts="$(date -Iseconds 2>/dev/null || date)"
      while IFS= read -r _line; do
        echo "${ts} ${_line}" >>"$GPU_MON_LOG" || true
      done < <(nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null) || true
      sleep "${GPU_MON_INTERVAL}"
    done
  ) &
  GPU_MONITOR_PID=$!
}

run_hrm_only_tee() {
  mkdir -p "$ROOT/logs"
  echo "==== $(date -Iseconds 2>/dev/null || date) HRM encoder-only pretrain ====" >>"$RUN_LOG"
  set +e
  run_hrm_only 2>&1 | tee -a "$RUN_LOG"
  st="${PIPESTATUS[0]}"
  set -e
  return "$st"
}

if [[ "${1:-}" == "__inner" ]]; then
  GPU_MON_LOG="${THESIS_GPU_MONITOR_LOG:-$ROOT/logs/hrm_gpu_monitor.log}"
  start_gpu_monitor
  trap stop_gpu_monitor EXIT INT TERM
  if [[ -n "${GPU_MONITOR_PID}" ]]; then
    echo "[hrm-encoder-pretrain-only] GPU usage log: ${GPU_MON_LOG} (interval ${THESIS_GPU_MONITOR_INTERVAL_SEC:-5}s)" | tee -a "$RUN_LOG"
  fi
  run_hrm_only_tee
  exit $?
fi

DETACH="${THESIS_DETACH:-none}"
SESS="${THESIS_SESSION:-hrm_encoder_pretrain}"

case "$DETACH" in
  tmux)
    tmux new-session -d -s "$SESS" "bash \"$SCRIPT_DIR/run_hrm_encoder_pretrain_only.sh\" __inner"
    echo "tmux session $SESS — attach: tmux attach -t $SESS"
    echo "Log file: $RUN_LOG (tail -f \"$RUN_LOG\")"
    if [[ "${THESIS_ENABLE_GPU_MONITOR:-1}" =~ ^(0|false|FALSE|no|NO)$ ]]; then
      :
    elif command -v nvidia-smi &>/dev/null; then
      echo "GPU usage log: ${THESIS_GPU_MONITOR_LOG:-$ROOT/logs/hrm_gpu_monitor.log} (tail -f \"${THESIS_GPU_MONITOR_LOG:-$ROOT/logs/hrm_gpu_monitor.log}\")"
    fi
    ;;
  screen)
    screen -dmS "$SESS" bash "$SCRIPT_DIR/run_hrm_encoder_pretrain_only.sh" __inner
    echo "screen $SESS — reattach: screen -r $SESS"
    echo "Log file: $RUN_LOG (tail -f \"$RUN_LOG\")"
    if [[ "${THESIS_ENABLE_GPU_MONITOR:-1}" =~ ^(0|false|FALSE|no|NO)$ ]]; then
      :
    elif command -v nvidia-smi &>/dev/null; then
      echo "GPU usage log: ${THESIS_GPU_MONITOR_LOG:-$ROOT/logs/hrm_gpu_monitor.log} (tail -f \"${THESIS_GPU_MONITOR_LOG:-$ROOT/logs/hrm_gpu_monitor.log}\")"
    fi
    ;;
  nohup)
    nohup bash "$SCRIPT_DIR/run_hrm_encoder_pretrain_only.sh" __inner >/dev/null 2>&1 &
    echo "nohup PID $! — output goes to $RUN_LOG (tee inside __inner)"
    echo "tail -f \"$RUN_LOG\""
    if [[ "${THESIS_ENABLE_GPU_MONITOR:-1}" =~ ^(0|false|FALSE|no|NO)$ ]]; then
      :
    elif command -v nvidia-smi &>/dev/null; then
      echo "GPU usage log: ${THESIS_GPU_MONITOR_LOG:-$ROOT/logs/hrm_gpu_monitor.log} (tail -f \"${THESIS_GPU_MONITOR_LOG:-$ROOT/logs/hrm_gpu_monitor.log}\")"
    fi
    ;;
  *)
    GPU_MON_LOG="${THESIS_GPU_MONITOR_LOG:-$ROOT/logs/hrm_gpu_monitor.log}"
    start_gpu_monitor
    trap stop_gpu_monitor EXIT INT TERM
    if [[ -n "${GPU_MONITOR_PID}" ]]; then
      echo "[hrm-encoder-pretrain-only] GPU usage log: ${GPU_MON_LOG} (interval ${THESIS_GPU_MONITOR_INTERVAL_SEC:-5}s)" | tee -a "$RUN_LOG"
    fi
    run_hrm_only_tee
    ;;
esac

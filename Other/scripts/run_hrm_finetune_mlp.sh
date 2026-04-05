#!/usr/bin/env bash
# HRM frozen-encoder supervised finetune with MLP sentiment head (2- or 3-class).
# Checkpoints: {THESIS_CHECKPOINT_ROOT:-checkpoints/hrm}/fine-tune/{stem}/{K-labels}/E_HRM1_4Level_ft_mlp_{K}label.safetensors
# Encoder weights: --hrm_encoder_ckpt (default checkpoints/hrm/pretrain/all-data/E_HRM1_4Level.safetensors)
#
# 4× L40 / L40S production: scripts/run_hrm_finetune_mlp_4xl40.sh [2|3] (CUDA 0–3, nproc=4, default batch 32).
# Sequential 3-label then 2-label: scripts/run_hrm_finetune_mlp_4xl40_sequential.sh — wrap the whole script in one tmux for detach.
# 3-label (2 ep) then 2-label (2 ep), separate logs: scripts/run_hrm_finetune_mlp_3then2_two_epochs.sh
# 2-label pos/neg only from data/processed/all-data.parquet: scripts/run_hrm_finetune_mlp_2label_posneg_all_data.sh
#
# Repo root:  bash scripts/run_hrm_finetune_mlp.sh [2|3]
#   or:        THESIS_HRM_FINETUNE_LABELS=2 bash scripts/run_hrm_finetune_mlp.sh
#
# Default: trains on **all** rows in data/processed/{stem}.parquet (no --max_samples unless THESIS_MAX_SAMPLES is set).
# For stem all-data: parquet has text, sentiment_value, source_stem; sentiment_value on disk is **{0,1,2}**
# (canonical 3-way). Re-merge normalization: Code/thesis/data/rewrite_all_data_sentiment_three_class.py
# Default detach: **tmux** (background session). Foreground: THESIS_DETACH=none bash scripts/run_hrm_finetune_mlp.sh 3
#
# GPU targets (per-GPU --batch_size; tune with THESIS_HRM_FINETUNE_BATCH or THESIS_HRM_BATCH):
#   - 2× RTX Pro 6000 Blackwell 96GB: try 32–64 (defaults below use same VRAM tiers as encoder pretrain).
#   - 2× NVIDIA L40S (48–50GB): tier below picks 8 / 32 by VRAM detect (override with THESIS_HRM_FINETUNE_BATCH).
#
# Env (see also run_hrm_encoder_pretrain_only.sh):
#   THESIS_PYTHON, THESIS_DETACH=tmux|none|screen|nohup (default tmux), THESIS_SESSION
#   CUDA_VISIBLE_DEVICES  — e.g. 0,1 for 2 GPUs; must match nproc count
#   THESIS_NPROC_PER_NODE — default: number of GPUs listed in CUDA_VISIBLE_DEVICES
#   THESIS_HRM_FINETUNE_BATCH — per-GPU batch (overrides tier); else THESIS_HRM_BATCH; else VRAM tier
#   THESIS_NUM_WORKERS, THESIS_GC_EVERY
#   THESIS_SAVE_EVERY_MINUTES, THESIS_SAVE_EVERY_STEPS, THESIS_MIN_SAVE_INTERVAL_SEC
#   THESIS_EPOCHS_FINETUNE — default 3
#   THESIS_MAX_SAMPLES — optional cap for smoke tests only; omit for full dataset
#   THESIS_CHECKPOINT_ROOT, THESIS_HRM_PRETRAIN_STEM (dataset stem, default all-data)
#   THESIS_DATA_ROOT — default $ROOT/data; set to e.g. $ROOT/BACKUP/data to use an alternate processed/ tree
#   THESIS_HRM_ENCODER_CKPT — override path to E_HRM1_4Level.safetensors
#   THESIS_AMP_BF16, THESIS_DATALOADER_PERSISTENT, THESIS_RESUME_TEMP
#   THESIS_HRM_FINETUNE_LOG — override log path
#   THESIS_NO_RESUME=1 — pass --no-resume (ignore live temp resume for this run)
#   THESIS_HRM_FINETUNE_CKPT_LAYOUT — 1|true|yes → same as --hrm_finetune_checkpoint_layout (default on in this script)

set -euo pipefail
export PYTHONUNBUFFERED=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
PY="${THESIS_PYTHON:-python3}"

# HF Hub: if HF_TOKEN is unset, read from $ROOT/.env (supports `HF_TOKEN = value` spacing). Does not log the secret.
ENV_DOT="$ROOT/.env"
if [[ -z "${HF_TOKEN:-}" ]] && [[ -f "$ENV_DOT" ]]; then
  HF_DOT_PY="${ROOT}/.venv/bin/python"
  [[ -x "$HF_DOT_PY" ]] || HF_DOT_PY=python3
  HF_TOKEN="$("$HF_DOT_PY" -c 'import re, sys
from pathlib import Path
for line in Path(sys.argv[1]).read_text().splitlines():
    m = re.match("^\\s*HF_TOKEN\\s*=\\s*(.+)$", line)
    if m:
        print(m.group(1).strip())
        break
' "$ENV_DOT")"
  if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN
  fi
fi
if [[ -n "${HF_TOKEN:-}" ]] && [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

export THESIS_RESUME_TEMP="${THESIS_RESUME_TEMP:-$ROOT/logs/resume}"
export NVIDIA_TF32_OVERRIDE="${NVIDIA_TF32_OVERRIDE:-1}"
mkdir -p "$ROOT/logs" "$THESIS_RESUME_TEMP"

INNER=0
if [[ "${1:-}" == "__inner" ]]; then
  INNER=1
  shift
  LABELS="${1:-3}"
else
  LABELS="${1:-${THESIS_HRM_FINETUNE_LABELS:-3}}"
fi

case "$LABELS" in
  2)
    N_CLASS=2
    CFG_NAME="E_HRM1_4Level_ft_mlp_2label.json"
    ;;
  3)
    N_CLASS=3
    CFG_NAME="E_HRM1_4Level_ft_mlp_3label.json"
    ;;
  *)
    echo "Usage: $0 [2|3]   (or set THESIS_HRM_FINETUNE_LABELS=2|3)" >&2
    exit 1
    ;;
esac

CFG="$ROOT/Code/thesis/config/hrm/$CFG_NAME"
RUN_LOG="${THESIS_HRM_FINETUNE_LOG:-$ROOT/logs/hrm_finetune_mlp_${LABELS}label.log}"

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

if [[ -n "${THESIS_HRM_FINETUNE_BATCH:-}" ]]; then
  FT_BS="${THESIS_HRM_FINETUNE_BATCH}"
elif [[ -n "${THESIS_HRM_BATCH:-}" ]]; then
  FT_BS="${THESIS_HRM_BATCH}"
elif [[ "${VRAM_MB:-0}" =~ ^[0-9]+$ ]] && [[ "${VRAM_MB:-0}" -lt 8000 ]]; then
  FT_BS=2
elif [[ "${VRAM_MB:-0}" =~ ^[0-9]+$ ]] && [[ "${VRAM_MB:-0}" -lt 24000 ]]; then
  FT_BS=8
elif [[ "${VRAM_MB:-0}" =~ ^[0-9]+$ ]] && [[ "${VRAM_MB:-0}" -lt 80000 ]]; then
  FT_BS=32
else
  FT_BS=64
fi

NUM_WORKERS="${THESIS_NUM_WORKERS:-8}"
# Match HRM MLM: periodic gc + CUDA cache clear (set THESIS_GC_EVERY=0 to disable)
GC_EVERY="${THESIS_GC_EVERY:-50}"
SAVE_MIN="${THESIS_SAVE_EVERY_MINUTES:-5}"
SAVE_STEPS="${THESIS_SAVE_EVERY_STEPS:-1000}"
MIN_SAVE_INT="${THESIS_MIN_SAVE_INTERVAL_SEC:-45}"

HRM_STEM="${THESIS_HRM_PRETRAIN_STEM:-all-data}"
DATA_ROOT="${THESIS_DATA_ROOT:-$ROOT/data}"
CHECKPOINT_ROOT="${THESIS_CHECKPOINT_ROOT:-$ROOT/checkpoints/hrm}"
ENCODER_CKPT="${THESIS_HRM_ENCODER_CKPT:-$CHECKPOINT_ROOT/pretrain/${HRM_STEM}/E_HRM1_4Level.safetensors}"

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
# Single stem parquet (e.g. data/processed/all-data.parquet): load once into RAM — fastest steps, higher startup RAM.
# Multi-shard lazy merge: THESIS_HRM_FINETUNE_SHARDED=1 → --hrm-finetune-sharded-processed (lower RAM, often slower per step).
case "${THESIS_HRM_FINETUNE_SHARDED:-0}" in
  1|true|TRUE|yes|YES) TRAIN_SINGLE_EXTRA+=(--hrm-finetune-sharded-processed) ;;
esac
case "${THESIS_NO_RESUME:-0}" in
  1|true|TRUE|yes|YES) TRAIN_SINGLE_EXTRA+=(--no-resume) ;;
esac

run_finetune() {
  echo "[hrm-finetune-mlp] nproc=$NPROC CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES batch=$FT_BS workers=$NUM_WORKERS labels=$LABELS"
  echo "[hrm-finetune-mlp] stem=$HRM_STEM data_root=$DATA_ROOT checkpoint_root=$CHECKPOINT_ROOT encoder=$ENCODER_CKPT"
  if [[ -n "${THESIS_MAX_SAMPLES:-}" ]]; then
    echo "[hrm-finetune-mlp] max_samples=${THESIS_MAX_SAMPLES} (capped; unset THESIS_MAX_SAMPLES for all rows)"
  else
    echo "[hrm-finetune-mlp] dataset: all samples (no --max_samples)"
  fi
  if [[ ! -f "$ENCODER_CKPT" ]]; then
    echo "[hrm-finetune-mlp] WARNING: encoder checkpoint missing at $ENCODER_CKPT" >&2
  fi
  "$PY" -m torch.distributed.run --nproc_per_node="$NPROC" \
    "$ROOT/Code/thesis/train/train_single.py" \
    --config "$CFG" \
    --dataset_stem "$HRM_STEM" \
    --n_classes "$N_CLASS" \
    --data_root "$DATA_ROOT" \
    --checkpoint_root "$CHECKPOINT_ROOT" \
    --log_dir "$ROOT/logs" \
    --phase finetune \
    --epochs_pretrain 0 \
    --epochs_finetune "${THESIS_EPOCHS_FINETUNE:-3}" \
    --batch_size "$FT_BS" \
    --hrm_encoder_ckpt "$ENCODER_CKPT" \
    --hrm_finetune_checkpoint_layout \
    "${TRAIN_SINGLE_EXTRA[@]}"
  echo "[hrm-finetune-mlp] done (${LABELS}-label MLP head; encoder loaded from pretrain ckpt)."
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
  GPU_MON_LOG="${THESIS_GPU_MONITOR_LOG:-$ROOT/logs/hrm_finetune_gpu_monitor.log}"
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

run_finetune_tee() {
  mkdir -p "$ROOT/logs"
  echo "==== $(date -Iseconds 2>/dev/null || date) HRM finetune MLP ${LABELS}-label ====" >>"$RUN_LOG"
  set +e
  run_finetune 2>&1 | tee -a "$RUN_LOG"
  st="${PIPESTATUS[0]}"
  set -e
  return "$st"
}

if [[ "$INNER" == "1" ]]; then
  GPU_MON_LOG="${THESIS_GPU_MONITOR_LOG:-$ROOT/logs/hrm_finetune_gpu_monitor.log}"
  start_gpu_monitor
  trap stop_gpu_monitor EXIT INT TERM
  if [[ -n "${GPU_MONITOR_PID}" ]]; then
    echo "[hrm-finetune-mlp] GPU usage log: ${GPU_MON_LOG} (interval ${THESIS_GPU_MONITOR_INTERVAL_SEC:-5}s)" | tee -a "$RUN_LOG"
  fi
  run_finetune_tee
  exit $?
fi

DETACH="${THESIS_DETACH:-tmux}"
SESS="${THESIS_SESSION:-hrm_finetune_mlp_${LABELS}label}"

case "$DETACH" in
  tmux)
    tmux new-session -d -s "$SESS" "bash \"$SCRIPT_DIR/run_hrm_finetune_mlp.sh\" __inner $LABELS"
    echo "tmux session $SESS — attach: tmux attach -t $SESS"
    echo "Log file: $RUN_LOG"
    ;;
  screen)
    screen -dmS "$SESS" bash "$SCRIPT_DIR/run_hrm_finetune_mlp.sh" __inner "$LABELS"
    echo "screen $SESS — reattach: screen -r $SESS"
    echo "Log file: $RUN_LOG"
    ;;
  nohup)
    nohup bash "$SCRIPT_DIR/run_hrm_finetune_mlp.sh" __inner "$LABELS" >/dev/null 2>&1 &
    echo "nohup PID $! — tail -f \"$RUN_LOG\""
    ;;
  *)
    GPU_MON_LOG="${THESIS_GPU_MONITOR_LOG:-$ROOT/logs/hrm_finetune_gpu_monitor.log}"
    start_gpu_monitor
    trap stop_gpu_monitor EXIT INT TERM
    if [[ -n "${GPU_MONITOR_PID}" ]]; then
      echo "[hrm-finetune-mlp] GPU usage log: ${GPU_MON_LOG}" | tee -a "$RUN_LOG"
    fi
    run_finetune_tee
    ;;
esac

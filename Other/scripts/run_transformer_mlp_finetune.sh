#!/usr/bin/env bash
# Frozen Hugging Face transformer backbone + trainable MLP head (768→1024→GELU→K).
# Checkpoints: {THESIS_CHECKPOINT_ROOT:-checkpoints}/{K}-labels/{stem}/{config_stem}.safetensors
#
# Usage (from repo root = directory containing Code/):
#   bash scripts/run_transformer_mlp_finetune.sh [2|3] [config_stem]
#   THESIS_TRANSFORMER_CFG_STEM=B4_E_DL3_BERT_mlp768_1024 bash scripts/run_transformer_mlp_finetune.sh 2
#   THESIS_TRANSFORMER_RUN_ALL=1 THESIS_DETACH=none bash scripts/run_transformer_mlp_finetune.sh 3
#
# 4× GPU: scripts/run_transformer_mlp_finetune_4xl40s.sh [2|3] [config_stem]
# All 8 (B3–B6 × 3 then 2), foreground: scripts/run_transformer_mlp_finetune_4xl40s_sequential_all8.sh
#
# Env: THESIS_PYTHON, THESIS_DETACH=tmux|none|screen|nohup (default tmux), THESIS_SESSION
#   CUDA_VISIBLE_DEVICES, THESIS_NPROC_PER_NODE
#   THESIS_TRANSFORMER_FINETUNE_BATCH (default 128 per GPU here; 4xl40s wrapper may override), THESIS_EPOCHS_FINETUNE (default 2)
#   THESIS_LR (default 1e-3, head-only)
#   THESIS_DATA_ROOT, THESIS_TRANSFORMER_DATASET_STEM (default all-data)
#   THESIS_CHECKPOINT_ROOT, THESIS_RESUME_TEMP, THESIS_NO_RESUME
#   THESIS_SAVE_EVERY_*, THESIS_NUM_WORKERS, THESIS_GC_EVERY
#   THESIS_TRANSFORMER_FINETUNE_LOG, THESIS_MAX_SAMPLES
#   THESIS_TORCH_MASTER_PORT — pass-through to torchrun (--master_port) when set (avoid EADDRINUSE)
#   THESIS_TRANSFORMER_RUN_ALL=1 — sequential runs for all four B3–B6 MLP configs (use THESIS_DETACH=none)
#   HF_TOKEN / .env — same as run_hrm_finetune_mlp.sh

set -euo pipefail
export PYTHONUNBUFFERED=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
PY="${THESIS_PYTHON:-python3}"

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
fi

LABELS="${1:-${THESIS_TRANSFORMER_FINETUNE_LABELS:-3}}"
CFG_ARG="${2:-}"

if [[ "$INNER" == "0" ]] && [[ -n "$CFG_ARG" ]]; then
  export THESIS_TRANSFORMER_CFG_STEM="$CFG_ARG"
fi

case "$LABELS" in
  2)
    N_CLASS=2
    LABEL_DIR="2_labels"
    ;;
  3)
    N_CLASS=3
    LABEL_DIR="3_labels"
    ;;
  *)
    echo "Usage: $0 [2|3] [config_stem]   (or set THESIS_TRANSFORMER_FINETUNE_LABELS=2|3)" >&2
    exit 1
    ;;
esac

if [[ "$INNER" == "0" ]] && [[ "${THESIS_TRANSFORMER_RUN_ALL:-0}" =~ ^(1|true|TRUE|yes|YES)$ ]]; then
  export THESIS_DETACH="${THESIS_DETACH:-none}"
  STEMS=(
    B3_E_DL1_DistilBERT_mlp768_1024
    B4_E_DL3_BERT_mlp768_1024
    B5_E_DL2_RoBERTa_mlp768_1024
    B6_BART_mlp768_1024
  )
  echo "[transformer-mlp-finetune] RUN_ALL: ${#STEMS[@]} models, labels=$LABELS detach=$THESIS_DETACH"
  for s in "${STEMS[@]}"; do
    echo "========== $s =========="
    _mk="${s%%_mlp768_1024}"
    THESIS_TRANSFORMER_CFG_STEM="$s" \
      THESIS_TRANSFORMER_FINETUNE_LOG="$ROOT/logs/transformer_finetune_mlp_${_mk}_${LABELS}label.log" \
      bash "$SCRIPT_DIR/run_transformer_mlp_finetune.sh" __inner "$LABELS" || exit $?
  done
  echo "[transformer-mlp-finetune] RUN_ALL done."
  exit 0
fi

CFG_STEM="${THESIS_TRANSFORMER_CFG_STEM:-B3_E_DL1_DistilBERT_mlp768_1024}"
CFG="$ROOT/Code/thesis/config/transformers/${LABEL_DIR}/${CFG_STEM}.json"
if [[ ! -f "$CFG" ]]; then
  echo "Missing config: $CFG" >&2
  exit 1
fi

MODEL_KEY="${CFG_STEM%%_mlp768_1024}"
MODEL_KEY="${MODEL_KEY:-$CFG_STEM}"
RUN_LOG="${THESIS_TRANSFORMER_FINETUNE_LOG:-$ROOT/logs/transformer_finetune_mlp_${MODEL_KEY}_${LABELS}label.log}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ -z "${THESIS_NPROC_PER_NODE:-}" ]]; then
  NPROC="$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')"
  if ! [[ "${NPROC}" =~ ^[1-9][0-9]*$ ]]; then
    NPROC=1
  fi
else
  NPROC="${THESIS_NPROC_PER_NODE}"
fi

# Full matrix (RUN_ALL × 2 label modes): 2 epochs; batch from env (4xl40s wrapper sets its own default).
FT_BS="${THESIS_TRANSFORMER_FINETUNE_BATCH:-128}"

NUM_WORKERS="${THESIS_NUM_WORKERS:-8}"
GC_EVERY="${THESIS_GC_EVERY:-50}"
SAVE_MIN="${THESIS_SAVE_EVERY_MINUTES:-5}"
SAVE_STEPS="${THESIS_SAVE_EVERY_STEPS:-1000}"
MIN_SAVE_INT="${THESIS_MIN_SAVE_INTERVAL_SEC:-45}"

DS_STEM="${THESIS_TRANSFORMER_DATASET_STEM:-all-data}"
DATA_ROOT="${THESIS_DATA_ROOT:-$ROOT/data}"
CHECKPOINT_ROOT="${THESIS_CHECKPOINT_ROOT:-$ROOT/checkpoints}"
LR="${THESIS_LR:-1e-3}"

TRAIN_SINGLE_EXTRA=(
  --num_workers "$NUM_WORKERS"
  --gc_every "$GC_EVERY"
  --save_every_minutes "$SAVE_MIN"
  --save_every_steps "$SAVE_STEPS"
  --min_save_interval_sec "$MIN_SAVE_INT"
  --lr "$LR"
)
if [[ -n "${THESIS_MAX_SAMPLES:-}" ]]; then
  TRAIN_SINGLE_EXTRA+=(--max_samples "${THESIS_MAX_SAMPLES}")
fi
case "${THESIS_DATALOADER_PERSISTENT:-0}" in
  1|true|TRUE|yes|YES) TRAIN_SINGLE_EXTRA+=(--dataloader_persistent_workers) ;;
esac
case "${THESIS_NO_RESUME:-0}" in
  1|true|TRUE|yes|YES) TRAIN_SINGLE_EXTRA+=(--no-resume) ;;
esac

run_finetune() {
  echo "[transformer-mlp-finetune] nproc=$NPROC CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES batch=$FT_BS labels=$LABELS lr=$LR"
  echo "[transformer-mlp-finetune] cfg=$CFG stem=$DS_STEM data_root=$DATA_ROOT checkpoint_root=$CHECKPOINT_ROOT"
  if [[ -n "${THESIS_MAX_SAMPLES:-}" ]]; then
    echo "[transformer-mlp-finetune] max_samples=${THESIS_MAX_SAMPLES}"
  fi
  TORCHRUN_EXTRA=()
  if [[ -n "${THESIS_TORCH_MASTER_PORT:-}" ]]; then
    TORCHRUN_EXTRA+=(--master_port "${THESIS_TORCH_MASTER_PORT}")
  fi
  "$PY" -m torch.distributed.run --nproc_per_node="$NPROC" "${TORCHRUN_EXTRA[@]}" \
    "$ROOT/Code/thesis/train/train_single.py" \
    --config "$CFG" \
    --dataset_stem "$DS_STEM" \
    --n_classes "$N_CLASS" \
    --data_root "$DATA_ROOT" \
    --checkpoint_root "$CHECKPOINT_ROOT" \
    --log_dir "$ROOT/logs" \
    --phase finetune \
    --epochs_pretrain 0 \
    --epochs_finetune "${THESIS_EPOCHS_FINETUNE:-2}" \
    --batch_size "$FT_BS" \
    "${TRAIN_SINGLE_EXTRA[@]}"
  echo "[transformer-mlp-finetune] done ($CFG_STEM ${LABELS}-label)."
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
  GPU_MON_LOG="${THESIS_GPU_MONITOR_LOG:-$ROOT/logs/transformer_finetune_gpu_monitor.log}"
  GPU_MON_INTERVAL="${THESIS_GPU_MONITOR_INTERVAL_SEC:-5}"
  case "${THESIS_ENABLE_GPU_MONITOR:-1}" in
    0|false|FALSE|no|NO) return 0 ;;
  esac
  if ! command -v nvidia-smi &>/dev/null; then
    return 0
  fi
  mkdir -p "$(dirname "$GPU_MON_LOG")"
  {
    echo "==== GPU monitor start $(date -Iseconds 2>/dev/null || date) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} nproc=${NPROC} cfg=${CFG_STEM} ===="
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
  echo "==== $(date -Iseconds 2>/dev/null || date) Transformer MLP finetune ${LABELS}-label ${CFG_STEM} ====" >>"$RUN_LOG"
  set +e
  run_finetune 2>&1 | tee -a "$RUN_LOG"
  st="${PIPESTATUS[0]}"
  set -e
  return "$st"
}

if [[ "$INNER" == "1" ]]; then
  GPU_MON_LOG="${THESIS_GPU_MONITOR_LOG:-$ROOT/logs/transformer_finetune_gpu_monitor.log}"
  start_gpu_monitor
  trap stop_gpu_monitor EXIT INT TERM
  if [[ -n "${GPU_MONITOR_PID}" ]]; then
    echo "[transformer-mlp-finetune] GPU usage log: ${GPU_MON_LOG}" | tee -a "$RUN_LOG"
  fi
  run_finetune_tee
  exit $?
fi

DETACH="${THESIS_DETACH:-tmux}"
SESS="${THESIS_SESSION:-transformer_finetune_mlp_${CFG_STEM}_${LABELS}label}"
SESS="${SESS//./_}"

case "$DETACH" in
  tmux)
    tmux new-session -d -s "$SESS" "bash \"$SCRIPT_DIR/run_transformer_mlp_finetune.sh\" __inner \"$LABELS\""
    echo "tmux session $SESS — attach: tmux attach -t $SESS"
    echo "Log file: $RUN_LOG"
    ;;
  screen)
    screen -dmS "$SESS" bash "$SCRIPT_DIR/run_transformer_mlp_finetune.sh" __inner "$LABELS"
    echo "screen $SESS — reattach: screen -r $SESS"
    echo "Log file: $RUN_LOG"
    ;;
  nohup)
    nohup bash "$SCRIPT_DIR/run_transformer_mlp_finetune.sh" __inner "$LABELS" >/dev/null 2>&1 &
    echo "nohup PID $! — tail -f \"$RUN_LOG\""
    ;;
  *)
    GPU_MON_LOG="${THESIS_GPU_MONITOR_LOG:-$ROOT/logs/transformer_finetune_gpu_monitor.log}"
    start_gpu_monitor
    trap stop_gpu_monitor EXIT INT TERM
    if [[ -n "${GPU_MONITOR_PID}" ]]; then
      echo "[transformer-mlp-finetune] GPU usage log: ${GPU_MON_LOG}" | tee -a "$RUN_LOG"
    fi
    run_finetune_tee
    ;;
esac

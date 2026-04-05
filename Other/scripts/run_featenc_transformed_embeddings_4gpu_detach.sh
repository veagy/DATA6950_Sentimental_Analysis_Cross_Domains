#!/usr/bin/env bash
# FeatureEncoderClassifier on 100-D transformed embeddings: 4× GPU DDP via torchrun + train_queue.
#
# Data:   data/transformed/all-data.parquet (columns features_100d + sentiment_value; full row set unless THESIS_MAX_SAMPLES).
# Models: Code/thesis/config/feature_encoder/{2_labels,3_labels}/FeatEnc_*.json (CNN, LSTM, GRU, RNN, FFNN).
# Out:    REAL finetune weights (safetensors) under:
#            $CHECKPOINT_ROOT/{2,3}-labels/all-data/FeatEnc_<ARCH>.safetensors
#          Example: checkpoints/3-labels/all-data/FeatEnc_CNN.safetensors
#          Then (unless THESIS_FEATENC_INCLUDE_ML_BC=0) docs/ml Tracks B/C under:
#            $CHECKPOINT_ROOT/moe/ml_stack/{2,3}-labels/all-data/trackB_*.safetensors and proc_*.joblib
#          Logs: $LOG_DIR (queue + per-job logs under queue_cnn_rnn_*/).
# Resume: live bundles under THESIS_RESUME_TEMP (default ./logs/resume) + train_queue state under logs/queue_cnn_rnn_*/queue_state.json
#
# Defaults: per-GPU batch 4096, 2 finetune epochs, 4 processes, periodic resume save every 200 steps (override via env).
#
# Usage:
#   cd /path/to/TEMP && bash scripts/run_featenc_transformed_embeddings_4gpu_detach.sh
# Foreground (no detach):  THESIS_DETACH=none bash scripts/run_featenc_transformed_embeddings_4gpu_detach.sh
# Resume queue after crash: export THESIS_QUEUE_RESUME_DIR=/path/to/logs/queue_cnn_rnn_<id>  then re-run this script.
#
# Env (optional):
#   THESIS_PYTHON, THESIS_DETACH=tmux|screen|nohup|none (default tmux)
#   THESIS_SESSION, THESIS_TORCH_MASTER_PORT, THESIS_FEATENC_SKIP_PRETRAIN (default 1)
#   THESIS_BATCH_SIZE (default 1024), THESIS_FEATENC_EPOCHS (default 2), THESIS_NPROC_PER_NODE (default 4)
#   THESIS_SAVE_EVERY_STEPS, THESIS_MIN_SAVE_INTERVAL_SEC, THESIS_NUM_WORKERS, THESIS_DIST_TIMEOUT_SEC
#   THESIS_CHECKPOINT_ROOT, THESIS_LOG_DIR, THESIS_DATA_ROOT, CUDA_VISIBLE_DEVICES
#   THESIS_FEATENC_INCLUDE_ML_BC=1 (default) — after FeatEnc jobs, queue docs/ml Tracks B & C on
#     data/processed/{stem}.parquet (frozen DistilBERT + linear head or sklearn LR/LinearSVC).
#     Set to 0 to run feature encoders only. Requires data/processed/all-data.parquet when enabled.

set -euo pipefail
export PYTHONUNBUFFERED=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

PY="${THESIS_PYTHON:-python3}"
[[ -x "$ROOT/.venv/bin/python" ]] && PY="$ROOT/.venv/bin/python"
export PYTHONPATH="${PYTHONPATH:-}:$ROOT"

# A sticky THESIS_TORCH_MASTER_PORT in the shell breaks sequential torchrun jobs (EADDRINUSE).
# train_queue picks a free port per job when this var is unset (see train_queue._torchrun_extra_args_from_env).
if [[ -z "${THESIS_FEATENC_USE_FIXED_TORCH_PORT:-}" ]]; then
  unset THESIS_TORCH_MASTER_PORT
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export THESIS_NPROC_PER_NODE="${THESIS_NPROC_PER_NODE:-4}"
export THESIS_BATCH_SIZE="${THESIS_BATCH_SIZE:-1024}"
export THESIS_FEATENC_EPOCHS="${THESIS_FEATENC_EPOCHS:-2}"
export THESIS_NUM_WORKERS="${THESIS_NUM_WORKERS:-4}"
export THESIS_QUEUE_INCLUDE_ALL_DATA="${THESIS_QUEUE_INCLUDE_ALL_DATA:-1}"
export THESIS_DIST_TIMEOUT_SEC="${THESIS_DIST_TIMEOUT_SEC:-7200}"

CKPT_ROOT="${THESIS_CHECKPOINT_ROOT:-$ROOT/checkpoints}"
LOG_ROOT="${THESIS_LOG_DIR:-$ROOT/logs}"
DATA_ROOT="${THESIS_DATA_ROOT:-$ROOT/data}"
export THESIS_CHECKPOINT_ROOT="$CKPT_ROOT"
export THESIS_LOG_DIR="$LOG_ROOT"
export THESIS_DATA_ROOT="$DATA_ROOT"

export THESIS_RESUME_TEMP="${THESIS_RESUME_TEMP:-$LOG_ROOT/resume}"
# Wider interval reduces DDP+disk stalls (was 200 → noticeable ~12s pauses on 4×GPU FeatEnc).
export THESIS_SAVE_EVERY_STEPS="${THESIS_SAVE_EVERY_STEPS:-800}"
export THESIS_MIN_SAVE_INTERVAL_SEC="${THESIS_MIN_SAVE_INTERVAL_SEC:-60}"

STEMS_FILE="${THESIS_FEATENC_STEMS_FILE:-$LOG_ROOT/featenc_stems_all_data_only.txt}"
mkdir -p "$LOG_ROOT" "$CKPT_ROOT" "$THESIS_RESUME_TEMP"
printf '%s\n' "all-data" >"$STEMS_FILE"

PQ="$DATA_ROOT/transformed/all-data.parquet"
if [[ ! -f "$PQ" ]]; then
  echo "ERROR: missing transformed parquet: $PQ" >&2
  exit 1
fi

NROWS="$("$PY" -c "import pyarrow.parquet as pq; print(pq.ParquetFile(r'''$PQ''').metadata.num_rows)" 2>/dev/null || echo "?")"
echo "[featenc-4gpu] Parquet: $PQ rows=$NROWS (all rows unless THESIS_MAX_SAMPLES is set on train_single — not passed here by default)" >&2
echo "[featenc-4gpu] DDP: nproc=$THESIS_NPROC_PER_NODE CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES per-GPU batch=$THESIS_BATCH_SIZE epochs=$THESIS_FEATENC_EPOCHS" >&2
echo "[featenc-4gpu] Checkpoints: $CKPT_ROOT  logs: $LOG_ROOT  live resume: $THESIS_RESUME_TEMP  save_every_steps=$THESIS_SAVE_EVERY_STEPS" >&2

_STAMP_ON="${THESIS_FEATENC_LOG_RUN_STAMP:-1}"
if [[ "${_STAMP_ON}" =~ ^(0|false|FALSE|no|NO)$ ]]; then
  RUN_LOG="${THESIS_FEATENC_RUN_LOG:-$LOG_ROOT/featenc_transformed_embeddings_4gpu.log}"
  GPU_MON_LOG="${THESIS_GPU_MONITOR_LOG:-$LOG_ROOT/featenc_transformed_embeddings_4gpu_gpu.log}"
else
  _stamp="$(date -u +%Y%m%d_%H%M%S)"
  RUN_LOG="${THESIS_FEATENC_RUN_LOG:-$LOG_ROOT/featenc_transformed_embeddings_4gpu_${_stamp}.log}"
  GPU_MON_LOG="${THESIS_GPU_MONITOR_LOG:-$LOG_ROOT/featenc_transformed_embeddings_4gpu_gpu_${_stamp}.log}"
  echo "[featenc-4gpu] Training log: $RUN_LOG" >&2
  echo "[featenc-4gpu] GPU log:     $GPU_MON_LOG" >&2
fi
# Detached modes (tmux/screen/nohup) re-exec __inner; export so child sees paths.
export RUN_LOG GPU_MON_LOG LOG_ROOT

EXTRA_QUEUE=(--feature-encoder-only --skip-wait --stems-file "$STEMS_FILE")
if [[ "${THESIS_FEATENC_SKIP_PRETRAIN:-1}" =~ ^(1|true|TRUE|yes|YES)$ ]]; then
  EXTRA_QUEUE+=(--skip-feature-encoder-pretrain)
fi
if [[ "${THESIS_FEATENC_INCLUDE_ML_BC:-1}" =~ ^(1|true|TRUE|yes|YES)$ ]]; then
  EXTRA_QUEUE+=(--include-ml-bc-after-feature-encoder)
fi

run_queue() {
  "$PY" "$ROOT/Code/thesis/train/train_queue.py" \
    "${EXTRA_QUEUE[@]}" \
    --epochs_finetune "$THESIS_FEATENC_EPOCHS" \
    --batch_size "$THESIS_BATCH_SIZE" \
    --num_workers "$THESIS_NUM_WORKERS" \
    --phase finetune \
    --data_root "$DATA_ROOT" \
    --checkpoint_root "$CKPT_ROOT" \
    --log_dir "$LOG_ROOT"
}

GPU_MONITOR_PID=""

stop_gpu_monitor() {
  local had=0
  if [[ -n "${GPU_MONITOR_PID:-}" ]]; then
    had=1
    kill "$GPU_MONITOR_PID" 2>/dev/null || true
    wait "$GPU_MONITOR_PID" 2>/dev/null || true
    GPU_MONITOR_PID=""
  fi
  if [[ "$had" == "1" ]] && [[ -n "${GPU_MON_LOG:-}" ]]; then
    echo "==== GPU monitor stop $(date -Iseconds 2>/dev/null || date) ====" >>"${GPU_MON_LOG}" 2>/dev/null || true
  fi
}

# Same sampling layout as scripts/run_transformer_mlp_finetune.sh (ISO prefix + nvidia-smi -L + csv rows).
start_gpu_monitor() {
  GPU_MON_LOG="${THESIS_GPU_MONITOR_LOG:-${GPU_MON_LOG:-}}"
  if [[ -z "$GPU_MON_LOG" ]]; then
    GPU_MON_LOG="${LOG_ROOT:-$ROOT/logs}/featenc_transformed_embeddings_4gpu_gpu.log"
  fi
  GPU_MON_INTERVAL="${THESIS_GPU_MONITOR_INTERVAL_SEC:-5}"
  case "${THESIS_ENABLE_GPU_MONITOR:-1}" in
    0|false|FALSE|no|NO) return 0 ;;
  esac
  if ! command -v nvidia-smi &>/dev/null; then
    return 0
  fi
  mkdir -p "$(dirname "$GPU_MON_LOG")"
  {
    echo "==== GPU monitor start $(date -Iseconds 2>/dev/null || date) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} nproc=${THESIS_NPROC_PER_NODE:-4} cfg=featenc_train_queue ===="
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

run_inner_tee() {
  mkdir -p "$LOG_ROOT"
  echo "==== $(date -Iseconds 2>/dev/null || date) featenc transformed embeddings 4-GPU queue ====" >>"$RUN_LOG"
  set +e
  run_queue 2>&1 | tee -a "$RUN_LOG"
  local st="${PIPESTATUS[0]}"
  set -e
  return "$st"
}

if [[ "${1:-}" == "__inner" ]]; then
  start_gpu_monitor
  trap stop_gpu_monitor EXIT INT TERM
  if [[ -n "${GPU_MONITOR_PID:-}" ]]; then
    echo "[featenc-4gpu] GPU monitor → $GPU_MON_LOG" | tee -a "$RUN_LOG"
  fi
  run_inner_tee
  exit $?
fi

DETACH="${THESIS_DETACH:-tmux}"
SESS="${THESIS_SESSION:-featenc_transformed_4gpu}"
SESS="${SESS//./_}"
NOHUP_PID_FILE="${THESIS_FEATENC_NOHUP_PID_FILE:-$LOG_ROOT/featenc_transformed_embeddings_4gpu_nohup.pid}"

case "$DETACH" in
  tmux)
    tmux new-session -d -s "$SESS" "bash \"$SCRIPT_DIR/run_featenc_transformed_embeddings_4gpu_detach.sh\" __inner"
    echo "tmux session $SESS — attach: tmux attach -t $SESS"
    echo "Log: $RUN_LOG"
    ;;
  screen)
    screen -dmS "$SESS" bash "$SCRIPT_DIR/run_featenc_transformed_embeddings_4gpu_detach.sh" __inner
    echo "screen $SESS — reattach: screen -r $SESS"
    echo "Log: $RUN_LOG"
    ;;
  nohup)
    nohup bash "$SCRIPT_DIR/run_featenc_transformed_embeddings_4gpu_detach.sh" __inner >/dev/null 2>&1 &
    echo $! >"$NOHUP_PID_FILE"
    echo "nohup PID $! (saved $NOHUP_PID_FILE) — tail -f \"$RUN_LOG\""
    ;;
  none|foreground|"")
    start_gpu_monitor
    trap stop_gpu_monitor EXIT INT TERM
    if [[ -n "${GPU_MONITOR_PID:-}" ]]; then
      echo "[featenc-4gpu] GPU monitor → $GPU_MON_LOG" | tee -a "$RUN_LOG"
    fi
    run_inner_tee
    ;;
  *)
    echo "ERROR: THESIS_DETACH must be tmux, screen, nohup, or none — got: $DETACH" >&2
    exit 1
    ;;
esac

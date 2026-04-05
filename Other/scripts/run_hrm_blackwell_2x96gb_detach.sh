#!/usr/bin/env bash
# HRM MLM pretrain for 2x RTX Pro 6000 Blackwell-class (96GB): DDP, per-GPU batch 64, 2 epochs.
#
# Outputs (under TEMP repo root):
#   checkpoints/hrm/pretrain/all-data/E_HRM1_4Level.safetensors
#   checkpoints/hrm/tokenizer/          (google-bert/bert-base-uncased via HuggingFace save_pretrained)
#   logs/hrm_blackwell_pretrain_<UTC>.log  (stdout+stderr; default one file per launch)
#   logs/hrm_blackwell_gpu_usage_<UTC>.log (nvidia-smi samples; pair matches training log)
#   logs/resume/                        live resume bundles
#
# Data: data/processed/all-data.parquet (build: python Code/thesis/data/merge_all_data_parquet.py)
#
# Usage:
#   cd /path/to/TEMP && bash scripts/run_hrm_blackwell_2x96gb_detach.sh
#
# tmux:   tmux attach -t hrm_blackwell_pretrain
# follow: stderr prints exact paths; or: ls -t logs/hrm_blackwell_pretrain_*.log | head -1
#
# Tunables (env): THESIS_HRM_BATCH, THESIS_EPOCHS_PRETRAIN, THESIS_NUM_WORKERS,
#   THESIS_SESSION, THESIS_AMP_BF16 (default on), THESIS_DATALOADER_PERSISTENT (default on)
#   THESIS_ALLOW_TINY_PRETRAIN_DATASET=1 — only if you intend a smoke run; otherwise
#   rows < THESIS_MIN_PRETRAIN_ROWS (default 10000) aborts to avoid training on a wrong/placeholder merge.
#   THESIS_PRETRAIN_LOG_RUN_STAMP=0 — disable timestamped logs (append to fixed paths below; confusing if many runs).
#     Default is ON: each launch gets logs/hrm_blackwell_pretrain_<UTC>.log and matching GPU log.

set -euo pipefail
export PYTHONUNBUFFERED=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export THESIS_NPROC_PER_NODE="${THESIS_NPROC_PER_NODE:-2}"
export THESIS_HRM_BATCH="${THESIS_HRM_BATCH:-64}"
export THESIS_EPOCHS_PRETRAIN="${THESIS_EPOCHS_PRETRAIN:-2}"
export THESIS_NUM_WORKERS="${THESIS_NUM_WORKERS:-12}"
export THESIS_CHECKPOINT_ROOT="${THESIS_CHECKPOINT_ROOT:-$ROOT/checkpoints/hrm}"
# Fresh log file per launch by default (set THESIS_PRETRAIN_LOG_RUN_STAMP=0 to append to fixed filenames).
_STAMP_ON="${THESIS_PRETRAIN_LOG_RUN_STAMP:-1}"
if [[ "${_STAMP_ON}" =~ ^(0|false|FALSE|no|NO)$ ]]; then
  export THESIS_HRM_PRETRAIN_LOG="${THESIS_HRM_PRETRAIN_LOG:-$ROOT/logs/hrm_blackwell_pretrain.log}"
  export THESIS_GPU_MONITOR_LOG="${THESIS_GPU_MONITOR_LOG:-$ROOT/logs/hrm_blackwell_gpu_usage.log}"
else
  _stamp="$(date -u +%Y%m%d_%H%M%S)"
  export THESIS_HRM_PRETRAIN_LOG="${THESIS_HRM_PRETRAIN_LOG:-$ROOT/logs/hrm_blackwell_pretrain_${_stamp}.log}"
  export THESIS_GPU_MONITOR_LOG="${THESIS_GPU_MONITOR_LOG:-$ROOT/logs/hrm_blackwell_gpu_usage_${_stamp}.log}"
  echo "[hrm-blackwell] This run's training log: $THESIS_HRM_PRETRAIN_LOG" >&2
  echo "[hrm-blackwell] This run's GPU log:     $THESIS_GPU_MONITOR_LOG" >&2
fi
export THESIS_RESUME_TEMP="${THESIS_RESUME_TEMP:-$ROOT/logs/resume}"
export THESIS_DETACH="${THESIS_DETACH:-tmux}"
export THESIS_SESSION="${THESIS_SESSION:-hrm_blackwell_pretrain}"
export THESIS_AMP_BF16="${THESIS_AMP_BF16:-1}"
export THESIS_DATALOADER_PERSISTENT="${THESIS_DATALOADER_PERSISTENT:-1}"
# NCCL collective watchdog (PyTorch): avoid 600s default timeout on long stalls between ranks
export THESIS_DIST_TIMEOUT_SEC="${THESIS_DIST_TIMEOUT_SEC:-7200}"

DATA_PQ="$ROOT/data/processed/all-data.parquet"
if [[ ! -f "$DATA_PQ" ]]; then
  echo "ERROR: missing $DATA_PQ" >&2
  echo "Build with: cd \"$ROOT\" && python Code/thesis/data/merge_all_data_parquet.py" >&2
  exit 1
fi

MIN_ROWS="${THESIS_MIN_PRETRAIN_ROWS:-10000}"
NROWS="$(python3 -c "import pyarrow.parquet as pq; print(pq.ParquetFile(r'''$DATA_PQ''').metadata.num_rows)" 2>/dev/null || echo "0")"
if [[ "${NROWS:-0}" =~ ^[0-9]+$ ]] && [[ "${NROWS:-0}" -lt "$MIN_ROWS" ]]; then
  case "${THESIS_ALLOW_TINY_PRETRAIN_DATASET:-0}" in
    1|true|TRUE|yes|YES) ;;
    *)
      echo "ERROR: $DATA_PQ has only ${NROWS} rows (minimum ${MIN_ROWS} unless you know what you are doing)." >&2
      echo "Restore your real merged parquet or rebuild with merge_all_data_parquet.py." >&2
      echo "For an intentional tiny smoke run: export THESIS_ALLOW_TINY_PRETRAIN_DATASET=1" >&2
      exit 1
      ;;
  esac
fi

mkdir -p "$ROOT/logs" "$ROOT/checkpoints/hrm" "$THESIS_RESUME_TEMP"

echo "[hrm-blackwell] all-data.parquet row count: ${NROWS} — training uses every loaded row unless THESIS_MAX_SAMPLES is set." >&2
if [[ -n "${THESIS_MAX_SAMPLES:-}" ]]; then
  echo "[hrm-blackwell] WARNING: THESIS_MAX_SAMPLES=${THESIS_MAX_SAMPLES} will cap pretrain rows (for smoke tests only)." >&2
fi

exec bash "$SCRIPT_DIR/run_hrm_encoder_pretrain_only.sh"

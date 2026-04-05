#!/usr/bin/env bash
# Full MoE run: flush TEMP/logs, background GPU sampling, periodic checkpoints, optional VRAM-target batch size.
#
#   FLUSH_LOGS=1              — remove all files in ROOT/logs (default 1)
#   GPU_MONITOR=1             — append nvidia-smi CSV every GPU_INTERVAL sec (default 1)
#   GPU_INTERVAL=5
#   EPOCHS=1
#   SAVE_EVERY_STEPS=100
#   AUTO_BATCH_VRAM_TARGET=0.8 — omit to use fixed BATCH_SIZE
#   RESUME=1                  — pass --resume to train_moe
#   NPROC=2
#   SYNTHETIC=1 / VERIFY_EXPERT_CHECKPOINTS=0
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-}:${ROOT}"

MOE_REPO_ROOT="$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/moe_ddp_launch.sh"

mkdir -p "$ROOT/logs"
if [[ "${FLUSH_LOGS:-1}" == "1" ]]; then
  find "$ROOT/logs" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  echo "[run_moe_training_resilient] flushed $ROOT/logs"
fi

TS="$(date +%Y%m%d_%H%M%S)"
GPU_LOG="$ROOT/logs/nvidia_smi_${TS}.log"
TRAIN_LOG="$ROOT/logs/moe_train_${TS}.log"
GPU_INTERVAL="${GPU_INTERVAL:-5}"

cleanup() {
  if [[ -n "${GPU_MON_PID:-}" ]] && kill -0 "$GPU_MON_PID" 2>/dev/null; then
    kill "$GPU_MON_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if command -v nvidia-smi >/dev/null 2>&1 && [[ "${GPU_MONITOR:-1}" == "1" ]]; then
  (
    while true; do
      echo "=== $(date -Iseconds 2>/dev/null || date) ==="
      nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader
      sleep "$GPU_INTERVAL"
    done
  ) >> "$GPU_LOG" 2>&1 &
  GPU_MON_PID=$!
  echo "[run_moe_training_resilient] GPU log: $GPU_LOG (pid $GPU_MON_PID)" | tee "$TRAIN_LOG"
else
  echo "[run_moe_training_resilient] GPU monitor skipped" | tee "$TRAIN_LOG"
fi

NPROC="${NPROC:-2}"
CKPT_DIR="${CKPT_DIR:-$ROOT/checkpoints/moe/training_state}"
mkdir -p "$CKPT_DIR" "$ROOT/DUMMY/data/processed" "$ROOT/DUMMY/data/transformed"

DUMMY_PROC="$ROOT/DUMMY/data/processed/moe_dummy_1k.parquet"
if [[ "${SYNTHETIC:-0}" == "1" ]]; then
  python3 "$ROOT/scripts/prepare_moe_dummy_data.py" --repo-root "$ROOT" --synthetic --n-rows "${N_ROWS:-1000}" >>"$TRAIN_LOG" 2>&1
elif [[ ! -f "$DUMMY_PROC" ]]; then
  if [[ -f "$ROOT/data/processed/all-data.parquet" && -f "$ROOT/data/transformed/all-data.parquet" ]]; then
    python3 "$ROOT/scripts/prepare_moe_dummy_data.py" --repo-root "$ROOT" --source-stem "${SOURCE_STEM:-all-data}" --n-rows "${N_ROWS:-1000}" >>"$TRAIN_LOG" 2>&1
  else
    echo "Missing dummy data; set SYNTHETIC=1 or prepare parquets." | tee -a "$TRAIN_LOG" >&2
    exit 1
  fi
fi

EXPERTS="${EXPERTS:-$ROOT/Code/thesis/config/moe/experts_smoke_2label.json}"
if [[ "${VERIFY_EXPERT_CHECKPOINTS:-1}" != "0" ]]; then
  python3 - "$ROOT" "$EXPERTS" << 'PY' >>"$TRAIN_LOG" 2>&1 || exit 1
import json, sys
from pathlib import Path
repo, exp = Path(sys.argv[1]), Path(sys.argv[2])
spec = json.loads(exp.read_text(encoding="utf-8"))
for e in spec:
    ck = Path(e["checkpoint"])
    if not ck.is_absolute():
        ck = repo / ck
    assert ck.is_file(), ck
PY
fi

MOE_ARGS=(
  --experts_json "$EXPERTS"
  --dataset_stem "${DATASET_STEM:-moe_dummy_1k}"
  --data_root "${DATA_ROOT:-$ROOT/DUMMY/data}"
  --epochs "${EPOCHS:-1}"
  --batch_size "${BATCH_SIZE:-4}"
  --max_samples "${MAX_SAMPLES:-1000}"
  --gate-hidden-dim "${GATE_HIDDEN_DIM:-0}"
  --lora-preset "${LORA_PRESET:-tiny10k}"
  --sync-labels
  --checkpoint-dir "$CKPT_DIR"
  --save-every-steps "${SAVE_EVERY_STEPS:-100}"
  --out_path "$ROOT/checkpoints/moe/gate_${DATASET_STEM:-moe_dummy_1k}_train.safetensors"
)
if [[ -n "${AUTO_BATCH_VRAM_TARGET:-}" ]]; then
  MOE_ARGS+=(--auto-batch-vram-target "$AUTO_BATCH_VRAM_TARGET")
fi
if [[ "${RESUME:-0}" == "1" ]]; then
  MOE_ARGS+=(--resume)
fi
EXTRA=( ${EXTRA_ARGS:-} )

echo "[run_moe_training_resilient] train log: $TRAIN_LOG" | tee -a "$TRAIN_LOG"
if [[ "$NPROC" -le 1 ]]; then
  python3 "$ROOT/Code/thesis/train/train_moe.py" "${MOE_ARGS[@]}" "${EXTRA[@]}" 2>&1 | tee -a "$TRAIN_LOG"
else
  moe_ddp_launch "$NPROC" "$ROOT/Code/thesis/train/train_moe.py" "${MOE_ARGS[@]}" "${EXTRA[@]}" 2>&1 | tee -a "$TRAIN_LOG"
fi

echo "[run_moe_training_resilient] done. Resume: RESUME=1 CKPT_DIR=$CKPT_DIR" | tee -a "$TRAIN_LOG"

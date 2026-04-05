#!/usr/bin/env bash
# Run train_moe for every expert manifest under Code/thesis/config/moe/ (excluding example_experts.json).
# Intended: 2-label and 3-label all-data configs; separate log per manifest under TEMP/logs.
#
# Defaults (full merged corpus):
#   DATA_ROOT=$ROOT/data
#   DATASET_STEM=all-data
#   No max_samples cap (omit --max_samples)
#
# Env:
#   MOE_MATRIX_PROFILE — all_data (default: dense feature-encoder experts only, 2+3 label) or
#     with_distilbert (DistilBERT + FFN like old all_data) or full (every experts_*.json except example/with_distilbert)
#   MOE_MANIFEST_LIST — space-separated basenames or paths (optional); default = all experts_*.json
#   MOE_SKIP_MANIFESTS — regex or space list to skip (optional)
#   VERIFY_EXPERT_CHECKPOINTS — 1 validate ckpts (default 1)
#   NPROC — torchrun ranks (default 1)
#   EPOCHS BATCH_SIZE SAVE_EVERY_STEPS AUTO_BATCH_VRAM_TARGET RESUME EXTRA_ARGS
#   FLUSH_LOGS — if 1, rm TEMP/logs/moe_matrix_* before run
#   SYNC_LABELS — default 1 (--sync-labels)
#
# Per-manifest checkpoint dirs: checkpoints/moe/training_state_<manifest_stem>/
# Per-manifest gate: checkpoints/moe/gate_<manifest_stem>.safetensors
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-}:${ROOT}"

MOE_REPO_ROOT="$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/moe_ddp_launch.sh"

MOE_DIR="$ROOT/Code/thesis/config/moe"
LOG_PREFIX="${LOG_PREFIX:-moe_matrix}"
TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ROOT/logs"

if [[ "${FLUSH_LOGS:-0}" == "1" ]]; then
  rm -f "$ROOT/logs/${LOG_PREFIX}"_*.log
  echo "[moe_matrix] flushed $ROOT/logs/${LOG_PREFIX}_*.log"
fi

shopt -s nullglob
MANIFESTS=()
if [[ -n "${MOE_MANIFEST_LIST:-}" ]]; then
  for item in $MOE_MANIFEST_LIST; do
    if [[ -f "$item" ]]; then
      MANIFESTS+=( "$item" )
    elif [[ -f "$MOE_DIR/$item" ]]; then
      MANIFESTS+=( "$MOE_DIR/$item" )
    else
      echo "Manifest not found: $item" >&2
      exit 1
    fi
  done
elif [[ "${MOE_MATRIX_PROFILE:-all_data}" == "all_data" ]]; then
  MANIFESTS=(
    "$MOE_DIR/experts_all_data_2label.json"
    "$MOE_DIR/experts_all_data_3label.json"
  )
elif [[ "${MOE_MATRIX_PROFILE:-}" == "with_distilbert" ]]; then
  MANIFESTS=(
    "$MOE_DIR/experts_all_data_2label_with_distilbert.json"
    "$MOE_DIR/experts_all_data_3label_with_distilbert.json"
  )
else
  for f in "$MOE_DIR"/experts_*.json; do
    b="$(basename "$f")"
    [[ "$b" == "example_experts.json" ]] && continue
    [[ "$b" == *_with_distilbert.json ]] && continue
    MANIFESTS+=( "$f" )
  done
fi
shopt -u nullglob

if [[ ${#MANIFESTS[@]} -eq 0 ]]; then
  echo "No manifests under $MOE_DIR (experts_*.json)." >&2
  exit 1
fi

DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
DATASET_STEM="${DATASET_STEM:-all-data}"
NPROC="${NPROC:-1}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SAVE_STEPS="${SAVE_EVERY_STEPS:-200}"
SYNC=()
[[ "${SYNC_LABELS:-1}" == "1" ]] && SYNC=(--sync-labels)

MAX_ARG=()
[[ -n "${MAX_SAMPLES:-}" ]] && MAX_ARG=(--max_samples "$MAX_SAMPLES")

AUTO=()
if [[ -n "${AUTO_BATCH_VRAM_TARGET:-}" ]]; then
  AUTO=(
    --auto-batch-vram-target "$AUTO_BATCH_VRAM_TARGET"
    --auto-batch-min "${AUTO_BATCH_MIN:-1}"
    --auto-batch-max "${AUTO_BATCH_MAX:-4096}"
  )
fi

RESUME_ARG=()
[[ "${RESUME:-0}" == "1" ]] && RESUME_ARG=(--resume)

EXTRA=( ${EXTRA_ARGS:-} )

for EXP in "${MANIFESTS[@]}"; do
  base="$(basename "$EXP" .json)"
  LOG="$ROOT/logs/${LOG_PREFIX}_${base}_${TS}.log"
  CKPT_DIR="$ROOT/checkpoints/moe/training_state_${base}"
  OUT_GATE="$ROOT/checkpoints/moe/gate_${base}.safetensors"

  echo "========================================" | tee "$LOG"
  echo "[moe_matrix] $(date -Iseconds 2>/dev/null || date) manifest=$EXP" | tee -a "$LOG"
  echo "[moe_matrix] data_root=$DATA_ROOT dataset_stem=$DATASET_STEM epochs=$EPOCHS batch=$BATCH_SIZE nproc=$NPROC" | tee -a "$LOG"

  if [[ "${VERIFY_EXPERT_CHECKPOINTS:-1}" != "0" ]]; then
    python3 - "$ROOT" "$EXP" << 'PY' 2>&1 | tee -a "$LOG" || exit 1
import json, sys
from pathlib import Path
repo, exp = Path(sys.argv[1]), Path(sys.argv[2])
spec = json.loads(exp.read_text(encoding="utf-8"))
for e in spec:
    ck = Path(e["checkpoint"])
    if not ck.is_absolute():
        ck = repo / ck
    if not ck.is_file():
        raise SystemExit(f"Missing checkpoint: {ck}")
PY
  fi

  MOE_ARGS=(
    --experts_json "$EXP"
    --dataset_stem "$DATASET_STEM"
    --data_root "$DATA_ROOT"
    --epochs "$EPOCHS"
    --batch_size "$BATCH_SIZE"
    "${MAX_ARG[@]}"
    --gate-hidden-dim "${GATE_HIDDEN_DIM:-0}"
    --lora-preset "${LORA_PRESET:-tiny10k}"
    "${SYNC[@]}"
    --checkpoint-dir "$CKPT_DIR"
    --save-every-steps "$SAVE_STEPS"
    --out_path "$OUT_GATE"
    "${AUTO[@]}"
    "${RESUME_ARG[@]}"
  )

  if [[ "$NPROC" -le 1 ]]; then
    python3 "$ROOT/Code/thesis/train/train_moe.py" "${MOE_ARGS[@]}" "${EXTRA[@]}" 2>&1 | tee -a "$LOG"
  else
    echo "[moe_matrix] DDP launcher: $(command -v torchrun 2>/dev/null || echo python3 -m torch.distributed.run)" | tee -a "$LOG"
    moe_ddp_launch "$NPROC" "$ROOT/Code/thesis/train/train_moe.py" "${MOE_ARGS[@]}" "${EXTRA[@]}" 2>&1 | tee -a "$LOG"
  fi

  echo "[moe_matrix] done $base log=$LOG gate=$OUT_GATE" | tee -a "$LOG"
done

echo "[moe_matrix] all manifests finished."

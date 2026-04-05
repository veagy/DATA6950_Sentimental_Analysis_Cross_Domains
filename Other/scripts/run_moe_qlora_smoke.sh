#!/usr/bin/env bash
# MoE gate smoke: optional torchrun DDP, tiny LoRA preset, logs under ROOT/logs only.
#
# Dual L40S (example):
#   export CUDA_VISIBLE_DEVICES=0,1
#   export MASTER_ADDR=127.0.0.1
#   export MASTER_PORT=29505
#   NPROC=2 bash scripts/run_moe_qlora_smoke.sh
#
# Single GPU / CPU:
#   NPROC=1 bash scripts/run_moe_qlora_smoke.sh
#
# Env:
#   EXPERTS   — path to experts JSON (default: Code/thesis/config/moe/experts_smoke_2label.json)
#   NPROC     — torchrun --nproc_per_node (default 1; use 2 on dual L40S)
#   SYNTHETIC — if 1 and DUMMY parquets missing, use synthetic data
#   EXTRA_ARGS — extra CLI passed to train_moe.py
#   VERIFY_EXPERT_CHECKPOINTS — set to 0 to skip JSON checkpoint file checks (random-init experts)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-}:${ROOT}"

MOE_REPO_ROOT="$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/moe_ddp_launch.sh"

mkdir -p "$ROOT/logs" "$ROOT/DUMMY/data/processed" "$ROOT/DUMMY/data/transformed"
NPROC="${NPROC:-1}"
LOG="$ROOT/logs/moe_smoke_${NPROC}proc_$(date +%Y%m%d_%H%M%S).log"

DUMMY_PROC="$ROOT/DUMMY/data/processed/moe_dummy_1k.parquet"
if [[ "${SYNTHETIC:-0}" == "1" ]]; then
  python3 "$ROOT/scripts/prepare_moe_dummy_data.py" --repo-root "$ROOT" --synthetic --n-rows "${N_ROWS:-1000}" || exit $?
elif [[ ! -f "$DUMMY_PROC" ]]; then
  if [[ -f "$ROOT/data/processed/all-data.parquet" && -f "$ROOT/data/transformed/all-data.parquet" ]]; then
    python3 "$ROOT/scripts/prepare_moe_dummy_data.py" --repo-root "$ROOT" --source-stem "${SOURCE_STEM:-all-data}" --n-rows "${N_ROWS:-1000}" || exit $?
  else
    echo "Missing $DUMMY_PROC. Run: bash scripts/prepare_moe_dummy_parquet.sh" >&2
    echo "Or set SYNTHETIC=1 for synthetic parquets." >&2
    exit 1
  fi
fi

EXPERTS="${EXPERTS:-$ROOT/Code/thesis/config/moe/experts_smoke_2label.json}"
if [[ ! -f "$EXPERTS" ]]; then
  echo "Experts JSON not found: $EXPERTS" >&2
  exit 1
fi

if [[ "${VERIFY_EXPERT_CHECKPOINTS:-1}" != "0" ]]; then
  python3 - "$ROOT" "$EXPERTS" << 'PY' || exit 1
import json, sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
exp = Path(sys.argv[2]).resolve()
with open(exp, encoding="utf-8") as f:
    spec = json.load(f)
for e in spec:
    ck = Path(e["checkpoint"])
    if not ck.is_absolute():
        ck = repo / ck
    if not ck.is_file():
        print("Missing expert checkpoint:", ck, file=sys.stderr)
        sys.exit(2)
PY
else
  echo "[run_moe_qlora_smoke] VERIFY_EXPERT_CHECKPOINTS=0 — skipping expert .safetensors check" >&2
fi

MOE_ARGS=(
  --experts_json "$EXPERTS"
  --dataset_stem "${DATASET_STEM:-moe_dummy_1k}"
  --data_root "$ROOT/DUMMY/data"
  --epochs "${EPOCHS:-1}"
  --batch_size "${BATCH_SIZE:-4}"
  --max_samples "${MAX_SAMPLES:-1000}"
  --gate-hidden-dim "${GATE_HIDDEN_DIM:-0}"
  --lora-preset "${LORA_PRESET:-tiny10k}"
  --sync-labels
  --out_path "$ROOT/checkpoints/moe/gate_${DATASET_STEM:-moe_dummy_1k}_smoke.safetensors"
)
# shellcheck disable=2206
EXTRA=( ${EXTRA_ARGS:-} )

if [[ "$NPROC" -le 1 ]]; then
  echo "[run_moe_qlora_smoke] single process (log: $LOG)" | tee "$LOG"
  python3 "$ROOT/Code/thesis/train/train_moe.py" "${MOE_ARGS[@]}" "${EXTRA[@]}" 2>&1 | tee -a "$LOG"
else
  echo "[run_moe_qlora_smoke] DDP x${NPROC} (log: $LOG)" | tee "$LOG"
  moe_ddp_launch "$NPROC" "$ROOT/Code/thesis/train/train_moe.py" "${MOE_ARGS[@]}" "${EXTRA[@]}" 2>&1 | tee -a "$LOG"
fi

#!/usr/bin/env bash
# docs/ml Track A: train E_ML1–E_ML4 (LR, LinearSVC, DecisionTree, RandomForest) on
# data/transformed/{dataset_stem}.parquet — default dataset_stem=all-data.
# DT/RF checkpoints are .joblib (full state); LR/SVC remain .safetensors.
#
# Usage (from TEMP = repo root):
#   bash scripts/run_ml_tabular_track_a.sh
#   THESIS_DATA_ROOT=... THESIS_PYTHON=... bash scripts/run_ml_tabular_track_a.sh
#
# Logs: logs/ml_tabular_track_a.log (override THESIS_ML_TABULAR_LOG) plus per-job
# logs/ml_tabular_{dataset_stem}_{2|3}label_{config_stem}.log under logs/.
# Optional: THESIS_MAX_SAMPLES — passed as --max_samples to train_single (smoke / subsample).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
PY="${THESIS_PYTHON:-python3}"
[[ -x "$ROOT/.venv/bin/python" ]] && PY="$ROOT/.venv/bin/python"

export PYTHONPATH="${PYTHONPATH:-}:$ROOT"
LOG="${THESIS_ML_TABULAR_LOG:-$ROOT/logs/ml_tabular_track_a.log}"
DS="${THESIS_ML_DATASET_STEM:-all-data}"
DATA_ROOT="${THESIS_DATA_ROOT:-$ROOT/data}"
CKPT_ROOT="${THESIS_CHECKPOINT_ROOT:-$ROOT/checkpoints}"
MAX_SAMPLES_ARGS=()
if [[ -n "${THESIS_MAX_SAMPLES:-}" ]]; then
  MAX_SAMPLES_ARGS=(--max_samples "${THESIS_MAX_SAMPLES}")
fi

mkdir -p "$(dirname "$LOG")" "$ROOT/logs"

run_one() {
  local labels="$1" cfg="$2"
  local cfg_stem
  cfg_stem="$(basename "$cfg" .json)"
  local jlog="$ROOT/logs/ml_tabular_${DS}_${labels}label_${cfg_stem}.log"
  echo "==== $(date -Iseconds 2>/dev/null || date) ML tabular labels=$labels $(basename "$cfg") dataset_stem=$DS ====" | tee -a "$LOG" | tee -a "$jlog"
  "$PY" "$ROOT/Code/thesis/train/train_single.py" \
    --config "$cfg" \
    --dataset_stem "$DS" \
    --n_classes "$labels" \
    --data_root "$DATA_ROOT" \
    --checkpoint_root "$CKPT_ROOT" \
    --log_dir "$ROOT/logs" \
    --phase finetune \
    --epochs_pretrain 0 \
    --epochs_finetune 1 \
    "${MAX_SAMPLES_ARGS[@]}" \
    2>&1 | tee -a "$LOG" | tee -a "$jlog"
}

run_one 3 "$ROOT/Code/thesis/config/ml/3_labels/E_ML1_LogisticRegression.json"
run_one 3 "$ROOT/Code/thesis/config/ml/3_labels/E_ML2_LinearSVC.json"
run_one 3 "$ROOT/Code/thesis/config/ml/3_labels/E_ML3_DecisionTreeClassifier.json"
run_one 3 "$ROOT/Code/thesis/config/ml/3_labels/E_ML4_RandomForestClassifier.json"
run_one 2 "$ROOT/Code/thesis/config/ml/2_labels/E_ML1_LogisticRegression.json"
run_one 2 "$ROOT/Code/thesis/config/ml/2_labels/E_ML2_LinearSVC.json"
run_one 2 "$ROOT/Code/thesis/config/ml/2_labels/E_ML3_DecisionTreeClassifier.json"
run_one 2 "$ROOT/Code/thesis/config/ml/2_labels/E_ML4_RandomForestClassifier.json"

echo "[ml-tabular-track-a] done. Checkpoints: LR/SVC .safetensors; DT/RF .joblib under $CKPT_ROOT/{2,3}-labels/$DS/" | tee -a "$LOG"

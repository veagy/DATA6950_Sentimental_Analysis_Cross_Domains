#!/usr/bin/env bash
# Frozen-backbone finetune for transformers + HRM (processed parquets except all-data).
# HRM uses Code/thesis/config/hrm/E_HRM1_4Level.json with --n_classes 2 then 3 per stem.
# Repo root: bash scripts/run_thesis_finetune_text.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
PY="${THESIS_PYTHON:-python3}"

export THESIS_RESUME_TEMP="${THESIS_RESUME_TEMP:-$ROOT/logs/resume}"
mkdir -p "$ROOT/logs" "$THESIS_RESUME_TEMP"

NPROC="${THESIS_NPROC_PER_NODE:-2}"
TXT_BS="${THESIS_TEXT_FINETUNE_BATCH:-4}"

mapfile -t CONFIGS < <(find "$ROOT/Code/thesis/config/transformers" "$ROOT/Code/thesis/config/hrm" -name '*.json' 2>/dev/null | sort -u)

run_inner() {
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
  if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    echo "No configs under Code/thesis/config/transformers or hrm." >&2
    exit 1
  fi
  for pq in "$ROOT/data/processed"/*.parquet; do
    [[ -f "$pq" ]] || continue
    stem=$(basename "$pq" .parquet)
    [[ "$stem" == "all-data" ]] && continue
    for cfg in "${CONFIGS[@]}"; do
      [[ "$cfg" == *"/moe/"* ]] && continue
      if [[ "$(basename "$cfg")" == "E_HRM1_4Level.json" ]] && [[ "$cfg" == *"/config/hrm/"* ]]; then
        for nc in 2 3; do
          echo "[finetune] $stem $(basename "$cfg") n_classes=$nc"
          "$PY" -m torch.distributed.run --nproc_per_node="$NPROC" \
            "$ROOT/Code/thesis/train/train_single.py" \
            --config "$cfg" \
            --n_classes "$nc" \
            --dataset_stem "$stem" \
            --data_root "$ROOT/data" \
            --checkpoint_root "$ROOT/checkpoints" \
            --log_dir "$ROOT/logs" \
            --phase finetune \
            --pretrain_text_source dataset \
            --epochs_pretrain 0 \
            --epochs_finetune "${THESIS_EPOCHS_TEXT_FINETUNE:-3}" \
            --batch_size "$TXT_BS" \
            --gc_every 50
        done
        continue
      fi
      echo "[finetune] $stem $(basename "$cfg")"
      "$PY" -m torch.distributed.run --nproc_per_node="$NPROC" \
        "$ROOT/Code/thesis/train/train_single.py" \
        --config "$cfg" \
        --dataset_stem "$stem" \
        --data_root "$ROOT/data" \
        --checkpoint_root "$ROOT/checkpoints" \
        --log_dir "$ROOT/logs" \
        --phase finetune \
        --pretrain_text_source dataset \
        --epochs_pretrain 0 \
        --epochs_finetune "${THESIS_EPOCHS_TEXT_FINETUNE:-3}" \
        --batch_size "$TXT_BS" \
        --gc_every 50
    done
  done
}

if [[ "${1:-}" == "__inner" ]]; then
  run_inner
  exit 0
fi

DETACH="${THESIS_DETACH:-none}"
SESS="${THESIS_SESSION:-thesis_finetune_text}"
LOG="$ROOT/logs/finetune_text_pipeline.log"

case "$DETACH" in
  tmux)
    tmux new-session -d -s "$SESS" "bash \"$SCRIPT_DIR/run_thesis_finetune_text.sh\" __inner"
    echo "tmux session $SESS"
    ;;
  screen)
    screen -dmS "$SESS" bash "$SCRIPT_DIR/run_thesis_finetune_text.sh" __inner
    echo "screen $SESS"
    ;;
  nohup)
    nohup bash "$SCRIPT_DIR/run_thesis_finetune_text.sh" __inner >>"$LOG" 2>&1 &
    echo "nohup PID $! — tail -f $LOG"
    ;;
  *)
    run_inner
    ;;
esac

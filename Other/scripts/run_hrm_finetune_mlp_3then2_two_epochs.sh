#!/usr/bin/env bash
# 3-label MLP finetune (2 epochs) → clean exit + final .safetensors → 2-label (2 epochs).
# Logs: logs/hrm_finetune_mlp_3label.log and logs/hrm_finetune_mlp_2label.log
#
# Default: THESIS_DETACH=none inside phases so this script blocks (wrap in one outer tmux for SSH-safe runs).
#   cd /path/to/TEMP && bash scripts/run_hrm_finetune_mlp_3then2_two_epochs.sh
#   tmux new-session -d -s hrm_mlp_3then2 "cd /path/to/TEMP && bash scripts/run_hrm_finetune_mlp_3then2_two_epochs.sh"
#
# Env:
#   THESIS_EPOCHS_FINETUNE_3LABEL, THESIS_EPOCHS_FINETUNE_2LABEL (default 2 each)
#   THESIS_PYTHON, CUDA_VISIBLE_DEVICES, THESIS_HRM_FINETUNE_BATCH, THESIS_HRM_FINETUNE_SHARDED, etc. (passed through)
#   THESIS_NO_RESUME=1 — skip live-resume temp state for each phase (optional)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

E3="${THESIS_EPOCHS_FINETUNE_3LABEL:-2}"
E2="${THESIS_EPOCHS_FINETUNE_2LABEL:-2}"
export THESIS_HRM_FINETUNE_SHARDED="${THESIS_HRM_FINETUNE_SHARDED:-0}"

LOG3="${ROOT}/logs/hrm_finetune_mlp_3label.log"
LOG2="${ROOT}/logs/hrm_finetune_mlp_2label.log"
mkdir -p "${ROOT}/logs"

{
  echo "==== $(date -Iseconds 2>/dev/null || date) [3then2-two-ep] pipeline start 3-label_epochs=${E3} then 2-label_epochs=${E2} ===="
} >>"$LOG3"

echo "[3then2-two-ep] phase 1/2: 3-label, epochs=${E3} (log tail: $LOG3)"
export THESIS_EPOCHS_FINETUNE="$E3"
export THESIS_HRM_FINETUNE_LOG="$LOG3"
export THESIS_DETACH=none
bash "$SCRIPT_DIR/run_hrm_finetune_mlp.sh" 3
echo "[3then2-two-ep] phase 1 done (checkpoints under checkpoint_root/fine-tune/.../3-labels/)."

echo "[3then2-two-ep] phase 2/2: 2-label, epochs=${E2} (log tail: $LOG2)"
export THESIS_EPOCHS_FINETUNE="$E2"
export THESIS_HRM_FINETUNE_LOG="$LOG2"
bash "$SCRIPT_DIR/run_hrm_finetune_mlp.sh" 2
echo "[3then2-two-ep] phase 2 done."

{
  echo "==== $(date -Iseconds 2>/dev/null || date) [3then2-two-ep] pipeline complete ===="
} >>"$LOG2"

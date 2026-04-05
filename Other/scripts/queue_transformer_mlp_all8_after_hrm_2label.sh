#!/usr/bin/env bash
# Wait until the current HRM 2-label MLP finetune (train_single workers on
# E_HRM1_4Level_ft_mlp_2label.json) has fully exited, then start the eight
# transformer MLP finetunes in a new detached tmux session.
#
# Does not signal or attach to the running HRM job — polling only.
#
# Env:
#   THESIS_PYTHON — passed into the transformer run (default: $ROOT/.venv/bin/python)
#   THESIS_QUEUE_POLL_SEC — sleep between checks (default: 60)
#   THESIS_QUEUE_AFTER_PGPATTERN — extended regex for pgrep -f (default: HRM 2-label finetune)
#   THESIS_QUEUE_TMUX_SESSION — tmux session for transformer run (default: transformer_ft_all8)
#   Extra env vars are forwarded into the tmux command via inline export (set before launching this script).
#
# Typical use (from anywhere):
#   nohup bash /path/to/TEMP/scripts/queue_transformer_mlp_all8_after_hrm_2label.sh \
#     >> /path/to/TEMP/logs/queue_transformer_after_hrm_2l.log 2>&1 &

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

POLL="${THESIS_QUEUE_POLL_SEC:-60}"
PATTERN="${THESIS_QUEUE_AFTER_PGPATTERN:-train_single\\.py.*E_HRM1_4Level_ft_mlp_2label\\.json}"
SESS="${THESIS_QUEUE_TMUX_SESSION:-transformer_ft_all8}"
PY="${THESIS_PYTHON:-$ROOT/.venv/bin/python}"
LOG_TAG="[queue-transformer-after-hrm-2l]"

log() {
  echo "$(date -Iseconds 2>/dev/null || date) $LOG_TAG $*"
}

log "waiting for no pgrep match: $PATTERN (poll ${POLL}s)"
while pgrep -f "$PATTERN" >/dev/null 2>&1; do
  log "HRM 2-label finetune still running"
  sleep "$POLL"
done

log "HRM 2-label workers gone; brief pause before GPU handoff"
sleep 15

if tmux has-session -t "$SESS" 2>/dev/null; then
  log "ERROR: tmux session already exists: $SESS — remove it or set THESIS_QUEUE_TMUX_SESSION" >&2
  exit 1
fi

# Inline cd so child scripts see correct repo layout; match prior HRM tmux style.
CMD="cd $(printf '%q' "$ROOT") && export THESIS_PYTHON=$(printf '%q' "$PY") && exec bash scripts/run_transformer_mlp_finetune_4xl40s_sequential_all8.sh"
tmux new-session -d -s "$SESS" bash -lc "$CMD"

log "started transformer all-8 in tmux session: $SESS (attach: tmux attach -t $SESS)"

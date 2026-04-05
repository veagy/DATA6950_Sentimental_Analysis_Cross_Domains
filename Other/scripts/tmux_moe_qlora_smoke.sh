#!/usr/bin/env bash
# Detached tmux session running MoE QLoRA/smoke training (see run_moe_qlora_smoke.sh).
# Usage: bash scripts/tmux_moe_qlora_smoke.sh
# Attach: tmux attach -t moe_smoke
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${MOE_TMUX_SESSION:-moe_smoke}"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists. Attach with: tmux attach -t $SESSION" >&2
  exit 1
fi
tmux new-session -d -s "$SESSION" "cd '$ROOT' && bash ./scripts/run_moe_qlora_smoke.sh; echo; echo 'Done. Press Enter to close.'; read _"
echo "Started tmux session: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Logs:   $ROOT/logs/moe_smoke_*.log"

#!/usr/bin/env bash
# Stop all thesis training processes rooted at this TEMP checkout (torchrun + train_single + train_queue).
# Invoked as a file so interactive tools don't match pkill -f on their own command line.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
for sig in TERM TERM KILL; do
  pkill -$sig -f "${PY}.*train_queue\.py" 2>/dev/null || true
  pkill -$sig -f "${PY}.*torch\.distributed\.run.*train_single" 2>/dev/null || true
  pkill -$sig -f "${PY} -u .*train_single\.py" 2>/dev/null || true
  sleep 1
done
echo "[stop-temp-train] done (best-effort)."

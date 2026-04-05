#!/usr/bin/env bash
# Run from TEMP repo root (parent of output/).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
if command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi
"$PY" "$ROOT/output/scripts/generate_all_charts.py" --repo-root "$ROOT" "$@"

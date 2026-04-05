#!/usr/bin/env bash
# Sequential 4×L40 finetune with HF_TOKEN / HUGGINGFACE_HUB_TOKEN from repo-root .env (if present).
# Does not print the token.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$ROOT/.env"
if [[ -f "$ENV_FILE" ]]; then
  PY="${ROOT}/.venv/bin/python"
  [[ -x "$PY" ]] || PY=python3
  HF_TOKEN="$("$PY" -c 'import re, sys
from pathlib import Path
for line in Path(sys.argv[1]).read_text().splitlines():
    m = re.match("^\\s*HF_TOKEN\\s*=\\s*(.+)$", line)
    if m:
        print(m.group(1).strip())
        break
' "$ENV_FILE")"
  if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN
    export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}"
  fi
fi
exec bash "$SCRIPT_DIR/run_hrm_finetune_mlp_4xl40_sequential.sh" "$@"

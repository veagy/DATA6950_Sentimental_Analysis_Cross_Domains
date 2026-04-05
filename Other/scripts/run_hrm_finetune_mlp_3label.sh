#!/usr/bin/env bash
# Thin wrapper: 3-label HRM MLP finetune. See run_hrm_finetune_mlp.sh.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_hrm_finetune_mlp.sh" 3

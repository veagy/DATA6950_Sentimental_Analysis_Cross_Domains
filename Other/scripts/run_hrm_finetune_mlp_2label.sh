#!/usr/bin/env bash
# Thin wrapper: 2-label HRM MLP finetune (drops neutral rows). See run_hrm_finetune_mlp.sh.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_hrm_finetune_mlp.sh" 2

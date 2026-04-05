#!/usr/bin/env bash
# Wrapper: 2-class transformer MLP head finetune.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_transformer_mlp_finetune.sh" 2 "$@"

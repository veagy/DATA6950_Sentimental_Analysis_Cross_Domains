#!/usr/bin/env bash
# Cloud / dual-GPU: HRM encoder-only MLM — interrupt mid-epoch and resume (DDP).
# Repo root. Requires: 2 GPUs, same env as run_hrm_encoder_pretrain_only.sh, HF cache.
#
# This script does NOT auto-kill (too environment-specific). Copy-paste pattern:
#
# 1) Terminal A — start training (tmux recommended):
#    export CUDA_VISIBLE_DEVICES=0,1
#    export THESIS_EPOCHS_PRETRAIN=1
#    export THESIS_SAVE_EVERY_STEPS=200
#    export THESIS_DETACH=tmux
#    export THESIS_SESSION=hrm_ddp_resume_test
#    bash scripts/run_hrm_encoder_pretrain_only.sh
#
# 2) After you see several hundred steps in logs/hrm_encoder_pretrain_only.log, send SIGTERM to the
#    train_single parent (or close tmux pane). Verify logs/resume/.../meta.json exists.
#
# 3) Relaunch the SAME command (same THESIS_RESUME_TEMP, data_root, checkpoint_root, stem).
#    Training should skip completed batches and continue; final artifact:
#    checkpoints/pretrain/<stem>/E_HRM1_4Level.safetensors
#
# See also: docs/plans/pretrain_scripts_blackwell_resume_3c0bbcf9.plan.md

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
echo "Read the comments in: $SCRIPT_DIR/smoke_hrm_ddp_sigterm_resume.sh"
echo "Repo: $ROOT"

# HRM MLP-head finetune (2- or 3-class). From repo root:
#   .\scripts\run_hrm_finetune_mlp.ps1        # default 3-class
#   .\scripts\run_hrm_finetune_mlp.ps1 2      # 2-class
# Forwards to WSL bash. Defaults: all parquet rows (no THESIS_MAX_SAMPLES), detach via tmux (THESIS_DETACH=tmux).
# Foreground in WSL: wsl bash -c 'export THESIS_DETACH=none; bash scripts/run_hrm_finetune_mlp.sh 3'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
wsl bash scripts/run_hrm_finetune_mlp.sh @args

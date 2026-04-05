# HRM encoder-only MLM pretrain ONLY (no queue). From repo root:
#   .\scripts\run_hrm_encoder_pretrain_only.ps1
# Forwards extra args to WSL bash (e.g. nothing; env vars must be set in WSL or exported).
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
wsl bash scripts/run_hrm_encoder_pretrain_only.sh @args

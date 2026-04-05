# Run from repo root (Windows): merges to WSL bash
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
wsl bash scripts/run_thesis_pretrain.sh @args

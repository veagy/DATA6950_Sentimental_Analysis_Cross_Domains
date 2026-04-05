$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
wsl bash scripts/run_thesis_finetune_text.sh @args

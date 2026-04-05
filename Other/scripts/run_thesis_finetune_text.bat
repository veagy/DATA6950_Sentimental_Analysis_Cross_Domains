@echo off
cd /d "%~dp0.."
wsl bash scripts/run_thesis_finetune_text.sh %*

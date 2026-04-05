@echo off
REM Run pretrain pipeline under WSL from repo root.
cd /d "%~dp0.."
wsl bash scripts/run_thesis_pretrain.sh %*

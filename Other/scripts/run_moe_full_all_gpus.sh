#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-}:${ROOT}"

mkdir -p "$ROOT/logs"
TS="$(date +%Y%m%d_%H%M%S)"
GPU_LOG="$ROOT/logs/nvidia_smi_moe_full_${TS}.log"
GPU_INTERVAL="${GPU_INTERVAL:-5}"

cleanup() {
  if [[ -n "${GPU_MON_PID:-}" ]] && kill -0 "$GPU_MON_PID" 2>/dev/null; then
    kill "$GPU_MON_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if command -v nvidia-smi >/dev/null 2>&1 && [[ "${GPU_MONITOR:-1}" == "1" ]]; then
  {
    echo "# columns: index, memory.used_MiB, memory.free_MiB, memory.total_MiB, util.gpu_%, util.mem_%, temp_C"
    echo "# (nounits MiB from nvidia-smi; avoids ambiguous 0 MiB CSV parsing)"
  } >>"$GPU_LOG"
  (
    while true; do
      echo "=== $(date -Iseconds 2>/dev/null || date) ==="
      nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total,utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader,nounits
      sleep "$GPU_INTERVAL"
    done
  ) >>"$GPU_LOG" 2>&1 &
  GPU_MON_PID=$!
  printf '%s\n' "$GPU_LOG" >"$ROOT/logs/GPU_MONITOR_LATEST_PATH"
  {
    echo "================================================================"
    echo "GPU MONITOR LOG (this run): $GPU_LOG"
    echo "Also saved for: cat $ROOT/logs/GPU_MONITOR_LATEST_PATH"
    echo "All MoE logs live under: $ROOT/logs/"
    echo "================================================================"
  } | tee "$ROOT/logs/moe_full_run_${TS}.txt"
else
  printf '%s\n' "disabled (GPU_MONITOR=0 or nvidia-smi not found)" >"$ROOT/logs/GPU_MONITOR_LATEST_PATH"
  echo "[run_moe_full_all_gpus] GPU_MONITOR=0 or nvidia-smi missing — no GPU log file." | tee "$ROOT/logs/moe_full_run_${TS}.txt"
  echo "(Training logs still go to $ROOT/logs/moe_full_*.log)" | tee -a "$ROOT/logs/moe_full_run_${TS}.txt"
fi

export DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
export DATASET_STEM="${DATASET_STEM:-all-data}"
export EPOCHS="${EPOCHS:-1}"
unset MAX_SAMPLES
export SYNC_LABELS="${SYNC_LABELS:-1}"
export LOG_PREFIX="${LOG_PREFIX:-moe_full}"
export SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-500}"
export GATE_HIDDEN_DIM="${GATE_HIDDEN_DIM:-0}"
# Default none: dense-only expert manifests do not use LLM LoRA. For DistilBERT experts use LORA_PRESET=tiny10k.
export LORA_PRESET="${LORA_PRESET:-none}"

# Per-GPU batch when not using VRAM auto-tuning. MoE runs multiple DistilBERT forwards; ~48GB L40
# class GPUs: try 1024–2048; raise to 4096 if stable, or lower if OOM.
export BATCH_SIZE="${BATCH_SIZE:-1024}"

# MOE_DISABLE_AUTO_BATCH=1 — fixed BATCH_SIZE only (no probe). Default: VRAM-target probe up to 4096.
if [[ "${MOE_DISABLE_AUTO_BATCH:-0}" == "1" ]]; then
  unset AUTO_BATCH_VRAM_TARGET AUTO_BATCH_MIN AUTO_BATCH_MAX
else
  export AUTO_BATCH_VRAM_TARGET="${AUTO_BATCH_VRAM_TARGET:-0.92}"
  export AUTO_BATCH_MIN="${AUTO_BATCH_MIN:-512}"
  export AUTO_BATCH_MAX="${AUTO_BATCH_MAX:-4096}"
fi

export MOE_MATRIX_PROFILE="${MOE_MATRIX_PROFILE:-all_data}"

if command -v nvidia-smi >/dev/null 2>&1; then
  export NPROC="$(nvidia-smi -L 2>/dev/null | wc -l)"
else
  export NPROC=1
fi
NPROC="$(echo "$NPROC" | tr -d '[:space:]')"
if [[ -z "$NPROC" || "$NPROC" -lt 1 ]]; then
  NPROC=1
fi
export NPROC

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29531}"

bash "$ROOT/scripts/run_moe_expert_matrix.sh"
status=$?
echo "[run_moe_full_all_gpus] finished exit=$status" | tee -a "$ROOT/logs/moe_full_run_${TS}.txt"
exit "$status"

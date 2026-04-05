# shellcheck shell=bash
# Source from repo scripts after setting MOE_REPO_ROOT to the TEMP checkout root.
# torchrun is often missing from PATH; pip/venv installs expose the same via python -m.
moe_ddp_launch() {
  local nproc=$1
  shift
  if command -v torchrun >/dev/null 2>&1; then
    torchrun --nproc_per_node="$nproc" "$@"
  elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/torchrun" ]]; then
    "${VIRTUAL_ENV}/bin/torchrun" --nproc_per_node="$nproc" "$@"
  elif [[ -n "${MOE_REPO_ROOT:-}" && -x "${MOE_REPO_ROOT}/.venv/bin/torchrun" ]]; then
    "${MOE_REPO_ROOT}/.venv/bin/torchrun" --nproc_per_node="$nproc" "$@"
  else
    # Single-node multi-GPU: --standalone avoids rdzv setup; same role as torchrun here.
    python3 -m torch.distributed.run --standalone --nproc_per_node="$nproc" "$@"
  fi
}

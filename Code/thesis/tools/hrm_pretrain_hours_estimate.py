#!/usr/bin/env python3
"""
Rough wall-time estimate for HRM MLM epochs on dual-GPU (or any) setup.

  T ≈ num_epochs * ceil(N / (R * B)) * t_step

N = parquet rows (text), R = GPU count, B = per-GPU batch, t_step = seconds per
optimizer step **measured on the target hardware** (not a laptop 4070).

Usage:
  python Code/thesis/tools/hrm_pretrain_hours_estimate.py --rows 9500000 --gpus 2 --batch 64 --t-step 0.5 --epochs 2
  python Code/thesis/tools/hrm_pretrain_hours_estimate.py  # reads row count from data/processed/all-data.parquet if present
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]


def main() -> None:
    ap = argparse.ArgumentParser(description="HRM MLM epoch hours (approximate).")
    ap.add_argument(
        "--parquet",
        type=Path,
        default=_REPO / "data" / "processed" / "all-data.parquet",
        help="Merged processed parquet (row count from metadata).",
    )
    ap.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Override row count N if parquet is missing.",
    )
    ap.add_argument("--gpus", type=int, default=2, help="DDP world size (processes).")
    ap.add_argument("--batch", type=int, default=64, help="Per-GPU batch size.")
    ap.add_argument("--t-step", type=float, default=0.5, help="Measured seconds per step on target GPUs.")
    ap.add_argument("--epochs", type=int, default=2, help="Number of full data passes.")
    args = ap.parse_args()

    n = args.rows
    pq = args.parquet.resolve()
    if n is None:
        if pq.is_file():
            import pyarrow.parquet as pq_mod

            n = pq_mod.ParquetFile(str(pq)).metadata.num_rows
            print(f"Rows N = {n} ({pq})")
        else:
            raise SystemExit(f"No --rows and parquet not found: {pq}")

    r = max(1, args.gpus)
    b = max(1, args.batch)
    per_rank = math.ceil(n / r)
    steps = math.ceil(per_rank / b)
    hours_per_epoch = (steps * args.t_step) / 3600.0
    total = hours_per_epoch * args.epochs

    print(f"Approx steps/epoch/GPU ~ ceil(ceil(N/R)/B) = ceil({per_rank}/{b}) = {steps}")
    print(f"Hours/epoch ~ {steps} * {args.t_step}s / 3600 = {hours_per_epoch:.2f}")
    print(f"Total for {args.epochs} epochs ~ {total:.2f} h (measure t_step on target hardware)")


if __name__ == "__main__":
    main()

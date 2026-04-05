#!/usr/bin/env python3
"""
Sample small Parquet shards from ``data/processed`` and ``data/transformed`` into
``DUMMY/DATA`` for lightweight mock runs.

Run from repository root::

    python Code/thesis/data/create_dummy_data.py
    python Code/thesis/data/create_dummy_data.py --samples 100
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[3]


def create_dummy_data(
    source_dirs: list[Path],
    target_dir: Path,
    *,
    num_samples: int = 50,
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for src_dir in source_dirs:
        if not src_dir.is_dir():
            print(f"[SKIP] missing directory: {src_dir}")
            continue
        for file in sorted(src_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(file)
                if len(df) == 0:
                    print(f"[SKIP] empty: {file}")
                    continue
                n = min(num_samples, len(df))
                sampled_df = df.sample(n=n, random_state=42)
                out_path = target_dir / f"{src_dir.name}_{file.name}"
                sampled_df.to_parquet(out_path, index=False)
                print(f"Created {out_path} with {len(sampled_df)} samples.")
            except Exception as e:
                print(f"Failed to process {file}: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build small parquet samples under DUMMY/DATA")
    ap.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Max rows per source parquet (default: 50)",
    )
    ap.add_argument(
        "--target",
        type=Path,
        default=None,
        help="Output directory (default: <repo>/DUMMY/DATA)",
    )
    args = ap.parse_args()
    target = args.target.resolve() if args.target else (_REPO / "DUMMY" / "DATA")
    sources = [_REPO / "data" / "processed", _REPO / "data" / "transformed"]
    print("Starting creation of dummy data...")
    create_dummy_data(sources, target, num_samples=max(1, args.samples))
    print("Dummy data creation complete.")


if __name__ == "__main__":
    main()

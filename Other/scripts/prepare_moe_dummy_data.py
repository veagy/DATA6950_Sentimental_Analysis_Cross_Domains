#!/usr/bin/env python3
"""
Build aligned MoE smoke parquets under ``DUMMY/data/processed`` and ``DUMMY/data/transformed``.

Default: slice the first N rows from ``{data_root}/processed/{stem}.parquet`` and the same
row range from ``{data_root}/transformed/{stem}.parquet`` (positional alignment; same schema
order as the merged all-data pipeline).

With ``--synthetic``: write N aligned rows without source files (CPU smoke when parquets
or checkpoints are absent elsewhere).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_REPO = _SCRIPT_DIR.parent


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare 1k-row aligned MoE dummy parquets.")
    ap.add_argument("--repo-root", type=Path, default=_DEFAULT_REPO, help="TEMP repo root (default: parent of scripts/).")
    ap.add_argument("--data-root", type=Path, default=None, help="Source data root (default: repo-root/data).")
    ap.add_argument("--source-stem", type=str, default="all-data", help="Stem for source processed/transformed parquets.")
    ap.add_argument("--out-stem", type=str, default="moe_dummy_1k", help="Stem for output parquets under DUMMY/data.")
    ap.add_argument("--n-rows", type=int, default=1000, help="Number of aligned rows to write.")
    ap.add_argument("--synthetic", action="store_true", help="Generate synthetic text/features/labels (no source files).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    data_root = (args.data_root or (repo / "data")).resolve()
    out_proc = repo / "DUMMY" / "data" / "processed" / f"{args.out_stem}.parquet"
    out_trans = repo / "DUMMY" / "data" / "transformed" / f"{args.out_stem}.parquet"
    out_proc.parent.mkdir(parents=True, exist_ok=True)
    out_trans.parent.mkdir(parents=True, exist_ok=True)

    n = int(args.n_rows)
    if n <= 0:
        print("ERROR: --n-rows must be positive", file=sys.stderr)
        return 1

    if args.synthetic:
        rng = np.random.default_rng(int(args.seed))
        texts = [f"synthetic smoke review {i} for moe dummy data." for i in range(n)]
        y = np.array([i % 2 for i in range(n)], dtype=np.int64)
        feats = rng.standard_normal((n, 100)).astype(np.float32)
        dfp = pd.DataFrame({"text": texts, "sentiment_value": y})
        dft = pd.DataFrame({"features_100d": [row.tolist() for row in feats], "sentiment_value": y})
        dfp.to_parquet(out_proc, index=False)
        dft.to_parquet(out_trans, index=False)
        print(f"Wrote synthetic {n} rows:\n  {out_proc}\n  {out_trans}")
        return 0

    src_p = data_root / "processed" / f"{args.source_stem}.parquet"
    src_t = data_root / "transformed" / f"{args.source_stem}.parquet"
    if not src_p.is_file():
        print(f"ERROR: missing processed parquet: {src_p}", file=sys.stderr)
        return 1
    if not src_t.is_file():
        print(f"ERROR: missing transformed parquet: {src_t}", file=sys.stderr)
        return 1

    df_p = pd.read_parquet(src_p)
    df_t = pd.read_parquet(src_t)
    if len(df_p) != len(df_t):
        print(
            f"WARNING: row count mismatch processed={len(df_p)} transformed={len(df_t)}; "
            "slice uses the first min(len) rows for both (verify merge alignment).",
            file=sys.stderr,
        )
    m = min(len(df_p), len(df_t))
    if m < n:
        print(f"ERROR: need at least {n} aligned rows, got {m}", file=sys.stderr)
        return 1

    df_p.head(n).reset_index(drop=True).to_parquet(out_proc, index=False)
    df_t.head(n).reset_index(drop=True).to_parquet(out_trans, index=False)
    print(f"Wrote first {n} aligned rows:\n  {out_proc}\n  {out_trans}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

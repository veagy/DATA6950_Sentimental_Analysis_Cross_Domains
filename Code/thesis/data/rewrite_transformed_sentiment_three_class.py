#!/usr/bin/env python3
"""
Rewrite **every** row of ``data/transformed/all-data.parquet`` so ``sentiment_value`` is only 0, 1, or 2.

Uses the same mapping as ``rewrite_all_data_sentiment_three_class.py`` on processed data:
``normalize_label_for_n_classes(coerce_label_int(y), 3, source_stem=...)`` from
``common.datasets``. Row order must match ``data/processed/all-data.parquet`` (for
``source_stem`` and raw label); ``features_100d`` is copied through unchanged.

Does not modify ``processed/all-data.parquet``.

Run from TEMP::

    python Code/thesis/data/rewrite_transformed_sentiment_three_class.py
    python Code/thesis/data/rewrite_transformed_sentiment_three_class.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_REPO = Path(__file__).resolve().parents[3]
_THESIS = _REPO / "Code" / "thesis"
if str(_THESIS) not in sys.path:
    sys.path.insert(0, str(_THESIS))

from common.datasets import coerce_label_int, normalize_label_for_n_classes  # noqa: E402


def _stem_key(v) -> str | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip()
    return s if s else None


def _map_all_batch(y_proc: np.ndarray, stem_obj: np.ndarray) -> np.ndarray:
    out = np.empty(len(y_proc), dtype=np.int64)
    for i in range(len(y_proc)):
        yi = coerce_label_int(y_proc[i])
        sk = _stem_key(stem_obj[i])
        r = normalize_label_for_n_classes(yi, 3, source_stem=sk)
        out[i] = 0 if r is None else int(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--processed",
        type=Path,
        default=_REPO / "data" / "processed" / "all-data.parquet",
        help="Reference parquet with source_stem + sentiment_value.",
    )
    ap.add_argument(
        "--transformed",
        type=Path,
        default=_REPO / "data" / "transformed" / "all-data.parquet",
        help="Transformed parquet to rewrite.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: atomic replace of --transformed).",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=256_000,
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    proc_path = args.processed.resolve()
    trans_path = args.transformed.resolve()
    if not proc_path.is_file():
        raise SystemExit(f"Missing processed: {proc_path}")
    if not trans_path.is_file():
        raise SystemExit(f"Missing transformed: {trans_path}")

    pf_p = pq.ParquetFile(proc_path)
    pf_t = pq.ParquetFile(trans_path)
    n_p = pf_p.metadata.num_rows
    n_t = pf_t.metadata.num_rows
    if n_p != n_t:
        raise SystemExit(f"Row count mismatch: processed {n_p} vs transformed {n_t}")

    bp = pf_p.iter_batches(
        batch_size=args.batch_size,
        columns=["source_stem", "sentiment_value"],
    )
    bt = pf_t.iter_batches(
        batch_size=args.batch_size,
        columns=["features_100d", "sentiment_value"],
    )

    if args.dry_run:
        b_p = next(bp)
        b_t = next(bt)
        if b_p.num_rows != b_t.num_rows:
            raise SystemExit("First batch row count mismatch")
        y_p = b_p.column("sentiment_value").to_numpy(zero_copy_only=False)
        st = b_p.column("source_stem").to_pandas().to_numpy(dtype=object, copy=False)
        m = _map_all_batch(y_p, st)
        print(f"[dry-run] rows={len(m)}")
        print(f"  mapped unique: {sorted(set(m.tolist()))}")
        return

    out_path = (
        args.output.resolve()
        if args.output
        else trans_path.with_suffix(trans_path.suffix + ".three_class.tmp")
    )
    writer: pq.ParquetWriter | None = None
    total = 0
    try:
        for b_p, b_t in zip(bp, bt):
            if b_p.num_rows != b_t.num_rows:
                raise RuntimeError(
                    f"Batch mismatch: {b_p.num_rows} vs {b_t.num_rows}"
                )
            y_p = b_p.column("sentiment_value").to_numpy(zero_copy_only=False)
            st = b_p.column("source_stem").to_pandas().to_numpy(dtype=object, copy=False)
            feats = b_t.column("features_100d")
            mapped = _map_all_batch(y_p, st)
            new_t = pa.Table.from_arrays(
                [feats, pa.array(mapped, type=pa.int64())],
                names=["features_100d", "sentiment_value"],
            )
            if writer is None:
                writer = pq.ParquetWriter(out_path, new_t.schema, compression="snappy")
            writer.write_table(new_t)
            total += new_t.num_rows
            print(f"  wrote rows total={total}")
    finally:
        if writer is not None:
            writer.close()

    if args.output:
        print(f"Wrote {out_path} ({total} rows)")
        return

    if total != n_p:
        out_path.unlink(missing_ok=True)
        raise SystemExit(f"Expected {n_p} rows, wrote {total}")

    backup = trans_path.with_suffix(trans_path.suffix + ".before_three_class_all_rows.bak")
    if not backup.is_file():
        shutil.copy2(trans_path, backup)
        print(f"Backup: {backup}")
    os.replace(out_path, trans_path)
    print(f"Replaced {trans_path} ({total} rows)")


if __name__ == "__main__":
    main()

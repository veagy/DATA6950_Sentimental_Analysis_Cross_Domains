#!/usr/bin/env python3
"""
Append ``source_stem`` to ``data/transformed/all-data.parquet`` from aligned rows in
``data/processed/all-data.parquet`` (same row count / order).

Does not read or write ``processed/all-data.parquet`` except to read ``source_stem``.

Run from TEMP::

    python Code/thesis/data/add_source_stem_to_transformed_parquet.py
    python Code/thesis/data/add_source_stem_to_transformed_parquet.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_REPO = Path(__file__).resolve().parents[3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--processed",
        type=Path,
        default=_REPO / "data" / "processed" / "all-data.parquet",
    )
    ap.add_argument(
        "--transformed",
        type=Path,
        default=_REPO / "data" / "transformed" / "all-data.parquet",
    )
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=256_000)
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

    tcols = set(pf_t.schema_arrow.names)
    if "source_stem" in tcols:
        raise SystemExit("transformed parquet already has source_stem")

    bp = pf_p.iter_batches(batch_size=args.batch_size, columns=["source_stem"])
    bt = pf_t.iter_batches(
        batch_size=args.batch_size,
        columns=["features_100d", "sentiment_value"],
    )

    if args.dry_run:
        b_p = next(bp)
        b_t = next(bt)
        if b_p.num_rows != b_t.num_rows:
            raise SystemExit("First batch row count mismatch")
        st = b_p.column(0)
        print(f"[dry-run] chunk rows={b_t.num_rows}")
        print(f"  source_stem type={st.type}")
        print(f"  output columns: features_100d, sentiment_value, source_stem")
        return

    out_path = (
        args.output.resolve()
        if args.output
        else trans_path.with_suffix(trans_path.suffix + ".with_stem.tmp")
    )
    writer: pq.ParquetWriter | None = None
    total = 0
    try:
        for b_p, b_t in zip(bp, bt):
            if b_p.num_rows != b_t.num_rows:
                raise RuntimeError(
                    f"Batch mismatch: {b_p.num_rows} vs {b_t.num_rows}"
                )
            stem = b_p.column("source_stem")
            feats = b_t.column("features_100d")
            sent = b_t.column("sentiment_value")
            new_t = pa.Table.from_arrays(
                [feats, sent, stem],
                names=["features_100d", "sentiment_value", "source_stem"],
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

    backup = trans_path.with_suffix(trans_path.suffix + ".before_source_stem.bak")
    if not backup.is_file():
        shutil.copy2(trans_path, backup)
        print(f"Backup: {backup}")
    os.replace(out_path, trans_path)
    print(f"Replaced {trans_path} ({total} rows)")


if __name__ == "__main__":
    main()

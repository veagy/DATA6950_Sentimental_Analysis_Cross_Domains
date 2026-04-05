#!/usr/bin/env python3
"""
Rewrite ``data/transformed/all-data.parquet`` ``sentiment_value`` for star-rating sources only.

``transformed/all-data.parquet`` from ``embed_reduce.py --only all-data`` has no
``source_stem``; row order matches ``data/processed/all-data.parquet`` (same row count).
For each row whose ``source_stem`` is in ``STAR_RATING_SOURCE_STEMS``, set
``sentiment_value`` using the same rules as ``normalize_label_for_n_classes(..., n_classes=3)``
in ``common.datasets`` (and ``rewrite_all_data_sentiment_three_class.py`` on processed data).
All other rows keep their current ``sentiment_value`` from the transformed file.

Does not modify ``processed/all-data.parquet`` (safe while training reads processed).

Run from TEMP (repo root containing ``Code/``)::

    python Code/thesis/data/rewrite_transformed_star_labels_three_class.py
    python Code/thesis/data/rewrite_transformed_star_labels_three_class.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_REPO = Path(__file__).resolve().parents[3]
_THESIS = _REPO / "Code" / "thesis"
if str(_THESIS) not in sys.path:
    sys.path.insert(0, str(_THESIS))

from common.datasets import (  # noqa: E402
    STAR_RATING_SOURCE_STEMS,
    coerce_label_int,
    normalize_label_for_n_classes,
)


def _stem_key(v) -> str | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip()
    return s if s else None


def _map_star_batch(
    y_proc: np.ndarray,
    stem_obj: np.ndarray,
    y_trans: np.ndarray,
) -> np.ndarray:
    """Return new sentiment_value column; non-star rows = y_trans."""
    out = np.asarray(y_trans, dtype=np.int64).copy()
    n = len(out)
    for i in range(n):
        sk = _stem_key(stem_obj[i])
        if sk not in STAR_RATING_SOURCE_STEMS:
            continue
        yi = coerce_label_int(y_proc[i])
        r = normalize_label_for_n_classes(yi, 3, source_stem=sk)
        out[i] = 0 if r is None else int(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--processed",
        type=Path,
        default=_REPO / "data" / "processed" / "all-data.parquet",
        help="Reference parquet with source_stem + sentiment_value (same row order as transformed).",
    )
    ap.add_argument(
        "--transformed",
        type=Path,
        default=_REPO / "data" / "transformed" / "all-data.parquet",
        help="Transformed parquet (features_100d + sentiment_value).",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: overwrite transformed via atomic replace).",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=256_000,
        help="Rows per read chunk (must match between the two readers' zip).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Process first chunk only; print stats, no write.",
    )
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

    cols_p = set(pf_p.schema_arrow.names)
    cols_t = set(pf_t.schema_arrow.names)
    if "source_stem" not in cols_p or "sentiment_value" not in cols_p:
        raise SystemExit(f"processed missing columns; have {sorted(cols_p)}")
    if "sentiment_value" not in cols_t or "features_100d" not in cols_t:
        raise SystemExit(f"transformed missing columns; have {sorted(cols_t)}")

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
            raise SystemExit("First batch row count mismatch between files")
        y_p = b_p.column("sentiment_value").to_numpy(zero_copy_only=False)
        st = b_p.column("source_stem").to_pandas().to_numpy(dtype=object, copy=False)
        y_tr = b_t.column("sentiment_value").to_numpy(zero_copy_only=False)
        mapped = _map_star_batch(y_p, st, y_tr)
        star_mask = np.array(
            [_stem_key(st[i]) in STAR_RATING_SOURCE_STEMS for i in range(len(st))],
            dtype=bool,
        )
        print(f"[dry-run] chunk rows={len(mapped)} star_rows={int(star_mask.sum())}")
        print(f"  STAR_RATING_SOURCE_STEMS={sorted(STAR_RATING_SOURCE_STEMS)}")
        print(f"  sentiment before (trans) unique sample: {np.unique(y_tr)[:20]}")
        print(f"  sentiment after (mapped) unique: {np.unique(mapped)}")
        changed = star_mask & (mapped != y_tr)
        print(f"  rows changed among star stems: {int(changed.sum())}")
        return

    out_path = (
        args.output.resolve()
        if args.output
        else trans_path.with_suffix(trans_path.suffix + ".star_three.tmp")
    )
    writer: pq.ParquetWriter | None = None
    total = 0
    try:
        for b_p, b_t in zip(bp, bt):
            if b_p.num_rows != b_t.num_rows:
                raise RuntimeError(
                    f"Batch size mismatch mid-file: {b_p.num_rows} vs {b_t.num_rows}"
                )
            y_p = b_p.column("sentiment_value").to_numpy(zero_copy_only=False)
            st = b_p.column("source_stem").to_pandas().to_numpy(dtype=object, copy=False)
            y_tr = b_t.column("sentiment_value").to_numpy(zero_copy_only=False)
            feats = b_t.column("features_100d")
            mapped = _map_star_batch(y_p, st, y_tr)
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

    backup = trans_path.with_suffix(trans_path.suffix + ".before_star_three_class.bak")
    if not backup.exists():
        import shutil

        shutil.copy2(trans_path, backup)
        print(f"Backup: {backup}")
    os.replace(out_path, trans_path)
    print(f"Replaced {trans_path} ({total} rows)")


if __name__ == "__main__":
    main()

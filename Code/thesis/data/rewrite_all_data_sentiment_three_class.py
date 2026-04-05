#!/usr/bin/env python3
"""
Rewrite ``data/processed/all-data.parquet`` so ``sentiment_value`` is only 0, 1, or 2.

Uses the same rules as ``normalize_label_for_n_classes(..., n_classes=3, source_stem=...)``
in ``Code.thesis.common.datasets`` (star-rating stems + generic merge bucketing).

Run from repo root (directory that contains ``Code/``), e.g.::

    python Code/thesis/data/rewrite_all_data_sentiment_three_class.py
    python Code/thesis/data/rewrite_all_data_sentiment_three_class.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Code.thesis.common.datasets import normalize_label_for_n_classes  # noqa: E402


def _stem_str(v) -> str | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip()
    return s if s else None


def _map_chunk(y_arr: np.ndarray, stem_arr: np.ndarray) -> np.ndarray:
    out = np.empty(len(y_arr), dtype=np.int64)
    for i in range(len(y_arr)):
        yi = int(y_arr[i])
        st = _stem_str(stem_arr[i])
        r = normalize_label_for_n_classes(yi, 3, source_stem=st)
        out[i] = 0 if r is None else int(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=_REPO_ROOT / "data" / "processed" / "all-data.parquet",
        help="Input merged parquet (default: data/processed/all-data.parquet under repo root).",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: overwrite input via atomic replace).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan first row group only; print label stats, do not write.",
    )
    args = ap.parse_args()
    inp = args.input.resolve()
    if not inp.is_file():
        raise SystemExit(f"Missing input: {inp}")

    reader = pq.ParquetFile(inp)
    names = reader.schema_arrow.names
    if "sentiment_value" not in names or "source_stem" not in names:
        raise SystemExit(f"Expected columns sentiment_value, source_stem; got {names}")

    if args.dry_run:
        t = reader.read_row_group(0)
        y = t.column("sentiment_value").to_numpy(zero_copy_only=False)
        st = t.column("source_stem").to_pandas().to_numpy(dtype=object, copy=False)
        m = _map_chunk(y, st)
        print(f"[dry-run] row_group 0 rows={len(m)}")
        print(f"  raw sentiment_value unique: {sorted(set(y.tolist()))[:30]}...")
        print(f"  mapped unique: {sorted(set(m.tolist()))}")
        return

    out_path = args.output.resolve() if args.output else inp.with_suffix(inp.suffix + ".three_class.tmp")
    writer: pq.ParquetWriter | None = None
    total = 0
    try:
        for rg in range(reader.num_row_groups):
            t = reader.read_row_group(rg)
            y = t.column("sentiment_value").to_numpy(zero_copy_only=False)
            st = t.column("source_stem").to_pandas().to_numpy(dtype=object, copy=False)
            mapped = _map_chunk(y, st)
            new_col = pa.array(mapped, type=pa.int64())
            arrays = []
            for name in t.column_names:
                if name == "sentiment_value":
                    arrays.append(new_col)
                else:
                    arrays.append(t.column(name))
            new_t = pa.Table.from_arrays(arrays, names=t.column_names)
            if writer is None:
                writer = pq.ParquetWriter(out_path, new_t.schema, compression="snappy")
            writer.write_table(new_t)
            total += new_t.num_rows
            print(f"  row_group {rg + 1}/{reader.num_row_groups} rows={new_t.num_rows} (total {total})")
    finally:
        if writer is not None:
            writer.close()

    if args.output:
        print(f"Wrote {out_path} ({total} rows)")
        return

    backup = inp.with_suffix(inp.suffix + ".before_three_class.bak")
    if not backup.exists():
        import shutil

        shutil.copy2(inp, backup)
        print(f"Backup: {backup}")
    os.replace(out_path, inp)
    print(f"Replaced {inp} ({total} rows)")


if __name__ == "__main__":
    main()

"""
Remove rows with empty `text` from processed parquets in-place (tmp + replace).

Skips ``all-data.parquet`` by default so a merged corpus is not rewritten unless you
opt in with ``--include-all-data``.
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROCESSED = _REPO / "data" / "processed"
ALL_DATA_NAME = "all-data.parquet"


def filter_file(filepath: Path) -> None:
    print(f"Filtering {filepath.name}...")
    try:
        parquet_file = pq.ParquetFile(filepath)
        writer: pq.ParquetWriter | None = None
        tmp_path = filepath.with_suffix(".tmp.parquet")
        for i, batch in enumerate(parquet_file.iter_batches(batch_size=250000)):
            df = batch.to_pandas()
            df = df.dropna(subset=["text"])
            df = df[df["text"].str.strip() != ""]

            table = pa.Table.from_pandas(df)
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, table.schema)
            writer.write_table(table)

        if writer:
            writer.close()

        del parquet_file
        gc.collect()
        time.sleep(0.5)

        if tmp_path.exists():
            filepath.unlink()
            tmp_path.rename(filepath)
            print(f"  [OK] Cleaned {filepath.name}")
        else:
            print(f"  [WARN] No tmp file saved for {filepath.name}")
    except Exception as e:
        print(f"  [ERROR] Failed to process {filepath.name}: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Drop empty-text rows from processed parquets.")
    ap.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED,
        help=f"Directory of *.parquet (default: {DEFAULT_PROCESSED})",
    )
    ap.add_argument(
        "--include-all-data",
        action="store_true",
        help=f"Also rewrite {ALL_DATA_NAME} (default: skip merged file)",
    )
    args = ap.parse_args()
    proc = args.processed_dir.resolve()
    if not proc.is_dir():
        print(f"[ERROR] Not a directory: {proc}", file=sys.stderr)
        sys.exit(1)

    print("Filtering processed parquet files (empty text rows)...")
    for filepath in sorted(proc.glob("*.parquet")):
        if filepath.name == ALL_DATA_NAME and not args.include_all_data:
            print(f"[SKIP] {filepath.name} (use --include-all-data to process merged corpus)")
            continue
        filter_file(filepath)


if __name__ == "__main__":
    main()

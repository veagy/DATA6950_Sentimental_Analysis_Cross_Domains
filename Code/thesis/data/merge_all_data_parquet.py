"""
Build ``data/processed/all-data.parquet`` by concatenating every ``*.parquet`` in
``data/processed`` except ``all-data.parquet``, with a unified schema:

- ``text`` (from ``text``, ``cleaned_text``, or inferred)
- ``sentiment_value`` (int labels via ``coerce_label_int``)
- ``source_stem`` (optional, for traceability)

Writes row-by-row per source file so peak memory is roughly one source parquet at a time.
Run from repo root::

    python Code/thesis/data/merge_all_data_parquet.py
    python Code/thesis/data/merge_all_data_parquet.py --delete-sources-after --confirm-delete-sources

Use ``--delete-sources-after`` only after verifying the merge, and always pass
``--confirm-delete-sources`` together (safety guard). Sources can be recreated by
splitting ``all-data.parquet`` on ``source_stem``.

Then regenerate ``data/transformed/all-data.parquet`` with ``embed_reduce.py --only all-data --force``
(or merge existing transformed shards with ``merge_all_transformed_parquet.py``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_REPO = Path(__file__).resolve().parents[3]
_THESIS = _REPO / "Code" / "thesis"
if str(_THESIS) not in sys.path:
    sys.path.insert(0, str(_THESIS))

from common.datasets import (  # noqa: E402
    _infer_label_column,
    _infer_text_column,
    coerce_label_int,
)

ALL_DATA_STEM = "all-data"


def _resolve_text_column(columns: list[str]) -> str | None:
    t = _infer_text_column(columns)
    if t:
        return t
    cols_lower = {c.lower(): c for c in columns}
    if "cleaned_text" in cols_lower:
        return cols_lower["cleaned_text"]
    return None


def normalize_processed_file(path: Path, stem: str, include_source_stem: bool) -> pa.Table:
    table = pq.read_table(path)
    df = table.to_pandas().reset_index(drop=True)
    text_col = _resolve_text_column(list(df.columns))
    if not text_col:
        raise ValueError(f"{path.name}: could not infer text column (columns={list(df.columns)})")
    label_col = _infer_label_column(list(df.columns))

    texts = df[text_col].astype(str)
    if label_col:
        labels = df[label_col].map(coerce_label_int).astype("int64")
    else:
        labels = pd.Series(0, index=df.index, dtype="int64")

    out: dict = {"text": texts, "sentiment_value": labels}
    if include_source_stem:
        out["source_stem"] = pd.Series([stem] * len(df), dtype="string")

    out_df = pd.DataFrame(out)
    return pa.Table.from_pandas(out_df, preserve_index=False)


def merge_all_data(
    processed_dir: Path,
    output_path: Path,
    *,
    include_source_stem: bool = True,
    delete_sources_after: bool = False,
) -> int:
    sources = sorted(
        p for p in processed_dir.glob("*.parquet") if p.is_file() and p.stem != ALL_DATA_STEM
    )
    if not sources:
        raise SystemExit(f"No source parquets under {processed_dir} (excluding {ALL_DATA_STEM})")

    expected_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in sources)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    writer: pq.ParquetWriter | None = None
    total_written = 0
    try:
        for path in sources:
            stem = path.stem
            batch = normalize_processed_file(path, stem, include_source_stem)
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, batch.schema, compression="snappy")
            writer.write_table(batch)
            total_written += batch.num_rows
        if writer:
            writer.close()
            writer = None
    except Exception:
        if writer:
            try:
                writer.close()
            except Exception:
                pass
        if tmp_path.is_file():
            tmp_path.unlink(missing_ok=True)
        raise

    if total_written != expected_rows:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Row count mismatch: wrote {total_written}, expected sum of parts {expected_rows}"
        )

    tmp_path.replace(output_path)
    if delete_sources_after:
        removed = 0
        for p in sources:
            try:
                p.unlink()
                removed += 1
            except OSError as e:
                raise RuntimeError(f"Merged OK but failed to delete source {p}: {e}") from e
        print(f"Removed {removed} source parquet(s) under {processed_dir}")
    return total_written


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge processed parquets into all-data.parquet")
    ap.add_argument(
        "--processed-dir",
        type=Path,
        default=_REPO / "data" / "processed",
        help="Directory containing per-dataset processed parquets",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output parquet (default: <processed-dir>/{ALL_DATA_STEM}.parquet)",
    )
    ap.add_argument(
        "--no-source-stem",
        action="store_true",
        help="Omit source_stem column",
    )
    ap.add_argument(
        "--delete-sources-after",
        action="store_true",
        help="After successful merge and row-count check, delete each source *.parquet (not all-data)",
    )
    ap.add_argument(
        "--confirm-delete-sources",
        action="store_true",
        help="Required with --delete-sources-after (prevents accidental shard deletion)",
    )
    args = ap.parse_args()
    if args.delete_sources_after and not args.confirm_delete_sources:
        ap.error(
            "--delete-sources-after requires --confirm-delete-sources (see script docstring)"
        )
    processed_dir = args.processed_dir.resolve()
    out = args.output.resolve() if args.output else (processed_dir / f"{ALL_DATA_STEM}.parquet")

    n = merge_all_data(
        processed_dir,
        out,
        include_source_stem=not args.no_source_stem,
        delete_sources_after=args.delete_sources_after,
    )
    print(f"Wrote {out} ({n} rows)")


if __name__ == "__main__":
    main()

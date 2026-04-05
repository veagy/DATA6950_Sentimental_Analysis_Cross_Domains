"""
Merge ``data/transformed/*.parquet`` except ``all-data.parquet`` into
``data/transformed/all-data.parquet``, appending ``source_stem`` on every row so you can
split back with ``groupby(source_stem)``.

Row counts: sum of per-file ``ParquetFile.metadata.num_rows`` must equal rows written; on
mismatch the temp output is removed and ``RuntimeError`` is raised.

**UMAP caveat:** Each stem is usually embedded with its own UMAP fit in ``embed_reduce.py``;
concatenating transformed shards preserves rows and labels but **mixes embedding spaces**.
For a single consistent 100D space across the corpus, merge processed first, then run
``embed_reduce.py --only all-data --force`` instead of (or understanding the difference
from) this merge.

**``sentiment`` column:** If present (exact name), this script logs dtype / string-like
checks only; it never casts or rewrites columns.

Unmerge example (pandas)::

    df = pd.read_parquet("data/transformed/all-data.parquet")
    for stem, g in df.groupby("source_stem"):
        g.drop(columns=["source_stem"]).to_parquet(f"data/transformed/{stem}.parquet")

Run from repo root::

    python Code/thesis/data/merge_all_transformed_parquet.py
    python Code/thesis/data/merge_all_transformed_parquet.py --delete-sources-after --confirm-delete-sources
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_REPO = Path(__file__).resolve().parents[3]

ALL_DATA_STEM = "all-data"
# Written by some pandas.to_parquet paths when the index was not dropped.
_JUNK_PARQUET_COLS = frozenset({"__index_level_0__"})


def _data_column_names(path: Path) -> list[str]:
    names = list(pq.ParquetFile(path).schema_arrow.names)
    return [c for c in names if c not in _JUNK_PARQUET_COLS]


def _prepare_transformed_batch(df: pd.DataFrame, ref_cols: list[str]) -> pd.DataFrame:
    drop_junk = [c for c in df.columns if c in _JUNK_PARQUET_COLS]
    if drop_junk:
        df = df.drop(columns=drop_junk)
    missing = [c for c in ref_cols if c not in df.columns]
    if missing:
        raise ValueError(f"batch missing columns {missing!r} (have {list(df.columns)!r})")
    extra = [c for c in df.columns if c not in ref_cols]
    if extra:
        raise ValueError(f"batch has unexpected columns {extra!r} (expected {ref_cols!r})")
    return df[ref_cols]


def _audit_sentiment_string_like(df: pd.DataFrame, path: Path) -> None:
    if "sentiment" not in df.columns:
        return
    s = df["sentiment"]
    string_like = bool(
        pd.api.types.is_string_dtype(s)
        or pd.api.types.is_object_dtype(s)
        or getattr(s.dtype, "name", "") == "string"
    )
    print(
        f"[sentiment audit] {path.name}: column 'sentiment' dtype={s.dtype!r} "
        f"string_like={string_like} (read-only, not modified)"
    )


def merge_all_transformed(
    transformed_dir: Path,
    output_path: Path,
    *,
    delete_sources_after: bool = False,
) -> int:
    sources = sorted(
        p for p in transformed_dir.glob("*.parquet") if p.is_file() and p.stem != ALL_DATA_STEM
    )
    if not sources:
        raise SystemExit(f"No source parquets under {transformed_dir} (excluding {ALL_DATA_STEM})")

    ref_cols = _data_column_names(sources[0])
    ref_set = set(ref_cols)
    for p in sources[1:]:
        cols = _data_column_names(p)
        if set(cols) != ref_set:
            raise ValueError(
                f"{p.name}: data columns {cols!r} must match first source "
                f"{sources[0].name!r} {ref_cols!r} (ignoring {_JUNK_PARQUET_COLS})"
            )

    any_sentiment = "sentiment" in ref_cols
    if not any_sentiment:
        print(
            "[sentiment audit] No 'sentiment' column in transformed sources "
            "(typical for embed_reduce output; using sentiment_value only)."
        )

    expected_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in sources)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    writer: pq.ParquetWriter | None = None
    total_written = 0
    sentiment_logged: set[str] = set()

    try:
        for path in sources:
            stem = path.stem
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=100_000):
                df = batch.to_pandas().reset_index(drop=True)
                df = _prepare_transformed_batch(df, ref_cols)
                if stem not in sentiment_logged and "sentiment" in df.columns:
                    _audit_sentiment_string_like(df, path)
                    sentiment_logged.add(stem)
                df["source_stem"] = pd.Series([stem] * len(df), dtype="string")
                table = pa.Table.from_pandas(df, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(tmp_path, table.schema, compression="snappy")
                elif not table.schema.equals(writer.schema, check_metadata=False):
                    raise ValueError(
                        f"{path.name}: batch schema {table.schema} != writer schema {writer.schema}"
                    )
                writer.write_table(table)
                total_written += table.num_rows
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
        print(f"Removed {removed} source parquet(s) under {transformed_dir}")
    return total_written


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Merge transformed parquets into all-data.parquet with source_stem"
    )
    ap.add_argument(
        "--transformed-dir",
        type=Path,
        default=_REPO / "data" / "transformed",
        help="Directory containing per-dataset transformed parquets",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output parquet (default: <transformed-dir>/{ALL_DATA_STEM}.parquet)",
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
    transformed_dir = args.transformed_dir.resolve()
    out = args.output.resolve() if args.output else (transformed_dir / f"{ALL_DATA_STEM}.parquet")

    n = merge_all_transformed(
        transformed_dir,
        out,
        delete_sources_after=args.delete_sources_after,
    )
    print(f"Wrote {out} ({n} rows)")


if __name__ == "__main__":
    main()

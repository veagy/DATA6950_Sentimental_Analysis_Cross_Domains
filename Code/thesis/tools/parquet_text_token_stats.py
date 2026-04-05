#!/usr/bin/env python3
"""
Compute max and mean token counts for a parquet text column using the HRM tokenizer
(google-bert/bert-base-uncased by default). Full encode, no truncation.

Run from repository root:
  python Code/thesis/tools/parquet_text_token_stats.py
  python Code/thesis/tools/parquet_text_token_stats.py --parquet path/to/file.parquet --max-rows 10000
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

_REPO = Path(__file__).resolve().parents[3]


def main() -> None:
    ap = argparse.ArgumentParser(description="Max/mean token lengths for a parquet text column (BERT, no truncation).")
    ap.add_argument(
        "--parquet",
        type=Path,
        default=_REPO / "data" / "processed" / "all-data.parquet",
        help="Path to parquet file",
    )
    ap.add_argument("--text-col", type=str, default="text", help="Text column name")
    ap.add_argument(
        "--tokenizer",
        type=str,
        default="google-bert/bert-base-uncased",
        help="Hugging Face tokenizer id (HRM default)",
    )
    ap.add_argument(
        "--arrow-batch-rows",
        type=int,
        default=8192,
        help="Rows per PyArrow read batch",
    )
    ap.add_argument(
        "--encode-batch-size",
        type=int,
        default=512,
        help="Strings per tokenizer batch (lower if OOM on long texts)",
    )
    ap.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Stop after this many rows (smoke test)",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=500_000,
        help="Print partial stats every N rows (0 to disable)",
    )
    args = ap.parse_args()

    parquet_path = args.parquet.resolve()
    if not parquet_path.is_file():
        raise SystemExit(f"Parquet not found: {parquet_path}")

    from transformers import AutoTokenizer
    import pyarrow.parquet as pq

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    pf = pq.ParquetFile(parquet_path)

    if args.text_col not in pf.schema_arrow.names:
        raise SystemExit(
            f"Column {args.text_col!r} not in parquet. Columns: {pf.schema_arrow.names}"
        )

    total_rows_meta = pf.metadata.num_rows
    if args.max_rows is not None:
        total_target = min(args.max_rows, total_rows_meta)
    else:
        total_target = total_rows_meta

    global_max = 0
    sum_tokens = 0
    count = 0

    def flush_chunk(strs: list[str]) -> None:
        nonlocal global_max, sum_tokens, count
        if not strs:
            return
        bs = args.encode_batch_size
        for i in range(0, len(strs), bs):
            chunk = strs[i : i + bs]
            out = tok(
                chunk,
                add_special_tokens=True,
                truncation=False,
                padding=False,
                return_attention_mask=False,
            )
            for ids in out["input_ids"]:
                n = len(ids)
                if n > global_max:
                    global_max = n
                sum_tokens += n
                count += 1

    pending: list[str] = []
    rows_done = 0
    last_progress_at = 0

    for batch in pf.iter_batches(
        batch_size=args.arrow_batch_rows,
        columns=[args.text_col],
    ):
        col = batch.column(0)
        # PyArrow may return large_string; iterate values
        for j in range(batch.num_rows):
            if args.max_rows is not None and rows_done >= args.max_rows:
                break
            v = col[j].as_py()
            if v is None:
                s = ""
            else:
                s = str(v)
            pending.append(s)
            rows_done += 1

            if len(pending) >= args.encode_batch_size:
                flush_chunk(pending)
                pending = []
                if args.progress_every and count >= last_progress_at + args.progress_every:
                    last_progress_at = (count // args.progress_every) * args.progress_every
                    print(
                        f"[progress] rows={count} max_tokens={global_max} mean={sum_tokens / count:.4f}",
                        flush=True,
                    )

        if args.max_rows is not None and rows_done >= args.max_rows:
            break

    flush_chunk(pending)
    if args.progress_every and count and count > last_progress_at:
        print(
            f"[progress] rows={count} max_tokens={global_max} mean={sum_tokens / count:.4f}",
            flush=True,
        )

    mean = sum_tokens / count if count else float("nan")
    print(f"parquet: {parquet_path}")
    print(f"tokenizer: {args.tokenizer}")
    print(f"rows_scanned: {count} (file_metadata_rows={total_rows_meta})")
    print(f"max_tokens: {global_max}")
    print(f"mean_tokens: {mean:.6f}")


if __name__ == "__main__":
    main()

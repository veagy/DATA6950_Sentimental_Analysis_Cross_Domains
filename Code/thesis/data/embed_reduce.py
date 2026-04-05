import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Please install sentence_transformers: pip install -U sentence-transformers")
    sys.exit(1)

try:
    import umap
except ImportError:
    print("Please install umap-learn: pip install umap-learn")
    sys.exit(1)

_REPO = Path(__file__).resolve().parents[3]


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _default_paths() -> tuple[Path, Path, Path]:
    processed = _REPO / "data" / "processed"
    transformed = _REPO / "data" / "transformed"
    checkpoint = _REPO / "checkpoints" / "transformer" / "all-MiniLM-L6-v2"
    return processed, transformed, checkpoint


def _text_series(df: pd.DataFrame) -> pd.Series:
    cols = list(df.columns)
    cols_lower = {c.lower(): c for c in cols}
    if "text" in cols_lower:
        return df[cols_lower["text"]].astype(str)
    if "cleaned_text" in cols_lower:
        return df[cols_lower["cleaned_text"]].astype(str)
    raise ValueError(
        f"No text column (expected 'text' or 'cleaned_text'); columns={cols}"
    )


def perform_embedding(
    filepath: Path,
    model: SentenceTransformer,
    *,
    transformed_dir: Path,
    force: bool = False,
) -> None:
    filepath = filepath.resolve()
    print(f"\n=========================================")
    print(f" Processing {filepath.name}...")
    print(f"=========================================")
    output_path = transformed_dir / filepath.name

    if output_path.exists() and not force:
        print(f"Skipping {filepath.name}, already transformed (use --force to overwrite).")
        return

    if output_path.exists() and force:
        output_path.unlink()

    dataset_parquet = pq.ParquetFile(filepath)
    total_rows = dataset_parquet.metadata.num_rows

    print(f"[{filepath.name}] Total rows: {total_rows}")
    fit_samples = min(total_rows, 100000)

    sample_texts: list[str] = []
    frac = fit_samples / total_rows if total_rows > 0 else 1.0

    print(f"[{filepath.name}] Collecting {fit_samples} samples for UMAP topological fitting...")
    for batch in dataset_parquet.iter_batches(batch_size=100000):
        df_batch = batch.to_pandas().reset_index(drop=True)
        sample_size = int(len(df_batch) * frac) + 1
        if sample_size > 0:
            ser = _text_series(df_batch)
            sample_texts.extend(
                ser.sample(n=min(sample_size, len(df_batch)), random_state=42).tolist()
            )
        if len(sample_texts) >= fit_samples:
            break

    sample_texts = sample_texts[:fit_samples]

    if not sample_texts:
        print(f"[{filepath.name}] No valid texts found.")
        return

    print(f"[{filepath.name}] Encoding sample corpus (size: {len(sample_texts)})...")
    sample_embeddings = model.encode(
        sample_texts, batch_size=128, show_progress_bar=True, convert_to_numpy=True
    )

    print(f"[{filepath.name}] Fitting UMAP (384 -> 100)...")
    reducer = umap.UMAP(n_components=100, random_state=42, verbose=True)
    reducer.fit(sample_embeddings)

    del sample_embeddings
    del sample_texts
    gc.collect()

    print(f"[{filepath.name}] Streaming batch transformations...")
    writer: pq.ParquetWriter | None = None

    num_chunks = (total_rows // 100000) + 1

    for batch in tqdm(
        dataset_parquet.iter_batches(batch_size=100000),
        total=num_chunks,
        desc=f"Transforming {filepath.name}",
    ):
        df_batch = batch.to_pandas().reset_index(drop=True)
        texts = _text_series(df_batch).tolist()
        if not texts:
            continue

        embeddings_384d = model.encode(
            texts, batch_size=128, show_progress_bar=False, convert_to_numpy=True
        )
        embeddings_100d = reducer.transform(embeddings_384d)

        df_batch["features_100d"] = list(embeddings_100d.astype(np.float32))
        if "sentiment_value" not in df_batch.columns:
            raise ValueError(
                f"{filepath.name}: batch missing 'sentiment_value' (columns={list(df_batch.columns)})"
            )
        df_batch = df_batch[["features_100d", "sentiment_value"]]

        table = pa.Table.from_pandas(df_batch, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)
        writer.write_table(table)

    if writer:
        writer.close()
    print(f"[{filepath.name}] Transformation Complete!")


def main() -> None:
    processed_dir, transformed_dir, checkpoint_dir = _default_paths()
    transformed_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ap = argparse.ArgumentParser(
        description="Embed processed parquets with MiniLM + UMAP -> data/transformed/*.parquet"
    )
    ap.add_argument(
        "--processed-dir",
        type=Path,
        default=processed_dir,
        help="Input directory (default: data/processed under repo root)",
    )
    ap.add_argument(
        "--transformed-dir",
        type=Path,
        default=transformed_dir,
        help="Output directory (default: data/transformed under repo root)",
    )
    ap.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=checkpoint_dir,
        help="Local SentenceTransformer cache directory",
    )
    ap.add_argument(
        "--only",
        type=str,
        default=None,
        metavar="STEM",
        help="Process only this parquet stem (e.g. all-data for merged corpus)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs in transformed/",
    )
    args = ap.parse_args()

    proc = args.processed_dir.resolve()
    trans = args.transformed_dir.resolve()
    ckpt = args.checkpoint_dir.resolve()
    trans.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)

    dev = _device()
    print("Loading or downloading sentence transformer...")
    if not (ckpt / "config.json").exists():
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev)
        model.save(str(ckpt))
    else:
        model = SentenceTransformer(str(ckpt), device=dev)
    print(f"Model ready on {model.device}.")

    files = sorted(proc.glob("*.parquet"))
    if args.only:
        want = f"{args.only}.parquet"
        files = [p for p in files if p.name == want]
        if not files:
            raise SystemExit(f"No parquet matching stem {args.only!r} under {proc}")

    for p in files:
        perform_embedding(p, model, transformed_dir=trans, force=args.force)


if __name__ == "__main__":
    main()

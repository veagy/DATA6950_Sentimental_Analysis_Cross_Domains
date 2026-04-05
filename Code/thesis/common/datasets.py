from __future__ import annotations

import os
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pq = None  # type: ignore[misc, assignment]

# Review / star-style sources in merged ``all-data.parquet`` (``source_stem`` column).
# Raw ``sentiment_value`` may be 1–5 stars or 0–10 scores; map to 3-way 0/1/2 for finetune.
STAR_RATING_SOURCE_STEMS = frozenset(
    {
        "yelp_review",
        "yelp_business",
        "amazon_reviews",
    }
)


def _infer_text_column(columns: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in columns}
    for key in ("text", "review", "statement", "tweet", "sentence", "body"):
        if key in cols_lower:
            return cols_lower[key]
    for c in columns:
        cl = c.lower()
        if "text" in cl or "review" in cl or "statement" in cl:
            return c
    return None


def _infer_label_column(columns: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in columns}
    for key in ("label", "sentiment", "target", "sentiment_value", "class"):
        if key in cols_lower:
            return cols_lower[key]
    for c in columns:
        if "label" in c.lower() or "sentiment" in c.lower():
            return c
    return None


def coerce_label_int(v) -> int:
    """
    Map a parquet label cell to int for classification (2- or 3-way sentiment and numeric).
    Handles string labels like 'negative' / 'positive' / 'neutral' that break raw int().
    """
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return 0
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, float) and not pd.isna(v):
        if v == int(v):
            return int(v)
        return 0
    s = str(v).strip().lower()
    if s in ("positive", "pos", "1", "yes"):
        return 1
    if s in ("negative", "neg", "0", "no"):
        return 0
    if s in ("neutral", "neu", "2"):
        return 2
    try:
        return int(float(s))
    except ValueError:
        return 0


def remap_star_rating_label_if_applicable(stem: Optional[str], y: int) -> Optional[int]:
    """
    For known review/star stems, map rating-like integers to 3-way sentiment.

    - ``y <= 2`` is treated as already neg/neu/pos (0/1/2) and left unchanged.
    - ``amazon_reviews`` with ``y in {0, 1}`` is kept (binary sentiment).
    - ``3–5``: 5-star scale → 1–2 stars → 0, 3 → 1, 4–5 → 2.
    - ``6–10``: 0–10 style thirds → 6 → neutral (1), 7–10 → positive (2).

    Returns ``None`` if ``stem`` is not a star-rating source or if ``y > 10`` (caller
    buckets with the generic merge rule).
    """
    if not stem or stem not in STAR_RATING_SOURCE_STEMS:
        return None
    if stem == "amazon_reviews" and y in (0, 1):
        return y
    if y <= 2:
        return y
    if 3 <= y <= 5:
        if y == 3:
            return 1
        return 2
    if 6 <= y <= 10:
        if y >= 7:
            return 2
        return 1
    return None


def normalize_label_for_n_classes(
    y: int, n_classes: int, source_stem: Optional[str] = None
) -> Optional[int]:
    """
    Map ``coerce_label_int`` output into ``0..n_classes-1`` for CrossEntropyLoss.

    Merged ``all-data.parquet`` can mix canonical 3-way labels (0/1/2) with coarse
    numeric strings (e.g. "10" → 10) from per-source normalization; values above 2
    are bucketed into three sentiment classes so training does not hit CUDA NLL
    assertions (targets must be ``< n_classes``).

    For rows with ``source_stem`` in ``STAR_RATING_SOURCE_STEMS``, integers in the
    3–10 range are interpreted as star / 10-point ratings and mapped to 0/1/2
    before generic bucketing.
    """
    if n_classes == 3:
        if y < 0:
            return None
        sr = remap_star_rating_label_if_applicable(source_stem, y)
        if sr is not None:
            return sr
        if y <= 2:
            return y
        # Bucket coarse scores (typical merged range 3–19) into neg / neu / pos.
        if y < 7:
            return 0
        if y < 14:
            return 1
        return 2
    if n_classes == 2:
        if y in (0, 1):
            return y
        if y == 2:
            return None
        t3 = normalize_label_for_n_classes(y, 3, source_stem=source_stem)
        if t3 is None or t3 == 2:
            return None
        return t3
    if 0 <= y < n_classes:
        return y
    return None


def _infer_features_column(columns: List[str]) -> Optional[str]:
    for c in columns:
        cl = c.lower()
        if "features_100d" in cl or cl == "features_100d":
            return c
    for c in columns:
        cl = c.lower()
        if "features" in cl or "embedding" in cl or "vector" in cl:
            return c
    return None


class ParquetTextDataset(Dataset):
    """Processed parquet: raw text + integer labels."""

    def __init__(
        self,
        parquet_path: str,
        max_samples: Optional[int] = None,
        text_col: Optional[str] = None,
        label_col: Optional[str] = None,
        *,
        allow_missing_label: bool = False,
        dummy_label: int = 0,
        exclude_neutral: bool = False,
        n_classes: Optional[int] = None,
    ):
        self.samples: List[Tuple[str, int]] = []
        if not os.path.isfile(parquet_path):
            return
        df = pd.read_parquet(parquet_path)
        if max_samples is not None:
            df = df.head(max_samples)
        cols = list(df.columns)
        tc = text_col or _infer_text_column(cols)
        lc = label_col or _infer_label_column(cols)
        if not tc:
            return
        if not lc and not allow_missing_label:
            return
        stem_col = "source_stem" if "source_stem" in cols else None
        for _, row in df.iterrows():
            t = row[tc]
            if not isinstance(t, str):
                t = str(t) if t is not None else ""
            if lc and lc in row.index:
                y = coerce_label_int(row[lc])
            else:
                y = dummy_label
            stem: Optional[str] = None
            if stem_col and stem_col in row.index:
                sv = row[stem_col]
                if sv is not None and not (isinstance(sv, float) and pd.isna(sv)):
                    stem = str(sv).strip()
            if n_classes is not None:
                yn = normalize_label_for_n_classes(int(y), int(n_classes), source_stem=stem)
                if yn is None:
                    continue
                y = yn
            if exclude_neutral and y == 2:
                continue
            self.samples.append((t, y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.samples[idx]


class ParquetFeaturesDataset(Dataset):
    """Transformed parquet: 100-D features + integer labels.

    If ``n_classes`` is 2 or 3, labels are normalized like ``ParquetTextDataset`` (merged
    ``all-data`` / ``source_stem`` / star-rating rules). Rows that map to ``None`` are
    skipped. When ``n_classes`` is omitted, raw ``coerce_label_int`` labels are kept (e.g.
    MoE dual datasets aligned by row index with processed text).
    """

    def __init__(
        self,
        parquet_path: str,
        max_samples: Optional[int] = None,
        features_col: Optional[str] = None,
        label_col: Optional[str] = None,
        *,
        n_classes: Optional[int] = None,
    ):
        self.samples: List[Tuple[torch.Tensor, int]] = []
        if not os.path.isfile(parquet_path):
            return
        df = pd.read_parquet(parquet_path)
        if max_samples is not None:
            df = df.head(max_samples)
        cols = list(df.columns)
        fc = features_col or _infer_features_column(cols)
        lc = label_col or _infer_label_column(cols)
        if not fc or not lc:
            return
        stem_col = "source_stem" if "source_stem" in cols else None
        for _, row in df.iterrows():
            vec = row[fc]
            if isinstance(vec, (list, np.ndarray)):
                x = np.asarray(vec, dtype=np.float32).reshape(-1)
            elif hasattr(vec, "tolist"):
                x = np.asarray(vec.tolist(), dtype=np.float32).reshape(-1)
            else:
                continue
            y = coerce_label_int(row[lc]) if lc in row.index else 0
            stem: Optional[str] = None
            if stem_col and stem_col in row.index:
                sv = row[stem_col]
                if sv is not None and not (isinstance(sv, float) and pd.isna(sv)):
                    stem = str(sv).strip()
            if n_classes is not None:
                yn = normalize_label_for_n_classes(int(y), int(n_classes), source_stem=stem)
                if yn is None:
                    continue
                y = yn
            self.samples.append((torch.from_numpy(x.copy()), y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.samples[idx]


def _vec_to_tensor(vec, target_dim: int) -> Optional[torch.Tensor]:
    if isinstance(vec, (list, np.ndarray)):
        x = np.asarray(vec, dtype=np.float32).reshape(-1)
    elif hasattr(vec, "tolist"):
        x = np.asarray(vec.tolist(), dtype=np.float32).reshape(-1)
    else:
        return None
    if x.size == 0:
        return None
    if x.shape[0] >= target_dim:
        x = x[:target_dim].copy()
    else:
        pad = np.zeros(target_dim, dtype=np.float32)
        pad[: x.shape[0]] = x
        x = pad
    return torch.from_numpy(x)


class StreamingParquetFeaturesIterable(IterableDataset):
    """
    Stream feature rows from a large parquet without loading the full file.
    Each DDP rank consumes a disjoint row range: ``[rank * N // W, (rank+1) * N // W)``.
    """

    def __init__(
        self,
        parquet_path: str,
        rank: int,
        world_size: int,
        *,
        max_samples: Optional[int] = None,
        features_col: Optional[str] = None,
        label_col: Optional[str] = None,
        input_dim: int = 100,
        batch_read: int = 4096,
        n_classes: Optional[int] = None,
    ) -> None:
        super().__init__()
        if pq is None:
            raise RuntimeError("StreamingParquetFeaturesIterable requires pyarrow")
        if not os.path.isfile(parquet_path):
            raise FileNotFoundError(parquet_path)
        self.parquet_path = parquet_path
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.max_samples = max_samples
        self.features_col = features_col
        self.label_col = label_col
        self.input_dim = int(input_dim)
        self.batch_read = int(batch_read)
        self.n_classes = int(n_classes) if n_classes is not None else None
        self.stem_col: Optional[str] = None

        pf = pq.ParquetFile(parquet_path)
        self._num_rows = int(pf.metadata.num_rows)
        # PyArrow Schema: use .names (num_fields / field(i) API differs across versions).
        schema_names = list(pf.schema_arrow.names)
        self.fc = features_col or _infer_features_column(schema_names)
        self.lc = label_col or _infer_label_column(schema_names)
        if not self.fc:
            raise ValueError(f"No features column in {parquet_path}")
        if "source_stem" in schema_names:
            self.stem_col = "source_stem"

        start = (self._num_rows * self.rank) // self.world_size
        end = (self._num_rows * (self.rank + 1)) // self.world_size
        if max_samples is not None:
            end = min(end, start + int(max_samples))
        self._start = start
        self._end = end

    def __len__(self) -> int:
        return max(0, self._end - self._start)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, int]]:
        if pq is None:
            raise RuntimeError("pyarrow is required")
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            wid, nw = 0, 1
        else:
            wid, nw = worker_info.id, worker_info.num_workers
        span = self._end - self._start
        if span <= 0:
            return
        chunk = (span + nw - 1) // nw
        w_lo = self._start + wid * chunk
        w_hi = min(self._start + (wid + 1) * chunk, self._end)

        pf = pq.ParquetFile(self.parquet_path)
        cols = [self.fc]
        if self.lc:
            cols.append(self.lc)
        if self.n_classes is not None and self.stem_col:
            cols.append(self.stem_col)

        row_cursor = 0
        yielded = 0
        max_y = w_hi - w_lo
        last_pair: Optional[Tuple[torch.Tensor, int]] = None

        # Rows may be skipped (bad vec / filtered label). Without padding, some ranks finish
        # fewer batches than others; DDP then deadlocks on live.save/dist.barrier near epoch end.
        for batch in pf.iter_batches(batch_size=self.batch_read, columns=cols):
            names = batch.schema.names
            i_f = names.index(self.fc)
            col_f = batch.column(i_f)
            col_y = None
            if self.lc and self.lc in names:
                col_y = batch.column(names.index(self.lc))
            col_stem = None
            if self.stem_col and self.stem_col in names:
                col_stem = batch.column(names.index(self.stem_col))
            bs = batch.num_rows
            batch_start = row_cursor
            row_cursor += bs
            stop_shard = False
            for i in range(bs):
                global_i = batch_start + i
                if global_i < w_lo:
                    continue
                if global_i >= w_hi:
                    stop_shard = True
                    break
                vec = col_f[i].as_py()
                t = _vec_to_tensor(vec, self.input_dim)
                if t is None:
                    continue
                y_raw = coerce_label_int(col_y[i].as_py()) if col_y is not None else 0
                source_stem: Optional[str] = None
                if col_stem is not None:
                    sv = col_stem[i].as_py()
                    if sv is not None and not (isinstance(sv, float) and pd.isna(sv)):
                        source_stem = str(sv).strip()
                if self.n_classes is not None:
                    yn = normalize_label_for_n_classes(
                        int(y_raw), int(self.n_classes), source_stem=source_stem
                    )
                    if yn is None:
                        continue
                    y = yn
                else:
                    y = int(y_raw)
                yield t, y
                last_pair = (t, y)
                yielded += 1
            if stop_shard:
                break

        while yielded < max_y:
            if last_pair is None:
                yield torch.zeros(self.input_dim, dtype=torch.float32), 0
            else:
                yield last_pair[0].clone(), last_pair[1]
            yielded += 1


def text_collate(batch: List[Tuple[str, int]]) -> Tuple[List[str], torch.Tensor]:
    texts = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return texts, labels


def features_collate(
    batch: List[Tuple[torch.Tensor, int]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return xs, ys

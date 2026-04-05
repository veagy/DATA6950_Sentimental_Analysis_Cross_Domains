"""
Merge all processed text parquets for HRM (or transformer) unsupervised pretraining.

For a **single** merged file use ``--pretrain_text_source all_data_parquet`` in ``train_single.py``
(``data/processed/all-data.parquet``). This module supports **all** ``data/processed/*.parquet`` via
``all_processed``. The MLM loss does not use labels; a dummy ``0`` is
stored so the same ``text_collate`` contract as ``ParquetTextDataset`` applies.

When ``pretrain_num_classes`` is 2 or 3 on ``LazyShardedMergedParquetTextDataset``,
rows are restricted so the corpus matches **sentiment** semantics:
``negative``/``positive`` (ids 0/1) for 2-label configs, and
``negative``/``neutral``/``positive`` (ids 0/1/2) for 3-label. String labels
``positive``, ``negative``, ``neutral`` (and common aliases) are normalized; ints
in ``{0, 1, 2}`` are kept as-is. Files without an inferred label column keep all
rows (same as unfiltered mode).

For large corpora use ``LazyShardedMergedParquetTextDataset`` (metadata + one-file
text-column cache) to avoid loading every row into Python lists.
"""
from __future__ import annotations

import bisect
import gc
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from torch.utils.data import Dataset

from Code.thesis.common.datasets import _infer_label_column, _infer_text_column, coerce_label_int


def sentiment_id_from_value(v) -> Optional[int]:
    """
    Map a parquet cell to sentiment id: 0 negative, 1 positive, 2 neutral.
    Returns None if missing or not in the project vocabulary.
    """
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, np.integer)):
        i = int(v)
        return i if i in (0, 1, 2) else None
    if isinstance(v, float) and not pd.isna(v):
        if v == int(v):
            i = int(v)
            return i if i in (0, 1, 2) else None
        return None
    s = str(v).strip().lower()
    if s in ("positive", "pos", "1"):
        return 1
    if s in ("negative", "neg", "0"):
        return 0
    if s in ("neutral", "neu", "2"):
        return 2
    try:
        i = int(float(s))
        return i if i in (0, 1, 2) else None
    except ValueError:
        return None


def _coerce_label(row, lc: Optional[str]) -> int:
    if not lc or lc not in row.index:
        return 0
    return coerce_label_int(row[lc])


def _filtered_row_indices_for_file(
    path: Path,
    label_col: str,
    n_raw: int,
    pretrain_num_classes: int,
) -> np.ndarray:
    """Physical row indices in ``0..n_raw-1`` that pass sentiment filter."""
    allowed = {0, 1} if pretrain_num_classes == 2 else {0, 1, 2}
    try:
        col = pd.read_parquet(path, columns=[label_col]).iloc[:n_raw, 0]
    except Exception:
        return np.arange(n_raw, dtype=np.int64)
    keep: List[int] = []
    for i, v in enumerate(col):
        sid = sentiment_id_from_value(v)
        if sid is not None and sid in allowed:
            keep.append(i)
    return np.asarray(keep, dtype=np.int64)


class MergedParquetTextDataset(Dataset):
    """Concatenate rows from every ``*.parquet`` in a processed directory (sorted paths)."""

    def __init__(
        self,
        processed_dir: str | Path,
        max_samples_per_file: Optional[int] = None,
        max_total: Optional[int] = None,
        text_col: Optional[str] = None,
        label_col: Optional[str] = None,
    ):
        self.samples: List[Tuple[str, int]] = []
        proc = Path(processed_dir)
        if not proc.is_dir():
            return
        paths = sorted(proc.glob("*.parquet"))
        for p in paths:
            if not p.is_file():
                continue
            self._append_file(
                p, text_col, label_col, max_samples_per_file, max_total
            )
            if max_total is not None and len(self.samples) >= max_total:
                self.samples = self.samples[:max_total]
                break

    def _append_file(
        self,
        path: Path,
        text_col: Optional[str],
        label_col: Optional[str],
        max_per: Optional[int],
        max_total: Optional[int],
    ) -> None:
        try:
            df = pd.read_parquet(path)
        except Exception:
            return
        if max_per is not None:
            df = df.head(max_per)
        cols = list(df.columns)
        tc = text_col or _infer_text_column(cols)
        lc = label_col or _infer_label_column(cols)
        if not tc:
            return
        for _, row in df.iterrows():
            if max_total is not None and len(self.samples) >= max_total:
                break
            t = row[tc]
            if not isinstance(t, str):
                t = str(t) if t is not None else ""
            if not t.strip():
                continue
            y = _coerce_label(row, lc)
            self.samples.append((t, y))
        del df
        gc.collect()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.samples[idx]


def _parquet_schema_column_names(path: Path) -> Optional[List[str]]:
    try:
        pf = pq.ParquetFile(path)
        sch = pf.schema_arrow
        return list(sch.names)
    except Exception:
        return None


def _parquet_num_rows(path: Path) -> int:
    try:
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        return 0


class LazyShardedMergedParquetTextDataset(Dataset):
    """Merged text with bounded RAM: metadata-only scan; one text column cached per open file.

    Each rank owns a contiguous slice of the global row stream (sorted ``*.parquet``)
    so DDP ranks need not duplicate a giant in-memory list. Tail rows are dropped
    so ``total`` aligns to ``world_size``. Use with ``shuffle=False`` and no
    ``DistributedSampler``.

    If ``pretrain_num_classes`` is 2, only rows labeled negative/positive (ids 0/1)
    are included. If 3, rows with neutral (2) are included as well. Requires a
    label column in each file unless you rely on the fallback (no column => all rows).

    Set ``return_supervised_labels=True`` for HRM/text finetune: ``__getitem__`` returns
    real sentiment ids (same filter as MLM when ``pretrain_num_classes`` is 2/3) instead
    of dummy ``0`` (MLM-only).
    """

    def __init__(
        self,
        processed_dir: str | Path,
        rank: int = 0,
        world_size: int = 1,
        max_samples_per_file: Optional[int] = None,
        max_total: Optional[int] = None,
        text_col: Optional[str] = None,
        pretrain_num_classes: Optional[int] = None,
        *,
        return_supervised_labels: bool = False,
    ):
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.return_supervised_labels = bool(return_supervised_labels)
        self.paths: List[Path] = []
        self._text_cols: List[str] = []
        self._label_cols: List[Optional[str]] = []
        self._row_indices: List[np.ndarray] = []
        self._prefix: List[int] = [0]

        pnc = pretrain_num_classes
        use_sentiment_filter = pnc in (2, 3)

        proc = Path(processed_dir)
        if proc.is_dir():
            for p in sorted(proc.glob("*.parquet")):
                if not p.is_file():
                    continue
                names = _parquet_schema_column_names(p)
                if not names:
                    continue
                tc = text_col or _infer_text_column(names)
                if not tc:
                    continue
                n = _parquet_num_rows(p)
                if max_samples_per_file is not None:
                    n = min(n, max_samples_per_file)
                if n <= 0:
                    continue
                lc = _infer_label_column(names)
                if use_sentiment_filter and lc:
                    idx_arr = _filtered_row_indices_for_file(p, lc, n, int(pnc))
                else:
                    idx_arr = np.arange(n, dtype=np.int64)

                if idx_arr.size == 0:
                    continue

                self.paths.append(p)
                self._text_cols.append(tc)
                self._label_cols.append(lc if (use_sentiment_filter or self.return_supervised_labels) else None)
                self._row_indices.append(idx_arr)
                self._prefix.append(self._prefix[-1] + int(idx_arr.size))

        total = self._prefix[-1]
        if max_total is not None:
            total = min(total, max_total)
            if total < self._prefix[-1]:
                self._truncate_prefix_to_total(total)

        usable = (total // self.world_size) * self.world_size
        self._per_rank = total // self.world_size if self.world_size else 0
        self._global_base = self.rank * self._per_rank
        self._usable_total = usable

        self._cache_path: Optional[Path] = None
        self._cache_text_series = None
        self._cache_label_series = None

    def _truncate_prefix_to_total(self, total: int) -> None:
        """Shrink ``paths`` / ``_row_indices`` / ``_prefix`` so global length is ``total``."""
        new_paths: List[Path] = []
        new_tc: List[str] = []
        new_lc: List[Optional[str]] = []
        new_idx: List[np.ndarray] = []
        prefix: List[int] = [0]
        remaining = total
        for i, p in enumerate(self.paths):
            arr = self._row_indices[i]
            if remaining <= 0:
                break
            take = min(remaining, int(arr.size))
            if take <= 0:
                continue
            sl = arr[:take] if take == arr.size else arr[:take].copy()
            new_paths.append(p)
            new_tc.append(self._text_cols[i])
            new_lc.append(self._label_cols[i])
            new_idx.append(sl)
            prefix.append(prefix[-1] + take)
            remaining -= take
        self.paths = new_paths
        self._text_cols = new_tc
        self._label_cols = new_lc
        self._row_indices = new_idx
        self._prefix = prefix

    def __len__(self) -> int:
        return self._per_rank

    def _clear_cache(self) -> None:
        self._cache_text_series = None
        self._cache_label_series = None
        self._cache_path = None
        gc.collect()

    def _text_and_label_at(
        self, path: Path, row_in_file: int, tc: str, lc: Optional[str]
    ) -> Tuple[str, int]:
        need_label = bool(self.return_supervised_labels and lc)
        if self._cache_path != path:
            self._clear_cache()
            cols = [tc] + ([lc] if need_label else [])
            try:
                df = pd.read_parquet(path, columns=cols)
            except Exception:
                return "", 0
            self._cache_path = path
            self._cache_text_series = df[tc]
            self._cache_label_series = df[lc].values if need_label else None
            del df
            gc.collect()
        assert self._cache_text_series is not None
        t = self._cache_text_series.iloc[row_in_file]
        if not isinstance(t, str):
            t = str(t) if t is not None else ""
        y = 0
        if need_label and self._cache_label_series is not None:
            v = self._cache_label_series[row_in_file]
            sid = sentiment_id_from_value(v)
            y = int(sid) if sid is not None else 0
        return t, y

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        if idx < 0 or idx >= self._per_rank:
            raise IndexError(idx)
        g = self._global_base + idx
        if g >= self._usable_total:
            raise IndexError(idx)
        fi = bisect.bisect_right(self._prefix, g) - 1
        if fi < 0 or fi >= len(self.paths):
            return "", 0
        off = g - self._prefix[fi]
        row_in_file = int(self._row_indices[fi][off])
        path = self.paths[fi]
        tc = self._text_cols[fi]
        lc = self._label_cols[fi]
        return self._text_and_label_at(path, row_in_file, tc, lc)

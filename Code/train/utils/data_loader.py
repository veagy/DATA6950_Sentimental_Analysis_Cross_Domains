"""
Dataset factory: wraps PyTorch DataLoader, routing to the correct storage backend
via data_source.py based on sentinel.conf [data] settings.
"""
from __future__ import annotations

import os
from torch.utils.data import DataLoader


def build_dataloader(split: str, batch_size: int = 32, config: dict | None = None) -> DataLoader:
    """
    Build a DataLoader for the given split (``'train'``, ``'val'``, or ``'test'``).

    Source priority
    ---------------
    1. ``SENTINEL_DATA_SOURCE`` env var (overrides everything)
    2. ``sentinel.conf [data]`` section
    3. Fallback: CSV from ``data/{split}.csv`` relative to project root

    Parameters
    ----------
    split :
        One of ``'train'``, ``'val'``, ``'test'``.
    batch_size :
        Batch size for the loader.
    config :
        Parsed sentinel.conf dict (from ``get_sentinel_config()``).
    """
    from Code.train.utils.data_source import make_loader_from_source

    config = config or {}
    data_cfg = config.get("data", {})

    source = os.environ.get("SENTINEL_DATA_SOURCE") or data_cfg.get("source", "local")
    target_col = data_cfg.get("target_col", "label")

    path_key = f"{split}_path"
    data_path = data_cfg.get(path_key, f"data/{split}.csv")

    return make_loader_from_source(
        source=source,
        path=data_path,
        split=split,
        batch_size=batch_size,
        target_col=target_col,
    )

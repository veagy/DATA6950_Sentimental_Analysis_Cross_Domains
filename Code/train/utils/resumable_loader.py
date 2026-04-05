"""
ResumableDataLoader: Checkpoint-aware DataLoader wrapper for mid-epoch resume support.
Tracks batch index and global step for resumption after interrupt.
"""
from __future__ import annotations

from typing import Any, Iterator, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset


class ResumableDataLoader:
    """
    Wraps a DataLoader with checkpoint/resume support.
    Tracks batch index and global step for mid-epoch resume when training is interrupted.
    """

    def __init__(
        self,
        loader: DataLoader,
        start_batch: int = 0,
        start_global_step: int = 0,
        batch_offset: Optional[int] = None,
    ):
        """
        Args:
            loader: Underlying PyTorch DataLoader.
            start_batch: Batch index to start from (0 = beginning of epoch).
            start_global_step: Global step count at resume (for LR scheduling, etc.).
            batch_offset: Alias for start_batch (deprecated; use start_batch).
        """
        if batch_offset is not None:
            start_batch = batch_offset
        self._loader = loader
        self._start_batch = max(0, start_batch)
        self._global_step = start_global_step
        self._current_batch = 0
        self._current_epoch = 0

    @property
    def dataset(self) -> Dataset:
        return self._loader.dataset

    @property
    def batch_size(self) -> int:
        return self._loader.batch_size

    @property
    def num_workers(self) -> int:
        return self._loader.num_workers

    @property
    def current_batch(self) -> int:
        return self._current_batch

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    def set_epoch(self, epoch: int) -> None:
        self._current_epoch = epoch

    def get_resume_state(self) -> dict[str, Any]:
        """Return state for resume.json: batch index and global step."""
        return {
            "batch": self._current_batch,
            "global_step": self._global_step,
            "epoch": self._current_epoch,
        }

    def __iter__(self) -> Iterator[Any]:
        self._current_batch = 0
        skipped = 0
        target_skip = self._start_batch
        for batch in self._loader:
            if skipped < target_skip:
                skipped += 1
                self._global_step += 1
                continue
            self._current_batch += 1
            self._global_step += 1
            yield batch
        self._start_batch = 0  # Reset after completing an epoch

    def __len__(self) -> int:
        return len(self._loader)


def wrap_resumable(
    loader: Union[DataLoader, Tuple[DataLoader, DataLoader]],
    start_batch: int = 0,
    start_global_step: int = 0,
) -> Union[ResumableDataLoader, Tuple[ResumableDataLoader, ResumableDataLoader]]:
    """
    Wrap a DataLoader or (train_loader, val_loader) pair with ResumableDataLoader.
    Only train loader is wrapped with resume state; val loader is passed through as-is
    (validation doesn't need batch-level resume).
    """
    if isinstance(loader, tuple):
        train_wrapped = ResumableDataLoader(
            loader[0], start_batch=start_batch, start_global_step=start_global_step
        )
        val_wrapped = ResumableDataLoader(loader[1])  # No resume state for val
        return train_wrapped, val_wrapped
    return ResumableDataLoader(
        loader, start_batch=start_batch, start_global_step=start_global_step
    )

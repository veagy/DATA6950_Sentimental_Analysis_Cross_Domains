from __future__ import annotations

import os
import random
from datetime import timedelta
from typing import Any, Callable, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def is_dist_avail() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed_from_env() -> Tuple[bool, Optional[int], Optional[int]]:
    """Returns (use_ddp, local_rank, world_size)."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1 or not torch.cuda.is_available():
        return False, None, None
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    if not is_dist_avail():
        # Default 30m; long MLM epochs can hit 10m PyTorch default if a rank stalls (I/O, GIL).
        _tsec = int(os.environ.get("THESIS_DIST_TIMEOUT_SEC", "1800"))
        _timeout = timedelta(seconds=max(300, _tsec))
        _kw: dict[str, Any] = dict(backend="nccl", timeout=_timeout)
        try:
            _kw["device_id"] = torch.device("cuda", local_rank)
            dist.init_process_group(**_kw)
        except TypeError:
            try:
                dist.init_process_group(backend="nccl", timeout=_timeout, device_id=local_rank)
            except TypeError:
                dist.init_process_group(backend="nccl", timeout=_timeout)
    torch.cuda.set_device(local_rank)
    return True, local_rank, world


def wrap_ddp(
    model: torch.nn.Module,
    local_rank: int,
    *,
    find_unused_parameters: bool = True,
) -> torch.nn.Module:
    return DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=find_unused_parameters,
    )


def make_sampler(dataset, shuffle: bool, use_ddp: bool) -> Optional[DistributedSampler]:
    if not use_ddp:
        return None
    return DistributedSampler(dataset, shuffle=shuffle)


def set_sampler_epoch(sampler: Any, epoch: int) -> None:
    """DDP shuffle seed per epoch; safe no-op if no sampler."""
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(int(epoch))


def make_dataloader_worker_init_fn(rank: int, base_seed: int = 3407) -> Callable[[int], None]:
    """Stable worker RNG per rank for reproducible resume (use with ``set_sampler_epoch``)."""

    def _fn(worker_id: int) -> None:
        s = int(base_seed) + int(worker_id) + int(rank) * 1_000_003
        random.seed(s)
        torch.manual_seed(s)

    return _fn


def loader_with_sampler(
    dataset,
    batch_size: int,
    collate_fn,
    sampler: Optional[DistributedSampler],
    num_workers: int = 0,
    worker_init_fn: Optional[Callable[[int], None]] = None,
    generator: Optional[torch.Generator] = None,
    *,
    persistent_workers: bool = False,
    prefetch_factor: Optional[int] = None,
    drop_last: bool = False,
) -> DataLoader:
    kw: dict[str, Any] = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=bool(drop_last),
    )
    if worker_init_fn is not None:
        kw["worker_init_fn"] = worker_init_fn
    if generator is not None:
        kw["generator"] = generator
    if num_workers > 0:
        if persistent_workers:
            kw["persistent_workers"] = True
        if prefetch_factor is not None:
            kw["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(**kw)

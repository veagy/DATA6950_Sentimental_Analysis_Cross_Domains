"""
Distributed training helpers for DDP (DistributedDataParallel) and FSDP.
Used when sentinel.conf [training] distributed = true, launched via torchrun.
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def init_distributed() -> tuple[int, int]:
    """
    Initialise the distributed process group from environment variables set by
    ``torchrun`` / ``torch.distributed.launch``.

    Returns
    -------
    (local_rank, world_size)
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "12355")

    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", master_port)

    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        torch.cuda.set_device(local_rank)

    return local_rank, world_size


def wrap_ddp(model: torch.nn.Module, local_rank: int) -> torch.nn.Module:
    """Wrap model in DistributedDataParallel for multi-GPU training."""
    return DDP(model, device_ids=[local_rank], output_device=local_rank)


def cleanup_distributed() -> None:
    """Tear down the process group gracefully."""
    if dist.is_initialized():
        dist.destroy_process_group()

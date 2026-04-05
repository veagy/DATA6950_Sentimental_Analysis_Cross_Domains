# src/train/train_ddp.py
"""
Multi-GPU DDP training script.

Launch with torchrun:
  Single node, 2 GPUs:
    torchrun --nproc_per_node=2 src/train/train_ddp.py --model MyModel --epochs 30

  Multi-node, 3 VMs × 2 GPUs each:
    torchrun --nnodes=3 --node_rank=0 --master_addr=192.168.1.10 --master_port=29500 \
             --nproc_per_node=2 src/train/train_ddp.py --model MyModel --epochs 50

Note: each process handles batch_size samples.
      Effective batch size = batch_size × world_size.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ..config.deep_learning import instantiate_model
from ..train.utils.data_loader import make_loader
from ..train.utils.distributed import (
    init_distributed, wrap_model_ddp,
    make_distributed_loader, cleanup_distributed,
)
from ..train.utils.checkpoint import save_checkpoint, load_resume_state

# module-level epoch tracker for DistributedSampler.set_epoch()
_epoch = 0


def train_epoch(model, loader, optimizer, criterion, scaler, accum_steps, rank):
    """Run one training epoch in a DDP context."""
    model.train()
    total_loss = 0.0
    if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(_epoch)        # reshuffle per epoch

    optimizer.zero_grad()
    for i, batch in enumerate(loader):
        if isinstance(batch, (tuple, list)):
            X = batch[0].cuda(rank, non_blocking=True) if torch.cuda.is_available() else batch[0]
            y = batch[1].cuda(rank, non_blocking=True) if len(batch) > 1 and torch.cuda.is_available() else (batch[1] if len(batch) > 1 else None)
        else:
            X = batch.cuda(rank, non_blocking=True) if torch.cuda.is_available() else batch
            y = None

        with torch.amp.autocast("cuda", enabled=scaler is not None and torch.cuda.is_available()):
            out  = model(X)
            loss = (criterion(out, y) / accum_steps
                    if y is not None else out.mean() / accum_steps)

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % accum_steps == 0:
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps

    return total_loss / max(1, len(loader))


def main():
    global _epoch

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",           type=str,   required=True)
    parser.add_argument("--data_source",     type=str,   default=None,
                        help="Path/URI to training data compatible with make_loader")
    parser.add_argument("--epochs",          type=int,   default=20)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--grad_accum",      type=int,   default=1)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--save_dir",        type=str,   default="checkpoints/ddp")
    parser.add_argument("--checkpoint",      type=str,   default=None)
    parser.add_argument("--num_workers",     type=int,   default=4)
    args = parser.parse_args()

    # ── Init DDP ──────────────────────────────────────────────────────
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    rank, world_size = init_distributed(backend)

    # ── Model ─────────────────────────────────────────────────────────
    if args.checkpoint:
        model_inner = type(instantiate_model(args.model)).load_model(args.checkpoint)
    else:
        model_inner = instantiate_model(args.model)

    model = wrap_model_ddp(model_inner, rank)

    # ── Data ──────────────────────────────────────────────────────────
    if args.data_source and str(args.data_source).strip():
        try:
            p = Path(args.data_source)
            if not p.is_absolute(): p = Path.cwd() / p
            if not p.exists(): raise FileNotFoundError(f"Data source not found: {p}")
            out = make_loader(args.data_source, batch_size=args.batch_size, num_workers=min(args.num_workers, os.cpu_count() or 1))
            dataset_loader = out[0] if isinstance(out, tuple) else out
            loader = make_distributed_loader(dataset_loader, rank, world_size, args.batch_size, args.num_workers)
        except Exception as e:
            if rank == 0: 
                print(f"[ERROR] Failed data_source loading for DDP: {e}.")
            cleanup_distributed()
            raise ValueError(f"Failed to load data from '{args.data_source}'. Error: {e}") from e
    else:
        cleanup_distributed()
        raise ValueError("A valid --data_source must be provided for DDP training. Synthetic fallback data is no longer supported.")

    # ── Optimizer + Loss ──────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scaler    = (torch.amp.GradScaler("cuda")
                 if args.mixed_precision and torch.cuda.is_available() else None)

    # ── Training loop ─────────────────────────────────────────────────
    for _epoch in range(args.epochs):
        loss = train_epoch(model, loader, optimizer, criterion,
                           scaler, args.grad_accum, rank)
        if rank == 0:
            print(f"Epoch {_epoch+1}/{args.epochs}  loss={loss:.4f}")
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated(rank) / 1e9
                print(f"  GPU {rank} memory: {alloc:.2f} GB")

    # ── Save from rank 0 only ─────────────────────────────────────────
    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        save_checkpoint(
            model_inner,   # save unwrapped model, not the DDP wrapper
            args.save_dir,
            "final.pt",
            manifest_extra={
                "training_state": {
                    "optimizer": "adamw",
                    "lr": args.lr,
                    "epochs": args.epochs,
                    "world_size": world_size,
                }
            }
        )
        print(f"[DONE] Saved to {args.save_dir}/final.pt")

    cleanup_distributed()


if __name__ == "__main__":
    main()

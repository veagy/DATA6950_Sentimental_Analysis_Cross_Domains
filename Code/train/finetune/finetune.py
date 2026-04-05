"""
Fine-tuning script: load a pretrained checkpoint, optionally freeze early layers,
then train on a downstream task.

Invoked by ``sentinel train`` when ``sentinel.conf [training] mode = finetune``,
or directly via ``python -m src.train.finetune.finetune``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def main() -> None:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    def get_sentinel_config():
        config_path = Path("./sentinel.conf")
        try:
            import json
            with open(config_path) as f:
                return json.load(f)
        except:
            return {}

    def log_event(*args, **kwargs):
        pass
    from Code.train.train import _build_model, _validate
    from Code.train.utils.checkpoint import CheckpointManager
    from Code.train.utils.data_loader import build_dataloader

    parser = argparse.ArgumentParser(description="Sentinel fine-tuning loop")
    parser.add_argument("--epochs",        type=int,   default=None)
    parser.add_argument("--batch_size",    type=int,   default=None)
    parser.add_argument("--lr",            type=float, default=None)
    parser.add_argument("--freeze_to",     type=int,   default=None,
                        help="Freeze the first N parameter tensors")
    parser.add_argument("--pretrain_ckpt", type=str,   default=None,
                        help="Path to pretrained .pth file to load before fine-tuning")
    args = parser.parse_args()

    config    = get_sentinel_config()
    train_cfg = config.get("training", {})
    epochs     = args.epochs     or int(train_cfg.get("finetune_epochs", 20))
    batch_size = args.batch_size or int(train_cfg.get("batch_size",      32))
    lr         = args.lr         or float(train_cfg.get("finetune_lr",   1e-5))
    freeze_to  = (
        args.freeze_to
        if args.freeze_to is not None
        else int(train_cfg.get("freeze_to", 0))
    )

    device = torch.device(config.get("hardware", {}).get("device", "cpu"))
    model  = _build_model(config).to(device)

    if args.pretrain_ckpt:
        state = torch.load(args.pretrain_ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print(f"[finetune] Loaded pretrained weights from {args.pretrain_ckpt}")

    # Freeze first N parameter tensors
    params = list(model.parameters())
    frozen = 0
    for i, p in enumerate(params):
        if i < freeze_to:
            p.requires_grad_(False)
            frozen += 1
    if frozen:
        print(f"[finetune] Froze {frozen}/{len(params)} parameter tensors.")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer  = optim.AdamW(trainable, lr=lr)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion  = nn.CrossEntropyLoss()

    train_loader = build_dataloader("train", batch_size=batch_size, config=config)
    val_loader   = build_dataloader("val",   batch_size=batch_size, config=config)
    ckpt         = CheckpointManager(config)

    log_event("FINETUNE_START", {"epochs": epochs, "freeze_to": freeze_to, "lr": lr})

    for epoch in range(epochs):
        model.train()
        total = 0.0

        for batch in train_loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()

        scheduler.step()
        avg      = total / max(len(train_loader), 1)
        val_loss = _validate(model, val_loader, criterion, device)
        ckpt.save(model, epoch=epoch, metrics={"loss": avg, "val_loss": val_loss})
        log_event("FINETUNE_EPOCH", {"epoch": epoch, "loss": avg, "val_loss": val_loss})
        print(f"[FineTune Epoch {epoch + 1}/{epochs}] loss={avg:.4f} val_loss={val_loss:.4f}")

    log_event("FINETUNE_END", {})
    print("[finetune] Fine-tuning complete.")


if __name__ == "__main__":
    main()

"""
Pre-training script (self-supervised / masked / generative objective).

Invoked by ``sentinel train`` when ``sentinel.conf [training] mode = pretrain``,
or directly via ``python -m src.train.pretrain.pretrain``.
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
    from Code.train.train import _build_model
    from Code.train.utils.checkpoint import CheckpointManager
    from Code.train.utils.data_loader import build_dataloader

    parser = argparse.ArgumentParser(description="Sentinel pre-training loop")
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--batch_size", type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    args = parser.parse_args()

    config     = get_sentinel_config()
    train_cfg  = config.get("training", {})
    epochs     = args.epochs     or int(train_cfg.get("epochs",     100))
    batch_size = args.batch_size or int(train_cfg.get("batch_size", 64))
    lr         = args.lr         or float(train_cfg.get("lr",        1e-4))

    device    = torch.device(config.get("hardware", {}).get("device", "cpu"))
    model     = _build_model(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loader    = build_dataloader("train", batch_size=batch_size, config=config)
    ckpt      = CheckpointManager(config)

    log_event("PRETRAIN_START", {"epochs": epochs})

    for epoch in range(epochs):
        model.train()
        total = 0.0

        for batch in loader:
            x = batch[0].to(device)
            # 15 % masking — replace masked positions with 0
            mask    = torch.rand_like(x) < 0.15
            x_in    = x.clone()
            x_in[mask] = 0.0
            pred    = model(x_in)
            loss    = nn.MSELoss()(pred[mask], x[mask])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()

        avg = total / max(len(loader), 1)
        ckpt.save(model, epoch=epoch, metrics={"pretrain_loss": avg})
        log_event("PRETRAIN_EPOCH", {"epoch": epoch, "loss": avg})
        print(f"[PreTrain Epoch {epoch + 1}/{epochs}] loss={avg:.4f}")

    log_event("PRETRAIN_END", {})
    print("[pretrain] Pre-training complete.")


if __name__ == "__main__":
    main()

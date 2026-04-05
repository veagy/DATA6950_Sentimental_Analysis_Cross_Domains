#!/usr/bin/env python
"""
Main training entrypoint for ``sentinel train``.

Discovered automatically by ``src/interface/commands/execution.py``
via ``_discover_train_script()``.

Environment variables injected by the CLI
------------------------------------------
SENTINEL_CONFIG   Path to sentinel.conf
SENTINEL_MODELS   Path to model registry root
"""
from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

# Allow running as a standalone script from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sentinel training loop")
    parser.add_argument("--epochs",      type=int,   default=None, help="Number of training epochs")
    parser.add_argument("--batch_size",  type=int,   default=None, help="Mini-batch size")
    parser.add_argument("--lr",          type=float, default=None, help="Learning rate")
    parser.add_argument("--resume",      action="store_true",      help="Resume from last checkpoint")
    parser.add_argument("--distributed", action="store_true",      help="Enable DDP multi-GPU training")
    parser.add_argument("--mode",        type=str,   default=None,
                        choices=["supervised", "pretrain", "finetune"],
                        help="Training mode (overrides sentinel.conf)")
    parser.add_argument("--model-class", dest="model_class", type=str, default=None,
                        help="Classifier class name from MODEL_CATALOG (e.g. DeepClassifier)")
    return parser.parse_args()


def _build_model(config: dict, model_class_override: str | None = None):
    """Instantiate a model from sentinel config.

    Priority order:
    0. model_class_override (from --model-class CLI arg / POST /api/train)
    1. sentinel.conf [model] section — class name in MODEL_CATALOG.
    2. AdvancedPipeline from a .mmd flowchart file.
    3. First entry in src/config/deep_learning/model_config_registry.json.
    """
    import importlib
    import json
    from Code.models.simple_classifier import MODEL_CATALOG

    model_cfg = config.get("model", {})
    in_features  = int(model_cfg.get("in_features",  128))
    out_features = int(model_cfg.get("out_features", 2))
    hidden       = model_cfg.get("hidden", "64,32")
    dropout      = float(model_cfg.get("dropout", 0.0))

    # --- Priority 0: explicit --model-class override ---
    if model_class_override and model_class_override in MODEL_CATALOG:
        cls = MODEL_CATALOG[model_class_override]
        print(f"[train] Using model class from override: {model_class_override}")
        return cls(in_features=in_features, out_features=out_features,
                   hidden=hidden, dropout=dropout)

    # --- Priority 1: sentinel.conf [model] section ---
    if model_cfg.get("class"):
        cls_name = model_cfg["class"].strip()
        if cls_name in MODEL_CATALOG:
            cls = MODEL_CATALOG[cls_name]
            print(f"[train] Using model class from sentinel.conf: {cls_name}")
            return cls(in_features=in_features, out_features=out_features,
                       hidden=hidden, dropout=dropout)
        # Generic: try to import from dotted path or Code.models
        for module_path in (f"Code.models.{cls_name.lower()}", "Code.models"):
            try:
                mod = importlib.import_module(module_path)
                cls = getattr(mod, cls_name)
                kwargs = {k: v for k, v in model_cfg.items() if k != "class"}
                return cls(**kwargs)
            except (ImportError, AttributeError):
                continue

    PATH_MERMAID = Path("./configs/pipeline/mermaid")
    PATH_MODEL_CONFIG_REGISTRY = Path("./configs/models/model_config_registry.json")

    # --- Priority 2: Pipeline from mermaid ---
    try:
        from Code.models.models import Pipeline
        mmd_files = sorted(PATH_MERMAID.glob("*.mmd")) if PATH_MERMAID.exists() else []
        if mmd_files:
            content = mmd_files[0].read_text(encoding="utf-8")
            return Pipeline(mermaid_flowchart=content, modules={})
    except Exception:
        pass

    # --- Priority 3: model_config_registry.json ---
    if PATH_MODEL_CONFIG_REGISTRY.exists():
        registry = json.loads(PATH_MODEL_CONFIG_REGISTRY.read_text())
        if registry:
            entry = next(iter(registry.values()))
            module_path, cls_name = entry["class"].rsplit(".", 1)
            mod = importlib.import_module(module_path)
            cls = getattr(mod, cls_name)
            return cls(**entry.get("kwargs", {}))

    # --- Fallback: simple binary classifier on default 128-dim input ---
    print("[train] No model config found — using default SimpleClassifier(128→64→32→2)")
    from Code.models.simple_classifier import SimpleClassifier
    return SimpleClassifier(in_features=128, out_features=2)


def _validate(model, loader, criterion, device) -> float:
    import torch

    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            total += criterion(model(inputs), labels).item()
    return total / max(len(loader), 1)


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

    from Code.train.utils.checkpoint import CheckpointManager
    from Code.train.utils.data_loader import build_dataloader

    args = parse_args()
    config = get_sentinel_config()
    train_cfg = config.get("training", {})

    mode = args.mode or train_cfg.get("mode", "supervised")

    # Delegate to specialised scripts for non-supervised modes
    if mode == "pretrain":
        from Code.train.pretrain.pretrain import main as pretrain_main

        sys.argv = [sys.argv[0]] + [
            a for a in sys.argv[1:] if not a.startswith("--mode")
        ]
        return pretrain_main()

    if mode == "finetune":
        from Code.train.finetune.finetune import main as finetune_main

        sys.argv = [sys.argv[0]] + [
            a for a in sys.argv[1:] if not a.startswith("--mode")
        ]
        return finetune_main()

    # --- Supervised training ---
    epochs     = args.epochs     or int(train_cfg.get("epochs",     50))
    batch_size = args.batch_size or int(train_cfg.get("batch_size", 32))
    lr         = args.lr         or float(train_cfg.get("lr",       1e-3))

    train_loader = build_dataloader("train", batch_size=batch_size, config=config)
    val_loader   = build_dataloader("val",   batch_size=batch_size, config=config)

    device = torch.device(config.get("hardware", {}).get("device", "cpu"))
    model  = _build_model(config, model_class_override=getattr(args, "model_class", None)).to(device)

    if args.distributed:
        from Code.train.utils.distributed import init_distributed, wrap_ddp

        local_rank, _ = init_distributed()
        model = wrap_ddp(model, local_rank)

    ckpt        = CheckpointManager(config)
    start_epoch = ckpt.resume(model) if args.resume else 0
    optimizer   = optim.AdamW(model.parameters(), lr=lr)
    criterion   = nn.CrossEntropyLoss()

    # SIGINT / SIGTERM → save interrupted checkpoint
    interrupted = False

    def _handle_signal(sig, frame):
        nonlocal interrupted
        interrupted = True
        print("\n[train] Signal received — saving interrupted checkpoint …")

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    log_event("TRAINING_START", {"epochs": epochs, "batch_size": batch_size, "lr": lr})

    avg_loss = 0.0
    try:
        for epoch in range(start_epoch, epochs):
            if interrupted:
                break

            model.train()
            total_loss = 0.0

            for batch in train_loader:
                if interrupted:
                    break
                inputs, labels = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss    = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / max(len(train_loader), 1)
            val_loss = _validate(model, val_loader, criterion, device)

            ckpt.save(model, epoch=epoch, metrics={"loss": avg_loss, "val_loss": val_loss})
            log_event("EPOCH_COMPLETE", {"epoch": epoch, "loss": avg_loss, "val_loss": val_loss})
            print(f"[Epoch {epoch + 1}/{epochs}] loss={avg_loss:.4f} val_loss={val_loss:.4f}")

    except Exception as exc:
        ckpt.save(model, epoch=start_epoch, metrics={"error": str(exc)})
        log_event("TRAINING_ERROR", {"error": str(exc)})
        raise
    finally:
        if args.distributed:
            from Code.train.utils.distributed import cleanup_distributed
            cleanup_distributed()

    log_event("TRAINING_END", {"final_loss": avg_loss})
    print("[train] Training complete.")


if __name__ == "__main__":
    main()

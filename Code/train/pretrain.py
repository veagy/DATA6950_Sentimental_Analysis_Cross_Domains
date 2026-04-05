"""
Scenario A/B Phase 1: single training run or pre-training pass.

Usage:
  python src/train/pretrain.py --model StandardDense
  python src/train/pretrain.py --model StandardDense --epochs 50
  python src/train/pretrain.py --model StandardDense --checkpoint checkpoints/pretrained/interrupted.pt
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path

import torch

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ..config.deep_learning import instantiate_model
from ..train.utils import (
    clear_resume_state,
    load_resume_state,
    save_checkpoint,
    save_resume_state,
)
from ..train.utils.data_loader import make_loader

# ── Module-level state for interrupt handler ─────────────────────────
_current_epoch = 0
_current_batch = 0
_global_step = 0
_model = None
_save_dir: str = "checkpoints"


def _handle_interrupt(sig, frame):
    print(
        f"\n[INTERRUPT] Signal {sig}. epoch={_current_epoch} batch={_current_batch}. "
        "Saving checkpoint...",
        flush=True,
    )
    if _model is not None:
        interrupted_path = os.path.join(_save_dir, "interrupted.pt")
        _model.save_model(interrupted_path)
        save_resume_state(_save_dir, _current_epoch, _current_batch, _global_step, interrupted_path)
        print(
            f"[INTERRUPT] Saved to {interrupted_path}. Re-run with "
            f"--checkpoint {interrupted_path} to resume.",
            flush=True,
        )
    sys.exit(0)


signal.signal(signal.SIGINT, _handle_interrupt)
signal.signal(signal.SIGTERM, _handle_interrupt)


def _build_data(args) -> torch.utils.data.DataLoader:
    """
    Build training DataLoader. Uses real data when --data_source provided,
    otherwise synthetic placeholder. Integrates with Phase 1 make_loader
    (file paths, URIs, clean, scale, paradigm).
    """
    data_source = getattr(args, "data_source", None)
    label_col = getattr(args, "label_col", "label")
    val_split = getattr(args, "val_split", 0.0)
    clean = getattr(args, "clean", False)
    scale = getattr(args, "scale", None)
    paradigm = getattr(args, "paradigm", "supervised")

    if data_source and str(data_source).strip():
        try:
            p = Path(data_source)
            if not p.is_absolute():
                p = Path.cwd() / p
            if not p.exists():
                raise FileNotFoundError(f"Data source not found: {data_source}")
            out = make_loader(
                data_source,
                batch_size=args.batch_size,
                num_workers=min(4, os.cpu_count() or 1),
                pin_memory=True,
                val_split=val_split,
                label_col=label_col,
                clean=clean,
                scale=scale,
                paradigm=paradigm,
            )
            if isinstance(out, tuple):
                return out[0]
            return out
        except Exception as e:
            raise ValueError(f"Failed to load data from '{data_source}'. Error: {e}") from e

    raise ValueError("A valid --data_source must be provided for production training. Synthetic fallback data is no longer supported.")


def main():
    global _current_epoch, _current_batch, _global_step, _model, _save_dir

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="StandardDense")
    parser.add_argument("--in_features", type=int, default=128)
    parser.add_argument("--out_features", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--loss", type=str, default="CrossEntropyLoss")
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="checkpoints/pretrained")
    parser.add_argument("--save_type", type=str, default="pt")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--no_progress_bar", action="store_true")
    parser.add_argument("--data_source", type=str, default=None)
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--val_split", type=float, default=0.0)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--scale", type=str, default=None)
    parser.add_argument("--paradigm", type=str, default="supervised")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    _save_dir = args.save_dir
    os.makedirs(_save_dir, exist_ok=True)

    start_epoch = 0

    # ── Resume / init logic ──────────────────────────────────────────
    if args.checkpoint:
        print(f"[RESUME] Loading checkpoint: {args.checkpoint}")
        try:
        # Use dimensions from args
        in_features = args.in_features
        out_features = args.out_features
        try:
            _model = instantiate_model(
                args.model,
                overrides={"in_features": in_features, "out_features": out_features},
            )
        except (KeyError, Exception):
            _model = instantiate_model(
                "StandardDense",
                overrides={"in_features": in_features, "out_features": out_features},
            )
        _model = type(_model).load_model(args.checkpoint)
        info = load_resume_state(_save_dir)
        if info:
            start_epoch = info.get("epoch", 0)
            print(f"[RESUME] Resuming from epoch {start_epoch}")
    else:
        info = load_resume_state(_save_dir)
        if info:
            checkpoint_path = info.get("checkpoint")
            if checkpoint_path and Path(checkpoint_path).exists():
                try:
                    _model = instantiate_model(
                        args.model,
                        overrides={"in_features": args.in_features, "out_features": args.out_features},
                    )
                except (KeyError, Exception):
                    _model = instantiate_model(
                        "StandardDense",
                        overrides={"in_features": args.in_features, "out_features": args.out_features},
                    )
                _model = type(_model).load_model(checkpoint_path)
                start_epoch = info.get("epoch", 0)
                print(f"[RESUME] Auto-resuming from {checkpoint_path} epoch {start_epoch}")
            else:
                try:
                    _model = instantiate_model(
                        args.model,
                        overrides={"in_features": args.in_features, "out_features": args.out_features},
                    )
                except (KeyError, Exception):
                    _model = instantiate_model(
                        "StandardDense",
                        overrides={"in_features": args.in_features, "out_features": args.out_features},
                    )
                print(f"[INIT] Instantiated model: {_model.get_class_type()}")
        else:
            try:
                _model = instantiate_model(
                    args.model,
                    overrides={"in_features": 128, "out_features": 3},
                )
            except (KeyError, Exception):
                _model = instantiate_model(
                    "StandardDense",
                    overrides={"in_features": 128, "out_features": 3},
                )
            print(f"[INIT] Instantiated model: {_model.get_class_type()}")

    # ── Data ─────────────────────────────────────────────────────────
    loader = _build_data(args)

    # ── Training ─────────────────────────────────────────────────────
    effective_epochs = args.epochs - start_epoch
    if effective_epochs <= 0:
        print("[SKIP] Already completed requested epochs.")
        return

    try:
        from ..train.utils.logging import trace
        trace(f"[TRAIN] Starting pretrain for {effective_epochs} epochs.")
    except Exception:
        pass

    try:
        history = _model.fit(
            data=loader,
            epochs=effective_epochs,
            learning_rate=args.lr,
            loss=args.loss,
            optimizer=args.optimizer,
            weight_decay=args.weight_decay,
            gradient_accumulation_steps=args.grad_accum,
            mixed_precision=args.mixed_precision,
            show_progress_bar=not args.no_progress_bar,
            verbose=True,
            save_dir=_save_dir,
            save_type=args.save_type,
        )
    except torch.cuda.OutOfMemoryError as oom:
        try:
            from ..train.utils.logging import trace
            trace(f"[FAILURE-MGMT] OOM at epoch {_current_epoch}. Dumping recovery checkpoint.")
        except Exception:
            pass
        torch.cuda.empty_cache()
        error_path = os.path.join(_save_dir, "oom_recovery.pt")
        _model.save_model(error_path)
        save_resume_state(
            _save_dir, _current_epoch, checkpoint_path=error_path, error="OOM"
        )
        raise RuntimeError(
            f"OOM at epoch {_current_epoch}. Checkpoint: {error_path}"
        ) from oom
    except Exception as exc:
        try:
            from ..train.utils.logging import trace
            trace(f"[FAILURE-MGMT] Critical error: {str(exc)}. Dumping recovery checkpoint.")
        except Exception:
            pass
        error_path = os.path.join(_save_dir, "error_recovery.pt")
        _model.save_model(error_path)
        save_resume_state(
            _save_dir, _current_epoch, checkpoint_path=error_path, error=str(exc)
        )
        print(f"[ERROR] Saved recovery checkpoint: {error_path}")
        raise

    # ── Save final ───────────────────────────────────────────────────
    try:
        from ..train.utils.logging import trace
        trace("[TRAIN] Epochs complete. Saving final model.")
    except Exception:
        pass
    
    if hasattr(_model, "detach_pipeline"):
        _model.detach_pipeline()
    if hasattr(_model, "_system_pipeline"):
        _model._system_pipeline = None  # Break circular ref before save
    save_checkpoint(
        _model,
        _save_dir,
        filename="final.pt",
        save_type=args.save_type,
        manifest_extra={
            "training_state": {
                "optimizer": args.optimizer,
                "lr": args.lr,
                "epochs": args.epochs,
                "loss": args.loss,
            }
        },
    )
    print("[DONE] Final model and manifest.json written.")

    # ── Clear resume file ────────────────────────────────────────────
    clear_resume_state(_save_dir)

    if history is not None:
        print(history.tail())


if __name__ == "__main__":
    main()

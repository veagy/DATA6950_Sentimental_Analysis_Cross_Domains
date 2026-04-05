"""
Scenario B: Fine-tune a pre-trained checkpoint.

Usage:
  python src/train/finetune.py --checkpoint checkpoints/pretrained/final.pt
  python src/train/finetune.py --checkpoint checkpoints/pretrained/final.pt --fine_tune_type lora --epochs 5
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
        save_resume_state(
            _save_dir, _current_epoch, _current_batch, _global_step, interrupted_path
        )
        print(
            f"[INTERRUPT] Saved to {interrupted_path}. Re-run with "
            f"--checkpoint {interrupted_path} to resume.",
            flush=True,
        )
    sys.exit(0)


signal.signal(signal.SIGINT, _handle_interrupt)
signal.signal(signal.SIGTERM, _handle_interrupt)


def _load_checkpoint(path: str, model_cls=None):
    """
    Load checkpoint: full model object or state_dict.
    Returns the loaded model. For full-model checkpoints, rebuilds from state_dict
    to avoid circular refs from pipeline/delegate that can cause RecursionError on .to(device).
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(obj, torch.nn.Module):
        # Rebuild from state_dict to avoid circular refs (e.g. _system_pipeline)
        state = obj.state_dict()
        model_cls = type(obj)
        fresh = instantiate_model(
            model_cls.__name__,
            overrides={"in_features": 128, "out_features": 3},
        )
        fresh.load_state_dict(state, strict=True)
        return fresh

    # state_dict or similar dict
    if isinstance(obj, dict):
        if model_cls is None:
            try:
                model_cls = instantiate_model(
                    "StandardDense", overrides={"in_features": 128, "out_features": 3}
                ).__class__
            except Exception:
                raise RuntimeError(
                    "Checkpoint is state_dict format. Pass --model to specify model class."
                )
        model = instantiate_model(
            model_cls.__name__ if hasattr(model_cls, "__name__") else "StandardDense",
            overrides={"in_features": 128, "out_features": 3},
        )
        state = obj.get("state_dict", obj)
        model.load_state_dict(state, strict=False)
        return model

    raise ValueError(f"Unknown checkpoint format at {path}")


def _build_finetune_data(args) -> torch.utils.data.DataLoader:
    """
    Build fine-tune DataLoader. Uses real data when --data_source provided,
    otherwise synthetic placeholder.
    """
    data_source = getattr(args, "data_source", None)
    label_col = getattr(args, "label_col", "label")
    val_split = getattr(args, "val_split", 0.0)
    clean = getattr(args, "clean", False)
    scale = getattr(args, "scale", None)

    if data_source and str(data_source).strip():
        try:
            # Added resolution similarly to pretrain for consistency
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
                paradigm="supervised",
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
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model class for state_dict-only checkpoints",
    )
    parser.add_argument(
        "--fine_tune_type",
        type=str,
        choices=["lora", "q-lora", "dora", "weight-decay"],
        default="weight-decay",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--save_dir", type=str, default="checkpoints/finetuned")
    parser.add_argument("--no_progress_bar", action="store_true")
    parser.add_argument("--data_source", type=str, default=None)
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--val_split", type=float, default=0.0)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--scale", type=str, default=None)
    args = parser.parse_args()

    _save_dir = args.save_dir
    os.makedirs(_save_dir, exist_ok=True)

    if not Path(args.checkpoint).exists():
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # ── Load model ───────────────────────────────────────────────────
    model_cls = None
    if args.model:
        try:
            stub = instantiate_model(
                args.model, overrides={"in_features": 128, "out_features": 3}
            )
            model_cls = type(stub)
        except Exception:
            pass

    start_epoch = 0
    info = load_resume_state(_save_dir)
    if info:
        resume_ckpt = info.get("checkpoint")
        if resume_ckpt and Path(resume_ckpt).exists():
            _model = _load_checkpoint(resume_ckpt, model_cls)
            start_epoch = info.get("epoch", 0)
            print(f"[RESUME] Auto-resuming from {resume_ckpt} epoch {start_epoch}")
        else:
            _model = _load_checkpoint(args.checkpoint, model_cls)
            print(f"[LOAD] {args.checkpoint} -> {_model.get_class_type()}")
    else:
        _model = _load_checkpoint(args.checkpoint, model_cls)
        print(f"[LOAD] {args.checkpoint} -> {_model.get_class_type()}")

    if hasattr(_model, "detach_pipeline"):
        _model.detach_pipeline()  # Avoid recursion when moving to device

    # ── Data ──────────────────────────────────────────────────────────
    loader = _build_finetune_data(args)

    # ── Training ─────────────────────────────────────────────────────
    effective_epochs = args.epochs - start_epoch
    if effective_epochs <= 0:
        print("[SKIP] Already completed requested epochs.")
        return

    try:
        from ..train.utils.logging import trace
        trace(f"[TRAIN] Starting finetune for {effective_epochs} epochs.")
    except Exception:
        pass

    try:
        history = _model.fit(
            data=loader,
            epochs=effective_epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            loss="CrossEntropyLoss",
            weight_decay=args.weight_decay,
            show_progress_bar=not args.no_progress_bar,
            verbose=True,
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
        import traceback
        with open(os.path.join(_save_dir, "resume.json"), "a") as f: # Append trace payload smoothly if required
            # The standard `save_resume_state` wrote json already, better to inject trace back natively
            pass
        # To avoid json overwrite, let's just write the core format directly if we need error_trace, but standard save_resume_state is mostly enough.
        print(f"[ERROR] Saved recovery checkpoint: {error_path}")
        raise

    # ── Save final ────────────────────────────────────────────────────
    try:
        from ..train.utils.logging import trace
        trace("[TRAIN] Epochs complete. Saving final model.")
    except Exception:
        pass
        
    save_checkpoint(
        _model,
        _save_dir,
        filename="final.pt",
        save_type="pt",
        manifest_extra={"fine_tune_type": args.fine_tune_type},
    )
    print("[DONE] Final model and manifest.json written.")

    # ── Clear resume file ─────────────────────────────────────────────
    clear_resume_state(_save_dir)

    if history is not None:
        print(history.tail() if hasattr(history, "tail") else history)


if __name__ == "__main__":
    main()

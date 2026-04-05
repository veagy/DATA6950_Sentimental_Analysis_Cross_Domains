"""
Scenario C Phase 3: fine-tune assembled MoE pipeline to correct offset errors.

Three strategies:
  A — freeze sub-models, train LAZY adapters only (fast, safest)
  B — differential LR (sub-models + adapters with controlled update rates)
  C — LoRA on full pipeline (PEFT-based, fewest trainable params)

Usage:
  python src/train/finetune_pipeline.py \
      --checkpoint checkpoints/moe/assembled.pt \
      --strategy A --epochs 5 --lr 1e-4
"""

import argparse
import json
import os
import signal
import sys
from pathlib import Path

import torch
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ..models.models import Pipeline
from ..train.utils.data_loader import make_loader

_pipeline      = None
_save_dir      = "checkpoints/moe"
_current_epoch = 0


def _handle_interrupt(sig, frame):
    if _pipeline is not None:
        from ..train.utils.checkpoint import save_checkpoint, save_resume_state
        try:
            interrupted_path = save_checkpoint(
                _pipeline, 
                save_dir=_save_dir, 
                filename="interrupted.pt",
                write_wal_hooks=False
            )
            save_resume_state(
                save_dir=_save_dir,
                epoch=_current_epoch,
                checkpoint_path=interrupted_path
            )
            print(f"[INTERRUPT] Saved to {interrupted_path}", flush=True)
        except Exception as e:
            print(f"[INTERRUPT] Error during recovery save: {e}", flush=True)
    sys.exit(0)


signal.signal(signal.SIGINT,  _handle_interrupt)
signal.signal(signal.SIGTERM, _handle_interrupt)


def _build_combined_data(args):
    """Build the fine-tuning dataset loader."""
    data_source = getattr(args, "data_source", None)
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
            )
            if isinstance(out, tuple):
                return out[0]
            return out
        except Exception as e:
            raise ValueError(f"Failed to load data from '{data_source}'. Error: {e}") from e

    raise ValueError("A valid --data_source must be provided for pipeline fine-tuning. Synthetic fallback data is no longer supported.")


def main():
    global _pipeline, _save_dir, _current_epoch

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   type=str, required=True)
    parser.add_argument("--strategy",     type=str, default="A",
                        choices=["A", "B", "C"],
                        help="A=freeze sub-models, B=differential LR, C=LoRA")
    parser.add_argument("--data_source",  type=str, default=None,
                        help="Path/URI to combined fine-tuning data source")
    parser.add_argument("--epochs",       type=int,   default=5)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--adapter_lr",   type=float, default=1e-3,
                        help="[Strategy B] LR for LAZY adapter layers")
    parser.add_argument("--submodel_lr",  type=float, default=1e-6,
                        help="[Strategy B] LR for pre-trained sub-model layers")
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--save_dir",     type=str,   default="checkpoints/moe")
    parser.add_argument("--no_progress_bar", action="store_true")
    args = parser.parse_args()

    _save_dir = args.save_dir
    os.makedirs(_save_dir, exist_ok=True)

    # ── Load assembled pipeline ───────────────────────────────────────
    print(f"[LOAD] Loading assembled pipeline: {args.checkpoint}")
    _pipeline = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(_pipeline, Pipeline):
        print("[WARN] Checkpoint is not a Pipeline — proceeding as-is.")

    loader = _build_combined_data(args)

    # ═══════════════════════════════════════════════════════════════════
    # STRATEGY A — Freeze sub-models; train LAZY adapters only
    # ═══════════════════════════════════════════════════════════════════
    if args.strategy == "A":
        # Identify sub-model node IDs (those provided during assembly)
        sub_model_node_ids = list(_pipeline._modules_dict.keys())
        print(f"[STRATEGY A] Freezing nodes: {sub_model_node_ids}")
        print(f"[STRATEGY A] Training LAZY adapter nodes only.")
        _pipeline.freeze_nodes(sub_model_node_ids)

        try:
            history = _pipeline.fit(
                data=loader,
                epochs=args.epochs,
                learning_rate=args.lr,
                loss="CrossEntropyLoss",
                optimizer="adamw",
                show_progress_bar=not args.no_progress_bar,
                save_dir=_save_dir,
            )
        except Exception as exc:
            from ..train.utils.checkpoint import save_checkpoint
            save_checkpoint(_pipeline, save_dir=_save_dir, filename="error_recovery.pt", write_wal_hooks=False)
            raise

    # ═══════════════════════════════════════════════════════════════════
    # STRATEGY B — Differential LR: low for sub-models, high for adapters
    # ═══════════════════════════════════════════════════════════════════
    elif args.strategy == "B":
        sub_model_node_ids = list(_pipeline._modules_dict.keys())
        pre_trained_params = []
        for nid in sub_model_node_ids:
            pre_trained_params.extend(_pipeline._modules_dict[nid].parameters())
        pre_trained_set = set(id(p) for p in pre_trained_params)
        adapter_params  = [p for p in _pipeline.parameters()
                           if id(p) not in pre_trained_set]

        print(f"[STRATEGY B] sub-model LR={args.submodel_lr} "
              f"({len(pre_trained_params)} params)")
        print(f"[STRATEGY B] adapter LR={args.adapter_lr} "
              f"({len(adapter_params)} params)")

        optimizer = optim.AdamW([
            {"params": pre_trained_params, "lr": args.submodel_lr},
            {"params": adapter_params,     "lr": args.adapter_lr},
        ], weight_decay=0.01)

        try:
            history = _pipeline.fit(
                data=loader,
                epochs=args.epochs,
                optimizer=optimizer,          # pass pre-built optimizer
                loss="CrossEntropyLoss",
                show_progress_bar=not args.no_progress_bar,
                save_dir=_save_dir,
            )
        except Exception as exc:
            from ..train.utils.checkpoint import save_checkpoint
            save_checkpoint(_pipeline, save_dir=_save_dir, filename="error_recovery.pt", write_wal_hooks=False)
            raise

    # ═══════════════════════════════════════════════════════════════════
    # STRATEGY C — LoRA on the full assembled pipeline (PEFT)
    # ═══════════════════════════════════════════════════════════════════
    elif args.strategy == "C":
        print("[STRATEGY C] Adding LoRA adapters to full pipeline...")
        try:
            history = _pipeline.fine_tune(
                data=loader,
                fine_tune_type="lora",
                epochs=args.epochs,
                learning_rate=args.lr,
                show_progress_bar=not args.no_progress_bar,
                save_dir=_save_dir,
            )
        except Exception as exc:
            from ..train.utils.checkpoint import save_checkpoint
            save_checkpoint(_pipeline, save_dir=_save_dir, filename="error_recovery.pt", write_wal_hooks=False)
            raise

    # ── Save ──────────────────────────────────────────────────────────
    from ..train.utils.checkpoint import save_checkpoint
    final_path = save_checkpoint(
        _pipeline,
        save_dir=_save_dir,
        filename="final.pt",
        manifest_extra={"training_state": {"fine_tune_strategy": args.strategy}}
    )

    print(f"[DONE] Fine-tuned MoE pipeline → {final_path}")

    # Clear resume file
    resume_file = Path(_save_dir) / "resume.json"
    if resume_file.exists():
        resume_file.unlink()


if __name__ == "__main__":
    main()

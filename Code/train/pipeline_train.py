"""
Scenario D: pipeline-first — build Mermaid template, pre-train end-to-end,
then fine-tune on second dataset.

Usage:
  python src/train/pipeline_train.py \
      --template moe_pipeline \
      --pretrain_epochs 50 \
      --finetune_epochs 10
"""

import argparse
import json
import os
import signal
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ..models.models import Pipeline
from ..train.utils.data_loader import make_loader
from ..config.deep_learning import instantiate_model

_pipeline      = None
_save_dir      = "checkpoints/pipeline_train"
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


def main():
    global _pipeline, _save_dir, _current_epoch

    parser = argparse.ArgumentParser()
    parser.add_argument("--template",          type=str, default="moe_pipeline",
                        help="Template name (resolves to .configs/pipeline/mermaid/<name>.mmd)")
    
    # Model configs
    parser.add_argument("--encoder_model",     type=str, default=None,
                        help="Model name for encoder")
    parser.add_argument("--decoder_model",     type=str, default=None,
                        help="Model name for decoder")
    parser.add_argument("--classifier_model",  type=str, default=None,
                        help="Model name for classifier")
                        
    # Data params
    parser.add_argument("--data_source_pretrain", type=str, default=None,
                        help="Path/URI to pre-training dataset")
    parser.add_argument("--data_source_finetune", type=str, default=None,
                        help="Path/URI to fine-tuning dataset")

    parser.add_argument("--pretrain_epochs",   type=int,   default=50)
    parser.add_argument("--finetune_epochs",   type=int,   default=10)
    parser.add_argument("--pretrain_lr",       type=float, default=1e-3)
    parser.add_argument("--finetune_lr",       type=float, default=1e-5)
    parser.add_argument("--fine_tune_type",    type=str,   default="weight-decay",
                        choices=["lora", "q-lora", "dora", "weight-decay"])
    parser.add_argument("--batch_size",        type=int,   default=32)
    parser.add_argument("--mixed_precision",   action="store_true")
    parser.add_argument("--save_dir",          type=str,   default="checkpoints/pipeline_train")
    parser.add_argument("--no_progress_bar",   action="store_true")
    args = parser.parse_args()

    _save_dir    = args.save_dir
    pretrain_dir = os.path.join(_save_dir, "pretrain")
    finetune_dir = os.path.join(_save_dir, "finetune")
    os.makedirs(pretrain_dir,  exist_ok=True)
    os.makedirs(finetune_dir,  exist_ok=True)

    # ── Build pipeline from Mermaid template ─────────────────────────
    # modules dict provides nn.Module instances for named nodes.
    # Nodes not in modules are handled by LAZY auto-injection.
    modules_dict = {}
    if args.encoder_model:
        modules_dict["encoder"] = instantiate_model(args.encoder_model)
    if args.decoder_model:
        modules_dict["decoder"] = instantiate_model(args.decoder_model)
    if args.classifier_model:
        modules_dict["classifier"] = instantiate_model(args.classifier_model)
    
    # Fallback for required nodes if not specified (for testing)
    if "encoder" not in modules_dict:
        print("[WARN] No --encoder_model specified, using standalone TransformerLayer")
        modules_dict["encoder"] = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True)
    if "classifier" not in modules_dict:
        print("[WARN] No --classifier_model specified, using standalone Linear")
        modules_dict["classifier"] = nn.Linear(256, 5)

    _pipeline = Pipeline(
        template_name=args.template,
        modules=modules_dict,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(f"[BUILD] Pipeline from template '{args.template}' constructed.")

    # ── Mandatory validation before first epoch ───────────────────────
    print("[VALIDATE] Running params_calculator()...")
    result = _pipeline.params_calculator()
    print(f"  consistency_status = {result['consistency_status']}")
    if result["consistency_status"] != "SUCCESS":
        print(f"[ERROR] {result.get('errors')}")
        sys.exit(1)

    print("[DRY-RUN] Running dummy_propagate((1, 128))...")
    out_shape, trace = _pipeline.dummy_propagate((1, 128))
    for line in trace:
        print(f"  {line}")
    print(f"[DRY-RUN] Output shape: {out_shape}")

    # ── Dataset factory (replace with real data) ──────────────────────
    def _get_loader(data_source, is_pretrain=True):
        if data_source and str(data_source).strip():
            try:
                p = Path(data_source)
                if not p.is_absolute():
                    p = Path.cwd() / p
                if not p.exists():
                    raise FileNotFoundError(f"Data source not found: {p}")
                out = make_loader(data_source, batch_size=args.batch_size, num_workers=min(4, os.cpu_count() or 1))
                return out[0] if isinstance(out, tuple) else out
            except Exception as e:
                raise ValueError(f"Failed to load data source '{data_source}': {e}") from e

        phase_str = 'pretrain' if is_pretrain else 'finetune'
        raise ValueError(f"A valid --data_source_{phase_str} must be provided. Synthetic fallback data is no longer supported.")

    pretrain_loader = _get_loader(args.data_source_pretrain, is_pretrain=True)
    finetune_loader = _get_loader(args.data_source_finetune, is_pretrain=False)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1 — Pre-training from random initialisation
    # ═══════════════════════════════════════════════════════════════════
    print(f"[PHASE 1] Pre-training pipeline ({args.pretrain_epochs} epochs)...")
    try:
        history_pt = _pipeline.fit(
            data=pretrain_loader,
            epochs=args.pretrain_epochs,
            learning_rate=args.pretrain_lr,
            loss="CrossEntropyLoss",
            optimizer="adamw",
            weight_decay=0.01,
            mixed_precision=args.mixed_precision,
            show_progress_bar=not args.no_progress_bar,
            save_dir=pretrain_dir,
        )
    except Exception as exc:
        from ..train.utils.checkpoint import save_checkpoint
        save_checkpoint(_pipeline, save_dir=pretrain_dir, filename="error_recovery.pt", write_wal_hooks=False)
        raise

    from ..train.utils.checkpoint import save_checkpoint
    final_path = save_checkpoint(
        _pipeline,
        save_dir=pretrain_dir,
        filename="final.pt",
        manifest_extra={"training_phase": "pretrain"}
    )
    print(f"[PHASE 1 DONE] Saved to {final_path}")

    # ── Reload from clean checkpoint for Phase 2 ─────────────────────
    # Reloading eliminates any accumulated optimizer states from Phase 1
    # and ensures a clean PEIC baseline for the fine-tuning phase.
    _pipeline = Pipeline.load_model(os.path.join(pretrain_dir, "final.pt"))
    print("[RELOAD] Reloaded pre-trained pipeline for Phase 2.")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 2 — Fine-tuning on second dataset
    # ═══════════════════════════════════════════════════════════════════
    print(f"[PHASE 2] Fine-tuning ({args.finetune_epochs} epochs, "
          f"fine_tune_type='{args.fine_tune_type}')...")
    try:
        history_ft = _pipeline.fine_tune(
            data=finetune_loader,
            fine_tune_type=args.fine_tune_type,
            epochs=args.finetune_epochs,
            learning_rate=args.finetune_lr,
            show_progress_bar=not args.no_progress_bar,
            save_dir=finetune_dir,
        )
    except Exception as exc:
        from ..train.utils.checkpoint import save_checkpoint
        save_checkpoint(_pipeline, save_dir=finetune_dir, filename="error_recovery.pt", write_wal_hooks=False)
        raise

    from ..train.utils.checkpoint import save_checkpoint
    final_path = save_checkpoint(
        _pipeline,
        save_dir=finetune_dir,
        filename="final.pt",
        manifest_extra={"training_phase": "finetune", "fine_tune_type": args.fine_tune_type}
    )

    print(f"[DONE] Final model → {final_path}")

    # ── Clean up resume files ─────────────────────────────────────────
    for resume_json in [Path(pretrain_dir) / "resume.json",
                        Path(finetune_dir) / "resume.json"]:
        if resume_json.exists():
            resume_json.unlink()


if __name__ == "__main__":
    main()

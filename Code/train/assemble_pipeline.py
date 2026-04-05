"""
Scenario C Phase 2: load pre-trained sub-models and assemble into AdvancedPipeline.
Runs params_calculator() + dummy_propagate() to validate shape connectivity.
Saves assembled pipeline ready for Phase 3 fine-tuning.

Usage:
  python src/train/assemble_pipeline.py \
      --module encoder checkpoints/encoder/final.pt StandardDense \
      --module decoder checkpoints/decoder/final.pt StandardDense \
      --module classifier checkpoints/classifier/final.pt StandardDense \
      --mermaid    .configs/pipeline/mermaid/moe_pipeline.mmd
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ..models.models import Pipeline
from ..config.deep_learning import instantiate_model


def _load_submodel(model_name: str, checkpoint_path: str):
    """Instantiate a real model from registry and load its weights."""
    model = instantiate_model(model_name)
    print(f"  Instantiated {model_name}...")
    model = type(model).load_model(checkpoint_path)
    return model


def main():
    parser = argparse.ArgumentParser()
    # Dynamic sub-models: --module <name> <checkpoint_path> <model_class>
    parser.add_argument("--module", action="append", nargs=3,
                        metavar=("NAME", "PATH", "MODEL"),
                        help="Specify a sub-model (e.g., --module encoder checkpoints/enc.pt StandardDense)")

    parser.add_argument("--mermaid",      type=str,
                        default=".configs/pipeline/mermaid/moe_pipeline.mmd")
    parser.add_argument("--save_dir",     type=str, default="checkpoints/moe")
    parser.add_argument("--input_shape",  type=str, default="1,128",
                        help="Comma-separated input shape for dummy propagation")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Load pre-trained sub-models ──────────────────────────────────
    print("[LOAD] Loading pre-trained sub-models...")
    modules = {}
    if args.module:
        for name, path, model_cls in args.module:
            try:
                modules[name] = _load_submodel(model_cls, path)
            except Exception as e:
                print(f"[ERROR] Failed to load {name} via instantiate_model: {e}")
                print("Fallback to raw torch.load...")
                modules[name] = torch.load(path, map_location="cpu", weights_only=False)
            print(f"  {name}: {type(modules[name]).__name__}")
    else:
        print("[ERROR] No modules specified. Use --module NAME PATH MODEL")
        sys.exit(1)

    # ── Read Mermaid flowchart ────────────────────────────────────────
    mermaid = Path(args.mermaid).read_text(encoding="utf-8")
    print(f"[MERMAID] Loaded flowchart from {args.mermaid}")

    pipeline = Pipeline(
        mermaid_flowchart=mermaid,
        modules=modules,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print("[ASSEMBLE] Pipeline constructed.")

    # ── PARAMS-CALCULATOR validation ─────────────────────────────────
    print("[VALIDATE] Running params_calculator()...")
    result = pipeline.params_calculator()
    print(f"  consistency_status = {result['consistency_status']}")
    if result["consistency_status"] != "SUCCESS":
        print(f"[ERROR] PARAMS-CALCULATOR failed: {result.get('errors')}")
        print("        Fix Mermaid flowchart or sub-model output/input dimensions.")
        sys.exit(1)
    print(f"  total_params = {result.get('total_params', 'N/A'):,}")

    # ── Dummy propagation (shape trace) ──────────────────────────────
    shape = tuple(int(x) for x in args.input_shape.split(","))
    print(f"[DRY-RUN] dummy_propagate({shape})...")
    out_shape, trace = pipeline.dummy_propagate(shape)
    for line in trace:
        print(f"  {line}")
    print(f"[DRY-RUN] Output shape: {out_shape}")

    # ── Forward pass test ─────────────────────────────────────────────
    fwd = pipeline.forward_pass_test(shape, check_nan_inf=True)
    print(f"[TEST] inference_ready={fwd['inference_ready']}, "
          f"latency_ms={fwd.get('latency_ms', 0):.1f}ms")
    if not fwd["inference_ready"]:
        print("[WARN] Forward pass test failed. Check sub-model outputs for NaN/Inf.")

    # ── Save assembled pipeline checkpoint ───────────────────────────
    from ..train.utils.checkpoint import save_checkpoint
    try:
        manifest_extra = {"training_state": {"phase": "assembled"}}
    except Exception:
        manifest_extra = {}
        
    assembled_path = save_checkpoint(
        pipeline,
        save_dir=args.save_dir,
        filename="assembled.pt",
        manifest_extra=manifest_extra
    )
    print(f"[DONE] Assembled pipeline saved to {assembled_path}")
    print("[DONE] manifest.json & sentinel.runtime.env written.")
    print("       Run finetune_pipeline.py to correct offset errors.")


if __name__ == "__main__":
    main()

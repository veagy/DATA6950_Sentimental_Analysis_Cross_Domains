#!/usr/bin/env python3
"""Load one transformer MLP-head JSON and run a single forward pass (head + embeddings path)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from Code.thesis.common.pkg_bootstrap import install_lazy_code_models

install_lazy_code_models(_REPO)

from Code.thesis.common.model_factory import build_model_from_config_dict, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=_REPO
        / "Code/thesis/config/transformers/2_labels/B3_E_DL1_DistilBERT_mlp768_1024.json",
    )
    ap.add_argument("--n-classes", type=int, default=2)
    args = ap.parse_args()
    cfg = load_config(args.config)
    model, _ = build_model_from_config_dict(cfg, args.n_classes, "")
    model.eval()
    with sys.stdout:
        logits = model(["smoke test forward pass"], return_type="logits")
    print("ok logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()

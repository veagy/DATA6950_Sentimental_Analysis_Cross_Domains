"""
Meta-stacking: concatenate frozen expert logits and train a linear head.

Experts JSON format matches ``train_moe.py`` (list of config + checkpoint + modality).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from Code.thesis.common.pkg_bootstrap import install_lazy_code_models

install_lazy_code_models(_REPO)

from Code.thesis.common.checkpoint_io import load_safetensors_state, save_safetensors
from Code.thesis.common.datasets import ParquetFeaturesDataset, ParquetTextDataset
from Code.thesis.common.model_factory import build_model_from_config_dict, load_config
from Code.thesis.train.train_moe import DualDataset, dual_collate


def _expert_logits(
    experts: list[nn.Module],
    modalities: list[str],
    texts: list[str],
    feats: torch.Tensor,
    device: torch.device,
    n_classes: int,
) -> torch.Tensor:
    outs = []
    with torch.no_grad():
        for ex, mode in zip(experts, modalities):
            if mode == "dense":
                o = ex(feats)
            else:
                o = ex(list(texts), return_type="logits")  # type: ignore[operator]
            if isinstance(o, tuple):
                o = o[0]
            if o.dim() == 1:
                o = o.unsqueeze(-1)
            outs.append(o.to(device))
    return torch.cat(outs, dim=-1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Stack expert logits with a trainable linear meta-head.")
    ap.add_argument("--experts_json", type=Path, required=True)
    ap.add_argument("--dataset_stem", type=str, required=True)
    ap.add_argument("--data_root", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--out_path", type=Path, default=None)
    args = ap.parse_args()

    data_root = args.data_root or (_REPO / "data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.experts_json, encoding="utf-8") as f:
        spec = json.load(f)

    experts: list[nn.Module] = []
    modalities: list[str] = []
    n_classes: int | None = None
    for entry in spec:
        cfg_p = Path(entry["config"])
        if not cfg_p.is_absolute():
            cfg_p = _REPO / cfg_p
        cfg = load_config(cfg_p)
        stem = cfg_p.as_posix()
        if "/2_labels/" in stem:
            nc = 2
        elif "/3_labels/" in stem:
            nc = 3
        else:
            raise ValueError("Expert config must live under 2_labels or 3_labels")
        if n_classes is None:
            n_classes = nc
        elif n_classes != nc:
            raise ValueError("All experts must share label count")
        model, _ = build_model_from_config_dict(cfg, nc, "")
        ck = Path(entry["checkpoint"])
        if not ck.is_absolute():
            ck = _REPO / ck
        if ck.is_file():
            model.load_state_dict(load_safetensors_state(ck, map_location="cpu"), strict=False)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        experts.append(model.to(device))
        modalities.append(entry.get("modality", "text"))

    assert n_classes is not None
    ds = DualDataset(args.dataset_stem, data_root, args.max_samples)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=dual_collate)

    with torch.no_grad():
        texts0, f0, _ = ds[0]
        z0 = _expert_logits(experts, modalities, [texts0], f0.unsqueeze(0).to(device), device, n_classes)
    in_dim = z0.numel()

    meta = nn.Linear(in_dim, n_classes).to(device)
    opt = torch.optim.AdamW(meta.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    for _ in range(args.epochs):
        bar = tqdm(loader, desc="stack meta")
        for texts, feats, y in bar:
            feats_d = feats.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            z = _expert_logits(experts, modalities, texts, feats_d, device, n_classes)
            logits = meta(z)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            bar.set_postfix(loss=float(loss.item()))

    out = args.out_path or (
        _REPO / "checkpoints" / "stack" / f"{n_classes}-labels" / args.dataset_stem / "meta_head.safetensors"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    save_safetensors(meta.state_dict(), out)
    print("Saved meta head", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

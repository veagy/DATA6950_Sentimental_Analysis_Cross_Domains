"""Evaluate checkpoints produced by train_single (safetensors)."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from Code.thesis.common.pkg_bootstrap import install_lazy_code_models

install_lazy_code_models(_REPO)

from Code.thesis.common.checkpoint_io import load_safetensors_state
from Code.thesis.common.datasets import (
    ParquetFeaturesDataset,
    ParquetTextDataset,
    features_collate,
    text_collate,
)
from Code.thesis.common.model_factory import (
    build_model_from_config_dict,
    is_text_model_config_path,
    load_config,
)
from Code.thesis.common.torch_util import (
    sync_factory_kwargs_device,
    sync_ml_fitted_buffers_to_attrs,
)
from Code.thesis.common.wrappers import RNNClassifier
from Code.models.deep_learning.llm.llm_models import LLMModule
from Code.models.deep_learning.hrm.hrm_model import (
    HierarchicalReasoningModel,
    HRMClassifierWrapper,
)
from Code.models.utils.utils import MLModule


_ML_JOBLIB_CHECKPOINT_CLASSES = frozenset({"DecisionTreeClassifier", "RandomForestClassifier"})


def _n_classes_from_path(config_path: Path) -> int:
    p = str(config_path).replace("\\", "/")
    if "/2_labels/" in p:
        return 2
    if "/3_labels/" in p:
        return 3
    raise ValueError(f"Cannot infer n_classes from {config_path}")


def _parquet(data_root: Path, stem: str, text_mode: bool) -> Path:
    sub = "processed" if text_mode else "transformed"
    return data_root / sub / f"{stem}.parquet"


def _ckpt_path(ckpt_root: Path, n_classes: int, stem: str, cfg_stem: str) -> Path:
    return ckpt_root / f"{n_classes}-labels" / stem / f"{cfg_stem}.safetensors"


def _unwrap(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, nn.parallel.DistributedDataParallel) else m


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--dataset_stem", type=str, required=True)
    ap.add_argument("--data_root", type=Path, default=None)
    ap.add_argument("--checkpoint_root", type=Path, default=None)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out_csv", type=Path, default=None)
    args = ap.parse_args()

    data_root = args.data_root or (_REPO / "data")
    ckpt_root = args.checkpoint_root or (_REPO / "checkpoints")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg_path = args.config.resolve()
    n_classes = _n_classes_from_path(cfg_path)
    text_mode = is_text_model_config_path(cfg_path)
    pq = _parquet(data_root, args.dataset_stem, text_mode)
    ckpt = _ckpt_path(ckpt_root, n_classes, args.dataset_stem, cfg_path.stem)

    cfg = load_config(cfg_path)
    class_name = next(iter(cfg))
    ckpt_joblib = ckpt.with_suffix(".joblib")

    if class_name in _ML_JOBLIB_CHECKPOINT_CLASSES and ckpt_joblib.is_file():
        import joblib

        model = joblib.load(ckpt_joblib)
    else:
        model, class_name = build_model_from_config_dict(cfg, n_classes, "")
        if ckpt.is_file():
            state = load_safetensors_state(ckpt, map_location=device)
            if isinstance(model, MLModule) and hasattr(model, "_init_module_"):
                xs0 = torch.randn(2, 100, device=device)
                y0 = torch.tensor([0, 1], device=device)
                model._init_module_(xs0, y0)
            if isinstance(model, MLModule):
                # Estimators like LinearSVC only register _fit_* buffers after fit; pre-create so load_state_dict applies.
                for k, v in list(state.items()):
                    if k.startswith("_fit_") and isinstance(v, torch.Tensor) and k not in model._buffers:
                        model.register_buffer(k, torch.empty_like(v))
            model.load_state_dict(state, strict=False)
    model = model.to(device)
    if isinstance(model, MLModule):
        sync_ml_fitted_buffers_to_attrs(model)
    else:
        sync_factory_kwargs_device(model, device)
    model.eval()

    if text_mode:
        ds = ParquetTextDataset(str(pq), max_samples=args.max_samples)
        collate = text_collate
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    else:
        ds = ParquetFeaturesDataset(str(pq), max_samples=args.max_samples, n_classes=n_classes)
        collate = features_collate
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    ys: list[int] = []
    preds: list[int] = []

    raw = _unwrap(model)
    if isinstance(raw, MLModule):
        if text_mode:
            raise SystemExit("ML on text not supported")
        xs = torch.stack([ds[i][0] for i in range(len(ds))])
        y_true = np.array([ds[i][1] for i in range(len(ds))])
        pr = raw.predict(xs)
        if hasattr(pr, "detach"):
            pr = pr.detach().cpu().numpy()
        pr = np.asarray(pr).reshape(-1)
        acc = float((pr == y_true).mean())
        print(f"accuracy={acc:.4f} n={len(ds)}")
        return

    for batch in tqdm(loader, desc="eval"):
        if text_mode:
            texts, y = batch
            y = y.to(device)
            if isinstance(raw, LLMModule):
                logits = raw(list(texts), return_type="logits")
            elif isinstance(raw, HRMClassifierWrapper):
                logits = raw(list(texts), pretrain=False)
            elif isinstance(raw, HierarchicalReasoningModel):
                raise SystemExit(
                    "HRM evaluation expects a config built via model_factory (HRMClassifierWrapper)."
                )
            else:
                logits = raw(list(texts))
            pred = logits.argmax(dim=-1)
        else:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            cn = class_name.lower()
            if "cnn" in cn and "lstm" not in cn:
                x = x.unsqueeze(-1).expand(-1, -1, 3)
            elif isinstance(raw, RNNClassifier) or "lstm" in cn or "gru" in cn or "rnn" in cn:
                x = x.unsqueeze(1)
            logits = raw(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            pred = logits.argmax(dim=-1)
        ys.extend(y.cpu().tolist())
        preds.extend(pred.cpu().tolist())

    acc = sum(int(a == b) for a, b in zip(ys, preds)) / max(len(ys), 1)
    print(f"accuracy={acc:.4f} n={len(ys)}")
    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["y_true", "y_pred"])
            w.writerows(zip(ys, preds))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Split merged all-data parquets by ``source_stem``, then evaluate each checkpointed model
separately on split files only (under processed/transformed/{split_subdir}/).

See TEMP/docs/ml/TRAINING_PIPELINES.md for data conventions.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    jaccard_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

_ML_JOBLIB_CHECKPOINT_CLASSES = frozenset({"DecisionTreeClassifier", "RandomForestClassifier"})

# Default glob paths to skip when searching for run_meta.txt (bare backbones / pretrain).
_RUN_META_SKIP_PARTS = (
    "/pretrain/",
    "/deep_learning/llm/",
)


def filesystem_slug(name: str) -> str:
    s = str(name).strip()
    s = re.sub(r"[^\w\-.]+", "_", s)
    s = s.strip("_") or "unknown"
    return s[:200]


def split_merged_parquets(
    data_root: Path,
    *,
    split_subdir: str = "by_source_stem",
    processed_file: str = "all-data.parquet",
    transformed_file: str = "all-data.parquet",
    max_rows: Optional[int] = None,
) -> dict[str, str]:
    """
    Write per-source_stem shards; return mapping safe_stem -> original source_stem.
    """
    proc_in = data_root / "processed" / processed_file
    trans_in = data_root / "transformed" / transformed_file
    if not proc_in.is_file():
        raise FileNotFoundError(proc_in)
    if not trans_in.is_file():
        raise FileNotFoundError(trans_in)

    df_p = pd.read_parquet(proc_in)
    df_t = pd.read_parquet(trans_in)
    if max_rows is not None and max_rows > 0:
        df_p = df_p.iloc[:max_rows].copy()
        df_t = df_t.iloc[:max_rows].copy()
        print(f"[split] truncated to first {max_rows} rows (debug)", flush=True)
    if "source_stem" not in df_p.columns:
        raise ValueError(f"{proc_in} missing column source_stem")
    if "source_stem" not in df_t.columns:
        raise ValueError(f"{trans_in} missing column source_stem")

    out_p = data_root / "processed" / split_subdir
    out_t = data_root / "transformed" / split_subdir
    out_p.mkdir(parents=True, exist_ok=True)
    out_t.mkdir(parents=True, exist_ok=True)

    stem_map: dict[str, str] = {}
    used_slug: dict[str, str] = {}

    for orig in sorted(df_p["source_stem"].dropna().unique(), key=str):
        o = str(orig).strip()
        slug = filesystem_slug(o)
        if slug in used_slug and used_slug[slug] != o:
            raise ValueError(f"Slug collision: {o!r} and {used_slug[slug]!r} -> {slug!r}")
        used_slug[slug] = o
        stem_map[slug] = o

        sub_p = df_p[df_p["source_stem"] == orig]
        sub_t = df_t[df_t["source_stem"] == orig]
        p_out = out_p / f"{slug}.parquet"
        t_out = out_t / f"{slug}.parquet"
        sub_p.to_parquet(p_out, index=False)
        sub_t.to_parquet(t_out, index=False)
        np_p, np_t = len(sub_p), len(sub_t)
        if np_p != np_t:
            print(
                f"[split] warning stem={o!r} slug={slug}: processed rows={np_p} "
                f"!= transformed rows={np_t} (MoE dual will use min length)",
                flush=True,
            )

    map_path = out_p / "_stem_map.json"
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump({"safe_to_original": stem_map, "split_subdir": split_subdir}, f, indent=2)
    print(f"[split] wrote {len(stem_map)} stems under {out_p} and {out_t}", flush=True)
    return stem_map


def load_stem_map(data_root: Path, split_subdir: str) -> dict[str, str]:
    map_path = data_root / "processed" / split_subdir / "_stem_map.json"
    if map_path.is_file():
        with open(map_path, encoding="utf-8") as f:
            d = json.load(f)
        return dict(d["safe_to_original"])
    # Infer from existing parquet names only (original = slug).
    out_p = data_root / "processed" / split_subdir
    if not out_p.is_dir():
        return {}
    m = {}
    for p in sorted(out_p.glob("*.parquet")):
        if p.name.startswith("_"):
            continue
        slug = p.stem
        m[slug] = slug
    return m


def _n_classes_from_config_path(config_path: Path) -> int:
    p = str(config_path).replace("\\", "/")
    if "/2_labels/" in p:
        return 2
    if "/3_labels/" in p:
        return 3
    raise ValueError(f"Cannot infer n_classes from {config_path}")


def _unwrap(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, nn.parallel.DistributedDataParallel) else m


def parse_run_meta(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if "=" not in line or line.startswith("#"):
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
    return out


@dataclass
class EvalTarget:
    """One eval run: config + resolved weights (+ optional joblib)."""

    model_id: str
    config_path: Path
    n_classes: int
    weight_path: Path
    joblib_path: Optional[Path]
    source: str  # config_scan | run_meta


def _should_skip_run_meta_path(p: Path) -> bool:
    s = str(p).replace("\\", "/")
    return any(part in s for part in _RUN_META_SKIP_PARTS)


def discover_run_meta_targets(checkpoint_root: Path) -> list[EvalTarget]:
    targets: list[EvalTarget] = []
    for meta in checkpoint_root.rglob("run_meta.txt"):
        if _should_skip_run_meta_path(meta):
            continue
        kv = parse_run_meta(meta)
        cfg_s = kv.get("config")
        nc_s = kv.get("n_classes")
        if not cfg_s or not nc_s:
            continue
        cfg_path = Path(cfg_s)
        if not cfg_path.is_file():
            cfg_path = _REPO / cfg_s
        if not cfg_path.is_file():
            continue
        n_classes = int(nc_s)
        meta_dir = meta.parent
        rel = meta_dir.relative_to(checkpoint_root)
        for w in sorted(meta_dir.glob("*.safetensors")):
            mid = f"run_meta__{rel.as_posix().replace('/', '_')}__{w.stem}"
            targets.append(
                EvalTarget(
                    model_id=mid,
                    config_path=cfg_path.resolve(),
                    n_classes=n_classes,
                    weight_path=w.resolve(),
                    joblib_path=None,
                    source="run_meta",
                )
            )
    return targets


def discover_config_scan_targets(config_root: Path) -> list[tuple[Path, int, str, str]]:
    """Returns (config_path, n_classes, cfg_stem, model_id_base) for standard layout.

    ``model_id_base`` is unique per JSON (relative path under config_root) so two configs
    with the same filename (e.g. FeatEnc_FFN.json) in different folders do not collide.
    """
    rows: list[tuple[Path, int, str, str]] = []
    config_root = config_root.resolve()
    for json_path in config_root.rglob("*.json"):
        p = json_path.as_posix()
        if "/moe/" in p.replace("\\", "/") and json_path.name.startswith("experts"):
            continue
        # Stacked MLP fine-tune uses same FeatEnc_* checkpoint names; use run_meta entries instead.
        if "mlp_gelu_head_ddp" in p.replace("\\", "/"):
            continue
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(raw, dict) or not raw:
            continue
        class_name = next(iter(raw))
        if class_name == "FeaturePretrainAutoencoder":
            continue
        try:
            nc = _n_classes_from_config_path(json_path)
        except ValueError:
            continue
        rel = json_path.resolve().relative_to(config_root)
        id_base = rel.with_suffix("").as_posix().replace("/", "_")
        rows.append((json_path.resolve(), nc, json_path.stem, id_base))
    return rows


def resolve_standard_weights(
    checkpoint_root: Path,
    n_classes: int,
    original_stem: str,
    cfg_stem: str,
) -> tuple[Optional[Path], Optional[Path]]:
    """Prefer K-labels/{stem}/ then K-labels/all-data/. Returns (safetensors, joblib)."""
    kdir = f"{n_classes}-labels"
    p_stem = checkpoint_root / kdir / original_stem / f"{cfg_stem}.safetensors"
    j_stem = checkpoint_root / kdir / original_stem / f"{cfg_stem}.joblib"
    p_all = checkpoint_root / kdir / "all-data" / f"{cfg_stem}.safetensors"
    j_all = checkpoint_root / kdir / "all-data" / f"{cfg_stem}.joblib"
    if p_stem.is_file():
        w, js = p_stem, j_stem if j_stem.is_file() else None
        return w, js
    if p_all.is_file():
        w, ja = p_all, j_all if j_all.is_file() else None
        return w, ja
    return None, None


def build_eval_targets_for_stem(
    checkpoint_root: Path,
    config_root: Path,
    original_stem: str,
    run_meta_targets: list[EvalTarget],
    config_rows: list[tuple[Path, int, str, str]],
) -> list[EvalTarget]:
    out: list[EvalTarget] = []
    seen: set[tuple[str, str]] = set()

    for cfg_path, nc, cfg_stem, id_base in config_rows:
        w, jb = resolve_standard_weights(checkpoint_root, nc, original_stem, cfg_stem)
        if w is None:
            continue
        kdir = f"{nc}-labels"
        stem_ckpt = (checkpoint_root / kdir / original_stem / f"{cfg_stem}.safetensors").resolve()
        suffix = "__stem_ckpt" if w.resolve() == stem_ckpt else "__all_data_ckpt"
        mid = f"{id_base}{suffix}"
        key = (mid, str(w))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            EvalTarget(
                model_id=mid,
                config_path=cfg_path,
                n_classes=nc,
                weight_path=w,
                joblib_path=jb,
                source="config_scan",
            )
        )

    for t in run_meta_targets:
        key = (t.model_id, str(t.weight_path))
        if key in seen:
            continue
        seen.add(key)
        out.append(t)

    return out


def load_classifier_model(
    target: EvalTarget,
    device: torch.device,
) -> tuple[nn.Module | Any, str]:
    cfg = load_config(target.config_path)
    class_name = next(iter(cfg))
    if class_name in _ML_JOBLIB_CHECKPOINT_CLASSES and target.joblib_path and target.joblib_path.is_file():
        import joblib

        return joblib.load(target.joblib_path), class_name

    model, class_name = build_model_from_config_dict(cfg, target.n_classes, "")
    if target.weight_path.is_file():
        state = load_safetensors_state(target.weight_path, map_location=device)
        if isinstance(model, MLModule) and hasattr(model, "_init_module_"):
            xs0 = torch.randn(2, 100, device=device)
            y0 = torch.tensor([0, 1], device=device)
            model._init_module_(xs0, y0)
        if isinstance(model, MLModule):
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
    return model, class_name


class DualSplitDataset(Dataset):
    """Aligned processed + transformed rows from split parquets."""

    def __init__(
        self,
        data_root: Path,
        split_subdir: str,
        safe_stem: str,
        max_samples: Optional[int],
        n_classes: int,
    ):
        p_txt = data_root / "processed" / split_subdir / f"{safe_stem}.parquet"
        p_feat = data_root / "transformed" / split_subdir / f"{safe_stem}.parquet"
        self.text_ds = ParquetTextDataset(
            str(p_txt),
            max_samples=max_samples,
            n_classes=n_classes,
        )
        self.feat_ds = ParquetFeaturesDataset(
            str(p_feat),
            max_samples=max_samples,
            n_classes=n_classes,
        )
        self.n = min(len(self.text_ds), len(self.feat_ds))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        t, y = self.text_ds[i]
        x, _ = self.feat_ds[i]
        return t, x, y


def dual_split_collate(batch):
    texts = [b[0] for b in batch]
    feats = torch.stack([b[1] for b in batch])
    y = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return texts, feats, y


def _forward_batch_logits(
    raw: nn.Module,
    class_name: str,
    text_mode: bool,
    batch,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (logits [B,K], y [B])."""
    if text_mode:
        texts, y = batch
        y = y.to(device)
        if isinstance(raw, LLMModule):
            logits = raw(list(texts), return_type="logits")
        elif isinstance(raw, HRMClassifierWrapper):
            logits = raw(list(texts), pretrain=False)
        elif isinstance(raw, HierarchicalReasoningModel):
            raise RuntimeError("HRMClassifierWrapper expected for HRM eval")
        else:
            logits = raw(list(texts))
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
    return logits, y


@torch.no_grad()
def collect_predictions_torch(
    model: nn.Module,
    class_name: str,
    text_mode: bool,
    loader: DataLoader,
    device: torch.device,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = _unwrap(model)
    ys: list[int] = []
    preds: list[int] = []
    probas: list[np.ndarray] = []

    for batch in tqdm(loader, desc="infer", leave=False):
        logits, y = _forward_batch_logits(raw, class_name, text_mode, batch, device)
        pr = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        pred = logits.argmax(dim=-1)
        ys.extend(y.cpu().tolist())
        preds.extend(pred.cpu().tolist())
        probas.append(pr)
    y_true = np.array(ys, dtype=np.int64)
    y_pred = np.array(preds, dtype=np.int64)
    y_proba = np.concatenate(probas, axis=0) if probas else np.zeros((0, n_classes), dtype=np.float64)
    return y_true, y_pred, y_proba


def collect_predictions_ml_module(
    model: MLModule,
    ds: ParquetFeaturesDataset,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = torch.stack([ds[i][0] for i in range(len(ds))])
    y_true = np.array([ds[i][1] for i in range(len(ds))], dtype=np.int64)
    pr = model.predict(xs)
    if hasattr(pr, "detach"):
        pr = pr.detach().cpu().numpy()
    y_pred = np.asarray(pr).reshape(-1).astype(np.int64)
    try:
        pba = model.predict_proba(xs)
        if hasattr(pba, "detach"):
            pba = pba.detach().cpu().numpy()
        y_proba = np.asarray(pba, dtype=np.float64)
        if y_proba.ndim == 1:
            y_proba = np.column_stack([1.0 - y_proba, y_proba])
    except Exception:
        y_proba = np.zeros((len(y_true), n_classes), dtype=np.float64)
        y_proba[np.arange(len(y_pred)), np.clip(y_pred, 0, n_classes - 1)] = 1.0
    return y_true, y_pred, y_proba


def compute_metrics_bundle(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    n_classes: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {"n_samples": int(len(y_true))}
    if len(y_true) == 0:
        out["error"] = "no_samples"
        return out

    labels_present = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    out["labels_present"] = labels_present

    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    out["hamming_loss"] = float(hamming_loss(y_true, y_pred))
    out["matthews_corrcoef"] = float(matthews_corrcoef(y_true, y_pred))
    out["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))

    for avg in ("macro", "micro", "weighted"):
        out[f"precision_{avg}"] = float(
            precision_score(y_true, y_pred, average=avg, zero_division=0, labels=labels_present)
        )
        out[f"recall_{avg}"] = float(
            recall_score(y_true, y_pred, average=avg, zero_division=0, labels=labels_present)
        )
        out[f"f1_{avg}"] = float(
            f1_score(y_true, y_pred, average=avg, zero_division=0, labels=labels_present)
        )
        out[f"jaccard_{avg}"] = float(
            jaccard_score(y_true, y_pred, average=avg, zero_division=0, labels=labels_present)
        )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    out["confusion_matrix"] = cm.tolist()

    auc_reason = None
    auc_val = None
    try:
        if y_proba.shape[0] != len(y_true) or y_proba.shape[1] != n_classes:
            auc_reason = "proba_shape_mismatch"
        elif len(set(y_true.tolist())) < n_classes:
            auc_reason = "missing_class_in_y_true"
        else:
            auc_val = float(
                roc_auc_score(
                    y_true,
                    y_proba,
                    multi_class="ovr",
                    average="macro",
                    labels=list(range(n_classes)),
                )
            )
    except Exception as e:
        auc_reason = str(e)
    out["roc_auc_ovr_macro"] = auc_val
    out["roc_auc_skip_reason"] = auc_reason

    return out


def load_moe_experts(
    experts_json: Path,
    device: torch.device,
) -> tuple[nn.ModuleList, list[str], int]:
    spec = json.loads(experts_json.read_text(encoding="utf-8"))
    experts: list[nn.Module] = []
    modalities: list[str] = []
    n_classes: Optional[int] = None
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
            raise ValueError(f"Expert config must be under 2_labels or 3_labels: {cfg_p}")
        if n_classes is None:
            n_classes = nc
        elif n_classes != nc:
            raise ValueError("Expert label counts must match")
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
    return nn.ModuleList(experts), modalities, n_classes


@torch.no_grad()
def moe_collect_predictions(
    experts: nn.ModuleList,
    modalities: list[str],
    n_classes: int,
    gate_state: Optional[dict[str, torch.Tensor]],
    loader: DataLoader,
    device: torch.device,
    feat_dim: int,
    sparse_top_k: Optional[int],
    gate_hidden_dim: Optional[int],
    use_text_gate: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from Code.thesis.train.moe_facade import FeatureGatedMoE, HeterogeneousMoE

    if use_text_gate:
        moe = HeterogeneousMoE(
            n_classes=n_classes,
            expert_modules=experts,
            sparse_top_k=sparse_top_k,
            gate_hidden_dim=gate_hidden_dim or 256,
        ).to(device)
    else:
        moe = FeatureGatedMoE(
            n_classes=n_classes,
            expert_modules=experts,
            feat_dim=feat_dim,
            sparse_top_k=sparse_top_k,
            gate_hidden_dim=gate_hidden_dim,
        ).to(device)
    if gate_state is not None:
        moe.gate.load_state_dict(gate_state, strict=False)
    moe.eval()
    raw = _unwrap(moe)

    ys: list[int] = []
    preds: list[int] = []
    probas: list[np.ndarray] = []

    for batch in tqdm(loader, desc="moe_infer", leave=False):
        texts, feats, y = batch
        feats = feats.to(device)
        y = y.to(device)
        if gate_state is None:
            outs = []
            for ex, mode in zip(raw.experts, modalities):
                if mode == "dense":
                    o = ex(feats)
                else:
                    o = ex(list(texts), return_type="logits")
                if o.dim() == 1:
                    o = o.unsqueeze(-1)
                outs.append(o)
            stacked = torch.stack(outs, dim=1)
            logits = stacked.mean(dim=1)
        else:
            logits = moe(list(texts), feats, expert_modalities=modalities)
        pr = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        pred = logits.argmax(dim=-1)
        ys.extend(y.cpu().tolist())
        preds.extend(pred.cpu().tolist())
        probas.append(pr)

    y_true = np.array(ys, dtype=np.int64)
    y_pred = np.array(preds, dtype=np.int64)
    y_proba = np.concatenate(probas, axis=0) if probas else np.zeros((0, n_classes), dtype=np.float64)
    return y_true, y_pred, y_proba


def eval_one_target(
    target: EvalTarget,
    *,
    data_root: Path,
    split_subdir: str,
    safe_stem: str,
    device: torch.device,
    batch_size: int,
    max_samples: Optional[int],
    log_path: bool,
) -> dict[str, Any]:
    cfg_path = target.config_path
    n_classes = target.n_classes
    text_mode = is_text_model_config_path(cfg_path)
    proc_p = data_root / "processed" / split_subdir / f"{safe_stem}.parquet"
    trans_p = data_root / "transformed" / split_subdir / f"{safe_stem}.parquet"
    if log_path:
        print(f"[eval] parquet text={proc_p} features={trans_p}", flush=True)

    meta: dict[str, Any] = {
        "model_id": target.model_id,
        "source": target.source,
        "config_path": str(target.config_path),
        "weight_path": str(target.weight_path),
        "n_classes": n_classes,
        "text_mode": text_mode,
        "safe_stem": safe_stem,
    }

    try:
        model, class_name = load_classifier_model(target, device)
    except Exception as e:
        meta["error"] = f"load_model: {e}"
        return meta

    meta["class_name"] = class_name

    if isinstance(_unwrap(model), MLModule):
        if text_mode:
            meta["error"] = "MLModule on text not supported"
            return meta
        ds = ParquetFeaturesDataset(
            str(trans_p),
            max_samples=max_samples,
            n_classes=n_classes,
        )
        if len(ds) < 1:
            meta["error"] = "empty_dataset"
            return meta
        y_true, y_pred, y_proba = collect_predictions_ml_module(model, ds, n_classes)
    else:
        if text_mode:
            ds = ParquetTextDataset(
                str(proc_p),
                max_samples=max_samples,
                n_classes=n_classes,
            )
            collate = text_collate
        else:
            ds = ParquetFeaturesDataset(
                str(trans_p),
                max_samples=max_samples,
                n_classes=n_classes,
            )
            collate = features_collate
        if len(ds) < 1:
            meta["error"] = "empty_dataset"
            return meta
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate,
        )
        y_true, y_pred, y_proba = collect_predictions_torch(
            model, class_name, text_mode, loader, device, n_classes
        )

    meta["metrics"] = compute_metrics_bundle(y_true, y_pred, y_proba, n_classes)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return meta


def eval_moe_manifest(
    experts_json: Path,
    *,
    variant_id: str,
    data_root: Path,
    split_subdir: str,
    safe_stem: str,
    device: torch.device,
    batch_size: int,
    max_samples: Optional[int],
    gate_safetensors: Optional[Path],
    use_text_gate: bool,
    sparse_top_k: Optional[int],
    gate_hidden_dim: Optional[int],
    log_path: bool,
) -> dict[str, Any]:
    proc_p = data_root / "processed" / split_subdir / f"{safe_stem}.parquet"
    trans_p = data_root / "transformed" / split_subdir / f"{safe_stem}.parquet"
    if log_path:
        print(f"[moe] {variant_id} parquet text={proc_p} features={trans_p}", flush=True)

    meta: dict[str, Any] = {
        "model_id": variant_id,
        "source": "moe_manifest",
        "experts_json": str(experts_json),
        "safe_stem": safe_stem,
        "gate_path": str(gate_safetensors) if gate_safetensors else None,
    }
    try:
        experts, modalities, n_classes = load_moe_experts(experts_json, device)
    except Exception as e:
        meta["error"] = f"load_experts: {e}"
        return meta

    ds = DualSplitDataset(
        data_root,
        split_subdir,
        safe_stem,
        max_samples,
        n_classes,
    )
    if len(ds) < 1:
        meta["error"] = "empty_dual_dataset"
        return meta
    feat_dim = int(ds[0][1].shape[0])
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dual_split_collate,
    )

    gate_state = None
    if gate_safetensors is not None and gate_safetensors.is_file():
        gate_state = load_safetensors_state(gate_safetensors, map_location=device)

    try:
        y_true, y_pred, y_proba = moe_collect_predictions(
            experts,
            modalities,
            n_classes,
            gate_state,
            loader,
            device,
            feat_dim,
            sparse_top_k,
            gate_hidden_dim,
            use_text_gate,
        )
    except Exception as e:
        meta["error"] = f"moe_forward: {e}"
        return meta

    meta["n_classes"] = n_classes
    meta["metrics"] = compute_metrics_bundle(y_true, y_pred, y_proba, n_classes)
    del experts
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return meta


def moe_manifest_label_count(mj: Path) -> Optional[int]:
    try:
        raw = json.loads(mj.read_text(encoding="utf-8"))
    except Exception:
        return None
    nc0: Optional[int] = None
    for e in raw:
        cp = Path(e["config"])
        if not cp.is_absolute():
            cp = _REPO / cp
        s = cp.as_posix()
        nci = 2 if "/2_labels/" in s else (3 if "/3_labels/" in s else None)
        if nci is None:
            return None
        if nc0 is None:
            nc0 = nci
        elif nc0 != nci:
            return None
    return nc0


def write_metrics_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# Column order for 2label_metrics_table.csv / 3label_metrics_table.csv
_METRICS_TABLE_SCALAR_KEYS: tuple[str, ...] = (
    "n_samples",
    "accuracy",
    "balanced_accuracy",
    "hamming_loss",
    "matthews_corrcoef",
    "cohen_kappa",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "jaccard_macro",
    "precision_micro",
    "recall_micro",
    "f1_micro",
    "jaccard_micro",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "jaccard_weighted",
    "roc_auc_ovr_macro",
    "roc_auc_skip_reason",
)


def _metrics_path_to_safe_stem(metrics_json: Path, label_mode: int) -> str:
    """Infer safe_stem from .../{K}label/{safe_stem}/.../metrics.json."""
    parts = metrics_json.resolve().parts
    tag = f"{label_mode}label"
    try:
        i = parts.index(tag)
        if i + 1 < len(parts) and parts[i + 1] != "metrics.json":
            return parts[i + 1]
    except ValueError:
        pass
    return ""


def payload_to_metrics_table_row(
    payload: dict[str, Any],
    metrics_json_path: Path,
    label_mode: int,
) -> dict[str, Any]:
    """One flat row: dataset + model identity + all scalar metrics + JSON blobs for CM / labels."""
    m = payload.get("metrics")
    if not isinstance(m, dict):
        m = {}

    safe = str(payload.get("safe_stem") or _metrics_path_to_safe_stem(metrics_json_path, label_mode))
    orig = str(payload.get("original_stem") or safe)

    wp = payload.get("weight_path")
    if wp is None and payload.get("gate_path") is not None:
        wp = payload.get("gate_path")
    wp_s = str(wp) if wp is not None else ""

    row: dict[str, Any] = {
        "label_mode": label_mode,
        "safe_stem": safe,
        "original_stem": orig,
        "dataset": orig,
        "model_id": str(payload.get("model_id", "")),
        "source": str(payload.get("source", "")),
        "class_name": str(payload.get("class_name", "")),
        "error": str(payload.get("error", "")),
        "config_path": str(payload.get("config_path", "")),
        "weight_path": wp_s,
        "experts_json": str(payload.get("experts_json", "")),
        "metrics_json_path": str(metrics_json_path.resolve()),
    }

    for k in _METRICS_TABLE_SCALAR_KEYS:
        v = m.get(k, "")
        if v is None:
            row[k] = ""
        else:
            row[k] = v

    lp = m.get("labels_present")
    row["labels_present_json"] = json.dumps(lp) if lp is not None else ""

    cm = m.get("confusion_matrix")
    row["confusion_matrix_json"] = json.dumps(cm) if cm is not None else ""

    return row


def metrics_table_fieldnames() -> list[str]:
    return [
        "label_mode",
        "safe_stem",
        "original_stem",
        "dataset",
        "model_id",
        "source",
        "class_name",
        "error",
        "config_path",
        "weight_path",
        "experts_json",
        "metrics_json_path",
        *_METRICS_TABLE_SCALAR_KEYS,
        "labels_present_json",
        "confusion_matrix_json",
    ]


def export_metrics_tables_from_json(out_dir: Path) -> None:
    """
    Scan output/metrics/{2,3}label/**/metrics.json and write
    2label_metrics_table.csv and 3label_metrics_table.csv (one row per model × dataset).
    """
    out_dir = out_dir.resolve()
    fnames = metrics_table_fieldnames()
    for k in (2, 3):
        sub = out_dir / f"{k}label"
        rows: list[dict[str, Any]] = []
        if sub.is_dir():
            for p in sorted(sub.rglob("metrics.json")):
                try:
                    payload = json.loads(p.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                nc = int(payload.get("n_classes", k))
                if nc != k:
                    continue
                rows.append(payload_to_metrics_table_row(payload, p, k))
        out_csv = out_dir / f"{k}label_metrics_table.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"[csv] wrote {len(rows)} rows -> {out_csv}", flush=True)


def append_summary_row(summary_csv: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    new_file = not summary_csv.is_file()
    with open(summary_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Split all-data by source_stem, evaluate each checkpoint on split parquets only."
    )
    ap.add_argument("--repo-root", type=Path, default=None, help="TEMP root (default: infer from script)")
    ap.add_argument("--data-root", type=Path, default=None, help="Default: <repo>/data")
    ap.add_argument("--checkpoint-root", type=Path, default=None, help="Default: <repo>/checkpoints")
    ap.add_argument("--config-root", type=Path, default=None, help="Default: <repo>/Code/thesis/config")
    ap.add_argument("--output-dir", type=Path, default=None, help="Default: <repo>/output/metrics")
    ap.add_argument("--split-subdir", type=str, default="by_source_stem")
    ap.add_argument(
        "--split-max-rows",
        type=int,
        default=None,
        help="Optional cap on rows read from merged all-data (debug/smoke; omit for full split)",
    )
    ap.add_argument("--skip-split", action="store_true", help="Reuse existing split under split-subdir")
    ap.add_argument(
        "--no-run-meta",
        action="store_true",
        help="Do not add run_meta.txt-derived checkpoint targets",
    )
    ap.add_argument(
        "--moe-manifests",
        type=str,
        default="",
        help="Comma-separated experts JSON paths (repo-relative or absolute)",
    )
    ap.add_argument(
        "--moe-gate",
        type=Path,
        default=None,
        help="Optional gate .safetensors for MoE (per manifest run if multiple)",
    )
    ap.add_argument("--moe-text-gate", action="store_true", help="Use HeterogeneousMoE (DistilBERT gate)")
    ap.add_argument("--moe-sparse-top-k", type=int, default=2)
    ap.add_argument("--moe-gate-hidden", type=int, default=256)
    ap.add_argument("--only-stems", type=str, default="", help="Comma-separated safe_stem or original stems")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--log-parquet-paths", action="store_true")
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip writing if metrics.json already exists for that stem/model",
    )
    ap.add_argument(
        "--export-metrics-csv-only",
        action="store_true",
        help="Rebuild 2label_metrics_table.csv and 3label_metrics_table.csv from metrics.json only (no split/eval)",
    )
    ap.add_argument(
        "--no-export-metrics-csv",
        action="store_true",
        help="Do not rebuild label CSV tables at end of a normal eval run",
    )
    args = ap.parse_args()

    global _REPO
    if args.repo_root is not None:
        _REPO = args.repo_root.resolve()
        sys.path.insert(0, str(_REPO))

    data_root = args.data_root or (_REPO / "data")
    ckpt_root = args.checkpoint_root or (_REPO / "checkpoints")
    cfg_root = args.config_root or (_REPO / "Code" / "thesis" / "config")
    out_dir = args.output_dir or (_REPO / "output" / "metrics")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.export_metrics_csv_only:
        export_metrics_tables_from_json(out_dir)
        print(f"[done] metrics CSV tables under {out_dir}", flush=True)
        return

    if not args.skip_split:
        split_merged_parquets(
            data_root,
            split_subdir=args.split_subdir,
            max_rows=args.split_max_rows,
        )
    stem_map = load_stem_map(data_root, args.split_subdir)
    if not stem_map:
        raise SystemExit("No stems found; run split first or check split-subdir")

    only = {s.strip() for s in args.only_stems.split(",") if s.strip()}
    if only:
        filtered: dict[str, str] = {}
        for safe, orig in stem_map.items():
            if safe in only or orig in only:
                filtered[safe] = orig
        stem_map = filtered
        if not stem_map:
            raise SystemExit("--only-stems matched nothing")

    config_rows = discover_config_scan_targets(cfg_root)
    run_meta_targets: list[EvalTarget] = []
    if not args.no_run_meta:
        run_meta_targets = discover_run_meta_targets(ckpt_root)

    summary_csv = out_dir / "summary.csv"
    summary_fields = [
        "safe_stem",
        "original_stem",
        "n_classes",
        "model_id",
        "source",
        "accuracy",
        "f1_macro",
        "balanced_accuracy",
        "roc_auc_ovr_macro",
        "error",
        "config_path",
        "weight_path",
        "metrics_path",
    ]

    moe_paths = [Path(p.strip()) for p in args.moe_manifests.split(",") if p.strip()]
    resolved_moe = []
    for mp in moe_paths:
        p = mp if mp.is_absolute() else _REPO / mp
        if p.is_file():
            resolved_moe.append(p)
    moe_jobs = [(mj, moe_manifest_label_count(mj)) for mj in resolved_moe]

    for safe_stem, original_stem in sorted(stem_map.items(), key=lambda x: x[0]):
        if safe_stem.startswith("_"):
            continue
        targets = build_eval_targets_for_stem(
            ckpt_root, cfg_root, original_stem, run_meta_targets, config_rows
        )
        by_k: dict[int, list[EvalTarget]] = {}
        for t in targets:
            by_k.setdefault(t.n_classes, []).append(t)

        for n_classes, tlist in by_k.items():
            kdir = out_dir / f"{n_classes}label" / safe_stem
            for target in tlist:
                if target.n_classes != n_classes:
                    continue
                out_json = kdir / target.model_id / "metrics.json"
                if args.skip_existing and out_json.is_file():
                    continue
                print(f"[eval] stem={original_stem} ({safe_stem}) model={target.model_id}", flush=True)
                payload = eval_one_target(
                    target,
                    data_root=data_root,
                    split_subdir=args.split_subdir,
                    safe_stem=safe_stem,
                    device=device,
                    batch_size=args.batch_size,
                    max_samples=args.max_samples,
                    log_path=args.log_parquet_paths,
                )
                payload["original_stem"] = original_stem
                payload["safe_stem"] = safe_stem
                write_metrics_json(out_json, payload)
                m = payload.get("metrics") or {}
                append_summary_row(
                    summary_csv,
                    {
                        "safe_stem": safe_stem,
                        "original_stem": original_stem,
                        "n_classes": n_classes,
                        "model_id": target.model_id,
                        "source": target.source,
                        "accuracy": m.get("accuracy", ""),
                        "f1_macro": m.get("f1_macro", ""),
                        "balanced_accuracy": m.get("balanced_accuracy", ""),
                        "roc_auc_ovr_macro": m.get("roc_auc_ovr_macro", ""),
                        "error": payload.get("error", ""),
                        "config_path": str(target.config_path),
                        "weight_path": str(target.weight_path),
                        "metrics_path": str(out_json),
                    },
                    summary_fields,
                )

        for mj, nc in moe_jobs:
            if nc is None:
                continue
            base = mj.stem
            variants = [(f"moe_uniform__{base}", None)]
            if args.moe_gate and args.moe_gate.is_file():
                variants.append((f"moe_trained_gate__{base}", args.moe_gate))
            for vid, gpath in variants:
                out_json = out_dir / f"{nc}label" / safe_stem / vid / "metrics.json"
                if args.skip_existing and out_json.is_file():
                    continue
                print(f"[moe] stem={original_stem} variant={vid}", flush=True)
                payload = eval_moe_manifest(
                    mj,
                    variant_id=vid,
                    data_root=data_root,
                    split_subdir=args.split_subdir,
                    safe_stem=safe_stem,
                    device=device,
                    batch_size=args.batch_size,
                    max_samples=args.max_samples,
                    gate_safetensors=gpath,
                    use_text_gate=args.moe_text_gate,
                    sparse_top_k=args.moe_sparse_top_k if args.moe_sparse_top_k > 0 else None,
                    gate_hidden_dim=args.moe_gate_hidden if args.moe_gate_hidden > 0 else None,
                    log_path=args.log_parquet_paths,
                )
                payload["original_stem"] = original_stem
                payload["safe_stem"] = safe_stem
                write_metrics_json(out_json, payload)
                m = payload.get("metrics") or {}
                append_summary_row(
                    summary_csv,
                    {
                        "safe_stem": safe_stem,
                        "original_stem": original_stem,
                        "n_classes": nc,
                        "model_id": vid,
                        "source": "moe_manifest",
                        "accuracy": m.get("accuracy", ""),
                        "f1_macro": m.get("f1_macro", ""),
                        "balanced_accuracy": m.get("balanced_accuracy", ""),
                        "roc_auc_ovr_macro": m.get("roc_auc_ovr_macro", ""),
                        "error": payload.get("error", ""),
                        "config_path": str(mj),
                        "weight_path": str(gpath or "uniform"),
                        "metrics_path": str(out_json),
                    },
                    summary_fields,
                )

    if not args.no_export_metrics_csv:
        export_metrics_tables_from_json(out_dir)

    print(f"[done] metrics under {out_dir}", flush=True)


if __name__ == "__main__":
    main()


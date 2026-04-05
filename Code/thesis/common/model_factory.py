from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

import torch
import torch.nn as nn

from Code.models.deep_learning.hrm.hrm_model import (
    HRMClassifierWrapper,
    build_sentiment_mlp_head,
)
from Code.models.utils.utils import MLModule
from Code.models.deep_learning.rnn.base.modules import GRUModule, LSTMModule
from Code.thesis.common.feature_pretrain_models import FeatureEncoderClassifier, FeaturePretrainAutoencoder
from Code.thesis.common.wrappers import RNNClassifier


def _walk_find_class(class_name: str, models_root: Path, code_dir: Path) -> Optional[Type]:
    for py in models_root.rglob("*.py"):
        if py.name.startswith("__"):
            continue
        rel = py.relative_to(code_dir)
        mod_path = "Code." + rel.with_suffix("").as_posix().replace("/", ".")
        try:
            mod = importlib.import_module(mod_path)
        except Exception:
            continue
        if hasattr(mod, class_name):
            return getattr(mod, class_name)
    return None


def get_model_class(class_name: str) -> Optional[Type]:
    repo = Path(__file__).resolve().parents[3]
    code_dir = repo / "Code"
    return _walk_find_class(class_name, code_dir / "models", code_dir)


def instantiate_from_kwargs(cls: Type, kwargs: Dict[str, Any]) -> nn.Module:
    try:
        sig = inspect.signature(cls.__init__)
        valid = [
            p.name
            for p in sig.parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        ]
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        filtered = kwargs if has_varkw else {k: v for k, v in kwargs.items() if k in valid}
        return cls(**filtered)
    except Exception:
        return cls(**kwargs)


def build_model_from_config_dict(
    cfg: Dict[str, Any],
    n_classes: int,
    label_folder: str,
    *,
    hrm_encoder_only: bool = False,
) -> Tuple[nn.Module, str]:
    """Returns (model, class_name).

    If ``hrm_encoder_only`` is True and the config is ``HierarchicalReasoningModel``,
    return the bare encoder (MLM pretrain: trunk + ``lm_head`` only; no K-way head).
    """
    class_name = next(iter(cfg))
    params = dict(cfg[class_name])

    if class_name == "FeaturePretrainAutoencoder":
        model = FeaturePretrainAutoencoder(**params)
        return model, class_name
    if class_name == "FeatureEncoderClassifier":
        model = FeatureEncoderClassifier(n_classes=int(n_classes), **params)
        return model, class_name

    cls = get_model_class(class_name)
    if cls is None:
        raise RuntimeError(f"Unknown model class: {class_name}")

    if class_name == "LLMModule":
        params["n_classes"] = n_classes

    if class_name == "HierarchicalReasoningModel":
        params = dict(params)
        head_spec = params.pop("classification_head", None)
        n_json = params.pop("n_classes", None)
        n_final = int(n_json) if n_json is not None else int(n_classes)
        encoder = instantiate_from_kwargs(cls, params)
        if hrm_encoder_only:
            model = encoder
        else:
            head: Optional[nn.Module] = None
            if head_spec is not None:
                htype = str(head_spec.get("type", "")).lower()
                if htype == "mlp_sentiment_v1":
                    d = int(encoder.hrm_config.output_embed_dim)
                    nc = int(head_spec.get("num_classes", n_final))
                    head = build_sentiment_mlp_head(d, nc)
                    n_final = nc
            model = HRMClassifierWrapper(encoder, n_final, head=head)
    else:
        model = instantiate_from_kwargs(cls, params)

    if isinstance(model, (LSTMModule, GRUModule)):
        model = RNNClassifier(model, n_classes)

    return model, class_name


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def is_text_model_config_path(config_path: Path) -> bool:
    p = str(config_path).lower().replace("\\", "/")
    return "transformer" in p or "/hrm/" in p


def is_feature_pretrain_autoencoder_config(cfg: Dict[str, Any]) -> bool:
    return "FeaturePretrainAutoencoder" in cfg


def is_feature_encoder_classifier_config(cfg: Dict[str, Any]) -> bool:
    return "FeatureEncoderClassifier" in cfg

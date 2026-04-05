"""Minimal norm package."""

from __future__ import annotations

import torch.nn as nn

from ._base import BaseNorm

# PyTorch built-ins used when registry resolution is unavailable
LayerNorm = nn.LayerNorm

__all__ = ["BaseNorm", "LayerNorm", "get_norm"]


def get_norm(name: str, *args, **kwargs):
    import importlib

    from .....config.deep_learning.model_registry import get_model_module_path

    key = name.strip().lower()
    if key == "layernorm":
        return nn.LayerNorm(*args, **kwargs) if (args or kwargs) else nn.LayerNorm
    try:
        module_path = get_model_module_path(name)
        m = importlib.import_module(module_path)
        cls = getattr(m, name)
    except (KeyError, AttributeError, ImportError) as e:
        raise KeyError(f"Unknown norm {name!r}: {e}") from e
    if args or kwargs:
        return cls(*args, **kwargs)
    return cls

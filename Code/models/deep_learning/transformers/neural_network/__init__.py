"""Minimal neural_network facade (no ffnn dependency)."""

from __future__ import annotations

import importlib

__all__ = ["get_neural_network"]


def get_neural_network(name: str, *args, **kwargs):
    from .....config.deep_learning.model_registry import get_model_module_path

    try:
        module_path = get_model_module_path(name)
        m = importlib.import_module(module_path)
        cls = getattr(m, name)
    except (KeyError, AttributeError, ImportError) as e:
        raise KeyError(f"Unknown neural_network {name!r}: {e}") from e
    if args or kwargs:
        return cls(*args, **kwargs)
    return cls

"""Minimal token_selection (registry-only)."""

from __future__ import annotations

import importlib

from ._base import BaseTokenSelector

__all__ = ["BaseTokenSelector", "get_token_selection"]


def get_token_selection(name: str, *args, **kwargs):
    from .....config.deep_learning.model_registry import get_model_module_path

    try:
        module_path = get_model_module_path(name)
        m = importlib.import_module(module_path)
        cls = getattr(m, name)
    except (KeyError, AttributeError, ImportError) as e:
        raise KeyError(f"Unknown token_selection {name!r}: {e}") from e
    if args or kwargs:
        return cls(*args, **kwargs)
    return cls

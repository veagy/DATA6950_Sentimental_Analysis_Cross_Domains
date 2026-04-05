"""Minimal embeddings package (full language/vision submodules optional in this checkout)."""

from __future__ import annotations

from ._base import BaseEmbedding

__all__ = ["BaseEmbedding", "get_embeddings"]


def get_embeddings(name: str, *args, **kwargs):
    import importlib

    from .....config.deep_learning.model_registry import get_model_module_path

    try:
        module_path = get_model_module_path(name)
        m = importlib.import_module(module_path)
        cls = getattr(m, name)
    except (KeyError, AttributeError, ImportError) as e:
        raise KeyError(f"Unknown embeddings {name!r}: {e}") from e
    if args or kwargs:
        return cls(*args, **kwargs)
    return cls

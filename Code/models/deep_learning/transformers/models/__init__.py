"""
Minimal transformer models facade for thesis training (HRM EncoderLM, HF-backed LLMs).

Full model zoo imports were removed; ``get_models`` falls back to ``model_registry``.
"""

from __future__ import annotations

from ._base import BaseTransformerModel, _parse_model_config, _resolve_pipeline_module
from .configs import TransformerConfig, config_to_dict, resolve_decoder_config
from .decoder import DecoderLM
from .encoder import EncoderLM
from .huggingface import HuggingFaceTransformer
from .translational import TranslationalLM

__all__ = [
    "BaseTransformerModel",
    "_parse_model_config",
    "_resolve_pipeline_module",
    "TransformerConfig",
    "config_to_dict",
    "resolve_decoder_config",
    "DecoderLM",
    "EncoderLM",
    "TranslationalLM",
    "HuggingFaceTransformer",
    "get_models",
]


def get_models(name: str, *args, **kwargs):
    import importlib
    import sys

    mod = sys.modules[__name__]
    registry = {
        n: getattr(mod, n)
        for n in __all__
        if n != "get_models" and isinstance(getattr(mod, n, None), type)
    }
    key = name.strip().lower()
    cls = None
    for k, v in registry.items():
        if k.lower() == key:
            cls = v
            break
    if cls is None:
        try:
            from .....config.deep_learning.model_registry import get_model_module_path

            module_path = get_model_module_path(name)
            m = importlib.import_module(module_path)
            cls = getattr(m, name)
        except (KeyError, AttributeError, ImportError):
            pass
    if cls is None:
        raise KeyError(f"Unknown model '{name}'. Local: {sorted(registry.keys())}")
    if args or kwargs:
        return cls(*args, **kwargs)
    return cls

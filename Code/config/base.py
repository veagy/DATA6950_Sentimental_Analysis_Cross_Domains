"""
Base config infrastructure for model configuration.

Reusable across config types (machine_learning, deep_learning, etc.).
"""

from __future__ import annotations

import copy
import importlib
from typing import Any

# Reserved attributes for ConfigTemplate (not passed to model)
_RESERVED = frozenset({"model_name", "model_path", "_immutable", "_extra_kwargs", "_config_fields", "immutable"})


def resolve_config_value(val: Any) -> Any:
    """
    Resolve JSON placeholder strings to Python values.
    Values starting with @ are special (e.g. @torch.float); others are returned as-is.
    Extensible for other config types.
    """
    if not isinstance(val, str) or not val.startswith("@"):
        return val
    key = val[1:].lower()
    try:
        import torch
    except ImportError:
        torch = None
    if torch is None:
        if key in ("torch.float", "torch.float32", "torch.float64", "torch.eps"):
            return None  # Fallback when torch unavailable
        if key == "scale":
            return "scale"
        return val
    if key == "torch.float":
        return torch.float
    if key == "torch.float32":
        return torch.float32
    if key == "torch.float64":
        return torch.float64
    if key == "torch.eps":
        return torch.finfo(torch.float32).eps
    if key == "scale":
        return "scale"
    return val


class ConfigTemplate:
    """
    Base class for model configuration templates.
    Supports immutable (copy-on-write) and mutable modes.
    Reusable for any config type (ML, DL, etc.).
    """

    model_name: str = ""
    model_path: str = ""  # Full module path for dynamic import

    def __init__(self, immutable: bool = True, **kwargs):
        self._immutable = immutable
        self._extra_kwargs = kwargs.copy()
        self._config_fields: set[str] = set()

    def _get_config_fields(self) -> set[str]:
        """Return set of attribute names that are config params (not reserved)."""
        if self._config_fields:
            return self._config_fields
        return {
            k for k in vars(self).keys()
            if not k.startswith("_") and k not in _RESERVED
        }

    def to_kwargs(self) -> dict[str, Any]:
        """Return all config params as kwargs for model construction."""
        kwargs = {}
        for key in self._get_config_fields():
            try:
                val = getattr(self, key)
                kwargs[key] = val
            except AttributeError:
                pass
        kwargs.update(self._extra_kwargs)
        return kwargs

    def copy(self) -> ConfigTemplate:
        """Return a deep copy. The copy is mutable so you can modify it without affecting the original."""
        new = copy.deepcopy(self)
        new._immutable = False
        return new

    def set_mutable(self, mutable: bool = True) -> None:
        """Switch to mutable mode. When mutable, attribute changes persist on the template."""
        object.__setattr__(self, "_immutable", not mutable)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _RESERVED or name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        if self._immutable and hasattr(self, name):
            raise AttributeError(
                f"Cannot modify immutable config. Use config.copy() then modify the copy, "
                f"or call config.set_mutable(True) to allow in-place changes."
            )
        object.__setattr__(self, name, value)

    def instantiate(self, **overrides) -> Any:
        """
        Dynamically import the model class and instantiate with config params.
        Overrides are merged and take precedence over config values.
        """
        module_path = self.model_path or self._resolve_model_path()
        class_name = self.model_name or self.__class__.__name__.removesuffix("Config")
        module = importlib.import_module(module_path)
        model_cls = getattr(module, class_name)
        kwargs = self.to_kwargs()
        kwargs.update(overrides)
        return model_cls(**kwargs)

    def _resolve_model_path(self) -> str:
        """Resolve model module path. Override in subclasses or set model_path."""
        raise NotImplementedError(
            "Set model_path on the config instance/class or override _resolve_model_path in subclass."
        )


class JsonConfig(ConfigTemplate):
    """
    Config loaded from JSON dict. Wraps dict for ConfigTemplate interface.
    Supports instantiate(), copy(), immutable/mutable semantics.
    """

    def __init__(
        self,
        model_name: str,
        model_path: str,
        params: dict[str, Any],
        immutable: bool = True,
        **kwargs,
    ):
        super().__init__(immutable=immutable, **kwargs)
        object.__setattr__(self, "model_name", model_name)
        object.__setattr__(self, "model_path", model_path)
        for k, v in params.items():
            object.__setattr__(self, k, resolve_config_value(v))


def create_model(config: ConfigTemplate, **overrides):
    """Instantiate a model from config. Overrides are merged and take precedence."""
    return config.instantiate(**overrides)

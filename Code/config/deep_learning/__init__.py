"""
Config-driven deep learning model instantiation.

Config templates are stored as JSON files. Use instantiate_model() for simple key-value lookup:

  from ...config.deep_learning import instantiate_model

  model = instantiate_model("StandardDense", {"in_features": 128, "out_features": 64})
  model = instantiate_model("MeanPooling", {"word_embedding_dimension": 768})

Or use load_config_json() + create_model() for more control.
"""

from ...config import ConfigTemplate, JsonConfig, create_model
from .model_registry import get_model_module_path, register_model, list_registered_models

_MODEL_CONFIG_REGISTRY: dict[str, str] | None = None


def _load_model_config_registry() -> dict[str, str]:
    """Load model_config_registry.json."""
    global _MODEL_CONFIG_REGISTRY
    if _MODEL_CONFIG_REGISTRY is not None:
        return _MODEL_CONFIG_REGISTRY
    import json
    from pathlib import Path
    config_root = Path(__file__).resolve().parent
    registry_path = config_root / "model_config_registry.json"
    if registry_path.exists():
        _MODEL_CONFIG_REGISTRY = json.loads(registry_path.read_text(encoding="utf-8"))
    else:
        _MODEL_CONFIG_REGISTRY = {}
    return _MODEL_CONFIG_REGISTRY


def instantiate_model(
    model_name: str,
    overrides: dict | None = None,
    immutable: bool = True,
    **kwargs,
):
    """
    Instantiate a model by name. Looks up config from model_config_registry.json.

    Args:
        model_name: Model class name, e.g. "StandardDense", "MeanPooling", "WordPieceTokenizer"
        overrides: Param overrides (e.g. {"in_features": 128})
        immutable: Whether the loaded config is immutable
        **kwargs: Additional overrides (merged with overrides dict)

    Returns:
        Instantiated model instance.
    """
    reg = _load_model_config_registry()
    config_rel_path = reg.get(model_name)
    if config_rel_path is None:
        try:
            module_path = get_model_module_path(model_name)
            config_rel_path = module_path.replace("Code.models.deep_learning.", "")
        except KeyError:
            raise KeyError(
                f"Model {model_name!r} not in model_config_registry. "
                f"Run: python src/scripts/setup/generate_dl_configs.py --format json"
            )
    merged_overrides = dict(overrides) if overrides else {}
    merged_overrides.update(kwargs)
    config = load_config_json(config_rel_path, model_name, merged_overrides, immutable=immutable)
    return create_model(config)


def load_config_json(
    module_rel_path: str,
    model_name: str,
    overrides: dict | None = None,
    immutable: bool = True,
) -> JsonConfig:
    """
    Load config from JSON file.

    Args:
        module_rel_path: Relative path under config/deep_learning, e.g.
            "ffnn.nn_layers.standard.layers"
        model_name: Model class name, e.g. "StandardDense"
        overrides: Optional dict of param overrides (merged into config)
        immutable: Whether the returned config is immutable (default True)

    Returns:
        JsonConfig instance with instantiate(), copy(), etc.
    """
    import json
    from pathlib import Path

    config_root = Path(__file__).resolve().parent
    json_path = config_root / f"{module_rel_path.replace('.', '/')}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Config JSON not found: {json_path}")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if model_name not in data:
        raise KeyError(f"Model {model_name!r} not in {json_path}")
    params = dict(data[model_name])
    if overrides:
        params.update(overrides)
    module_path = f"Code.models.deep_learning.{module_rel_path}"
    return JsonConfig(model_name, module_path, params, immutable=immutable)


__all__ = [
    "ConfigTemplate",
    "JsonConfig",
    "create_model",
    "instantiate_model",
    "load_config_json",
    "get_model_module_path",
    "register_model",
    "list_registered_models",
]

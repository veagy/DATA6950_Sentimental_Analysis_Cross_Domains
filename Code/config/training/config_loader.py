"""
Training configuration loader: load, merge, and validate training configs.
Supports preset configs and CLI overrides.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def _get_config_root() -> Path:
    """Return path to src/config/training."""
    return Path(__file__).resolve().parent


def load_training_config(
    config_path: Optional[str] = None,
    preset: Optional[str] = None,
    overrides: Optional[dict] = None,
) -> dict[str, Any]:
    """
    Load training configuration from file or preset, merge with overrides.

    Args:
        config_path: Path to JSON config file. If None, uses preset or defaults.
        preset: Preset name (minimal, standard, large). Ignored if config_path set.
        overrides: Dict of values to merge (take precedence over loaded config).

    Returns:
        Merged configuration dict.
    """
    root = _get_config_root()
    config: dict[str, Any] = {}

    if config_path:
        path = Path(config_path)
        if not path.is_absolute():
            path = root.parent.parent.parent / path  # PROJECT_ROOT relative
        if path.exists():
            config = json.loads(path.read_text(encoding="utf-8"))
    elif preset:
        preset_path = root / "presets" / f"{preset}.json"
        if preset_path.exists():
            config = json.loads(preset_path.read_text(encoding="utf-8"))

    if not config:
        config = _default_training_config()

    if overrides:
        config = _deep_merge(config, overrides)

    return config


def _default_training_config() -> dict[str, Any]:
    """Default minimal config when no file or preset provided."""
    return {
        "data": {
            "source": "data/train.csv",
            "label_col": "label",
            "test_size": 0.2,
            "preprocessing": {"clean": False, "engineer": False, "scale": None, "paradigm": "supervised"},
        },
        "model": {"name": "StandardDense", "overrides": {"in_features": 128, "out_features": 3}},
        "training": {
            "epochs": 20,
            "batch_size": 32,
            "loss": "CrossEntropyLoss",
            "optimizer": "adamw",
            "lr": 0.001,
            "val_split": 0.2,
            "run_id": "default",
        },
    }


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override values take precedence."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def validate_training_config(config: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate config against schema. Returns (valid, list of error messages).
    """
    errors: list[str] = []

    # Model name must exist in registry
    model_name = config.get("model", {}).get("name")
    if model_name:
        try:
            from ...config.deep_learning import _load_model_config_registry
            reg = _load_model_config_registry()
            if model_name not in reg:
                try:
                    from ...config.deep_learning.model_registry import get_model_module_path
                    get_model_module_path(model_name)
                except KeyError:
                    errors.append(f"Model '{model_name}' not found in model_config_registry")
        except Exception as e:
            errors.append(f"Could not verify model registry: {e}")

    # Training epochs must be positive
    epochs = config.get("training", {}).get("epochs")
    if epochs is not None and (not isinstance(epochs, int) or epochs < 1):
        errors.append("training.epochs must be a positive integer")

    # Val split in [0, 1]
    val_split = config.get("training", {}).get("val_split")
    if val_split is not None and (not isinstance(val_split, (int, float)) or val_split < 0 or val_split > 1):
        errors.append("training.val_split must be between 0 and 1")

    return len(errors) == 0, errors


def config_to_cli_args(config: dict[str, Any]) -> list[str]:
    """
    Convert config dict to argv-like list for subprocess.
    Used when dispatching to pretrain.py / finetune.py.
    """
    args: list[str] = []
    t = config.get("training", {})
    m = config.get("model", {})
    d = config.get("data", {})

    if t.get("epochs"):
        args.extend(["--epochs", str(t["epochs"])])
    if t.get("batch_size"):
        args.extend(["--batch_size", str(t["batch_size"])])
    if t.get("lr"):
        args.extend(["--lr", str(t["lr"])])
    if t.get("loss"):
        args.extend(["--loss", str(t["loss"])])
    if t.get("optimizer"):
        args.extend(["--optimizer", str(t["optimizer"])])
    if m.get("name"):
        args.extend(["--model", str(m["name"])])
    if d.get("source"):
        args.extend(["--data_source", str(d["source"])])
    if d.get("label_col"):
        args.extend(["--label_col", str(d["label_col"])])

    return args

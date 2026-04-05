"""
ModelRegistry: maps model class names to their module paths for dynamic import.

Built from src/models/machine_learning package layout and __registry__.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_MODELS_REGISTRY_JSON = _PROJECT_ROOT / "src" / "models" / "machine_learning" / "__registry__.json"
_MODELS_ROOT = _PROJECT_ROOT / "src" / "models" / "machine_learning"

# Manual mapping: model_name -> (subpath under machine_learning, e.g. "classification.linear_model.linear_models")
# The registry JSON has nested structure; we flatten and map to actual module paths by scanning __init__.py
_MODEL_TO_MODULE: dict[str, str] = {}


def _flatten_registry(obj: Any, acc: list[str]) -> None:
    """Flatten nested __registry__.json structure to a list of model names."""
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, str):
                acc.append(item)
            elif isinstance(item, dict):
                for v in item.values():
                    _flatten_registry(v, acc)
    elif isinstance(obj, dict):
        for v in obj.values():
            _flatten_registry(v, acc)


def _model_to_module_from_imports() -> dict[str, str]:
    """Scan machine_learning package .py files: map each class to its module path."""
    import ast

    result: dict[str, str] = {}

    for py_path in _MODELS_ROOT.rglob("*.py"):
        if "__pycache__" in str(py_path):
            continue
        try:
            rel = py_path.relative_to(_MODELS_ROOT)
        except ValueError:
            continue
        parts = list(rel.parts)
        if not parts:
            continue
        # classification/linear_model/linear_models.py -> classification.linear_model.linear_models
        module_rel = ".".join(parts).replace(".py", "").rstrip(".__init__")
        if module_rel.endswith(".__init__"):
            module_rel = module_rel[:-9]
        module_path = f"Code.models.machine_learning.{module_rel}"
        try:
            tree = ast.parse(py_path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    result[node.name] = module_path
        except Exception:
            pass

    return result


def _build_registry() -> dict[str, str]:
    """Build model name -> module path mapping by scanning model Python files."""
    return _model_to_module_from_imports()


def _load_registry() -> dict[str, str]:
    """Load or build registry, cache result."""
    if not _MODEL_TO_MODULE:
        _MODEL_TO_MODULE.update(_build_registry())
    return _MODEL_TO_MODULE


def get_model_module_path(model_name: str) -> str:
    """Return full module path for a model class name."""
    reg = _load_registry()
    path = reg.get(model_name)
    if path:
        return path
    # Fallback: search common locations by naming convention
    if "Classifier" in model_name:
        for sub in ["classification.linear_model.linear_models", "classification.linear_model.linear_optimizer_models",
                    "classification.svm.svm", "classification.knn.knn", "classification.tree_models.trees",
                    "classification.other.other_models", "classification.ensemble.bagging.bagging",
                    "classification.ensemble.boosting.boosting", "classification.ensemble.meta_estimators.meta_estimators",
                    "classification.ensemble.stacking_and_voting.stacking_and_voting"]:
            mod = f"Code.models.machine_learning.{sub}"
            try:
                m = __import__(mod, fromlist=[model_name])
                if hasattr(m, model_name):
                    _MODEL_TO_MODULE[model_name] = mod
                    return mod
            except Exception:
                continue
    if "Regressor" in model_name or "Ridge" == model_name or "Lasso" == model_name:
        for sub in ["regression.linear_model.linear_models", "regression.linear_model.linear_optimizer_models",
                    "regression.svm.svm", "regression.knn.knn", "regression.tree_models.trees"]:
            mod = f"Code.models.machine_learning.{sub}"
            try:
                m = __import__(mod, fromlist=[model_name])
                if hasattr(m, model_name):
                    _MODEL_TO_MODULE[model_name] = mod
                    return mod
            except Exception:
                continue
    if "Cluster" in model_name or "KMeans" in model_name or "DBSCAN" in model_name:
        for sub in ["clustering.centroid_based.centroid_based", "clustering.density_based.density_based"]:
            mod = f"Code.models.machine_learning.{sub}"
            try:
                m = __import__(mod, fromlist=[model_name])
                if hasattr(m, model_name):
                    _MODEL_TO_MODULE[model_name] = mod
                    return mod
            except Exception:
                continue
    raise KeyError(f"Model {model_name!r} not found in registry.")


def register_model(model_name: str, module_path: str) -> None:
    """Manually register a model -> module path."""
    _load_registry()
    _MODEL_TO_MODULE[model_name] = module_path


def list_registered_models() -> list[str]:
    """Return sorted list of registered model names."""
    return sorted(_load_registry().keys())

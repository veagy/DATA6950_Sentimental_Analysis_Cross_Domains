"""
ModelRegistry: maps model class names to their module paths for dynamic import.

Built from src/models/ package layout via AST scan (all DLModule-compatible models).
"""

from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_MODELS_ROOT = _PROJECT_ROOT / "src" / "models"

_MODEL_TO_MODULE: dict[str, str] = {}


def _model_to_module_from_imports() -> dict[str, str]:
    """Scan src/models/ package .py files: map each class to its module path."""
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
        module_rel = ".".join(parts).replace(".py", "")
        if module_rel.endswith(".__init__"):
            module_rel = module_rel[:-9]
        module_path = f"Code.models.{module_rel}"
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
    # Fallback: try known packages (DL and ML families)
    _KNOWN_DL_SUBPATHS = [
        "ffnn.nn_layers",
        "ffnn",
        "transformers.tokenizer",
        "transformers.pooling",
        "transformers.norm",
        "transformers.attention",
        "transformers.embeddings",
        "transformers.positional_encoders",
        "transformers.logits_calculation",
        "transformers.token_selection",
        "transformers.neural_network",
        "cnn",
        "rnn",
        "gan",
        "dbn",
        "auto_encoders",
        "llm",
    ]
    _KNOWN_ML_SUBPATHS = [
        "machine_learning",
        "machine_learning.classification",
        "machine_learning.clustering",
        "machine_learning.regression",
        "machine_learning.preprocessing",
        "machine_learning.feature_selection",
        "machine_learning.feature_extraction",
        "machine_learning.transformer",
        "machine_learning.cross_validation",
    ]
    for sub in _KNOWN_DL_SUBPATHS:
        mod = f"Code.models.deep_learning.{sub}"
        try:
            m = __import__(mod, fromlist=[model_name])
            if hasattr(m, model_name):
                _MODEL_TO_MODULE[model_name] = mod
                return mod
        except Exception:
            continue
    for sub in _KNOWN_ML_SUBPATHS:
        mod = f"Code.models.{sub}"
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


# Register Pipeline (lives natively in Code.models.models)
register_model("Pipeline", "Code.models.models")

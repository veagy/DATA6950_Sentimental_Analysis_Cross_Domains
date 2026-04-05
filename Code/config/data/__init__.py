"""
Data pipeline configuration and registry.
"""
from pathlib import Path
import json

_REGISTRY_PATH = Path(__file__).resolve().parent / "pipeline_registry.json"
_REGISTRY_CACHE: dict | None = None


def get_pipeline_registry() -> dict:
    """Load pipeline registry (source URI patterns -> pipeline config)."""
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is not None:
        return _REGISTRY_CACHE
    if _REGISTRY_PATH.exists():
        _REGISTRY_CACHE = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
    else:
        _REGISTRY_CACHE = {}
    return _REGISTRY_CACHE


def get_pipeline_for_source(source: str) -> dict | None:
    """
    Look up pipeline config for a data source URI.
    Returns dict with pipeline type and optional params, or None if no match.
    """
    registry = get_pipeline_registry()
    source_lower = source.lower()
    for pattern, config in registry.get("sources", {}).items():
        if pattern in source_lower or source_lower.endswith(pattern.lstrip("*")):
            return config
    # Default fallback by extension
    ext = Path(source).suffix.lower().lstrip(".")
    return registry.get("extensions", {}).get(ext, registry.get("extensions", {}).get("default"))

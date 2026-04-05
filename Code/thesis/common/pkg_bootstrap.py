"""Install a namespace package for ``Code.models`` without running the heavy ``__init__.py``."""

from __future__ import annotations

import sys
import types
from pathlib import Path


def install_lazy_code_models(repo_root: Path) -> None:
    """Must run before any ``from Code.models...`` that would load the full package."""
    key = "Code.models"
    if key in sys.modules:
        return
    m = types.ModuleType(key)
    m.__path__ = [str((repo_root / "Code" / "models").resolve())]  # type: ignore[attr-defined]
    sys.modules[key] = m

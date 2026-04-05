"""Avoid importing the heavy ``Code.models`` top-level ``__init__.py`` during tests."""
from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]

from Code.thesis.common.pkg_bootstrap import install_lazy_code_models

install_lazy_code_models(_REPO)

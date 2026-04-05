# src/test/unittests/deep_learning/test_model_registry.py
"""
Tests for the DL model registry.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from ....config.deep_learning.model_registry import (  # noqa: E402
    get_model_module_path,
    list_registered_models,
    register_model,
)

@pytest.mark.unit
def test_list_registered_models_non_empty():
    models = list_registered_models()
    assert isinstance(models, list)
    assert len(models) >= 1

@pytest.mark.unit
def test_get_model_module_path_pipeline():
    path = get_model_module_path("Pipeline")
    assert path == "Code.models.models"

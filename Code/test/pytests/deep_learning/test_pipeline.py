import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from ....models.models import Pipeline

def _make_pipeline(mermaid: str = None, modules: dict = None) -> Pipeline:
    if mermaid is None:
        mermaid = "flowchart LR\n  input[IDENTITY] --> linear[Linear(16,4)]\n"
    if modules is None:
        modules = {"linear": nn.Linear(16, 4)}
    return Pipeline(mermaid_flowchart=mermaid, modules=modules)

@pytest.mark.unit
def test_params_calculator_success():
    pipeline = _make_pipeline()
    result = pipeline.params_calculator()
    assert result["status"] == "Native Math Verified"
    assert "total_params" in result
    assert "trainable_params" in result
    assert "param_ranges" in result

@pytest.mark.unit
def test_dummy_propagate():
    pipeline = _make_pipeline()
    out_shape, trace = pipeline.dummy_propagate((1, 16))
    assert isinstance(out_shape, tuple)
    assert len(out_shape) > 0
    assert out_shape[0] == 1  # batch dim preserved

@pytest.mark.unit
def test_forward_pass_test():
    pipeline = _make_pipeline()
    result = pipeline.forward_pass_test((1, 16))
    assert result.get("success") is True

@pytest.mark.unit
def test_backprop_test():
    pipeline = _make_pipeline()
    result = pipeline.backprop_test((1, 16))
    assert result.get("success") is True
    assert result.get("input_grad_computed") is True

@pytest.mark.unit
def test_get_config_combinations():
    pipeline = _make_pipeline()
    combos = pipeline.get_config_combinations()
    assert isinstance(combos, list)
    assert len(combos) > 0

@pytest.mark.unit
def test_get_build_signature():
    pipeline = _make_pipeline()
    sig = pipeline.get_build_signature()
    assert "nodes" in sig
    assert "edges" in sig

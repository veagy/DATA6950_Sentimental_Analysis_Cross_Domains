# src/test/test_dl_model_layers.py
"""
Smoke tests for DLModelLayers.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ..models.deep_learning.models import DLModelLayers  # noqa: E402

@pytest.mark.unit
def test_dl_model_layers_basic():
    """Test with a simple list of layers."""
    act_funcs = [nn.ReLU()]
    layers_config = [
        ("nn", {"in_features": 10, "out_features": 20}),
        ("act", None), 
        ("nn", {"in_features": 20, "out_features": 5})
    ]
    
    model = DLModelLayers(layers=layers_config, act_funcs=act_funcs)
    x = torch.randn(1, 10)
    y = model(x)
    assert y.shape == (1, 5)

@pytest.mark.unit
def test_dl_model_layers_rnn_handling():
    """Test with tuple return (simulated with LSTM)."""
    lstm_config = [
        ("lstm", {"input_size": 10, "hidden_size": 20, "batch_first": True}),
        ("nn", {"in_features": 20, "out_features": 5})
    ]
    
    model = DLModelLayers(layers=lstm_config, act_funcs=[])
    x_seq = torch.randn(2, 5, 10)
    y_seq = model(x_seq)
    assert y_seq.shape == (2, 5, 5)

# src/test/verify_rnn_family.py
"""
Verification tests for RNN family modules.
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ..models.deep_learning.rnn import (  # noqa: E402
    RNNModule, LSTMModule, GRUModule, 
    RNNCell, GRUCell, LSTMCell,
    ESNModule, RWKVModule, MambaModule, NTMModule, HopfieldNetworkModule
)

@pytest.mark.unit
def test_rnn_module():
    input_size, hidden_size, num_layers = 10, 20, 2
    x = torch.randn(5, 3, input_size)
    model = RNNModule(input_size, hidden_size, num_layers)
    out, hn = model(x)
    assert out.shape == (5, 3, hidden_size)

@pytest.mark.unit
def test_lstm_module():
    input_size, hidden_size, num_layers = 10, 20, 2
    x = torch.randn(5, 3, input_size)
    model = LSTMModule(input_size, hidden_size, num_layers)
    out, (hn, cn) = model(x)
    assert out.shape == (5, 3, hidden_size)

@pytest.mark.unit
def test_gru_module():
    input_size, hidden_size, num_layers = 10, 20, 2
    x = torch.randn(5, 3, input_size)
    model = GRUModule(input_size, hidden_size, num_layers, funcs=[['sigmoid']*3]*2)
    out, hn = model(x)
    assert out.shape == (5, 3, hidden_size)

@pytest.mark.unit
def test_esn_module():
    input_size, hidden_size, num_layers = 10, 20, 2
    x = torch.randn(5, 3, input_size)
    model = ESNModule(input_size, hidden_size, num_layers)
    out, hn = model(x)
    assert out.shape == (5, 3, hidden_size)

@pytest.mark.unit
def test_rwkv_module():
    input_size, hidden_size, num_layers = 10, 20, 2
    x = torch.randn(5, 3, input_size)
    model = RWKVModule(input_size, hidden_size, num_layers)
    out, _ = model(x)
    assert out.shape == (5, 3, hidden_size)

@pytest.mark.unit
def test_mamba_module():
    input_size, hidden_size, num_layers = 10, 20, 2
    x = torch.randn(5, 3, input_size)
    model = MambaModule(input_size, hidden_size, num_layers)
    out, _ = model(x)
    assert out.shape == (5, 3, hidden_size)

@pytest.mark.unit
def test_ntm_module():
    input_size, hidden_size, num_layers = 10, 20, 2
    model = NTMModule(input_size, hidden_size, num_layers, 5, 8)
    x = torch.randn(5, 3, input_size)
    out, _ = model(x)
    assert out.shape == (5, 3, 8)

@pytest.mark.unit
def test_hopfield_module():
    input_size, hidden_size, num_layers = 10, 20, 2
    model = HopfieldNetworkModule(input_size, hidden_size, num_layers)
    x = torch.randn(5, 3, input_size)
    out = model(x)
    assert out.shape == (5, 3, hidden_size)

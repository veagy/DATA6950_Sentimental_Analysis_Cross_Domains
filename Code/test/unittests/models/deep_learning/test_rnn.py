"""
Unit tests for RNN architectures.
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from .....models.deep_learning.rnn import (
    RNNModule,
    LSTMModule,
    GRUModule,
)

@pytest.mark.unit
def test_rnn_module_forward():
    input_size = 10
    hidden_size = 20
    batch_size = 2
    seq_len = 5
    
    model = RNNModule(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
    x = torch.randn(batch_size, seq_len, input_size)
    y, h = model(x)
    
    assert y.shape == (batch_size, seq_len, hidden_size)
    assert h.shape == (1, batch_size, hidden_size)

@pytest.mark.unit
def test_lstm_module_forward():
    input_size = 10
    hidden_size = 20
    batch_size = 2
    seq_len = 5
    
    model = LSTMModule(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
    x = torch.randn(batch_size, seq_len, input_size)
    y, (h, c) = model(x)
    
    assert y.shape == (batch_size, seq_len, hidden_size)
    assert h.shape == (1, batch_size, hidden_size)
    assert c.shape == (1, batch_size, hidden_size)

@pytest.mark.unit
def test_gru_module_forward():
    input_size = 10
    hidden_size = 20
    batch_size = 2
    seq_len = 5
    
    model = GRUModule(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
    x = torch.randn(batch_size, seq_len, input_size)
    y, h = model(x)
    
    assert y.shape == (batch_size, seq_len, hidden_size)
    assert h.shape == (1, batch_size, hidden_size)

@pytest.mark.unit
def test_rnn_module_bidirectional():
    input_size = 10
    hidden_size = 20
    batch_size = 2
    seq_len = 5
    
    model = RNNModule(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)
    x = torch.randn(batch_size, seq_len, input_size)
    y, h = model(x)
    
    assert y.shape == (batch_size, seq_len, hidden_size * 2)
    assert h.shape == (2, batch_size, hidden_size)

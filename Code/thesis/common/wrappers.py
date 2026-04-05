"""Thin wrappers so RNN cores expose (batch, n_classes) logits."""

from __future__ import annotations

import torch
import torch.nn as nn


class RNNClassifier(nn.Module):
    """Wraps LSTMModule / GRUModule: last timestep -> Linear -> logits."""

    def __init__(self, rnn: nn.Module, n_classes: int):
        super().__init__()
        self.rnn = rnn
        hidden = rnn.hidden_size * (2 if getattr(rnn, "bidirectional", False) else 1)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, feat] batch_first expected from collate
        out, _ = self.rnn(x)
        if getattr(self.rnn, "batch_first", False):
            last = out[:, -1, :]
        else:
            last = out[-1, :, :]
        return self.head(last)

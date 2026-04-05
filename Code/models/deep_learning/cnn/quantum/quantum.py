"""
Quantum Convolutional Layer - stub for future implementation.
Reference: docs/deep-learning/cnn/cnn.md Part IV
"""

import torch
import torch.nn as nn
from typing import Union
from .....models.utils import DLModule


__all__ = ["QCNNLayer"]


class QCNNLayer(DLModule):
    """
    Stub: Quantum Convolutional Layer. Requires quantum hardware.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 n_qubits: int = 4,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.linear = nn.Linear(in_channels, out_channels, **self.factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.view(B, C, -1).mean(dim=-1)
        return self.linear(x)

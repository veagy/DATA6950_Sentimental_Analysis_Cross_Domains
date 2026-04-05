"""
TopologyNet - stub for persistent homology.
Reference: docs/deep-learning/cnn/cnn.md Part IV
"""

import torch
import torch.nn as nn
from typing import Union
from .....models.utils import DLModule


__all__ = ["TopologyNetLayer"]


class TopologyNetLayer(DLModule):
    """
    Stub: TopologyNet with persistent homology.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.proj = nn.Linear(in_channels, out_channels, **self.factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.flatten(2).mean(dim=-1)
        return self.proj(x)

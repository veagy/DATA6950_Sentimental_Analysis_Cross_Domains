"""
Event-Based Conv - stub for neuromorphic vision.
Reference: docs/deep-learning/cnn/cnn.md Part IV
"""

import torch
import torch.nn as nn
from typing import Union
from .....models.utils import DLModule


__all__ = ["EventConvLayer"]


class EventConvLayer(DLModule):
    """
    Stub: Event-based convolution for spike/event tensors.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, **self.factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

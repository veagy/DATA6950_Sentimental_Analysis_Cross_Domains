"""
SEAFEC: Spatial-Edge Adaptive Feature Enhancement Convolution.
Reference: docs/deep-learning/cnn/cnn.md Part IV
"""

import torch
import torch.nn as nn
from typing import Union
from .....models.utils import DLModule


__all__ = ["SEAFECLayer"]


class SEAFECLayer(DLModule):
    """
    Dual-branch: SCARF (spatial-channel attention) + MEFE (edge enhancement).
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 scarf_reduction: int = 4,
                 mefe_kernel: int = 3,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}
        mid = max(in_channels // scarf_reduction, 1)
        self.scarf = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels),
            nn.Sigmoid()
        )
        self.mefe = nn.Conv2d(in_channels, in_channels, mefe_kernel, padding=mefe_kernel // 2, **self.factory_kwargs)
        self.conv = nn.Conv2d(in_channels * 2, out_channels, kernel_size, padding=kernel_size // 2, **self.factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.scarf(x).view(x.shape[0], -1, 1, 1)
        scarf_out = x * a
        mefe_out = self.mefe(x)
        cat = torch.cat([scarf_out, mefe_out], dim=1)
        return self.conv(cat)

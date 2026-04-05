"""
Shiftwise convolution: mimics large kernels with small shift-based operations.
Reference: docs/deep-learning/cnn/cnn.md - CVPR 2025 Shiftwise
"""

import torch
import torch.nn as nn
from typing import Union
from .....models.utils import DLModule


__all__ = ["ShiftwiseConvLayer"]


class ShiftwiseConvLayer(DLModule):
    """
    Shiftwise convolution: granularity extraction + multi-path fusion.
    Mimics large kernels (e.g. 31x31) with small 3x3 kernels via shift operations.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 shift_groups: int = 4,
                 dimensionality: Union[int, float] = 2,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.shift_groups = min(shift_groups, in_channels)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        s = max(1, in_channels // shift_groups)
        self.split_size = s
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, **self.factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        s = self.split_size
        shifts = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
        parts = []
        for i, (dx, dy) in enumerate(shifts):
            if (i + 1) * s <= C:
                xi = x[:, i * s:(i + 1) * s]
                parts.append(torch.roll(xi, (dx, dy), (-2, -1)))
        if len(parts) * s < C:
            parts.append(x[:, len(parts) * s:])
        x = torch.cat(parts, dim=1)
        return self.conv(x)

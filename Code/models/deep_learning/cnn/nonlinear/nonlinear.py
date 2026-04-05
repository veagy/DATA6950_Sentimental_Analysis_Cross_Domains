"""
Nonlinear Volterra convolution operators.
Reference: docs/deep-learning/cnn/cnn.md Part IV - Higher-Order Volterra
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from .....models.utils import DLModule


__all__ = [
    "VolterraConvLayer",
]


class VolterraConvLayer(DLModule):
    """
    Higher-order Volterra convolution: y = h0 + sum(h1*x) + sum(h2*x*x) + ...
    Captures multiplicative pixel interactions.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 order: int = 2,
                 stride: int = 1,
                 padding: Union[int, None] = None,
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
        self.order = min(max(order, 2), 3)
        self.stride = stride
        self.padding = padding if padding is not None else kernel_size // 2
        self.dilation = dilation
        self.groups = groups
        k = kernel_size
        self.conv1 = nn.Conv2d(in_channels, out_channels, k, stride, self.padding, dilation, groups, bias, **self.factory_kwargs)
        if self.order >= 2:
            self.unfold = nn.Unfold(k, dilation=dilation, padding=self.padding, stride=stride)
            self.conv2 = nn.Conv2d(in_channels * k * k, out_channels, 1, **self.factory_kwargs)
        else:
            self.unfold = None
            self.conv2 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        if self.order >= 2 and self.conv2 is not None:
            B, C, H, W = x.shape
            k = self.kernel_size
            u = self.unfold(x)
            u2 = (u * u).view(B, C * k * k, -1)
            h_out = (H + 2 * self.padding - k) // self.stride + 1
            w_out = (W + 2 * self.padding - k) // self.stride + 1
            u2 = u2.view(B, C * k * k, h_out, w_out)
            out = out + self.conv2(u2)
        return out

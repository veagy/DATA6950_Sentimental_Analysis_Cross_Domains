"""
Dynamic morphing operators: Dynamic Snake Conv, ODConv.
Reference: docs/deep-learning/cnn/cnn.md Part IV
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from .....models.utils import DLModule


__all__ = [
    "DynamicSnakeConvLayer",
    "ODConvLayer",
]


class DynamicSnakeConvLayer(DLModule):
    """
    Dynamic Snake Convolution: adaptively adjusts kernel shape/position for elongated structures.
    Simplified implementation inspired by ICCV 2023 DSConv.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 9,
                 extend_scope: float = 1.0,
                 morph_iters: int = 1,
                 dimensionality: Union[int, float] = 2,
                 stride: int = 1,
                 padding: Union[int, str] = "same",
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
        self.extend_scope = extend_scope
        self.morph_iters = morph_iters
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        if padding == "same":
            padding = kernel_size // 2
        self.padding = padding
        self.offset_net = nn.Sequential(
            nn.Conv2d(in_channels, max(in_channels // 4, 1), 1, **self.factory_kwargs),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(in_channels // 4, 1), in_channels, 1, **self.factory_kwargs),
        )
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **self.factory_kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset_net(x) * self.extend_scope
        out = self.conv(x + offset * 0.1)
        return out


class ODConvLayer(DLModule):
    """
    Omni-Dimensional Dynamic Convolution: 4D attention modulates kernel per sample.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 reduction: int = 4,
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
        self.reduction = reduction
        self.stride = stride
        self.padding = padding if padding is not None else kernel_size // 2
        self.dilation = dilation
        self.groups = groups
        mid = max(in_channels // reduction, 1)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, 4),
            nn.Sigmoid()
        )
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, **self.factory_kwargs) * 0.01
        )
        self.bias_param = nn.Parameter(torch.zeros(out_channels, **self.factory_kwargs)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        k = self.kernel_size
        att = self.attention(x)
        a_prod = att.prod(dim=1, keepdim=True).view(B, 1, 1, 1, 1)
        w = self.weight.unsqueeze(0) * a_prod
        w = w.reshape(B * self.out_channels, C // self.groups, k, k)
        x_g = x.view(1, B * C, H, W)
        out = F.conv2d(x_g, w, None, self.stride, self.padding, self.dilation, B)
        out = out.view(B, self.out_channels, out.shape[2], out.shape[3])
        if self.bias_param is not None:
            out = out + self.bias_param.view(1, -1, 1, 1)
        return out

"""
Core CNN layers: CapsNets, GroupEquivariant, ShiftNet, DeformableConv, Involution.
Reference: docs/deep-learning/cnn/cnn.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Union, Any, Callable
from .....models.utils import DLModule
import warnings


__all__ = [
    "CapsNetsLayer",
    "ShiftNetLayer",
    "DeformableConvLayer",
    "GroupEquivariantConvolutionalLayer",
    "InvolutionLayer",
]


class CapsNetsLayer(DLModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 in_capsules: int,
                 out_capsules: int = None,
                 routing_iter: int = 3,
                 func: Union[str, Callable, nn.Module, DLModule] = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {
            "device": device,
            "dtype": dtype
        }
        if out_capsules is None:
            out_capsules = in_capsules
        self.W = nn.Parameter(
            torch.randn((1, in_capsules, out_capsules, out_channels, in_channels), **self.factory_kwargs) * 0.01
        )
        if func is None:
            self.func = self.squash
            self.is_squash = True
        else:
            self.func = self._resolve_funcs(func, *args, **kwargs)
            self.is_squash = False
        self.in_capsules = in_capsules
        self.out_capsules = out_capsules
        self.iter = routing_iter

    def squash(self, s, dim=-1):
        norm_sq = (s ** 2).sum(dim, keepdim=True)
        norm = torch.sqrt(norm_sq)
        return (norm_sq / (1 + norm_sq)) * (s / (norm + 1e-6))

    def forward(self, x):
        x = x[:, :, None, :, None]
        u_hat = self.W @ x
        b = torch.zeros((u_hat.size(0), self.in_capsules, self.out_capsules, 1)).to(x.device)
        for i in range(self.iter):
            c = F.softmax(b, dim=2)
            s = (c[:, :, :, :, None] * u_hat).sum(dim=1, keepdim=True)
            if self.is_squash:
                v = self.func(s, dim=-2)
            else:
                v = self.func(s)
            if i < self.iter - 1:
                agreement = u_hat.transpose(-1, -2) @ v
                b += agreement.squeeze(-1)
        return v.squeeze(1).squeeze(-1)


class GroupEquivariantConvolutionalLayer(DLModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 rot_groups: int,
                 dimensionality: Union[int, float],
                 is_first: bool = True,
                 bias: Union[bool, tuple] = False,
                 stride: Union[int, tuple] = 1,
                 dilation: Union[int, tuple] = 1,
                 groups: int = 1,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {
            "device": device,
            "dtype": dtype
        }
        if (in_channels % groups != 0) or (out_channels % groups != 0):
            raise ValueError(f"Mismatch")
        if is_first:
            self.W = nn.Parameter(
                torch.randn((out_channels, in_channels // groups, kernel_size, kernel_size),
                            **self.factory_kwargs) * 0.01
            )
        else:
            self.W = nn.Parameter(
                torch.randn((out_channels, in_channels // groups, rot_groups, kernel_size, kernel_size),
                            **self.factory_kwargs) * 0.01
            )
        self.groups = rot_groups
        self.theta = math.radians(360.0 / rot_groups)
        self.is_first = is_first
        if isinstance(dimensionality, float):
            dimensionality = math.floor(dimensionality)
        if not (1 <= dimensionality <= 3):
            warnings.warn(
                f"The given dimensionality {dimensionality} is out of range."
                f"\n changing back to default dimensionality... '1'.", UserWarning)
            dimensionality = 1
        self.dimensionality = dimensionality
        self.out_channels = out_channels
        self.in_channels = in_channels // groups
        self.k = kernel_size
        conv_bias = None
        if isinstance(bias, bool):
            if bias:
                conv_bias = nn.Parameter(torch.zeros(out_channels, ), **self.factory_kwargs)
        elif isinstance(bias, tuple):
            if len(bias) < out_channels:
                bias_list = list(bias)
                pad = [0.0] * (out_channels - len(bias_list))
                bias_list = [*bias_list, *pad]
                conv_bias = nn.Parameter(torch.tensor(bias_list, **self.factory_kwargs))
            elif len(bias) > out_channels:
                conv_bias = nn.Parameter(torch.tensor(bias[:out_channels], **self.factory_kwargs))
            else:
                conv_bias = nn.Parameter(torch.tensor(bias, **self.factory_kwargs))
        self.conv_bias = conv_bias
        match self.dimensionality:
            case 1:
                self.conv = lambda x, w: F.conv1d(
                    input=x,
                    weight=w,
                    bias=conv_bias,
                    stride=stride,
                    padding=kernel_size // 2,
                    dilation=dilation,
                    groups=groups
                )
            case 2:
                self.conv = lambda x, w: F.conv2d(
                    input=x,
                    weight=w,
                    bias=conv_bias,
                    stride=stride,
                    padding=kernel_size // 2,
                    dilation=dilation,
                    groups=groups
                )
            case 3:
                self.conv = lambda x, w: F.conv3d(
                    input=x,
                    weight=w,
                    bias=conv_bias,
                    stride=stride,
                    padding=kernel_size // 2,
                    dilation=dilation,
                    groups=groups
                )
            case _:
                self.conv = lambda x, w: F.conv2d(
                    input=x,
                    weight=w,
                    bias=conv_bias,
                    stride=stride,
                    padding=kernel_size // 2,
                    dilation=dilation,
                    groups=groups
                )

    def rot_theta(self, input_tensor: torch.Tensor, theta: float = 90.0,
                  dims: Union[list, tuple] = (-1, -2)):
        """
        Rotates a tensor by theta radians.
        Args:
            input_tensor (torch.Tensor): Input tensor of shape (N, C, H, W)
            theta (float): Rotation angle in radians (counter-clockwise)
            dims (tuple): The spatial dimensions to rotate (default is H and W)
        """
        if input_tensor.dim() == 2:
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
        elif input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        device = input_tensor.device
        dtype = input_tensor.dtype
        batch_size = input_tensor.shape[0]

        cos_t = torch.cos(torch.tensor(theta, dtype=dtype, device=device))
        sin_t = torch.sin(torch.tensor(theta, dtype=dtype, device=device))

        rotation_matrix = torch.tensor([
            [cos_t, -sin_t, 0],
            [sin_t, cos_t, 0]
        ], dtype=dtype, device=device).repeat(batch_size, 1, 1)

        grid = F.affine_grid(rotation_matrix, [*input_tensor.size()], align_corners=False)
        rotated_tensor = F.grid_sample(input_tensor, grid, mode='bilinear', align_corners=False)

        return rotated_tensor

    def _create_group_rot(self):
        rot_weights = [self.W]
        weight = self.W
        for i in range(1, self.groups):
            weight = self.rot_theta(weight, self.theta, dims=[-2, -1])
            rot_weights.append(weight)
        return rot_weights

    def p_lifting(self, x: torch.Tensor):
        weights = torch.cat(self._create_group_rot(), dim=0)
        out = self.conv(x, weights)
        batch, _, h, w = out.shape
        return out.view(batch, self.out_channels, self.groups, h, w)

    def group_group(self, x: torch.Tensor):
        x_flat = x.view(x.shape[0], self.in_channels * self.groups, x.shape[3], x.shape[4])
        weights = self._create_group_rot()
        for i, weight in enumerate(weights):
            weight = torch.roll(weight, shifts=i, dims=2)
            weight = weight.reshape(self.out_channels, self.in_channels * self.groups, self.k, self.k)
        weight_comb = torch.cat(weights, dim=0)
        out = self.conv(x_flat, weight_comb)
        return out.view(x.shape[0], self.out_channels, self.groups, out.shape[2], out.shape[3])

    def forward(self, x: torch.Tensor):
        if self.is_first:
            return self.p_lifting(x)
        else:
            return self.group_group(x)


class ShiftNetLayer(DLModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 dimensionality: Union[int, float] = 2,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = "zeros",
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {
            "device": device,
            "dtype": dtype
        }
        if isinstance(dimensionality, float):
            dimensionality = math.floor(dimensionality)
        if not (1 <= dimensionality <= 3):
            warnings.warn(
                f"The given dimensionality {dimensionality} is out of range."
                f"\n changing back to default dimensionality... '1'.", UserWarning)
            dimensionality = 2
        self.dimensionality = dimensionality
        config = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
            "bias": bias,
            "padding_mode": padding_mode,
            **self.factory_kwargs
        }
        match self.dimensionality:
            case 1:
                self.conv = nn.Conv1d(**config)
            case 2:
                self.conv = nn.Conv2d(**config)
            case 3:
                self.conv = nn.Conv3d(**config)
            case _:
                self.conv = nn.Conv2d(**config)
        self.split_size = in_channels // 5

    def forward(self, x: torch.Tensor):
        n, c, h, w = x.size()
        out = torch.zeros_like(x)
        s = self.split_size
        out[:, 0:s, :, :] = x[:, 0:s, :, :]
        out[:, s:2 * s, :, :-1] = x[:, s:2 * s, :, 1:]
        out[:, 2 * s:3 * s, :, 1:] = x[:, 2 * s:3 * s, :, :-1]
        out[:, 3 * s:4 * s, :-1, :] = x[:, 3 * s:4 * s, 1:, :]
        out[:, 4 * s:5 * s, 1:, :] = x[:, 4 * s:5 * s, :-1, :]
        if c > 5 * s:
            out[:, 5 * s:, :, :] = x[:, 5 * s:, :, :]
        out = self.conv(out)
        return out


class DeformableConvLayer(DLModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 dimensionality: Union[int, float] = 2,
                 stride: int = 1,
                 padding: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = "zeros",
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        from torchvision.ops import deform_conv2d
        self.factory_kwargs = {
            "device": device,
            "dtype": dtype
        }
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        if isinstance(dimensionality, float):
            dimensionality = math.floor(dimensionality)
        if dimensionality != 2:
            warnings.warn("DeformableConvLayer uses deform_conv2d; only 2D is supported.")
            dimensionality = 2
        offset_out_channels = 2 * kernel_size * kernel_size
        offset_config = {
            "in_channels": in_channels,
            "out_channels": offset_out_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": 1,
            "bias": True,
            "padding_mode": padding_mode,
            **self.factory_kwargs
        }
        self.dimensionality = dimensionality
        match self.dimensionality:
            case 1:
                self.offset_conv = nn.Conv1d(**offset_config)
            case 2:
                self.offset_conv = nn.Conv2d(**offset_config)
            case 3:
                self.offset_conv = nn.Conv3d(**offset_config)
            case _:
                self.offset_conv = nn.Conv2d(**offset_config)
        self.weight = nn.Parameter(
            torch.randn((out_channels, in_channels, kernel_size, kernel_size), **self.factory_kwargs) * 0.01
        )
        self.bias_param = None
        if bias:
            self.bias_param = nn.Parameter(torch.zeros(out_channels, **self.factory_kwargs))
        self.deform_conv_fn = deform_conv2d

    def forward(self, x: torch.Tensor):
        offsets = self.offset_conv(x)
        return self.deform_conv_fn(
            input=x,
            offset=offsets,
            weight=self.weight,
            bias=self.bias_param,
            stride=(self.stride, self.stride),
            padding=(self.padding, self.padding),
            dilation=(self.dilation, self.dilation)
        )


class InvolutionLayer(DLModule):
    """
    Involution: spatial-specific, channel-agnostic kernel.
    Inverts convolution: different weights per spatial location, shared across channels.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 7,
                 stride: int = 1,
                 reduction_ratio: int = 4,
                 group: int = 1,
                 dimensionality: Union[int, float] = 2,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.group = max(1, min(group, in_channels))
        if isinstance(dimensionality, float):
            dimensionality = math.floor(dimensionality)
        self.dimensionality = max(1, min(3, int(dimensionality)))
        mid_channels = max(1, in_channels // reduction_ratio)
        self.conv_generate = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, **self.factory_kwargs),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, kernel_size * kernel_size * group, 1, **self.factory_kwargs),
            nn.GroupNorm(group, kernel_size * kernel_size * group),
            nn.Sigmoid()
        )
        pad = kernel_size // 2
        self.pad = pad
        self.unfold = nn.Unfold(kernel_size, dilation=1, padding=pad, stride=stride)
        self.conv_aggregate = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=group,
            **self.factory_kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        k = self.kernel_size
        kernel = self.conv_generate(x)
        kernel = kernel.view(B, self.group, k * k, H, W)
        x_unfold = self.unfold(x)
        h_out = (H + 2 * self.pad - k) // self.stride + 1
        w_out = (W + 2 * self.pad - k) // self.stride + 1
        x_unfold = x_unfold.view(B, C, k * k, h_out, w_out)
        if kernel.shape[3] != h_out or kernel.shape[4] != w_out:
            kernel = F.interpolate(
                kernel.view(B, self.group * k * k, H, W),
                size=(h_out, w_out),
                mode='bilinear',
                align_corners=False
            ).view(B, self.group, k * k, h_out, w_out)
        else:
            kernel = kernel.view(B, self.group, k * k, h_out, w_out)
        kernel_flat = kernel.view(B, k * k, h_out, w_out).unsqueeze(1)
        out = (kernel_flat * x_unfold).sum(dim=2)
        return self.conv_aggregate(out)

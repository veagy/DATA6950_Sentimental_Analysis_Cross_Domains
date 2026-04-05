"""
Capsule-ConvKAN: B-spline weights instead of linear W in capsules.
Reference: docs/deep-learning/cnn/cnn.md Part IV
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from .....models.utils import DLModule


__all__ = [
    "CapsuleConvKANLayer",
]


class CapsuleConvKANLayer(DLModule):
    """
    Capsule layer with B-spline (KAN-style) weights instead of linear W.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 in_capsules: int,
                 out_capsules: int,
                 kernel_size: int = 3,
                 num_spline_basis: int = 5,
                 routing_iter: int = 3,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.in_capsules = in_capsules
        self.out_capsules = out_capsules
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.iter = routing_iter
        self.num_basis = num_spline_basis
        self.spline_weight = nn.Parameter(
            torch.randn(1, in_capsules, out_capsules, out_channels, in_channels, num_spline_basis, **self.factory_kwargs) * 0.01
        )

    def _spline_basis(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(-1, 1)
        k = self.num_basis
        t = torch.linspace(-1, 1, k + 1, device=x.device, dtype=x.dtype)
        basis = []
        for i in range(k):
            left = (x >= t[i]).float()
            right = (x < t[i + 1]).float()
            basis.append(left * right)
        return torch.stack(basis, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        basis = self._spline_basis(x)
        basis = basis.unsqueeze(2).unsqueeze(3)
        w = (self.spline_weight * basis).sum(dim=-1)
        x = x[:, :, None, :, None]
        u_hat = w @ x
        b = torch.zeros((u_hat.size(0), self.in_capsules, self.out_capsules, 1), device=x.device)
        for i in range(self.iter):
            c = F.softmax(b, dim=2)
            s = (c[:, :, :, :, None] * u_hat).sum(dim=1, keepdim=True)
            norm_sq = (s ** 2).sum(dim=-2, keepdim=True)
            norm = torch.sqrt(norm_sq + 1e-6)
            v = (norm_sq / (1 + norm_sq)) * (s / norm)
            if i < self.iter - 1:
                agreement = u_hat.transpose(-1, -2) @ v
                b = b + agreement.squeeze(-1)
        return v.squeeze(1).squeeze(-1)

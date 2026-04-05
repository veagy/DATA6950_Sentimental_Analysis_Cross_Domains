import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Union, Any, Optional, Callable

from .....models.deep_learning.activations.ActivationFunction import Activation
from .....models.utils import DLModule
from .....models.deep_learning.activations.Complex.complex_ import ComplexLinear


__all__ = [
    "ComplexLinear",
    "KANLayer",
    "SlimLinear"
]


class KANLayer(DLModule):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 num_intervals: int = 5,
                 spline_order: int = 3,
                 func: Union[str, Callable, nn.Module, DLModule] = "silu",
                 bias: bool = True,
                 grid_strategy: str = "clamped",
                 grid_momentum: float = 0.9,
                 device: str = None,
                 dtype: torch.dtype = None,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {
            "device": device,
            "dtype": dtype
        }
        self.G = num_intervals
        self.k = spline_order
        self.in_features = in_features
        self.out_features = out_features
        self.grid_momentum = grid_momentum
        self.strategy = grid_strategy.lower()

        self.W_func = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            **self.factory_kwargs
        )

        # Grid initialization
        # self.grid shape: (in_features, G + 2*k + 1)
        grid = self.generate_grid(
            torch.ones(in_features, **self.factory_kwargs) * -1,
            torch.ones(in_features, **self.factory_kwargs)
        )
        self.register_buffer("grid", grid)

        self.spline_weight = nn.Parameter(
            torch.randn((out_features, in_features, self.G + self.k), **self.factory_kwargs) * 0.1
        )
        self.spline_scalar = nn.Parameter(
            torch.randn((out_features, in_features), **self.factory_kwargs) * 0.1
        )

        self.func = self._resolve_funcs(func, *args, **kwargs)
        self.bias_vector = nn.Parameter(torch.zeros(out_features, **self.factory_kwargs)) if bias else None
        self.bias = bias
        self.kwargs = kwargs

    def _resolve_func(self, func, *args, **kwargs):
        if isinstance(func, str):
            if func.lower() == "silu":
                return nn.SiLU()
        return super()._resolve_func(func, *args, **kwargs)

    def generate_grid(self, x_min: torch.Tensor, x_max: torch.Tensor, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        if torch.any(torch.isclose(x_min, x_max, atol=1e-5)):
            mask = torch.isclose(x_min, x_max, atol=1e-5)
            x_min[mask] -= 0.1
            x_max[mask] += 0.1

        t = torch.linspace(0, 1, self.G + 1, device=x_min.device, dtype=x_min.dtype)

        if self.strategy == "quantile" and x is not None:
            q = torch.linspace(0, 1, self.G + 1, device=x.device, dtype=x.dtype)
            grid_core = torch.quantile(x, q, dim=0).T

            eps = torch.linspace(0, 1e-5, self.G + 1, device=x.device, dtype=x.dtype)
            grid_core = grid_core + eps.unsqueeze(0)

        elif self.strategy == "exp_decay":
            gamma = abs(self.kwargs.get("gamma", 5.0))
            t_warped = torch.sign(t - 0.5) * (1 - torch.exp(-gamma * torch.abs(t - 0.5)))
            t_warped = (t_warped - t_warped.min()) / (t_warped.max() - t_warped.min())
            grid_core = x_min.unsqueeze(1) + (x_max - x_min).unsqueeze(1) * t_warped.unsqueeze(0)

        elif self.strategy == "gaussian":
            gamma = abs(self.kwargs.get("gamma", 5.0))
            t_warped = torch.erf((t - 0.5) * gamma)
            t_warped = (t_warped - t_warped.min()) / (t_warped.max() - t_warped.min())
            grid_core = x_min.unsqueeze(1) + (x_max - x_min).unsqueeze(1) * t_warped.unsqueeze(0)

        else:
            grid_core = x_min.unsqueeze(1) + (x_max - x_min).unsqueeze(1) * t.unsqueeze(0)

        step = (grid_core[:, -1] - grid_core[:, 0]) / self.G

        pad_range_left = torch.arange(-self.k, 0, device=x_min.device, dtype=x_min.dtype)
        left_pads = grid_core[:, 0:1] + step.unsqueeze(1) * pad_range_left.unsqueeze(0)

        pad_range_right = torch.arange(1, self.k + 1, device=x_min.device, dtype=x_min.dtype)
        right_pads = grid_core[:, -1:] + step.unsqueeze(1) * pad_range_right.unsqueeze(0)

        grid = torch.cat([left_pads, grid_core, right_pads], dim=1)
        return grid

    def compute_b_splines(self, x: torch.Tensor, grid: torch.Tensor = None) -> torch.Tensor:
        if grid is None:
            grid = self.grid

        x = x.unsqueeze(-1)
        grid = grid.unsqueeze(0)

        value = ((x >= grid[:, :, :-1]) & (x < grid[:, :, 1:])).to(x.dtype)

        for p in range(1, self.k + 1):
            denom1 = grid[:, :, p:-1] - grid[:, :, :-(p + 1)]
            denom2 = grid[:, :, p + 1:] - grid[:, :, 1:-p]

            term1 = (x - grid[:, :, :-(p + 1)]) / (denom1 + 1e-8) * value[..., :-1]
            term2 = (grid[:, :, p + 1:] - x) / (denom2 + 1e-8) * value[..., 1:]
            value = term1 + term2

        return value

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin: float = 0.01):
        # x: (B, in_features)

        batch_size = x.size(0)
        x = x.view(-1, self.in_features)

        # 1. Update grid points based on x distribution
        if batch_size > 100:
            mi = torch.quantile(x, 0.01, dim=0)
            ma = torch.quantile(x, 0.99, dim=0)
        else:
            mi = x.min(dim=0).values - margin
            ma = x.max(dim=0).values + margin

        if self.strategy == "quantile":
            new_grid_core = self.generate_grid(mi, ma, x)
        else:
            new_grid_core = self.generate_grid(mi, ma)

        updated_grid = (1 - self.grid_momentum) * new_grid_core + self.grid_momentum * self.grid

        # 2. Least Squares Projection
        old_grid = self.grid.clone()
        self.grid.copy_(updated_grid)

        old_basis = self.compute_b_splines(x, old_grid)
        y_old = torch.einsum("bik,oik->boi", old_basis, self.spline_weight)

        new_basis = self.compute_b_splines(x, self.grid)

        A = new_basis.permute(1, 0, 2)
        Y = y_old.permute(2, 0, 1)

        AtA = torch.bmm(A.transpose(1, 2), A)
        AtY = torch.bmm(A.transpose(1, 2), Y)

        reg = torch.eye(AtA.shape[1], device=AtA.device, dtype=AtA.dtype).unsqueeze(0) * 1e-4

        try:
            sol = torch.linalg.solve(AtA + reg, AtY)
        except RuntimeError:
            sol = torch.linalg.lstsq(AtA + reg, AtY).solution

        self.spline_weight.data.copy_(sol.permute(2, 0, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grid_min = self.grid[:, self.k].view(1, -1)
        grid_max = self.grid[:, -(self.k + 1)].view(1, -1)

        x_clamped = torch.clamp(x, grid_min, grid_max)
        base_chk = self.W_func(self.func(x))

        basis = self.compute_b_splines(x_clamped, grid=self.grid)
        spline_out_per_feature = torch.einsum("...ik,oik->...oi", basis, self.spline_weight)

        spline_out_scaled = spline_out_per_feature * self.spline_scalar.unsqueeze(0)
        spline_total = spline_out_scaled.sum(dim=-1)

        out = base_chk + spline_total

        if self.bias_vector is not None:
            out = out + self.bias_vector

        return out


class SlimLinear(DLModule):
    def __init__(self,
                 max_in_features: int,
                 max_out_features: int,
                 bias: bool = True,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 ):
        super().__init__()
        self.factory_args = {
            "device": device,
            "dtype": dtype
        }
        self.max_in_features = max_in_features
        self.max_out_features = max_out_features
        self.weight = nn.Parameter(
            torch.randn((max_out_features, max_in_features), **self.factory_args)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.zeros((max_out_features,), **self.factory_args)
            )
        else:
            self.bias = None
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform(self.weight, a=5 ** 0.5)

    def forward(self, x, out_features=None):
        if out_features is None:
            out_features = self.max_out_features
        else:
            if out_features > self.max_out_features:
                raise RuntimeError(f"Tensor Mismatch: mat1 and mat2 shapes cannot be multiplied.\n"
                                   f"Expected within {self.max_out_features} but got {out_features}.")
        in_features = x.size(-1)
        if in_features > self.max_in_features:
            raise RuntimeError(f"Tensor Mismatch: mat1 and mat2 shapes cannot be multiplied.\n"
                               f"Expected within {self.max_in_features} but got {in_features}.")
        weight = self.weight[:out_features, :in_features]
        bias = self.bias[:out_features] if self.bias is not None else None
        return F.linear(x, weight, bias)
"""Efficient RNN cells: RWKVCell, MambaCell."""
import torch
import torch.nn as nn
from typing import Optional, Union, Callable, List
from .....models.utils import DLModule


class RWKVCell(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 func: Union[str, Callable, nn.Module],
                 bias: bool = True,
                 decay: float = 0.0,
                 mu: Optional[List[float]] = None,
                 proj_size: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {
            "device": device,
            "dtype": dtype
        }
        in_kwargs = {
            "in_features": input_size,
            "out_features": hidden_size,
            "bias": bias,
            **self.factory_kwargs
        }
        self.Wk = nn.Linear(**in_kwargs)
        self.Wv = nn.Linear(**in_kwargs)
        self.Wr = nn.Linear(**in_kwargs)

        if mu is None:
            mu_k, mu_v, mu_r = 0.5, 0.5, 0.5
        else:
            if len(mu) > 3:
                mu = mu[:3]
            elif len(mu) < 3:
                try:
                    mean = sum(mu) / len(mu)
                except ZeroDivisionError:
                    mean = 0.5
                mu = mu + [mean] * (3 - len(mu))
            mu_k, mu_v, mu_r = mu[0], mu[1] if len(mu) > 1 else mu[0], mu[2] if len(mu) > 2 else mu[0]
        self.mu_k = nn.Parameter(torch.tensor(mu_k, **self.factory_kwargs))
        self.mu_v = nn.Parameter(torch.tensor(mu_v, **self.factory_kwargs))
        self.mu_r = nn.Parameter(torch.tensor(mu_r, **self.factory_kwargs))

        self.w = nn.Parameter(torch.full((hidden_size,), decay, **self.factory_kwargs))
        self.u = nn.Parameter(torch.zeros((hidden_size,), **self.factory_kwargs))

        self.Wo = nn.Linear(
            in_features=hidden_size,
            out_features=proj_size,
            bias=bias,
            **self.factory_kwargs
        ) if proj_size is not None else None
        self.proj = False if proj_size is None else True
        self.func = self._resolve_funcs(func, *args, **kwargs)

        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, a_prev: Optional[torch.Tensor] = None, b_prev: Optional[torch.Tensor] = None,
                x_prev: Optional[torch.Tensor] = None):
        if a_prev is None:
            a_prev = torch.zeros((x.size(0), self.hidden_size), **self.factory_kwargs)
        if b_prev is None:
            b_prev = torch.zeros((x.size(0), self.hidden_size), **self.factory_kwargs)
        if x_prev is None:
            x_prev = torch.zeros_like(x, **self.factory_kwargs)

        x_k = (self.mu_k * x) + ((1 - self.mu_k) * x_prev)
        x_v = (self.mu_v * x) + ((1 - self.mu_v) * x_prev)
        x_r = (self.mu_r * x) + ((1 - self.mu_r) * x_prev)

        k_t = self.Wk(x_k)
        v_t = self.Wv(x_v)
        r_t = self.func(self.Wr(x_r))

        w = torch.exp(-self.w)
        k_exp = torch.exp(k_t)

        a_t = (w * a_prev) + (k_exp * v_t)
        b_t = (w * b_prev) + k_exp

        u = torch.exp(self.u + k_t)
        wkv_t = (a_prev + (u * v_t)) / (b_prev + u)

        out = r_t * wkv_t

        if self.proj:
            out = self.Wo(out)
        return out, a_t, b_t, x


class MambaCell(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 rank: Union[str, int] = "auto",
                 bias: bool = True,
                 func: Union[str, Callable, nn.Module] = None,
                 proj_size: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {
            "device": device,
            "dtype": dtype
        }
        self.A_log = nn.Parameter(torch.log(torch.arange(1, hidden_size + 1, **self.factory_kwargs).float()))
        self.D = nn.Parameter(torch.ones((hidden_size,), **self.factory_kwargs))

        rank = hidden_size // 16 if rank == "auto" else rank
        self.x_proj = nn.Linear(
            in_features=input_size,
            out_features=rank + hidden_size * 2,
            bias=bias,
            **self.factory_kwargs
        )
        self.dt_proj = nn.Linear(
            in_features=rank,
            out_features=hidden_size,
            bias=bias,
            **self.factory_kwargs
        )
        self.Wo = nn.Linear(
            in_features=hidden_size,
            out_features=proj_size,
            bias=bias,
            **self.factory_kwargs
        ) if proj_size is not None else None
        self.proj = False if proj_size is None else True
        self.func = self._resolve_funcs(func, *args, **kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rank = rank

        if input_size != hidden_size:
            self.u_proj = nn.Linear(input_size, hidden_size, bias=bias, **self.factory_kwargs)
        else:
            self.u_proj = None

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None):
        if h_prev is None:
            h_prev = torch.zeros((x.size(0), self.hidden_size), **self.factory_kwargs)

        u = self.u_proj(x) if self.u_proj is not None else x

        proj = self.x_proj(x)
        dt, B, C = torch.split(proj, [self.rank, self.hidden_size, self.hidden_size], dim=-1)

        dt = self.func(self.dt_proj(dt))

        A = -torch.exp(self.A_log)
        A_bar = torch.exp(dt * A)

        B_bar = dt * B

        h_t = A_bar * h_prev + B_bar * u

        y = C * h_t + self.D * u

        if self.proj:
            y = self.Wo(y)
        return y, h_t

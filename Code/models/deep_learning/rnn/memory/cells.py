"""Memory RNN cells: ESNCell, NTMCell, HopfieldNetworkCell."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Callable, List, Tuple, Dict
from .....models.utils import DLModule
from ..base import LSTMCell


class ESNCell(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 func: Union[str, Callable, nn.Module],
                 bias: bool = True,
                 leaking_rate: float = 0.8,
                 spectral_radius: float = 0.9,
                 proj_size: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {
            "device": device,
            "dtype": dtype
        }
        self.Win = nn.Linear(
            in_features=input_size,
            out_features=hidden_size,
            bias=bias,
            **self.factory_kwargs
        )
        self.Wres = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=bias,
            **self.factory_kwargs
        )
        if proj_size is not None:
            self.Wo = nn.Linear(
                in_features=hidden_size,
                out_features=proj_size,
                bias=bias,
                **self.factory_kwargs
            )
        self.alpha = leaking_rate
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.proj = False if proj_size is None else True
        self.reset_params(spectral_radius)

        self.func = self._resolve_funcs(func, *args, **kwargs)

    def reset_params(self, spectral_radius):
        nn.init.uniform_(self.Win.weight, -1, 1)
        nn.init.uniform_(self.Wres.weight, -1, 1)
        if self.bias:
            nn.init.zeros_(self.Win.bias)
            nn.init.zeros_(self.Wres.bias)
        with torch.no_grad():
            values = torch.linalg.eigvals(self.Wres.weight)
            max_eigen = torch.max(torch.abs(values))
            self.Wres.weight.mul_(spectral_radius / max_eigen)

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if h_prev is None:
            h_prev = torch.zeros((x.size(0), self.hidden_size), **self.factory_kwargs)
        h_bar = self.Win(x) + self.Wres(h_prev)
        h_state = ((1 - self.alpha) * h_prev) + (self.alpha * self.func(h_bar))
        if self.proj:
            output = self.Wo(h_state)
        else:
            output = h_state
        return output, h_state


class NTMCell(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 mem_rows: int,
                 mem_columns: int,
                 funcs: Union[List[Tuple[str, Callable, nn.Module]],
                 Tuple[Union[str, Callable, nn.Module]],
                 Dict[str, Union[str, Callable, nn.Module]]] = None,
                 bias: bool = True,
                 proj_size: int = 0,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {
            "device": device,
            "dtype": dtype
        }
        self.mem_rows = mem_rows
        self.mem_columns = mem_columns

        self.controller = LSTMCell(
            input_size=input_size + mem_columns,
            hidden_size=hidden_size,
            funcs=funcs,
            bias=bias,
            proj_size=proj_size,
            **self.factory_kwargs
        )
        self.head_interface = nn.Linear(
            in_features=hidden_size,
            out_features=mem_columns * 3 + 1,
            bias=bias,
            **self.factory_kwargs
        )
        funcs = funcs[5:7] if funcs is not None and len(funcs) >= 7 else None
        if funcs is None:
            funcs = ["softplus", "sigmoid"]
        self.func = self._resolve_funcs(funcs, *args, **kwargs)

    def forward(self, x: torch.Tensor,
                h_ctrl: Optional[torch.Tensor],
                c_ctrl: Optional[torch.Tensor],
                M_prev: Optional[torch.Tensor],
                r_prev: Optional[torch.Tensor]):
        func_0, func_1 = self.func
        combined_input = torch.cat([x, r_prev], dim=-1)
        h_t, c_t = self.controller(combined_input, h_ctrl, c_ctrl)

        params = self.head_interface(h_t)
        k_t, e_t, a_t, beta_t = torch.split(params, [self.mem_columns] * 3 + [1], dim=-1)

        sim = F.cosine_similarity(k_t.unsqueeze(1), M_prev, dim=-1)
        w_t = F.softmax(sim * func_0(beta_t), dim=-1)

        r_t = torch.matmul(w_t.unsqueeze(1), M_prev).squeeze(1)

        erase = torch.matmul(w_t.unsqueeze(-1), func_1(e_t).unsqueeze(1))
        add = torch.matmul(w_t.unsqueeze(-1), a_t.unsqueeze(1))

        M_t = M_prev * (1 - erase) + add
        return r_t, h_t, c_t, M_t, w_t


class HopfieldNetworkCell(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 bias: bool = True,
                 beta: float = 1.0,
                 proj_size: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {
            "device": device,
            "dtype": dtype
        }
        self.weight = nn.Linear(
            in_features=input_size,
            out_features=hidden_size,
            bias=bias,
            **self.factory_kwargs
        )
        self.threshold = nn.Parameter(torch.zeros((hidden_size,), **self.factory_kwargs))
        self.beta = nn.Parameter(torch.tensor(beta, **self.factory_kwargs))
        self.W = nn.Linear(
            in_features=hidden_size,
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

    def forward(self, x: torch.Tensor):
        activation = self.weight(x) - self.threshold
        s = torch.sign(activation)
        net = self.W(s) * self.beta
        s_t = F.softmax(net, dim=-1)
        if self.proj:
            s_t = self.Wo(s_t)
        return s_t

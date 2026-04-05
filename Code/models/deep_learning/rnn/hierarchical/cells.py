"""Hierarchical RNN cells: HierarchicalRNNCell, HierarchicalLSTMCell, HierarchicalGRUCell."""
import torch
import torch.nn as nn
from typing import Optional, Union, Callable, List, Tuple, Dict
from .....models.utils import DLModule

from ..base import RNNCell, LSTMCell, GRUCell


class HierarchicalRNNCell(DLModule):
    def __init__(self,
                 input_size: int,
                 high_hidden_size: int,
                 low_hidden_size: int,
                 proj_size: int,
                 funcs: Union[List[Union[str, Callable, DLModule, nn.Module]],
                 Tuple[Union[str, Callable, DLModule, nn.Module]],
                 Dict[str, Union[str, Callable, DLModule, nn.Module]]] = None,
                 bias: bool = True,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {
            "device": device,
            "dtype": dtype
        }
        self.Wxh = nn.Linear(
            in_features=input_size,
            out_features=low_hidden_size,
            bias=bias,
            **self.factory_kwargs
        )
        self.Whh = nn.Linear(
            in_features=low_hidden_size,
            out_features=low_hidden_size,
            bias=bias,
            **self.factory_kwargs
        )

        self.Weh = nn.Linear(
            in_features=low_hidden_size,
            out_features=high_hidden_size,
            bias=bias,
            **self.factory_kwargs
        )
        self.WHH = nn.Linear(
            in_features=high_hidden_size,
            out_features=high_hidden_size,
            bias=bias,
            **self.factory_kwargs
        )

        self.Wo = nn.Linear(
            in_features=low_hidden_size + high_hidden_size,
            out_features=proj_size,
            bias=bias,
            **self.factory_kwargs
        )

        self.funcs = nn.ModuleList([])
        if funcs is None:
            funcs = ["tanh", "sigmoid"]
        self.funcs = self._resolve_funcs(funcs, *args, **kwargs)
        self.hidden_size = low_hidden_size + high_hidden_size
        self.low_hidden_size = low_hidden_size

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None):
        if h_prev is None:
            h_prev = torch.zeros((self.hidden_size, 1), **self.factory_kwargs)
        h_prev_low = h_prev[..., :self.low_hidden_size]
        h_prev_high = h_prev[..., self.low_hidden_size:]
        sig_low, sig_high = self.funcs

        h_t_low = sig_low(self.Wxh(x) + self.Whh(h_prev_low))
        h_t_high = sig_high(self.Weh(h_t_low) + self.WHH(h_prev_high))

        h_t = torch.cat([h_t_low, h_t_high], dim=-1)
        y_t = torch.softmax(self.Wo(h_t), dim=-1)
        return y_t, h_t


class HierarchicalLSTMCell(DLModule):
    def __init__(self,
                 input_size: int,
                 low_hidden_size: int,
                 high_hidden_size: int,
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
        lstm_kwargs_low = {
            "input_size": input_size,
            "hidden_size": low_hidden_size,
            "funcs": funcs,
            "bias": bias,
            "proj_size": proj_size,
            "args": args,
            "kwargs": kwargs,
            **self.factory_kwargs
        }
        lstm_kwargs_high = {
            "input_size": input_size,
            "hidden_size": high_hidden_size,
            "funcs": funcs,
            "bias": bias,
            "proj_size": proj_size,
            "args": args,
            "kwargs": kwargs,
            **self.factory_kwargs
        }
        self.cell_low = LSTMCell(**lstm_kwargs_low)
        self.cell_high = LSTMCell(**lstm_kwargs_high)

        # Boundary Layer
        self.Gate_low = nn.Linear(
            in_features=low_hidden_size,
            out_features=1,
            bias=bias,
            **self.factory_kwargs
        )
        self.Gate_high = nn.Linear(
            in_features=high_hidden_size,
            out_features=1,
            bias=bias,
            **self.factory_kwargs
        )
        self.low_hidden_size = low_hidden_size
        self.high_hidden_size = high_hidden_size

        funcs = funcs[:2] if funcs is not None else ["sigmoid", "sigmoid"]
        self.funcs = self._resolve_funcs(funcs, *args, **kwargs)

        self.hidden_size = low_hidden_size + high_hidden_size

    def forward(self, x: torch.Tensor,
                h_prev: Optional[torch.Tensor] = None,
                c_prev: Optional[torch.Tensor] = None,
                z_prev: Optional[torch.Tensor] = None):
        if h_prev is None:
            h_prev = torch.zeros((self.low_hidden_size + self.high_hidden_size, 1), **self.factory_kwargs)
        if c_prev is None:
            c_prev = torch.zeros((self.low_hidden_size + self.high_hidden_size, 1), **self.factory_kwargs)
        if z_prev is None:
            z_prev = torch.zeros((2, 1), **self.factory_kwargs)
        h_prev_low = h_prev[..., :self.low_hidden_size]
        h_prev_high = h_prev[..., self.low_hidden_size:]
        c_prev_low = c_prev[..., :self.low_hidden_size]
        c_prev_high = c_prev[..., self.low_hidden_size:]
        z_prev_low = z_prev[..., 0:1]
        z_prev_high = z_prev[..., 1:2]
        sig_low, sig_high = self.funcs

        c_low_in = c_prev_low * (1 - z_prev_low)
        h_t_low, c_t_low = self.cell_low(x, h_prev_low, c_low_in)
        z_t_low = sig_low(self.Gate_low(h_t_low))
        z_t_low = (z_t_low > 0.5).float()

        if z_t_low.sum() > 0:
            c_high_in = c_prev_high * (1 - z_prev_high)
            h_t_high, c_t_high = self.cell_high(h_t_low, h_prev_high, c_high_in)
        else:
            h_t_high, c_t_high = h_prev_high, c_prev_high
        z_t_high = sig_high(self.Gate_high(h_t_high))
        z_t_high = (z_t_high > 0.5).float()

        h_t = torch.cat([h_t_low, h_t_high], dim=-1)
        c_t = torch.cat([c_t_low, c_t_high], dim=-1)
        z_t = torch.cat([z_t_low, z_t_high], dim=-1)
        return h_t, c_t, z_t


class HierarchicalGRUCell(nn.Module):
    def __init__(self,
                 input_size: int,
                 low_hidden_size: int,
                 high_hidden_size: int,
                 funcs: Union[List[Tuple[str, Callable, nn.Module]],
                 Tuple[Union[str, Callable, nn.Module]],
                 Dict[str, Union[str, Callable, nn.Module]]],
                 bias: bool = True,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {
            "device": device,
            "dtype": dtype
        }
        gru_kwargs_low = {
            "input_size": input_size,
            "hidden_size": low_hidden_size,
            "funcs": funcs,
            "bias": bias,
            "args": args,
            "kwargs": kwargs,
            **self.factory_kwargs
        }
        gru_kwargs_high = {
            "input_size": input_size,
            "hidden_size": high_hidden_size,
            "funcs": funcs,
            "bias": bias,
            "args": args,
            "kwargs": kwargs,
            **self.factory_kwargs
        }
        self.cell_low = GRUCell(**gru_kwargs_low)
        self.cell_high = GRUCell(**gru_kwargs_high)

        self.low_hidden_size = low_hidden_size
        self.high_hidden_size = high_hidden_size
        self.hidden_size = low_hidden_size + high_hidden_size

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None):
        if h_prev is None:
            h_prev = torch.zeros((x.size(0), self.hidden_size), **self.factory_kwargs)

        h_prev_low = h_prev[..., :self.low_hidden_size]
        h_prev_high = h_prev[..., self.low_hidden_size:]

        h_t_low = self.cell_low(x, h_prev_low)
        h_t_high = self.cell_high(x, h_prev_high)

        h_t = torch.cat([h_t_low, h_t_high], dim=-1)
        return h_t

"""Tree RNN cells: TreeRNNCell, TreeLSTMCell, TreeGRUCell."""
import torch
import torch.nn as nn
from typing import Optional, Union, Callable, List, Tuple, Dict
from .....models.utils import DLModule

from ..base import RNNCell, LSTMCell, GRUCell


class TreeRNNCell(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 non_linearity: Union[str, Callable, nn.Module] = 'tanh',
                 funcs: Union[List, Tuple, Dict] = None,
                 n_branching: int = 2,
                 bias: bool = True,
                 proj_size: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {
            "device": device,
            "dtype": dtype
        }
        rnn_kwargs = {
            "input_size": input_size,
            "hidden_size": hidden_size * n_branching,
            "non_linearity": non_linearity,
            "funcs": funcs,
            "bias": bias,
            **kwargs,
            **self.factory_kwargs
        }
        self.rnn_cell = RNNCell(**rnn_kwargs)
        self.total_hidden = hidden_size * n_branching
        self.hidden_size = hidden_size
        self.N = n_branching
        self.Wo = nn.ModuleList([
            nn.Linear(
                in_features=hidden_size,
                out_features=proj_size,
                bias=bias,
                **self.factory_kwargs
            ) for _ in range(n_branching)
        ]) if proj_size is not None else None
        self.proj_size = proj_size

    def forward(self, x: torch.Tensor, h_prev: Optional[List[torch.Tensor]] = None):
        if h_prev is None:
            h_prev = torch.zeros((x.size(0), self.total_hidden), **self.factory_kwargs)

        if isinstance(h_prev, (list, tuple)):
            h_prev = torch.cat(h_prev, dim=-1)

        if h_prev.size(-1) < self.total_hidden:
            h_index = list(h_prev.shape[:-1])
            rem_size = self.total_hidden - h_prev.size(-1)
            h_index.append(rem_size)
            pad_zeros = torch.zeros(tuple(h_index), **self.factory_kwargs)
            h_prev = torch.cat([h_prev, pad_zeros], dim=-1)
        elif h_prev.size(-1) > self.total_hidden:
            h_prev = h_prev[..., :self.total_hidden]

        h_t = self.rnn_cell(x, h_prev)
        h_out = []
        for i in range(self.N):
            h = h_t[..., i * self.hidden_size: (i + 1) * self.hidden_size]
            if self.proj_size is not None:
                h = self.Wo[i](h)
            h_out.append(h)
        return h_out


class TreeLSTMCell(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 funcs: Union[List[Tuple[str, Callable, nn.Module]],
                 Tuple[Union[str, Callable, nn.Module]],
                 Dict[str, Union[str, Callable, nn.Module]]] = None,
                 n_branching: int = 2,
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
        lstm_kwargs = {
            "input_size": input_size,
            "hidden_size": hidden_size * n_branching,
            "funcs": funcs,
            "bias": bias,
            "proj_size": proj_size,
            **self.factory_kwargs
        }
        self.cell = LSTMCell(**lstm_kwargs)
        self.total_hidden = hidden_size * n_branching
        self.hidden_size = hidden_size
        self.N = n_branching

    def forward(self, x: torch.Tensor,
                h_prev: Optional[torch.Tensor] = None,
                c_prev: Optional[torch.Tensor] = None):
        if h_prev is None:
            h_prev = torch.zeros((x.size(0), self.total_hidden), **self.factory_kwargs)
        if isinstance(h_prev, (list, tuple)):
            h_prev = torch.cat(h_prev, dim=-1)

        if c_prev is None:
            c_prev = torch.zeros((x.size(0), self.total_hidden), **self.factory_kwargs)
        if isinstance(c_prev, (list, tuple)):
            c_prev = torch.cat(c_prev, dim=-1)

        if h_prev.size(-1) < self.total_hidden:
            h_index = list(h_prev.shape[:-1])
            rem_size = self.total_hidden - h_prev.size(-1)
            h_index.append(rem_size)
            pad_zeros = torch.zeros(tuple(h_index), **self.factory_kwargs)
            h_prev = torch.cat([h_prev, pad_zeros], dim=-1)
        elif h_prev.size(-1) > self.total_hidden:
            h_prev = h_prev[..., :self.total_hidden]

        if c_prev.size(-1) < self.total_hidden:
            c_index = list(c_prev.shape[:-1])
            rem_size = self.total_hidden - c_prev.size(-1)
            c_index.append(rem_size)
            pad_zeros = torch.zeros(tuple(c_index), **self.factory_kwargs)
            c_prev = torch.cat([c_prev, pad_zeros], dim=-1)
        elif c_prev.size(-1) > self.total_hidden:
            c_prev = c_prev[..., :self.total_hidden]

        h_t, c_t = self.cell(x, h_prev, c_prev)
        h_out = []
        c_out = []
        for i in range(self.N):
            h = h_t[..., i * self.hidden_size: (i + 1) * self.hidden_size]
            c = c_t[..., i * self.hidden_size: (i + 1) * self.hidden_size]
            h_out.append(h)
            c_out.append(c)
        return h_out, c_out


class TreeGRUCell(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 funcs: Union[List[Tuple[str, Callable, nn.Module]],
                 Tuple[Union[str, Callable, nn.Module]],
                 Dict[str, Union[str, Callable, nn.Module]]],
                 n_branching: int = 2,
                 bias: bool = True,
                 proj_size: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {
            "device": device,
            "dtype": dtype
        }
        gru_kwargs = {
            "input_size": input_size,
            "hidden_size": hidden_size * n_branching,
            "funcs": funcs,
            "bias": bias,
            **kwargs,
            **self.factory_kwargs
        }
        self.cell = GRUCell(**gru_kwargs)
        self.total_hidden = hidden_size * n_branching
        self.hidden_size = hidden_size
        self.N = n_branching
        self.Wo = nn.ModuleList([
            nn.Linear(
                in_features=hidden_size,
                out_features=proj_size,
                bias=bias,
                **self.factory_kwargs
            ) for _ in range(n_branching)
        ]) if proj_size is not None else None
        self.proj_size = proj_size

    def forward(self, x: torch.Tensor, h_prev: Optional[List[torch.Tensor]] = None):
        if h_prev is None:
            h_prev = torch.zeros((x.size(0), self.total_hidden), **self.factory_kwargs)
        if isinstance(h_prev, (list, tuple)):
            h_prev = torch.cat(h_prev, dim=-1)

        if h_prev.size(-1) < self.total_hidden:
            h_index = list(h_prev.shape[:-1])
            rem_size = self.total_hidden - h_prev.size(-1)
            h_index.append(rem_size)
            pad_zeros = torch.zeros(tuple(h_index), **self.factory_kwargs)
            h_prev = torch.cat([h_prev, pad_zeros], dim=-1)
        elif h_prev.size(-1) > self.total_hidden:
            h_prev = h_prev[..., :self.total_hidden]

        h_t = self.cell(x, h_prev)
        h_out = []
        for i in range(self.N):
            h = h_t[..., i * self.hidden_size: (i + 1) * self.hidden_size]
            if self.proj_size is not None:
                h = self.Wo[i](h)
            h_out.append(h)
        return h_out

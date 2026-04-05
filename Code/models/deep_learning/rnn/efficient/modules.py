"""Efficient RNN modules: RWKVModule, MambaModule."""
import torch
import torch.nn as nn
from typing import Optional, Union, Callable, List, Tuple
from .....models.utils import DLModule

from .cells import RWKVCell, MambaCell


class RWKVModule(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 func: Union[str, Callable, nn.Module] = "tanh",
                 bias: bool = True,
                 batch_first: bool = False,
                 dropout: float = 0.0,
                 bidirectional: bool = False,
                 decay: float = 0.0,
                 mu: Optional[Union[List[float], List[List[float]]]] = None,
                 proj_size: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {
            "device": device,
            "dtype": dtype
        }
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout_val = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.proj_size = proj_size if proj_size is not None else 0

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.layers = nn.ModuleList()

        for layer_idx in range(num_layers):
            if layer_idx == 0:
                layer_in = input_size
            else:
                prev_out = self.proj_size if self.proj_size > 0 else hidden_size
                layer_in = prev_out * self.num_directions

            layer_func = func
            if isinstance(func, (list, tuple)):
                if layer_idx < len(func):
                    layer_func = func[layer_idx]

            layer_mu = None
            if mu is not None:
                if isinstance(mu, (list, tuple)) and len(mu) > 0 and isinstance(mu[0], (list, tuple)):
                    if layer_idx < len(mu):
                        layer_mu = mu[layer_idx]
                else:
                    layer_mu = mu

            layer_cells = nn.ModuleDict()
            layer_cells['fwd'] = RWKVCell(
                input_size=layer_in,
                hidden_size=hidden_size,
                func=layer_func,
                bias=bias,
                decay=decay,
                mu=layer_mu,
                proj_size=proj_size,
                **self.factory_kwargs,
                **kwargs
            )

            if bidirectional:
                layer_cells['bwd'] = RWKVCell(
                    input_size=layer_in,
                    hidden_size=hidden_size,
                    func=layer_func,
                    bias=bias,
                    decay=decay,
                    mu=layer_mu,
                    proj_size=proj_size,
                    **self.factory_kwargs,
                    **kwargs
                )
            self.layers.append(layer_cells)

    def forward(self, x: torch.Tensor,
                states: Optional[Tuple[list, list, list]] = None):
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape

        if states is None:
            a_states = [None] * (self.num_layers * self.num_directions)
            b_states = [None] * (self.num_layers * self.num_directions)
            x_states = [None] * (self.num_layers * self.num_directions)
        else:
            a_states, b_states, x_states = states

        current_input = x
        a_all, b_all, x_prev_out_all = [], [], []

        for layer_idx, layer_cells in enumerate(self.layers):
            if layer_idx == 0:
                l_in = self.input_size
            else:
                prev_out = self.proj_size if self.proj_size > 0 else self.hidden_size
                l_in = prev_out * self.num_directions

            idx_fwd = layer_idx * self.num_directions
            a_t = a_states[idx_fwd]
            if a_t is None:
                a_t = torch.zeros((batch_size, self.hidden_size), **self.factory_kwargs)

            b_t = b_states[idx_fwd]
            if b_t is None:
                b_t = torch.zeros((batch_size, self.hidden_size), **self.factory_kwargs)

            x_prev = x_states[idx_fwd]
            if x_prev is None:
                x_prev = torch.zeros((batch_size, l_in), **self.factory_kwargs)

            cell_fwd = layer_cells['fwd']
            h_fwd_list = []

            for t in range(seq_len):
                inp_t = current_input[t]
                out_t, a_t, b_t, x_prev = cell_fwd(inp_t, a_t, b_t, x_prev)
                h_fwd_list.append(out_t)

            out_fwd = torch.stack(h_fwd_list, dim=0)
            a_all.append(a_t)
            b_all.append(b_t)
            x_prev_out_all.append(x_prev)

            if self.bidirectional:
                idx_bwd = idx_fwd + 1
                cell_bwd = layer_cells['bwd']

                a_t_b = a_states[idx_bwd]
                if a_t_b is None:
                    a_t_b = torch.zeros((batch_size, self.hidden_size), **self.factory_kwargs)

                b_t_b = b_states[idx_bwd]
                if b_t_b is None:
                    b_t_b = torch.zeros((batch_size, self.hidden_size), **self.factory_kwargs)

                x_prev_b = x_states[idx_bwd]
                if x_prev_b is None:
                    x_prev_b = torch.zeros((batch_size, l_in), **self.factory_kwargs)

                h_bwd_list = []
                for t in range(seq_len - 1, -1, -1):
                    inp_t = current_input[t]
                    out_t, a_t_b, b_t_b, x_prev_b = cell_bwd(inp_t, a_t_b, b_t_b, x_prev_b)
                    h_bwd_list.append(out_t)

                h_bwd_list.reverse()
                out_bwd = torch.stack(h_bwd_list, dim=0)

                output = torch.cat([out_fwd, out_bwd], dim=2)

                a_all.append(a_t_b)
                b_all.append(b_t_b)
                x_prev_out_all.append(x_prev_b)
            else:
                output = out_fwd

            if self.dropout is not None and layer_idx < self.num_layers - 1:
                output = self.dropout(output)
            current_input = output

        if self.batch_first:
            current_input = current_input.transpose(0, 1)

        return current_input, a_all, b_all, x_prev_out_all


class MambaModule(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 rank: Union[str, int] = "auto",
                 bias: bool = True,
                 batch_first: bool = False,
                 dropout: float = 0.0,
                 bidirectional: bool = False,
                 func: Union[str, Callable, nn.Module] = 'tanh',
                 proj_size: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {
            "device": device,
            "dtype": dtype
        }
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout_val = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.proj_size = proj_size if proj_size is not None else 0

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.layers = nn.ModuleList()

        for layer_idx in range(num_layers):
            if layer_idx == 0:
                layer_in = input_size
            else:
                prev_out = self.proj_size if self.proj_size > 0 else hidden_size
                layer_in = prev_out * self.num_directions

            layer_func = func
            if isinstance(func, (list, tuple)):
                if layer_idx < len(func):
                    layer_func = func[layer_idx]

            layer_rank = rank
            if isinstance(rank, (list, tuple)):
                if layer_idx < len(rank):
                    layer_rank = rank[layer_idx]

            layer_cells = nn.ModuleDict()
            layer_cells['fwd'] = MambaCell(
                input_size=layer_in,
                hidden_size=hidden_size,
                rank=layer_rank,
                bias=bias,
                func=layer_func,
                proj_size=proj_size,
                **self.factory_kwargs,
                **kwargs
            )

            if bidirectional:
                layer_cells['bwd'] = MambaCell(
                    input_size=layer_in,
                    hidden_size=hidden_size,
                    rank=layer_rank,
                    bias=bias,
                    func=layer_func,
                    proj_size=proj_size,
                    **self.factory_kwargs,
                    **kwargs
                )
            self.layers.append(layer_cells)

    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape

        if h0 is None:
            num_h = self.num_layers * self.num_directions
            h0 = torch.zeros(num_h, batch_size, self.hidden_size, **self.factory_kwargs)
        else:
            expected_h_dim = self.num_layers * self.num_directions
            if h0.size(0) != expected_h_dim:
                raise ValueError(
                    f"Expected h0 to have {expected_h_dim} layers, got {h0.size(0)}")

        current_input = x
        hn_all = []

        for layer_idx, layer_cells in enumerate(self.layers):
            h_idx_fwd = layer_idx * self.num_directions
            h_prev_fwd = h0[h_idx_fwd]

            cell_fwd = layer_cells['fwd']
            h_t = h_prev_fwd
            h_fwd_list = []

            for t in range(seq_len):
                inp_t = current_input[t]
                y_t, h_t = cell_fwd(inp_t, h_t)
                h_fwd_list.append(y_t)

            out_fwd = torch.stack(h_fwd_list, dim=0)
            hn_all.append(h_t)

            if self.bidirectional:
                h_idx_bwd = h_idx_fwd + 1
                h_prev_bwd = h0[h_idx_bwd]
                cell_bwd = layer_cells['bwd']
                h_t_bwd = h_prev_bwd
                h_bwd_list = []

                for t in range(seq_len - 1, -1, -1):
                    inp_t = current_input[t]
                    y_t, h_t_bwd = cell_bwd(inp_t, h_t_bwd)
                    h_bwd_list.append(y_t)

                h_bwd_list.reverse()
                out_bwd = torch.stack(h_bwd_list, dim=0)
                hn_all.append(h_t_bwd)

                output = torch.cat([out_fwd, out_bwd], dim=2)
            else:
                output = out_fwd

            if self.dropout is not None and layer_idx < self.num_layers - 1:
                output = self.dropout(output)
            current_input = output

        hn = torch.stack(hn_all, dim=0)
        if self.batch_first:
            current_input = current_input.transpose(0, 1)
        return current_input, hn

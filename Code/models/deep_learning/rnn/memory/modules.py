"""Memory RNN modules: ESNModule, NTMModule, HopfieldNetworkModule."""
import torch
import torch.nn as nn
from typing import Optional, Union, Callable, List, Tuple, Dict, Any
from .....models.utils import DLModule

from .cells import ESNCell, NTMCell, HopfieldNetworkCell


class ESNModule(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 func: Union[str, Callable, nn.Module] = 'tanh',
                 bias: bool = True,
                 batch_first: bool = False,
                 dropout: float = 0.0,
                 bidirectional: bool = False,
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

            layer_cells = nn.ModuleDict()
            layer_cells['fwd'] = ESNCell(
                input_size=layer_in,
                hidden_size=hidden_size,
                func=layer_func,
                bias=bias,
                leaking_rate=leaking_rate,
                spectral_radius=spectral_radius,
                proj_size=proj_size,
                **self.factory_kwargs,
                **kwargs
            )

            if bidirectional:
                layer_cells['bwd'] = ESNCell(
                    input_size=layer_in,
                    hidden_size=hidden_size,
                    func=layer_func,
                    bias=bias,
                    leaking_rate=leaking_rate,
                    spectral_radius=spectral_radius,
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
            if h0.size(1) != batch_size:
                raise ValueError(f"Expected h0 batch size {batch_size}, got {h0.size(1)}")

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
                out_t, h_t = cell_fwd(inp_t, h_t)
                h_fwd_list.append(out_t)

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
                    out_t, h_t_bwd = cell_bwd(inp_t, h_t_bwd)
                    h_bwd_list.append(out_t)

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


class NTMModule(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 mem_rows: int,
                 mem_columns: int,
                 funcs: Union[List[Tuple[str, Callable, nn.Module]],
                 Tuple[Union[str, Callable, nn.Module]],
                 Dict[str, Union[str, Callable, nn.Module]]] = None,
                 bias: bool = True,
                 batch_first: bool = False,
                 dropout: float = 0.0,
                 bidirectional: bool = False,
                 proj_size: int = 0,
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
        self.proj_size = proj_size
        self.mem_rows = mem_rows
        self.mem_columns = mem_columns

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.layers = nn.ModuleList()

        for layer_idx in range(num_layers):
            if layer_idx == 0:
                layer_in = input_size
            else:
                prev_out = mem_columns
                layer_in = prev_out * self.num_directions

            layer_funcs = None
            if funcs is not None:
                if isinstance(funcs, (list, tuple)):
                    if layer_idx < len(funcs):
                        layer_funcs = funcs[layer_idx]
                elif isinstance(funcs, dict):
                    layer_funcs = funcs

            layer_cells = nn.ModuleDict()
            layer_cells['fwd'] = NTMCell(
                input_size=layer_in,
                hidden_size=hidden_size,
                mem_rows=mem_rows,
                mem_columns=mem_columns,
                funcs=layer_funcs,
                bias=bias,
                proj_size=proj_size,
                **self.factory_kwargs,
                **kwargs
            )

            if bidirectional:
                layer_cells['bwd'] = NTMCell(
                    input_size=layer_in,
                    hidden_size=hidden_size,
                    mem_rows=mem_rows,
                    mem_columns=mem_columns,
                    funcs=layer_funcs,
                    bias=bias,
                    proj_size=proj_size,
                    **self.factory_kwargs,
                    **kwargs
                )
            self.layers.append(layer_cells)

    def forward(self, x: torch.Tensor, states: Optional[Any] = None) -> Tuple[torch.Tensor, Any]:
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape
        current_input = x
        states_out = []

        for layer_idx, layer_cells in enumerate(self.layers):
            h_ctrl = torch.zeros(batch_size, self.hidden_size, **self.factory_kwargs)
            c_ctrl = torch.zeros(batch_size, self.hidden_size, **self.factory_kwargs)
            M_prev = torch.zeros(batch_size, self.mem_rows, self.mem_columns, **self.factory_kwargs)
            r_prev = torch.zeros(batch_size, self.mem_columns, **self.factory_kwargs)

            cell_fwd = layer_cells['fwd']
            out_list = []

            for t in range(seq_len):
                inp_t = current_input[t]
                r_prev, h_ctrl, c_ctrl, M_prev, w_t = cell_fwd(inp_t, h_ctrl, c_ctrl, M_prev, r_prev)
                out_list.append(r_prev)

            out_fwd = torch.stack(out_list, dim=0)
            states_out.append((h_ctrl, c_ctrl, M_prev, r_prev))

            if self.bidirectional:
                cell_bwd = layer_cells['bwd']
                h_b = torch.zeros(batch_size, self.hidden_size, **self.factory_kwargs)
                c_b = torch.zeros(batch_size, self.hidden_size, **self.factory_kwargs)
                M_b = torch.zeros(batch_size, self.mem_rows, self.mem_columns, **self.factory_kwargs)
                r_b = torch.zeros(batch_size, self.mem_columns, **self.factory_kwargs)

                out_list_bwd = []
                for t in range(seq_len - 1, -1, -1):
                    inp_t = current_input[t]
                    r_b, h_b, c_b, M_b, w_t = cell_bwd(inp_t, h_b, c_b, M_b, r_b)
                    out_list_bwd.append(r_b)

                out_list_bwd.reverse()
                out_bwd = torch.stack(out_list_bwd, dim=0)

                output = torch.cat([out_fwd, out_bwd], dim=2)
                states_out.append((h_b, c_b, M_b, r_b))
            else:
                output = out_fwd

            if self.dropout is not None and layer_idx < self.num_layers - 1:
                output = self.dropout(output)
            current_input = output

        if self.batch_first:
            current_input = current_input.transpose(0, 1)

        return current_input, states_out


class HopfieldNetworkModule(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 bias: bool = True,
                 beta: float = 1.0,
                 batch_first: bool = False,
                 dropout: float = 0.0,
                 bidirectional: bool = False,
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

            layer_cells = nn.ModuleDict()
            layer_cells['fwd'] = HopfieldNetworkCell(
                input_size=layer_in,
                hidden_size=hidden_size,
                bias=bias,
                beta=beta,
                proj_size=proj_size,
                **self.factory_kwargs,
                **kwargs
            )

            if bidirectional:
                layer_cells['bwd'] = HopfieldNetworkCell(
                    input_size=layer_in,
                    hidden_size=hidden_size,
                    bias=bias,
                    beta=beta,
                    proj_size=proj_size,
                    **self.factory_kwargs,
                    **kwargs
                )
            self.layers.append(layer_cells)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape
        current_input = x

        for layer_idx, layer_cells in enumerate(self.layers):
            cell_fwd = layer_cells['fwd']
            out_list = []
            for t in range(seq_len):
                inp_t = current_input[t]
                out_t = cell_fwd(inp_t)
                out_list.append(out_t)

            out_fwd = torch.stack(out_list, dim=0)

            if self.bidirectional:
                cell_bwd = layer_cells['bwd']
                out_list_b = []
                for t in range(seq_len - 1, -1, -1):
                    inp_t = current_input[t]
                    out_t = cell_bwd(inp_t)
                    out_list_b.append(out_t)
                out_list_b.reverse()
                out_bwd = torch.stack(out_list_b, dim=0)
                output = torch.cat([out_fwd, out_bwd], dim=2)
            else:
                output = out_fwd

            if self.dropout is not None and layer_idx < self.num_layers - 1:
                output = self.dropout(output)
            current_input = output

        if self.batch_first:
            current_input = current_input.transpose(0, 1)

        return current_input

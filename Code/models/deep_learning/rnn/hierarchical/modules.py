"""Hierarchical RNN modules: HierarchicalRNNModule, HierarchicalLSTMModule, HierarchicalGRUModule."""
import torch
import torch.nn as nn
from typing import Optional, Union, List, Tuple, Dict
from .....models.utils import DLModule

from .cells import HierarchicalRNNCell, HierarchicalLSTMCell, HierarchicalGRUCell


class HierarchicalRNNModule(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 low_hidden_size: int = None,
                 funcs: Union[List, Tuple, Dict] = None,
                 bias: bool = True,
                 proj_size: int = None,
                 batch_first: bool = False,
                 dropout: float = 0.0,
                 bidirectional: bool = False,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout_val = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.proj_size = proj_size if proj_size is not None else 0
        self.low_hidden_size = low_hidden_size if low_hidden_size is not None else hidden_size // 2

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.layers = nn.ModuleList()

        for layer_idx in range(num_layers):
            if layer_idx == 0:
                layer_in = input_size
            else:
                prev_out = self.proj_size if self.proj_size > 0 else hidden_size
                layer_in = prev_out * self.num_directions

            layer_cells = nn.ModuleDict()
            layer_cells['fwd'] = HierarchicalRNNCell(
                input_size=layer_in,
                low_hidden_size=self.low_hidden_size, high_hidden_size=self.hidden_size - self.low_hidden_size,
                funcs=funcs,
                bias=bias,
                proj_size=proj_size,
                **self.factory_kwargs,
                **kwargs
            )
            if bidirectional:
                layer_cells['bwd'] = HierarchicalRNNCell(
                    input_size=layer_in,
                    low_hidden_size=self.low_hidden_size, high_hidden_size=self.hidden_size - self.low_hidden_size,
                    funcs=funcs,
                    bias=bias,
                    proj_size=proj_size,
                    **self.factory_kwargs,
                    **kwargs
                )
            self.layers.append(layer_cells)

    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None):
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape

        if h0 is None:
            num = self.num_layers * self.num_directions
            h0 = torch.zeros(num, batch_size, self.hidden_size, **self.factory_kwargs)

        current_input = x
        hn_all = []

        for layer_idx, layer_cells in enumerate(self.layers):
            idx_fwd = layer_idx * self.num_directions
            h_prev = h0[idx_fwd]
            cell_fwd = layer_cells['fwd']
            h_list = []

            for t in range(seq_len):
                inp_t = current_input[t]
                y_t, h_prev = cell_fwd(inp_t, h_prev)
                h_list.append(y_t)

            out_fwd = torch.stack(h_list, dim=0)
            hn_all.append(h_prev)

            if self.bidirectional:
                idx_bwd = idx_fwd + 1
                h_prev_b = h0[idx_bwd]
                cell_bwd = layer_cells['bwd']
                h_bwd_list = []
                for t in range(seq_len - 1, -1, -1):
                    inp_t = current_input[t]
                    y_t_b, h_prev_b = cell_bwd(inp_t, h_prev_b)
                    h_bwd_list.append(y_t_b)
                h_bwd_list.reverse()
                out_bwd = torch.stack(h_bwd_list, dim=0)
                hn_all.append(h_prev_b)
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


class HierarchicalLSTMModule(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 low_hidden_size: int = None,
                 funcs: Union[List, Tuple, Dict] = None,
                 bias: bool = True,
                 proj_size: int = None,
                 batch_first: bool = False,
                 dropout: float = 0.0,
                 bidirectional: bool = False,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout_val = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.proj_size = proj_size if proj_size is not None else 0
        self.low_hidden_size = low_hidden_size if low_hidden_size is not None else hidden_size // 2

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.layers = nn.ModuleList()

        for layer_idx in range(num_layers):
            if layer_idx == 0:
                layer_in = input_size
            else:
                prev_out = self.proj_size if self.proj_size > 0 else hidden_size
                layer_in = prev_out * self.num_directions

            layer_cells = nn.ModuleDict()
            layer_cells['fwd'] = HierarchicalLSTMCell(
                input_size=layer_in,
                low_hidden_size=self.low_hidden_size, high_hidden_size=self.hidden_size - self.low_hidden_size,
                funcs=funcs,
                bias=bias,
                proj_size=proj_size,
                **self.factory_kwargs,
                **kwargs
            )
            if bidirectional:
                layer_cells['bwd'] = HierarchicalLSTMCell(
                    input_size=layer_in,
                    low_hidden_size=self.low_hidden_size, high_hidden_size=self.hidden_size - self.low_hidden_size,
                    funcs=funcs,
                    bias=bias,
                    proj_size=proj_size,
                    **self.factory_kwargs,
                    **kwargs
                )
            self.layers.append(layer_cells)

    def forward(self, x: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None):
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape

        if hx is None:
            num = self.num_layers * self.num_directions
            h0 = torch.zeros(num, batch_size, self.hidden_size, **self.factory_kwargs)
            c0 = torch.zeros(num, batch_size, self.hidden_size, **self.factory_kwargs)
            z0 = torch.zeros(num, batch_size, 2, **self.factory_kwargs)
        else:
            h0, c0, z0 = hx

        current_input = x
        hn_all = []
        cn_all = []
        zn_all = []

        for layer_idx, layer_cells in enumerate(self.layers):
            idx_fwd = layer_idx * self.num_directions
            h_prev = h0[idx_fwd]
            c_prev = c0[idx_fwd]
            z_prev = z0[idx_fwd] if z0 is not None else None

            cell_fwd = layer_cells['fwd']
            h_list = []

            for t in range(seq_len):
                inp_t = current_input[t]
                h_prev, c_prev, z_prev = cell_fwd(inp_t, h_prev, c_prev, z_prev)
                h_list.append(h_prev)

            out_fwd = torch.stack(h_list, dim=0)
            hn_all.append(h_prev)
            cn_all.append(c_prev)
            zn_all.append(z_prev)

            if self.bidirectional:
                idx_bwd = idx_fwd + 1
                h_prev_b = h0[idx_bwd]
                c_prev_b = c0[idx_bwd]
                z_prev_b = z0[idx_bwd] if z0 is not None else None
                cell_bwd = layer_cells['bwd']
                h_bwd_list = []
                for t in range(seq_len - 1, -1, -1):
                    inp_t = current_input[t]
                    h_prev_b, c_prev_b, z_prev_b = cell_bwd(inp_t, h_prev_b, c_prev_b, z_prev_b)
                    h_bwd_list.append(h_prev_b)
                h_bwd_list.reverse()
                out_bwd = torch.stack(h_bwd_list, dim=0)
                hn_all.append(h_prev_b)
                cn_all.append(c_prev_b)
                zn_all.append(z_prev_b)
                output = torch.cat([out_fwd, out_bwd], dim=2)
            else:
                output = out_fwd

            if self.dropout is not None and layer_idx < self.num_layers - 1:
                output = self.dropout(output)
            current_input = output

        hn = torch.stack(hn_all, dim=0)
        cn = torch.stack(cn_all, dim=0)
        zn = torch.stack(zn_all, dim=0)
        if self.batch_first:
            current_input = current_input.transpose(0, 1)
        return current_input, (hn, cn, zn)


class HierarchicalGRUModule(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 low_hidden_size: int = None,
                 funcs: Union[List, Tuple, Dict] = None,
                 bias: bool = True,
                 proj_size: int = None,
                 batch_first: bool = False,
                 dropout: float = 0.0,
                 bidirectional: bool = False,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout_val = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.proj_size = proj_size if proj_size is not None else 0
        self.low_hidden_size = low_hidden_size if low_hidden_size is not None else hidden_size // 2

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.layers = nn.ModuleList()

        for layer_idx in range(num_layers):
            if layer_idx == 0:
                layer_in = input_size
            else:
                prev_out = self.proj_size if self.proj_size > 0 else hidden_size
                layer_in = prev_out * self.num_directions

            layer_cells = nn.ModuleDict()
            layer_cells['fwd'] = HierarchicalGRUCell(
                input_size=layer_in,
                low_hidden_size=self.low_hidden_size, high_hidden_size=self.hidden_size - self.low_hidden_size,
                funcs=funcs,
                bias=bias,
                proj_size=proj_size,
                **self.factory_kwargs,
                **kwargs
            )
            if bidirectional:
                layer_cells['bwd'] = HierarchicalGRUCell(
                    input_size=layer_in,
                    low_hidden_size=self.low_hidden_size, high_hidden_size=self.hidden_size - self.low_hidden_size,
                    funcs=funcs,
                    bias=bias,
                    proj_size=proj_size,
                    **self.factory_kwargs,
                    **kwargs
                )
            self.layers.append(layer_cells)

    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None):
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape

        if h0 is None:
            num = self.num_layers * self.num_directions
            h0 = torch.zeros(num, batch_size, self.hidden_size, **self.factory_kwargs)

        current_input = x
        hn_all = []

        for layer_idx, layer_cells in enumerate(self.layers):
            idx_fwd = layer_idx * self.num_directions
            h_prev = h0[idx_fwd]
            cell_fwd = layer_cells['fwd']
            h_list = []

            for t in range(seq_len):
                inp_t = current_input[t]
                h_prev = cell_fwd(inp_t, h_prev)
                h_list.append(h_prev)

            out_fwd = torch.stack(h_list, dim=0)
            hn_all.append(h_prev)

            if self.bidirectional:
                idx_bwd = idx_fwd + 1
                h_prev_b = h0[idx_bwd]
                cell_bwd = layer_cells['bwd']
                h_bwd_list = []
                for t in range(seq_len - 1, -1, -1):
                    inp_t = current_input[t]
                    h_prev_b = cell_bwd(inp_t, h_prev_b)
                    h_bwd_list.append(h_prev_b)
                h_bwd_list.reverse()
                out_bwd = torch.stack(h_bwd_list, dim=0)
                hn_all.append(h_prev_b)
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

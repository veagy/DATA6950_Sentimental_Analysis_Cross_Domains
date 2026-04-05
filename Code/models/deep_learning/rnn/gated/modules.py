"""Gated RNN modules: SRUModule, QRNNModule, MGUModule, GORUModule, JANETModule."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Callable, List, Tuple, Dict
from .....models.utils import DLModule

from .cells import SRUCell, QRNNCell, MGUCell, GORUCell, JANETCell
from .._base import _BaseRNNModule


class SRUModule(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 funcs: Union[List, Tuple, Dict] = None,
                 bias: bool = True,
                 batch_first: bool = False,
                 dropout: float = 0.0,
                 bidirectional: bool = False,
                 proj_size: int = None,
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

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.layers = nn.ModuleList()

        for layer_idx in range(num_layers):
            if layer_idx == 0:
                layer_in = input_size
            else:
                prev_out = self.proj_size if self.proj_size > 0 else hidden_size
                layer_in = prev_out * self.num_directions

            layer_funcs = funcs
            if isinstance(funcs, (list, tuple)):
                if layer_idx < len(funcs):
                    layer_funcs = funcs[layer_idx]

            layer_cells = nn.ModuleDict()
            layer_cells['fwd'] = SRUCell(
                input_size=layer_in,
                hidden_size=hidden_size,
                funcs=layer_funcs,
                bias=bias,
                proj_size=proj_size,
                **self.factory_kwargs,
                **kwargs
            )
            if bidirectional:
                layer_cells['bwd'] = SRUCell(
                    input_size=layer_in,
                    hidden_size=hidden_size,
                    funcs=layer_funcs,
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
            num_h = self.num_layers * self.num_directions
            h0 = torch.zeros(num_h, batch_size, self.hidden_size, **self.factory_kwargs)

        current_input = x
        hn_all = []

        for layer_idx, layer_cells in enumerate(self.layers):
            h_idx_fwd = layer_idx * self.num_directions
            c_prev_fwd = h0[h_idx_fwd]

            cell_fwd = layer_cells['fwd']
            h_list = []

            for t in range(seq_len):
                inp_t = current_input[t]
                h_t, c_prev_fwd = cell_fwd(inp_t, c_prev_fwd)
                h_list.append(h_t)

            out_fwd = torch.stack(h_list, dim=0)
            hn_all.append(c_prev_fwd)

            if self.bidirectional:
                h_idx_bwd = h_idx_fwd + 1
                c_prev_bwd = h0[h_idx_bwd]
                cell_bwd = layer_cells['bwd']
                h_bwd_list = []

                for t in range(seq_len - 1, -1, -1):
                    inp_t = current_input[t]
                    h_t_b, c_prev_bwd = cell_bwd(inp_t, c_prev_bwd)
                    h_bwd_list.append(h_t_b)

                h_bwd_list.reverse()
                out_bwd = torch.stack(h_bwd_list, dim=0)
                hn_all.append(c_prev_bwd)
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


class QRNNModule(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 kernel_size: int = 2,
                 dimensionality: Union[int, float] = 1,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 funcs: Union[List, Tuple, Dict] = None,
                 groups: int = 1,
                 bias: bool = True,
                 ifo_pooling: bool = False,
                 dataset_type: str = 'ts',
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
        self.batch_first = batch_first
        self.dropout_val = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.layers = nn.ModuleList()

        for layer_idx in range(num_layers):
            if layer_idx == 0:
                layer_in = input_size
            else:
                layer_in = hidden_size * self.num_directions

            layer_funcs = funcs
            if isinstance(funcs, (list, tuple)):
                if layer_idx < len(funcs):
                    layer_funcs = funcs[layer_idx]

            layer_cells = nn.ModuleDict()
            layer_cells['fwd'] = QRNNCell(
                input_size=layer_in,
                hidden_size=hidden_size,
                kernel_size=kernel_size,
                dimensionality=dimensionality,
                stride=stride,
                padding=padding,
                dilation=dilation,
                funcs=layer_funcs,
                groups=groups,
                bias=bias,
                ifo_pooling=ifo_pooling,
                **self.factory_kwargs,
                **kwargs
            )
            if bidirectional:
                layer_cells['bwd'] = QRNNCell(
                    input_size=layer_in,
                    hidden_size=hidden_size,
                    kernel_size=kernel_size,
                    dimensionality=dimensionality,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    funcs=layer_funcs,
                    groups=groups,
                    bias=bias,
                    ifo_pooling=ifo_pooling,
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
            c0 = torch.zeros(num, batch_size, self.hidden_size, **self.factory_kwargs)
        else:
            c0 = h0

        current_input = x
        hn_all = []

        for layer_idx, layer_cells in enumerate(self.layers):
            idx_fwd = layer_idx * self.num_directions
            c_prev = c0[idx_fwd]

            cell_fwd = layer_cells['fwd']

            inp_fwd = current_input.permute(1, 2, 0)

            if layer_cells['fwd'].kernel_size > 1:
                padding_size = layer_cells['fwd'].kernel_size - 1
                inp_fwd = F.pad(inp_fwd, (padding_size, 0))

            z_all, f_all, o_all, i_all = cell_fwd.process_gates(inp_fwd)

            c_list = []
            h_list = []

            for t in range(seq_len):
                z_t = z_all[:, :, t]
                f_t = f_all[:, :, t]
                o_t = o_all[:, :, t]
                i_t = i_all[:, :, t] if i_all is not None else None

                h_t, c_prev = cell_fwd.pool(c_prev, z_t, f_t, o_t, i_t)
                h_list.append(h_t)

            out_fwd = torch.stack(h_list, dim=0)
            hn_all.append(c_prev)

            if self.bidirectional:
                idx_bwd = idx_fwd + 1
                c_prev_b = c0[idx_bwd]
                cell_bwd = layer_cells['bwd']

                inp_bwd = current_input.permute(1, 2, 0)
                inp_bwd = torch.flip(inp_bwd, [2])

                if layer_cells['bwd'].kernel_size > 1:
                    padding_size = layer_cells['bwd'].kernel_size - 1
                    inp_bwd = F.pad(inp_bwd, (padding_size, 0))

                z_all_b, f_all_b, o_all_b, i_all_b = cell_bwd.process_gates(inp_bwd)

                h_list_b = []
                for t in range(seq_len):
                    z_t = z_all_b[:, :, t]
                    f_t = f_all_b[:, :, t]
                    o_t = o_all_b[:, :, t]
                    i_t = i_all_b[:, :, t] if i_all_b is not None else None

                    h_t_b, c_prev_b = cell_bwd.pool(c_prev_b, z_t, f_t, o_t, i_t)
                    h_list_b.append(h_t_b)

                h_list_b.reverse()
                out_bwd = torch.stack(h_list_b, dim=0)
                hn_all.append(c_prev_b)

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


class GORUModule(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 num_reflections: Union[int, List[int]],
                 funcs: Union[List[Tuple[str, Callable, nn.Module]],
                 Tuple[Union[str, Callable, nn.Module]],
                 Dict[str, Union[str, Callable, nn.Module]]] = None,
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

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.layers = nn.ModuleList()

        def get_layer_arg(arg, layer_idx):
            if isinstance(arg, (list, tuple)):
                if len(arg) == num_layers:
                    return arg[layer_idx]
                if len(arg) > 0 and isinstance(arg[0], (list, tuple, dict)):
                    if layer_idx < len(arg):
                        return arg[layer_idx]
                    return arg[-1]
                return arg
            return arg

        def get_scalar_or_list_arg(arg, layer_idx):
            if isinstance(arg, (list, tuple)):
                if layer_idx < len(arg):
                    return arg[layer_idx]
                return arg[-1]
            return arg

        for layer_idx in range(num_layers):
            if layer_idx == 0:
                layer_in = input_size
            else:
                prev_out = self.proj_size if self.proj_size > 0 else hidden_size
                layer_in = prev_out * self.num_directions

            cur_reflections = get_scalar_or_list_arg(num_reflections, layer_idx)
            cur_funcs = get_layer_arg(funcs, layer_idx)

            layer_cells = nn.ModuleDict()
            layer_cells['fwd'] = GORUCell(
                input_size=layer_in,
                hidden_size=hidden_size,
                num_reflections=cur_reflections,
                funcs=cur_funcs,
                bias=bias,
                proj_size=proj_size,
                **self.factory_kwargs,
                **kwargs
            )
            if bidirectional:
                layer_cells['bwd'] = GORUCell(
                    input_size=layer_in,
                    hidden_size=hidden_size,
                    num_reflections=cur_reflections,
                    funcs=cur_funcs,
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


class MGUModule(_BaseRNNModule):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, funcs=None, bias=True,
                 proj_size=None, dropout=0.0, bidirectional=False, batch_first=False,
                 device="cpu", dtype=torch.float32, *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        funcs_exp = self._expand_arg(funcs, num_layers, bidirectional, is_container=True)
        l_ins = self._init_layer_args(input_size, hidden_size, num_layers, 2 if bidirectional else 1, proj_size)

        self.layers = nn.ModuleList()
        k = 0
        for i in range(num_layers):
            ld = nn.ModuleDict()
            ld['fwd'] = MGUCell(l_ins[i], hidden_size, funcs_exp[k], bias, proj_size, **self.factory_kwargs)
            k += 1
            if bidirectional:
                ld['bwd'] = MGUCell(l_ins[i], hidden_size, funcs_exp[k], bias, proj_size, **self.factory_kwargs)
                k += 1
            self.layers.append(ld)

    def forward(self, x, h0=None):
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len = x.size(0)
        curr = x
        for i, layer in enumerate(self.layers):
            fwd_out = []
            h = None
            for t in range(seq_len):
                h = layer['fwd'](curr[t], h)
                fwd_out.append(h)
            fwd = torch.stack(fwd_out, dim=0)
            if self.bidirectional:
                bwd_out = []
                h_b = None
                for t in range(seq_len - 1, -1, -1):
                    h_b = layer['bwd'](curr[t], h_b)
                    bwd_out.append(h_b)
                bwd_out.reverse()
                bwd = torch.stack(bwd_out, dim=0)
                curr = torch.cat([fwd, bwd], dim=2)
            else:
                curr = fwd
            if self.dropout and i < self.num_layers - 1:
                curr = self.dropout(curr)
        if self.batch_first:
            curr = curr.transpose(0, 1)
        return curr


class JANETModule(MGUModule):
    def __init__(self, input_size, hidden_size, num_layers, funcs=None, bias=True,
                 proj_size=None, dropout=0.0, bidirectional=False, batch_first=False,
                 device="cpu", dtype=torch.float32, *args, **kwargs):
        super().__init__(input_size, hidden_size, num_layers, funcs, bias, proj_size, dropout,
                         bidirectional, batch_first, device, dtype, *args, **kwargs)
        self.layers = nn.ModuleList()
        funcs_exp = self._expand_arg(funcs, num_layers, bidirectional, is_container=True)
        l_ins = self._init_layer_args(input_size, hidden_size, num_layers, 2 if bidirectional else 1, proj_size)
        k = 0
        for i in range(num_layers):
            ld = nn.ModuleDict()
            ld['fwd'] = JANETCell(l_ins[i], hidden_size, funcs_exp[k], bias, proj_size, **self.factory_kwargs)
            k += 1
            if bidirectional:
                ld['bwd'] = JANETCell(l_ins[i], hidden_size, funcs_exp[k], bias, proj_size, **self.factory_kwargs)
                k += 1
            self.layers.append(ld)

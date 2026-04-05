"""Continuous RNN modules: NeuralODEModule, LTCModule, CfCModule."""
import torch
import torch.nn as nn
from typing import Optional, Union, List, Tuple, Dict
from .....models.utils import DLModule

from .cells import NeuralODECell, LTCCell, CfCCell
from .._base import _BaseRNNModule


class NeuralODEModule(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 nn_module: Union[nn.Module, DLModule, dict] = "auto",
                 funcs: Union[List, Tuple, Dict] = None,
                 update: Union[str, nn.Module, DLModule] = "auto",
                 solve_method: str = "rk_4",
                 bias: bool = True,
                 batch_first: bool = False,
                 dropout: float = 0.0,
                 bidirectional: bool = False,
                 proj_size: int = 0,
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
        self.proj_size = proj_size

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.layers = nn.ModuleList()

        for layer_idx in range(num_layers):
            if layer_idx == 0:
                layer_in = input_size
            else:
                prev_out = proj_size if proj_size > 0 else hidden_size
                layer_in = prev_out * self.num_directions

            layer_funcs = funcs
            if isinstance(funcs, (list, tuple)):
                if layer_idx < len(funcs):
                    layer_funcs = funcs[layer_idx]

            layer_cells = nn.ModuleDict()
            layer_cells['fwd'] = NeuralODECell(
                input_size=layer_in,
                hidden_size=hidden_size,
                nn_module=nn_module,
                funcs=layer_funcs,
                update=update,
                solve_method=solve_method,
                bias=bias,
                proj_size=proj_size,
                **self.factory_kwargs,
                **kwargs
            )
            if bidirectional:
                layer_cells['bwd'] = NeuralODECell(
                    input_size=layer_in,
                    hidden_size=hidden_size,
                    nn_module=nn_module,
                    funcs=layer_funcs,
                    update=update,
                    solve_method=solve_method,
                    bias=bias,
                    proj_size=proj_size,
                    **self.factory_kwargs,
                    **kwargs
                )
            self.layers.append(layer_cells)

    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None, t_span: Optional[torch.Tensor] = None):
        if self.batch_first:
            x = x.transpose(0, 1)  # (Seq, Batch, Input)

        seq_len, batch_size, _ = x.shape

        # Prepare t_span
        if t_span is None:
            t_span = torch.linspace(0, seq_len, steps=seq_len + 1, device=x.device, dtype=x.dtype)
        else:
            if t_span.size(0) != seq_len + 1:
                if t_span.size(0) == seq_len:
                    t0 = torch.zeros(1, device=x.device, dtype=x.dtype)
                    t_span = torch.cat([t0, t_span])

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
                t_interval = t_span[t:t + 2]
                h_prev = cell_fwd(inp_t, h_prev, t_interval)
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
                    t_interval = torch.flip(t_span[t:t + 2], [0])
                    h_prev_b = cell_bwd(inp_t, h_prev_b, t_interval)
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


class LTCModule(_BaseRNNModule):
    def __init__(self, input_size, hidden_size, num_layers, seq_len: int, funcs="sigmoid",
                 bias=True, time_delta=0.1, solver_type="euler", proj_size=None, dropout=0.0,
                 bidirectional=False, batch_first=False, device="cpu", dtype=torch.float32, *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        funcs_exp = self._expand_arg(funcs, num_layers, bidirectional)
        l_ins = self._init_layer_args(input_size, hidden_size, num_layers, 2 if bidirectional else 1, proj_size)

        self.layers = nn.ModuleList()
        k = 0
        for i in range(num_layers):
            ld = nn.ModuleDict()
            ld['fwd'] = LTCCell(l_ins[i], hidden_size, seq_len, funcs_exp[k], bias, time_delta,
                                solver_type, proj_size, **self.factory_kwargs)
            k += 1
            if bidirectional:
                ld['bwd'] = LTCCell(l_ins[i], hidden_size, seq_len, funcs_exp[k], bias, time_delta,
                                    solver_type, proj_size, **self.factory_kwargs)
                k += 1
            self.layers.append(ld)

    def forward(self, x, h0=None):
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len = x.size(0)
        curr = x
        for i, layer in enumerate(self.layers):
            fwd_out = []
            In_fwd = []

            for t in range(seq_len):
                h = layer['fwd'](curr[t], In_fwd)
                In_fwd.append(h)
                fwd_out.append(h)

            fwd = torch.stack(fwd_out, dim=0)

            if self.bidirectional:
                bwd_out = []
                In_bwd = []
                for t in range(seq_len - 1, -1, -1):
                    h = layer['bwd'](curr[t], In_bwd)
                    In_bwd.append(h)
                    bwd_out.append(h)
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


class CfCModule(_BaseRNNModule):
    def __init__(self, input_size, hidden_size, num_layers, funcs="sigmoid", bias=True,
                 time_delta=0.1, proj_size=None, dropout=0.0, bidirectional=False,
                 batch_first=False, device="cpu", dtype=torch.float32, *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        funcs_exp = self._expand_arg(funcs, num_layers, bidirectional)
        l_ins = self._init_layer_args(input_size, hidden_size, num_layers, 2 if bidirectional else 1, proj_size)

        self.layers = nn.ModuleList()
        k = 0
        for i in range(num_layers):
            ld = nn.ModuleDict()
            ld['fwd'] = CfCCell(l_ins[i], hidden_size, funcs_exp[k], bias, time_delta, proj_size, **self.factory_kwargs)
            k += 1
            if bidirectional:
                ld['bwd'] = CfCCell(l_ins[i], hidden_size, funcs_exp[k], bias, time_delta, proj_size, **self.factory_kwargs)
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

from typing import Callable
"""Specialized RNN modules extracted from RNNFamily."""
import torch
import torch.nn as nn
from typing import Optional, Union, List, Tuple, Dict, Any

from .....models.utils import DLModule
from .._base import _BaseRNNModule

from .cells import (
    LMUCell,
    IndRNNCell,
    PhasedLSTMCell,
    mRNNCell,
    FastWeightsRNCell,
    SkipRNNCell,
    JumpLSTMCell,
    ACTRNNCell,
    uRNNCell,
    AntiSymRNNCell,
    CTRNNCell,
    StackRNNCell,
    VariationalRecurrentUnitCell,
    NARXCell,
    MorgifierRecurrentUnitCell,
)


class PhasedLSTMModule(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 funcs: Union[List, Tuple, Dict] = None,
                 bias: bool = True,
                 use_feature_timestamp: bool = False,
                 alpha: float = 0.001,
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
        self.use_feature_timestamp = use_feature_timestamp

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
            layer_cells['fwd'] = PhasedLSTMCell(
                input_size=layer_in,
                hidden_size=hidden_size,
                funcs=layer_funcs,
                bias=bias,
                use_feature_timestamp=use_feature_timestamp,
                alpha=alpha,
                proj_size=proj_size,
                **self.factory_kwargs,
                **kwargs
            )
            if bidirectional:
                layer_cells['bwd'] = PhasedLSTMCell(
                    input_size=layer_in,
                    hidden_size=hidden_size,
                    funcs=layer_funcs,
                    bias=bias,
                    use_feature_timestamp=use_feature_timestamp,
                    alpha=alpha,
                    proj_size=proj_size,
                    **self.factory_kwargs,
                    **kwargs
                )
            self.layers.append(layer_cells)

    def forward(self, x: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                times: Optional[torch.Tensor] = None):
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape

        if self.use_feature_timestamp:
            pass
        else:
            if times is None:
                times = torch.arange(seq_len, device=x.device, dtype=x.dtype).view(seq_len, 1).expand(seq_len, batch_size)
            else:
                if times.dim() == 1:
                    times = times.view(-1, 1).expand(seq_len, batch_size)
                if self.batch_first and times.shape[0] == batch_size:
                    times = times.transpose(0, 1)

        if hx is None:
            num = self.num_layers * self.num_directions
            real_hidden = self.proj_size if self.proj_size > 0 else self.hidden_size
            h0 = torch.zeros(num, batch_size, real_hidden, **self.factory_kwargs)
            c0 = torch.zeros(num, batch_size, self.hidden_size, **self.factory_kwargs)
        else:
            h0, c0 = hx

        current_input = x
        hn_all, cn_all = [], []

        for layer_idx, layer_cells in enumerate(self.layers):
            idx_fwd = layer_idx * self.num_directions
            h_prev = h0[idx_fwd]
            c_prev = c0[idx_fwd]

            cell_fwd = layer_cells['fwd']
            h_list = []

            for t in range(seq_len):
                inp_t = current_input[t]
                if self.use_feature_timestamp:
                    time_t = None
                else:
                    time_t = times[t]
                    if time_t.dim() == 1:
                        time_t = time_t.unsqueeze(1)

                h_prev, c_prev = cell_fwd(inp_t, h_prev, c_prev, t=time_t)
                h_list.append(h_prev)

            out_fwd = torch.stack(h_list, dim=0)
            hn_all.append(h_prev)
            cn_all.append(c_prev)

            if self.bidirectional:
                idx_bwd = idx_fwd + 1
                h_prev_b = h0[idx_bwd]
                c_prev_b = c0[idx_bwd]
                cell_bwd = layer_cells['bwd']
                h_bwd_list = []

                for t in range(seq_len - 1, -1, -1):
                    inp_t = current_input[t]
                    if self.use_feature_timestamp:
                        time_t = None
                    else:
                        time_t = times[t]
                        if time_t.dim() == 1:
                            time_t = time_t.unsqueeze(1)

                    h_prev_b, c_prev_b = cell_bwd(inp_t, h_prev_b, c_prev_b, t=time_t)
                    h_bwd_list.append(h_prev_b)

                h_bwd_list.reverse()
                out_bwd = torch.stack(h_bwd_list, dim=0)
                hn_all.append(h_prev_b)
                cn_all.append(c_prev_b)
                output = torch.cat([out_fwd, out_bwd], dim=2)
            else:
                output = out_fwd

            if self.dropout is not None and layer_idx < self.num_layers - 1:
                output = self.dropout(output)
            current_input = output

        hn = torch.stack(hn_all, dim=0)
        cn = torch.stack(cn_all, dim=0)
        if self.batch_first:
            current_input = current_input.transpose(0, 1)
        return current_input, (hn, cn)


class IndRNNModule(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 func: Union[str, Callable, nn.Module] = "relu",
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

            layer_func = func
            if isinstance(func, (list, tuple)):
                if layer_idx < len(func):
                    layer_func = func[layer_idx]

            layer_cells = nn.ModuleDict()
            layer_cells['fwd'] = IndRNNCell(
                input_size=layer_in,
                hidden_size=hidden_size,
                func=layer_func,
                bias=bias,
                proj_size=proj_size,
                **self.factory_kwargs,
                **kwargs
            )
            if bidirectional:
                layer_cells['bwd'] = IndRNNCell(
                    input_size=layer_in,
                    hidden_size=hidden_size,
                    func=layer_func,
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
                h_prev, _ = cell_fwd(inp_t, h_prev)
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
                    h_prev_b, _ = cell_bwd(inp_t, h_prev_b)
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


class LMUModule(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 memory_size: int,
                 theta: int = 1,
                 func: Union[str, Callable, nn.Module] = "sigmoid",
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
        self.memory_size = memory_size
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
            layer_cells['fwd'] = LMUCell(
                input_size=layer_in,
                hidden_size=hidden_size,
                memory_size=memory_size,
                theta=theta,
                func=layer_func,
                bias=bias,
                proj_size=proj_size,
                **self.factory_kwargs,
                **kwargs
            )
            if bidirectional:
                layer_cells['bwd'] = LMUCell(
                    input_size=layer_in,
                    hidden_size=hidden_size,
                    memory_size=memory_size,
                    theta=theta,
                    func=layer_func,
                    bias=bias,
                    proj_size=proj_size,
                    **self.factory_kwargs,
                    **kwargs
                )
            self.layers.append(layer_cells)

    def forward(self, x: torch.Tensor, h0: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.shape

        if h0 is None:
            num = self.num_layers * self.num_directions
            h_init = torch.zeros(num, batch_size, self.hidden_size, **self.factory_kwargs)
            m_init = torch.zeros(num, batch_size, self.memory_size, **self.factory_kwargs)
        else:
            h_init, m_init = h0

        current_input = x
        h_final = []
        m_final = []

        for layer_idx, layer_cells in enumerate(self.layers):
            idx_fwd = layer_idx * self.num_directions
            h_prev = h_init[idx_fwd]
            m_prev = m_init[idx_fwd]

            cell_fwd = layer_cells['fwd']
            out_list = []

            for t in range(seq_len):
                inp_t = current_input[t]
                h_t, m_prev = cell_fwd(inp_t, m_prev, h_prev)
                h_prev = h_t
                out_list.append(h_t)

            out_fwd = torch.stack(out_list, dim=0)
            h_final.append(h_prev)
            m_final.append(m_prev)

            if self.bidirectional:
                idx_bwd = idx_fwd + 1
                h_prev_b = h_init[idx_bwd]
                m_prev_b = m_init[idx_bwd]
                cell_bwd = layer_cells['bwd']
                out_list_b = []

                for t in range(seq_len - 1, -1, -1):
                    inp_t = current_input[t]
                    h_t_b, m_prev_b = cell_bwd(inp_t, m_prev_b, h_prev_b)
                    h_prev_b = h_t_b
                    out_list_b.append(h_t_b)

                out_list_b.reverse()
                out_bwd = torch.stack(out_list_b, dim=0)
                h_final.append(h_prev_b)
                m_final.append(m_prev_b)
                output = torch.cat([out_fwd, out_bwd], dim=2)
            else:
                output = out_fwd

            if self.dropout is not None and layer_idx < self.num_layers - 1:
                output = self.dropout(output)
            current_input = output

        hn = torch.stack(h_final, dim=0)
        mn = torch.stack(m_final, dim=0)
        if self.batch_first:
            current_input = current_input.transpose(0, 1)
        return current_input, (hn, mn)


class mRNNModule(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 funcs: Union[str, Callable, nn.Module, DLModule] = None,
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

        for layer_idx in range(num_layers):
            if layer_idx == 0:
                layer_in = input_size
            else:
                prev_out = self.proj_size if self.proj_size > 0 else hidden_size
                layer_in = prev_out * self.num_directions

            layer_func = funcs
            if isinstance(funcs, (list, tuple)):
                if layer_idx < len(funcs):
                    layer_func = funcs[layer_idx]

            layer_cells = nn.ModuleDict()
            layer_cells['fwd'] = mRNNCell(
                input_size=layer_in,
                hidden_size=hidden_size,
                funcs=layer_func,
                bias=bias,
                proj_size=proj_size,
                **self.factory_kwargs,
                **kwargs
            )
            if bidirectional:
                layer_cells['bwd'] = mRNNCell(
                    input_size=layer_in,
                    hidden_size=hidden_size,
                    funcs=layer_func,
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


class FastWeightsRNModule(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 funcs: Union[str, Callable, nn.Module, DLModule] = None,
                 bias: bool = True,
                 proj_size: int = None,
                 batch_first: bool = False,
                 dropout: float = 0.0,
                 bidirectional: bool = False,
                 delta_rule: bool = False,
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
        self.delta_rule = delta_rule

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.layers = nn.ModuleList()

        for layer_idx in range(num_layers):
            if layer_idx == 0:
                layer_in = input_size
            else:
                prev_out = self.proj_size if self.proj_size > 0 else hidden_size
                layer_in = prev_out * self.num_directions

            layer_func = funcs
            if isinstance(funcs, (list, tuple)):
                if layer_idx < len(funcs):
                    layer_func = funcs[layer_idx]

            layer_cells = nn.ModuleDict()
            layer_cells['fwd'] = FastWeightsRNCell(
                input_size=layer_in,
                hidden_size=hidden_size,
                funcs=layer_func,
                bias=bias,
                proj_size=proj_size,
                delta_rule=delta_rule,
                **self.factory_kwargs,
                **kwargs
            )
            if bidirectional:
                layer_cells['bwd'] = FastWeightsRNCell(
                    input_size=layer_in,
                    hidden_size=hidden_size,
                    funcs=layer_func,
                    bias=bias,
                    proj_size=proj_size,
                    delta_rule=delta_rule,
                    **self.factory_kwargs,
                    **kwargs
                )
            self.layers.append(layer_cells)

    def forward(self, x: torch.Tensor, h0: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape

        W_init = None
        current_input = x
        W_last_all = []

        for layer_idx, layer_cells in enumerate(self.layers):
            W_prev = None
            cell_fwd = layer_cells['fwd']
            h_list = []

            for t in range(seq_len):
                inp_t = current_input[t]
                y_t, W_prev = cell_fwd(inp_t, W_prev)
                h_list.append(y_t)

            out_fwd = torch.stack(h_list, dim=0)
            W_last_all.append(W_prev)

            if self.bidirectional:
                cell_bwd = layer_cells['bwd']
                W_prev_b = None
                h_bwd_list = []
                for t in range(seq_len - 1, -1, -1):
                    inp_t = current_input[t]
                    y_t_b, W_prev_b = cell_bwd(inp_t, W_prev_b)
                    h_bwd_list.append(y_t_b)

                h_bwd_list.reverse()
                out_bwd = torch.stack(h_bwd_list, dim=0)
                W_last_all.append(W_prev_b)
                output = torch.cat([out_fwd, out_bwd], dim=2)
            else:
                output = out_fwd

            if self.dropout is not None and layer_idx < self.num_layers - 1:
                output = self.dropout(output)
            current_input = output

        try:
            W_n = torch.stack(W_last_all, dim=0)
        except Exception:
            W_n = W_last_all

        if self.batch_first:
            current_input = current_input.transpose(0, 1)
        return current_input, W_n


class SkipRNNModule(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 funcs: Union[List, Tuple, Dict] = None,
                 func: Union[nn.Module, DLModule, str, Callable] = None,
                 bias: bool = True,
                 update: Union[str, DLModule, nn.Module] = "rnn",
                 proj_size: int = 0,
                 threshold: float = 1.0,
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
        self.threshold = threshold

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.layers = nn.ModuleList()

        for layer_idx in range(num_layers):
            if layer_idx == 0:
                layer_in = input_size
            else:
                prev_out = self.proj_size if self.proj_size > 0 else hidden_size
                layer_in = prev_out * self.num_directions

            layer_cells = nn.ModuleDict()
            layer_cells['fwd'] = SkipRNNCell(
                input_size=layer_in,
                hidden_size=hidden_size,
                funcs=funcs,
                func=func,
                bias=bias,
                update=update,
                proj_size=proj_size,
                threshold=threshold,
                **self.factory_kwargs,
                **kwargs
            )
            if bidirectional:
                layer_cells['bwd'] = SkipRNNCell(
                    input_size=layer_in,
                    hidden_size=hidden_size,
                    funcs=funcs,
                    func=func,
                    bias=bias,
                    update=update,
                    proj_size=proj_size,
                    threshold=threshold,
                    **self.factory_kwargs,
                    **kwargs
                )
            self.layers.append(layer_cells)

    def forward(self, x: torch.Tensor, h0: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape

        num = self.num_layers * self.num_directions
        if h0 is None:
            h_init = torch.zeros(num, batch_size, self.hidden_size, **self.factory_kwargs)
            G_init = torch.zeros(num, batch_size, self.hidden_size, **self.factory_kwargs)
        else:
            h_init, G_init = h0

        if h0 is None:
            G_init = torch.zeros(num, batch_size, self.hidden_size, **self.factory_kwargs)

        current_input = x
        hn_all = []
        Gn_all = []

        for layer_idx, layer_cells in enumerate(self.layers):
            idx_fwd = layer_idx * self.num_directions
            h_prev = h_init[idx_fwd]
            G_prev = G_init[idx_fwd]
            cell_fwd = layer_cells['fwd']
            h_list = []

            for t in range(seq_len):
                inp_t = current_input[t]
                h_prev, G_prev = cell_fwd(inp_t, h_prev, G_prev)
                h_list.append(h_prev)

            out_fwd = torch.stack(h_list, dim=0)
            hn_all.append(h_prev)
            Gn_all.append(G_prev)

            if self.bidirectional:
                idx_bwd = idx_fwd + 1
                h_prev_b = h_init[idx_bwd]
                G_prev_b = G_init[idx_bwd]
                cell_bwd = layer_cells['bwd']
                h_bwd_list = []
                for t in range(seq_len - 1, -1, -1):
                    inp_t = current_input[t]
                    h_prev_b, G_prev_b = cell_bwd(inp_t, h_prev_b, G_prev_b)
                    h_bwd_list.append(h_prev_b)

                h_bwd_list.reverse()
                out_bwd = torch.stack(h_bwd_list, dim=0)
                hn_all.append(h_prev_b)
                Gn_all.append(G_prev_b)
                output = torch.cat([out_fwd, out_bwd], dim=2)
            else:
                output = out_fwd

            if self.dropout is not None and layer_idx < self.num_layers - 1:
                output = self.dropout(output)
            current_input = output

        hn = torch.stack(hn_all, dim=0)
        Gn = torch.stack(Gn_all, dim=0)
        if self.batch_first:
            current_input = current_input.transpose(0, 1)
        return current_input, (hn, Gn)


class JumpLSTMModule(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 funcs: Union[List, Tuple, Dict] = None,
                 bias: bool = True,
                 hard: bool = False,
                 tau: float = 1.0,
                 proj_size: int = 0,
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
            layer_cells['fwd'] = JumpLSTMCell(
                input_size=layer_in,
                hidden_size=hidden_size,
                funcs=layer_funcs,
                bias=bias,
                hard=hard,
                tau=tau,
                proj_size=proj_size,
                **self.factory_kwargs,
                **kwargs
            )
            if bidirectional:
                layer_cells['bwd'] = JumpLSTMCell(
                    input_size=layer_in,
                    hidden_size=hidden_size,
                    funcs=layer_funcs,
                    bias=bias,
                    hard=hard,
                    tau=tau,
                    proj_size=proj_size,
                    **self.factory_kwargs,
                    **kwargs
                )
            self.layers.append(layer_cells)

    def forward(self, x: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape

        if hx is None:
            num = self.num_layers * self.num_directions
            h0 = torch.zeros(num, batch_size, self.proj_size if self.proj_size > 0 else self.hidden_size, **self.factory_kwargs)
            c0 = torch.zeros(num, batch_size, self.hidden_size, **self.factory_kwargs)
        else:
            h0, c0 = hx

        current_input = x
        hn_all = []
        cn_all = []

        for layer_idx, layer_cells in enumerate(self.layers):
            idx_fwd = layer_idx * self.num_directions
            h_prev = h0[idx_fwd]
            c_prev = c0[idx_fwd]

            cell_fwd = layer_cells['fwd']
            h_list = []

            for t in range(seq_len):
                inp_t = current_input[t]
                h_prev, c_prev, u_t = cell_fwd(inp_t, h_prev, c_prev)
                h_list.append(h_prev)

            out_fwd = torch.stack(h_list, dim=0)
            hn_all.append(h_prev)
            cn_all.append(c_prev)

            if self.bidirectional:
                idx_bwd = idx_fwd + 1
                h_prev_b = h0[idx_bwd]
                c_prev_b = c0[idx_bwd]
                cell_bwd = layer_cells['bwd']
                h_bwd_list = []
                for t in range(seq_len - 1, -1, -1):
                    inp_t = current_input[t]
                    h_prev_b, c_prev_b, u_t = cell_bwd(inp_t, h_prev_b, c_prev_b)
                    h_bwd_list.append(h_prev_b)

                h_bwd_list.reverse()
                out_bwd = torch.stack(h_bwd_list, dim=0)
                hn_all.append(h_prev_b)
                cn_all.append(c_prev_b)
                output = torch.cat([out_fwd, out_bwd], dim=2)
            else:
                output = out_fwd

            if self.dropout is not None and layer_idx < self.num_layers - 1:
                output = self.dropout(output)
            current_input = output

        hn = torch.stack(hn_all, dim=0)
        cn = torch.stack(cn_all, dim=0)
        if self.batch_first:
            current_input = current_input.transpose(0, 1)
        return current_input, (hn, cn)


class ACTRNNModule(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 non_linearity: Union[str, Callable, nn.Module] = 'tanh',
                 funcs: Union[List, Tuple, Dict] = None,
                 halting_func: Union[str, Callable, nn.Module, DLModule] = 'sigmoid',
                 bias: bool = True,
                 max_step_size: int = 50,
                 cell_type: str = 'rnn',
                 proj_size: int = None,
                 epsilon: float = 0.0001,
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

        for layer_idx in range(num_layers):
            if layer_idx == 0:
                layer_in = input_size
            else:
                prev_out = self.proj_size if self.proj_size > 0 else hidden_size
                layer_in = prev_out * self.num_directions

            layer_cells = nn.ModuleDict()
            layer_cells['fwd'] = ACTRNNCell(
                input_size=layer_in,
                hidden_size=hidden_size,
                non_linearity=non_linearity,
                funcs=funcs,
                halting_func=halting_func,
                bias=bias,
                max_step_size=max_step_size,
                cell_type=cell_type,
                proj_size=proj_size,
                epsilon=epsilon,
                **self.factory_kwargs,
                **kwargs
            )
            if bidirectional:
                layer_cells['bwd'] = ACTRNNCell(
                    input_size=layer_in,
                    hidden_size=hidden_size,
                    non_linearity=non_linearity,
                    funcs=funcs,
                    halting_func=halting_func,
                    bias=bias,
                    max_step_size=max_step_size,
                    cell_type=cell_type,
                    proj_size=proj_size,
                    epsilon=epsilon,
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
        ponder_all = []

        for layer_idx, layer_cells in enumerate(self.layers):
            idx_fwd = layer_idx * self.num_directions
            h_prev = h0[idx_fwd]
            cell_fwd = layer_cells['fwd']
            h_list = []
            ponder_list = []

            for t in range(seq_len):
                inp_t = current_input[t]
                h_prev, ponder = cell_fwd(inp_t, h_prev)
                h_list.append(h_prev)
                ponder_list.append(ponder)

            out_fwd = torch.stack(h_list, dim=0)
            hn_all.append(h_prev)
            ponder_all.append(torch.stack(ponder_list, dim=0))

            if self.bidirectional:
                idx_bwd = idx_fwd + 1
                h_prev_b = h0[idx_bwd]
                cell_bwd = layer_cells['bwd']
                h_bwd_list = []
                ponder_bwd_list = []
                for t in range(seq_len - 1, -1, -1):
                    inp_t = current_input[t]
                    h_prev_b, ponder_b = cell_bwd(inp_t, h_prev_b)
                    h_bwd_list.append(h_prev_b)
                    ponder_bwd_list.append(ponder_b)

                h_bwd_list.reverse()
                ponder_bwd_list.reverse()
                out_bwd = torch.stack(h_bwd_list, dim=0)
                hn_all.append(h_prev_b)
                ponder_all.append(torch.stack(ponder_bwd_list, dim=0))
                output = torch.cat([out_fwd, out_bwd], dim=2)
            else:
                output = out_fwd

            if self.dropout is not None and layer_idx < self.num_layers - 1:
                output = self.dropout(output)
            current_input = output

        hn = torch.stack(hn_all, dim=0)
        ponder_out = torch.stack(ponder_all, dim=0)

        if self.batch_first:
            current_input = current_input.transpose(0, 1)
        return current_input, (hn, ponder_out)


class uRNNModule(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 proj_size: int = None,
                 num_reflections: int = 2,
                 func: Union[str, nn.Module, DLModule, Callable] = "tanh",
                 bias: bool = True,
                 dim: int = 0,
                 arrangement: str = "split",
                 is_stacked_flag: bool = False,
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
        if proj_size is None:
            proj_size = hidden_size

        for layer_idx in range(num_layers):
            if layer_idx == 0:
                layer_in = input_size
            else:
                prev_out = self.proj_size if self.proj_size > 0 else hidden_size
                layer_in = prev_out * self.num_directions

            layer_cells = nn.ModuleDict()
            layer_cells['fwd'] = uRNNCell(
                input_size=layer_in,
                hidden_size=hidden_size,
                proj_size=proj_size,
                num_reflections=num_reflections,
                func=func,
                bias=bias,
                dim=dim,
                arrangement=arrangement,
                is_stacked_flag=is_stacked_flag,
                **self.factory_kwargs,
                **kwargs
            )
            if bidirectional:
                layer_cells['bwd'] = uRNNCell(
                    input_size=layer_in,
                    hidden_size=hidden_size,
                    proj_size=proj_size,
                    num_reflections=num_reflections,
                    func=func,
                    bias=bias,
                    dim=dim,
                    arrangement=arrangement,
                    is_stacked_flag=is_stacked_flag,
                    **self.factory_kwargs,
                    **kwargs
                )
            self.layers.append(layer_cells)

    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None):
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape
        from .....models.deep_learning.activations.Complex.complex_ import Complex

        states = [None] * (self.num_layers * self.num_directions)
        if h0 is not None:
            pass

        current_input = x
        hn_all = []
        layer_output_list = []

        for layer_idx, layer_cells in enumerate(self.layers):
            idx_fwd = layer_idx * self.num_directions
            h_prev = states[idx_fwd]

            cell_fwd = layer_cells['fwd']
            y_list = []

            for t in range(seq_len):
                inp_t = current_input[t]
                y_t, h_prev, W = cell_fwd(inp_t, h_prev)
                y_list.append(y_t)

            out_fwd_tensor = torch.stack([y.tensor for y in y_list], dim=1)
            out_fwd = Complex(out_fwd_tensor, dim=0, **self.factory_kwargs)
            hn_all.append(h_prev)

            if self.bidirectional:
                idx_bwd = idx_fwd + 1
                h_prev_b = states[idx_bwd]
                cell_bwd = layer_cells['bwd']
                y_bwd_list = []
                for t in range(seq_len - 1, -1, -1):
                    inp_t = current_input[t]
                    y_t_b, h_prev_b, W = cell_bwd(inp_t, h_prev_b)
                    y_bwd_list.append(y_t_b)
                y_bwd_list.reverse()

                out_bwd_tensor = torch.stack([y.tensor for y in y_bwd_list], dim=1)
                out_bwd = Complex(out_bwd_tensor, dim=0, **self.factory_kwargs)
                hn_all.append(h_prev_b)

                output_tensor = torch.cat([out_fwd.tensor, out_bwd.tensor], dim=-1)
                output = Complex(output_tensor, dim=0, **self.factory_kwargs)
            else:
                output = out_fwd
            current_input = output

        if self.batch_first:
            if hasattr(current_input, 'tensor'):
                t = current_input.tensor
                t = t.transpose(1, 2)
                current_input = Complex(t, dim=0, **self.factory_kwargs)

        return current_input, hn_all


class AntiSymRNNModule(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 func: Union[str, nn.Module, DLModule, Callable] = "sigmoid",
                 bias: bool = True,
                 proj_size: int = None,
                 epsilon: float = 0.5,
                 gamma: float = 0.01,
                 tensor_type: str = "real",
                 dim: Optional[int] = None,
                 arrangement: str = "split",
                 is_stacked_flag: bool = False,
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

        for layer_idx in range(num_layers):
            if layer_idx == 0:
                layer_in = input_size
            else:
                prev_out = self.proj_size if self.proj_size > 0 else hidden_size
                layer_in = prev_out * self.num_directions

            layer_cells = nn.ModuleDict()
            layer_cells['fwd'] = AntiSymRNNCell(
                input_size=layer_in,
                hidden_size=hidden_size,
                func=func,
                bias=bias,
                proj_size=proj_size,
                epsilon=epsilon,
                gamma=gamma,
                tensor_type=tensor_type,
                dim=dim,
                arrangement=arrangement,
                is_stacked_flag=is_stacked_flag,
                **self.factory_kwargs,
                **kwargs
            )
            if bidirectional:
                layer_cells['bwd'] = AntiSymRNNCell(
                    input_size=layer_in,
                    hidden_size=hidden_size,
                    func=func,
                    bias=bias,
                    proj_size=proj_size,
                    epsilon=epsilon,
                    gamma=gamma,
                    tensor_type=tensor_type,
                    dim=dim,
                    arrangement=arrangement,
                    is_stacked_flag=is_stacked_flag,
                    **self.factory_kwargs,
                    **kwargs
                )
            self.layers.append(layer_cells)

    def forward(self, x: Union[torch.Tensor, Any], h0: Optional[Union[torch.Tensor, Any]] = None):
        if self.batch_first and isinstance(x, torch.Tensor):
            x = x.transpose(0, 1)

        if hasattr(x, 'shape'):
            seq_len = x.shape[0]
            batch_size = x.shape[1]
        else:
            seq_len = x.tensor.shape[1]
            batch_size = x.tensor.shape[2]

        states = [None] * (self.num_layers * self.num_directions)
        if h0 is not None:
            if isinstance(h0, (list, tuple)):
                states = list(h0)
            elif isinstance(h0, torch.Tensor):
                states = [h0[i] for i in range(h0.shape[0])]
            else:
                states = [h0] * (self.num_layers * self.num_directions)

        current_input = x
        hn_all = []

        for layer_idx, layer_cells in enumerate(self.layers):
            idx_fwd = layer_idx * self.num_directions
            h_prev = states[idx_fwd]

            cell_fwd = layer_cells['fwd']
            y_list = []

            for t in range(seq_len):
                inp_t = current_input[t] if hasattr(current_input, 'shape') else current_input[t]
                y_t, h_prev, W = cell_fwd(inp_t, h_prev)
                y_list.append(y_t)

            is_complex = hasattr(y_list[0], 'tensor')
            if is_complex:
                from .....models.deep_learning.activations.Complex.complex_ import Complex
                stacked = torch.stack([y.tensor for y in y_list], dim=1)
                out_fwd = Complex(stacked, dim=0, **self.factory_kwargs)
            else:
                out_fwd = torch.stack(y_list, dim=0)

            hn_all.append(h_prev)

            if self.bidirectional:
                idx_bwd = idx_fwd + 1
                h_prev_b = states[idx_bwd]
                if h0 is not None:
                    try:
                        h_prev_b = h0[idx_bwd]
                    except Exception:
                        pass

                cell_bwd = layer_cells['bwd']
                y_bwd_list = []
                for t in range(seq_len - 1, -1, -1):
                    inp_t = current_input[t] if hasattr(current_input, 'shape') else current_input[t]
                    y_t_b, h_prev_b, W = cell_bwd(inp_t, h_prev_b)
                    y_bwd_list.append(y_t_b)
                y_bwd_list.reverse()

                if is_complex:
                    stacked_b = torch.stack([y.tensor for y in y_bwd_list], dim=1)
                    out_bwd = Complex(stacked_b, dim=0, **self.factory_kwargs)
                    output_tensor = torch.cat([out_fwd.tensor, out_bwd.tensor], dim=-1)
                    output = Complex(output_tensor, dim=0, **self.factory_kwargs)
                else:
                    out_bwd = torch.stack(y_bwd_list, dim=0)
                    output = torch.cat([out_fwd, out_bwd], dim=2)
                hn_all.append(h_prev_b)
            else:
                output = out_fwd

            if self.dropout is not None and layer_idx < self.num_layers - 1 and not is_complex:
                output = self.dropout(output)
            current_input = output

        if isinstance(hn_all[0], torch.Tensor):
            hn = torch.stack(hn_all, dim=0)
        else:
            hn = hn_all

        if self.batch_first and isinstance(current_input, torch.Tensor):
            current_input = current_input.transpose(0, 1)

        return current_input, hn


class CTRNNModule(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 funcs: Union[str, Callable, nn.Module, DLModule] = None,
                 bias: bool = True,
                 proj_size: int = None,
                 alpha: float = 0.1,
                 tau: float = 1.0,
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
        self.alpha = alpha
        self.tau = tau

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.layers = nn.ModuleList()

        for layer_idx in range(num_layers):
            if layer_idx == 0:
                layer_in = input_size
            else:
                prev_out = self.proj_size if self.proj_size > 0 else hidden_size
                layer_in = prev_out * self.num_directions

            layer_cells = nn.ModuleDict()
            layer_cells['fwd'] = CTRNNCell(
                input_size=layer_in,
                hidden_size=hidden_size,
                funcs=funcs,
                bias=bias,
                proj_size=proj_size,
                time_delta=alpha,
                **self.factory_kwargs,
                **kwargs
            )
            if bidirectional:
                layer_cells['bwd'] = CTRNNCell(
                    input_size=layer_in,
                    hidden_size=hidden_size,
                    funcs=funcs,
                    bias=bias,
                    proj_size=proj_size,
                    time_delta=alpha,
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


class StackRNNModule(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 funcs: Union[List, Tuple, Dict] = None,
                 bias: bool = True,
                 proj_size: int = None,
                 stack_width: int = 10,
                 stack_depth: int = 10,
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
        self.stack_width = stack_width
        self.stack_depth = stack_depth

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.layers = nn.ModuleList()

        for layer_idx in range(num_layers):
            if layer_idx == 0:
                layer_in = input_size
            else:
                prev_out = self.proj_size if self.proj_size > 0 else hidden_size
                layer_in = prev_out * self.num_directions

            layer_cells = nn.ModuleDict()
            layer_cells['fwd'] = StackRNNCell(
                input_size=layer_in,
                hidden_size=hidden_size,
                funcs=funcs,
                bias=bias,
                proj_size=proj_size,
                stack_size=stack_width,
                max_stack_vectors=stack_depth,
                **self.factory_kwargs,
                **kwargs
            )
            if bidirectional:
                layer_cells['bwd'] = StackRNNCell(
                    input_size=layer_in,
                    hidden_size=hidden_size,
                    funcs=funcs,
                    bias=bias,
                    proj_size=proj_size,
                    stack_size=stack_width,
                    max_stack_vectors=stack_depth,
                    **self.factory_kwargs,
                    **kwargs
                )
            self.layers.append(layer_cells)

    def forward(self, x: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None):
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape

        if hx is None:
            num = self.num_layers * self.num_directions
            h0 = torch.zeros(num, batch_size, self.hidden_size, **self.factory_kwargs)
            s0 = torch.zeros(num, batch_size, self.layers[0]['fwd'].k * self.stack_width, **self.factory_kwargs)
            M0 = torch.zeros(num, batch_size, self.stack_depth, self.stack_width, **self.factory_kwargs)
            Str0 = torch.zeros(num, batch_size, self.stack_depth, **self.factory_kwargs)
        else:
            h0, s0, M0, Str0 = hx

        current_input = x
        hn_all = []
        sn_all = []
        Mn_all = []
        Strn_all = []

        for layer_idx, layer_cells in enumerate(self.layers):
            idx_fwd = layer_idx * self.num_directions
            h_prev = h0[idx_fwd]
            s_prev = s0[idx_fwd]
            M_prev = M0[idx_fwd]
            Str_prev = Str0[idx_fwd]

            cell_fwd = layer_cells['fwd']
            h_list = []

            for t in range(seq_len):
                inp_t = current_input[t]
                h_prev, s_prev, M_prev, Str_prev = cell_fwd(inp_t, h_prev, s_prev, M_prev, Str_prev)
                h_list.append(h_prev)

            out_fwd = torch.stack(h_list, dim=0)
            hn_all.append(h_prev)
            sn_all.append(s_prev)
            Mn_all.append(M_prev)
            Strn_all.append(Str_prev)

            if self.bidirectional:
                idx_bwd = idx_fwd + 1
                h_prev_b = h0[idx_bwd]
                s_prev_b = s0[idx_bwd]
                M_prev_b = M0[idx_bwd]
                Str_prev_b = Str0[idx_bwd]

                cell_bwd = layer_cells['bwd']
                h_bwd_list = []
                for t in range(seq_len - 1, -1, -1):
                    inp_t = current_input[t]
                    h_prev_b, s_prev_b, M_prev_b, Str_prev_b = cell_bwd(inp_t, h_prev_b, s_prev_b, M_prev_b, Str_prev_b)
                    h_bwd_list.append(h_prev_b)
                h_bwd_list.reverse()
                out_bwd = torch.stack(h_bwd_list, dim=0)
                hn_all.append(h_prev_b)
                sn_all.append(s_prev_b)
                Mn_all.append(M_prev_b)
                Strn_all.append(Str_prev_b)
                output = torch.cat([out_fwd, out_bwd], dim=2)
            else:
                output = out_fwd

            if self.dropout is not None and layer_idx < self.num_layers - 1:
                output = self.dropout(output)
            current_input = output

        hn = torch.stack(hn_all, dim=0)
        sn = torch.stack(sn_all, dim=0)
        Mn = torch.stack(Mn_all, dim=0)
        Strn = torch.stack(Strn_all, dim=0)

        if self.batch_first:
            current_input = current_input.transpose(0, 1)
        return current_input, (hn, sn, Mn, Strn)


class VariationalRecurrentUnitModule(_BaseRNNModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 latent_size: int,
                 z_dim: int,
                 num_layers: int,
                 non_linearity: Union[str, Callable, nn.Module] = 'tanh',
                 funcs: Union[List, Tuple, Dict] = None,
                 enc_funcs: Union[List, Tuple, Dict] = None,
                 dec_funcs: Union[List, Tuple, Dict] = None,
                 bias: bool = True,
                 cell_type: str = "rnn",
                 generative: bool = False,
                 proj_size: int = 0,
                 dropout: float = 0.0,
                 bidirectional: bool = False,
                 batch_first: bool = False,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.dropout_val = dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        self.funcs_exp = self._expand_arg(funcs, num_layers, bidirectional, is_container=True)
        self.enc_funcs_exp = self._expand_arg(enc_funcs, num_layers, bidirectional, is_container=True)
        self.dec_funcs_exp = self._expand_arg(dec_funcs, num_layers, bidirectional, is_container=True)

        self.layers = nn.ModuleList()
        directions = 2 if bidirectional else 1
        layer_ins = self._init_layer_args(input_size, hidden_size, num_layers, directions, proj_size or 0)

        k = 0
        for i in range(num_layers):
            l_dict = nn.ModuleDict()
            l_dict['fwd'] = VariationalRecurrentUnitCell(
                input_size=layer_ins[i], hidden_size=hidden_size, latent_size=latent_size, z_dim=z_dim,
                non_linearity=non_linearity, funcs=self.funcs_exp[k], enc_funcs=self.enc_funcs_exp[k],
                dec_funcs=self.dec_funcs_exp[k], bias=bias, cell_type=cell_type, generative=generative,
                proj_size=proj_size, **self.factory_kwargs, **kwargs
            )
            k += 1
            if bidirectional:
                l_dict['bwd'] = VariationalRecurrentUnitCell(
                    input_size=layer_ins[i], hidden_size=hidden_size, latent_size=latent_size, z_dim=z_dim,
                    non_linearity=non_linearity, funcs=self.funcs_exp[k], enc_funcs=self.enc_funcs_exp[k],
                    dec_funcs=self.dec_funcs_exp[k], bias=bias, cell_type=cell_type, generative=generative,
                    proj_size=proj_size, **self.factory_kwargs, **kwargs
                )
                k += 1
            self.layers.append(l_dict)

    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None):
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch, _ = x.shape
        curr_x = x

        for layer_idx, layer in enumerate(self.layers):
            cell = layer['fwd']
            out_seq = []
            h_state, c_state = None, None

            for t in range(seq_len):
                res = cell(curr_x[t], h_state, c_state)
                if isinstance(res, tuple):
                    h_next, c_next = res
                    c_state = c_next
                else:
                    h_next = res
                h_state = h_next
                out_seq.append(h_next)

            fwd_out = torch.stack(out_seq, dim=0)

            if self.bidirectional:
                cell_b = layer['bwd']
                out_seq_b = []
                h_state_b, c_state_b = None, None
                for t in range(seq_len - 1, -1, -1):
                    res = cell_b(curr_x[t], h_state_b, c_state_b)
                    if isinstance(res, tuple):
                        h_next, c_next = res
                        c_state_b = c_next
                    else:
                        h_next = res
                    h_state_b = h_next
                    out_seq_b.append(h_next)
                out_seq_b.reverse()
                bwd_out = torch.stack(out_seq_b, dim=0)
                curr_x = torch.cat([fwd_out, bwd_out], dim=2)
            else:
                curr_x = fwd_out

            if self.dropout and layer_idx < self.num_layers - 1:
                curr_x = self.dropout(curr_x)

        if self.batch_first:
            curr_x = curr_x.transpose(0, 1)
        return curr_x


class NARXModule(_BaseRNNModule):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 proj_size: int = None, non_linearity: Union[str, Callable] = 'tanh',
                 bias: bool = True, dropout: float = 0.0, bidirectional: bool = False,
                 batch_first: bool = False, device: str = "cpu", dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        self.nl_exp = self._expand_arg(non_linearity, num_layers, bidirectional)
        layer_ins = self._init_layer_args(input_size, hidden_size, num_layers, 2 if bidirectional else 1, proj_size or 0)

        self.layers = nn.ModuleList()
        k = 0
        for i in range(num_layers):
            ld = nn.ModuleDict()
            ld['fwd'] = NARXCell(layer_ins[i], hidden_size, proj_size, self.nl_exp[k], bias, **self.factory_kwargs)
            k += 1
            if bidirectional:
                ld['bwd'] = NARXCell(layer_ins[i], hidden_size, proj_size, self.nl_exp[k], bias, **self.factory_kwargs)
                k += 1
            self.layers.append(ld)

    def forward(self, x, h0=None):
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, _, _ = x.shape
        curr_x = x
        for i, layer in enumerate(self.layers):
            out_seq = []
            h = None
            for t in range(seq_len):
                y, h = layer['fwd'](curr_x[t], h)
                out_seq.append(y)
            fwd = torch.stack(out_seq, dim=0)

            if self.bidirectional:
                out_seq_b = []
                h_b = None
                for t in range(seq_len - 1, -1, -1):
                    y, h_b = layer['bwd'](curr_x[t], h_b)
                    out_seq_b.append(y)
                out_seq_b.reverse()
                bwd = torch.stack(out_seq_b, dim=0)
                curr_x = torch.cat([fwd, bwd], dim=2)
            else:
                curr_x = fwd
            if self.dropout and i < self.num_layers - 1:
                curr_x = self.dropout(curr_x)
        if self.batch_first:
            curr_x = curr_x.transpose(0, 1)
        return curr_x


class MorgifierRecurrentUnitModule(_BaseRNNModule):
    def __init__(self, input_size, hidden_size, num_layers, num_rounds=3, non_linearity='tanh',
                 funcs=None, bias=True, cell_type="lstm", mogrification_funcs="sigmoid", proj_size=None,
                 dropout=0.0, bidirectional=False, batch_first=False, device="cpu", dtype=torch.float32, *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        funcs_exp = self._expand_arg(funcs, num_layers, bidirectional, is_container=True)
        nl_exp = self._expand_arg(non_linearity, num_layers, bidirectional)
        mog_funcs_exp = self._expand_arg(mogrification_funcs, num_layers, bidirectional)

        rounds_exp = self._expand_arg(num_rounds, num_layers, bidirectional)
        cell_type_exp = self._expand_arg(cell_type, num_layers, bidirectional)

        l_ins = self._init_layer_args(input_size, hidden_size, num_layers, 2 if bidirectional else 1, proj_size)

        self.layers = nn.ModuleList()
        k = 0
        for i in range(num_layers):
            ld = nn.ModuleDict()
            ld['fwd'] = MorgifierRecurrentUnitCell(
                l_ins[i], hidden_size, rounds_exp[k], nl_exp[k], funcs_exp[k], bias,
                cell_type_exp[k], mog_funcs_exp[k], proj_size, **self.factory_kwargs
            )
            k += 1
            if bidirectional:
                ld['bwd'] = MorgifierRecurrentUnitCell(
                    l_ins[i], hidden_size, rounds_exp[k], nl_exp[k], funcs_exp[k], bias,
                    cell_type_exp[k], mog_funcs_exp[k], proj_size, **self.factory_kwargs
                )
                k += 1
            self.layers.append(ld)

    def forward(self, x, h0=None):
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len = x.size(0)
        curr = x
        for i, layer in enumerate(self.layers):
            fwd_out = []
            h_state, c_state = None, None
            for t in range(seq_len):
                res = layer['fwd'](curr[t], h_state, c_state)
                if isinstance(res, tuple):
                    h_next, c_next = res
                    c_state = c_next
                else:
                    h_next = res
                h_state = h_next
                fwd_out.append(h_next)
            fwd = torch.stack(fwd_out, dim=0)

            if self.bidirectional:
                bwd_out = []
                h_state_b, c_state_b = None, None
                for t in range(seq_len - 1, -1, -1):
                    res = layer['bwd'](curr[t], h_state_b, c_state_b)
                    if isinstance(res, tuple):
                        h_next_b, c_next_b = res
                        c_state_b = c_next_b
                    else:
                        h_next_b = res
                    h_state_b = h_next_b
                    bwd_out.append(h_next_b)
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

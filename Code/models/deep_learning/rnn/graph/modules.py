from typing import Callable
"""Graph RNN modules: GraphRecurrentUnitModule, DynamicGraphRecurrentUnitModule."""
import torch
import torch.nn as nn
from typing import Optional, Union, List, Tuple, Dict
from .....models.utils import DLModule

from .cells import GraphRecurrentUnitCell, DynamicGraphRecurrentUnitCell
from .._base import _BaseRNNModule


class GraphRecurrentUnitModule(_BaseRNNModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 non_linearity: Union[str, Callable, nn.Module] = 'tanh',
                 funcs: Union[List, Tuple, Dict] = None,
                 bias: bool = True,
                 proj_size: int = None,
                 dropout: float = 0.0,
                 bidirectional: bool = False,
                 batch_first: bool = False,
                 cell_type: str = "rnn",
                 gcn_type: str = "gcn",
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
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        directions = 2 if bidirectional else 1
        self.funcs_exp = self._expand_arg(funcs, num_layers, bidirectional, is_container=True)
        self.cell_type_exp = self._expand_arg(cell_type, num_layers, bidirectional)

        self.layers = nn.ModuleList()
        self.proj_size = proj_size
        eff_hidden = proj_size if proj_size is not None and proj_size > 0 else hidden_size

        layer_ins = []
        for i in range(num_layers):
            if i == 0:
                layer_ins.append(input_size)
            else:
                layer_ins.append(eff_hidden * directions)

        k = 0
        for i in range(num_layers):
            l_dict = nn.ModuleDict()
            l_dict['fwd'] = GraphRecurrentUnitCell(
                input_size=layer_ins[i],
                hidden_size=hidden_size,
                non_linearity=non_linearity,
                funcs=self.funcs_exp[k],
                bias=bias,
                proj_size=proj_size,
                cell_type=self.cell_type_exp[k],
                gcn_type=gcn_type,
                **self.factory_kwargs, **kwargs
            )
            k += 1
            if bidirectional:
                l_dict['bwd'] = GraphRecurrentUnitCell(
                    input_size=layer_ins[i],
                    hidden_size=hidden_size,
                    non_linearity=non_linearity,
                    funcs=self.funcs_exp[k],
                    bias=bias,
                    proj_size=proj_size,
                    cell_type=self.cell_type_exp[k],
                    gcn_type=gcn_type,
                    **self.factory_kwargs, **kwargs
                )
                k += 1
            self.layers.append(l_dict)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, h_0: Optional[torch.Tensor] = None):
        if self.batch_first:
            x = x.transpose(0, 1)

        time_steps, batch_size, num_nodes, _ = x.shape
        current_input = x
        all_layer_outputs = []
        final_h_n = []

        for layer_idx, layer in enumerate(self.layers):
            cell = layer['fwd']
            h_state = None
            c_state = None
            outputs = []

            for t in range(time_steps):
                inp = current_input[t]
                h_prev_in = (h_state, c_state) if "lstm" in cell.cell_type and h_state is not None else h_state

                res = cell(inp, adj, h_prev_in)

                if isinstance(res, tuple):
                    h_next, c_next = res
                    c_state = c_next
                else:
                    h_next = res
                h_state = h_next
                outputs.append(h_next)

            out_fwd = torch.stack(outputs, dim=0)
            final_h_n.append(h_state)

            if self.bidirectional:
                cell_b = layer['bwd']
                h_state_b = None
                c_state_b = None
                outputs_b = []

                for t in range(time_steps - 1, -1, -1):
                    inp = current_input[t]
                    h_prev_in = (h_state_b, c_state_b) if "lstm" in cell_b.cell_type and h_state_b is not None else h_state_b

                    res = cell_b(inp, adj, h_prev_in)

                    if isinstance(res, tuple):
                        h_next, c_next = res
                        c_state_b = c_next
                    else:
                        h_next = res
                    h_state_b = h_next
                    outputs_b.append(h_next)

                outputs_b.reverse()
                out_bwd = torch.stack(outputs_b, dim=0)
                final_h_n.append(h_state_b)

                current_output = torch.cat([out_fwd, out_bwd], dim=-1)
            else:
                current_output = out_fwd

            if self.dropout and layer_idx < self.num_layers - 1:
                current_output = self.dropout(current_output)

            current_input = current_output

        if self.batch_first:
            current_input = current_input.transpose(0, 1)

        return current_input, final_h_n


class DynamicGraphRecurrentUnitModule(_BaseRNNModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 non_linearity: Union[str, Callable, nn.Module] = 'tanh',
                 funcs: Union[List, Tuple, Dict] = None,
                 bias: bool = True,
                 proj_size: int = None,
                 dropout: float = 0.0,
                 bidirectional: bool = False,
                 batch_first: bool = False,
                 cell_type: str = "rnn",
                 sim_type: str = "attn",
                 dist_type: str = "dist",
                 p: int = 2,
                 gamma: float = 1.0,
                 beta: float = None,
                 max_iter: int = 100,
                 soft_threshold: float = 0.5,
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
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        directions = 2 if bidirectional else 1
        self.funcs_exp = self._expand_arg(funcs, num_layers, bidirectional, is_container=True)
        self.cell_type_exp = self._expand_arg(cell_type, num_layers, bidirectional)

        self.layers = nn.ModuleList()
        self.proj_size = proj_size
        eff_hidden = proj_size if proj_size is not None and proj_size > 0 else hidden_size

        layer_ins = []
        for i in range(num_layers):
            if i == 0:
                layer_ins.append(input_size)
            else:
                layer_ins.append(eff_hidden * directions)

        k = 0
        for i in range(num_layers):
            l_dict = nn.ModuleDict()
            l_dict['fwd'] = DynamicGraphRecurrentUnitCell(
                input_size=layer_ins[i],
                hidden_size=hidden_size,
                non_linearity=non_linearity,
                funcs=self.funcs_exp[k],
                bias=bias,
                proj_size=proj_size,
                cell_type=self.cell_type_exp[k],
                sim_type=sim_type,
                dist_type=dist_type,
                p=p,
                gamma=gamma,
                beta=beta,
                max_iter=max_iter,
                soft_threshold=soft_threshold,
                **self.factory_kwargs, **kwargs
            )
            k += 1
            if bidirectional:
                l_dict['bwd'] = DynamicGraphRecurrentUnitCell(
                    input_size=layer_ins[i],
                    hidden_size=hidden_size,
                    non_linearity=non_linearity,
                    funcs=self.funcs_exp[k],
                    bias=bias,
                    proj_size=proj_size,
                    cell_type=self.cell_type_exp[k],
                    sim_type=sim_type,
                    dist_type=dist_type,
                    p=p,
                    gamma=gamma,
                    beta=beta,
                    max_iter=max_iter,
                    soft_threshold=soft_threshold,
                    **self.factory_kwargs, **kwargs
                )
                k += 1
            self.layers.append(l_dict)

    def forward(self, x: torch.Tensor, h_0: Optional[torch.Tensor] = None):
        if self.batch_first:
            x = x.transpose(0, 1)

        time_steps, batch_size, num_nodes, _ = x.shape
        current_input = x
        final_h_n = []

        for layer_idx, layer in enumerate(self.layers):
            cell = layer['fwd']
            h_state = None
            c_state = None
            outputs = []

            for t in range(time_steps):
                inp = current_input[t]
                h_prev_in = (h_state, c_state) if "lstm" in cell.cell_type and h_state is not None else h_state

                res = cell(inp, h_prev_in)

                if isinstance(res, tuple):
                    h_next, c_next = res
                    c_state = c_next
                else:
                    h_next = res
                h_state = h_next
                outputs.append(h_next)

            out_fwd = torch.stack(outputs, dim=0)
            final_h_n.append(h_state)

            if self.bidirectional:
                cell_b = layer['bwd']
                h_state_b = None
                c_state_b = None
                outputs_b = []

                for t in range(time_steps - 1, -1, -1):
                    inp = current_input[t]
                    h_prev_in = (h_state_b, c_state_b) if "lstm" in cell_b.cell_type and h_state_b is not None else h_state_b

                    res = cell_b(inp, h_prev_in)

                    if isinstance(res, tuple):
                        h_next, c_next = res
                        c_state_b = c_next
                    else:
                        h_next = res
                    h_state_b = h_next
                    outputs_b.append(h_next)

                outputs_b.reverse()
                out_bwd = torch.stack(outputs_b, dim=0)
                final_h_n.append(h_state_b)
                current_output = torch.cat([out_fwd, out_bwd], dim=-1)
            else:
                current_output = out_fwd

            if self.dropout and layer_idx < self.num_layers - 1:
                current_output = self.dropout(current_output)
            current_input = current_output

        if self.batch_first:
            current_input = current_input.transpose(0, 1)

        return current_input, final_h_n

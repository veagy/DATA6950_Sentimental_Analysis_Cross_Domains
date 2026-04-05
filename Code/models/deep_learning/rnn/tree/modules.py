from typing import Callable
"""Tree RNN modules: TreeRNNModule, TreeLSTMModule, TreeGRUModule."""
import torch
import torch.nn as nn
from typing import Optional, Union, List, Tuple, Dict, Any
from .....models.utils import DLModule

from .cells import TreeRNNCell, TreeLSTMCell, TreeGRUCell
from .._base import _BaseRNNModule


class TreeRNNModule(_BaseRNNModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 n_branching: int = 2,
                 non_linearity: Union[str, Callable, nn.Module] = 'tanh',
                 funcs: Union[List, Tuple, Dict] = None,
                 bias: bool = True,
                 proj_size: int = None,
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
        self.proj_size = proj_size
        self.num_directions = 2 if bidirectional else 1

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        self.funcs_expanded = self._expand_arg(funcs, num_layers, bidirectional, is_container=True)
        self.non_linearity_expanded = self._expand_arg(non_linearity, num_layers, bidirectional)
        self.n_branching_expanded = self._expand_arg(n_branching, num_layers, bidirectional)

        self.layers = nn.ModuleList()

        layer_ins = []
        for i in range(num_layers):
            if i == 0:
                layer_ins.append(input_size)
            else:
                hs = proj_size if proj_size is not None and proj_size > 0 else hidden_size
                layer_ins.append(hs)

        k = 0
        for i in range(num_layers):
            num_nodes = self.n_branching_expanded[0] ** i

            level_cells = nn.ModuleDict()

            fwd_nodes = nn.ModuleList()
            for _ in range(num_nodes):
                fwd_nodes.append(TreeRNNCell(
                    input_size=layer_ins[i],
                    hidden_size=hidden_size,
                    non_linearity=self.non_linearity_expanded[k],
                    funcs=self.funcs_expanded[k],
                    n_branching=self.n_branching_expanded[i],
                    bias=bias,
                    proj_size=proj_size,
                    **self.factory_kwargs,
                    **kwargs
                ))
            level_cells['fwd'] = fwd_nodes
            k += 1

            if bidirectional:
                bwd_nodes = nn.ModuleList()
                for _ in range(num_nodes):
                    bwd_nodes.append(TreeRNNCell(
                        input_size=layer_ins[i],
                        hidden_size=hidden_size,
                        non_linearity=self.non_linearity_expanded[k],
                        funcs=self.funcs_expanded[k],
                        n_branching=self.n_branching_expanded[i],
                        bias=bias,
                        proj_size=proj_size,
                        **self.factory_kwargs,
                        **kwargs
                    ))
                level_cells['bwd'] = bwd_nodes
                k += 1

            self.layers.append(level_cells)

    def forward(self, x: torch.Tensor, h_prev: Optional[List[torch.Tensor]] = None):
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.shape

        current_inputs = [x]

        for layer_idx, layer in enumerate(self.layers):
            nodes = layer['fwd']
            next_inputs = []

            if len(current_inputs) != len(nodes):
                raise RuntimeError(f"TreeRNN topology mismatch at layer {layer_idx}.")

            for node_idx, node in enumerate(nodes):
                inp_seq = current_inputs[node_idx]
                h_state = None
                outs_per_step = []
                for t in range(seq_len):
                    inp = inp_seq[t]
                    h_out_list = node(inp, h_state)
                    h_state = h_out_list
                    outs_per_step.append(h_out_list)

                n_branch = len(outs_per_step[0])
                for b in range(n_branch):
                    branch_seq = torch.stack([step_out[b] for step_out in outs_per_step], dim=0)
                    next_inputs.append(branch_seq)

            current_inputs = next_inputs

            if self.dropout is not None and layer_idx < self.num_layers - 1:
                current_inputs = [self.dropout(inp) for inp in current_inputs]

        fwd_leaves = current_inputs
        fwd_out = torch.cat(fwd_leaves, dim=-1)

        out_final = fwd_out

        if self.bidirectional:
            x_bwd = torch.flip(x, [0])
            current_inputs = [x_bwd]

            for layer_idx, layer in enumerate(self.layers):
                nodes = layer['bwd']
                next_inputs = []

                if len(current_inputs) != len(nodes):
                    raise RuntimeError(f"TreeRNN Bwd topology mismatch.")

                for node_idx, node in enumerate(nodes):
                    inp_seq = current_inputs[node_idx]
                    h_state = None
                    outs_per_step = []
                    for t in range(seq_len):
                        inp = inp_seq[t]
                        h_out_list = node(inp, h_state)
                        h_state = h_out_list
                        outs_per_step.append(h_out_list)

                    n_branch = len(outs_per_step[0])
                    for b in range(n_branch):
                        branch_seq = torch.stack([step_out[b] for step_out in outs_per_step], dim=0)
                        next_inputs.append(branch_seq)

                current_inputs = next_inputs
                if self.dropout is not None and layer_idx < self.num_layers - 1:
                    current_inputs = [self.dropout(inp) for inp in current_inputs]

            bwd_leaves = [torch.flip(leaf_seq, [0]) for leaf_seq in current_inputs]
            bwd_out = torch.cat(bwd_leaves, dim=-1)

            out_final = torch.cat([fwd_out, bwd_out], dim=-1)

        if self.batch_first:
            out_final = out_final.transpose(0, 1)

        return out_final, None


class TreeLSTMModule(_BaseRNNModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 n_branching: int = 2,
                 funcs: Union[List, Tuple, Dict] = None,
                 bias: bool = True,
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
        self.num_directions = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        self.funcs_expanded = self._expand_arg(funcs, num_layers, bidirectional, is_container=True)
        self.n_branching_expanded = self._expand_arg(n_branching, num_layers, bidirectional)

        self.layers = nn.ModuleList()
        layer_ins = []
        for i in range(num_layers):
            if i == 0:
                layer_ins.append(input_size)
            else:
                hs = proj_size if proj_size is not None and proj_size > 0 else hidden_size
                layer_ins.append(hs)

        k = 0
        for i in range(num_layers):
            num_nodes = self.n_branching_expanded[0] ** i
            level_cells = nn.ModuleDict()

            fwd_nodes = nn.ModuleList()
            for _ in range(num_nodes):
                fwd_nodes.append(TreeLSTMCell(
                    input_size=layer_ins[i],
                    hidden_size=hidden_size,
                    funcs=self.funcs_expanded[k],
                    n_branching=self.n_branching_expanded[i],
                    bias=bias,
                    proj_size=proj_size,
                    **self.factory_kwargs,
                    **kwargs
                ))
            level_cells['fwd'] = fwd_nodes
            k += 1
            if bidirectional:
                bwd_nodes = nn.ModuleList()
                for _ in range(num_nodes):
                    bwd_nodes.append(TreeLSTMCell(
                        input_size=layer_ins[i],
                        hidden_size=hidden_size,
                        funcs=self.funcs_expanded[k],
                        n_branching=self.n_branching_expanded[i],
                        bias=bias,
                        proj_size=proj_size,
                        **self.factory_kwargs,
                        **kwargs
                    ))
                level_cells['bwd'] = bwd_nodes
                k += 1
            self.layers.append(level_cells)

    def forward(self, x: torch.Tensor, state: Any = None):
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.shape

        current_inputs = [x]

        for layer_idx, layer in enumerate(self.layers):
            nodes = layer['fwd']
            next_inputs = []

            if len(current_inputs) != len(nodes):
                raise RuntimeError(f"TreeLSTM topology mismatch at layer {layer_idx}.")

            for node_idx, node in enumerate(nodes):
                inp_seq = current_inputs[node_idx]
                h_list, c_list = None, None
                outs_per_step = []

                for t in range(seq_len):
                    inp = inp_seq[t]
                    h_out, c_out = node(inp, h_list, c_list)
                    h_list, c_list = h_out, c_out
                    outs_per_step.append(h_out)

                n_branch = len(outs_per_step[0])
                for b in range(n_branch):
                    branch_seq = torch.stack([step_out[b] for step_out in outs_per_step], dim=0)
                    next_inputs.append(branch_seq)

            current_inputs = next_inputs
            if self.dropout is not None and layer_idx < self.num_layers - 1:
                current_inputs = [self.dropout(inp) for inp in current_inputs]

        fwd_leaves = current_inputs
        fwd_out = torch.cat(fwd_leaves, dim=-1)
        out_final = fwd_out

        if self.bidirectional:
            x_bwd = torch.flip(x, [0])
            current_inputs = [x_bwd]

            for layer_idx, layer in enumerate(self.layers):
                nodes = layer['bwd']
                next_inputs = []

                for node_idx, node in enumerate(nodes):
                    inp_seq = current_inputs[node_idx]
                    h_list, c_list = None, None
                    outs_per_step = []

                    for t in range(seq_len):
                        inp = inp_seq[t]
                        h_out, c_out = node(inp, h_list, c_list)
                        h_list, c_list = h_out, c_out
                        outs_per_step.append(h_out)

                    n_branch = len(outs_per_step[0])
                    for b in range(n_branch):
                        branch_seq = torch.stack([step_out[b] for step_out in outs_per_step], dim=0)
                        next_inputs.append(branch_seq)

                current_inputs = next_inputs
                if self.dropout is not None and layer_idx < self.num_layers - 1:
                    current_inputs = [self.dropout(inp) for inp in current_inputs]

            bwd_leaves = [torch.flip(leaf_seq, [0]) for leaf_seq in current_inputs]
            bwd_out = torch.cat(bwd_leaves, dim=-1)
            out_final = torch.cat([fwd_out, bwd_out], dim=-1)

        if self.batch_first:
            current_input = out_final.transpose(0, 1)
        else:
            current_input = out_final

        return current_input, None


class TreeGRUModule(_BaseRNNModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 n_branching: int = 2,
                 funcs: Union[List, Tuple, Dict] = None,
                 bias: bool = True,
                 proj_size: int = None,
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
        self.num_directions = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        self.funcs_expanded = self._expand_arg(funcs, num_layers, bidirectional, is_container=True)
        self.n_branching_expanded = self._expand_arg(n_branching, num_layers, bidirectional)

        self.layers = nn.ModuleList()
        layer_ins = []
        for i in range(num_layers):
            if i == 0:
                layer_ins.append(input_size)
            else:
                hs = proj_size if proj_size is not None and proj_size > 0 else hidden_size
                layer_ins.append(hs)

        k = 0
        for i in range(num_layers):
            num_nodes = self.n_branching_expanded[0] ** i
            level_cells = nn.ModuleDict()
            fwd_nodes = nn.ModuleList()
            for _ in range(num_nodes):
                fwd_nodes.append(TreeGRUCell(
                    input_size=layer_ins[i],
                    hidden_size=hidden_size,
                    funcs=self.funcs_expanded[k],
                    n_branching=self.n_branching_expanded[i],
                    bias=bias,
                    proj_size=proj_size,
                    **self.factory_kwargs,
                    **kwargs
                ))
            level_cells['fwd'] = fwd_nodes
            k += 1

            if bidirectional:
                bwd_nodes = nn.ModuleList()
                for _ in range(num_nodes):
                    bwd_nodes.append(TreeGRUCell(
                        input_size=layer_ins[i],
                        hidden_size=hidden_size,
                        funcs=self.funcs_expanded[k],
                        n_branching=self.n_branching_expanded[i],
                        bias=bias,
                        proj_size=proj_size,
                        **self.factory_kwargs,
                        **kwargs
                    ))
                level_cells['bwd'] = bwd_nodes
                k += 1
            self.layers.append(level_cells)

    def forward(self, x: torch.Tensor, state: Any = None):
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.shape

        current_inputs = [x]

        for layer_idx, layer in enumerate(self.layers):
            nodes = layer['fwd']
            next_inputs = []

            if len(current_inputs) != len(nodes):
                raise RuntimeError(f"TreeGRU topology mismatch at layer {layer_idx}.")

            for node_idx, node in enumerate(nodes):
                inp_seq = current_inputs[node_idx]
                h_list = None
                outs_per_step = []

                for t in range(seq_len):
                    inp = inp_seq[t]
                    h_out = node(inp, h_list)
                    h_list = h_out
                    outs_per_step.append(h_out)

                n_branch = len(outs_per_step[0])
                for b in range(n_branch):
                    branch_seq = torch.stack([step_out[b] for step_out in outs_per_step], dim=0)
                    next_inputs.append(branch_seq)

            current_inputs = next_inputs
            if self.dropout is not None and layer_idx < self.num_layers - 1:
                current_inputs = [self.dropout(inp) for inp in current_inputs]

        fwd_leaves = current_inputs
        fwd_out = torch.cat(fwd_leaves, dim=-1)
        out_final = fwd_out

        if self.bidirectional:
            x_bwd = torch.flip(x, [0])
            current_inputs = [x_bwd]

            for layer_idx, layer in enumerate(self.layers):
                nodes = layer['bwd']
                next_inputs = []

                for node_idx, node in enumerate(nodes):
                    inp_seq = current_inputs[node_idx]
                    h_list = None
                    outs_per_step = []

                    for t in range(seq_len):
                        inp = inp_seq[t]
                        h_out = node(inp, h_list)
                        h_list = h_out
                        outs_per_step.append(h_out)

                    n_branch = len(outs_per_step[0])
                    for b in range(n_branch):
                        branch_seq = torch.stack([step_out[b] for step_out in outs_per_step], dim=0)
                        next_inputs.append(branch_seq)

                current_inputs = next_inputs
                if self.dropout is not None and layer_idx < self.num_layers - 1:
                    current_inputs = [self.dropout(inp) for inp in current_inputs]

            bwd_leaves = [torch.flip(leaf_seq, [0]) for leaf_seq in current_inputs]
            bwd_out = torch.cat(bwd_leaves, dim=-1)
            out_final = torch.cat([fwd_out, bwd_out], dim=-1)

        if self.batch_first:
            current_input = out_final.transpose(0, 1)
        else:
            current_input = out_final
        return current_input, None

from typing import Callable
"""Forest RNN modules: ForestRNNModule, ForestLSTMModule, ForestGRUModule."""
import torch
import torch.nn as nn
from typing import Optional, Union, List, Tuple, Dict, Any

from .....models.utils import DLModule

from ..tree import TreeRNNModule, TreeLSTMModule, TreeGRUModule


class ForestRNNModule(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 n_trees: int = 5,
                 tree_input_size: int = None,
                 tree_hidden_size: int = None,
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
        self.n_trees = n_trees
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.tree_input_size = tree_input_size if tree_input_size is not None else max(1, input_size // 2)
        self.tree_hidden_size = tree_hidden_size if tree_hidden_size is not None else hidden_size

        self.feature_selectors = nn.ModuleList([
            nn.Linear(input_size, self.tree_input_size, bias=bias, **self.factory_kwargs)
            for _ in range(n_trees)
        ])

        self.trees = nn.ModuleList([
            TreeRNNModule(
                input_size=self.tree_input_size,
                hidden_size=self.tree_hidden_size,
                num_layers=num_layers,
                n_branching=n_branching,
                non_linearity=non_linearity,
                funcs=funcs,
                bias=bias,
                proj_size=proj_size,
                dropout=dropout,
                bidirectional=bidirectional,
                batch_first=False,
                device=device,
                dtype=dtype,
                **kwargs
            )
            for _ in range(n_trees)
        ])

        num_directions = 2 if bidirectional else 1
        self.aggregator = nn.LazyLinear(
             out_features=proj_size if proj_size is not None else hidden_size,
             bias=bias,
             **self.factory_kwargs
        )

        self.proj_size = proj_size

    def forward(self, x: torch.Tensor, h_prev: Optional[List[Any]] = None):
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.shape

        tree_outputs = []

        for i in range(self.n_trees):
            selector = self.feature_selectors[i]
            x_tree = selector(x)

            h_tree_prev = h_prev[i] if h_prev is not None else None
            out_tree, _ = self.trees[i](x_tree, h_tree_prev)

            tree_outputs.append(out_tree)

        concat_out = torch.cat(tree_outputs, dim=-1)
        final_out = self.aggregator(concat_out)

        if self.batch_first:
            final_out = final_out.transpose(0, 1)

        return final_out, None


class ForestLSTMModule(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 n_trees: int = 5,
                 tree_input_size: int = None,
                 tree_hidden_size: int = None,
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
        self.n_trees = n_trees
        self.input_size = input_size
        self.batch_first = batch_first

        self.tree_input_size = tree_input_size if tree_input_size is not None else max(1, input_size // 2)
        self.tree_hidden_size = tree_hidden_size if tree_hidden_size is not None else hidden_size

        self.feature_selectors = nn.ModuleList([
            nn.Linear(input_size, self.tree_input_size, bias=bias, **self.factory_kwargs)
            for _ in range(n_trees)
        ])

        self.trees = nn.ModuleList([
            TreeLSTMModule(
                input_size=self.tree_input_size,
                hidden_size=self.tree_hidden_size,
                num_layers=num_layers,
                n_branching=n_branching,
                funcs=funcs,
                bias=bias,
                proj_size=proj_size,
                dropout=dropout,
                bidirectional=bidirectional,
                batch_first=False,
                device=device,
                dtype=dtype,
                **kwargs
            )
            for _ in range(n_trees)
        ])

        self.aggregator = nn.LazyLinear(
             out_features=proj_size if proj_size is not None and proj_size > 0 else hidden_size,
             bias=bias,
             **self.factory_kwargs
        )

    def forward(self, x: torch.Tensor, state: Any = None):
        if self.batch_first:
            x = x.transpose(0, 1)

        tree_outputs = []
        for i in range(self.n_trees):
            x_tree = self.feature_selectors[i](x)
            out_tree, _ = self.trees[i](x_tree, None)
            tree_outputs.append(out_tree)

        concat_out = torch.cat(tree_outputs, dim=-1)
        final_out = self.aggregator(concat_out)

        if self.batch_first:
            final_out = final_out.transpose(0, 1)
        return final_out, None


class ForestGRUModule(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 n_trees: int = 5,
                 tree_input_size: int = None,
                 tree_hidden_size: int = None,
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
        self.n_trees = n_trees
        self.input_size = input_size
        self.batch_first = batch_first

        self.tree_input_size = tree_input_size if tree_input_size is not None else max(1, input_size // 2)
        self.tree_hidden_size = tree_hidden_size if tree_hidden_size is not None else hidden_size

        self.feature_selectors = nn.ModuleList([
            nn.Linear(input_size, self.tree_input_size, bias=bias, **self.factory_kwargs)
            for _ in range(n_trees)
        ])

        self.trees = nn.ModuleList([
            TreeGRUModule(
                input_size=self.tree_input_size,
                hidden_size=self.tree_hidden_size,
                num_layers=num_layers,
                n_branching=n_branching,
                funcs=funcs,
                bias=bias,
                proj_size=proj_size,
                dropout=dropout,
                bidirectional=bidirectional,
                batch_first=False,
                device=device,
                dtype=dtype,
                **kwargs
            )
            for _ in range(n_trees)
        ])

        self.aggregator = nn.LazyLinear(
             out_features=proj_size if proj_size is not None else hidden_size,
             bias=bias,
             **self.factory_kwargs
        )

    def forward(self, x: torch.Tensor, state: Any = None):
        if self.batch_first:
            x = x.transpose(0, 1)

        tree_outputs = []
        for i in range(self.n_trees):
            x_tree = self.feature_selectors[i](x)
            out_tree, _ = self.trees[i](x_tree, None)
            tree_outputs.append(out_tree)

        concat_out = torch.cat(tree_outputs, dim=-1)
        final_out = self.aggregator(concat_out)

        if self.batch_first:
            final_out = final_out.transpose(0, 1)
        return final_out, None

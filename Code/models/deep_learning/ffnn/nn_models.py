import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Union, Any, Optional, Callable

from ....models.deep_learning.activations.ActivationFunction import Activation
from ....models.utils import DLModule
from ....models.deep_learning.models import DLModelLayers
from .nn_layers import KANLayer, SlimLinear


__all__ = [
    "FeedForwardNeuralNetwork",
    "FeedForwardNeuralNetworkOp",
    "KANLayer",
    "KANNetwork",
    "SlimLinear",
    "CustomLinearNetworkModule",
]


def _resolve_layer_class(layer_type: Any, device: str, dtype: torch.dtype):
    """Resolve layer_type to a class (or None if pre-built instance)."""
    if layer_type is None or layer_type == "linear" or layer_type == "nn.Linear":
        return nn.Linear
    if isinstance(layer_type, type):
        return layer_type
    if isinstance(layer_type, nn.Module):
        return None
    if isinstance(layer_type, str):
        from . import nn_layers
        cls = getattr(nn_layers, layer_type, None)
        if cls is not None:
            return cls
        try:
            from ....config.deep_learning import _load_model_config_registry
            reg = _load_model_config_registry()
            module_path = reg.get(layer_type)
            if module_path:
                import importlib
                mod = importlib.import_module("Code.models.deep_learning." + module_path)
                return getattr(mod, layer_type, None)
        except Exception:
            pass
    return None


class FeedForwardNeuralNetwork(DLModule):
    def __init__(self,
                 dims: Union[List[int], Dict[str, int], Tuple[int]],
                 biases: Union[List[bool], Tuple[bool], Dict[str, bool]],
                 act_funcs: Union[List[Union[nn.Module, Callable, str]],
                 Tuple[Union[nn.Module, Callable, str]],
                 Dict[str, Union[nn.Module, Callable, str]]],
                 bypass: Optional[Union[Dict[str, Union[list, tuple]], Dict[str, dict]]] = None,  # New Argument
                 device: str = 'cpu', dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.bypass = bypass

        if type(dims) != type(biases):
            raise TypeError(f"Both values must be of same type.")

        self.nn_layers = nn.ModuleList([])

        # Normalize dims and biases
        if isinstance(dims, dict):
            dims_list = list(dims.values())
        else:
            dims_list = list(dims)

        if isinstance(biases, dict):
            biases_list = list(biases.values())
        else:
            biases_list = list(biases)

        dim_prev = dims_list.pop(0)

        for dim, bias in zip(dims_list, biases_list):
            self.nn_layers.append(nn.Linear(dim_prev, dim, bias=bias, device=device, dtype=dtype))
            dim_prev = dim

        self.funcs = nn.ModuleList([])

        if isinstance(act_funcs, dict):
            acts_list = list(act_funcs.values())
        else:
            acts_list = list(act_funcs)

        for func in acts_list:
            if isinstance(func, Callable):
                from ....models.utils import ActFuncWrapper
                self.funcs.append(ActFuncWrapper(func, *args, **kwargs))
            elif isinstance(func, nn.Module):
                for param in func.parameters():
                    param.requires_grad = False
                self.funcs.append(func)
            elif isinstance(func, str):
                self.funcs.append(Activation(func, *args, **kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        layer_outputs = {}

        num_layers = len(self.nn_layers)
        num_funcs = len(self.funcs)

        for i in range(num_layers):
            out = self.nn_layers[i](out)

            if i < num_funcs:
                out = self.funcs[i](out)

            layer_outputs[i] = out

            if self.bypass is not None and i in self.bypass:
                source_idx = self.bypass[i]

                if not isinstance(source_idx, (list, tuple)):
                    sources = [source_idx]
                else:
                    sources = source_idx

                for src in sources:
                    if src in layer_outputs:
                        src_out = layer_outputs[src]
                        if src_out.shape == out.shape:
                            out = out + src_out

        return out


class FeedForwardNeuralNetworkOp(DLModelLayers):
    """
    Overloaded version of FFNN using DLModelLayers primarily.
    Arguments mimic FeedForwardNeuralNetwork but construct a layers config for DLModelLayers.
    """

    def __init__(self,
                 dims: Union[List[int], Dict[str, int], Tuple[int]],
                 biases: Union[List[bool], Tuple[bool], Dict[str, bool]],
                 act_funcs: Union[List[Union[nn.Module, Callable, str]],
                 Tuple[Union[nn.Module, Callable, str]],
                 Dict[str, Union[nn.Module, Callable, str]]],
                 bypass: Optional[Union[Dict[str, Union[list, tuple]], Dict[str, dict]]] = None,
                 device: str = 'cpu', dtype: torch.dtype = torch.float32,
                 *args, **kwargs):

        self.bypass = bypass

        if isinstance(dims, dict):
            dims_list = list(dims.values())
        else:
            dims_list = list(dims)

        if isinstance(biases, dict):
            biases_list = list(biases.values())
        else:
            biases_list = list(biases)

        if isinstance(act_funcs, dict):
            acts_list = list(act_funcs.values())
        else:
            acts_list = list(act_funcs)

        dim_prev = dims_list.pop(0)

        layers_config = []

        for i, (dim, bias) in enumerate(zip(dims_list, biases_list)):
            layers_config.append(
                ('linear', {'in_features': dim_prev, 'out_features': dim, 'bias': bias})
            )
            dim_prev = dim

            if i < len(acts_list):
                func = acts_list[i]
                if isinstance(func, str):
                    layers_config.append(
                        ('act', {'activation_func': func, 'in_features': dim})
                    )
                elif isinstance(func, (nn.Module, Callable)):
                    layers_config.append(
                        ('act', {})
                    )

        dl_act_funcs = [f for f in acts_list if isinstance(f, (nn.Module, Callable))]

        super().__init__(layers=layers_config, act_funcs=dl_act_funcs, device=device, dtype=dtype, *args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        layer_outputs = {}

        current_block_idx = 0

        # Convert ModuleList to list to iterate manually if needed, or index
        if isinstance(self.layers, nn.ModuleList):
            modules = list(self.layers)
        else:
            modules = list(self.layers.values())

        idx = 0
        while idx < len(modules):
            layer = modules[idx]
            out = layer(out)
            idx += 1

            if idx < len(modules):
                next_mod = modules[idx]
                is_linear = isinstance(next_mod, (nn.Linear, nn.LazyLinear, nn.Bilinear))

                if not is_linear:
                    out = next_mod(out)
                    idx += 1

            layer_outputs[current_block_idx] = out

            if self.bypass is not None and current_block_idx in self.bypass:
                source_idx = self.bypass[current_block_idx]
                if not isinstance(source_idx, (list, tuple)):
                    sources = [source_idx]
                else:
                    sources = source_idx

                for src in sources:
                    if src in layer_outputs:
                        src_out = layer_outputs[src]
                        if src_out.shape == out.shape:
                            out = out + src_out

            current_block_idx += 1

        return out


class KANNetwork(DLModule):
    def __init__(self,
                 dims: Union[List[int], Dict[str, int], Tuple[int]],
                 bypass: Optional[Union[Dict[str, Union[list, tuple]], Dict[str, dict]]] = None,
                 kan_kwargs: Dict[str, Any] = None,
                 device: str = 'cpu',
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.bypass = bypass
        self.kan_kwargs = kan_kwargs if kan_kwargs is not None else {}
        self.factory_kwargs = {"device": device, "dtype": dtype}

        if isinstance(dims, dict):
            dims_list = list(dims.values())
        else:
            dims_list = list(dims)

        self.layers = nn.ModuleList([])

        for i in range(len(dims_list) - 1):
            in_dim = dims_list[i]
            out_dim = dims_list[i + 1]

            self.layers.append(
                KANLayer(
                    in_features=in_dim,
                    out_features=out_dim,
                    **self.kan_kwargs,
                    **self.factory_kwargs
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        layer_outputs = {}

        for i, layer in enumerate(self.layers):
            out = layer(out)

            layer_outputs[i] = out

            if self.bypass is not None and i in self.bypass:
                source_idx = self.bypass[i]

                if not isinstance(source_idx, (list, tuple)):
                    sources = [source_idx]
                else:
                    sources = source_idx

                for src in sources:
                    if src in layer_outputs:
                        src_out = layer_outputs[src]
                        if src_out.shape == out.shape:
                            out = out + src_out

        return out


class CustomLinearNetworkModule(DLModule):
    def __init__(self,
                 dims: Union[List[int], Tuple[int], Dict],
                 act_funcs: Union[List[Union[nn.Module, Callable, str]],
                 Tuple[Union[nn.Module, Callable, str]],
                 Dict[str, Union[nn.Module, Callable, str]]],
                 layer_types: Union[List, Tuple, Dict, str, type, nn.Module] = "linear",
                 layer_configs: Optional[Union[List[dict], Dict]] = None,
                 biases: Union[List[bool], Tuple[bool], Dict] = True,
                 bypass: Optional[Dict[int, Union[int, List[int]]]] = None,
                 device: str = 'cpu', dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.bypass = bypass

        if isinstance(dims, dict):
            dims_list = list(dims.values())
        else:
            dims_list = list(dims)

        if isinstance(biases, bool):
            biases_list = [biases] * (len(dims_list) - 1)
        elif isinstance(biases, dict):
            biases_list = list(biases.values())
        else:
            biases_list = list(biases)
        if type(dims) != type(biases) and not isinstance(biases, bool):
            raise TypeError("dims and biases must be of same type (list, tuple, or dict), or biases can be bool.")

        if isinstance(act_funcs, dict):
            acts_list = list(act_funcs.values())
        else:
            acts_list = list(act_funcs)

        num_blocks = len(dims_list) - 1
        if len(biases_list) != num_blocks:
            raise ValueError(f"biases length ({len(biases_list)}) must equal num_blocks ({num_blocks}).")
        if len(acts_list) != num_blocks:
            raise ValueError(f"act_funcs length ({len(acts_list)}) must equal num_blocks ({num_blocks}).")

        if isinstance(layer_types, (list, tuple, dict)):
            if isinstance(layer_types, dict):
                layer_types_list = list(layer_types.values())
            else:
                layer_types_list = list(layer_types)
            if len(layer_types_list) != num_blocks:
                if len(layer_types_list) == 1:
                    layer_types_list = layer_types_list * num_blocks
                else:
                    raise ValueError(
                        f"layer_types length ({len(layer_types_list)}) must equal num_blocks ({num_blocks}) or 1."
                    )
        else:
            layer_types_list = [layer_types] * num_blocks

        if layer_configs is None:
            layer_configs_list = [{}] * num_blocks
        elif isinstance(layer_configs, dict):
            layer_configs_list = [layer_configs.get(i, {}) for i in range(num_blocks)]
        else:
            layer_configs_list = list(layer_configs)
            while len(layer_configs_list) < num_blocks:
                layer_configs_list.append({})

        self.nn_layers = nn.ModuleList([])
        factory_kwargs = {"device": device, "dtype": dtype}

        for i in range(num_blocks):
            in_f = dims_list[i]
            out_f = dims_list[i + 1]
            bias = biases_list[i]
            layer_type = layer_types_list[i]
            cfg = layer_configs_list[i] if i < len(layer_configs_list) else {}

            if isinstance(layer_type, nn.Module):
                self.nn_layers.append(layer_type)
            else:
                cls = _resolve_layer_class(layer_type, device, dtype)
                if cls is None:
                    raise ValueError(f"Could not resolve layer type: {layer_type}")
                layer_kwargs = {
                    "in_features": in_f,
                    "out_features": out_f,
                    "bias": bias,
                    **factory_kwargs,
                    **cfg,
                }
                self.nn_layers.append(cls(**layer_kwargs))

        self.funcs = nn.ModuleList([])
        for func in acts_list:
            if isinstance(func, Callable):
                from ....models.utils import ActFuncWrapper
                self.funcs.append(ActFuncWrapper(func, *args, **kwargs))
            elif isinstance(func, nn.Module):
                for param in func.parameters():
                    param.requires_grad = False
                self.funcs.append(func)
            elif isinstance(func, str):
                self.funcs.append(Activation(func, *args, **kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        layer_outputs = {}

        num_layers = len(self.nn_layers)
        num_funcs = len(self.funcs)

        for i in range(num_layers):
            out = self.nn_layers[i](out)

            if i < num_funcs:
                out = self.funcs[i](out)

            layer_outputs[i] = out

            if self.bypass is not None and i in self.bypass:
                source_idx = self.bypass[i]

                if not isinstance(source_idx, (list, tuple)):
                    sources = [source_idx]
                else:
                    sources = source_idx

                for src in sources:
                    if src in layer_outputs:
                        src_out = layer_outputs[src]
                        if src_out.shape == out.shape:
                            out = out + src_out

        return out




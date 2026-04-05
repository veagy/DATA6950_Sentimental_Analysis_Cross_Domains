"""
CNN models: CNNetworks, CapsNetsModule, GroupEquivariantModule, ShiftNetsModule, DeformableConvModule.
Reference: docs/deep-learning/cnn/cnn.md
"""

import torch
import torch.nn as nn
import math
from .....models.deep_learning.activations.ActivationFunction import Activation
from ...models import ConvolutionLayer, PoolingLayer, PaddingLayer, DropoutLayer
from .....models.deep_learning.models import DLModelLayers
from typing import Union, Any, Tuple, List, Dict, Callable
from .....models.utils import DLModule


__all__ = [
    "CNNetworks",
    "CNNetworksOp",
    "CapsNetsModule",
    "GroupEquivariantConvolutionalModule",
    "ShiftNetsModule",
    "DeformableConvModule",
    "InvolutionModule",
    "VolterraConvModule",
    "DynamicSnakeConvModule",
    "ODConvModule",
    "ShiftwiseConvModule",
    "SEAFECModule",
    "IncoherentMotifModule",
]


class CNNetworks(DLModule):
    def __init__(self,
                 dimensionality: Union[int, float],
                 layer_types: Union[List[str], Tuple[str], Dict[str, str]],
                 act_funcs: Union[List[Tuple[str, Callable, nn.Module]],
                 Tuple[Union[str, Callable, nn.Module]],
                 Dict[str, Union[str, Callable, nn.Module]]],
                 channels: Union[List[int], Tuple[int], Dict[str, int]],
                 pool_type: Union[List[str], Tuple[str], Dict[str, str]],
                 pad_type: Union[List[str], Tuple[str], Dict[str, str]],
                 dropout_percent: Union[List[float], Tuple[float], Dict[str, float]],
                 kernel_size: Union[List[int], Tuple[int], Dict[str, int]],
                 stride: Union[List[int], Tuple[int], Dict[str, int]],
                 padding: Union[List[int], Tuple[int], Dict[str, int]],
                 dilation: Union[List[int], Tuple[int], Dict[str, int]],
                 groups: Union[List[int], Tuple[int], Dict[str, int]],
                 bias: Union[List[bool], Tuple[bool], Dict[str, bool]],
                 padding_mode: Union[List[str], Tuple[str], Dict[str, str]],
                 lazy: bool = False,
                 transpose: bool = False,
                 device: str = 'cpu',
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        if isinstance(layer_types, (list, tuple)):
            self.layers = nn.ModuleList([])
            in_channels = channels.pop(0)
            for layer_type in layer_types:
                match layer_type:
                    case 'conv' | 'convolution':
                        conv_config = {
                            "in_channels": in_channels,
                            "out_channels": channels.pop(0),
                            "kernel_size": kernel_size.pop(0),
                            "stride": stride.pop(0),
                            "padding": padding.pop(0),
                            "dilation": dilation.pop(0),
                            "groups": groups.pop(0),
                            "bias": bias.pop(0),
                            "padding_mode": padding_mode.pop(0),
                            "device": device,
                            "dtype": dtype
                        }
                        in_channels = conv_config["out_channels"]
                        self.layers.append(
                            ConvolutionLayer(dimensionality, conv_config, lazy, transpose, *args, **kwargs))
                    case "pool" | "pooling":
                        pool_config = {
                            "kernel_size": kernel_size.pop(0),
                            "stride": stride.pop(0),
                            "padding": padding.pop(0),
                            "dilation": dilation.pop(0)
                        }
                        self.layers.append(PoolingLayer(pool_type.pop(0), dimensionality, pool_config, *args, **kwargs))
                    case "pad" | "padding":
                        pad_config = {"padding": padding.pop(0)}
                        self.layers.append(PaddingLayer(pad_type.pop(0), dimensionality, pad_config, *args, **kwargs))
                    case 'act' | 'activation' | 'act_func' | 'activation_func' | 'act_function' | 'activation_function':
                        func = act_funcs.pop(0)
                        if isinstance(func, str):
                            self.layers.append(Activation(func, *args, **kwargs))
                        elif isinstance(func, Callable):
                            from .....models.utils import ActFuncWrapper
                            self.layers.append(ActFuncWrapper(func, *args, **kwargs))
                        elif isinstance(func, nn.Module):
                            for param in func.parameters():
                                param.requires_grad = False
                            self.layers.append(func)
                    case "fc" | "fully_connected" | "nn" | "linear":
                        out_c = channels.pop(0)
                        self.layers.append(
                            nn.Linear(in_channels, out_c, bias=bias.pop(0), device=device, dtype=dtype))
                        in_channels = out_c
                    case "dropout":
                        self.layers.append(DropoutLayer(dropout_percent.pop(0), dimensionality, *args, **kwargs))

        elif isinstance(layer_types, dict):
            self.layers = nn.ModuleDict({})

            def get_param(param_collection, key, default=None):
                if isinstance(param_collection, dict):
                    return param_collection.get(key, default)
                elif isinstance(param_collection, (list, tuple)) and len(param_collection) > 0:
                    return param_collection.pop(0)
                return default

            if isinstance(channels, (list, tuple)):
                in_channels = channels.pop(0)
            elif isinstance(channels, dict) and "in_channels" in channels:
                in_channels = channels["in_channels"]
            else:
                in_channels = 1

            for key, layer_type in layer_types.items():
                match layer_type:
                    case 'conv' | 'convolution':
                        conv_config = {
                            "in_channels": in_channels,
                            "out_channels": get_param(channels, key),
                            "kernel_size": get_param(kernel_size, key),
                            "stride": get_param(stride, key, 1),
                            "padding": get_param(padding, key, 0),
                            "dilation": get_param(dilation, key, 1),
                            "groups": get_param(groups, key, 1),
                            "bias": get_param(bias, key, True),
                            "padding_mode": get_param(padding_mode, key, 'zeros'),
                            "device": device,
                            "dtype": dtype
                        }
                        in_channels = conv_config["out_channels"]
                        self.layers[key] = ConvolutionLayer(dimensionality, conv_config, lazy, transpose, *args, **kwargs)

                    case "pool" | "pooling":
                        pool_config = {
                            "kernel_size": get_param(kernel_size, key),
                            "stride": get_param(stride, key, get_param(kernel_size, key)),
                            "padding": get_param(padding, key, 0),
                            "dilation": get_param(dilation, key, 1)
                        }
                        self.layers[key] = PoolingLayer(get_param(pool_type, key, 'max'), dimensionality, pool_config, *args, **kwargs)

                    case "pad" | "padding":
                        pad_config = {"padding": get_param(padding, key)}
                        self.layers[key] = PaddingLayer(get_param(pad_type, key, 'zero'), dimensionality, pad_config, *args, **kwargs)

                    case 'act' | 'activation' | 'act_func' | 'activation_func' | 'act_function' | 'activation_function':
                        func = get_param(act_funcs, key)
                        if isinstance(func, str):
                            self.layers[key] = Activation(func, *args, **kwargs)
                        elif isinstance(func, Callable):
                            from .....models.utils import ActFuncWrapper
                            self.layers[key] = ActFuncWrapper(func, *args, **kwargs)
                        elif isinstance(func, nn.Module):
                            for param in func.parameters():
                                param.requires_grad = False
                            self.layers[key] = func

                    case "fc" | "fully_connected" | "nn" | "linear":
                        out_c = get_param(channels, key)
                        self.layers[key] = nn.Linear(in_channels, out_c, bias=get_param(bias, key, True), device=device, dtype=dtype)
                        in_channels = out_c

                    case "dropout":
                        self.layers[key] = DropoutLayer(get_param(dropout_percent, key, 0.5), dimensionality, *args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.layers, nn.ModuleList):
            for layer in self.layers:
                if isinstance(layer, nn.Linear) and x.dim() > 2:
                    x = x.view(x.size(0), -1)
                x = layer(x)
        elif isinstance(self.layers, nn.ModuleDict):
            for layer in self.layers.values():
                if isinstance(layer, nn.Linear) and x.dim() > 2:
                    x = x.view(x.size(0), -1)
                x = layer(x)
        return x


class CNNetworksOp(DLModelLayers):
    def __init__(self,
                 dimensionality: Union[int, float],
                 layer_types: Union[List[str], Tuple[str], Dict[str, str]],
                 act_funcs: Union[List[Tuple[str, Callable, nn.Module]],
                 Tuple[Union[str, Callable, nn.Module]],
                 Dict[str, Union[str, Callable, nn.Module]]],
                 channels: Union[List[int], Tuple[int], Dict[str, int]],
                 pool_type: Union[List[str], Tuple[str], Dict[str, str]],
                 pad_type: Union[List[str], Tuple[str], Dict[str, str]],
                 dropout_percent: Union[List[float], Tuple[float], Dict[str, float]],
                 kernel_size: Union[List[int], Tuple[int], Dict[str, int]],
                 stride: Union[List[int], Tuple[int], Dict[str, int]],
                 padding: Union[List[int], Tuple[int], Dict[str, int]],
                 dilation: Union[List[int], Tuple[int], Dict[str, int]],
                 groups: Union[List[int], Tuple[int], Dict[str, int]],
                 bias: Union[List[bool], Tuple[bool], Dict[str, bool]],
                 padding_mode: Union[List[str], Tuple[str], Dict[str, str]],
                 lazy: bool = False,
                 transpose: bool = False,
                 device: str = 'cpu',
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):

        def to_list(x):
            return list(x) if isinstance(x, (list, tuple)) else x

        layers_config = None

        if isinstance(layer_types, (list, tuple)):
            layers_config = []
            l_channels = to_list(channels)
            l_kernel_size = to_list(kernel_size)
            l_stride = to_list(stride)
            l_padding = to_list(padding)
            l_dilation = to_list(dilation)
            l_groups = to_list(groups)
            l_bias = to_list(bias)
            l_padding_mode = to_list(padding_mode)
            l_pool_type = to_list(pool_type)
            l_pad_type = to_list(pad_type)
            l_act_funcs = to_list(act_funcs)
            l_dropout_percent = to_list(dropout_percent)

            in_channels = l_channels.pop(0)

            for layer_type in layer_types:
                match layer_type:
                    case 'conv' | 'convolution':
                        out_c = l_channels.pop(0)
                        conv_config = {
                            "in_channels": in_channels,
                            "out_channels": out_c,
                            "kernel_size": l_kernel_size.pop(0),
                            "stride": l_stride.pop(0),
                            "padding": l_padding.pop(0),
                            "dilation": l_dilation.pop(0),
                            "groups": l_groups.pop(0),
                            "bias": l_bias.pop(0),
                            "padding_mode": l_padding_mode.pop(0)
                        }
                        real_config = {
                            "dimensionality": dimensionality,
                            "conv_config": conv_config,
                            "lazy": lazy,
                            "transpose": transpose
                        }
                        in_channels = out_c
                        layers_config.append(('conv', real_config))

                    case "pool" | "pooling":
                        pt = l_pool_type.pop(0)
                        k_s = l_kernel_size.pop(0)
                        real_pool_config = {
                            "pool_type": pt,
                            "dimensionality": dimensionality,
                            "pool_config": {
                                "kernel_size": k_s,
                                "stride": l_stride.pop(0),
                                "padding": l_padding.pop(0),
                                "dilation": l_dilation.pop(0)
                            }
                        }
                        layers_config.append(('pool', real_pool_config))

                    case "pad" | "padding":
                        pt = l_pad_type.pop(0)
                        real_config = {
                            "pad_type": pt,
                            "dimensionality": dimensionality,
                            "pad_config": {"padding": l_padding.pop(0)}
                        }
                        layers_config.append(('pad', real_config))

                    case 'act' | 'activation' | 'act_func' | 'activation_func' | 'act_function' | 'activation_function':
                        func = l_act_funcs.pop(0)
                        layers_config.append(('act', {'activation': func} if isinstance(func, str) else {}))

                    case "fc" | "fully_connected" | "nn" | "linear":
                        out_c = l_channels.pop(0)
                        fc_config = {
                            "in_features": in_channels,
                            "out_features": out_c,
                            "bias": l_bias.pop(0)
                        }
                        in_channels = out_c
                        layers_config.append(('linear', fc_config))

                    case "dropout":
                        drop_config = {
                            "dropout_percent": l_dropout_percent.pop(0),
                            "dimensionality": dimensionality
                        }
                        layers_config.append(('dropout', drop_config))

        elif isinstance(layer_types, dict):
            layers_config = {}

            def get_param(pc, key, default=None):
                return pc.get(key, default) if isinstance(pc, dict) else default

            in_channels = channels.get("in_channels", 1)

            for key, layer_type in layer_types.items():
                match layer_type:
                    case 'conv' | 'convolution':
                        out_c = get_param(channels, key)
                        conv_config = {
                            "in_channels": in_channels,
                            "out_channels": out_c,
                            "kernel_size": get_param(kernel_size, key),
                            "stride": get_param(stride, key, 1),
                            "padding": get_param(padding, key, 0),
                            "dilation": get_param(dilation, key, 1),
                            "groups": get_param(groups, key, 1),
                            "bias": get_param(bias, key, True),
                            "padding_mode": get_param(padding_mode, key, 'zeros')
                        }
                        real_config = {
                            "dimensionality": dimensionality,
                            "conv_config": conv_config,
                            "lazy": lazy,
                            "transpose": transpose
                        }
                        in_channels = out_c
                        layers_config[key] = {'type': 'conv', 'config': real_config}

                    case "pool" | "pooling":
                        real_config = {
                            "pool_type": get_param(pool_type, key, 'max'),
                            "dimensionality": dimensionality,
                            "pool_config": {
                                "kernel_size": get_param(kernel_size, key),
                                "stride": get_param(stride, key, get_param(kernel_size, key)),
                                "padding": get_param(padding, key, 0),
                                "dilation": get_param(dilation, key, 1)
                            }
                        }
                        layers_config[key] = {'type': 'pool', 'config': real_config}

                    case "pad" | "padding":
                        real_config = {
                            "pad_type": get_param(pad_type, key, 'zero'),
                            "dimensionality": dimensionality,
                            "pad_config": {"padding": get_param(padding, key)}
                        }
                        layers_config[key] = {'type': 'pad', 'config': real_config}

                    case 'act' | 'activation' | 'act_func' | 'activation_func' | 'act_function' | 'activation_function':
                        func = get_param(act_funcs, key)
                        layers_config[key] = {'type': 'act', 'config': {'activation': func} if isinstance(func, str) else {}}

                    case "fc" | "fully_connected" | "nn" | "linear":
                        out_c = get_param(channels, key)
                        fc_config = {
                            "in_features": in_channels,
                            "out_features": out_c,
                            "bias": get_param(bias, key, True)
                        }
                        in_channels = out_c
                        layers_config[key] = {'type': 'linear', 'config': fc_config}

                    case "dropout":
                        drop_config = {
                            "dropout_percent": get_param(dropout_percent, key, 0.5),
                            "dimensionality": dimensionality
                        }
                        layers_config[key] = {'type': 'dropout', 'config': drop_config}

        dl_act_funcs = []
        if isinstance(act_funcs, (list, tuple)):
            dl_act_funcs = [f for f in act_funcs if isinstance(f, (nn.Module, Callable))]
        elif isinstance(act_funcs, dict):
            dl_act_funcs = {k: v for k, v in act_funcs.items() if isinstance(v, (nn.Module, Callable))}

        super().__init__(layers=layers_config, act_funcs=dl_act_funcs, device=device, dtype=dtype, *args, **kwargs)


class CapsNetsModule(DLModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 in_capsules: int,
                 out_capsules: int,
                 kernel_size: Union[int, tuple],
                 num_layers: int,
                 dimensionality: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 routing_iter: int = 3,
                 decay_factor: int = 0.1,
                 iter_decay: bool = False,
                 funcs: Union[List[Tuple[str, Callable, nn.Module]],
                 Tuple[Union[str, Callable, nn.Module]],
                 Dict[str, Union[str, Callable, nn.Module]],
                 str, Callable, nn.Module, DLModule, None] = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}
        args_kwargs = {"args": args, "kwargs": kwargs}
        conv_config = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
            "bias": bias,
            "padding_mode": padding_mode,
            **self.factory_kwargs
        }
        self.conv = ConvolutionLayer(
            dimensionality=dimensionality,
            conv_config=conv_config,
            lazy=False,
            transpose=False
        )
        if funcs is None:
            funcs = [None] * num_layers
        if isinstance(funcs, dict):
            funcs = list(funcs.values())
        if isinstance(funcs, (list, tuple)):
            if len(funcs) > num_layers:
                funcs = funcs[:num_layers]
            elif len(funcs) < num_layers:
                funcs = [*funcs, *funcs[:num_layers - len(funcs)]]

        from ..layers.layers import CapsNetsLayer
        layers = [
            CapsNetsLayer(
                in_channels=out_channels,
                out_channels=out_channels,
                in_capsules=in_capsules,
                out_capsules=out_capsules,
                routing_iter=routing_iter,
                func=funcs.pop(0),
                **self.factory_kwargs,
                **args_kwargs
            )
        ]
        for i, func in enumerate(funcs):
            r_iter = int(routing_iter * math.exp(-decay_factor * (i + 1))) if iter_decay else routing_iter
            if iter_decay and r_iter < 2:
                r_iter = 2
            layers.append(
                CapsNetsLayer(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    in_capsules=out_capsules,
                    out_capsules=out_capsules,
                    routing_iter=r_iter,
                    func=func,
                    **self.factory_kwargs,
                    **args_kwargs
                )
            )
        self.layers = nn.Sequential(*layers)
        self.in_capsules = in_capsules

    def squash(self, s, dim=-1):
        norm_sq = (s ** 2).sum(dim, keepdim=True)
        norm = torch.sqrt(norm_sq)
        return (norm_sq / (1 + norm_sq)) * (s / (norm + 1e-6))

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        batch = x.size(0)
        x = x.view(batch, -1, self.in_capsules)
        x = self.squash(x, dim=-1)
        return self.layers(x)


class GroupEquivariantConvolutionalModule(DLModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 rot_groups: int,
                 num_layers: int,
                 dimensionality: Union[int, float],
                 bias: Union[bool, tuple] = False,
                 stride: Union[int, tuple] = 1,
                 dilation: Union[int, tuple] = 1,
                 groups: int = 1,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        from ..layers.layers import GroupEquivariantConvolutionalLayer
        config = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "rot_groups": rot_groups,
            "dimensionality": dimensionality,
            "is_first": False,
            "bias": bias,
            "stride": stride,
            "dilation": dilation,
            "groups": groups,
            "device": device,
            "dtype": dtype
        }
        configs = [dict(config) for _ in range(num_layers)]
        configs[0]["is_first"] = True

        self.module = nn.Sequential(*[GroupEquivariantConvolutionalLayer(**c) for c in configs])

    def forward(self, x: torch.Tensor):
        return self.module(x)


class ShiftNetsModule(DLModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 num_layers: int,
                 dimensionality: Union[int, float] = 2,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = "zeros",
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}
        config = {
            "in_channels": out_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "dimensionality": dimensionality,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
            "bias": bias,
            "padding_mode": padding_mode,
            **self.factory_kwargs
        }
        configs = [dict(config) for _ in range(num_layers)]
        configs[0]["in_channels"] = in_channels

        from ..layers.layers import ShiftNetLayer
        self.module = nn.Sequential(*[ShiftNetLayer(**c) for c in configs])

    def forward(self, x: torch.Tensor):
        return self.module(x)


class DeformableConvModule(DLModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_layers: int,
                 kernel_size: int = 3,
                 dimensionality: Union[int, float] = 2,
                 stride: int = 1,
                 padding: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = "zeros",
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}
        config = {
            "in_channels": out_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "dimensionality": dimensionality,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
            "bias": bias,
            "padding_mode": padding_mode,
            **self.factory_kwargs
        }
        configs = [dict(config) for _ in range(num_layers)]
        configs[0]["in_channels"] = in_channels

        from ..layers.layers import DeformableConvLayer
        self.module = nn.Sequential(*[DeformableConvLayer(**c) for c in configs])

    def forward(self, x: torch.Tensor):
        return self.module(x)


class InvolutionModule(DLModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_layers: int,
                 kernel_size: int = 7,
                 stride: int = 1,
                 reduction_ratio: int = 4,
                 group: int = 1,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        from ..layers.layers import InvolutionLayer
        configs = [{"in_channels": out_channels, "out_channels": out_channels, "kernel_size": kernel_size,
                    "stride": stride, "reduction_ratio": reduction_ratio, "group": group,
                    "device": device, "dtype": dtype} for _ in range(num_layers)]
        configs[0]["in_channels"] = in_channels
        self.module = nn.Sequential(*[InvolutionLayer(**c) for c in configs])

    def forward(self, x: torch.Tensor):
        return self.module(x)


class VolterraConvModule(DLModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_layers: int,
                 kernel_size: int = 3,
                 order: int = 2,
                 stride: int = 1,
                 padding: int = 1,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        from ..nonlinear.nonlinear import VolterraConvLayer
        configs = [{"in_channels": out_channels, "out_channels": out_channels, "kernel_size": kernel_size,
                    "order": order, "stride": stride, "padding": padding,
                    "device": device, "dtype": dtype} for _ in range(num_layers)]
        configs[0]["in_channels"] = in_channels
        self.module = nn.Sequential(*[VolterraConvLayer(**c) for c in configs])

    def forward(self, x: torch.Tensor):
        return self.module(x)


class DynamicSnakeConvModule(DLModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_layers: int,
                 kernel_size: int = 9,
                 extend_scope: float = 1.0,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        from ..dynamic.dynamic import DynamicSnakeConvLayer
        configs = [{"in_channels": out_channels, "out_channels": out_channels, "kernel_size": kernel_size,
                    "extend_scope": extend_scope, "device": device, "dtype": dtype} for _ in range(num_layers)]
        configs[0]["in_channels"] = in_channels
        self.module = nn.Sequential(*[DynamicSnakeConvLayer(**c) for c in configs])

    def forward(self, x: torch.Tensor):
        return self.module(x)


class ODConvModule(DLModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_layers: int,
                 kernel_size: int = 3,
                 reduction: int = 4,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        from ..dynamic.dynamic import ODConvLayer
        configs = [{"in_channels": out_channels, "out_channels": out_channels, "kernel_size": kernel_size,
                    "reduction": reduction, "device": device, "dtype": dtype} for _ in range(num_layers)]
        configs[0]["in_channels"] = in_channels
        self.module = nn.Sequential(*[ODConvLayer(**c) for c in configs])

    def forward(self, x: torch.Tensor):
        return self.module(x)


class ShiftwiseConvModule(DLModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_layers: int,
                 kernel_size: int = 3,
                 shift_groups: int = 4,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        from ..shiftwise.shiftwise import ShiftwiseConvLayer
        configs = [{"in_channels": out_channels, "out_channels": out_channels, "kernel_size": kernel_size,
                    "shift_groups": shift_groups, "device": device, "dtype": dtype} for _ in range(num_layers)]
        configs[0]["in_channels"] = in_channels
        self.module = nn.Sequential(*[ShiftwiseConvLayer(**c) for c in configs])

    def forward(self, x: torch.Tensor):
        return self.module(x)


class SEAFECModule(DLModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_layers: int,
                 kernel_size: int = 3,
                 scarf_reduction: int = 4,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        from ..spatial_edge.spatial_edge import SEAFECLayer
        configs = [{"in_channels": out_channels, "out_channels": out_channels, "kernel_size": kernel_size,
                    "scarf_reduction": scarf_reduction, "device": device, "dtype": dtype} for _ in range(num_layers)]
        configs[0]["in_channels"] = in_channels
        self.module = nn.Sequential(*[SEAFECLayer(**c) for c in configs])

    def forward(self, x: torch.Tensor):
        return self.module(x)


class IncoherentMotifModule(DLModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_layers: int,
                 motif_type: str = "IFFL",
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        from ..structural.structural import IncoherentMotifLayer
        configs = [{"in_channels": out_channels, "out_channels": out_channels,
                    "motif_type": motif_type, "device": device, "dtype": dtype} for _ in range(num_layers)]
        configs[0]["in_channels"] = in_channels
        self.module = nn.Sequential(*[IncoherentMotifLayer(**c) for c in configs])

    def forward(self, x: torch.Tensor):
        return self.module(x)

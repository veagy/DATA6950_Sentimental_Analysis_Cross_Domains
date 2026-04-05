import torch
import torch.nn as nn
import re
import math
import json
import os
from typing import Optional, Any, List, Tuple, Union, Dict, Callable
__all__ = [
    'ConvolutionActivation',
    'CustomStringConvolutionActivation',
]


class ConvolutionActivation(nn.Module):
    """
    Activation function based on fast Fourier transform (FFT) convolution.
    Applies initial linear transformations, followed by a gate-up mechanism 
    and FFT-based convolution with dynamic activations.
    """
    def __init__(
        self,
        in_features: int,
        activations: Union[str, List[str], nn.Module, nn.ModuleList, Callable, List[Callable], Any],
        dim: int = -1,
        **kwargs,
    ):
        super().__init__()
        self.in_features = in_features
        self.dim = dim
        self.bias = kwargs.get('bias', False)
        
        # Projections
        self.initial = nn.Linear(in_features, 2 * in_features, bias=self.bias)
        self.gate = nn.Linear(in_features, in_features, bias=self.bias)
        self.up = nn.Linear(in_features, in_features, bias=self.bias)
        
        self.out_layer = kwargs.get('out_layer', False)
        if self.out_layer:
            self.out = nn.Linear(in_features, in_features, bias=self.bias)
        
        if activations is None:
            raise ValueError("No activations provided for ConvolutionActivation.")
        
        # Handle single activation name or list
        if isinstance(activations, str):
            act_names = [activations, activations]
        elif isinstance(activations, (list, tuple)):
            if len(activations) == 0:
                 raise ValueError("activations list cannot be empty.")
            act_names = list(activations)
            if len(act_names) == 1:
                act_names = [act_names[0], act_names[0]]
            else:
                act_names = act_names[:2]
        else:
            # Fallback if someone passed an object directly
            act_names = [activations, activations]

        from ..ActivationFunction import Activation
        self.funcs = nn.ModuleList()
        for name in act_names:
            if isinstance(name, str):
                self.funcs.append(Activation(name, **kwargs))
            else:
                # If it's already a module, use it but possibly freeze it
                if isinstance(name, nn.Module):
                    if kwargs.get('freeze', True):
                        for param in name.parameters():
                            param.requires_grad = False
                    self.funcs.append(name)
                else:
                    # Generic case, try to wrap it if possible or error
                    raise TypeError(f"Unsupported activation type: {type(name)}")

        self.arrange = kwargs.get('arrange', 'alternate')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Move target dim to last for linear layers
        x = x.transpose(self.dim, -1)
        
        x = self.initial(x)
        if self.arrange == 'alternate':
            x1, x2 = x[..., 0::2], x[..., 1::2]
        else:
            x1, x2 = x[..., :self.in_features], x[..., self.in_features:]
            
        x1 = self.gate(x1)
        x2 = self.up(x2)
        
        func1, func2 = self.funcs
        x1 = func1(x1)
        
        # FFT Convolution
        conv_val = self.convolution(x1, x2)
        conv_val = func2(conv_val)
        
        if self.out_layer:
            conv_val = self.out(conv_val)
            
        # Transpose back
        return conv_val.transpose(self.dim, -1)

    def convolution(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Fixed bug: x.shape(-1) -> x.shape[-1]
        if x.shape[-1] != y.shape[-1]:
            raise RuntimeError(f"Tensor dimension mismatch: {x.shape[-1]} vs {y.shape[-1]}")
            
        n = x.shape[-1]
        pad_size = n + n - 1
        
        # Spectral convolution
        x_freq = torch.fft.rfft(x, n=pad_size)
        y_freq = torch.fft.rfft(y, n=pad_size)
        
        out = x_freq * y_freq
        
        # Inverse transform
        out = torch.fft.irfft(out, n=pad_size)
        
        # Crop to original size
        start_idx = (n - 1) // 2
        return out[..., start_idx:start_idx + n]


class CustomStringConvolutionActivation(nn.Module):
    """
    CustomStringConvolutionActivation.
    
    A flexible activation layer that supports FFT-based convolution operations via a custom string.
    Combines the workflow of ConvolutionActivation (transpose -> process -> transpose back) with
    dynamic string parsing for complex operational definitions.
    
    Args:
        in_features (int): Number of input features.
        act_operation (str): String describing the operation (e.g., "conv(x1, x2) + w1 * A1(x3)").
        act_funcs (Union[str, nn.Module, ...]): Activation functions.
        dims (Union[int, Tuple[int]]): Dimensions for custom parameters (w, b).
        dim (int): Dimension to apply operations on (default -1).
        biases (List[bool]): Whether to use bias for each linear transformation x1, x2...
    """
    def __init__(self,
                 in_features: int,
                 act_operation: str,
                 act_funcs: Union[str, nn.Module, Callable, List[str], nn.ModuleList, List[Any]],
                 dims: Union[int, Tuple[int]] = None,
                 dim: int = -1,
                 biases: List[bool] = None,
                 name: str = None,
                 *args, **kwargs):
        super().__init__()
        from ..ActivationFunction import ExpressionParser
        self.in_features = in_features
        # self.raw_operation = act_operation # Replaced by registry check
        self.dim = dim
        self.dims = dims
        self.args = args
        self.kwargs = kwargs
        
        # Registry Check
        self.registry_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'custom_string.json')
        self.raw_operation = self._check_registry(act_operation, name)

        self.funcs = nn.ModuleList([])
        self.__pre_process_funcs__(act_funcs)
        
        # Parse x1, x2... for linear transformations
        x_indices = [int(n) for n in re.findall(r'\bx(\d+)\b', self.raw_operation)]
        num_transforms = max(x_indices) if x_indices else 0
        
        if biases is None:
            biases = [kwargs.get('bias', False) for _ in range(num_transforms)]
        elif len(biases) < num_transforms:
            biases = biases + [kwargs.get('bias', False) for _ in range(num_transforms - len(biases))]
            
        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features, in_features, bias=b) for b in biases[:num_transforms]
        ])
        
        # Parse w1, w2... and b1, b2... for custom parameters
        self.custom_params = nn.ParameterDict()
        w_indices = set(int(n) for n in re.findall(r'\bw(\d+)\b', self.raw_operation))
        b_indices = set(int(n) for n in re.findall(r'\bb(\d+)\b', self.raw_operation))
        
        if (w_indices or b_indices) and self.dims is None:
             raise ValueError("dims must be provided when using w or b parameters in abstract strings.")

        for idx in w_indices:
            if isinstance(self.dims, int):
                shape = (self.dims,)
            else:
                shape = self.dims
            self.custom_params[f'w{idx}'] = nn.Parameter(torch.randn(*shape))

        for idx in b_indices:
            if isinstance(self.dims, int):
                b_dim = self.dims
            else:
                b_dim = self.dims[-1]
            self.custom_params[f'b{idx}'] = nn.Parameter(torch.zeros(b_dim))
            
        self.parser = ExpressionParser()
        self.math_ops = self.parser.get_executable_context()
        self.math_ops['conv'] = self.convolution # Add convolution operator
        
        self.executable_expression = self.parser.transpile(self.raw_operation, len(self.funcs), extra_ops=['conv'])
        self.compiled_code = compile(self.executable_expression, '<string>', 'eval')

    def _check_registry(self, act_operation: str, name: str) -> str:
        registry = {}
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    content = f.read().strip()
                    if content:
                         registry = json.loads(content)
            except (json.JSONDecodeError, IOError):
                pass
        
        # Check if act_operation is a name in registry
        if act_operation in registry:
            return registry[act_operation]
            
        # If name provided and not in registry, save it
        if name and name not in registry:
            registry[name] = act_operation
            try:
                with open(self.registry_path, 'w') as f:
                    json.dump(registry, f, indent=4)
            except IOError:
                pass
                
        return act_operation

    def __pre_process_funcs__(self, funcs: Any) -> None:
        from ..ActivationFunction import Activation, ExpressionParser
        if not isinstance(funcs, (list, nn.ModuleList)):
            funcs = [funcs]
        for func in funcs:
            if isinstance(func, str):
                self.funcs.append(Activation(func, *self.args, **self.kwargs))
            elif isinstance(func, nn.Module):
                self.funcs.append(func)
            elif callable(func):
                self.funcs.append(func)
            else:
                 raise TypeError(f"Unsupported activation type: {type(func)}")



    def convolution(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Same logic as ConvolutionActivation
        if x.shape[-1] != y.shape[-1]:
            # Simple broadcasting check or error
             raise RuntimeError(f"Tensor dimension mismatch for conv: {x.shape[-1]} vs {y.shape[-1]}")
        
        n = x.shape[-1]
        pad_size = n + n - 1
        x_freq = torch.fft.rfft(x, n=pad_size)
        y_freq = torch.fft.rfft(y, n=pad_size)
        out = x_freq * y_freq
        out = torch.fft.irfft(out, n=pad_size)
        start_idx = (n - 1) // 2
        return out[..., start_idx:start_idx + n]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transpose to put target dim at -1 for Linear and FFT
        x = x.transpose(self.dim, -1)
        
        raw_x = x
        feats = [layer(x) for layer in self.linear_layers]
        
        context = {
            'raw_x': raw_x,
            'feats': feats,
            'acts': self.funcs,
            'custom_params': self.custom_params,
            'ops': self.math_ops,
            'torch': torch
        }
        
        try:
            res = eval(self.compiled_code, {}, context)
            if not isinstance(res, torch.Tensor):
                res = torch.as_tensor(res, device=x.device, dtype=x.dtype)
            
            # Transpose back
            return res.transpose(self.dim, -1)
            
        except Exception as e:
            raise RuntimeError(f"Failed to evaluate expression: '{self.raw_operation}'\nError: {e}") from e


import torch
import torch.nn as nn
import math
from typing import Tuple, Any, Callable, Union, List, Optional, Dict
from .complex_ import Complex as _Complex
from ..ActivationFunction import Activation
import re
import json
import os
__all__ = [
    'ComplexActivation',
    'ComplexCReLU',
    'ComplexCustomStringActivationLayer',
    'ComplexELU',
    'ComplexGELU',
    'ComplexGLU',
    'ComplexHardshrink',
    'ComplexHardsigmoid',
    'ComplexHardswish',
    'ComplexHardtanh',
    'ComplexHeaviside',
    'ComplexLeakyReLU',
    'ComplexLogSigmoid',
    'ComplexLogSoftmax',
    'ComplexMish',
    'ComplexReLU',
    'ComplexReLU6',
    'ComplexSELU',
    'ComplexSiLU',
    'ComplexSigmoid',
    'ComplexSinc',
    'ComplexSoftmax',
    'ComplexSoftplus',
    'ComplexSoftshrink',
    'ComplexSoftsign',
    'ComplexTanh',
    'ComplexTanhshrink',
    'ModReLU',
    'zReLU',
]


class ComplexActivation(_Complex):
    """
    Robust Complex Activation wrapper.
    Handles 'complex-' prefix and supports multiple operation modes:
    - 'sep': Separate activation for real and imaginary parts.
    - 'mag': Activation applied to the magnitude, preserving phase.
    - 'rot': Parametric rotation and activation.
    """
    def __init__(self, activation: str, mode: str = 'sep', is_stacked_flag: bool = False, *args, **kwargs):
        # We need to extract common settings before calling super().__init__ 
        # because _Complex might consume data.
        # But ComplexActivation is used as nn.Module factory here.
        # super().__init__ handles data initialization if provided.
        super().__init__(data=None, is_stacked_flag=is_stacked_flag, *args, **kwargs)
        
        all_modes = ['sep', 'rot', 'mag', 'complex']
        if mode not in all_modes:
            raise ValueError(f"Mode '{mode}' is not supported. Use one of {all_modes}.")
        
        # Robustly handle the activation name and "complex-" prefix
        self.raw_activation_name = activation
        self.clean_activation_name = re.sub(r"^complex-", "", activation)
        self.mode = mode
        
        # Ensure deep copies of flags to avoid side-effects if shared
        self.flags = kwargs.get('flags', {}).copy()
        # Default flags if missing
        self.flags.setdefault('bias', False)
        self.flags.setdefault('gain', False)
        self.flags.setdefault('phase', True)
        
        self.dim = kwargs.get('dim', -1)
        self.dims = kwargs.get('dims', ())

        match self.mode:
            case 'sep':
                self.real_act = Activation(self.clean_activation_name, *args, **kwargs)
                self.imag_act = Activation(self.clean_activation_name, *args, **kwargs)

            case 'mag':
                self.act_func = Activation(self.clean_activation_name, *args, **kwargs)

            case 'rot':
                # Parametric Phase
                if not self.dims:
                    # If dims not provided, we might be in a state where we can't initialize Parameters correctly
                    # but we'll try to follow the existing logic.
                    pass
                
                self.parametric_phase = nn.Parameter(torch.zeros(self.dims))
                # Note: original code used stack of ones and zeros, which is essentially initializing phase.
                # Here we initialize to 0 and let initialize helper handle it if needed.
                self.__phase_initialize__(self.parametric_phase)
                
                if self.flags.get('bias'):
                    self.bias_param = nn.Parameter(torch.zeros(self.dims))
                    # Initialize bias as zeros in complex space (0+0j)
                else:
                    self.register_parameter('bias_param', None)

                if self.flags.get('gain'):
                    self.gain_param = nn.Parameter(torch.ones(self.dims))
                else:
                    self.register_parameter('gain_param', None)
                
                self.act_func = Activation(self.clean_activation_name, *args, **kwargs)

            case 'complex':
                # Map names to specialized classes or methods in _Complex
                name = self.clean_activation_name.lower()
                match name:
                    case 'crelu':
                        self.act_func = ComplexCReLU(*args, **kwargs)
                    case 'relu':
                        self.act_func = ComplexReLU(*args, **kwargs)
                    case 'leaky_relu':
                        self.act_func = ComplexLeakyReLU(*args, **kwargs)
                    case 'elu':
                        self.act_func = ComplexELU(*args, **kwargs)
                    case 'selu':
                        self.act_func = ComplexSELU(*args, **kwargs)
                    case 'sigmoid':
                        self.act_func = ComplexSigmoid(*args, **kwargs)
                    case 'tanh':
                        self.act_func = ComplexTanh(*args, **kwargs)
                    case 'silu':
                        self.act_func = ComplexSiLU(*args, **kwargs)
                    case 'mish':
                        self.act_func = ComplexMish(*args, **kwargs)
                    case 'softplus':
                        self.act_func = ComplexSoftplus(*args, **kwargs)
                    case 'gelu':
                        self.act_func = ComplexGELU(*args, **kwargs)
                    case 'relu6':
                        self.act_func = ComplexReLU6(*args, **kwargs)
                    case 'modrelu':
                        self.act_func = ModReLU(*args, **kwargs)
                    case 'zrelu':
                        self.act_func = zReLU(*args, **kwargs)
                    case 'glu':
                        self.act_func = ComplexGLU(*args, **kwargs)
                    case 'softmax':
                        self.act_func = ComplexSoftmax(*args, **kwargs)
                    case 'complex_softmax':
                        self.act_func = ComplexSoftmax(*args, **kwargs)
                    case 'log_softmax':
                        self.act_func = ComplexLogSoftmax(*args, **kwargs)
                    case 'tanhshrink':
                        self.act_func = ComplexTanhshrink(*args, **kwargs)
                    case 'softsign':
                        self.act_func = ComplexSoftsign(*args, **kwargs)
                    case 'hardtanh':
                        self.act_func = ComplexHardtanh(*args, **kwargs)
                    case 'hardsigmoid':
                        self.act_func = ComplexHardsigmoid(*args, **kwargs)
                    case 'hardswish':
                        self.act_func = ComplexHardswish(*args, **kwargs)
                    case 'softshrink':
                        self.act_func = ComplexSoftshrink(*args, **kwargs)
                    case 'hardshrink':
                        self.act_func = ComplexHardshrink(*args, **kwargs)
                    case 'log_sigmoid':
                        self.act_func = ComplexLogSigmoid(*args, **kwargs)
                    case 'heaviside':
                        self.act_func = ComplexHeaviside(*args, **kwargs)
                    case 'sinc':
                        self.act_func = ComplexSinc(*args, **kwargs)
                    case _:
                        # Default behavior for 'complex' mode: 
                        # Try to find it in Activation registry if not complex-specific
                        self.act_func = Activation(self.clean_activation_name, *args, **kwargs)

    def __phase_initialize__(self, param: torch.Tensor):
        # Original logic seems to want to set some initial phase
        with torch.no_grad():
            param.fill_(0.0) # Start with 0 phase

    def forward(self, x: Any):
        # Ensure input is a Complex object
        # We check for .tensor to identify our custom Complex objects (both Small and Big classes)
        if not hasattr(x, 'tensor') or not hasattr(x, 'dim'):
            actual_dim = self.dim
            if actual_dim is not None and actual_dim < 0:
                 if hasattr(x, 'dim'): # Standard torch.Tensor
                     actual_dim = x.dim() + 1 + actual_dim
            
            x = _Complex(x, dim=actual_dim, device=self.device, dtype=self.dtype)
        
        # Consistent dimension for the result
        target_dim = x.dim
        
        match self.mode:
            case 'sep':
                res_tensor = torch.stack([self.real_act(x.real), self.imag_act(x.imag)], dim=target_dim)
                # Force unbind to ensure we are not hiding a rank-3 tensor
                r_out, i_out = res_tensor.unbind(target_dim)
                # Re-construct using property setter logic (safe) or new Complex
                new_c = _Complex(r_out, dim=target_dim, dtype=self.dtype, device=self.device)
                new_c.imag = i_out
                return new_c

            case 'mag':
                mag = x.mag()
                phi = x.phi()
                new_mag = self.act_func(mag)
                return _Complex.from_polar(new_mag, phi, dim=target_dim)

            case 'rot':
                # Apply rotation: z * exp(i * phase)
                phi = x.phi() + self.parametric_phase
                mag = x.mag()
                new_mag = self.act_func(mag)
                
                # Apply gain if present
                if hasattr(self, 'gain_param') and self.gain_param is not None:
                    new_mag = new_mag * self.gain_param
                
                return _Complex.from_polar(new_mag, phi, dim=target_dim)

            case 'complex':
                # Special modules usually handle _Complex inputs or return them
                res = self.act_func(x)
                # If the underlying module returned a raw tensor, re-wrap it
                if isinstance(res, torch.Tensor) and not isinstance(res, _Complex):
                    return _Complex(res, dim=x.dim, _is_stacked=True) # Assuming it same stack-dim
                return res

        return x


class ModReLU(nn.Module):
    def __init__(self, bias: float = 0.0, learnable: bool = False, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        if learnable:
            self.bias = nn.Parameter(torch.tensor(bias))
        else:
            self.register_buffer('bias', torch.tensor(bias))

    def forward(self, x: Any) -> _Complex:
        if not isinstance(x, _Complex):
            x = _Complex(x, *self.args, **self.kwargs)
        return x.modReLU(self.bias, inplace=False)


class zReLU(nn.Module):
    def __init__(self, *args, **kwargs):
       super().__init__()
       self.args = args
       self.kwargs = kwargs

    def forward(self, x: Any) -> _Complex:
        if not isinstance(x, _Complex):
            x = _Complex(x, *self.args, **self.kwargs)
        phi = x.phi()
        # Condition: 0 <= phi <= pi/2
        condition = (phi >= 0.0) & (phi <= math.pi / 2.0)
        return _Complex.where_(condition, x, _Complex.zeros_like(x))


class ComplexCReLU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: Any) -> _Complex:
        if not isinstance(x, _Complex):
            x = _Complex(x, *self.args, **self.kwargs)
        return x.cReLU()


class ComplexReLU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: Any) -> _Complex:
        if not isinstance(x, _Complex):
            x = _Complex(x, *self.args, **self.kwargs)
        return x.relu()


class ComplexLeakyReLU(nn.Module):
    def __init__(self, negative_slope: float = 0.01, *args, **kwargs):
        super().__init__()
        self.negative_slope = negative_slope
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: Any) -> _Complex:
        if not isinstance(x, _Complex):
            x = _Complex(x, *self.args, **self.kwargs)
        return x.leaky_relu(negative_slope=self.negative_slope)


class ComplexELU(nn.Module):
    def __init__(self, alpha: float = 1.0, *args, **kwargs):
        super().__init__()
        self.alpha = alpha
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: Any) -> _Complex:
        if not isinstance(x, _Complex):
            x = _Complex(x, *self.args, **self.kwargs)
        return x.elu(alpha=self.alpha)


class ComplexSELU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: Any) -> _Complex:
        if not isinstance(x, _Complex):
            x = _Complex(x, *self.args, **self.kwargs)
        return x.selu()


class ComplexSigmoid(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: Any) -> _Complex:
        if not isinstance(x, _Complex):
            x = _Complex(x, *self.args, **self.kwargs)
        return x.sigmoid()


class ComplexTanh(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: Any) -> _Complex:
        if not isinstance(x, _Complex):
            x = _Complex(x, *self.args, **self.kwargs)
        return x.tanh()


class ComplexSiLU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: Any) -> _Complex:
        if not isinstance(x, _Complex):
            x = _Complex(x, *self.args, **self.kwargs)
        return x.silu()


class ComplexMish(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: Any) -> _Complex:
        if not isinstance(x, _Complex):
            x = _Complex(x, *self.args, **self.kwargs)
        return x.mish()


class ComplexSoftplus(nn.Module):
    def __init__(self, beta: float = 1.0, threshold: float = 20.0, *args, **kwargs):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: Any) -> _Complex:
        if not isinstance(x, _Complex):
            x = _Complex(x, *self.args, **self.kwargs)
        return x.softplus(beta=self.beta, threshold=self.threshold)


class ComplexGELU(nn.Module):
    def __init__(self, approximate: str = 'none', *args, **kwargs):
        super().__init__()
        self.approximate = approximate
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: Any) -> _Complex:
        if not isinstance(x, _Complex):
            x = _Complex(x, *self.args, **self.kwargs)
        return x.gelu(approximate=self.approximate)


class ComplexReLU6(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: Any) -> _Complex:
        if not isinstance(x, _Complex):
            x = _Complex(x, *self.args, **self.kwargs)
        return x.relu6()


class ComplexGLU(nn.Module):
    def __init__(self, dim: int = -1, *args, **kwargs):
        super().__init__()
        self.dim = dim
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: Any) -> _Complex:
        if not isinstance(x, _Complex):
            x = _Complex(x, *self.args, **self.kwargs)
        return x.glu(dim=self.dim)


class ComplexSoftmax(nn.Module):
    def __init__(self, dim: int = -1, *args, **kwargs):
        super().__init__()
        self.dim = dim
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: Any) -> _Complex:
        if not isinstance(x, _Complex):
            x = _Complex(x, *self.args, **self.kwargs)
        return x.softmax(dim=self.dim)


# Renamed to avoid confusion, but it was already ComplexSoftmax
class _ComplexSoftmax(nn.Module):
    def __init__(self, dim: int = -1, *args, **kwargs):
        super().__init__()
        self.dim = dim
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: Any) -> _Complex:
        if not isinstance(x, _Complex):
            x = _Complex(x, *self.args, **self.kwargs)
        return x.complex_softmax(dim=self.dim)


class ComplexLogSoftmax(nn.Module):
    def __init__(self, dim: int = -1, *args, **kwargs):
        super().__init__()
        self.dim = dim
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: Any) -> _Complex:
        if not isinstance(x, _Complex):
            x = _Complex(x, *self.args, **self.kwargs)
        return x.log_softmax(dim=self.dim)


class ComplexTanhshrink(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: Any) -> _Complex:
        if not isinstance(x, _Complex):
            x = _Complex(x, *self.args, **self.kwargs)
        return x.tanhshrink()


class ComplexSoftsign(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: Any) -> _Complex:
        if not isinstance(x, _Complex):
            x = _Complex(x, *self.args, **self.kwargs)
        return x.softsign()


class ComplexHardtanh(nn.Module):
    def __init__(self, min_val: float = -1.0, max_val: float = 1.0, *args, **kwargs):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: Any) -> _Complex:
        if not isinstance(x, _Complex):
            x = _Complex(x, *self.args, **self.kwargs)
        return x.hardtanh(min_val=self.min_val, max_val=self.max_val)


class ComplexHardsigmoid(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: Any) -> _Complex:
        if not isinstance(x, _Complex):
            x = _Complex(x, *self.args, **self.kwargs)
        return x.hardsigmoid()


class ComplexHardswish(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: Any) -> _Complex:
        if not isinstance(x, _Complex):
            x = _Complex(x, *self.args, **self.kwargs)
        return x.hardswish()


class ComplexSoftshrink(nn.Module):
    def __init__(self, lambd: float = 0.5, *args, **kwargs):
        super().__init__()
        self.lambd = lambd
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: Any) -> _Complex:
        if not isinstance(x, _Complex):
            x = _Complex(x, *self.args, **self.kwargs)
        return x.softshrink(lambd=self.lambd)


class ComplexHardshrink(nn.Module):
    def __init__(self, lambd: float = 0.5, *args, **kwargs):
        super().__init__()
        self.lambd = lambd
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: Any) -> _Complex:
        if not isinstance(x, _Complex):
            x = _Complex(x, *self.args, **self.kwargs)
        return x.hardshrink(lambd=self.lambd)


class ComplexLogSigmoid(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: Any) -> _Complex:
        if not isinstance(x, _Complex):
            x = _Complex(x, *self.args, **self.kwargs)
        return x.log_sigmoid()


class ComplexHeaviside(nn.Module):
    def __init__(self, values: Any, *args, **kwargs):
        super().__init__()
        self.values = values
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: Any) -> _Complex:
        if not isinstance(x, _Complex):
            x = _Complex(x, *self.args, **self.kwargs)
        return x.heaviside(values=self.values)


class ComplexSinc(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: Any) -> _Complex:
        if not isinstance(x, _Complex):
            x = _Complex(x, *self.args, **self.kwargs)
        return x.sinc()

class _ComplexLinear(nn.Module):
    """
    Complex linear transformation module for Complex objects.
    y = Wz + b = (Wr + iWi)(xr + ixi) + (br + ibi)
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.real_linear = nn.Linear(in_features, out_features, bias=bias)
        self.imag_linear = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, x: _Complex) -> _Complex:
        # Standard complex linear implementation:
        # y_real = Wr*xr - Wi*xi + br
        # y_imag = Wr*xi + Wi*xr + bi
        xr, xi = x.real, x.imag
        
        out_real = self.real_linear(xr) - self.imag_linear(xi)
        out_imag = self.real_linear(xi) + self.imag_linear(xr)
        
        # Combine back into Complex object
        return _Complex(torch.stack([out_real, out_imag], dim=x.dim), dim=x.dim, is_stacked_flag=True)


class ComplexCustomStringActivationLayer(_Complex):
    """
    ComplexCustomStringActivationLayer.
    
    A flexible complex activation layer that performs operations based on a custom string.
    Supports complex linear transformations (x1, x2, ...) and multiple activation functions (A1, A2, ...).
    
    Args:
        in_features (int): Number of input features (per real/imag part).
        act_operation (str): String describing the operation (e.g., "x + abs(x1) + A1(x2)").
        act_funcs (Union[str, nn.Module, Callable, List]): One or more activation functions.
        biases (List[bool]): Whether to use bias for each linear transformation.
    """
    def __init__(self,
                 in_features: int,
                 act_operation: str,
                 act_funcs: Union[str, nn.Module, Callable, List[str], nn.ModuleList, List[Any]],
                 dims: Optional[Union[int, Tuple[int]]] = None,
                 biases: Optional[List[bool]] = None,
                 name: Optional[str] = None,
                 *args, **kwargs):
        # Initialize as a Complex object with no initial data
        super().__init__(data=None, *args, **kwargs)
        from ..ActivationFunction import ExpressionParser
        
        self.in_features = in_features
        
        self.in_features = in_features

        # Registry Check
        self.registry_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'custom_string.json')
        self.raw_operation = self._check_registry(act_operation, name)
        
        self.funcs = nn.ModuleList([])
        self.args = args
        self.kwargs = kwargs
        
        self.__pre_process_funcs__(act_funcs)
        
        # Parse x1, x2, ... to determine number of complex linear transformations
        x_indices = [int(n) for n in re.findall(r'\bx(\d+)\b', self.raw_operation)]
        num_transforms = max(x_indices) if x_indices else 0
        
        _biases: List[bool] = biases if biases is not None else []
        if len(_biases) < num_transforms:
            _biases.extend([True] * (num_transforms - len(_biases)))
            
        self.linear_layers = nn.ModuleList([
            _ComplexLinear(in_features, in_features, bias=_biases[i]) for i in range(num_transforms)
        ])

        # Parse w1, w2, ... and b1, b2, ...
        # Parse w1, w2, ... and b1, b2, ...
        self.dims = dims
        self.custom_params = nn.ParameterDict()
        
        w_indices = set(int(n) for n in re.findall(r'\bw(\d+)\b', self.raw_operation))
        b_indices = set(int(n) for n in re.findall(r'\bb(\d+)\b', self.raw_operation))
        
        _dims = self.dims
        if (w_indices or b_indices) and _dims is None:
             raise ValueError("dims must be provided when using w or b parameters in abstract strings.")

        for idx in w_indices:
            # simple initialization for weights
            if isinstance(_dims, int):
                shape = (_dims,)
            else:
                shape = tuple(_dims) # type: ignore
            self.custom_params[f'w{idx}'] = nn.Parameter(torch.randn(*shape))

        for idx in b_indices:
            # biases use last dim
            if isinstance(_dims, int):
                b_dim = _dims
            else:
                b_dim = _dims[-1] if _dims is not None else 1 # type: ignore
            self.custom_params[f'b{idx}'] = nn.Parameter(torch.zeros(b_dim))
        
        self.parser = ExpressionParser()
        self.math_ops = self.parser.get_executable_context()
        self.executable_expression = self.parser.transpile(self.raw_operation, len(self.funcs))
        self.compiled_code = compile(self.executable_expression, '<string>', 'eval')

    def _check_registry(self, act_operation: str, name: Optional[str]) -> str:
        registry: Dict[str, str] = {}
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    content = f.read().strip()
                    if content:
                         loaded = json.loads(content)
                         if isinstance(loaded, dict):
                             registry.update(loaded)
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
        if not isinstance(funcs, (list, nn.ModuleList)):
            funcs = [funcs]
            
        for func in funcs:
            if isinstance(func, str):
                self.funcs.append(ComplexActivation(func, **self.kwargs))
            elif isinstance(func, nn.Module):
                self.funcs.append(func)
            elif callable(func):
                # We assume generic callables are complex-safe or handle them as is
                self.funcs.append(func)
            else:
                raise TypeError(f"Unsupported complex activation function type: {type(func)}")



    def forward(self, x: Any) -> _Complex:
        if not isinstance(x, _Complex):
            x = _Complex(x, dim=self.dim if self.dim is not None else -1, device=self.device, dtype=self.dtype)
        
        # Consistent stacking dimension resolution
        actual_dim = x.dim
        if actual_dim < 0:
            actual_dim = x.tensor.dim() + actual_dim

        # 1. Convert everything to raw torch complex tensors for the 'eval' call
        def to_c(z):
            # Only unstack custom Complex objects (detecting .tensor and .dim)
            if hasattr(z, 'tensor') and hasattr(z, 'dim') and not isinstance(z, torch.Tensor):
                r = z.real
                i = z.imag
                if callable(r): r = r()
                if callable(i): i = i()
                
                # Failsafe: Ensure rank 2 for eval components BEFORE torch.complex
                if isinstance(r, torch.Tensor):
                    while r.dim() > 2: r = r.unbind(-1)[0]
                if isinstance(i, torch.Tensor):
                    while i.dim() > 2: i = i.unbind(-1)[0]

                res = torch.complex(r, i)
                return res
            return z

        raw_x_c = to_c(x)
        feats_c = [to_c(layer(x)) for layer in self.linear_layers]
        
        # 2. Wrap activation functions to return raw complex tensors
        acts_wrapped = []
        for func in self.funcs:
            def wrapped(z, f=func):
                res = f(z)
                return to_c(res)
            acts_wrapped.append(wrapped)
            
        context = {
            'raw_x': raw_x_c,
            'feats': feats_c,
            'acts': acts_wrapped,
            'acts': acts_wrapped,
            'custom_params': self.custom_params,
            'ops': self.math_ops,
            'torch': torch
        }
        
        try:
            # 3. Evaluate expression in standard PyTorch space
            res_c = eval(self.compiled_code, {}, context)
            
            # Ensure the result is a tensor and has correct type
            if not isinstance(res_c, torch.Tensor):
                res_c = torch.as_tensor(res_c, device=x.device, dtype=x.dtype)
            elif not res_c.is_complex() and not res_c.is_floating_point():
                res_c = res_c.to(x.dtype)

            # 4. Re-wrap the result into a Complex object using the original stacking dimension
            return _Complex(res_c, dim=actual_dim)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to evaluate expression: '{self.raw_operation}'\nError: {e}") from e

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        return self is other


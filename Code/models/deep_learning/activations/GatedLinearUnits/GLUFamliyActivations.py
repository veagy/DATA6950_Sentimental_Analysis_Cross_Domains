import torch
import torch.nn as nn
from typing import Any, Union, Tuple, List, Callable, Dict
import re
import re
import math
import json
import os
__all__ = [
    'CustomStringActivationLayer',
    'EGLUActivation',
    'GEGLUActivation',
    'GLUActivation',
    'GTUActivation',
    'GatedLinearUnits',
    'LiGLUActivation',
    'ReGLUActivation',
    'SEGLUActivation',
    'SwiGLUActivation',
]


class _LinearTransformation(nn.Module):
    def __init__(self, dims: tuple, **kwargs):
        super().__init__()
        self.dim = kwargs.get('dim', -1)
        in_features = dims[self.dim]
        out_features = kwargs.get('out_features', in_features)
        bias = kwargs.get('bias', False)
        self._linear1 = nn.Linear(in_features, out_features, bias)
        self._linear2 = nn.Linear(in_features, out_features, bias)

    def _apply_linear(self, x):
        if self.dim == -1 or self.dim == x.ndim - 1:
            return self._linear1(x), self._linear2(x)

        # Transpose if not operating on last dimension
        x = x.transpose(self.dim, -1)
        x1 = self._linear1(x)
        x2 = self._linear2(x)
        x1 = x1.transpose(self.dim, -1)
        x2 = x2.transpose(self.dim, -1)
        return x1, x2


class GatedLinearUnits(_LinearTransformation):
    """
    Generic Gated Linear Unit.
    """

    def __init__(self, activation: str, dims: tuple, **kwargs):
        super().__init__(dims, **kwargs)
        from ..ActivationFunction import Activation
        self.func = Activation(activation, **kwargs)
        self.activation_name = activation

    def forward(self, x):
        x1, x2 = self._apply_linear(x)
        return x1 * self.func(x2)


class GLUActivation(_LinearTransformation):
    """
    Generic Gated Linear Unit.
    """

    def __init__(self, activation1: str, activation2: str, dims: tuple, **kwargs):
        super().__init__(dims, **kwargs)
        from ..ActivationFunction import Activation
        self.func1 = Activation(activation1, **kwargs)
        self.func2 = Activation(activation2, **kwargs)

    def forward(self, x):
        x1, x2 = self._apply_linear(x)
        return self.func1(x1) * self.func2(x2)


class GTUActivation(_LinearTransformation):
    """
    Gated Tanh Unit: tanh(x1) * sigmoid(x2)
    """

    def __init__(self, dims: tuple, **kwargs):
        super().__init__(dims, **kwargs)

    def forward(self, x):
        x1, x2 = self._apply_linear(x)
        return torch.tanh(x1) * torch.sigmoid(x2)


class ReGLUActivation(_LinearTransformation):
    """
    ReLU Gated Linear Unit: x1 * relu(x2)
    """

    def __init__(self, dims: tuple, **kwargs):
        super().__init__(dims, **kwargs)

    def forward(self, x):
        x1, x2 = self._apply_linear(x)
        return x1 * torch.relu(x2)


class GEGLUActivation(_LinearTransformation):
    """
    GELU Gated Linear Unit: x1 * gelu(x2)
    """

    def __init__(self, dims: tuple, **kwargs):
        super().__init__(dims, **kwargs)
        from ..ActivationFunction import Activation
        # Use regex to find GELU variants if needed, or exact match
        act_name = "ParametricGELUActivation" if kwargs.get('gelu_trainable', False) else "GELUActivation"
        self.gelu = Activation(act_name, **kwargs)

    def forward(self, x):
        x1, x2 = self._apply_linear(x)
        return x1 * self.gelu(x2)


class SwiGLUActivation(_LinearTransformation):
    """
    Swish Gated Linear Unit (SiLU): x1 * silu(x2)
    """

    def __init__(self, dims: tuple, **kwargs):
        super().__init__(dims, **kwargs)
        from ..ActivationFunction import Activation
        act_name = "ParametricSILUActivation" if kwargs.get('gelu_trainable', False) else "SILUActivation"
        self.silu = Activation(act_name, **kwargs)

    def forward(self, x):
        x1, x2 = self._apply_linear(x)
        return x1 * self.silu(x2)


class EGLUActivation(_LinearTransformation):
    """
    Exponential Gated Linear Unit: x1 * gaussian(x2)
    """

    def __init__(self, dims: tuple, **kwargs):
        super().__init__(dims, **kwargs)

    def forward(self, x):
        x1, x2 = self._apply_linear(x)
        return x1 * torch.exp(-(x2 ** 2))


class LiGLUActivation(_LinearTransformation):
    """
    Exponential Gated Linear Unit: x1 * gaussian(x2)
    """

    def __init__(self, dims: tuple, **kwargs):
        super().__init__(dims, **kwargs)

    def forward(self, x):
        x1, x2 = self._apply_linear(x)
        return x1 * x2


class SEGLUActivation(_LinearTransformation):
    """
    Exponential Gated Linear Unit: x1 * gaussian(x2)
    """

    def __init__(self, dims: tuple, **kwargs):
        super().__init__(dims, **kwargs)

    def forward(self, x):
        x1, x2 = self._apply_linear(x)
        return torch.sigmoid(x1) * torch.exp(x2)


class CustomStringActivationLayer(nn.Module):
    """
    CustomStringActivationLayer.

    A flexible activation layer that performs operations based on a custom string.
    Supports trainable linear transformations (x1, x2, ...) and multiple activation functions (A1, A2, ...).

    Args:
        in_features (int): Number of input features.
        act_operation (str): String describing the operation (e.g., "x + abs(x1) + A1(x2)").
        act_funcs (Union[str, nn.Module, Callable, List]): One or more activation functions.
        biases (List[bool]): Whether to use bias for each linear transformation.
    """

    def __init__(self,
                 in_features: int,
                 act_operation: str,
                 act_funcs: Union[str, nn.Module, Callable, List[str], nn.ModuleList, List[Callable], Any],
                 dims: Union[int, Tuple[int]] = None,
                 biases: List[bool] = None,
                 name: str = None,
                 *args, **kwargs):
        super().__init__()
        from ..ActivationFunction import ExpressionParser
        self.in_features = in_features
        # self.raw_operation = act_operation # Replaced by registry check below
        self.funcs = nn.ModuleList([])
        self.args = args
        self.kwargs = kwargs

        # Registry Check
        self.registry_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                          'custom_string.json')
        self.raw_operation = self._check_registry(act_operation, name)

        self.__pre_process_funcs__(act_funcs)

        # Parse x1, x2, ... to determine number of linear transformations
        x_indices = [int(n) for n in re.findall(r'\bx(\d+)\b', self.raw_operation)]
        num_transforms = max(x_indices) if x_indices else 0

        if biases is None:
            biases = [True for _ in range(num_transforms)]
        elif len(biases) < num_transforms:
            # Pad biases if not enough provided
            biases = biases + [True for _ in range(num_transforms - len(biases))]

        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features, in_features, bias=b) for b in biases[:num_transforms]
        ])

        # Parse w1, w2, ... and b1, b2, ...
        self.dims = dims
        self.custom_params = nn.ParameterDict()

        w_indices = set(int(n) for n in re.findall(r'\bw(\d+)\b', self.raw_operation))
        b_indices = set(int(n) for n in re.findall(r'\bb(\d+)\b', self.raw_operation))

        if (w_indices or b_indices) and self.dims is None:
            raise ValueError("dims must be provided when using w or b parameters in abstract strings.")

        for idx in w_indices:
            # simple initialization for weights
            if isinstance(self.dims, int):
                shape = (self.dims,)
            else:
                shape = self.dims
            self.custom_params[f'w{idx}'] = nn.Parameter(torch.randn(*shape))

        for idx in b_indices:
            # biases use last dim
            if isinstance(self.dims, int):
                b_dim = self.dims
            else:
                b_dim = self.dims[-1]
            self.custom_params[f'b{idx}'] = nn.Parameter(torch.zeros(b_dim))

        self.parser = ExpressionParser()
        self.math_ops = self.parser.get_executable_context()
        self.executable_expression = self.parser.transpile(self.raw_operation, len(self.funcs))
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
        from ..Adaptive.Mixture.mixture_of_activations import ActFuncWrapper

        # Convert to list if it's a single item
        if not isinstance(funcs, (list, nn.ModuleList)):
            funcs = [funcs]

        for func in funcs:
            if isinstance(func, str):
                self.funcs.append(Activation(func, *self.args, **self.kwargs))
            elif isinstance(func, nn.Module):
                self.funcs.append(func)
            elif callable(func):
                self.funcs.append(ActFuncWrapper(func, *self.args, **self.kwargs))
            else:
                raise TypeError(f"Unsupported activation function type: {type(func)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_x = x
        # Apply linear transformations
        feats = [layer(x) for layer in self.linear_layers]

        # evaluation context
        context = {
            'raw_x': raw_x,
            'feats': feats,
            'acts': self.funcs,
            'custom_params': self.custom_params,
            'ops': self.math_ops,
            'torch': torch
        }

        try:
            return eval(self.compiled_code, {}, context)
        except Exception as e:
            raise RuntimeError(
                f"Failed to evaluate expression: '{self.raw_operation}'\n"
                f"Transpiled to: '{self.executable_expression}'\n"
                f"Error: {str(e)}"
            ) from e

"""Config templates for misc."""
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from .....config import ConfigTemplate

try:
    import torch
except ImportError:
    torch = None

"""Generated config for FunctionTransformer."""
class FunctionTransformerConfig(ConfigTemplate):
    model_name = "FunctionTransformer"
    model_path = "Code.models.machine_learning.preprocessing.misc.misc"

    def __init__(self,
        immutable: bool = True,
        func: Union[str, Callable, nn.Module] = None,
        inverse_func: Union[str, Callable, nn.Module] = None,
        validate: bool = False,
        accept_sparse: bool = False,
        check_inverse: bool = False,
        feature_names_out: Union[Literal['one-to-one'], Callable, nn.Module] = None,
        kw_args: dict = None,
        inv_kw_args: dict = None,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.func = func
        self.inverse_func = inverse_func
        self.validate = validate
        self.accept_sparse = accept_sparse
        self.check_inverse = check_inverse
        self.feature_names_out = feature_names_out
        self.kw_args = kw_args
        self.inv_kw_args = inv_kw_args
        self.device = device
        self.dtype = dtype


"""Generated config for KernelCenterer."""
class KernelCentererConfig(ConfigTemplate):
    model_name = "KernelCenterer"
    model_path = "Code.models.machine_learning.preprocessing.misc.misc"

    def __init__(self,
        immutable: bool = True,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.device = device
        self.dtype = dtype


"""Generated config for PolynomialFeatures."""
class PolynomialFeaturesConfig(ConfigTemplate):
    model_name = "PolynomialFeatures"
    model_path = "Code.models.machine_learning.preprocessing.misc.misc"

    def __init__(self,
        immutable: bool = True,
        degree: Union[int, tuple] = 2,
        interaction_only: bool = False,
        include_bias: bool = True,
        order: Literal['C', 'F'] = 'C',
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.order = order
        self.device = device
        self.dtype = dtype


"""Generated config for SplineTransformers."""
class SplineTransformersConfig(ConfigTemplate):
    model_name = "SplineTransformers"
    model_path = "Code.models.machine_learning.preprocessing.misc.misc"

    def __init__(self,
        immutable: bool = True,
        n_knots: int = 5,
        degree: int = 3,
        knots: Union[Literal['uniform', 'quantile'], list, tuple, torch.Tensor] = 'uniform',
        extrapolation: Literal['error', 'constant', 'linear', 'continue', 'periodic'] = 'constant',
        include_bias: bool = True,
        order: Literal['C', 'F'] = 'C',
        handle_missing: Literal['error', 'zeros'] = 'error',
        sparse_output: bool = False,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_knots = n_knots
        self.degree = degree
        self.knots = knots
        self.extrapolation = extrapolation
        self.include_bias = include_bias
        self.order = order
        self.handle_missing = handle_missing
        self.sparse_output = sparse_output
        self.device = device
        self.dtype = dtype

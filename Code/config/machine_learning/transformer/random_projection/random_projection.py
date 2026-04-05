"""Config templates for random_projection."""
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from .....config import ConfigTemplate

try:
    import torch
except ImportError:
    torch = None

"""Generated config for GaussianRandomProjection."""
class GaussianRandomProjectionConfig(ConfigTemplate):
    model_name = "GaussianRandomProjection"
    model_path = "Code.models.machine_learning.transformer.random_projection.random_projectio"

    def __init__(self,
        immutable: bool = True,
        n_components: Union[Literal['auto'], int] = 'auto',
        eps: float = 0.1,
        compute_inverse_components: bool = False,
        random_state: Union[int, torch.Generator] = None,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_components = n_components
        self.eps = eps
        self.compute_inverse_components = compute_inverse_components
        self.random_state = random_state
        self.device = device
        self.dtype = dtype


"""Generated config for SparseRandomProjection."""
class SparseRandomProjectionConfig(ConfigTemplate):
    model_name = "SparseRandomProjection"
    model_path = "Code.models.machine_learning.transformer.random_projection.random_projectio"

    def __init__(self,
        immutable: bool = True,
        n_components: Union[Literal['auto'], int] = 'auto',
        density: Union[Literal['auto'], float] = 'auto',
        eps: float = 0.1,
        dense_output: bool = False,
        compute_inverse_components: bool = False,
        random_state: Union[int, torch.Generator] = None,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_components = n_components
        self.density = density
        self.eps = eps
        self.dense_output = dense_output
        self.compute_inverse_components = compute_inverse_components
        self.random_state = random_state
        self.device = device
        self.dtype = dtype

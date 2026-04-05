"""Config templates for cross_decomp_models."""
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from .....config import ConfigTemplate

try:
    import torch
except ImportError:
    torch = None

"""Generated config for CCA."""
class CCAConfig(ConfigTemplate):
    model_name = "CCA"
    model_path = "Code.models.machine_learning.regression.cross_decomposition.cross_decomp_models"

    def __init__(self,
        immutable: bool = True,
        n_components: int = 2,
        scale: bool = True,
        max_iter: int = 500,
        tol: float = 1e-06,
        copy: bool = True,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_components = n_components
        self.scale = scale
        self.max_iter = max_iter
        self.tol = tol
        self.copy = copy
        self.device = device
        self.dtype = dtype


"""Generated config for PLSCanonical."""
class PLSCanonicalConfig(ConfigTemplate):
    model_name = "PLSCanonical"
    model_path = "Code.models.machine_learning.regression.cross_decomposition.cross_decomp_models"

    def __init__(self,
        immutable: bool = True,
        n_components: int = 2,
        scale: bool = True,
        algorithm: str = 'nipals',
        max_iter: int = 500,
        tol: float = 1e-06,
        copy: bool = True,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_components = n_components
        self.scale = scale
        self.algorithm = algorithm
        self.max_iter = max_iter
        self.tol = tol
        self.copy = copy
        self.device = device
        self.dtype = dtype


"""Generated config for PLSRegression."""
class PLSRegressionConfig(ConfigTemplate):
    model_name = "PLSRegression"
    model_path = "Code.models.machine_learning.regression.cross_decomposition.cross_decomp_models"

    def __init__(self,
        immutable: bool = True,
        n_components: int = 2,
        scale: bool = True,
        max_iter: int = 500,
        tol: float = 1e-06,
        copy: bool = True,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_components = n_components
        self.scale = scale
        self.max_iter = max_iter
        self.tol = tol
        self.copy = copy
        self.device = device
        self.dtype = dtype


"""Generated config for PLSSVD."""
class PLSSVDConfig(ConfigTemplate):
    model_name = "PLSSVD"
    model_path = "Code.models.machine_learning.regression.cross_decomposition.cross_decomp_models"

    def __init__(self,
        immutable: bool = True,
        n_components: int = 2,
        scale: bool = True,
        copy: bool = True,
        algorithm: Union[str, Callable] = 'full',
        max_iter: int = 500,
        tol: float = 1e-06,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_components = n_components
        self.scale = scale
        self.copy = copy
        self.algorithm = algorithm
        self.max_iter = max_iter
        self.tol = tol
        self.device = device
        self.dtype = dtype

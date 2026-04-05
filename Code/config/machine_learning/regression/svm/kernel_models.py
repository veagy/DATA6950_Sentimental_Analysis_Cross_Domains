"""Config templates for kernel_models."""
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from .....config import ConfigTemplate

try:
    import torch
except ImportError:
    torch = None

"""Generated config for BaseKernelModel."""
class BaseKernelModelConfig(ConfigTemplate):
    model_name = "BaseKernelModel"
    model_path = "Code.models.machine_learning.regression.svm.kernel_models"

    def __init__(self,
        immutable: bool = True,
        alpha: Union[float, List[float], Tuple[float], torch.Tensor] = 1.0,
        kernel: Union[str, Callable, nn.Module, MLModule] = 'linear',
        gamma: Union[float, str] = None,
        degree: float = 3,
        coef0: float = 1,
        kernel_params: dict = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.device = device
        self.dtype = dtype


"""Generated config for KernelElasticNet."""
class KernelElasticNetConfig(ConfigTemplate):
    model_name = "KernelElasticNet"
    model_path = "Code.models.machine_learning.regression.svm.kernel_models"

    def __init__(self,
        immutable: bool = True,
        alpha: Union[float, List[float], Tuple[float], torch.Tensor] = 1.0,
        l1_ratio: float = 0.5,
        kernel: Union[str, Callable] = 'linear',
        gamma: Union[float, str] = None,
        degree: float = 3,
        coef0: float = 1,
        kernel_params: dict = None,
        fit_intercept: bool = True,
        precompute: bool = False,
        max_iter: int = 1000,
        copy_X: bool = True,
        tol: float = 0.0001,
        warm_start: bool = False,
        positive: bool = False,
        random_state: int = None,
        selection: str = 'cyclic',
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.fit_intercept = fit_intercept
        self.precompute = precompute
        self.max_iter = max_iter
        self.copy_X = copy_X
        self.tol = tol
        self.warm_start = warm_start
        self.positive = positive
        self.random_state = random_state
        self.selection = selection
        self.device = device
        self.dtype = dtype


"""Generated config for KernelLars."""
class KernelLarsConfig(ConfigTemplate):
    model_name = "KernelLars"
    model_path = "Code.models.machine_learning.regression.svm.kernel_models"

    def __init__(self,
        immutable: bool = True,
        fit_intercept: bool = True,
        verbose: Union[bool, int] = False,
        precompute: Union[bool, str, List, Tuple, torch.Tensor] = 'auto',
        n_nonzero_coefs: int = 500,
        eps: float = torch.finfo(torch.float32).eps,
        fit_path: bool = True,
        max_iter: int = None,
        jitter: float = 1e-06,
        random_state: int = None,
        kernel: Union[str, Callable] = 'linear',
        gamma: Union[float, str] = None,
        degree: float = 3,
        coef0: float = 1,
        kernel_params: dict = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.precompute = precompute
        self.n_nonzero_coefs = n_nonzero_coefs
        self.eps = eps
        self.fit_path = fit_path
        self.max_iter = max_iter
        self.jitter = jitter
        self.random_state = random_state
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.device = device
        self.dtype = dtype


"""Generated config for KernelLasso."""
class KernelLassoConfig(ConfigTemplate):
    model_name = "KernelLasso"
    model_path = "Code.models.machine_learning.regression.svm.kernel_models"

    def __init__(self,
        immutable: bool = True,
        alpha: Union[float, List[float], Tuple[float], torch.Tensor] = 1.0,
        kernel: Union[str, Callable] = 'linear',
        gamma: Union[float, str] = None,
        degree: float = 3,
        coef0: float = 1,
        kernel_params: dict = None,
        fit_intercept: bool = True,
        precompute: bool = False,
        copy_X: bool = True,
        max_iter: int = 1000,
        tol: float = 0.0001,
        warm_start: bool = False,
        positive: bool = False,
        random_state: int = None,
        selection: str = 'cyclic',
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.fit_intercept = fit_intercept
        self.precompute = precompute
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.positive = positive
        self.random_state = random_state
        self.selection = selection
        self.device = device
        self.dtype = dtype


"""Generated config for KernelRidge."""
class KernelRidgeConfig(ConfigTemplate):
    model_name = "KernelRidge"
    model_path = "Code.models.machine_learning.regression.svm.kernel_models"

    def __init__(self,
        immutable: bool = True,
        alpha: Union[float, List[float], Tuple[float], torch.Tensor] = 1.0,
        kernel: Union[str, Callable] = 'linear',
        gamma: Union[float, str] = None,
        degree: float = 3,
        coef0: float = 1,
        kernel_params: dict = None,
        fit_intercept: bool = True,
        max_iter: int = None,
        tol: float = 0.001,
        solver: str = 'auto',
        random_state: int = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.random_state = random_state
        self.device = device
        self.dtype = dtype

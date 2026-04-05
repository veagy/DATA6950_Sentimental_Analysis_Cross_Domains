"""Config templates for svm."""
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from .....config import ConfigTemplate

try:
    import torch
except ImportError:
    torch = None

"""Generated config for LinearSVR."""
class LinearSVRConfig(ConfigTemplate):
    model_name = "LinearSVR"
    model_path = "Code.models.machine_learning.regression.svm.svm"

    def __init__(self,
        immutable: bool = True,
        tol: float = 0.001,
        C: float = 1.0,
        epsilon: float = 0.1,
        loss: str = 'epsilon_insensitive',
        fit_intercept: bool = True,
        intercept_scaling: float = 1.0,
        dual: Union[str, bool] = 'auto',
        shrinking: float = True,
        cache_size: float = 200,
        verbose: bool = False,
        max_iter: int = 1000,
        random_state: int = None,
        trainable_kernel: bool = False,
        n_support_vectors: int = 100,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.tol = tol
        self.C = C
        self.epsilon = epsilon
        self.loss = loss
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.dual = dual
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.random_state = random_state
        self.trainable_kernel = trainable_kernel
        self.n_support_vectors = n_support_vectors
        self.device = device
        self.dtype = dtype


"""Generated config for NuSVR."""
class NuSVRConfig(ConfigTemplate):
    model_name = "NuSVR"
    model_path = "Code.models.machine_learning.regression.svm.svm"

    def __init__(self,
        immutable: bool = True,
        kernel: Union[str, Callable, MLModule, nn.Module] = 'rbf',
        gamma: Union[str, float] = 'scale',
        degree: int = 3,
        coef0: float = 0.0,
        tol: float = 0.001,
        nu: float = 0.5,
        C: float = 1.0,
        epsilon: float = 0.1,
        shrinking: float = True,
        cache_size: float = 200,
        verbose: bool = False,
        max_iter: int = 1000,
        trainable_kernel: bool = False,
        n_support_vectors: int = 100,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.nu = nu
        self.C = C
        self.epsilon = epsilon
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.trainable_kernel = trainable_kernel
        self.n_support_vectors = n_support_vectors
        self.device = device
        self.dtype = dtype


"""Generated config for SVR."""
class SVRConfig(ConfigTemplate):
    model_name = "SVR"
    model_path = "Code.models.machine_learning.regression.svm.svm"

    def __init__(self,
        immutable: bool = True,
        kernel: Union[str, Callable, MLModule, nn.Module] = 'rbf',
        gamma: Union[str, float] = 'scale',
        degree: int = 3,
        coef0: float = 0.0,
        tol: float = 0.001,
        C: float = 1.0,
        epsilon: float = 0.1,
        shrinking: float = True,
        cache_size: float = 200,
        verbose: bool = False,
        max_iter: int = 1000,
        trainable_kernel: bool = False,
        n_support_vectors: int = 100,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.C = C
        self.epsilon = epsilon
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.trainable_kernel = trainable_kernel
        self.n_support_vectors = n_support_vectors
        self.device = device
        self.dtype = dtype

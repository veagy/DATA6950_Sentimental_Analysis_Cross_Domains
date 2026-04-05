"""Config templates for svm."""
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from .....config import ConfigTemplate

try:
    import torch
except ImportError:
    torch = None

"""Generated config for LinearSVC."""
class LinearSVCConfig(ConfigTemplate):
    model_name = "LinearSVC"
    model_path = "Code.models.machine_learning.classification.svm.svm"

    def __init__(self,
        immutable: bool = True,
        penalty: str = 'l2',
        loss: str = 'squared_hinge',
        dual: Union[str, bool] = 'auto',
        tol: float = 0.0001,
        C: float = 1.0,
        multi_class: str = 'ovr',
        fit_intercept: bool = True,
        intercept_scaling: float = 1.0,
        class_weight: Union[str, dict] = None,
        verbose: bool = False,
        random_state: int = None,
        max_iter: int = 1000,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.penalty = penalty
        self.loss = loss
        self.dual = dual
        self.tol = tol
        self.C = C
        self.multi_class = multi_class
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.verbose = verbose
        self.random_state = random_state
        self.max_iter = max_iter
        self.device = device
        self.dtype = dtype


"""Generated config for NuSVC."""
class NuSVCConfig(ConfigTemplate):
    model_name = "NuSVC"
    model_path = "Code.models.machine_learning.classification.svm.svm"

    def __init__(self,
        immutable: bool = True,
        nu: float = 0.5,
        kernel: Union[str, Callable, MLModule, nn.Module] = 'rbf',
        degree: int = 3,
        gamma: Union[str, float] = 'scale',
        coef0: float = 0.0,
        shrinking: bool = True,
        probability: bool = False,
        tol: float = 0.001,
        cache_size: float = 200,
        class_weight: Union[str, dict] = None,
        verbose: bool = False,
        max_iter: int = -1,
        decision_function_shape: str = 'ovr',
        break_ties: bool = False,
        random_state: int = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.nu = nu
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.break_ties = break_ties
        self.random_state = random_state
        self.device = device
        self.dtype = dtype


"""Generated config for SVC."""
class SVCConfig(ConfigTemplate):
    model_name = "SVC"
    model_path = "Code.models.machine_learning.classification.svm.svm"

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
        probability: bool = False,
        class_weight: Union[str, dict] = None,
        decision_function_shape: str = 'ovr',
        break_ties: bool = False,
        random_state: int = None,
        cache_size: float = 200,
        verbose: bool = False,
        max_iter: int = 1000,
        trainable_kernel: bool = False,
        n_support_vectors: int = 100,
        warm_start: bool = False,
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
        self.probability = probability
        self.class_weight = class_weight
        self.decision_function_shape = decision_function_shape
        self.break_ties = break_ties
        self.random_state = random_state
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.trainable_kernel = trainable_kernel
        self.n_support_vectors = n_support_vectors
        self.warm_start = warm_start
        self.device = device
        self.dtype = dtype

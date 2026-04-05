"""Config templates for pca."""
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from .....config import ConfigTemplate

try:
    import torch
except ImportError:
    torch = None

"""Generated config for IncrementalPCA."""
class IncrementalPCAConfig(ConfigTemplate):
    model_name = "IncrementalPCA"
    model_path = "Code.models.machine_learning.transformer.pca.pca"

    def __init__(self,
        immutable: bool = True,
        n_components: Optional[int] = None,
        whiten: bool = False,
        copy: bool = True,
        batch_size: Optional[int] = None,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        method_solver_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_components = n_components
        self.whiten = whiten
        self.copy = copy
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.method_solver_kwargs = method_solver_kwargs


"""Generated config for KernelPCA."""
class KernelPCAConfig(ConfigTemplate):
    model_name = "KernelPCA"
    model_path = "Code.models.machine_learning.transformer.pca.pca"

    def __init__(self,
        immutable: bool = True,
        n_components: Optional[int] = None,
        kernel: Union[Literal['linear', 'poly', 'rbf', 'sigmoid', 'cosine', 'precomputed'], Callable[..., torch.Tensor], nn.Module] = 'linear',
        gamma: Optional[Union[int, float]] = None,
        degree: Union[int, float] = 3,
        coef0: float = 1,
        kernel_params: Optional[Dict[str, Any]] = None,
        alpha: float = 1.0,
        fit_inverse_transform: bool = False,
        eigen_solver: Union[Literal['auto', 'dense', 'arpack', 'randomized'], Callable[..., Tuple[torch.Tensor, torch.Tensor]], nn.Module] = 'auto',
        tol: float = 0.0,
        max_iter: Optional[int] = None,
        iterated_power: Union[int, Literal['auto']] = 'auto',
        remove_zero_eig: bool = False,
        random_state: Optional[Union[int, torch.Generator]] = None,
        copy_X: bool = True,
        n_jobs: Optional[int] = None,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        eigen_solver_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.alpha = alpha
        self.fit_inverse_transform = fit_inverse_transform
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.iterated_power = iterated_power
        self.remove_zero_eig = remove_zero_eig
        self.random_state = random_state
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype
        self.eigen_solver_kwargs = eigen_solver_kwargs


"""Generated config for MiniBatchSparsePCA."""
class MiniBatchSparsePCAConfig(ConfigTemplate):
    model_name = "MiniBatchSparsePCA"
    model_path = "Code.models.machine_learning.transformer.pca.pca"

    def __init__(self,
        immutable: bool = True,
        n_components: Optional[int] = None,
        alpha: float = 1.0,
        ridge_alpha: float = 0.01,
        max_iter: int = 1000,
        tol: float = 0.001,
        method: Union[Literal['lars', 'cd'], Callable[..., torch.Tensor], nn.Module] = 'lars',
        callback: Optional[Union[Callable[..., None], nn.Module]] = None,
        batch_size: int = 3,
        shuffle: bool = True,
        max_no_improvement: Optional[int] = 10,
        n_jobs: Optional[int] = None,
        U_init: Optional[Union[List, Tuple, torch.Tensor]] = None,
        V_init: Optional[Union[List, Tuple, torch.Tensor]] = None,
        verbose: Union[int, bool] = False,
        random_state: Optional[Union[torch.Generator, int]] = None,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        method_solver_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_components = n_components
        self.alpha = alpha
        self.ridge_alpha = ridge_alpha
        self.max_iter = max_iter
        self.tol = tol
        self.method = method
        self.callback = callback
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_no_improvement = max_no_improvement
        self.n_jobs = n_jobs
        self.U_init = U_init
        self.V_init = V_init
        self.verbose = verbose
        self.random_state = random_state
        self.device = device
        self.dtype = dtype
        self.method_solver_kwargs = method_solver_kwargs


"""Generated config for PCA."""
class PCAConfig(ConfigTemplate):
    model_name = "PCA"
    model_path = "Code.models.machine_learning.transformer.pca.pca"

    def __init__(self,
        immutable: bool = True,
        n_components: Optional[Union[int, float, Literal['mle']]] = None,
        copy: bool = True,
        whiten: bool = False,
        svd_solver: Union[Literal['auto', 'full', 'covariance_eigh', 'arpack', 'randomized'], Callable[..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], nn.Module] = 'auto',
        tol: float = 0.0,
        iterated_power: Union[int, Literal['auto']] = 'auto',
        n_oversamples: int = 10,
        power_iteration_normalizer: Union[Literal['auto', 'QR', 'LU', 'none'], Callable[..., Any], nn.Module] = 'auto',
        random_state: Optional[Union[int, torch.Generator]] = None,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        svd_solver_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.n_oversamples = n_oversamples
        self.power_iteration_normalizer = power_iteration_normalizer
        self.random_state = random_state
        self.device = device
        self.dtype = dtype
        self.svd_solver_kwargs = svd_solver_kwargs


"""Generated config for SparsePCA."""
class SparsePCAConfig(ConfigTemplate):
    model_name = "SparsePCA"
    model_path = "Code.models.machine_learning.transformer.pca.pca"

    def __init__(self,
        immutable: bool = True,
        n_components: Optional[int] = None,
        alpha: float = 1.0,
        ridge_alpha: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-08,
        method: Union[Literal['lars', 'cd'], Callable[..., torch.Tensor], nn.Module] = 'lars',
        n_jobs: Optional[int] = None,
        U_init: Optional[Union[List, Tuple, torch.Tensor]] = None,
        V_init: Optional[Union[List, Tuple, torch.Tensor]] = None,
        verbose: Union[int, bool] = False,
        random_state: Optional[Union[torch.Generator, int]] = None,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        method_solver_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_components = n_components
        self.alpha = alpha
        self.ridge_alpha = ridge_alpha
        self.max_iter = max_iter
        self.tol = tol
        self.method = method
        self.n_jobs = n_jobs
        self.U_init = U_init
        self.V_init = V_init
        self.verbose = verbose
        self.random_state = random_state
        self.device = device
        self.dtype = dtype
        self.method_solver_kwargs = method_solver_kwargs

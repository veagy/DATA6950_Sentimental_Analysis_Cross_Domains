import torch
import torch.nn as nn
from typing import Callable, Union, List, Tuple
from .....models.utils import MLModule
import warnings
from .kernels import *
from ..linear_model import Lasso, ElasticNet, Lars
from torch.func import vmap
import joblib

__all__ = [
    "KernelRidge",
    "KernelLasso",
    "KernelElasticNet",
    "KernelLars",

]


class BaseKernelModel(MLModule):
    def __init__(self,
                 alpha: Union[float, List[float], Tuple[float], torch.Tensor] = 1.0,
                 kernel: Union[str, Callable, nn.Module, MLModule] = "linear",
                 gamma: Union[float, str] = None,
                 degree: float = 3,
                 coef0: float = 1,
                 kernel_params: dict = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.alpha = alpha
        # Separate kernel args from model args
        self.kernel_config = {
            "gamma": gamma,
            "degree": degree,
            "coef0": coef0,
            **(kernel_params if kernel_params else {}),
            **kwargs
        }
        self.kernel_name = kernel
        self.kernel = None # Instantiated in _init_module_
        self.gamma = gamma
        self.device = device
        self.dtype = dtype
        
        self.weights = None
        self.train_data = None
        self.in_features = None
        self.out_features = None
        self.intercept_ = None

    @property
    def dual_coef_(self):
        return self.weights

    @property
    def X_fit_(self):
        return self.train_data

    @property
    def n_features_in_(self):
        return self.in_features

    def _init_module_(self, X: torch.Tensor, y: torch.Tensor):
        in_features = X.size(-1)
        out_features = y.size(-1) if y.ndim > 1 else 1

        # Gamma handling
        gamma_val = self.gamma
        if isinstance(self.gamma, str):
            if self.gamma.lower() == "scale":
                gamma_val = 1 / (in_features * X.var()) if X.numel() > 1 and X.var() > 0 else 1.0
            elif self.gamma.lower() == "auto":
                gamma_val = 1 / in_features
        elif isinstance(self.gamma, (float, int)):
             gamma_val = float(self.gamma)
             
        self.in_features = in_features
        self.out_features = out_features

        # Update kernel_config with computed gamma and dimensions
        self.kernel_config.update({
            "gamma": gamma_val,
            "num_features": in_features,
            "num_classes": out_features,
            "device": self.device,
            "dtype": self.dtype
        })

        if isinstance(self.kernel_name, str):
            kernel_class = get_kernel_class(self.kernel_name)
            if kernel_class:
                self.kernel = kernel_class(**self.kernel_config)
            else:
                available_kernels = KernelRegistry.list_kernels()
                raise ValueError(f"Unknown kernel type: '{self.kernel_name}'.\nAvailable kernels: {available_kernels}")
        elif isinstance(self.kernel_name, (nn.Module, MLModule)):
             self.kernel = self.kernel_name
        
        if self.kernel is not None:
             self.kernel.to(self.device)
             
        return self

    def forward(self, X: torch.Tensor, y: torch.Tensor = None):
        if self.in_features is None:
            self._init_module_(X, y)
            self.fit(X, y)
        else:
            return self._predict(X)

    def fit(self, data_or_X, y=None, **kwargs):
        raise NotImplementedError

    def _predict(self, X: torch.Tensor):
        if not isinstance(X, torch.Tensor):
             X = torch.tensor(X, device=self.device, dtype=self.dtype)
        
        # K(X, X_fit)
        K_trans = self.kernel(X, self.train_data)
        
        # Prediction = K * dual_coef + intercept
        
        if self.weights.ndim == 1:
             pred = K_trans @ self.weights
        else:
             pred = K_trans @ self.weights
             
        if self.intercept_ is not None:
             pred += self.intercept_

        return pred


class KernelRidge(BaseKernelModel):
    """
    Kernel ridge regression.
    """
    def __init__(self, 
                 alpha: Union[float, List[float], Tuple[float], torch.Tensor] = 1.0,
                 kernel: Union[str, Callable] = "linear",
                 gamma: Union[float, str] = None,
                 degree: float = 3,
                 coef0: float = 1,
                 kernel_params: dict = None,
                 # Ridge specific args
                 fit_intercept: bool = True, # KRR typically centers kernel, but handled in fit
                 max_iter: int = None,
                 tol: float = 1e-3,
                 solver: str = "auto",
                 random_state: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(alpha=alpha, kernel=kernel, gamma=gamma, degree=degree, 
                         coef0=coef0, kernel_params=kernel_params, 
                         device=device, dtype=dtype, *args, **kwargs)
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.random_state = random_state

    def fit(self, data_or_X, y=None, **kwargs):
        if y is None and isinstance(data_or_X, (list, tuple)):
            X, y = data_or_X
        else:
            X = data_or_X

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, device=self.device, dtype=self.dtype)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, device=self.device, dtype=self.dtype)
        
        if y.ndim == 1:
            y = y.unsqueeze(-1)

        self._init_module_(X, y)
        self.train_data = X if kwargs.get('copy_X', True) else X

        # Handle intercept by centering y
        if self.fit_intercept:
             y_mean = y.mean(dim=0, keepdim=True)
             y_train = y - y_mean
             self.intercept_ = y_mean.squeeze(0) # (n_targets,)
        else:
             y_train = y
             self.intercept_ = torch.zeros(y.shape[1], device=self.device, dtype=self.dtype)

        # Kernel Matrix K
        K = self.kernel(X) # (N, N)

        N = X.size(0)
        
        # Process alpha
        if isinstance(self.alpha, (float, int)):
             alpha_val = float(self.alpha)
             reg = alpha_val * torch.eye(N, device=self.device, dtype=self.dtype)
        elif isinstance(self.alpha, torch.Tensor):
             if self.alpha.numel() == 1:
                  reg = float(self.alpha) * torch.eye(N, device=self.device, dtype=self.dtype)
             else:
                  # Per-target alpha? KRR usually (K + alpha I)^-1. 
                  # For now simplified handling
                  reg = float(self.alpha.mean()) * torch.eye(N, device=self.device, dtype=self.dtype)

        alpha_val = float(self.alpha) if isinstance(self.alpha, (float, int)) else 1.0
        
        # Ridge solution: weights = (K + alpha I)^-1 y
        reg_K = K + alpha_val * torch.eye(N, device=self.device, dtype=self.dtype)
        
        try:
             self.weights = torch.linalg.solve(reg_K, y_train)
        except torch.linalg.LinAlgError:
             warnings.warn("Singular matrix in KRR, trying lstsq")
             self.weights = torch.linalg.lstsq(reg_K, y_train).solution
             
        return self


class KernelLasso(BaseKernelModel):
    def __init__(self, 
                 alpha: Union[float, List[float], Tuple[float], torch.Tensor] = 1.0,
                 kernel: Union[str, Callable] = "linear",
                 gamma: Union[float, str] = None,
                 degree: float = 3,
                 coef0: float = 1,
                 kernel_params: dict = None,
                 # Lasso specific
                 fit_intercept: bool = True,
                 precompute: bool = False,
                 copy_X: bool = True,
                 max_iter: int = 1000,
                 tol: float = 1e-4,
                 warm_start: bool = False,
                 positive: bool = False,
                 random_state: int = None,
                 selection: str = 'cyclic',
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(alpha=alpha, kernel=kernel, gamma=gamma, degree=degree, 
                         coef0=coef0, kernel_params=kernel_params, 
                         device=device, dtype=dtype, *args, **kwargs)
        self.fit_intercept = fit_intercept
        self.precompute = precompute
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.positive = positive
        self.random_state = random_state
        self.selection = selection

    def fit(self, data_or_X, y=None, **kwargs):
        if y is None and isinstance(data_or_X, (list, tuple)):
            X, y = data_or_X
        else:
            X = data_or_X

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, device=self.device, dtype=self.dtype)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, device=self.device, dtype=self.dtype)
            
        self._init_module_(X, y)
        self.train_data = X

        # Kernel Matrix K becomes our features
        K = self.kernel(X)
        
        # Delegate to Linear Lasso
        lasso = Lasso(alpha=self.alpha, 
                      fit_intercept=self.fit_intercept, 
                      precompute=self.precompute,
                      copy_X=self.copy_X,
                      max_iter=self.max_iter,
                      tol=self.tol,
                      warm_start=self.warm_start,
                      positive=self.positive,
                      random_state=self.random_state,
                      selection=self.selection,
                      device=self.device, 
                      dtype=self.dtype)
        
        lasso.fit(K, y)
        
        if lasso.weight.ndim == 2:
             self.weights = lasso.weight.T
        else:
             self.weights = lasso.weight
        
        # Extract intercept
        if hasattr(lasso, 'bias') and lasso.bias is not None:
             self.intercept_ = lasso.bias
        elif hasattr(lasso, 'intercept_'): # Check other_decomposition naming convention if applicable
             self.intercept_ = lasso.intercept_
        else:
             self.intercept_ = None
             
        return self

class KernelElasticNet(BaseKernelModel):
    def __init__(self, 
                 alpha: Union[float, List[float], Tuple[float], torch.Tensor] = 1.0,
                 l1_ratio: float = 0.5,
                 kernel: Union[str, Callable] = "linear",
                 gamma: Union[float, str] = None,
                 degree: float = 3,
                 coef0: float = 1,
                 kernel_params: dict = None,
                 # ElasticNet specific
                 fit_intercept: bool = True,
                 precompute: bool = False,
                 max_iter: int = 1000,
                 copy_X: bool = True,
                 tol: float = 1e-4,
                 warm_start: bool = False,
                 positive: bool = False,
                 random_state: int = None,
                 selection: str = 'cyclic',
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
         super().__init__(alpha=alpha, kernel=kernel, gamma=gamma, degree=degree, 
                          coef0=coef0, kernel_params=kernel_params, 
                          device=device, dtype=dtype, *args, **kwargs)
         self.l1_ratio = l1_ratio
         self.fit_intercept = fit_intercept
         self.precompute = precompute
         self.max_iter = max_iter
         self.copy_X = copy_X
         self.tol = tol
         self.warm_start = warm_start
         self.positive = positive
         self.random_state = random_state
         self.selection = selection

    def fit(self, data_or_X, y=None, **kwargs):
        if y is None and isinstance(data_or_X, (list, tuple)):
            X, y = data_or_X
        else:
            X = data_or_X

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, device=self.device, dtype=self.dtype)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, device=self.device, dtype=self.dtype)
            
        self._init_module_(X, y)
        self.train_data = X
        
        K = self.kernel(X)
        
        en = ElasticNet(alpha=self.alpha, 
                        l1_ratio=self.l1_ratio, 
                        fit_intercept=self.fit_intercept, 
                        precompute=self.precompute,
                        max_iter=self.max_iter,
                        copy_X=self.copy_X,
                        tol=self.tol,
                        warm_start=self.warm_start,
                        positive=self.positive,
                        random_state=self.random_state,
                        selection=self.selection,
                        device=self.device, 
                        dtype=self.dtype)
        en.fit(K, y)
        
        if en.weight.ndim == 2:
             self.weights = en.weight.T
        else:
             self.weights = en.weight

        # Extract intercept
        if hasattr(en, 'bias') and en.bias is not None:
             self.intercept_ = en.bias
        elif hasattr(en, 'intercept_'):
             self.intercept_ = en.intercept_
        else:
             self.intercept_ = None
             
        return self

class KernelLars(BaseKernelModel):
    def __init__(self, 
                 fit_intercept: bool = True,
                 verbose: Union[bool, int] = False,
                 precompute: Union[bool, str, List, Tuple, torch.Tensor] = "auto",
                 n_nonzero_coefs: int = 500,
                 eps: float = torch.finfo(torch.float32).eps,
                 fit_path: bool = True,
                 max_iter: int = None,
                 jitter: float = 1e-6,
                 random_state: int = None,
                 kernel: Union[str, Callable] = "linear",
                 gamma: Union[float, str] = None,
                 degree: float = 3,
                 coef0: float = 1,
                 kernel_params: dict = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        # Note constraint: Lars doesn't take alpha in init in same way as others usually, 
        # but KernelLars usually implies some regularization. 
        # BaseKernelModel expects alpha. We pass it, but Lars class uses n_nonzero_coefs or alpha only in fit?
        # Standard Lars doesn't have 'alpha' in init, but LassoLars does.
        # Check standard usage: KernelLars usually ~= LassoLars on Kernel.
        super().__init__(alpha=1.0, kernel=kernel, gamma=gamma, degree=degree, 
                         coef0=coef0, kernel_params=kernel_params, 
                         device=device, dtype=dtype, *args, **kwargs)
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.precompute = precompute
        self.n_nonzero_coefs = n_nonzero_coefs
        self.eps = eps
        self.fit_path = fit_path
        self.max_iter = max_iter
        self.jitter = jitter
        self.random_state = random_state

    def fit(self, data_or_X, y=None, **kwargs):
        if y is None and isinstance(data_or_X, (list, tuple)):
             X, y = data_or_X
        else:
             X = data_or_X
             
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, device=self.device, dtype=self.dtype)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, device=self.device, dtype=self.dtype)
            
        self._init_module_(X, y)
        self.train_data = X
        
        K = self.kernel(X)
        
        # Use Lars
        lars = Lars(fit_intercept=self.fit_intercept, 
                    verbose=self.verbose,
                    precompute=self.precompute,
                    n_nonzero_coefs=self.n_nonzero_coefs,
                    eps=max(self.eps, 1e-4), # Ensure stability
                    fit_path=self.fit_path,
                    max_iter=self.max_iter,
                    jitter=self.jitter,
                    random_state=self.random_state,
                    device=self.device, 
                    dtype=self.dtype)
                    
        lars.fit(K, y)
        
        if hasattr(lars, 'weight'):
            if lars.weight.ndim == 2:
                self.weights = lars.weight.T
            else:
                self.weights = lars.weight
        elif hasattr(lars, 'coef_'):
             self.weights = torch.tensor(lars.coef_.T, device=self.device, dtype=self.dtype)
        
        # Extract intercept
        # Lars in lars.py might use intercept_ or bias
        if hasattr(lars, 'bias') and lars.bias is not None:
             self.intercept_ = lars.bias
        elif hasattr(lars, 'intercept_'):
             self.intercept_ = lars.intercept_
        else:
             self.intercept_ = None
        
        return self

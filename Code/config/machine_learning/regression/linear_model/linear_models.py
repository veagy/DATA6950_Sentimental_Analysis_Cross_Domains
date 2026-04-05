"""Config templates for linear_models."""
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from .....config import ConfigTemplate

try:
    import torch
except ImportError:
    torch = None

"""Generated config for ElasticNet."""
class ElasticNetConfig(ConfigTemplate):
    model_name = "ElasticNet"
    model_path = "Code.models.machine_learning.regression.linear_model.linear_models"

    def __init__(self,
        immutable: bool = True,
        alpha: float = 1.0,
        l1_norm: float = 0.5,
        fit_intercept: bool = True,
        precompute: Union[bool, List[list], Tuple[tuple], torch.Tensor] = False,
        copy_X: bool = True,
        max_iter: int = None,
        tol: float = 1e-06,
        warm_start: bool = False,
        solver: str = 'auto',
        n_jobs: int = None,
        positive: bool = False,
        random_state: int = None,
        selection: str = 'cyclic',
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.alpha = alpha
        self.l1_norm = l1_norm
        self.fit_intercept = fit_intercept
        self.precompute = precompute
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.solver = solver
        self.n_jobs = n_jobs
        self.positive = positive
        self.random_state = random_state
        self.selection = selection
        self.device = device
        self.dtype = dtype


"""Generated config for ElasticNetCV."""
class ElasticNetCVConfig(ConfigTemplate):
    model_name = "ElasticNetCV"
    model_path = "Code.models.machine_learning.regression.linear_model.linear_models"

    def __init__(self,
        immutable: bool = True,
        l1_ratio: Union[float, List[float], Tuple[float], torch.Tensor] = 0.5,
        eps: float = 0.001,
        n_alphas: int = 100,
        alphas: Union[List[float], Tuple[float], torch.Tensor] = None,
        fit_intercept: bool = True,
        precompute: Union[bool, str, List[list], Tuple[tuple], torch.Tensor] = 'auto',
        max_iter: int = 1000,
        tol: float = 0.0001,
        cv: Union[str, int, Iterable, Callable, MLModule] = None,
        cv_config: dict = None,
        copy_X: bool = True,
        verbose: Union[bool, int] = 0,
        n_jobs: int = None,
        positive: bool = False,
        random_state: int = None,
        selection: str = 'cyclic',
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        store_cv_values: bool = False,
        alpha_per_target: bool = False,
        scoring: Union[str, Callable] = None,
        warm_start: bool = False,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.l1_ratio = l1_ratio
        self.eps = eps
        self.n_alphas = n_alphas
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.precompute = precompute
        self.max_iter = max_iter
        self.tol = tol
        self.cv = cv
        self.cv_config = cv_config
        self.copy_X = copy_X
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.positive = positive
        self.random_state = random_state
        self.selection = selection
        self.device = device
        self.dtype = dtype
        self.store_cv_values = store_cv_values
        self.alpha_per_target = alpha_per_target
        self.scoring = scoring
        self.warm_start = warm_start


"""Generated config for HuberRegressor."""
class HuberRegressorConfig(ConfigTemplate):
    model_name = "HuberRegressor"
    model_path = "Code.models.machine_learning.regression.linear_model.linear_models"

    def __init__(self,
        immutable: bool = True,
        epsilon: float = 1.35,
        alpha: float = 0.0001,
        fit_intercept: bool = True,
        copy_X: bool = True,
        max_iter: int = None,
        tol: float = 1e-06,
        n_jobs: int = None,
        positive: bool = False,
        warm_start: bool = False,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.epsilon = epsilon
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.n_jobs = n_jobs
        self.positive = positive
        self.warm_start = warm_start
        self.device = device
        self.dtype = dtype


"""Generated config for Lasso."""
class LassoConfig(ConfigTemplate):
    model_name = "Lasso"
    model_path = "Code.models.machine_learning.regression.linear_model.linear_models"

    def __init__(self,
        immutable: bool = True,
        alpha: Union[float, List[float], Tuple[float], torch.Tensor] = 1.0,
        fit_intercept: bool = True,
        precompute: Union[bool, List[list], Tuple[tuple], torch.Tensor] = False,
        copy_X: bool = True,
        max_iter: int = None,
        tol: float = 1e-06,
        warm_start: bool = False,
        solver: str = 'auto',
        n_jobs: int = None,
        positive: bool = False,
        random_state: int = None,
        selection: str = 'cyclic',
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.precompute = precompute
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.solver = solver
        self.n_jobs = n_jobs
        self.positive = positive
        self.random_state = random_state
        self.selection = selection
        self.device = device
        self.dtype = dtype


"""Generated config for LassoCV."""
class LassoCVConfig(ConfigTemplate):
    model_name = "LassoCV"
    model_path = "Code.models.machine_learning.regression.linear_model.linear_models"

    def __init__(self,
        immutable: bool = True,
        eps: float = 0.001,
        n_alphas: int = 100,
        alphas: Union[list, tuple, torch.Tensor] = None,
        fit_intercept: bool = True,
        precompute: Union[bool, List[list], Tuple[tuple], torch.Tensor] = False,
        copy_X: bool = True,
        max_iter: Union[int, List[int]] = 1000,
        tol: Union[float, List[float]] = 0.0001,
        warm_start: bool = False,
        cv: Union[str, int, Iterable, Callable, MLModule] = None,
        cv_config: dict = None,
        scoring: Union[str, Callable, Iterable] = None,
        verbose: Union[bool, int] = False,
        solver: Union[str, List[str]] = 'auto',
        n_jobs: int = None,
        positive: bool = False,
        random_state: int = None,
        selection: Union[str, List[str]] = 'cyclic',
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        store_cv_values: bool = False,
        alpha_per_target: bool = False,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.eps = eps
        self.n_alphas = n_alphas
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.precompute = precompute
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.cv = cv
        self.cv_config = cv_config
        self.scoring = scoring
        self.verbose = verbose
        self.solver = solver
        self.n_jobs = n_jobs
        self.positive = positive
        self.random_state = random_state
        self.selection = selection
        self.device = device
        self.dtype = dtype
        self.store_cv_values = store_cv_values
        self.alpha_per_target = alpha_per_target


"""Generated config for LinearRegression."""
class LinearRegressionConfig(ConfigTemplate):
    model_name = "LinearRegression"
    model_path = "Code.models.machine_learning.regression.linear_model.linear_models"

    def __init__(self,
        immutable: bool = True,
        fit_intercept: bool = True,
        copy_X: bool = True,
        tol: float = 1e-06,
        n_jobs: int = None,
        positive: bool = False,
        warm_start: bool = False,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.tol = tol
        self.n_jobs = n_jobs
        self.positive = positive
        self.warm_start = warm_start
        self.device = device
        self.dtype = dtype


"""Generated config for OrthogonalMatchingPursuit."""
class OrthogonalMatchingPursuitConfig(ConfigTemplate):
    model_name = "OrthogonalMatchingPursuit"
    model_path = "Code.models.machine_learning.regression.linear_model.linear_models"

    def __init__(self,
        immutable: bool = True,
        n_nonzero_coefs: int = None,
        tol: float = None,
        fit_intercept: bool = True,
        precompute: Union[str, bool] = 'auto',
        copy_X: bool = False,
        positive: bool = False,
        max_iter: int = None,
        n_jobs: int = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_nonzero_coefs = n_nonzero_coefs
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.precompute = precompute
        self.copy_X = copy_X
        self.positive = positive
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype


"""Generated config for OrthogonalMatchingPursuitCV."""
class OrthogonalMatchingPursuitCVConfig(ConfigTemplate):
    model_name = "OrthogonalMatchingPursuitCV"
    model_path = "Code.models.machine_learning.regression.linear_model.linear_models"

    def __init__(self,
        immutable: bool = True,
        n_nonzero_coefs: Union[int, List[int], Tuple[int], torch.Tensor] = None,
        tol: float = None,
        fit_intercept: bool = True,
        precompute: Union[str, bool] = 'auto',
        copy_X: bool = True,
        cv: Union[int, str, Callable, Iterable, MLModule] = None,
        cv_config: dict = None,
        store_cv_results: bool = False,
        verbose: Union[int, bool] = False,
        scoring: Union[str, Callable, nn.Module] = None,
        positive: bool = False,
        max_iter: int = None,
        n_jobs: int = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        alpha_per_target: bool = False,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_nonzero_coefs = n_nonzero_coefs
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.precompute = precompute
        self.copy_X = copy_X
        self.cv = cv
        self.cv_config = cv_config
        self.store_cv_results = store_cv_results
        self.verbose = verbose
        self.scoring = scoring
        self.positive = positive
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype
        self.alpha_per_target = alpha_per_target


"""Generated config for QuantileRegressor."""
class QuantileRegressorConfig(ConfigTemplate):
    model_name = "QuantileRegressor"
    model_path = "Code.models.machine_learning.regression.linear_model.linear_models"

    def __init__(self,
        immutable: bool = True,
        fit_intercept: bool = True,
        quantile: float = 0.5,
        alpha: float = 1.0,
        tol: float = None,
        copy_X: bool = False,
        positive: bool = False,
        solver: str = 'admm',
        solver_options: dict = None,
        max_iter: int = 1000,
        n_jobs: int = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.fit_intercept = fit_intercept
        self.quantile = quantile
        self.alpha = alpha
        self.tol = tol
        self.copy_X = copy_X
        self.positive = positive
        self.solver = solver
        self.solver_options = solver_options
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype


"""Generated config for RANSACRegressor."""
class RANSACRegressorConfig(ConfigTemplate):
    model_name = "RANSACRegressor"
    model_path = "Code.models.machine_learning.regression.linear_model.linear_models"

    def __init__(self,
        immutable: bool = True,
        fit_intercept: bool = True,
        estimator: Union[Callable, object] = None,
        min_samples: int = None,
        residual_threshold: float = None,
        is_data_valid: Callable = None,
        is_model_valid: Callable = None,
        tol: float = None,
        copy_X: bool = False,
        positive: bool = False,
        max_trials: int = None,
        max_skips: int = None,
        stop_n_inliers: float = float('inf'),
        stop_score: float = float('inf'),
        stop_probability: float = 0.99,
        loss: Union[str, Callable, nn.Module, MLModule] = 'absolute_error',
        random_state: int = None,
        n_jobs: int = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.fit_intercept = fit_intercept
        self.estimator = estimator
        self.min_samples = min_samples
        self.residual_threshold = residual_threshold
        self.is_data_valid = is_data_valid
        self.is_model_valid = is_model_valid
        self.tol = tol
        self.copy_X = copy_X
        self.positive = positive
        self.max_trials = max_trials
        self.max_skips = max_skips
        self.stop_n_inliers = stop_n_inliers
        self.stop_score = stop_score
        self.stop_probability = stop_probability
        self.loss = loss
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype


"""Generated config for Ridge."""
class RidgeConfig(ConfigTemplate):
    model_name = "Ridge"
    model_path = "Code.models.machine_learning.regression.linear_model.linear_models"

    def __init__(self,
        immutable: bool = True,
        alpha: Union[float, List[float], Tuple[float], torch.Tensor] = 1.0,
        fit_intercept: bool = True,
        copy_X: bool = True,
        max_iter: int = None,
        tol: float = 1e-06,
        solver: str = 'auto',
        n_jobs: int = None,
        positive: bool = False,
        random_state: int = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.n_jobs = n_jobs
        self.positive = positive
        self.random_state = random_state
        self.device = device
        self.dtype = dtype


"""Generated config for RidgeCV."""
class RidgeCVConfig(ConfigTemplate):
    model_name = "RidgeCV"
    model_path = "Code.models.machine_learning.regression.linear_model.linear_models"

    def __init__(self,
        immutable: bool = True,
        alphas: Union[List[float], Tuple[float], torch.Tensor] = (0.1, 1.0, 10.0),
        fit_intercept: bool = True,
        scoring: Union[str, Callable] = None,
        cv: Union[int, MLModule, Iterable] = None,
        cv_config: dict = None,
        gcv_mode: str = 'auto',
        store_cv_results: bool = False,
        alpha_per_target: bool = False,
        copy_X: bool = True,
        max_iter: int = None,
        tol: float = 1e-06,
        solver: str = 'auto',
        n_jobs: int = None,
        positive: bool = False,
        random_state: int = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.scoring = scoring
        self.cv = cv
        self.cv_config = cv_config
        self.gcv_mode = gcv_mode
        self.store_cv_results = store_cv_results
        self.alpha_per_target = alpha_per_target
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.n_jobs = n_jobs
        self.positive = positive
        self.random_state = random_state
        self.device = device
        self.dtype = dtype


"""Generated config for TheilSenRegressor."""
class TheilSenRegressorConfig(ConfigTemplate):
    model_name = "TheilSenRegressor"
    model_path = "Code.models.machine_learning.regression.linear_model.linear_models"

    def __init__(self,
        immutable: bool = True,
        fit_intercept: bool = True,
        max_subpopulation: int = 10000.0,
        n_subsamples: int = None,
        tol: float = None,
        copy_X: bool = False,
        positive: bool = False,
        max_iter: int = None,
        random_state: int = None,
        n_jobs: int = None,
        verbose: bool = False,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.fit_intercept = fit_intercept
        self.max_subpopulation = max_subpopulation
        self.n_subsamples = n_subsamples
        self.tol = tol
        self.copy_X = copy_X
        self.positive = positive
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.device = device
        self.dtype = dtype

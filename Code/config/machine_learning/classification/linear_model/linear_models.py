"""Config templates for linear_models."""
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from .....config import ConfigTemplate

try:
    import torch
except ImportError:
    torch = None

"""Generated config for ElasticNetClassifier."""
class ElasticNetClassifierConfig(ConfigTemplate):
    model_name = "ElasticNetClassifier"
    model_path = "Code.models.machine_learning.classification.linear_model.linear_models"

    def __init__(self,
        immutable: bool = True,
        alpha: Union[float, List[float], Tuple[float], torch.Tensor] = 1.0,
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
        class_weight: dict = None,
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
        self.class_weight = class_weight
        self.device = device
        self.dtype = dtype


"""Generated config for ElasticNetClassifierCV."""
class ElasticNetClassifierCVConfig(ConfigTemplate):
    model_name = "ElasticNetClassifierCV"
    model_path = "Code.models.machine_learning.classification.linear_model.linear_models"

    def __init__(self,
        immutable: bool = True,
        alphas: Union[float, List[float], Tuple[float], torch.Tensor] = None,
        l1_norms: Union[float, List[float], Tuple[float]] = None,
        fit_intercept: bool = True,
        precompute: Union[bool, List[list], Tuple[tuple], torch.Tensor] = False,
        copy_X: bool = True,
        max_iter: int = None,
        tol: float = 1e-06,
        warm_start: bool = False,
        positive: bool = False,
        selection: Union[str, List[str]] = 'cyclic',
        class_weights: Union[dict, List[dict], Tuple[dict]] = None,
        cv: Union[str, int, Callable, Iterable, nn.Module] = None,
        cv_config: dict = None,
        scoring: Union[str, Callable] = None,
        n_jobs: int = None,
        random_state: int = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.alphas = alphas
        self.l1_norms = l1_norms
        self.fit_intercept = fit_intercept
        self.precompute = precompute
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.positive = positive
        self.selection = selection
        self.class_weights = class_weights
        self.cv = cv
        self.cv_config = cv_config
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.device = device
        self.dtype = dtype


"""Generated config for ElasticNetLarsClassifier."""
class ElasticNetLarsClassifierConfig(ConfigTemplate):
    model_name = "ElasticNetLarsClassifier"
    model_path = "Code.models.machine_learning.classification.linear_model.linear_models"

    def __init__(self,
        immutable: bool = True,
        fit_intercept: bool = True,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        verbose: Union[bool, int] = False,
        precompute: Union[bool, str, List, Tuple, torch.Tensor] = 'auto',
        n_nonzero_coefs: int = 500,
        eps: float = torch.finfo(torch.float32).eps,
        fit_path: bool = True,
        max_iter: int = None,
        jitter: float = None,
        tol: float = None,
        copy_X: bool = False,
        positive: bool = False,
        random_state: int = None,
        n_jobs: int = None,
        class_weight: dict = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.verbose = verbose
        self.precompute = precompute
        self.n_nonzero_coefs = n_nonzero_coefs
        self.eps = eps
        self.fit_path = fit_path
        self.max_iter = max_iter
        self.jitter = jitter
        self.tol = tol
        self.copy_X = copy_X
        self.positive = positive
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.class_weight = class_weight
        self.device = device
        self.dtype = dtype


"""Generated config for ElasticNetLarsClassifierCV."""
class ElasticNetLarsClassifierCVConfig(ConfigTemplate):
    model_name = "ElasticNetLarsClassifierCV"
    model_path = "Code.models.machine_learning.classification.linear_model.linear_models"

    def __init__(self,
        immutable: bool = True,
        fit_intercept: bool = True,
        alphas: Union[float, List[float]] = None,
        l1_ratios: Union[float, List[float]] = None,
        verbose: Union[bool, int] = False,
        precompute: Union[bool, str, List, Tuple, torch.Tensor] = 'auto',
        n_nonzero_coefs_list: List[int] = None,
        eps: float = torch.finfo(torch.float32).eps,
        fit_path: bool = True,
        max_iter: int = None,
        jitter: float = None,
        tol: float = None,
        copy_X: bool = False,
        positive: bool = False,
        class_weights: Union[dict, List[dict], Tuple[dict]] = None,
        cv: Union[str, int, Callable, Iterable, nn.Module] = None,
        cv_config: dict = None,
        scoring: Union[str, Callable] = None,
        n_jobs: int = None,
        random_state: int = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.fit_intercept = fit_intercept
        self.alphas = alphas
        self.l1_ratios = l1_ratios
        self.verbose = verbose
        self.precompute = precompute
        self.n_nonzero_coefs_list = n_nonzero_coefs_list
        self.eps = eps
        self.fit_path = fit_path
        self.max_iter = max_iter
        self.jitter = jitter
        self.tol = tol
        self.copy_X = copy_X
        self.positive = positive
        self.class_weights = class_weights
        self.cv = cv
        self.cv_config = cv_config
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.device = device
        self.dtype = dtype


"""Generated config for LarsClassifier."""
class LarsClassifierConfig(ConfigTemplate):
    model_name = "LarsClassifier"
    model_path = "Code.models.machine_learning.classification.linear_model.linear_models"

    def __init__(self,
        immutable: bool = True,
        fit_intercept: bool = True,
        verbose: Union[bool, int] = False,
        precompute: Union[bool, str, List, Tuple, torch.Tensor] = 'auto',
        n_nonzero_coefs: int = 500,
        eps: float = torch.finfo(torch.float32).eps,
        fit_path: bool = True,
        max_iter: int = None,
        jitter: float = None,
        tol: float = None,
        copy_X: bool = True,
        positive: bool = False,
        random_state: int = None,
        n_jobs: int = None,
        class_weight: dict = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
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
        self.tol = tol
        self.copy_X = copy_X
        self.positive = positive
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.class_weight = class_weight
        self.device = device
        self.dtype = dtype


"""Generated config for LarsClassifierCV."""
class LarsClassifierCVConfig(ConfigTemplate):
    model_name = "LarsClassifierCV"
    model_path = "Code.models.machine_learning.classification.linear_model.linear_models"

    def __init__(self,
        immutable: bool = True,
        fit_intercept: bool = True,
        verbose: Union[bool, int] = False,
        precompute: Union[bool, str, List, Tuple, torch.Tensor] = 'auto',
        n_nonzero_coefs_list: List[int] = None,
        eps: float = torch.finfo(torch.float32).eps,
        fit_path: bool = True,
        max_iter: int = None,
        jitter: float = None,
        tol: float = None,
        copy_X: bool = True,
        positive: bool = False,
        class_weights: Union[dict, List[dict], Tuple[dict]] = None,
        cv: Union[str, int, Callable, Iterable, nn.Module] = None,
        cv_config: dict = None,
        scoring: Union[str, Callable] = None,
        n_jobs: int = None,
        random_state: int = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.precompute = precompute
        self.n_nonzero_coefs_list = n_nonzero_coefs_list
        self.eps = eps
        self.fit_path = fit_path
        self.max_iter = max_iter
        self.jitter = jitter
        self.tol = tol
        self.copy_X = copy_X
        self.positive = positive
        self.class_weights = class_weights
        self.cv = cv
        self.cv_config = cv_config
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.device = device
        self.dtype = dtype


"""Generated config for LassoClassifier."""
class LassoClassifierConfig(ConfigTemplate):
    model_name = "LassoClassifier"
    model_path = "Code.models.machine_learning.classification.linear_model.linear_models"

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
        class_weight: dict = None,
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
        self.class_weight = class_weight
        self.device = device
        self.dtype = dtype


"""Generated config for LassoClassifierCV."""
class LassoClassifierCVConfig(ConfigTemplate):
    model_name = "LassoClassifierCV"
    model_path = "Code.models.machine_learning.classification.linear_model.linear_models"

    def __init__(self,
        immutable: bool = True,
        alphas: Union[float, List[float], Tuple[float], torch.Tensor] = None,
        fit_intercept: bool = True,
        precompute: Union[bool, List[list], Tuple[tuple], torch.Tensor] = False,
        copy_X: bool = True,
        max_iter: int = None,
        tol: float = 1e-06,
        warm_start: bool = False,
        positive: bool = False,
        selection: Union[str, List[str]] = 'cyclic',
        class_weights: Union[dict, List[dict], Tuple[dict]] = None,
        cv: Union[str, int, Callable, Iterable, nn.Module] = None,
        cv_config: dict = None,
        scoring: Union[str, Callable] = None,
        n_jobs: int = None,
        random_state: int = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.precompute = precompute
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.positive = positive
        self.selection = selection
        self.class_weights = class_weights
        self.cv = cv
        self.cv_config = cv_config
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.device = device
        self.dtype = dtype


"""Generated config for LassoLarsClassifier."""
class LassoLarsClassifierConfig(ConfigTemplate):
    model_name = "LassoLarsClassifier"
    model_path = "Code.models.machine_learning.classification.linear_model.linear_models"

    def __init__(self,
        immutable: bool = True,
        fit_intercept: bool = True,
        alpha: float = 1.0,
        verbose: Union[bool, int] = False,
        precompute: Union[bool, str, List, Tuple, torch.Tensor] = 'auto',
        n_nonzero_coefs: int = 500,
        eps: float = torch.finfo(torch.float32).eps,
        fit_path: bool = True,
        max_iter: int = None,
        jitter: float = None,
        tol: float = None,
        copy_X: bool = False,
        positive: bool = False,
        random_state: int = None,
        n_jobs: int = None,
        class_weight: dict = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self.verbose = verbose
        self.precompute = precompute
        self.n_nonzero_coefs = n_nonzero_coefs
        self.eps = eps
        self.fit_path = fit_path
        self.max_iter = max_iter
        self.jitter = jitter
        self.tol = tol
        self.copy_X = copy_X
        self.positive = positive
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.class_weight = class_weight
        self.device = device
        self.dtype = dtype


"""Generated config for LassoLarsClassifierCV."""
class LassoLarsClassifierCVConfig(ConfigTemplate):
    model_name = "LassoLarsClassifierCV"
    model_path = "Code.models.machine_learning.classification.linear_model.linear_models"

    def __init__(self,
        immutable: bool = True,
        fit_intercept: bool = True,
        alphas: Union[float, List[float]] = None,
        verbose: Union[bool, int] = False,
        precompute: Union[bool, str, List, Tuple, torch.Tensor] = 'auto',
        n_nonzero_coefs_list: List[int] = None,
        eps: float = torch.finfo(torch.float32).eps,
        fit_path: bool = True,
        max_iter: int = None,
        jitter: float = None,
        tol: float = None,
        copy_X: bool = False,
        positive: bool = False,
        class_weights: Union[dict, List[dict], Tuple[dict]] = None,
        cv: Union[str, int, Callable, Iterable, nn.Module] = None,
        cv_config: dict = None,
        scoring: Union[str, Callable] = None,
        n_jobs: int = None,
        random_state: int = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.fit_intercept = fit_intercept
        self.alphas = alphas
        self.verbose = verbose
        self.precompute = precompute
        self.n_nonzero_coefs_list = n_nonzero_coefs_list
        self.eps = eps
        self.fit_path = fit_path
        self.max_iter = max_iter
        self.jitter = jitter
        self.tol = tol
        self.copy_X = copy_X
        self.positive = positive
        self.class_weights = class_weights
        self.cv = cv
        self.cv_config = cv_config
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.device = device
        self.dtype = dtype


"""Generated config for LogisticRegression."""
class LogisticRegressionConfig(ConfigTemplate):
    model_name = "LogisticRegression"
    model_path = "Code.models.machine_learning.classification.linear_model.linear_models"

    def __init__(self,
        immutable: bool = True,
        C: float = 1.0,
        l1_ratio: float = 0.0,
        dual: bool = False,
        tol: float = 0.0001,
        fit_intercept: bool = True,
        intercept_scaling: float = 1,
        class_weight: Union[dict, str] = None,
        random_state: int = None,
        solver: str = 'lbfgs',
        max_iter: int = 100,
        verbose: int = 0,
        warm_start: bool = False,
        n_jobs: int = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.C = C
        self.l1_ratio = l1_ratio
        self.dual = dual
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype


"""Generated config for LogisticRegressionCV."""
class LogisticRegressionCVConfig(ConfigTemplate):
    model_name = "LogisticRegressionCV"
    model_path = "Code.models.machine_learning.classification.linear_model.linear_models"

    def __init__(self,
        immutable: bool = True,
        Cs: Union[int, float, List[float], Tuple[float]] = 10,
        l1_ratios: Union[float, List[float], Tuple[float]] = None,
        dual: bool = False,
        tol: float = 0.0001,
        fit_intercept: bool = True,
        intercept_scalings: Union[float, List[float]] = 1,
        cv: Union[str, int, Callable, Iterable] = None,
        cv_config: Optional[dict] = None,
        scoring: Union[str, Callable] = None,
        class_weight: Union[dict, str] = None,
        random_state: int = None,
        refit: bool = True,
        solver: Union[str, List[str], Tuple[str]] = 'lbfgs',
        max_iter: int = 100,
        verbose: int = 0,
        warm_start: bool = False,
        n_jobs: int = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.Cs = Cs
        self.l1_ratios = l1_ratios
        self.dual = dual
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.intercept_scalings = intercept_scalings
        self.cv = cv
        self.cv_config = cv_config
        self.scoring = scoring
        self.class_weight = class_weight
        self.random_state = random_state
        self.refit = refit
        self.solver = solver
        self.max_iter = max_iter
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype


"""Generated config for RidgeClassifier."""
class RidgeClassifierConfig(ConfigTemplate):
    model_name = "RidgeClassifier"
    model_path = "Code.models.machine_learning.classification.linear_model.linear_models"

    def __init__(self,
        immutable: bool = True,
        alpha: Union[float, List[float], Tuple[float], torch.Tensor] = 1.0,
        fit_intercept: bool = True,
        copy_X: bool = True,
        max_iter: int = None,
        tol: float = 1e-06,
        solver: str = 'auto',
        class_weight: dict = None,
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
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.positive = positive
        self.random_state = random_state
        self.device = device
        self.dtype = dtype


"""Generated config for RidgeClassifierCV."""
class RidgeClassifierCVConfig(ConfigTemplate):
    model_name = "RidgeClassifierCV"
    model_path = "Code.models.machine_learning.classification.linear_model.linear_models"

    def __init__(self,
        immutable: bool = True,
        alphas: Union[int, float, List[float], Tuple[float], torch.Tensor] = 1.0,
        fit_intercept: bool = True,
        copy_X: bool = True,
        max_iter: int = None,
        tol: float = 1e-06,
        refit: bool = True,
        cv: Union[str, int, Callable, Iterable, nn.Module] = None,
        cv_config: dict = None,
        scoring: Union[str, Callable] = None,
        store_cv_values: bool = False,
        solver: Union[str, List[str], Tuple[str]] = 'auto',
        class_weights: Union[dict, List[dict], Tuple[dict]] = None,
        n_jobs: int = None,
        positive: bool = False,
        random_state: int = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.refit = refit
        self.cv = cv
        self.cv_config = cv_config
        self.scoring = scoring
        self.store_cv_values = store_cv_values
        self.solver = solver
        self.class_weights = class_weights
        self.n_jobs = n_jobs
        self.positive = positive
        self.random_state = random_state
        self.device = device
        self.dtype = dtype


"""Generated config for RidgeLarsClassifier."""
class RidgeLarsClassifierConfig(ConfigTemplate):
    model_name = "RidgeLarsClassifier"
    model_path = "Code.models.machine_learning.classification.linear_model.linear_models"

    def __init__(self,
        immutable: bool = True,
        fit_intercept: bool = True,
        alpha: float = 1.0,
        verbose: Union[bool, int] = False,
        precompute: Union[bool, str, List, Tuple, torch.Tensor] = 'auto',
        n_nonzero_coefs: int = 500,
        eps: float = torch.finfo(torch.float32).eps,
        fit_path: bool = True,
        max_iter: int = None,
        jitter: float = None,
        tol: float = None,
        copy_X: bool = True,
        positive: bool = False,
        random_state: int = None,
        n_jobs: int = None,
        class_weight: dict = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self.verbose = verbose
        self.precompute = precompute
        self.n_nonzero_coefs = n_nonzero_coefs
        self.eps = eps
        self.fit_path = fit_path
        self.max_iter = max_iter
        self.jitter = jitter
        self.tol = tol
        self.copy_X = copy_X
        self.positive = positive
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.class_weight = class_weight
        self.device = device
        self.dtype = dtype


"""Generated config for RidgeLarsClassifierCV."""
class RidgeLarsClassifierCVConfig(ConfigTemplate):
    model_name = "RidgeLarsClassifierCV"
    model_path = "Code.models.machine_learning.classification.linear_model.linear_models"

    def __init__(self,
        immutable: bool = True,
        fit_intercept: bool = True,
        alphas: Union[float, List[float]] = None,
        verbose: Union[bool, int] = False,
        precompute: Union[bool, str, List, Tuple, torch.Tensor] = 'auto',
        n_nonzero_coefs_list: List[int] = None,
        eps: float = torch.finfo(torch.float32).eps,
        fit_path: bool = True,
        max_iter: int = None,
        jitter: float = None,
        tol: float = None,
        copy_X: bool = True,
        positive: bool = False,
        class_weights: Union[dict, List[dict], Tuple[dict]] = None,
        cv: Union[str, int, Callable, Iterable, nn.Module] = None,
        cv_config: dict = None,
        scoring: Union[str, Callable] = None,
        n_jobs: int = None,
        random_state: int = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.fit_intercept = fit_intercept
        self.alphas = alphas
        self.verbose = verbose
        self.precompute = precompute
        self.n_nonzero_coefs_list = n_nonzero_coefs_list
        self.eps = eps
        self.fit_path = fit_path
        self.max_iter = max_iter
        self.jitter = jitter
        self.tol = tol
        self.copy_X = copy_X
        self.positive = positive
        self.class_weights = class_weights
        self.cv = cv
        self.cv_config = cv_config
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.device = device
        self.dtype = dtype

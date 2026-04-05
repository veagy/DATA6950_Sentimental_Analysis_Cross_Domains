"""Config templates for search_cv."""
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from ....config import ConfigTemplate

try:
    import torch
except ImportError:
    torch = None

"""Generated config for BaseSearchCV."""
class BaseSearchCVConfig(ConfigTemplate):
    model_name = "BaseSearchCV"
    model_path = "Code.models.machine_learning.cross_validation.search_cv"

    def __init__(self,
        immutable: bool = True,
        estimator: MLModule = None,
        scoring: Union[str, Callable] = None,
        n_jobs: int = None,
        refit: bool = True,
        cv: Union[str, int, Callable, Iterable, MLModule] = None,
        cv_config: dict = None,
        verbose: Union[int, bool] = 0,
        pre_dispatch: str = '2*n_jobs',
        error_score: Union[int, float] = float('nan'),
        return_train_score: bool = False,
        store_cv_values: bool = False,
        return_estimators: bool = False,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.cv = cv
        self.cv_config = cv_config
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.return_train_score = return_train_score
        self.store_cv_values = store_cv_values
        self.return_estimators = return_estimators
        self.device = device
        self.dtype = dtype


"""Generated config for GridSearchCV."""
class GridSearchCVConfig(ConfigTemplate):
    model_name = "GridSearchCV"
    model_path = "Code.models.machine_learning.cross_validation.search_cv"

    def __init__(self,
        immutable: bool = True,
        estimator = None,
        param_grid = None,
        scoring = None,
        n_jobs = None,
        refit = True,
        cv = None,
        cv_config = None,
        verbose = 0,
        pre_dispatch = '2*n_jobs',
        error_score = float('nan'),
        return_train_score = False,
        store_cv_values = False,
        return_estimators = False,
        device = 'cpu',
        dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.cv = cv
        self.cv_config = cv_config
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.return_train_score = return_train_score
        self.store_cv_values = store_cv_values
        self.return_estimators = return_estimators
        self.device = device
        self.dtype = dtype


"""Generated config for ParameterGrid."""
class ParameterGridConfig(ConfigTemplate):
    model_name = "ParameterGrid"
    model_path = "Code.models.machine_learning.cross_validation.search_cv"

    def __init__(self,
        immutable: bool = True,
        param_grid: Union[Dict[str, List[Any]], List[Dict[str, List[Any]]]] = None,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.param_grid = param_grid


"""Generated config for ParameterSampler."""
class ParameterSamplerConfig(ConfigTemplate):
    model_name = "ParameterSampler"
    model_path = "Code.models.machine_learning.cross_validation.search_cv"

    def __init__(self,
        immutable: bool = True,
        param_distributions: Dict[str, Union[List[Any], Any]] = None,
        n_iter: int = None,
        random_state: int = None,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state


"""Generated config for RandomizedSearchCV."""
class RandomizedSearchCVConfig(ConfigTemplate):
    model_name = "RandomizedSearchCV"
    model_path = "Code.models.machine_learning.cross_validation.search_cv"

    def __init__(self,
        immutable: bool = True,
        estimator = None,
        param_distributions = None,
        n_iter = 10,
        scoring = None,
        n_jobs = None,
        refit = True,
        cv = None,
        cv_config = None,
        verbose = 0,
        pre_dispatch = '2*n_jobs',
        random_state = None,
        error_score = float('nan'),
        return_train_score = False,
        store_cv_values = False,
        return_estimators = False,
        device = 'cpu',
        dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.cv = cv
        self.cv_config = cv_config
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.random_state = random_state
        self.error_score = error_score
        self.return_train_score = return_train_score
        self.store_cv_values = store_cv_values
        self.return_estimators = return_estimators
        self.device = device
        self.dtype = dtype

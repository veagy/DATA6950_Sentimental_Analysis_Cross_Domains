"""Config templates for imputation."""
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from .....config import ConfigTemplate

try:
    import torch
except ImportError:
    torch = None

"""Generated config for IterativeImputer."""
class IterativeImputerConfig(ConfigTemplate):
    model_name = "IterativeImputer"
    model_path = "Code.models.machine_learning.preprocessing.imputation.imputatio"

    def __init__(self,
        immutable: bool = True,
        estimator: MLModule = None,
        missing_values: Union[int, float, str, torch.nan] = torch.nan,
        sample_posterior: bool = False,
        max_iter: int = 10,
        tol: float = 0.001,
        n_nearest_features: int = None,
        initial_strategy: Literal['mean', 'median', 'most_frequent', 'constant'] = 'mean',
        fill_value: Union[int, float, torch.Tensor] = None,
        imputation_order: Literal['ascending', 'descending', 'roman', 'arabic', 'random'] = 'ascending',
        skip_complete: bool = False,
        min_value: Union[float, list, tuple, torch.Tensor] = -torch.inf,
        max_value: Union[float, list, tuple, torch.Tensor] = torch.inf,
        verbose: int = 0,
        random_state: Union[int, torch.Generator] = None,
        add_indicator: bool = False,
        keep_empty_features: bool = False,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.missing_values = missing_values
        self.sample_posterior = sample_posterior
        self.max_iter = max_iter
        self.tol = tol
        self.n_nearest_features = n_nearest_features
        self.initial_strategy = initial_strategy
        self.fill_value = fill_value
        self.imputation_order = imputation_order
        self.skip_complete = skip_complete
        self.min_value = min_value
        self.max_value = max_value
        self.verbose = verbose
        self.random_state = random_state
        self.add_indicator = add_indicator
        self.keep_empty_features = keep_empty_features
        self.device = device
        self.dtype = dtype


"""Generated config for KNNImputer."""
class KNNImputerConfig(ConfigTemplate):
    model_name = "KNNImputer"
    model_path = "Code.models.machine_learning.preprocessing.imputation.imputatio"

    def __init__(self,
        immutable: bool = True,
        missing_values: Union[int, float, str, torch.nan] = torch.nan,
        n_neighbors: int = 5,
        weights: Union[Literal['uniform', 'distance'], Callable] = 'uniform',
        metric: Union[Literal['nan_euclidean'], str, Callable, nn.Module] = 'nan_euclidean',
        metric_params: dict = None,
        copy: bool = True,
        add_indicator: bool = False,
        keep_empty_features: bool = False,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.missing_values = missing_values
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.metric_params = metric_params
        self.copy = copy
        self.add_indicator = add_indicator
        self.keep_empty_features = keep_empty_features
        self.device = device
        self.dtype = dtype


"""Generated config for MissingIndicator."""
class MissingIndicatorConfig(ConfigTemplate):
    model_name = "MissingIndicator"
    model_path = "Code.models.machine_learning.preprocessing.imputation.imputatio"

    def __init__(self,
        immutable: bool = True,
        missing_values: Union[int, float, str, torch.nan] = torch.nan,
        features: Literal['missing-only', 'all'] = 'missing-only',
        sparse: Union[Literal['auto'], bool] = 'auto',
        error_on_new: bool = True,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.missing_values = missing_values
        self.features = features
        self.sparse = sparse
        self.error_on_new = error_on_new
        self.device = device
        self.dtype = dtype


"""Generated config for SimpleImputer."""
class SimpleImputerConfig(ConfigTemplate):
    model_name = "SimpleImputer"
    model_path = "Code.models.machine_learning.preprocessing.imputation.imputatio"

    def __init__(self,
        immutable: bool = True,
        missing_values: Union[int, float, str, torch.nan] = torch.nan,
        fill_value: Union[int, float, torch.Tensor] = None,
        copy: bool = True,
        add_indicator: bool = False,
        keep_empty_features: bool = False,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.missing_values = missing_values
        self.fill_value = fill_value
        self.copy = copy
        self.add_indicator = add_indicator
        self.keep_empty_features = keep_empty_features
        self.device = device
        self.dtype = dtype


"""Generated config for _RidgeEstimator."""
class _RidgeEstimatorConfig(ConfigTemplate):
    model_name = "_RidgeEstimator"
    model_path = "Code.models.machine_learning.preprocessing.imputation.imputatio"

    def __init__(self,
        immutable: bool = True,
        alpha: float = 1.0,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.alpha = alpha
        self.device = device
        self.dtype = dtype

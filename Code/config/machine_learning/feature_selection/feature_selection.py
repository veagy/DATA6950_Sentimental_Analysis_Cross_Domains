"""Config templates for feature_selection."""
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from ....config import ConfigTemplate

try:
    import torch
except ImportError:
    torch = None

"""Generated config for GenericUnivariateSelect."""
class GenericUnivariateSelectConfig(ConfigTemplate):
    model_name = "GenericUnivariateSelect"
    model_path = "Code.models.machine_learning.feature_selection.feature_selectio"

    def __init__(self,
        immutable: bool = True,
        score_func: Union[Callable, nn.Module] = None,
        mode: Literal['percentile', 'k_best', 'fpr', 'fdr', 'fwe'] = 'percentile',
        param: Union[int, float, Literal['all']] = 1e-05,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.score_func = score_func
        self.mode = mode
        self.param = param
        self.device = device
        self.dtype = dtype


"""Generated config for RFE."""
class RFEConfig(ConfigTemplate):
    model_name = "RFE"
    model_path = "Code.models.machine_learning.feature_selection.feature_selectio"

    def __init__(self,
        immutable: bool = True,
        estimator: MLModule = None,
        n_features_to_select: Union[int, float] = None,
        step: Union[int, float] = 1,
        verbose: int = 0,
        importance_getter: Union[Literal['auto'], Callable, nn.Module] = 'auto',
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.verbose = verbose
        self.importance_getter = importance_getter
        self.device = device
        self.dtype = dtype


"""Generated config for RFECV."""
class RFECVConfig(ConfigTemplate):
    model_name = "RFECV"
    model_path = "Code.models.machine_learning.feature_selection.feature_selectio"

    def __init__(self,
        immutable: bool = True,
        estimator: MLModule = None,
        n_features_to_select: Union[int, float] = None,
        min_features_to_select: int = 1,
        cv: Union[str, int, Callable, MLModule] = None,
        cv_config: dict = None,
        scoring: Union[str, Callable, nn.Module] = None,
        step: Union[int, float] = 1,
        verbose: int = 0,
        importance_getter: Union[Literal['auto'], Callable, nn.Module] = 'auto',
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.min_features_to_select = min_features_to_select
        self.cv = cv
        self.cv_config = cv_config
        self.scoring = scoring
        self.step = step
        self.verbose = verbose
        self.importance_getter = importance_getter
        self.device = device
        self.dtype = dtype


"""Generated config for SelectFdr."""
class SelectFdrConfig(ConfigTemplate):
    model_name = "SelectFdr"
    model_path = "Code.models.machine_learning.feature_selection.feature_selectio"

    def __init__(self,
        immutable: bool = True,
        score_func: Union[Callable, nn.Module] = None,
        alpha: float = 0.05,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.score_func = score_func
        self.alpha = alpha
        self.device = device
        self.dtype = dtype


"""Generated config for SelectFpr."""
class SelectFprConfig(ConfigTemplate):
    model_name = "SelectFpr"
    model_path = "Code.models.machine_learning.feature_selection.feature_selectio"

    def __init__(self,
        immutable: bool = True,
        score_func: Union[Callable, nn.Module] = None,
        alpha: float = 0.05,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.score_func = score_func
        self.alpha = alpha
        self.device = device
        self.dtype = dtype


"""Generated config for SelectFromModel."""
class SelectFromModelConfig(ConfigTemplate):
    model_name = "SelectFromModel"
    model_path = "Code.models.machine_learning.feature_selection.feature_selectio"

    def __init__(self,
        immutable: bool = True,
        estimator: MLModule = None,
        threshold: Union[str, float] = None,
        prefit: bool = False,
        norm_order: Union[int, float] = 1,
        max_features: Union[int, Callable] = None,
        importance_getter: Union[Literal['auto'], Callable, nn.Module] = 'auto',
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.threshold = threshold
        self.prefit = prefit
        self.norm_order = norm_order
        self.max_features = max_features
        self.importance_getter = importance_getter
        self.device = device
        self.dtype = dtype


"""Generated config for SelectFwe."""
class SelectFweConfig(ConfigTemplate):
    model_name = "SelectFwe"
    model_path = "Code.models.machine_learning.feature_selection.feature_selectio"

    def __init__(self,
        immutable: bool = True,
        score_func: Union[Callable, nn.Module] = None,
        alpha: float = 0.05,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.score_func = score_func
        self.alpha = alpha
        self.device = device
        self.dtype = dtype


"""Generated config for SelectKBest."""
class SelectKBestConfig(ConfigTemplate):
    model_name = "SelectKBest"
    model_path = "Code.models.machine_learning.feature_selection.feature_selectio"

    def __init__(self,
        immutable: bool = True,
        score_func: Union[Callable, nn.Module] = None,
        k: Union[int, Literal['all']] = 10,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.score_func = score_func
        self.k = k
        self.device = device
        self.dtype = dtype


"""Generated config for SelectPercentile."""
class SelectPercentileConfig(ConfigTemplate):
    model_name = "SelectPercentile"
    model_path = "Code.models.machine_learning.feature_selection.feature_selectio"

    def __init__(self,
        immutable: bool = True,
        score_func: Union[Callable, nn.Module] = None,
        percentile: int = 10,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.score_func = score_func
        self.percentile = percentile
        self.device = device
        self.dtype = dtype


"""Generated config for SequentialFeatureSelector."""
class SequentialFeatureSelectorConfig(ConfigTemplate):
    model_name = "SequentialFeatureSelector"
    model_path = "Code.models.machine_learning.feature_selection.feature_selectio"

    def __init__(self,
        immutable: bool = True,
        estimator: MLModule = None,
        n_features_to_select: Union[Literal['auto'], int, float] = 'auto',
        tol: float = None,
        direction: Literal['forward', 'backward'] = 'forward',
        scoring: Union[str, Callable, nn.Module] = None,
        cv: Union[int, str, MLModule, Callable] = None,
        cv_config: dict = None,
        n_jobs: int = None,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.tol = tol
        self.direction = direction
        self.scoring = scoring
        self.cv = cv
        self.cv_config = cv_config
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype


"""Generated config for VarianceThreshold."""
class VarianceThresholdConfig(ConfigTemplate):
    model_name = "VarianceThreshold"
    model_path = "Code.models.machine_learning.feature_selection.feature_selectio"

    def __init__(self,
        immutable: bool = True,
        threshold: float = 0.0,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.threshold = threshold
        self.device = device
        self.dtype = dtype

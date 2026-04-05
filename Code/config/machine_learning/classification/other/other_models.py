"""Config templates for other_models."""
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from .....config import ConfigTemplate

try:
    import torch
except ImportError:
    torch = None

"""Generated config for CalibratedClassifierCV."""
class CalibratedClassifierCVConfig(ConfigTemplate):
    model_name = "CalibratedClassifierCV"
    model_path = "Code.models.machine_learning.classification.other.other_models"

    def __init__(self,
        immutable: bool = True,
        estimator: MLClassifier = None,
        method: str = 'sigmoid',
        cv: Union[str, Callable, Iterable, MLClassifier, int] = None,
        cv_config: dict = None,
        n_jobs: int = None,
        ensemble: Union[str, bool] = 'auto',
        warm_start: bool = False,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.method = method
        self.cv = cv
        self.cv_config = cv_config
        self.n_jobs = n_jobs
        self.ensemble = ensemble
        self.warm_start = warm_start
        self.device = device
        self.dtype = dtype


"""Generated config for DummyClassifier."""
class DummyClassifierConfig(ConfigTemplate):
    model_name = "DummyClassifier"
    model_path = "Code.models.machine_learning.classification.other.other_models"

    def __init__(self,
        immutable: bool = True,
        strategy: str = 'prior',
        random_state: Union[int, None] = None,
        constant: Union[int, str, List[Union[int, str]], Tuple[Union[int, str]], torch.Tensor, None] = None,
        warm_start: bool = False,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.strategy = strategy
        self.random_state = random_state
        self.constant = constant
        self.warm_start = warm_start
        self.device = device
        self.dtype = dtype


"""Generated config for FixedThresholdClassifier."""
class FixedThresholdClassifierConfig(ConfigTemplate):
    model_name = "FixedThresholdClassifier"
    model_path = "Code.models.machine_learning.classification.other.other_models"

    def __init__(self,
        immutable: bool = True,
        estimator: MLClassifier = None,
        threshold: Union[str, float] = 'auto',
        pos_label: Union[int, float, bool, str] = None,
        response_method: str = 'auto',
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.threshold = threshold
        self.pos_label = pos_label
        self.response_method = response_method
        self.device = device
        self.dtype = dtype


"""Generated config for LabelPropagation."""
class LabelPropagationConfig(ConfigTemplate):
    model_name = "LabelPropagation"
    model_path = "Code.models.machine_learning.classification.other.other_models"

    def __init__(self,
        immutable: bool = True,
        kernel: Union[str, MLModule, Callable] = 'rbf',
        gamma: float = 20.0,
        n_neighbors: int = 7,
        max_iter: int = 1000,
        tol: float = 0.001,
        n_jobs: int = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.kernel = kernel
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.max_iter = max_iter
        self.tol = tol
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype


"""Generated config for LabelSpreading."""
class LabelSpreadingConfig(ConfigTemplate):
    model_name = "LabelSpreading"
    model_path = "Code.models.machine_learning.classification.other.other_models"

    def __init__(self,
        immutable: bool = True,
        kernel: Union[str, MLModule, Callable] = 'rbf',
        gamma: float = 20.0,
        n_neighbors: int = 7,
        alpha: float = 0.2,
        max_iter: int = 1000,
        tol: float = 0.001,
        n_jobs: int = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.kernel = kernel
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype


"""Generated config for SelfTrainingClassifier."""
class SelfTrainingClassifierConfig(ConfigTemplate):
    model_name = "SelfTrainingClassifier"
    model_path = "Code.models.machine_learning.classification.other.other_models"

    def __init__(self,
        immutable: bool = True,
        estimator: MLClassifier = None,
        threshold: float = 0.75,
        criterion: str = 'threshold',
        k_best: int = 10,
        max_iter: int = 10,
        verbose: bool = False,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.threshold = threshold
        self.criterion = criterion
        self.k_best = k_best
        self.max_iter = max_iter
        self.verbose = verbose
        self.device = device
        self.dtype = dtype


"""Generated config for TunedThresholdClassifierCV."""
class TunedThresholdClassifierCVConfig(ConfigTemplate):
    model_name = "TunedThresholdClassifierCV"
    model_path = "Code.models.machine_learning.classification.other.other_models"

    def __init__(self,
        immutable: bool = True,
        estimator: MLClassifier = None,
        scoring: Union[str, Callable, nn.Module] = 'balanced_accuracy',
        response_method: str = 'auto',
        thresholds: Union[int, float, List[float], Tuple[float], torch.Tensor] = 100,
        cv: Union[str, Callable, Iterable, MLModule] = None,
        cv_config: dict = None,
        refit: bool = True,
        n_jobs: int = None,
        random_state: int = None,
        store_cv_results: bool = False,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.scoring = scoring
        self.response_method = response_method
        self.thresholds = thresholds
        self.cv = cv
        self.cv_config = cv_config
        self.refit = refit
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.store_cv_results = store_cv_results
        self.device = device
        self.dtype = dtype


"""Generated config for _PlattCalibrator."""
class _PlattCalibratorConfig(ConfigTemplate):
    model_name = "_PlattCalibrator"
    model_path = "Code.models.machine_learning.classification.other.other_models"

    def __init__(self,
        immutable: bool = True,
        n_classes: int = None,
        device: str = None,
        dtype: torch.dtype = None,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_classes = n_classes
        self.device = device
        self.dtype = dtype


"""Generated config for _TemperatureScaler."""
class _TemperatureScalerConfig(ConfigTemplate):
    model_name = "_TemperatureScaler"
    model_path = "Code.models.machine_learning.classification.other.other_models"

    def __init__(self,
        immutable: bool = True,
        device: str = None,
        dtype: torch.dtype = None,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.device = device
        self.dtype = dtype

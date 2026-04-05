"""Config templates for knn."""
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from .....config import ConfigTemplate

try:
    import torch
except ImportError:
    torch = None

"""Generated config for KNeighborsClassifier."""
class KNeighborsClassifierConfig(ConfigTemplate):
    model_name = "KNeighborsClassifier"
    model_path = "Code.models.machine_learning.classification.knn.k"

    def __init__(self,
        immutable: bool = True,
        n_neighbors: int = 5,
        weights: Union[str, Callable] = 'uniform',
        algorithm: str = 'auto',
        leaf_size: int = 30,
        p: float = 2,
        metric: Union[str, Callable, object] = 'minkowski',
        metric_params: dict = None,
        class_weights: Union[str, dict] = None,
        n_jobs: int = None,
        warm_start: bool = False,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.class_weights = class_weights
        self.n_jobs = n_jobs
        self.warm_start = warm_start
        self.device = device
        self.dtype = dtype


"""Generated config for NearestCentroid."""
class NearestCentroidConfig(ConfigTemplate):
    model_name = "NearestCentroid"
    model_path = "Code.models.machine_learning.classification.knn.k"

    def __init__(self,
        immutable: bool = True,
        p: float = 2,
        metric: Union[str, Callable, object] = 'minkowski',
        shrink_threshold: float = None,
        priors: Union[str, List, Tuple, torch.Tensor] = 'uniform',
        metric_params: dict = None,
        class_weights: Union[str, dict] = None,
        n_jobs: int = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.p = p
        self.metric = metric
        self.shrink_threshold = shrink_threshold
        self.priors = priors
        self.metric_params = metric_params
        self.class_weights = class_weights
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype


"""Generated config for RadiusNeighborsClassifier."""
class RadiusNeighborsClassifierConfig(ConfigTemplate):
    model_name = "RadiusNeighborsClassifier"
    model_path = "Code.models.machine_learning.classification.knn.k"

    def __init__(self,
        immutable: bool = True,
        radius: float = 1.0,
        weights: Union[str, Callable] = 'uniform',
        algorithm: str = 'auto',
        leaf_size: int = 30,
        p: float = 2,
        metric: Union[str, Callable, object] = 'minkowski',
        metric_params: dict = None,
        class_weight: Union[str, dict] = None,
        n_jobs: int = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.radius = radius
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype

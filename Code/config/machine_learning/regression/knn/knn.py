"""Config templates for knn."""
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from .....config import ConfigTemplate

try:
    import torch
except ImportError:
    torch = None

"""Generated config for BallTree."""
class BallTreeConfig(ConfigTemplate):
    model_name = "BallTree"
    model_path = "Code.models.machine_learning.regression.knn.k"

    def __init__(self,
        immutable: bool = True,
        data: torch.Tensor = None,
        leaf_size: int = 40,
        metric: Callable = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        n_jobs: int = 1,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.data = data
        self.leaf_size = leaf_size
        self.metric = metric
        self.device = device
        self.dtype = dtype
        self.n_jobs = n_jobs


"""Generated config for KDTree."""
class KDTreeConfig(ConfigTemplate):
    model_name = "KDTree"
    model_path = "Code.models.machine_learning.regression.knn.k"

    def __init__(self,
        immutable: bool = True,
        data: torch.Tensor = None,
        leaf_size: int = 40,
        metric: Callable = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        n_jobs: int = 1,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.data = data
        self.leaf_size = leaf_size
        self.metric = metric
        self.device = device
        self.dtype = dtype
        self.n_jobs = n_jobs


"""Generated config for KNeighboursRegression."""
class KNeighboursRegressionConfig(ConfigTemplate):
    model_name = "KNeighboursRegression"
    model_path = "Code.models.machine_learning.regression.knn.k"

    def __init__(self,
        immutable: bool = True,
        n_neighbors: int = 5,
        weights: Union[str, Callable] = 'uniform',
        algorithm: str = 'auto',
        leaf_size: int = 30,
        p: float = 2,
        metric: Union[str, Callable, object] = 'minkowski',
        metric_params: dict = None,
        n_jobs: int = None,
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
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype


"""Generated config for NeighborhoodComponentsAnalysis."""
class NeighborhoodComponentsAnalysisConfig(ConfigTemplate):
    model_name = "NeighborhoodComponentsAnalysis"
    model_path = "Code.models.machine_learning.regression.knn.k"

    def __init__(self,
        immutable: bool = True,
        n_components: int = None,
        init: Union[Literal['auto', 'pca', 'lda', 'identity', 'random'], list, tuple, torch.Tensor] = 'auto',
        warm_start: bool = False,
        max_iter: int = 50,
        tol: float = 1e-05,
        callback: Callable = None,
        verbose: int = 0,
        random_state: Union[int, torch.Generator] = None,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_components = n_components
        self.init = init
        self.warm_start = warm_start
        self.max_iter = max_iter
        self.tol = tol
        self.callback = callback
        self.verbose = verbose
        self.random_state = random_state
        self.device = device
        self.dtype = dtype


"""Generated config for Node."""
class NodeConfig(ConfigTemplate):
    model_name = "Node"
    model_path = "Code.models.machine_learning.regression.knn.k"

    def __init__(self,
        immutable: bool = True,
        idxs = None,
        point = None,
        left = None,
        right = None,
        axis = None,
        split_val = None,
        radius = None,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.idxs = idxs
        self.point = point
        self.left = left
        self.right = right
        self.axis = axis
        self.split_val = split_val
        self.radius = radius


"""Generated config for RadiusNeighborsRegressor."""
class RadiusNeighborsRegressorConfig(ConfigTemplate):
    model_name = "RadiusNeighborsRegressor"
    model_path = "Code.models.machine_learning.regression.knn.k"

    def __init__(self,
        immutable: bool = True,
        radius: float = 10.0,
        weights: Union[str, Callable] = 'uniform',
        algorithm: str = 'auto',
        leaf_size: int = 30,
        p: float = 2,
        metric: Union[str, Callable, object] = 'minkowski',
        metric_params: dict = None,
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
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype

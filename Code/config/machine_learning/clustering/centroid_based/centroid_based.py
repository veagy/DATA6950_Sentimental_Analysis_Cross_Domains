"""Config templates for centroid_based."""
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from .....config import ConfigTemplate

try:
    import torch
except ImportError:
    torch = None

"""Generated config for BregmanKMeans."""
class BregmanKMeansConfig(ConfigTemplate):
    model_name = "BregmanKMeans"
    model_path = "Code.models.machine_learning.clustering.centroid_based.centroid_based"

    def __init__(self,
        immutable: bool = True,
        n_clusters: int = 8,
        divergence: Optional[str] = None,
        divergence_params: Dict[str, Any] = None,
        eps: float = 1e-09,
        metric: Union[str, Callable, 'nn.Module'] = 'euclidean',
        metric_params: Dict[str, Any] = None,
        init: Union[str, Callable, 'nn.Module', List, Tuple, torch.Tensor] = 'k-means++',
        n_init: Union[str, int] = 'auto',
        max_iter: int = 300,
        tol: float = 0.0001,
        verbose: bool = False,
        random_state: Optional[int] = None,
        copy_X: bool = True,
        warm_start: bool = False,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_clusters = n_clusters
        self.divergence = divergence
        self.divergence_params = divergence_params
        self.eps = eps
        self.metric = metric
        self.metric_params = metric_params
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.copy_X = copy_X
        self.warm_start = warm_start
        self.device = device
        self.dtype = dtype


"""Generated config for KMeansCluster."""
class KMeansClusterConfig(ConfigTemplate):
    model_name = "KMeansCluster"
    model_path = "Code.models.machine_learning.clustering.centroid_based.centroid_based"

    def __init__(self,
        immutable: bool = True,
        n_clusters: int = 8,
        init: Union[str, Callable, nn.Module, list, tuple, torch.Tensor] = 'k-means++',
        n_init: Union[str, int] = 'auto',
        max_iter: int = 300,
        tol: float = 0.0001,
        verbose: bool = False,
        random_state: int = None,
        copy_X: bool = True,
        metric: Union[str, Callable, nn.Module] = 'euclidean',
        metric_params: dict = None,
        algorithm: Union[str, Callable, nn.Module] = 'lloyd',
        n_yinyang_groups: int = 10,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.copy_X = copy_X
        self.metric = metric
        self.metric_params = metric_params
        self.algorithm = algorithm
        self.n_yinyang_groups = n_yinyang_groups
        self.device = device
        self.dtype = dtype

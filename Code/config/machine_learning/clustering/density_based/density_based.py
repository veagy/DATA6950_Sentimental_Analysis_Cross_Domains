"""Config templates for density_based."""
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from .....config import ConfigTemplate

try:
    import torch
except ImportError:
    torch = None

"""Generated config for DBSCAN."""
class DBSCANConfig(ConfigTemplate):
    model_name = "DBSCAN"
    model_path = "Code.models.machine_learning.clustering.density_based.density_based"

    def __init__(self,
        immutable: bool = True,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: Union[str, Callable, nn.Module] = 'euclidean',
        metric_params: dict = None,
        algorithm: Union[str, Callable, nn.Module] = 'auto',
        leaf_size: int = 30,
        p: float = None,
        n_jobs: int = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.metric_params = metric_params
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype


"""Generated config for HDBSCAN."""
class HDBSCANConfig(ConfigTemplate):
    model_name = "HDBSCAN"
    model_path = "Code.models.machine_learning.clustering.density_based.density_based"

    def __init__(self,
        immutable: bool = True,
        min_cluster_size: int = 5,
        min_samples: int = None,
        eps: float = 0.0,
        metric: Union[str, Callable, nn.Module] = 'euclidean',
        metric_params: dict = None,
        cluster_selection_epsilon: float = 0.0,
        max_cluster_size: int = None,
        alpha: float = 0.0,
        cluster_selection_method: Union[str, Callable, nn.Module] = 'eom',
        allow_single_cluster: bool = False,
        store_centers: str = None,
        copy: bool = True,
        algorithm: Union[str, Callable, nn.Module] = 'auto',
        leaf_size: int = 30,
        p: float = None,
        n_jobs: int = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.eps = eps
        self.metric = metric
        self.metric_params = metric_params
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.max_cluster_size = max_cluster_size
        self.alpha = alpha
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster
        self.store_centers = store_centers
        self.copy = copy
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype


"""Generated config for OPTICS."""
class OPTICSConfig(ConfigTemplate):
    model_name = "OPTICS"
    model_path = "Code.models.machine_learning.clustering.density_based.density_based"

    def __init__(self,
        immutable: bool = True,
        min_samples: int = None,
        max_eps: float = float('inf'),
        eps: float = 0.0,
        metric: Union[str, Callable, nn.Module] = 'euclidean',
        metric_params: dict = None,
        cluster_method: Union[str, Callable, nn.Module] = 'xi',
        xi: float = 0.05,
        predecessor_correction: bool = True,
        min_cluster_size: Union[int, float] = None,
        algorithm: Union[str, Callable, nn.Module] = 'auto',
        leaf_size: int = 30,
        memory: Union[str] = None,
        p: float = None,
        n_jobs: int = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.min_samples = min_samples
        self.max_eps = max_eps
        self.eps = eps
        self.metric = metric
        self.metric_params = metric_params
        self.cluster_method = cluster_method
        self.xi = xi
        self.predecessor_correction = predecessor_correction
        self.min_cluster_size = min_cluster_size
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.memory = memory
        self.p = p
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype

"""Config templates for search."""
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from .....config import ConfigTemplate

try:
    import torch
except ImportError:
    torch = None

"""Generated config for NearestNeighbors."""
class NearestNeighborsConfig(ConfigTemplate):
    model_name = "NearestNeighbors"
    model_path = "Code.models.machine_learning.preprocessing.search.search"

    def __init__(self,
        immutable: bool = True,
        n_neighbors: int = 5,
        radius: float = 1.0,
        algorithm: Union[Literal['ball_tree', 'kd_tree', 'brute', 'auto'], Callable, nn.Module] = 'auto',
        leaf_size: int = 30,
        metric: Union[str, Callable, nn.Module] = 'minkowski',
        p: float = 2,
        metric_params: dict = None,
        n_jobs: int = None,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype

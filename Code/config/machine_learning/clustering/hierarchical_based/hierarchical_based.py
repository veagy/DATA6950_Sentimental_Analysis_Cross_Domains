"""Config templates for hierarchical_based."""
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from .....config import ConfigTemplate

try:
    import torch
except ImportError:
    torch = None

"""Generated config for AgglomerativeClustering."""
class AgglomerativeClusteringConfig(ConfigTemplate):
    model_name = "AgglomerativeClustering"
    model_path = "Code.models.machine_learning.clustering.hierarchical_based.hierarchical_based"

    def __init__(self,
        immutable: bool = True,
        n_clusters: Union[int, None] = 2,
        metric: Union[str, Callable, nn.Module] = 'euclidean',
        metric_params: dict = None,
        memory: Union[str, Any] = None,
        connectivity: Union[list, tuple, torch.Tensor, Callable, nn.Module] = None,
        compute_full_tree: Union[str, bool] = 'auto',
        linkage: Union[str, Callable, nn.Module] = 'ward',
        distance_threshold: float = None,
        compute_distances: bool = False,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_clusters = n_clusters
        self.metric = metric
        self.metric_params = metric_params
        self.memory = memory
        self.connectivity = connectivity
        self.compute_full_tree = compute_full_tree
        self.linkage = linkage
        self.distance_threshold = distance_threshold
        self.compute_distances = compute_distances
        self.device = device
        self.dtype = dtype


"""Generated config for Birch."""
class BirchConfig(ConfigTemplate):
    model_name = "Birch"
    model_path = "Code.models.machine_learning.clustering.hierarchical_based.hierarchical_based"

    def __init__(self,
        immutable: bool = True,
        threshold: float = 0.5,
        branching_factor: int = 50,
        n_clusters: Union[int, 'MLCluster', None] = 3,
        compute_labels: bool = True,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters
        self.compute_labels = compute_labels
        self.device = device
        self.dtype = dtype


"""Generated config for FeatureAgglomeration."""
class FeatureAgglomerationConfig(ConfigTemplate):
    model_name = "FeatureAgglomeration"
    model_path = "Code.models.machine_learning.clustering.hierarchical_based.hierarchical_based"

    def __init__(self,
        immutable: bool = True,
        n_clusters: Union[int, None] = 2,
        metric: Union[str, Callable, nn.Module] = 'euclidean',
        metric_params: dict = None,
        memory: Union[str, Any] = None,
        connectivity: Union[list, tuple, torch.Tensor, Callable, nn.Module] = None,
        compute_full_tree: Union[str, bool] = 'auto',
        linkage: Union[str, Callable, nn.Module] = 'ward',
        pooling_func: Union[str, Callable, nn.Module] = torch.mean,
        distance_threshold: float = None,
        compute_distances: bool = False,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_clusters = n_clusters
        self.metric = metric
        self.metric_params = metric_params
        self.memory = memory
        self.connectivity = connectivity
        self.compute_full_tree = compute_full_tree
        self.linkage = linkage
        self.pooling_func = pooling_func
        self.distance_threshold = distance_threshold
        self.compute_distances = compute_distances
        self.device = device
        self.dtype = dtype


"""Generated config for _CFNode."""
class _CFNodeConfig(ConfigTemplate):
    model_name = "_CFNode"
    model_path = "Code.models.machine_learning.clustering.hierarchical_based.hierarchical_based"

    def __init__(self,
        immutable: bool = True,
        threshold: float = None,
        branching_factor: int = None,
        is_leaf: bool = None,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.is_leaf = is_leaf


"""Generated config for _CFSubcluster."""
class _CFSubclusterConfig(ConfigTemplate):
    model_name = "_CFSubcluster"
    model_path = "Code.models.machine_learning.clustering.hierarchical_based.hierarchical_based"

    def __init__(self,
        immutable: bool = True,
        linear_sum: torch.Tensor = None,
        squared_sum: float = None,
        count: int = None,
        child = None,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.linear_sum = linear_sum
        self.squared_sum = squared_sum
        self.count = count
        self.child = child

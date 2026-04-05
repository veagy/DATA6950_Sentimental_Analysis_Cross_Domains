"""Config templates for misc."""
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from .....config import ConfigTemplate

try:
    import torch
except ImportError:
    torch = None

"""Generated config for ColumnTransformer."""
class ColumnTransformerConfig(ConfigTemplate):
    model_name = "ColumnTransformer"
    model_path = "Code.models.machine_learning.transformer.misc.misc"

    def __init__(self,
        immutable: bool = True,
        transformers: Union[List[Union[list, tuple]], Tuple[List[Union[list, tuple]]], torch.Tensor, None] = None,
        name: str = None,
        transformer: Union[Literal['drop', 'passthrough'], MLModule] = None,
        columns: Union[str, int, slice, List[Union[str, int, bool]], Tuple[Union[str, int, bool]], torch.Tensor, Callable, nn.Module] = None,
        remainder: Union[Literal['drop', 'passthrough'], MLModule] = 'drop',
        sparse_threshold: float = 0.3,
        n_jobs: int = None,
        transformer_weights: dict = None,
        verbose: bool = False,
        verbose_feature_names_out: Union[bool, str, Callable[[str, str], str]] = True,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.transformers = transformers
        self.name = name
        self.transformer = transformer
        self.columns = columns
        self.remainder = remainder
        self.sparse_threshold = sparse_threshold
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self.verbose = verbose
        self.verbose_feature_names_out = verbose_feature_names_out
        self.device = device
        self.dtype = dtype


"""Generated config for FeatureUnion."""
class FeatureUnionConfig(ConfigTemplate):
    model_name = "FeatureUnion"
    model_path = "Code.models.machine_learning.transformer.misc.misc"

    def __init__(self,
        immutable: bool = True,
        transformers: Union[List[Union[list, tuple]], Tuple[List[Union[list, tuple]]], torch.Tensor, None] = None,
        n_jobs: int = None,
        transformer_weights: dict = None,
        verbose: bool = False,
        verbose_feature_names_out: Union[bool, str, Callable[[str, str], str]] = True,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.transformers = transformers
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self.verbose = verbose
        self.verbose_feature_names_out = verbose_feature_names_out
        self.device = device
        self.dtype = dtype


"""Generated config for KNeighborsTransformer."""
class KNeighborsTransformerConfig(ConfigTemplate):
    model_name = "KNeighborsTransformer"
    model_path = "Code.models.machine_learning.transformer.misc.misc"

    def __init__(self,
        immutable: bool = True,
        mode: Literal['distance', 'connectivity'] = 'distance',
        n_neighbors: int = 5,
        algorithm: Union[Literal['auto', 'ball_tree', 'kd_tree', 'brute'], Callable, nn.Module] = 'auto',
        leaf_size: int = 30,
        metric: Union[str, Callable, nn.Module] = 'minkowski',
        p: float = 2,
        metric_params: dict = None,
        n_jobs: int = None,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.mode = mode
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype


"""Generated config for RadiusNeighborsTransformer."""
class RadiusNeighborsTransformerConfig(ConfigTemplate):
    model_name = "RadiusNeighborsTransformer"
    model_path = "Code.models.machine_learning.transformer.misc.misc"

    def __init__(self,
        immutable: bool = True,
        mode: Literal['distance', 'connectivity'] = 'distance',
        radius: float = 1.0,
        algorithm: Union[Literal['auto', 'ball_tree', 'kd_tree', 'brute'], Callable, nn.Module] = 'auto',
        leaf_size: int = 30,
        metric: Union[str, Callable, nn.Module] = 'minkowski',
        p: float = 2,
        metric_params: dict = None,
        n_jobs: int = None,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.mode = mode
        self.radius = radius
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype


"""Generated config for RandomTreesEmbeddings."""
class RandomTreesEmbeddingsConfig(ConfigTemplate):
    model_name = "RandomTreesEmbeddings"
    model_path = "Code.models.machine_learning.transformer.misc.misc"

    def __init__(self,
        immutable: bool = True,
        n_estimators: int = 100,
        max_depth: int = 5,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_leaf_nodes: int = None,
        min_impurity_decrease: float = 0.0,
        sparse_output: bool = True,
        n_jobs: int = None,
        random_state: Union[int, torch.Generator] = None,
        verbose: int = 0,
        warm_start: bool = False,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.sparse_output = sparse_output
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.device = device
        self.dtype = dtype

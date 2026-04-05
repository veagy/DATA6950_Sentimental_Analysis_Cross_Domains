"""Config templates for trees."""
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from .....config import ConfigTemplate

try:
    import torch
except ImportError:
    torch = None

"""Generated config for DecisionTreeClassifier."""
class DecisionTreeClassifierConfig(ConfigTemplate):
    model_name = "DecisionTreeClassifier"
    model_path = "Code.models.machine_learning.classification.tree_models.trees"

    def __init__(self,
        immutable: bool = True,
        criterion: str = 'gini',
        splitter: str = 'best',
        max_depth: int = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Union[int, float, str] = None,
        random_state: int = None,
        max_leaf_nodes: int = None,
        min_impurity_decrease: float = 0.0,
        class_weight: Union[dict, str] = None,
        ccp_alpha: float = 0.0,
        monotonic_cst: Union[List[int], Tuple[int], torch.Tensor] = None,
        interaction_cst: Union[str, List[Any], Tuple[Any]] = None,
        warm_start: bool = False,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.monotonic_cst = monotonic_cst
        self.interaction_cst = interaction_cst
        self.warm_start = warm_start
        self.device = device
        self.dtype = dtype


"""Generated config for ExtraTreeClassifier."""
class ExtraTreeClassifierConfig(ConfigTemplate):
    model_name = "ExtraTreeClassifier"
    model_path = "Code.models.machine_learning.classification.tree_models.trees"

    def __init__(self,
        immutable: bool = True,
        criterion: str = 'gini',
        splitter: str = 'random',
        max_depth: int = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Union[int, float, str] = 'sqrt',
        random_state: int = None,
        max_leaf_nodes: int = None,
        min_impurity_decrease: float = 0.0,
        class_weight: Union[dict, str] = None,
        ccp_alpha: float = 0.0,
        monotonic_cst: Union[List[int], Tuple[int], torch.Tensor] = None,
        interaction_cst: Union[str, List[Any], Tuple[Any]] = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.monotonic_cst = monotonic_cst
        self.interaction_cst = interaction_cst
        self.device = device
        self.dtype = dtype

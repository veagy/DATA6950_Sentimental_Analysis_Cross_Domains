"""Config templates for normalizer."""
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from .....config import ConfigTemplate

try:
    import torch
except ImportError:
    torch = None

"""Generated config for Binarizer."""
class BinarizerConfig(ConfigTemplate):
    model_name = "Binarizer"
    model_path = "Code.models.machine_learning.preprocessing.normalizer.normalizer"

    def __init__(self,
        immutable: bool = True,
        threshold: float = 0.0,
        copy: bool = True,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.threshold = threshold
        self.copy = copy
        self.device = device
        self.dtype = dtype


"""Generated config for MaxAbsScaler."""
class MaxAbsScalerConfig(ConfigTemplate):
    model_name = "MaxAbsScaler"
    model_path = "Code.models.machine_learning.preprocessing.normalizer.normalizer"

    def __init__(self,
        immutable: bool = True,
        copy: bool = True,
        clip: bool = False,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.copy = copy
        self.clip = clip
        self.device = device
        self.dtype = dtype


"""Generated config for MinMaxScaler."""
class MinMaxScalerConfig(ConfigTemplate):
    model_name = "MinMaxScaler"
    model_path = "Code.models.machine_learning.preprocessing.normalizer.normalizer"

    def __init__(self,
        immutable: bool = True,
        feature_range: Union[Tuple[float, float], List[float], torch.Tensor] = (0, 1),
        copy: bool = True,
        clip: bool = False,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.feature_range = feature_range
        self.copy = copy
        self.clip = clip
        self.device = device
        self.dtype = dtype


"""Generated config for Normalizer."""
class NormalizerConfig(ConfigTemplate):
    model_name = "Normalizer"
    model_path = "Code.models.machine_learning.preprocessing.normalizer.normalizer"

    def __init__(self,
        immutable: bool = True,
        norm: Union[Literal['l1', 'l2', 'max'], float] = 'l2',
        copy: bool = True,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.norm = norm
        self.copy = copy
        self.device = device
        self.dtype = dtype


"""Generated config for PowerTransformer."""
class PowerTransformerConfig(ConfigTemplate):
    model_name = "PowerTransformer"
    model_path = "Code.models.machine_learning.preprocessing.normalizer.normalizer"

    def __init__(self,
        immutable: bool = True,
        method: str = 'yeo-johnson',
        standardize: bool = True,
        copy: bool = True,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.method = method
        self.standardize = standardize
        self.copy = copy
        self.device = device
        self.dtype = dtype


"""Generated config for QuantileTransformer."""
class QuantileTransformerConfig(ConfigTemplate):
    model_name = "QuantileTransformer"
    model_path = "Code.models.machine_learning.preprocessing.normalizer.normalizer"

    def __init__(self,
        immutable: bool = True,
        n_quantiles: int = 1000,
        output_distribution: str = 'uniform',
        subsample: int = 10000,
        random_state: Optional[Union[int, torch.Generator]] = None,
        copy: bool = True,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.subsample = subsample
        self.random_state = random_state
        self.copy = copy
        self.device = device
        self.dtype = dtype


"""Generated config for RobustScaler."""
class RobustScalerConfig(ConfigTemplate):
    model_name = "RobustScaler"
    model_path = "Code.models.machine_learning.preprocessing.normalizer.normalizer"

    def __init__(self,
        immutable: bool = True,
        with_centering: bool = True,
        with_scaling: bool = True,
        quantile_range: Tuple[float, float] = (25.0, 75.0),
        unit_variance: bool = False,
        copy: bool = True,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.unit_variance = unit_variance
        self.copy = copy
        self.device = device
        self.dtype = dtype


"""Generated config for StandardScaler."""
class StandardScalerConfig(ConfigTemplate):
    model_name = "StandardScaler"
    model_path = "Code.models.machine_learning.preprocessing.normalizer.normalizer"

    def __init__(self,
        immutable: bool = True,
        copy: bool = True,
        with_mean: bool = True,
        with_std: bool = True,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self.device = device
        self.dtype = dtype

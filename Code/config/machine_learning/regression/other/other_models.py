"""Config templates for other_models."""
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from .....config import ConfigTemplate

try:
    import torch
except ImportError:
    torch = None

"""Generated config for DummyRegressor."""
class DummyRegressorConfig(ConfigTemplate):
    model_name = "DummyRegressor"
    model_path = "Code.models.machine_learning.regression.other.other_models"

    def __init__(self,
        immutable: bool = True,
        strategy: str = 'mean',
        constant: Union[int, float] = None,
        quantile: float = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.strategy = strategy
        self.constant = constant
        self.quantile = quantile
        self.device = device
        self.dtype = dtype


"""Generated config for IsotonicRegression."""
class IsotonicRegressionConfig(ConfigTemplate):
    model_name = "IsotonicRegression"
    model_path = "Code.models.machine_learning.regression.other.other_models"

    def __init__(self,
        immutable: bool = True,
        y_min: float = None,
        y_max: float = None,
        increasing: Union[str, bool] = True,
        out_of_bounds: str = 'nan',
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.y_min = y_min
        self.y_max = y_max
        self.increasing = increasing
        self.out_of_bounds = out_of_bounds
        self.device = device
        self.dtype = dtype

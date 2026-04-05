"""Config templates for generative_data."""
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from .....config import ConfigTemplate

try:
    import torch
except ImportError:
    torch = None

"""Generated config for KernelDensity."""
class KernelDensityConfig(ConfigTemplate):
    model_name = "KernelDensity"
    model_path = "Code.models.machine_learning.preprocessing.generative_data.generative_data"

    def __init__(self,
        immutable: bool = True,
        bandwidth: Union[Literal['scott', 'silverman'], float] = 1.0,
        algorithm: Union[Literal['kd_tree', 'ball_tree', 'auto'], Callable, nn.Module] = 'auto',
        kernel: Union[Literal['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'], str, Callable, nn.Module] = 'gaussian',
        metric: Union[str, Callable, nn.Module] = 'euclidean',
        atol: float = 0,
        rtol: float = 0,
        breadth_first: bool = True,
        leaf_size: int = 40,
        metric_params: dict = None,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.bandwidth = bandwidth
        self.algorithm = algorithm
        self.kernel = kernel
        self.metric = metric
        self.atol = atol
        self.rtol = rtol
        self.breadth_first = breadth_first
        self.leaf_size = leaf_size
        self.metric_params = metric_params
        self.device = device
        self.dtype = dtype

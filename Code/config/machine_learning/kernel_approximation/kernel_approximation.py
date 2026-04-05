"""Config templates for kernel_approximation."""
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from ....config import ConfigTemplate

try:
    import torch
except ImportError:
    torch = None

"""Generated config for AdditiveChi2Sampler."""
class AdditiveChi2SamplerConfig(ConfigTemplate):
    model_name = "AdditiveChi2Sampler"
    model_path = "Code.models.machine_learning.kernel_approximation.kernel_approximatio"

    def __init__(self,
        immutable: bool = True,
        sample_steps: int = 2,
        sample_interval: float = None,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.sample_steps = sample_steps
        self.sample_interval = sample_interval
        self.device = device
        self.dtype = dtype


"""Generated config for Nystroem."""
class NystroemConfig(ConfigTemplate):
    model_name = "Nystroem"
    model_path = "Code.models.machine_learning.kernel_approximation.kernel_approximatio"

    def __init__(self,
        immutable: bool = True,
        kernel: Union[str, MLModule, Callable] = 'rbf',
        gamma: float = None,
        coef0: float = None,
        degree: float = None,
        kernel_params: dict = None,
        n_components: int = 100,
        random_state: Union[int, torch.Generator] = None,
        n_jobs: int = None,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.kernel_params = kernel_params
        self.n_components = n_components
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype


"""Generated config for PolynomialCountSketch."""
class PolynomialCountSketchConfig(ConfigTemplate):
    model_name = "PolynomialCountSketch"
    model_path = "Code.models.machine_learning.kernel_approximation.kernel_approximatio"

    def __init__(self,
        immutable: bool = True,
        gamma: float = None,
        coef0: float = None,
        degree: float = None,
        n_components: int = 100,
        random_state: Union[int, torch.Generator] = None,
        n_jobs: int = None,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.n_components = n_components
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype


"""Generated config for RBFSampler."""
class RBFSamplerConfig(ConfigTemplate):
    model_name = "RBFSampler"
    model_path = "Code.models.machine_learning.kernel_approximation.kernel_approximatio"

    def __init__(self,
        immutable: bool = True,
        gamma: Union[float, Literal['scale']] = None,
        n_components: int = 100,
        random_state: Union[int, torch.Generator] = None,
        n_jobs: int = None,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype


"""Generated config for SkewedChi2Sampler."""
class SkewedChi2SamplerConfig(ConfigTemplate):
    model_name = "SkewedChi2Sampler"
    model_path = "Code.models.machine_learning.kernel_approximation.kernel_approximatio"

    def __init__(self,
        immutable: bool = True,
        skewedness: float = 1.0,
        n_components: int = 100,
        random_state: Union[int, torch.Generator] = None,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.skewedness = skewedness
        self.n_components = n_components
        self.random_state = random_state
        self.device = device
        self.dtype = dtype

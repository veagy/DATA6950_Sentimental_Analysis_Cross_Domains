"""Config templates for covariance."""
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from .....config import ConfigTemplate

try:
    import torch
except ImportError:
    torch = None

"""Generated config for EllipticEnvelope."""
class EllipticEnvelopeConfig(ConfigTemplate):
    model_name = "EllipticEnvelope"
    model_path = "Code.models.machine_learning.feature_selection.covariance.covariance"

    def __init__(self,
        immutable: bool = True,
        store_precision: bool = True,
        assume_centered: bool = False,
        support_fraction: float = None,
        contamination: float = 0.1,
        random_state: Union[int, torch.Generator] = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.store_precision = store_precision
        self.assume_centered = assume_centered
        self.support_fraction = support_fraction
        self.contamination = contamination
        self.random_state = random_state
        self.device = device
        self.dtype = dtype


"""Generated config for EmpiricalCovariance."""
class EmpiricalCovarianceConfig(ConfigTemplate):
    model_name = "EmpiricalCovariance"
    model_path = "Code.models.machine_learning.feature_selection.covariance.covariance"

    def __init__(self,
        immutable: bool = True,
        store_precision: bool = True,
        assume_centered: bool = False,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.store_precision = store_precision
        self.assume_centered = assume_centered
        self.device = device
        self.dtype = dtype


"""Generated config for GraphicalElasticNet."""
class GraphicalElasticNetConfig(ConfigTemplate):
    model_name = "GraphicalElasticNet"
    model_path = "Code.models.machine_learning.feature_selection.covariance.covariance"

    def __init__(self,
        immutable: bool = True,
        alpha: float = 0.01,
        l1_ratio: float = 0.5,
        mode: Literal['lars', 'cd'] = 'cd',
        covariance: Union[Literal['precomputed', 'none'], torch.Tensor] = None,
        tol: float = 0.0001,
        enet_tol: float = 0.0001,
        max_iter: int = 100,
        verbose: bool = False,
        eps: float = _FLOAT64_EPS,
        assume_centered: bool = False,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.mode = mode
        self.covariance = covariance
        self.tol = tol
        self.enet_tol = enet_tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.eps = eps
        self.assume_centered = assume_centered
        self.device = device
        self.dtype = dtype


"""Generated config for GraphicalElasticNetCV."""
class GraphicalElasticNetCVConfig(ConfigTemplate):
    model_name = "GraphicalElasticNetCV"
    model_path = "Code.models.machine_learning.feature_selection.covariance.covariance"

    def __init__(self,
        immutable: bool = True,
        alphas: Union[int, List[float], torch.Tensor] = 4,
        l1_ratios: Union[float, List[float], torch.Tensor] = 0.5,
        mode: Literal['lars', 'cd'] = 'cd',
        covariance: Union[Literal['precomputed', 'none'], torch.Tensor] = None,
        tol: float = 0.0001,
        enet_tol: float = 0.0001,
        cv: Union[int, str, MLModule] = None,
        cv_config: dict = None,
        max_iter: int = 100,
        verbose: bool = False,
        eps: float = _FLOAT64_EPS,
        assume_centered: bool = False,
        n_refinements: int = 4,
        n_jobs: Optional[int] = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.alphas = alphas
        self.l1_ratios = l1_ratios
        self.mode = mode
        self.covariance = covariance
        self.tol = tol
        self.enet_tol = enet_tol
        self.cv = cv
        self.cv_config = cv_config
        self.max_iter = max_iter
        self.verbose = verbose
        self.eps = eps
        self.assume_centered = assume_centered
        self.n_refinements = n_refinements
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype


"""Generated config for GraphicalLasso."""
class GraphicalLassoConfig(ConfigTemplate):
    model_name = "GraphicalLasso"
    model_path = "Code.models.machine_learning.feature_selection.covariance.covariance"

    def __init__(self,
        immutable: bool = True,
        alpha: float = 0.01,
        mode: Literal['lars', 'cd'] = 'cd',
        covariance: Union[Literal['precomputed', 'none'], torch.Tensor] = None,
        tol: float = 0.0001,
        enet_tol: float = 0.0001,
        max_iter: int = 100,
        verbose: bool = False,
        eps: float = 2.220446049250313e-16,
        assume_centered: bool = False,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.alpha = alpha
        self.mode = mode
        self.covariance = covariance
        self.tol = tol
        self.enet_tol = enet_tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.eps = eps
        self.assume_centered = assume_centered
        self.device = device
        self.dtype = dtype


"""Generated config for GraphicalLassoCV."""
class GraphicalLassoCVConfig(ConfigTemplate):
    model_name = "GraphicalLassoCV"
    model_path = "Code.models.machine_learning.feature_selection.covariance.covariance"

    def __init__(self,
        immutable: bool = True,
        alphas: [float, list, tuple, torch.Tensor] = 0.01,
        mode: Literal['lars', 'cd'] = 'cd',
        covariance: Union[Literal['precomputed', 'none'], torch.Tensor] = None,
        tol: float = 0.0001,
        cv: Union[int, str, MLModule] = None,
        cv_config: dict = None,
        enet_tol: float = 0.0001,
        max_iter: int = 100,
        verbose: bool = False,
        eps: float = 2.220446049250313e-16,
        assume_centered: bool = False,
        n_refinements: int = 4,
        n_jobs: Optional[int] = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.alphas = alphas
        self.mode = mode
        self.covariance = covariance
        self.tol = tol
        self.cv = cv
        self.cv_config = cv_config
        self.enet_tol = enet_tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.eps = eps
        self.assume_centered = assume_centered
        self.n_refinements = n_refinements
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype


"""Generated config for GraphicalRidge."""
class GraphicalRidgeConfig(ConfigTemplate):
    model_name = "GraphicalRidge"
    model_path = "Code.models.machine_learning.feature_selection.covariance.covariance"

    def __init__(self,
        immutable: bool = True,
        alpha: float = 0.01,
        covariance: Union[Literal['precomputed', 'none'], torch.Tensor] = None,
        tol: float = 0.0001,
        max_iter: int = 100,
        verbose: bool = False,
        eps: float = _FLOAT64_EPS,
        assume_centered: bool = False,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.alpha = alpha
        self.covariance = covariance
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.eps = eps
        self.assume_centered = assume_centered
        self.device = device
        self.dtype = dtype


"""Generated config for GraphicalRidgeCV."""
class GraphicalRidgeCVConfig(ConfigTemplate):
    model_name = "GraphicalRidgeCV"
    model_path = "Code.models.machine_learning.feature_selection.covariance.covariance"

    def __init__(self,
        immutable: bool = True,
        alphas: Union[int, List[float], torch.Tensor] = 4,
        covariance: Union[Literal['precomputed', 'none'], torch.Tensor] = None,
        tol: float = 0.0001,
        cv: Union[int, str, MLModule] = None,
        cv_config: dict = None,
        max_iter: int = 100,
        verbose: bool = False,
        eps: float = _FLOAT64_EPS,
        assume_centered: bool = False,
        n_refinements: int = 4,
        n_jobs: Optional[int] = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.alphas = alphas
        self.covariance = covariance
        self.tol = tol
        self.cv = cv
        self.cv_config = cv_config
        self.max_iter = max_iter
        self.verbose = verbose
        self.eps = eps
        self.assume_centered = assume_centered
        self.n_refinements = n_refinements
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype


"""Generated config for LedoitWolf."""
class LedoitWolfConfig(ConfigTemplate):
    model_name = "LedoitWolf"
    model_path = "Code.models.machine_learning.feature_selection.covariance.covariance"

    def __init__(self,
        immutable: bool = True,
        store_precision: bool = True,
        assume_centered: bool = False,
        block_size: int = 1000,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.store_precision = store_precision
        self.assume_centered = assume_centered
        self.block_size = block_size
        self.device = device
        self.dtype = dtype


"""Generated config for MinCovDet."""
class MinCovDetConfig(ConfigTemplate):
    model_name = "MinCovDet"
    model_path = "Code.models.machine_learning.feature_selection.covariance.covariance"

    def __init__(self,
        immutable: bool = True,
        store_precision: bool = True,
        assume_centered: bool = False,
        support_fraction: float = None,
        random_state: Union[int, torch.Generator] = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.store_precision = store_precision
        self.assume_centered = assume_centered
        self.support_fraction = support_fraction
        self.random_state = random_state
        self.device = device
        self.dtype = dtype


"""Generated config for OAS."""
class OASConfig(ConfigTemplate):
    model_name = "OAS"
    model_path = "Code.models.machine_learning.feature_selection.covariance.covariance"

    def __init__(self,
        immutable: bool = True,
        store_precision: bool = True,
        assume_centered: bool = False,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.store_precision = store_precision
        self.assume_centered = assume_centered
        self.device = device
        self.dtype = dtype


"""Generated config for ShrunkCovariance."""
class ShrunkCovarianceConfig(ConfigTemplate):
    model_name = "ShrunkCovariance"
    model_path = "Code.models.machine_learning.feature_selection.covariance.covariance"

    def __init__(self,
        immutable: bool = True,
        store_precision: bool = True,
        assume_centered: bool = False,
        shrinkage: float = 0.1,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.store_precision = store_precision
        self.assume_centered = assume_centered
        self.shrinkage = shrinkage
        self.device = device
        self.dtype = dtype

"""Config templates for manifold_learning."""
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from .....config import ConfigTemplate

try:
    import torch
except ImportError:
    torch = None

"""Generated config for ClassicalMDS."""
class ClassicalMDSConfig(ConfigTemplate):
    model_name = "ClassicalMDS"
    model_path = "Code.models.machine_learning.transformer.manifold_learning.manifold_learning"

    def __init__(self,
        immutable: bool = True,
        n_components: int = 2,
        metric: Union[str, Callable, nn.Module] = 'euclidean',
        metric_params: dict = None,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_components = n_components
        self.metric = metric
        self.metric_params = metric_params
        self.device = device
        self.dtype = dtype


"""Generated config for Isomap."""
class IsomapConfig(ConfigTemplate):
    model_name = "Isomap"
    model_path = "Code.models.machine_learning.transformer.manifold_learning.manifold_learning"

    def __init__(self,
        immutable: bool = True,
        n_neighbors: int = 5,
        radius: float = None,
        n_components: int = 2,
        eigen_solver: Union[Literal['auto', 'arpack', 'dense'], Callable, nn.Module] = 'auto',
        tol: float = 0,
        max_iter: int = None,
        path_method: Union[Literal['auto', 'FW', 'D'], Callable, nn.Module] = 'auto',
        neighbors_algorithm: Union[Literal['auto', 'brute', 'ball_tree', 'kd_tree'], Callable, nn.Module] = 'auto',
        n_jobs: int = None,
        metric: Union[str, Callable, nn.Module] = 'minkowski',
        p: float = 2,
        metric_params: dict = None,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.n_components = n_components
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.path_method = path_method
        self.neighbors_algorithm = neighbors_algorithm
        self.n_jobs = n_jobs
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.device = device
        self.dtype = dtype


"""Generated config for LocallyLinearEmbedding."""
class LocallyLinearEmbeddingConfig(ConfigTemplate):
    model_name = "LocallyLinearEmbedding"
    model_path = "Code.models.machine_learning.transformer.manifold_learning.manifold_learning"

    def __init__(self,
        immutable: bool = True,
        n_neighbors: int = 5,
        n_components: int = 2,
        reg: float = 0.001,
        eigen_solver: Union[Literal['auto', 'arpack', 'dense']] = 'auto',
        tol: float = 1e-06,
        max_iter: int = 100,
        method: Union[Literal['standard', 'hessian', 'modified', 'ltsa'], Callable, nn.Module] = 'standard',
        hessian_tol: float = 0.0001,
        modified_tol: float = 1e-12,
        neighbors_algorithm: Union[Literal['auto', 'brute', 'ball_tree', 'kd_tree'], Callable, nn.Module] = 'auto',
        random_state: Union[int, torch.Generator] = None,
        n_jobs: int = None,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.reg = reg
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.method = method
        self.hessian_tol = hessian_tol
        self.modified_tol = modified_tol
        self.neighbors_algorithm = neighbors_algorithm
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype


"""Generated config for MDS."""
class MDSConfig(ConfigTemplate):
    model_name = "MDS"
    model_path = "Code.models.machine_learning.transformer.manifold_learning.manifold_learning"

    def __init__(self,
        immutable: bool = True,
        n_components: int = 2,
        metric_mds: bool = True,
        n_init: int = 1,
        init: Union[Literal['random', 'classical_mds'], Callable, nn.Module] = 'classical_mds',
        max_iter: int = 100,
        verbose: int = 0,
        eps: float = 1e-06,
        n_jobs: int = None,
        random_state: Union[int, torch.Generator] = None,
        metric: Union[str, Callable, nn.Module] = 'euclidean',
        metric_params: dict = None,
        normalized_stress: Union[bool, Literal['auto']] = 'auto',
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_components = n_components
        self.metric_mds = metric_mds
        self.n_init = n_init
        self.init = init
        self.max_iter = max_iter
        self.verbose = verbose
        self.eps = eps
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.metric = metric
        self.metric_params = metric_params
        self.normalized_stress = normalized_stress
        self.device = device
        self.dtype = dtype


"""Generated config for SpectralEmbedding."""
class SpectralEmbeddingConfig(ConfigTemplate):
    model_name = "SpectralEmbedding"
    model_path = "Code.models.machine_learning.transformer.manifold_learning.manifold_learning"

    def __init__(self,
        immutable: bool = True,
        n_components: int = 2,
        affinity: Union[Literal['nearest_neighbors', 'rbf', 'precomputed', 'precomputed_nearest_neighbors'], Callable, nn.Module] = 'nearest_neighbors',
        gamma: float = None,
        random_state: Union[int, torch.Generator] = None,
        eigen_solver: Union[Literal['arpack', 'lobpcg', 'amg'], Callable, nn.Module] = None,
        eigen_tol: Union[Literal['auto'], float] = 'auto',
        n_neighbors: int = None,
        n_jobs: int = None,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_components = n_components
        self.affinity = affinity
        self.gamma = gamma
        self.random_state = random_state
        self.eigen_solver = eigen_solver
        self.eigen_tol = eigen_tol
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype


"""Generated config for TSNE."""
class TSNEConfig(ConfigTemplate):
    model_name = "TSNE"
    model_path = "Code.models.machine_learning.transformer.manifold_learning.manifold_learning"

    def __init__(self,
        immutable: bool = True,
        n_components: int = 2,
        perplexity: float = 30.0,
        early_exaggeration: float = 12.0,
        learning_rate: Union[Literal['auto'], float] = 'auto',
        max_iter: int = 1000,
        n_iter_without_progress: int = 300,
        min_grad_norm: float = 1e-07,
        metric: Union[str, Callable, nn.Module] = 'euclidean',
        metric_params: Optional[dict] = None,
        init: Union[Literal['random', 'pca'], tuple, list, torch.Tensor] = 'pca',
        verbose: int = 0,
        random_state: Optional[Union[int, torch.Generator]] = None,
        method: Union[Literal['barnes_hut', 'exact'], Callable, nn.Module] = 'barnes_hut',
        method_params: Optional[dict] = None,
        angle: float = 0.5,
        n_jobs: Optional[int] = None,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.metric = metric
        self.metric_params = metric_params
        self.init = init
        self.verbose = verbose
        self.random_state = random_state
        self.method = method
        self.method_params = method_params
        self.angle = angle
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype

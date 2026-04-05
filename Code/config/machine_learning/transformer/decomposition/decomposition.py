"""Config templates for decomposition."""
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from .....config import ConfigTemplate

try:
    import torch
except ImportError:
    torch = None

"""Generated config for DictionaryLearning."""
class DictionaryLearningConfig(ConfigTemplate):
    model_name = "DictionaryLearning"
    model_path = "Code.models.machine_learning.transformer.decomposition.decompositio"

    def __init__(self,
        immutable: bool = True,
        n_components: int = None,
        alpha: float = 1,
        max_iter: int = 1000,
        tol: float = 1e-08,
        fit_algorithm: Union[Literal['lars', 'cd'], Callable, nn.Module] = 'lars',
        transform_algorithm: Union[Literal['lars', 'lasso_lars', 'lasso_cd', 'omp', 'threshold'], Callable, nn.Module] = 'omp',
        transform_n_nonzero_coefs: int = None,
        transform_alpha: float = None,
        n_jobs: int = None,
        code_init: Union[list, tuple, torch.Tensor] = None,
        dict_init: Union[list, tuple, torch.Tensor] = None,
        callback: Union[Callable, nn.Module] = None,
        verbose: bool = False,
        split_sign: bool = False,
        random_state: Union[int, torch.Generator] = None,
        positive_code: bool = False,
        positive_dict: bool = False,
        transform_max_iter: int = 1000,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_components = n_components
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_algorithm = fit_algorithm
        self.transform_algorithm = transform_algorithm
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs
        self.transform_alpha = transform_alpha
        self.n_jobs = n_jobs
        self.code_init = code_init
        self.dict_init = dict_init
        self.callback = callback
        self.verbose = verbose
        self.split_sign = split_sign
        self.random_state = random_state
        self.positive_code = positive_code
        self.positive_dict = positive_dict
        self.transform_max_iter = transform_max_iter
        self.device = device
        self.dtype = dtype


"""Generated config for FactorAnalysis."""
class FactorAnalysisConfig(ConfigTemplate):
    model_name = "FactorAnalysis"
    model_path = "Code.models.machine_learning.transformer.decomposition.decompositio"

    def __init__(self,
        immutable: bool = True,
        n_components: int = None,
        tol: float = 0.01,
        copy: bool = True,
        max_iter: int = 1000,
        noise_variance_init: Union[list, tuple, torch.Tensor] = None,
        svd_method: Union[Literal['lapack', 'la_pack', 'randomized'], Callable, nn.Module] = 'randomized',
        iterated_power: int = 3,
        rotation: Union[Literal['varimax', 'quartimax'], Callable] = None,
        random_state: Optional[Union[int, torch.Generator]] = 0,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_components = n_components
        self.tol = tol
        self.copy = copy
        self.max_iter = max_iter
        self.noise_variance_init = noise_variance_init
        self.svd_method = svd_method
        self.iterated_power = iterated_power
        self.rotation = rotation
        self.random_state = random_state
        self.device = device
        self.dtype = dtype


"""Generated config for FastICA."""
class FastICAConfig(ConfigTemplate):
    model_name = "FastICA"
    model_path = "Code.models.machine_learning.transformer.decomposition.decompositio"

    def __init__(self,
        immutable: bool = True,
        n_components: Optional[int] = None,
        algorithm: Union[Literal['parallel', 'deflation'], Callable, nn.Module] = 'parallel',
        whiten: Union[bool, Literal['arbitrary-variance', 'unit-variance']] = 'unit-variance',
        fun: Union[str, Literal['logcosh', 'exp', 'cube'], Callable, nn.Module] = 'logcosh',
        func_args: Optional[dict] = None,
        max_iter: int = 200,
        tol: float = 0.0001,
        w_init: Optional[Union[List, Tuple, torch.Tensor]] = None,
        whiten_solver: Union[Literal['eigh', 'svd'], Callable, nn.Module] = 'svd',
        random_state: Optional[Union[int, torch.Generator]] = None,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        whiten_solver_kwargs: Optional[Dict[str, Any]] = None,
        algorithm_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_components = n_components
        self.algorithm = algorithm
        self.whiten = whiten
        self.fun = fun
        self.func_args = func_args
        self.max_iter = max_iter
        self.tol = tol
        self.w_init = w_init
        self.whiten_solver = whiten_solver
        self.random_state = random_state
        self.device = device
        self.dtype = dtype
        self.whiten_solver_kwargs = whiten_solver_kwargs
        self.algorithm_kwargs = algorithm_kwargs


"""Generated config for LatentDirichletAllocation."""
class LatentDirichletAllocationConfig(ConfigTemplate):
    model_name = "LatentDirichletAllocation"
    model_path = "Code.models.machine_learning.transformer.decomposition.decompositio"

    def __init__(self,
        immutable: bool = True,
        n_components: int = 10,
        doc_topic_prior: float = None,
        topic_word_prior: float = None,
        learning_method: Union[Literal['batch', 'online'], Callable, nn.Module] = 'batch',
        learning_decay: float = 0.7,
        learning_offset: float = 10.0,
        max_iter: int = 10,
        batch_size: int = 128,
        evaluate_every: int = -1,
        total_samples: int = 1000000.0,
        perp_tol: float = 0.1,
        mean_change_tol: float = 0.001,
        max_doc_update_iter: int = 100,
        n_jobs: int = None,
        verbose: int = 0,
        random_state: Union[int, torch.Generator] = None,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_components = n_components
        self.doc_topic_prior = doc_topic_prior
        self.topic_word_prior = topic_word_prior
        self.learning_method = learning_method
        self.learning_decay = learning_decay
        self.learning_offset = learning_offset
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.evaluate_every = evaluate_every
        self.total_samples = total_samples
        self.perp_tol = perp_tol
        self.mean_change_tol = mean_change_tol
        self.max_doc_update_iter = max_doc_update_iter
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        self.device = device
        self.dtype = dtype


"""Generated config for MiniBatchDictionaryLearning."""
class MiniBatchDictionaryLearningConfig(ConfigTemplate):
    model_name = "MiniBatchDictionaryLearning"
    model_path = "Code.models.machine_learning.transformer.decomposition.decompositio"

    def __init__(self,
        immutable: bool = True,
        n_components: int = None,
        alpha: float = 1,
        max_iter: int = 1000,
        tol: float = 1e-08,
        fit_algorithm: Union[Literal['lars', 'cd'], Callable, nn.Module] = 'lars',
        transform_algorithm: Union[Literal['lars', 'lasso_lars', 'lasso_cd', 'omp', 'threshold'], Callable, nn.Module] = 'omp',
        transform_n_nonzero_coefs: int = None,
        transform_alpha: float = None,
        batch_size: int = 256,
        shuffle = True,
        n_jobs: int = None,
        code_init: Union[list, tuple, torch.Tensor] = None,
        dict_init: Union[list, tuple, torch.Tensor] = None,
        callback: Union[Callable, nn.Module] = None,
        verbose: bool = False,
        split_sign: bool = False,
        random_state: Union[int, torch.Generator] = None,
        positive_code: bool = False,
        positive_dict: bool = False,
        transform_max_iter: int = 1000,
        max_no_improvement = 10,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_components = n_components
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_algorithm = fit_algorithm
        self.transform_algorithm = transform_algorithm
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs
        self.transform_alpha = transform_alpha
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_jobs = n_jobs
        self.code_init = code_init
        self.dict_init = dict_init
        self.callback = callback
        self.verbose = verbose
        self.split_sign = split_sign
        self.random_state = random_state
        self.positive_code = positive_code
        self.positive_dict = positive_dict
        self.transform_max_iter = transform_max_iter
        self.max_no_improvement = max_no_improvement
        self.device = device
        self.dtype = dtype


"""Generated config for MiniBatchNMF."""
class MiniBatchNMFConfig(ConfigTemplate):
    model_name = "MiniBatchNMF"
    model_path = "Code.models.machine_learning.transformer.decomposition.decompositio"

    def __init__(self,
        immutable: bool = True,
        n_components: Union[Literal['auto'], int, None] = 'auto',
        init: Union[Literal['random', 'nndsvd', 'nndsvda', 'nndsvdar', 'custom'], Callable, nn.Module, None] = None,
        solver: Union[Literal['cd', 'mu'], Callable, nn.Module] = 'cd',
        beta_loss: Union[float, Literal['frobenius', 'kullback-leibler', 'itakura-saito'], Callable] = 'frobenius',
        tol: float = 0.0001,
        batch_size: int = 1024,
        max_iter: int = 200,
        max_no_improvement: int = 10,
        random_state: Optional[Union[int, torch.Generator]] = None,
        alpha_W: float = 0.0,
        alpha_H: Union[float, Literal['same']] = 'same',
        l1_ratio: float = 0.0,
        forget_factor: float = 0.7,
        fresh_restarts = False,
        fresh_restarts_max_iter = 30,
        transform_max_iter = None,
        verbose: int = 0,
        shuffle: bool = False,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_components = n_components
        self.init = init
        self.solver = solver
        self.beta_loss = beta_loss
        self.tol = tol
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.max_no_improvement = max_no_improvement
        self.random_state = random_state
        self.alpha_W = alpha_W
        self.alpha_H = alpha_H
        self.l1_ratio = l1_ratio
        self.forget_factor = forget_factor
        self.fresh_restarts = fresh_restarts
        self.fresh_restarts_max_iter = fresh_restarts_max_iter
        self.transform_max_iter = transform_max_iter
        self.verbose = verbose
        self.shuffle = shuffle
        self.device = device
        self.dtype = dtype


"""Generated config for NMF."""
class NMFConfig(ConfigTemplate):
    model_name = "NMF"
    model_path = "Code.models.machine_learning.transformer.decomposition.decompositio"

    def __init__(self,
        immutable: bool = True,
        n_components: Union[Literal['auto'], int, None] = 'auto',
        init: Union[Literal['random', 'nndsvd', 'nndsvda', 'nndsvdar', 'custom'], Callable, nn.Module, None] = None,
        solver: Union[Literal['cd', 'mu'], Callable, nn.Module] = 'cd',
        beta_loss: Union[float, Literal['frobenius', 'kullback-leibler', 'itakura-saito'], Callable, nn.Module] = 'frobenius',
        tol: float = 0.0001,
        max_iter: int = 200,
        random_state: Optional[Union[int, torch.Generator]] = None,
        alpha_W: float = 0.0,
        alpha_H: Union[float, Literal['same']] = 'same',
        l1_ratio: float = 0.0,
        verbose: int = 0,
        shuffle: bool = False,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_components = n_components
        self.init = init
        self.solver = solver
        self.beta_loss = beta_loss
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha_W = alpha_W
        self.alpha_H = alpha_H
        self.l1_ratio = l1_ratio
        self.verbose = verbose
        self.shuffle = shuffle
        self.device = device
        self.dtype = dtype


"""Generated config for NeighborhoodComponentsAnalysis."""
class NeighborhoodComponentsAnalysisConfig(ConfigTemplate):
    model_name = "NeighborhoodComponentsAnalysis"
    model_path = "Code.models.machine_learning.transformer.decomposition.decompositio"

    def __init__(self,
        immutable: bool = True,
        n_components: int = None,
        init: Union[Literal['auto', 'pca', 'lda', 'identity', 'random'], list, tuple, torch.Tensor] = 'auto',
        warm_start: bool = False,
        max_iter: int = 50,
        tol: float = 1e-05,
        callback: Union[Callable, nn.Module] = None,
        verbose: int = 0,
        random_state: Union[int, torch.Generator] = None,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_components = n_components
        self.init = init
        self.warm_start = warm_start
        self.max_iter = max_iter
        self.tol = tol
        self.callback = callback
        self.verbose = verbose
        self.random_state = random_state
        self.device = device
        self.dtype = dtype


"""Generated config for SparseCoder."""
class SparseCoderConfig(ConfigTemplate):
    model_name = "SparseCoder"
    model_path = "Code.models.machine_learning.transformer.decomposition.decompositio"

    def __init__(self,
        immutable: bool = True,
        dictionary: Union[list, tuple, torch.Tensor, None] = None,
        transform_algorithm: Union[Literal['lasso_lars', 'ridge_lars', 'elasticnet_lars', 'lasso_cd', 'ridge_cd', 'elasticnet_cd', 'lasso', 'ridge', 'elasticnet', 'omp', 'threshold'], Callable, nn.Module] = 'omp',
        transform_n_nonzero_coefs: int = None,
        transform_alpha: float = None,
        split_sign: bool = False,
        n_jobs: int = None,
        positive_code: bool = False,
        transform_max_iter: int = 1000,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.dictionary = dictionary
        self.transform_algorithm = transform_algorithm
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs
        self.transform_alpha = transform_alpha
        self.split_sign = split_sign
        self.n_jobs = n_jobs
        self.positive_code = positive_code
        self.transform_max_iter = transform_max_iter
        self.device = device
        self.dtype = dtype


"""Generated config for TruncatedSVD."""
class TruncatedSVDConfig(ConfigTemplate):
    model_name = "TruncatedSVD"
    model_path = "Code.models.machine_learning.transformer.decomposition.decompositio"

    def __init__(self,
        immutable: bool = True,
        n_components: int = 2,
        algorithm: Union[Literal['arpack', 'randomized'], Callable, nn.Module] = 'randomized',
        n_iter: int = 5,
        n_oversamples: int = 10,
        power_iteration_normalizer: Union[Literal['auto', 'QR', 'LU', 'none'], Callable, nn.Module] = 'auto',
        random_state: Optional[Union[int, torch.Generator]] = None,
        tol: float = 0.0,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_components = n_components
        self.algorithm = algorithm
        self.n_iter = n_iter
        self.n_oversamples = n_oversamples
        self.power_iteration_normalizer = power_iteration_normalizer
        self.random_state = random_state
        self.tol = tol
        self.device = device
        self.dtype = dtype

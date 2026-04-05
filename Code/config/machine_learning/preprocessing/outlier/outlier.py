"""Config templates for outlier."""
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from .....config import ConfigTemplate

try:
    import torch
except ImportError:
    torch = None

"""Generated config for AdEMAMixEstimatorBasedOutlierDetection."""
class AdEMAMixEstimatorBasedOutlierDetectionConfig(ConfigTemplate):
    model_name = "AdEMAMixEstimatorBasedOutlierDetection"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        estimator: Optional[MLModule] = None,
        hidden_dim: int = 64,
        bottleneck_dim: int = 16,
        n_layers: int = 2,
        activation: str = 'relu',
        contamination: Union[Literal['auto'], float] = 'auto',
        max_iter: int = 100,
        tol: float = 0.0001,
        eta0: float = 0.001,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
        verbose: bool = False,
        beta1: float = 0.9,
        beta2: float = 0.999,
        alpha: float = 5.0,
        beta3: float = 0.9999,
        T_alpha_beta3: int = 1000,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_layers = n_layers
        self.activation = activation
        self.contamination = contamination
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.beta1 = beta1
        self.beta2 = beta2
        self.alpha = alpha
        self.beta3 = beta3
        self.T_alpha_beta3 = T_alpha_beta3
        self.device = device
        self.dtype = dtype


"""Generated config for AdEMAMixOneClassSVM."""
class AdEMAMixOneClassSVMConfig(ConfigTemplate):
    model_name = "AdEMAMixOneClassSVM"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        kernel = 'rbf',
        kernel_params = None,
        degree = 3,
        gamma = 'scale',
        coef0 = 0.0,
        tol = 0.001,
        fit_intercept = True,
        shuffle = True,
        random_state = None,
        learning_rate = 'invscaling',
        eta0 = 0.01,
        power_t = 0.5,
        warm_start = False,
        average = False,
        nu = 0.5,
        shrinking = True,
        cache_size = 200,
        verbose = False,
        max_iter = -1,
        beta1: float = 0.9,
        beta2: float = 0.999,
        smoothening_term: float = 1e-08,
        alpha: float = 5.0,
        beta3: float = 0.9999,
        T_alpha_beta3: int = 1000,
        device = 'cpu',
        dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.warm_start = warm_start
        self.average = average
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.beta1 = beta1
        self.beta2 = beta2
        self.smoothening_term = smoothening_term
        self.alpha = alpha
        self.beta3 = beta3
        self.T_alpha_beta3 = T_alpha_beta3
        self.device = device
        self.dtype = dtype


"""Generated config for AdaGradEstimatorBasedOutlierDetection."""
class AdaGradEstimatorBasedOutlierDetectionConfig(ConfigTemplate):
    model_name = "AdaGradEstimatorBasedOutlierDetection"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        estimator: Optional[MLModule] = None,
        hidden_dim: int = 64,
        bottleneck_dim: int = 16,
        n_layers: int = 2,
        activation: str = 'relu',
        contamination: Union[Literal['auto'], float] = 'auto',
        max_iter: int = 100,
        tol: float = 0.0001,
        eta0: float = 0.001,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
        verbose: bool = False,
        smoothening_term: float = 1e-10,
        weight_decay: float = 0.0,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_layers = n_layers
        self.activation = activation
        self.contamination = contamination
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.smoothening_term = smoothening_term
        self.weight_decay = weight_decay
        self.device = device
        self.dtype = dtype


"""Generated config for AdaGradOneClassSVM."""
class AdaGradOneClassSVMConfig(ConfigTemplate):
    model_name = "AdaGradOneClassSVM"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        kernel = 'rbf',
        kernel_params = None,
        degree = 3,
        gamma = 'scale',
        coef0 = 0.0,
        tol = 0.001,
        fit_intercept = True,
        shuffle = True,
        random_state = None,
        learning_rate = 'invscaling',
        eta0 = 0.01,
        power_t = 0.5,
        warm_start = False,
        average = False,
        nu = 0.5,
        shrinking = True,
        cache_size = 200,
        verbose = False,
        max_iter = -1,
        smoothening_term: float = 1e-08,
        device = 'cpu',
        dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.warm_start = warm_start
        self.average = average
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.smoothening_term = smoothening_term
        self.device = device
        self.dtype = dtype


"""Generated config for AdadeltaEstimatorBasedOutlierDetection."""
class AdadeltaEstimatorBasedOutlierDetectionConfig(ConfigTemplate):
    model_name = "AdadeltaEstimatorBasedOutlierDetection"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        estimator: Optional[MLModule] = None,
        hidden_dim: int = 64,
        bottleneck_dim: int = 16,
        n_layers: int = 2,
        activation: str = 'relu',
        contamination: Union[Literal['auto'], float] = 'auto',
        max_iter: int = 100,
        tol: float = 0.0001,
        eta0: float = 1.0,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
        verbose: bool = False,
        rho: float = 0.9,
        smoothening_term: float = 1e-06,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_layers = n_layers
        self.activation = activation
        self.contamination = contamination
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.rho = rho
        self.smoothening_term = smoothening_term
        self.device = device
        self.dtype = dtype


"""Generated config for AdadeltaOneClassSVM."""
class AdadeltaOneClassSVMConfig(ConfigTemplate):
    model_name = "AdadeltaOneClassSVM"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        kernel = 'rbf',
        kernel_params = None,
        degree = 3,
        gamma = 'scale',
        coef0 = 0.0,
        tol = 0.001,
        fit_intercept = True,
        shuffle = True,
        random_state = None,
        learning_rate = 'invscaling',
        eta0 = 0.01,
        power_t = 0.5,
        warm_start = False,
        average = False,
        nu = 0.5,
        shrinking = True,
        cache_size = 200,
        verbose = False,
        max_iter = -1,
        rho: float = 0.9,
        smoothening_term: float = 1e-06,
        device = 'cpu',
        dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.warm_start = warm_start
        self.average = average
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.rho = rho
        self.smoothening_term = smoothening_term
        self.device = device
        self.dtype = dtype


"""Generated config for AdafactorEstimatorBasedOutlierDetection."""
class AdafactorEstimatorBasedOutlierDetectionConfig(ConfigTemplate):
    model_name = "AdafactorEstimatorBasedOutlierDetection"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        estimator: Optional[MLModule] = None,
        hidden_dim: int = 64,
        bottleneck_dim: int = 16,
        n_layers: int = 2,
        activation: str = 'relu',
        contamination: Union[Literal['auto'], float] = 'auto',
        max_iter: int = 100,
        tol: float = 0.0001,
        eta0: float = 0.001,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
        verbose: bool = False,
        beta1: float = None,
        clip_threshold: float = 1.0,
        decay_rate: float = -0.8,
        eps1: float = 1e-30,
        eps2: float = 0.001,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_layers = n_layers
        self.activation = activation
        self.contamination = contamination
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.beta1 = beta1
        self.clip_threshold = clip_threshold
        self.decay_rate = decay_rate
        self.eps1 = eps1
        self.eps2 = eps2
        self.device = device
        self.dtype = dtype


"""Generated config for AdafactorOneClassSVM."""
class AdafactorOneClassSVMConfig(ConfigTemplate):
    model_name = "AdafactorOneClassSVM"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        kernel = 'rbf',
        kernel_params = None,
        degree = 3,
        gamma = 'scale',
        coef0 = 0.0,
        tol = 0.001,
        fit_intercept = True,
        shuffle = True,
        random_state = None,
        learning_rate = 'invscaling',
        eta0 = 0.01,
        power_t = 0.5,
        warm_start = False,
        average = False,
        nu = 0.5,
        shrinking = True,
        cache_size = 200,
        verbose = False,
        max_iter = -1,
        beta1: float = None,
        clip_threshold: float = 1.0,
        decay_rate: float = -0.8,
        eps1: float = 1e-30,
        eps2: float = 0.001,
        device = 'cpu',
        dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.warm_start = warm_start
        self.average = average
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.beta1 = beta1
        self.clip_threshold = clip_threshold
        self.decay_rate = decay_rate
        self.eps1 = eps1
        self.eps2 = eps2
        self.device = device
        self.dtype = dtype


"""Generated config for AdamEstimatorBasedOutlierDetection."""
class AdamEstimatorBasedOutlierDetectionConfig(ConfigTemplate):
    model_name = "AdamEstimatorBasedOutlierDetection"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        estimator: Optional[MLModule] = None,
        hidden_dim: int = 64,
        bottleneck_dim: int = 16,
        n_layers: int = 2,
        activation: str = 'relu',
        contamination: Union[Literal['auto'], float] = 'auto',
        max_iter: int = 100,
        tol: float = 0.0001,
        eta0: float = 0.001,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
        verbose: bool = False,
        beta1: float = 0.9,
        beta2: float = 0.999,
        smoothening_term: float = 1e-08,
        weight_decay: float = 0.0,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_layers = n_layers
        self.activation = activation
        self.contamination = contamination
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.beta1 = beta1
        self.beta2 = beta2
        self.smoothening_term = smoothening_term
        self.weight_decay = weight_decay
        self.device = device
        self.dtype = dtype


"""Generated config for AdamOneClassSVM."""
class AdamOneClassSVMConfig(ConfigTemplate):
    model_name = "AdamOneClassSVM"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        kernel = 'rbf',
        kernel_params = None,
        degree = 3,
        gamma = 'scale',
        coef0 = 0.0,
        tol = 0.001,
        fit_intercept = True,
        shuffle = True,
        random_state = None,
        learning_rate = 'invscaling',
        eta0 = 0.01,
        power_t = 0.5,
        warm_start = False,
        average = False,
        nu = 0.5,
        shrinking = True,
        cache_size = 200,
        verbose = False,
        max_iter = -1,
        beta1: float = 0.9,
        beta2: float = 0.999,
        smoothening_term: float = 1e-08,
        device = 'cpu',
        dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.warm_start = warm_start
        self.average = average
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.beta1 = beta1
        self.beta2 = beta2
        self.smoothening_term = smoothening_term
        self.device = device
        self.dtype = dtype


"""Generated config for AdamWEstimatorBasedOutlierDetection."""
class AdamWEstimatorBasedOutlierDetectionConfig(ConfigTemplate):
    model_name = "AdamWEstimatorBasedOutlierDetection"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        estimator: Optional[MLModule] = None,
        hidden_dim: int = 64,
        bottleneck_dim: int = 16,
        n_layers: int = 2,
        activation: str = 'relu',
        contamination: Union[Literal['auto'], float] = 'auto',
        max_iter: int = 100,
        tol: float = 0.0001,
        eta0: float = 0.001,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
        verbose: bool = False,
        beta1: float = 0.9,
        beta2: float = 0.999,
        smoothening_term: float = 1e-08,
        weight_decay: float = 0.01,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_layers = n_layers
        self.activation = activation
        self.contamination = contamination
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.beta1 = beta1
        self.beta2 = beta2
        self.smoothening_term = smoothening_term
        self.weight_decay = weight_decay
        self.device = device
        self.dtype = dtype


"""Generated config for AdamWOneClassSVM."""
class AdamWOneClassSVMConfig(ConfigTemplate):
    model_name = "AdamWOneClassSVM"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        kernel = 'rbf',
        kernel_params = None,
        degree = 3,
        gamma = 'scale',
        coef0 = 0.0,
        tol = 0.001,
        fit_intercept = True,
        shuffle = True,
        random_state = None,
        learning_rate = 'invscaling',
        eta0 = 0.01,
        power_t = 0.5,
        warm_start = False,
        average = False,
        nu = 0.5,
        shrinking = True,
        cache_size = 200,
        verbose = False,
        max_iter = -1,
        beta1: float = 0.9,
        beta2: float = 0.999,
        smoothening_term: float = 1e-08,
        weight_decay: float = 0.01,
        device = 'cpu',
        dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.warm_start = warm_start
        self.average = average
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.beta1 = beta1
        self.beta2 = beta2
        self.smoothening_term = smoothening_term
        self.weight_decay = weight_decay
        self.device = device
        self.dtype = dtype


"""Generated config for AdamaxEstimatorBasedOutlierDetection."""
class AdamaxEstimatorBasedOutlierDetectionConfig(ConfigTemplate):
    model_name = "AdamaxEstimatorBasedOutlierDetection"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        estimator: Optional[MLModule] = None,
        hidden_dim: int = 64,
        bottleneck_dim: int = 16,
        n_layers: int = 2,
        activation: str = 'relu',
        contamination: Union[Literal['auto'], float] = 'auto',
        max_iter: int = 100,
        tol: float = 0.0001,
        eta0: float = 0.002,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
        verbose: bool = False,
        beta1: float = 0.9,
        beta2: float = 0.999,
        smoothening_term: float = 1e-08,
        weight_decay: float = 0.0,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_layers = n_layers
        self.activation = activation
        self.contamination = contamination
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.beta1 = beta1
        self.beta2 = beta2
        self.smoothening_term = smoothening_term
        self.weight_decay = weight_decay
        self.device = device
        self.dtype = dtype


"""Generated config for AdanEstimatorBasedOutlierDetection."""
class AdanEstimatorBasedOutlierDetectionConfig(ConfigTemplate):
    model_name = "AdanEstimatorBasedOutlierDetection"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        estimator: Optional[MLModule] = None,
        hidden_dim: int = 64,
        bottleneck_dim: int = 16,
        n_layers: int = 2,
        activation: str = 'relu',
        contamination: Union[Literal['auto'], float] = 'auto',
        max_iter: int = 100,
        tol: float = 0.0001,
        eta0: float = 0.001,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
        verbose: bool = False,
        beta1: float = 0.98,
        beta2: float = 0.92,
        beta3: float = 0.99,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_layers = n_layers
        self.activation = activation
        self.contamination = contamination
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.device = device
        self.dtype = dtype


"""Generated config for AdanOneClassSVM."""
class AdanOneClassSVMConfig(ConfigTemplate):
    model_name = "AdanOneClassSVM"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        kernel = 'rbf',
        kernel_params = None,
        degree = 3,
        gamma = 'scale',
        coef0 = 0.0,
        tol = 0.001,
        fit_intercept = True,
        shuffle = True,
        random_state = None,
        learning_rate = 'invscaling',
        eta0 = 0.01,
        power_t = 0.5,
        warm_start = False,
        average = False,
        nu = 0.5,
        shrinking = True,
        cache_size = 200,
        verbose = False,
        max_iter = -1,
        beta1: float = 0.98,
        beta2: float = 0.92,
        beta3: float = 0.99,
        device = 'cpu',
        dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.warm_start = warm_start
        self.average = average
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.device = device
        self.dtype = dtype


"""Generated config for Autoencoder."""
class AutoencoderConfig(ConfigTemplate):
    model_name = "Autoencoder"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        encoder_layers = None,
        decoder_layers = None,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers


"""Generated config for BFGSEstimatorBasedOutlierDetection."""
class BFGSEstimatorBasedOutlierDetectionConfig(ConfigTemplate):
    model_name = "BFGSEstimatorBasedOutlierDetection"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        estimator: Optional[MLModule] = None,
        hidden_dim: int = 64,
        bottleneck_dim: int = 16,
        n_layers: int = 2,
        activation: str = 'relu',
        contamination: Union[Literal['auto'], float] = 'auto',
        max_iter: int = 100,
        tol: float = 0.0001,
        eta0: float = 1.0,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
        verbose: bool = False,
        history_size: int = 100,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_layers = n_layers
        self.activation = activation
        self.contamination = contamination
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.history_size = history_size
        self.device = device
        self.dtype = dtype


"""Generated config for BGDEstimatorBasedOutlierDetection."""
class BGDEstimatorBasedOutlierDetectionConfig(ConfigTemplate):
    model_name = "BGDEstimatorBasedOutlierDetection"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        estimator: Optional[MLModule] = None,
        hidden_dim: int = 64,
        bottleneck_dim: int = 16,
        n_layers: int = 2,
        activation: str = 'relu',
        contamination: Union[Literal['auto'], float] = 'auto',
        max_iter: int = 100,
        tol: float = 0.0001,
        eta0: float = 0.001,
        batch_size: int = 10000,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
        verbose: bool = False,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_layers = n_layers
        self.activation = activation
        self.contamination = contamination
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.device = device
        self.dtype = dtype


"""Generated config for BGDOneClassSVM."""
class BGDOneClassSVMConfig(ConfigTemplate):
    model_name = "BGDOneClassSVM"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        kernel = 'rbf',
        kernel_params = None,
        degree = 3,
        gamma = 'scale',
        coef0 = 0.0,
        tol = 0.001,
        fit_intercept = True,
        shuffle = True,
        random_state = None,
        learning_rate = 'invscaling',
        eta0 = 0.01,
        power_t = 0.5,
        warm_start = False,
        average = False,
        nu = 0.5,
        shrinking = True,
        cache_size = 200,
        verbose = False,
        max_iter = -1,
        batch_size: int = 32,
        device = 'cpu',
        dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.warm_start = warm_start
        self.average = average
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype


"""Generated config for BayesianOptimizationEstimatorBasedOutlierDetection."""
class BayesianOptimizationEstimatorBasedOutlierDetectionConfig(ConfigTemplate):
    model_name = "BayesianOptimizationEstimatorBasedOutlierDetection"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        estimator: Optional[MLModule] = None,
        hidden_dim: int = 64,
        bottleneck_dim: int = 16,
        n_layers: int = 2,
        activation: str = 'relu',
        contamination: Union[Literal['auto'], float] = 'auto',
        max_iter: int = 100,
        tol: float = 0.0001,
        eta0: float = 0.001,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
        verbose: bool = False,
        n_calls: int = 20,
        n_random_starts: int = 5,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_layers = n_layers
        self.activation = activation
        self.contamination = contamination
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.n_calls = n_calls
        self.n_random_starts = n_random_starts
        self.device = device
        self.dtype = dtype


"""Generated config for BayesianOptimizationOneClassSVM."""
class BayesianOptimizationOneClassSVMConfig(ConfigTemplate):
    model_name = "BayesianOptimizationOneClassSVM"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        kernel = 'rbf',
        kernel_params = None,
        degree = 3,
        gamma = 'scale',
        coef0 = 0.0,
        tol = 0.001,
        fit_intercept = True,
        shuffle = True,
        random_state = None,
        learning_rate = 'constant',
        eta0 = 0.01,
        power_t = 0.5,
        warm_start = False,
        average = False,
        nu = 0.5,
        shrinking = True,
        cache_size = 200,
        verbose = False,
        max_iter = -1,
        n_calls: int = 20,
        n_random_starts: int = 5,
        device = 'cpu',
        dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.warm_start = warm_start
        self.average = average
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.n_calls = n_calls
        self.n_random_starts = n_random_starts
        self.device = device
        self.dtype = dtype


"""Generated config for CMAESEstimatorBasedOutlierDetection."""
class CMAESEstimatorBasedOutlierDetectionConfig(ConfigTemplate):
    model_name = "CMAESEstimatorBasedOutlierDetection"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        estimator: Optional[MLModule] = None,
        hidden_dim: int = 64,
        bottleneck_dim: int = 16,
        n_layers: int = 2,
        activation: str = 'relu',
        contamination: Union[Literal['auto'], float] = 'auto',
        max_iter: int = 100,
        tol: float = 0.0001,
        eta0: float = 0.001,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
        verbose: bool = False,
        sigma0: float = 0.3,
        popsize: int = 10,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_layers = n_layers
        self.activation = activation
        self.contamination = contamination
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.sigma0 = sigma0
        self.popsize = popsize
        self.device = device
        self.dtype = dtype


"""Generated config for CMAESOneClassSVM."""
class CMAESOneClassSVMConfig(ConfigTemplate):
    model_name = "CMAESOneClassSVM"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        kernel = 'rbf',
        kernel_params = None,
        degree = 3,
        gamma = 'scale',
        coef0 = 0.0,
        tol = 0.001,
        fit_intercept = True,
        shuffle = True,
        random_state = None,
        learning_rate = 'constant',
        eta0 = 0.01,
        power_t = 0.5,
        warm_start = False,
        average = False,
        nu = 0.5,
        shrinking = True,
        cache_size = 200,
        verbose = False,
        max_iter = -1,
        sigma0: float = 0.3,
        popsize: int = 10,
        device = 'cpu',
        dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.warm_start = warm_start
        self.average = average
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.sigma0 = sigma0
        self.popsize = popsize
        self.device = device
        self.dtype = dtype


"""Generated config for DAdaptationEstimatorBasedOutlierDetection."""
class DAdaptationEstimatorBasedOutlierDetectionConfig(ConfigTemplate):
    model_name = "DAdaptationEstimatorBasedOutlierDetection"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        estimator: Optional[MLModule] = None,
        hidden_dim: int = 64,
        bottleneck_dim: int = 16,
        n_layers: int = 2,
        activation: str = 'relu',
        contamination: Union[Literal['auto'], float] = 'auto',
        max_iter: int = 100,
        tol: float = 0.0001,
        eta0: float = 1.0,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
        verbose: bool = False,
        growth_rate: float = float('inf'),
        decouple_lr: bool = False,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_layers = n_layers
        self.activation = activation
        self.contamination = contamination
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.growth_rate = growth_rate
        self.decouple_lr = decouple_lr
        self.device = device
        self.dtype = dtype


"""Generated config for DAdaptationOneClassSVM."""
class DAdaptationOneClassSVMConfig(ConfigTemplate):
    model_name = "DAdaptationOneClassSVM"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        kernel = 'rbf',
        kernel_params = None,
        degree = 3,
        gamma = 'scale',
        coef0 = 0.0,
        tol = 0.001,
        fit_intercept = True,
        shuffle = True,
        random_state = None,
        learning_rate = 'constant',
        eta0 = 1.0,
        power_t = 0.5,
        warm_start = False,
        average = False,
        nu = 0.5,
        shrinking = True,
        cache_size = 200,
        verbose = False,
        max_iter = -1,
        growth_rate: float = float('inf'),
        decouple_lr: bool = False,
        device = 'cpu',
        dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.warm_start = warm_start
        self.average = average
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.growth_rate = growth_rate
        self.decouple_lr = decouple_lr
        self.device = device
        self.dtype = dtype


"""Generated config for FireflyEstimatorBasedOutlierDetection."""
class FireflyEstimatorBasedOutlierDetectionConfig(ConfigTemplate):
    model_name = "FireflyEstimatorBasedOutlierDetection"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        estimator: Optional[MLModule] = None,
        hidden_dim: int = 64,
        bottleneck_dim: int = 16,
        n_layers: int = 2,
        activation: str = 'relu',
        contamination: Union[Literal['auto'], float] = 'auto',
        max_iter: int = 100,
        tol: float = 0.0001,
        eta0: float = 0.001,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
        verbose: bool = False,
        n_fireflies: int = 20,
        alpha_firefly: float = 0.2,
        beta_min: float = 0.2,
        gamma_firefly: float = 1.0,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_layers = n_layers
        self.activation = activation
        self.contamination = contamination
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.n_fireflies = n_fireflies
        self.alpha_firefly = alpha_firefly
        self.beta_min = beta_min
        self.gamma_firefly = gamma_firefly
        self.device = device
        self.dtype = dtype


"""Generated config for FireflyOneClassSVM."""
class FireflyOneClassSVMConfig(ConfigTemplate):
    model_name = "FireflyOneClassSVM"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        kernel = 'rbf',
        kernel_params = None,
        degree = 3,
        gamma = 'scale',
        coef0 = 0.0,
        tol = 0.001,
        fit_intercept = True,
        shuffle = True,
        random_state = None,
        learning_rate = 'constant',
        eta0 = 0.01,
        power_t = 0.5,
        warm_start = False,
        average = False,
        nu = 0.5,
        shrinking = True,
        cache_size = 200,
        verbose = False,
        max_iter = -1,
        n_fireflies: int = 20,
        alpha_firefly: float = 0.2,
        beta_min: float = 0.2,
        gamma_firefly: float = 1.0,
        device = 'cpu',
        dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.warm_start = warm_start
        self.average = average
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.n_fireflies = n_fireflies
        self.alpha_firefly = alpha_firefly
        self.beta_min = beta_min
        self.gamma_firefly = gamma_firefly
        self.device = device
        self.dtype = dtype


"""Generated config for IsolationForest."""
class IsolationForestConfig(ConfigTemplate):
    model_name = "IsolationForest"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        n_estimators: int = 100,
        max_samples: Union[Literal['auto'], int, float] = 'auto',
        contamination: Union[Literal['auto'], float] = 'auto',
        max_features: Union[int, float] = 1.0,
        bootstrap: bool = False,
        n_jobs: int = None,
        random_state: Union[int, torch.Generator] = None,
        verbose: int = 0,
        warm_start: bool = False,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.device = device
        self.dtype = dtype


"""Generated config for LAMBEstimatorBasedOutlierDetection."""
class LAMBEstimatorBasedOutlierDetectionConfig(ConfigTemplate):
    model_name = "LAMBEstimatorBasedOutlierDetection"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        estimator: Optional[MLModule] = None,
        hidden_dim: int = 64,
        bottleneck_dim: int = 16,
        n_layers: int = 2,
        activation: str = 'relu',
        contamination: Union[Literal['auto'], float] = 'auto',
        max_iter: int = 100,
        tol: float = 0.0001,
        eta0: float = 0.001,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
        verbose: bool = False,
        beta1: float = 0.9,
        beta2: float = 0.999,
        smoothening_term: float = 1e-06,
        weight_decay: float = 0.01,
        trust_ratio: float = 1.0,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_layers = n_layers
        self.activation = activation
        self.contamination = contamination
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.beta1 = beta1
        self.beta2 = beta2
        self.smoothening_term = smoothening_term
        self.weight_decay = weight_decay
        self.trust_ratio = trust_ratio
        self.device = device
        self.dtype = dtype


"""Generated config for LAMBOneClassSVM."""
class LAMBOneClassSVMConfig(ConfigTemplate):
    model_name = "LAMBOneClassSVM"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        kernel = 'rbf',
        kernel_params = None,
        degree = 3,
        gamma = 'scale',
        coef0 = 0.0,
        tol = 0.001,
        fit_intercept = True,
        shuffle = True,
        random_state = None,
        learning_rate = 'invscaling',
        eta0 = 0.01,
        power_t = 0.5,
        warm_start = False,
        average = False,
        nu = 0.5,
        shrinking = True,
        cache_size = 200,
        verbose = False,
        max_iter = -1,
        beta1: float = 0.9,
        beta2: float = 0.999,
        smoothening_term: float = 1e-06,
        weight_decay: float = 0.01,
        trust_ratio: float = 1.0,
        device = 'cpu',
        dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.warm_start = warm_start
        self.average = average
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.beta1 = beta1
        self.beta2 = beta2
        self.smoothening_term = smoothening_term
        self.weight_decay = weight_decay
        self.trust_ratio = trust_ratio
        self.device = device
        self.dtype = dtype


"""Generated config for LARSEstimatorBasedOutlierDetection."""
class LARSEstimatorBasedOutlierDetectionConfig(ConfigTemplate):
    model_name = "LARSEstimatorBasedOutlierDetection"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        estimator: Optional[MLModule] = None,
        hidden_dim: int = 64,
        bottleneck_dim: int = 16,
        n_layers: int = 2,
        activation: str = 'relu',
        contamination: Union[Literal['auto'], float] = 'auto',
        max_iter: int = 100,
        tol: float = 0.0001,
        eta0: float = 0.001,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
        verbose: bool = False,
        weight_decay: float = 0.0,
        eta_lars: float = 0.001,
        momentum_lars: float = 0.9,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_layers = n_layers
        self.activation = activation
        self.contamination = contamination
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.weight_decay = weight_decay
        self.eta_lars = eta_lars
        self.momentum_lars = momentum_lars
        self.device = device
        self.dtype = dtype


"""Generated config for LARSOneClassSVM."""
class LARSOneClassSVMConfig(ConfigTemplate):
    model_name = "LARSOneClassSVM"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        kernel = 'rbf',
        kernel_params = None,
        degree = 3,
        gamma = 'scale',
        coef0 = 0.0,
        tol = 0.001,
        fit_intercept = True,
        shuffle = True,
        random_state = None,
        learning_rate = 'constant',
        eta0 = 0.01,
        power_t = 0.5,
        warm_start = False,
        average = False,
        nu = 0.5,
        shrinking = True,
        cache_size = 200,
        verbose = False,
        max_iter = -1,
        weight_decay: float = 0.0,
        eta_lars: float = 0.001,
        momentum_lars: float = 0.9,
        device = 'cpu',
        dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.warm_start = warm_start
        self.average = average
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.weight_decay = weight_decay
        self.eta_lars = eta_lars
        self.momentum_lars = momentum_lars
        self.device = device
        self.dtype = dtype


"""Generated config for LBFGSEstimatorBasedOutlierDetection."""
class LBFGSEstimatorBasedOutlierDetectionConfig(ConfigTemplate):
    model_name = "LBFGSEstimatorBasedOutlierDetection"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        estimator: Optional[MLModule] = None,
        hidden_dim: int = 64,
        bottleneck_dim: int = 16,
        n_layers: int = 2,
        activation: str = 'relu',
        contamination: Union[Literal['auto'], float] = 'auto',
        max_iter: int = 100,
        tol: float = 0.0001,
        eta0: float = 1.0,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
        verbose: bool = False,
        history_size: int = 10,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_layers = n_layers
        self.activation = activation
        self.contamination = contamination
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.history_size = history_size
        self.device = device
        self.dtype = dtype


"""Generated config for LionEstimatorBasedOutlierDetection."""
class LionEstimatorBasedOutlierDetectionConfig(ConfigTemplate):
    model_name = "LionEstimatorBasedOutlierDetection"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        estimator: Optional[MLModule] = None,
        hidden_dim: int = 64,
        bottleneck_dim: int = 16,
        n_layers: int = 2,
        activation: str = 'relu',
        contamination: Union[Literal['auto'], float] = 'auto',
        max_iter: int = 100,
        tol: float = 0.0001,
        eta0: float = 0.0001,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
        verbose: bool = False,
        beta1: float = 0.9,
        beta2: float = 0.99,
        weight_decay: float = 0.0,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_layers = n_layers
        self.activation = activation
        self.contamination = contamination
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.device = device
        self.dtype = dtype


"""Generated config for LionOneClassSVM."""
class LionOneClassSVMConfig(ConfigTemplate):
    model_name = "LionOneClassSVM"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        kernel = 'rbf',
        kernel_params = None,
        degree = 3,
        gamma = 'scale',
        coef0 = 0.0,
        tol = 0.001,
        fit_intercept = True,
        shuffle = True,
        random_state = None,
        learning_rate = 'invscaling',
        eta0 = 0.01,
        power_t = 0.5,
        warm_start = False,
        average = False,
        nu = 0.5,
        shrinking = True,
        cache_size = 200,
        verbose = False,
        max_iter = -1,
        beta1: float = 0.9,
        beta2: float = 0.99,
        device = 'cpu',
        dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.warm_start = warm_start
        self.average = average
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.beta1 = beta1
        self.beta2 = beta2
        self.device = device
        self.dtype = dtype


"""Generated config for LocalOutlierFactor."""
class LocalOutlierFactorConfig(ConfigTemplate):
    model_name = "LocalOutlierFactor"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        n_neighbors: int = 20,
        algorithm: Union[Literal['auto', 'ball_tree', 'kd_tree', 'brute'], Callable, nn.Module] = 'auto',
        leaf_size: int = 30,
        metric: Union[str, Callable, nn.Module] = 'minkowski',
        p: float = 2,
        metric_params: dict = None,
        contamination: Union[Literal['auto'], float] = 'auto',
        novelty: bool = False,
        n_jobs: int = None,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.contamination = contamination
        self.novelty = novelty
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype


"""Generated config for LookaheadEstimatorBasedOutlierDetection."""
class LookaheadEstimatorBasedOutlierDetectionConfig(ConfigTemplate):
    model_name = "LookaheadEstimatorBasedOutlierDetection"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        estimator: Optional[MLModule] = None,
        hidden_dim: int = 64,
        bottleneck_dim: int = 16,
        n_layers: int = 2,
        activation: str = 'relu',
        contamination: Union[Literal['auto'], float] = 'auto',
        max_iter: int = 100,
        tol: float = 0.0001,
        eta0: float = 0.001,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
        verbose: bool = False,
        k: int = 5,
        alpha_la: float = 0.5,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_layers = n_layers
        self.activation = activation
        self.contamination = contamination
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.k = k
        self.alpha_la = alpha_la
        self.device = device
        self.dtype = dtype


"""Generated config for LookaheadOneClassSVM."""
class LookaheadOneClassSVMConfig(ConfigTemplate):
    model_name = "LookaheadOneClassSVM"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        kernel = 'rbf',
        kernel_params = None,
        degree = 3,
        gamma = 'scale',
        coef0 = 0.0,
        tol = 0.001,
        fit_intercept = True,
        shuffle = True,
        random_state = None,
        learning_rate = 'invscaling',
        eta0 = 0.01,
        power_t = 0.5,
        warm_start = False,
        average = False,
        nu = 0.5,
        shrinking = True,
        cache_size = 200,
        verbose = False,
        max_iter = -1,
        k: int = 5,
        alpha_la: float = 0.5,
        device = 'cpu',
        dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.warm_start = warm_start
        self.average = average
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.k = k
        self.alpha_la = alpha_la
        self.device = device
        self.dtype = dtype


"""Generated config for MARSEstimatorBasedOutlierDetection."""
class MARSEstimatorBasedOutlierDetectionConfig(ConfigTemplate):
    model_name = "MARSEstimatorBasedOutlierDetection"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        estimator: Optional[MLModule] = None,
        hidden_dim: int = 64,
        bottleneck_dim: int = 16,
        n_layers: int = 2,
        activation: str = 'relu',
        contamination: Union[Literal['auto'], float] = 'auto',
        max_iter: int = 100,
        tol: float = 0.0001,
        eta0: float = 0.001,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
        verbose: bool = False,
        beta1: float = 0.9,
        beta2: float = 0.99,
        gamma_mars: float = 0.025,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_layers = n_layers
        self.activation = activation
        self.contamination = contamination
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.beta1 = beta1
        self.beta2 = beta2
        self.gamma_mars = gamma_mars
        self.device = device
        self.dtype = dtype


"""Generated config for MARSOneClassSVM."""
class MARSOneClassSVMConfig(ConfigTemplate):
    model_name = "MARSOneClassSVM"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        kernel = 'rbf',
        kernel_params = None,
        degree = 3,
        gamma = 'scale',
        coef0 = 0.0,
        tol = 0.001,
        fit_intercept = True,
        shuffle = True,
        random_state = None,
        learning_rate = 'invscaling',
        eta0 = 0.01,
        power_t = 0.5,
        warm_start = False,
        average = False,
        nu = 0.5,
        shrinking = True,
        cache_size = 200,
        verbose = False,
        max_iter = -1,
        beta1: float = 0.9,
        beta2: float = 0.99,
        smoothening_term: float = 1e-08,
        gamma_mars: float = 0.025,
        device = 'cpu',
        dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.warm_start = warm_start
        self.average = average
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.beta1 = beta1
        self.beta2 = beta2
        self.smoothening_term = smoothening_term
        self.gamma_mars = gamma_mars
        self.device = device
        self.dtype = dtype


"""Generated config for MGDEstimatorBasedOutlierDetection."""
class MGDEstimatorBasedOutlierDetectionConfig(ConfigTemplate):
    model_name = "MGDEstimatorBasedOutlierDetection"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        estimator: Optional[MLModule] = None,
        hidden_dim: int = 64,
        bottleneck_dim: int = 16,
        n_layers: int = 2,
        activation: str = 'relu',
        contamination: Union[Literal['auto'], float] = 'auto',
        max_iter: int = 100,
        tol: float = 0.0001,
        eta0: float = 0.001,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
        verbose: bool = False,
        gamma_momentum: float = 0.9,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_layers = n_layers
        self.activation = activation
        self.contamination = contamination
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.gamma_momentum = gamma_momentum
        self.device = device
        self.dtype = dtype


"""Generated config for MGDOneClassSVM."""
class MGDOneClassSVMConfig(ConfigTemplate):
    model_name = "MGDOneClassSVM"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        kernel = 'rbf',
        kernel_params = None,
        degree = 3,
        gamma = 'scale',
        coef0 = 0.0,
        tol = 0.001,
        fit_intercept = True,
        shuffle = True,
        random_state = None,
        learning_rate = 'invscaling',
        eta0 = 0.01,
        power_t = 0.5,
        warm_start = False,
        average = False,
        nu = 0.5,
        shrinking = True,
        cache_size = 200,
        verbose = False,
        max_iter = -1,
        gamma_momentum: float = 0.9,
        device = 'cpu',
        dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.warm_start = warm_start
        self.average = average
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.gamma_momentum = gamma_momentum
        self.device = device
        self.dtype = dtype


"""Generated config for MadgradEstimatorBasedOutlierDetection."""
class MadgradEstimatorBasedOutlierDetectionConfig(ConfigTemplate):
    model_name = "MadgradEstimatorBasedOutlierDetection"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        estimator: Optional[MLModule] = None,
        hidden_dim: int = 64,
        bottleneck_dim: int = 16,
        n_layers: int = 2,
        activation: str = 'relu',
        contamination: Union[Literal['auto'], float] = 'auto',
        max_iter: int = 100,
        tol: float = 0.0001,
        eta0: float = 0.01,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
        verbose: bool = False,
        eps: float = 1e-06,
        momentum_madgrad: float = 0.9,
        weight_decay: float = 0.0,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_layers = n_layers
        self.activation = activation
        self.contamination = contamination
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.eps = eps
        self.momentum_madgrad = momentum_madgrad
        self.weight_decay = weight_decay
        self.device = device
        self.dtype = dtype


"""Generated config for MadgradOneClassSVM."""
class MadgradOneClassSVMConfig(ConfigTemplate):
    model_name = "MadgradOneClassSVM"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        kernel = 'rbf',
        kernel_params = None,
        degree = 3,
        gamma = 'scale',
        coef0 = 0.0,
        tol = 0.001,
        fit_intercept = True,
        shuffle = True,
        random_state = None,
        learning_rate = 'constant',
        eta0 = 0.01,
        power_t = 0.5,
        warm_start = False,
        average = False,
        nu = 0.5,
        shrinking = True,
        cache_size = 200,
        verbose = False,
        max_iter = -1,
        eps: float = 1e-06,
        momentum_madgrad: float = 0.9,
        weight_decay: float = 0.0,
        device = 'cpu',
        dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.warm_start = warm_start
        self.average = average
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.eps = eps
        self.momentum_madgrad = momentum_madgrad
        self.weight_decay = weight_decay
        self.device = device
        self.dtype = dtype


"""Generated config for MuonEstimatorBasedOutlierDetection."""
class MuonEstimatorBasedOutlierDetectionConfig(ConfigTemplate):
    model_name = "MuonEstimatorBasedOutlierDetection"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        estimator: Optional[MLModule] = None,
        hidden_dim: int = 64,
        bottleneck_dim: int = 16,
        n_layers: int = 2,
        activation: str = 'relu',
        contamination: Union[Literal['auto'], float] = 'auto',
        max_iter: int = 100,
        tol: float = 0.0001,
        eta0: float = 0.001,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
        verbose: bool = False,
        momentum: float = 0.9,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_layers = n_layers
        self.activation = activation
        self.contamination = contamination
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.momentum = momentum
        self.device = device
        self.dtype = dtype


"""Generated config for MuonOneClassSVM."""
class MuonOneClassSVMConfig(ConfigTemplate):
    model_name = "MuonOneClassSVM"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        kernel = 'rbf',
        kernel_params = None,
        degree = 3,
        gamma = 'scale',
        coef0 = 0.0,
        tol = 0.001,
        fit_intercept = True,
        shuffle = True,
        random_state = None,
        learning_rate = 'invscaling',
        eta0 = 0.01,
        power_t = 0.5,
        warm_start = False,
        average = False,
        nu = 0.5,
        shrinking = True,
        cache_size = 200,
        verbose = False,
        max_iter = -1,
        momentum: float = 0.9,
        device = 'cpu',
        dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.warm_start = warm_start
        self.average = average
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.momentum = momentum
        self.device = device
        self.dtype = dtype


"""Generated config for NAdamEstimatorBasedOutlierDetection."""
class NAdamEstimatorBasedOutlierDetectionConfig(ConfigTemplate):
    model_name = "NAdamEstimatorBasedOutlierDetection"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        estimator: Optional[MLModule] = None,
        hidden_dim: int = 64,
        bottleneck_dim: int = 16,
        n_layers: int = 2,
        activation: str = 'relu',
        contamination: Union[Literal['auto'], float] = 'auto',
        max_iter: int = 100,
        tol: float = 0.0001,
        eta0: float = 0.002,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
        verbose: bool = False,
        beta1: float = 0.9,
        beta2: float = 0.999,
        smoothening_term: float = 1e-08,
        weight_decay: float = 0.0,
        momentum_decay: float = 0.004,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_layers = n_layers
        self.activation = activation
        self.contamination = contamination
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.beta1 = beta1
        self.beta2 = beta2
        self.smoothening_term = smoothening_term
        self.weight_decay = weight_decay
        self.momentum_decay = momentum_decay
        self.device = device
        self.dtype = dtype


"""Generated config for OneClassSVM."""
class OneClassSVMConfig(ConfigTemplate):
    model_name = "OneClassSVM"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        kernel: Union[str, Callable, nn.Module] = 'rbf',
        kernel_params: dict = None,
        degree: int = 3,
        gamma: Union[Literal['scale', 'auto'], float] = 'scale',
        coef0: float = 0.0,
        tol: float = 0.001,
        nu: float = 0.5,
        shrinking: bool = True,
        cache_size: float = 200,
        verbose: bool = False,
        max_iter: int = -1,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.device = device
        self.dtype = dtype


"""Generated config for PSOEstimatorBasedOutlierDetection."""
class PSOEstimatorBasedOutlierDetectionConfig(ConfigTemplate):
    model_name = "PSOEstimatorBasedOutlierDetection"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        estimator: Optional[MLModule] = None,
        hidden_dim: int = 64,
        bottleneck_dim: int = 16,
        n_layers: int = 2,
        activation: str = 'relu',
        contamination: Union[Literal['auto'], float] = 'auto',
        max_iter: int = 100,
        tol: float = 0.0001,
        eta0: float = 0.001,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
        verbose: bool = False,
        n_particles: int = 20,
        inertia: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_layers = n_layers
        self.activation = activation
        self.contamination = contamination
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.n_particles = n_particles
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2
        self.device = device
        self.dtype = dtype


"""Generated config for PSOOneClassSVM."""
class PSOOneClassSVMConfig(ConfigTemplate):
    model_name = "PSOOneClassSVM"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        kernel = 'rbf',
        kernel_params = None,
        degree = 3,
        gamma = 'scale',
        coef0 = 0.0,
        tol = 0.001,
        fit_intercept = True,
        shuffle = True,
        random_state = None,
        learning_rate = 'constant',
        eta0 = 0.01,
        power_t = 0.5,
        warm_start = False,
        average = False,
        nu = 0.5,
        shrinking = True,
        cache_size = 200,
        verbose = False,
        max_iter = -1,
        n_particles: int = 20,
        inertia: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        device = 'cpu',
        dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.warm_start = warm_start
        self.average = average
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.n_particles = n_particles
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2
        self.device = device
        self.dtype = dtype


"""Generated config for PassiveAggressiveEstimatorBasedOutlierDetection."""
class PassiveAggressiveEstimatorBasedOutlierDetectionConfig(ConfigTemplate):
    model_name = "PassiveAggressiveEstimatorBasedOutlierDetection"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        estimator: Optional[MLModule] = None,
        hidden_dim: int = 64,
        bottleneck_dim: int = 16,
        n_layers: int = 2,
        activation: str = 'relu',
        contamination: Union[Literal['auto'], float] = 'auto',
        max_iter: int = 100,
        tol: float = 0.0001,
        eta0: float = 0.001,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
        verbose: bool = False,
        C: float = 1.0,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_layers = n_layers
        self.activation = activation
        self.contamination = contamination
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.C = C
        self.device = device
        self.dtype = dtype


"""Generated config for PassiveAggressiveOneClassSVM."""
class PassiveAggressiveOneClassSVMConfig(ConfigTemplate):
    model_name = "PassiveAggressiveOneClassSVM"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        kernel = 'rbf',
        kernel_params = None,
        degree = 3,
        gamma = 'scale',
        coef0 = 0.0,
        tol = 0.001,
        fit_intercept = True,
        shuffle = True,
        random_state = None,
        learning_rate = 'optimal',
        eta0 = 0.01,
        power_t = 0.5,
        warm_start = False,
        average = False,
        nu = 0.5,
        shrinking = True,
        cache_size = 200,
        verbose = False,
        max_iter = -1,
        C: float = 1.0,
        device = 'cpu',
        dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.warm_start = warm_start
        self.average = average
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.C = C
        self.device = device
        self.dtype = dtype


"""Generated config for ProdigyEstimatorBasedOutlierDetection."""
class ProdigyEstimatorBasedOutlierDetectionConfig(ConfigTemplate):
    model_name = "ProdigyEstimatorBasedOutlierDetection"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        estimator: Optional[MLModule] = None,
        hidden_dim: int = 64,
        bottleneck_dim: int = 16,
        n_layers: int = 2,
        activation: str = 'relu',
        contamination: Union[Literal['auto'], float] = 'auto',
        max_iter: int = 100,
        tol: float = 0.0001,
        eta0: float = 1.0,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
        verbose: bool = False,
        beta1: float = 0.9,
        beta2: float = 0.999,
        growth_rate: float = float('inf'),
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_layers = n_layers
        self.activation = activation
        self.contamination = contamination
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.beta1 = beta1
        self.beta2 = beta2
        self.growth_rate = growth_rate
        self.device = device
        self.dtype = dtype


"""Generated config for ProdigyOneClassSVM."""
class ProdigyOneClassSVMConfig(ConfigTemplate):
    model_name = "ProdigyOneClassSVM"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        kernel = 'rbf',
        kernel_params = None,
        degree = 3,
        gamma = 'scale',
        coef0 = 0.0,
        tol = 0.001,
        fit_intercept = True,
        shuffle = True,
        random_state = None,
        learning_rate = 'constant',
        eta0 = 1.0,
        power_t = 0.5,
        warm_start = False,
        average = False,
        nu = 0.5,
        shrinking = True,
        cache_size = 200,
        verbose = False,
        max_iter = -1,
        beta1: float = 0.9,
        beta2: float = 0.999,
        smoothening_term: float = 1e-08,
        growth_rate: float = float('inf'),
        device = 'cpu',
        dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.warm_start = warm_start
        self.average = average
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.beta1 = beta1
        self.beta2 = beta2
        self.smoothening_term = smoothening_term
        self.growth_rate = growth_rate
        self.device = device
        self.dtype = dtype


"""Generated config for QNSVRGEstimatorBasedOutlierDetection."""
class QNSVRGEstimatorBasedOutlierDetectionConfig(ConfigTemplate):
    model_name = "QNSVRGEstimatorBasedOutlierDetection"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        estimator: Optional[MLModule] = None,
        hidden_dim: int = 64,
        bottleneck_dim: int = 16,
        n_layers: int = 2,
        activation: str = 'relu',
        contamination: Union[Literal['auto'], float] = 'auto',
        max_iter: int = 100,
        tol: float = 0.0001,
        eta0: float = 1.0,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
        verbose: bool = False,
        memory: int = 10,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_layers = n_layers
        self.activation = activation
        self.contamination = contamination
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.memory = memory
        self.device = device
        self.dtype = dtype


"""Generated config for QNSVRGOneClassSVM."""
class QNSVRGOneClassSVMConfig(ConfigTemplate):
    model_name = "QNSVRGOneClassSVM"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        kernel = 'rbf',
        kernel_params = None,
        degree = 3,
        gamma = 'scale',
        coef0 = 0.0,
        tol = 0.001,
        fit_intercept = True,
        shuffle = True,
        random_state = None,
        learning_rate = 'constant',
        eta0 = 0.01,
        power_t = 0.5,
        warm_start = False,
        average = False,
        nu = 0.5,
        shrinking = True,
        cache_size = 200,
        verbose = False,
        max_iter = -1,
        memory: int = 10,
        device = 'cpu',
        dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.warm_start = warm_start
        self.average = average
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.memory = memory
        self.device = device
        self.dtype = dtype


"""Generated config for RAdamEstimatorBasedOutlierDetection."""
class RAdamEstimatorBasedOutlierDetectionConfig(ConfigTemplate):
    model_name = "RAdamEstimatorBasedOutlierDetection"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        estimator: Optional[MLModule] = None,
        hidden_dim: int = 64,
        bottleneck_dim: int = 16,
        n_layers: int = 2,
        activation: str = 'relu',
        contamination: Union[Literal['auto'], float] = 'auto',
        max_iter: int = 100,
        tol: float = 0.0001,
        eta0: float = 0.001,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
        verbose: bool = False,
        beta1: float = 0.9,
        beta2: float = 0.999,
        smoothening_term: float = 1e-08,
        weight_decay: float = 0.0,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_layers = n_layers
        self.activation = activation
        self.contamination = contamination
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.beta1 = beta1
        self.beta2 = beta2
        self.smoothening_term = smoothening_term
        self.weight_decay = weight_decay
        self.device = device
        self.dtype = dtype


"""Generated config for RMSPropEstimatorBasedOutlierDetection."""
class RMSPropEstimatorBasedOutlierDetectionConfig(ConfigTemplate):
    model_name = "RMSPropEstimatorBasedOutlierDetection"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        estimator: Optional[MLModule] = None,
        hidden_dim: int = 64,
        bottleneck_dim: int = 16,
        n_layers: int = 2,
        activation: str = 'relu',
        contamination: Union[Literal['auto'], float] = 'auto',
        max_iter: int = 100,
        tol: float = 0.0001,
        eta0: float = 0.001,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
        verbose: bool = False,
        decay_rate: float = 0.99,
        smoothening_term: float = 1e-08,
        momentum: float = 0.0,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_layers = n_layers
        self.activation = activation
        self.contamination = contamination
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.decay_rate = decay_rate
        self.smoothening_term = smoothening_term
        self.momentum = momentum
        self.device = device
        self.dtype = dtype


"""Generated config for RMSPropOneClassSVM."""
class RMSPropOneClassSVMConfig(ConfigTemplate):
    model_name = "RMSPropOneClassSVM"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        kernel = 'rbf',
        kernel_params = None,
        degree = 3,
        gamma = 'scale',
        coef0 = 0.0,
        tol = 0.001,
        fit_intercept = True,
        shuffle = True,
        random_state = None,
        learning_rate = 'invscaling',
        eta0 = 0.01,
        power_t = 0.5,
        warm_start = False,
        average = False,
        nu = 0.5,
        shrinking = True,
        cache_size = 200,
        verbose = False,
        max_iter = -1,
        smoothening_term: float = 1e-08,
        decay_rate: float = 0.9,
        device = 'cpu',
        dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.warm_start = warm_start
        self.average = average
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.smoothening_term = smoothening_term
        self.decay_rate = decay_rate
        self.device = device
        self.dtype = dtype


"""Generated config for RpropEstimatorBasedOutlierDetection."""
class RpropEstimatorBasedOutlierDetectionConfig(ConfigTemplate):
    model_name = "RpropEstimatorBasedOutlierDetection"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        estimator: Optional[MLModule] = None,
        hidden_dim: int = 64,
        bottleneck_dim: int = 16,
        n_layers: int = 2,
        activation: str = 'relu',
        contamination: Union[Literal['auto'], float] = 'auto',
        max_iter: int = 100,
        tol: float = 0.0001,
        eta0: float = 0.01,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
        verbose: bool = False,
        etas: Tuple[float, float] = (0.5, 1.2),
        step_sizes: Tuple[float, float] = (1e-06, 50.0),
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_layers = n_layers
        self.activation = activation
        self.contamination = contamination
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.etas = etas
        self.step_sizes = step_sizes
        self.device = device
        self.dtype = dtype


"""Generated config for RpropOneClassSVM."""
class RpropOneClassSVMConfig(ConfigTemplate):
    model_name = "RpropOneClassSVM"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        kernel = 'rbf',
        kernel_params = None,
        degree = 3,
        gamma = 'scale',
        coef0 = 0.0,
        tol = 0.001,
        fit_intercept = True,
        shuffle = True,
        random_state = None,
        learning_rate = 'invscaling',
        eta0 = 0.01,
        power_t = 0.5,
        warm_start = False,
        average = False,
        nu = 0.5,
        shrinking = True,
        cache_size = 200,
        verbose = False,
        max_iter = -1,
        etas: Tuple[float, float] = (0.5, 1.2),
        step_sizes: Tuple[float, float] = (1e-06, 50.0),
        device = 'cpu',
        dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.warm_start = warm_start
        self.average = average
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.etas = etas
        self.step_sizes = step_sizes
        self.device = device
        self.dtype = dtype


"""Generated config for SGDOneClassSVM."""
class SGDOneClassSVMConfig(ConfigTemplate):
    model_name = "SGDOneClassSVM"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        nu: float = 0.5,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 0.001,
        shuffle: bool = True,
        verbose: int = 0,
        random_state: Union[int, None] = None,
        learning_rate: Literal['constant', 'optimal', 'invscaling', 'adaptive'] = 'optimal',
        eta0: float = 0.01,
        power_t: float = 0.5,
        warm_start: bool = False,
        average: bool = False,
        kernel: Union[str, Callable, nn.Module] = 'rbf',
        kernel_params: dict = None,
        degree: int = 3,
        gamma: Union[Literal['scale', 'auto'], float] = 'scale',
        coef0: float = 0.0,
        shrinking: bool = True,
        cache_size: float = 200,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.nu = nu
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle = shuffle
        self.verbose = verbose
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.warm_start = warm_start
        self.average = average
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.device = device
        self.dtype = dtype


"""Generated config for ScheduleFreeEstimatorBasedOutlierDetection."""
class ScheduleFreeEstimatorBasedOutlierDetectionConfig(ConfigTemplate):
    model_name = "ScheduleFreeEstimatorBasedOutlierDetection"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        estimator: Optional[MLModule] = None,
        hidden_dim: int = 64,
        bottleneck_dim: int = 16,
        n_layers: int = 2,
        activation: str = 'relu',
        contamination: Union[Literal['auto'], float] = 'auto',
        max_iter: int = 100,
        tol: float = 0.0001,
        eta0: float = 0.001,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
        verbose: bool = False,
        beta: float = 0.9,
        r: float = 0.0,
        warmup_steps: int = 0,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_layers = n_layers
        self.activation = activation
        self.contamination = contamination
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.beta = beta
        self.r = r
        self.warmup_steps = warmup_steps
        self.device = device
        self.dtype = dtype


"""Generated config for ScheduleFreeOneClassSVM."""
class ScheduleFreeOneClassSVMConfig(ConfigTemplate):
    model_name = "ScheduleFreeOneClassSVM"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        kernel = 'rbf',
        kernel_params = None,
        degree = 3,
        gamma = 'scale',
        coef0 = 0.0,
        tol = 0.001,
        fit_intercept = True,
        shuffle = True,
        random_state = None,
        learning_rate = 'constant',
        eta0 = 0.01,
        power_t = 0.5,
        warm_start = False,
        average = False,
        nu = 0.5,
        shrinking = True,
        cache_size = 200,
        verbose = False,
        max_iter = -1,
        beta: float = 0.9,
        r: float = 0.0,
        warmup_steps: int = 0,
        device = 'cpu',
        dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.warm_start = warm_start
        self.average = average
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.beta = beta
        self.r = r
        self.warmup_steps = warmup_steps
        self.device = device
        self.dtype = dtype


"""Generated config for ShampooEstimatorBasedOutlierDetection."""
class ShampooEstimatorBasedOutlierDetectionConfig(ConfigTemplate):
    model_name = "ShampooEstimatorBasedOutlierDetection"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        estimator: Optional[MLModule] = None,
        hidden_dim: int = 64,
        bottleneck_dim: int = 16,
        n_layers: int = 2,
        activation: str = 'relu',
        contamination: Union[Literal['auto'], float] = 'auto',
        max_iter: int = 100,
        tol: float = 0.0001,
        eta0: float = 0.001,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
        verbose: bool = False,
        update_freq: int = 1,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_layers = n_layers
        self.activation = activation
        self.contamination = contamination
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.update_freq = update_freq
        self.device = device
        self.dtype = dtype


"""Generated config for ShampooOneClassSVM."""
class ShampooOneClassSVMConfig(ConfigTemplate):
    model_name = "ShampooOneClassSVM"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        kernel = 'rbf',
        kernel_params = None,
        degree = 3,
        gamma = 'scale',
        coef0 = 0.0,
        tol = 0.001,
        fit_intercept = True,
        shuffle = True,
        random_state = None,
        learning_rate = 'invscaling',
        eta0 = 0.01,
        power_t = 0.5,
        warm_start = False,
        average = False,
        nu = 0.5,
        shrinking = True,
        cache_size = 200,
        verbose = False,
        max_iter = -1,
        update_freq: int = 1,
        device = 'cpu',
        dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.warm_start = warm_start
        self.average = average
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.update_freq = update_freq
        self.device = device
        self.dtype = dtype


"""Generated config for SophiaEstimatorBasedOutlierDetection."""
class SophiaEstimatorBasedOutlierDetectionConfig(ConfigTemplate):
    model_name = "SophiaEstimatorBasedOutlierDetection"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        estimator: Optional[MLModule] = None,
        hidden_dim: int = 64,
        bottleneck_dim: int = 16,
        n_layers: int = 2,
        activation: str = 'relu',
        contamination: Union[Literal['auto'], float] = 'auto',
        max_iter: int = 100,
        tol: float = 0.0001,
        eta0: float = 0.001,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
        verbose: bool = False,
        beta1: float = 0.9,
        beta2: float = 0.999,
        rho: float = 0.04,
        update_freq: int = 10,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_layers = n_layers
        self.activation = activation
        self.contamination = contamination
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.beta1 = beta1
        self.beta2 = beta2
        self.rho = rho
        self.update_freq = update_freq
        self.device = device
        self.dtype = dtype


"""Generated config for SophiaOneClassSVM."""
class SophiaOneClassSVMConfig(ConfigTemplate):
    model_name = "SophiaOneClassSVM"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        kernel = 'rbf',
        kernel_params = None,
        degree = 3,
        gamma = 'scale',
        coef0 = 0.0,
        tol = 0.001,
        fit_intercept = True,
        shuffle = True,
        random_state = None,
        learning_rate = 'invscaling',
        eta0 = 0.01,
        power_t = 0.5,
        warm_start = False,
        average = False,
        nu = 0.5,
        shrinking = True,
        cache_size = 200,
        verbose = False,
        max_iter = -1,
        beta1: float = 0.9,
        beta2: float = 0.999,
        rho: float = 0.04,
        update_freq: int = 10,
        device = 'cpu',
        dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.warm_start = warm_start
        self.average = average
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.beta1 = beta1
        self.beta2 = beta2
        self.rho = rho
        self.update_freq = update_freq
        self.device = device
        self.dtype = dtype


"""Generated config for TRPOEstimatorBasedOutlierDetection."""
class TRPOEstimatorBasedOutlierDetectionConfig(ConfigTemplate):
    model_name = "TRPOEstimatorBasedOutlierDetection"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        estimator: Optional[MLModule] = None,
        hidden_dim: int = 64,
        bottleneck_dim: int = 16,
        n_layers: int = 2,
        activation: str = 'relu',
        contamination: Union[Literal['auto'], float] = 'auto',
        max_iter: int = 100,
        tol: float = 0.0001,
        eta0: float = 0.001,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
        verbose: bool = False,
        max_kl: float = 0.01,
        damping: float = 0.1,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_layers = n_layers
        self.activation = activation
        self.contamination = contamination
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.max_kl = max_kl
        self.damping = damping
        self.device = device
        self.dtype = dtype


"""Generated config for TRPOOneClassSVM."""
class TRPOOneClassSVMConfig(ConfigTemplate):
    model_name = "TRPOOneClassSVM"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        kernel = 'rbf',
        kernel_params = None,
        degree = 3,
        gamma = 'scale',
        coef0 = 0.0,
        tol = 0.001,
        fit_intercept = True,
        shuffle = True,
        random_state = None,
        learning_rate = 'constant',
        eta0 = 0.01,
        power_t = 0.5,
        warm_start = False,
        average = False,
        nu = 0.5,
        shrinking = True,
        cache_size = 200,
        verbose = False,
        max_iter = -1,
        max_kl: float = 0.01,
        damping: float = 0.1,
        device = 'cpu',
        dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.warm_start = warm_start
        self.average = average
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.max_kl = max_kl
        self.damping = damping
        self.device = device
        self.dtype = dtype


"""Generated config for YogiEstimatorBasedOutlierDetection."""
class YogiEstimatorBasedOutlierDetectionConfig(ConfigTemplate):
    model_name = "YogiEstimatorBasedOutlierDetection"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        estimator: Optional[MLModule] = None,
        hidden_dim: int = 64,
        bottleneck_dim: int = 16,
        n_layers: int = 2,
        activation: str = 'relu',
        contamination: Union[Literal['auto'], float] = 'auto',
        max_iter: int = 100,
        tol: float = 0.0001,
        eta0: float = 0.01,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
        verbose: bool = False,
        beta1: float = 0.9,
        beta2: float = 0.999,
        smoothening_term: float = 0.001,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_layers = n_layers
        self.activation = activation
        self.contamination = contamination
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.beta1 = beta1
        self.beta2 = beta2
        self.smoothening_term = smoothening_term
        self.device = device
        self.dtype = dtype


"""Generated config for _BaseEstimatorBasedOutlierDetection."""
class _BaseEstimatorBasedOutlierDetectionConfig(ConfigTemplate):
    model_name = "_BaseEstimatorBasedOutlierDetection"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        estimator: Optional[MLModule] = None,
        hidden_dim: int = 64,
        bottleneck_dim: int = 16,
        n_layers: int = 2,
        activation: Union[str, nn.Module, Callable] = 'relu',
        contamination: Union[Literal['auto'], float] = 'auto',
        max_iter: int = 100,
        tol: float = 0.0001,
        eta0: float = 0.001,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
        verbose: bool = False,
        warm_start: bool = False,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.estimator = estimator
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_layers = n_layers
        self.activation = activation
        self.contamination = contamination
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.device = device
        self.dtype = dtype


"""Generated config for _CallableAct."""
class _CallableActConfig(ConfigTemplate):
    model_name = "_CallableAct"
    model_path = "Code.models.machine_learning.preprocessing.outlier.outlier"

    def __init__(self,
        immutable: bool = True,
        self_ = None,
        fn = act,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.self_ = self_
        self.fn = fn

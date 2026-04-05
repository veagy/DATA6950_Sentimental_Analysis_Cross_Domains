"""Config templates for kernels."""
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from .....config import ConfigTemplate

try:
    import torch
except ImportError:
    torch = None

"""Generated config for AnovaKernel."""
class AnovaKernelConfig(ConfigTemplate):
    model_name = "AnovaKernel"
    model_path = "Code.models.machine_learning.regression.svm.kernels"

    def __init__(self,
        immutable: bool = True,
        degree: int = None,
        gamma: float = None,
        trainable: bool = False,
        num_support_vectors: int = None,
        num_features: int = None,
        num_classes: int = 1,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.degree = degree
        self.gamma = gamma
        self.trainable = trainable
        self.num_support_vectors = num_support_vectors
        self.num_features = num_features
        self.num_classes = num_classes
        self.device = device
        self.dtype = dtype


"""Generated config for BaseKernel."""
class BaseKernelConfig(ConfigTemplate):
    model_name = "BaseKernel"
    model_path = "Code.models.machine_learning.regression.svm.kernels"

    def __init__(self,
        immutable: bool = True,
        trainable: bool = False,
        num_support_vectors: int = None,
        num_features: int = None,
        num_classes: int = 1,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.trainable = trainable
        self.num_support_vectors = num_support_vectors
        self.num_features = num_features
        self.num_classes = num_classes
        self.device = device
        self.dtype = dtype


"""Generated config for ChiSquareKernel."""
class ChiSquareKernelConfig(ConfigTemplate):
    model_name = "ChiSquareKernel"
    model_path = "Code.models.machine_learning.regression.svm.kernels"

    def __init__(self,
        immutable: bool = True,
        gamma: float = None,
        trainable: bool = False,
        num_support_vectors: int = None,
        num_features: int = None,
        num_classes: int = 1,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.gamma = gamma
        self.trainable = trainable
        self.num_support_vectors = num_support_vectors
        self.num_features = num_features
        self.num_classes = num_classes
        self.device = device
        self.dtype = dtype


"""Generated config for ConstantKernel."""
class ConstantKernelConfig(ConfigTemplate):
    model_name = "ConstantKernel"
    model_path = "Code.models.machine_learning.regression.svm.kernels"

    def __init__(self,
        immutable: bool = True,
        constant_val: float = None,
        trainable: bool = False,
        num_support_vectors: int = None,
        num_features: int = None,
        num_classes: int = 1,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.constant_val = constant_val
        self.trainable = trainable
        self.num_support_vectors = num_support_vectors
        self.num_features = num_features
        self.num_classes = num_classes
        self.device = device
        self.dtype = dtype


"""Generated config for HistogramIntersectionKernel."""
class HistogramIntersectionKernelConfig(ConfigTemplate):
    model_name = "HistogramIntersectionKernel"
    model_path = "Code.models.machine_learning.regression.svm.kernels"

    def __init__(self,
        immutable: bool = True,
        trainable: bool = False,
        num_support_vectors: int = None,
        num_features: int = None,
        num_classes: int = 1,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.trainable = trainable
        self.num_support_vectors = num_support_vectors
        self.num_features = num_features
        self.num_classes = num_classes
        self.device = device
        self.dtype = dtype


"""Generated config for LaplacianKernel."""
class LaplacianKernelConfig(ConfigTemplate):
    model_name = "LaplacianKernel"
    model_path = "Code.models.machine_learning.regression.svm.kernels"

    def __init__(self,
        immutable: bool = True,
        sigma: float = None,
        trainable: bool = False,
        num_support_vectors: int = None,
        num_features: int = None,
        num_classes: int = 1,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.sigma = sigma
        self.trainable = trainable
        self.num_support_vectors = num_support_vectors
        self.num_features = num_features
        self.num_classes = num_classes
        self.device = device
        self.dtype = dtype


"""Generated config for LinearKernel."""
class LinearKernelConfig(ConfigTemplate):
    model_name = "LinearKernel"
    model_path = "Code.models.machine_learning.regression.svm.kernels"

    def __init__(self,
        immutable: bool = True,
        trainable: bool = False,
        num_support_vectors: int = None,
        num_features: int = None,
        num_classes: int = 1,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.trainable = trainable
        self.num_support_vectors = num_support_vectors
        self.num_features = num_features
        self.num_classes = num_classes
        self.device = device
        self.dtype = dtype


"""Generated config for PolyKernel."""
class PolyKernelConfig(ConfigTemplate):
    model_name = "PolyKernel"
    model_path = "Code.models.machine_learning.regression.svm.kernels"

    def __init__(self,
        immutable: bool = True,
        degree: float = None,
        gamma: float = None,
        bias: float = None,
        trainable: bool = False,
        num_support_vectors: int = None,
        num_features: int = None,
        num_classes: int = 1,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.degree = degree
        self.gamma = gamma
        self.bias = bias
        self.trainable = trainable
        self.num_support_vectors = num_support_vectors
        self.num_features = num_features
        self.num_classes = num_classes
        self.device = device
        self.dtype = dtype


"""Generated config for ProductKernel."""
class ProductKernelConfig(ConfigTemplate):
    model_name = "ProductKernel"
    model_path = "Code.models.machine_learning.regression.svm.kernels"

    def __init__(self,
        immutable: bool = True,
        k1: MLModule = None,
        k2: MLModule = None,
        trainable: bool = False,
        num_support_vectors: int = None,
        num_features: int = None,
        num_classes: int = 1,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.k1 = k1
        self.k2 = k2
        self.trainable = trainable
        self.num_support_vectors = num_support_vectors
        self.num_features = num_features
        self.num_classes = num_classes
        self.device = device
        self.dtype = dtype


"""Generated config for RBFKernel."""
class RBFKernelConfig(ConfigTemplate):
    model_name = "RBFKernel"
    model_path = "Code.models.machine_learning.regression.svm.kernels"

    def __init__(self,
        immutable: bool = True,
        gamma: float = None,
        trainable: bool = False,
        num_support_vectors: int = None,
        num_features: int = None,
        num_classes: int = 1,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.gamma = gamma
        self.trainable = trainable
        self.num_support_vectors = num_support_vectors
        self.num_features = num_features
        self.num_classes = num_classes
        self.device = device
        self.dtype = dtype


"""Generated config for SigmoidKernel."""
class SigmoidKernelConfig(ConfigTemplate):
    model_name = "SigmoidKernel"
    model_path = "Code.models.machine_learning.regression.svm.kernels"

    def __init__(self,
        immutable: bool = True,
        gamma: float = None,
        bias: float = None,
        trainable: bool = False,
        num_support_vectors: int = None,
        num_features: int = None,
        num_classes: int = 1,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.gamma = gamma
        self.bias = bias
        self.trainable = trainable
        self.num_support_vectors = num_support_vectors
        self.num_features = num_features
        self.num_classes = num_classes
        self.device = device
        self.dtype = dtype


"""Generated config for SumKernel."""
class SumKernelConfig(ConfigTemplate):
    model_name = "SumKernel"
    model_path = "Code.models.machine_learning.regression.svm.kernels"

    def __init__(self,
        immutable: bool = True,
        k1: MLModule = None,
        k2: MLModule = None,
        trainable: bool = False,
        num_support_vectors: int = None,
        num_features: int = None,
        num_classes: int = 1,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.k1 = k1
        self.k2 = k2
        self.trainable = trainable
        self.num_support_vectors = num_support_vectors
        self.num_features = num_features
        self.num_classes = num_classes
        self.device = device
        self.dtype = dtype


"""Generated config for WhiteKernel."""
class WhiteKernelConfig(ConfigTemplate):
    model_name = "WhiteKernel"
    model_path = "Code.models.machine_learning.regression.svm.kernels"

    def __init__(self,
        immutable: bool = True,
        noise_level: float = None,
        trainable: bool = False,
        num_support_vectors: int = None,
        num_features: int = None,
        num_classes: int = 1,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.noise_level = noise_level
        self.trainable = trainable
        self.num_support_vectors = num_support_vectors
        self.num_features = num_features
        self.num_classes = num_classes
        self.device = device
        self.dtype = dtype

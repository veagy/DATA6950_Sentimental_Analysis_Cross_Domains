import torch
import torch.nn as nn
import math
from typing import Optional, Any, List
from .....models.utils import MLModule

__all__ = [
    "KernelRegistry",
    "get_kernel_class",
    "LinearKernel",
    "PolyKernel",
    "RBFKernel",
    "SigmoidKernel",
    "ConstantKernel",
    "WhiteKernel",
    "SumKernel",
    "ProductKernel",
    "LaplacianKernel",
    "HistogramIntersectionKernel",
    "ChiSquareKernel",
    "AnovaKernel"
]


# Kernel Registry
class KernelRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str):
        def decorator(kernel_cls):
            cls._registry[name.lower()] = kernel_cls
            return kernel_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[Any]:  # Returns class, not instance
        return cls._registry.get(name.lower())

    @classmethod
    def list_kernels(cls) -> List[str]:
        return list(cls._registry.keys())


# Helper aliases for easier import
register_kernel = KernelRegistry.register
get_kernel_class = KernelRegistry.get


class BaseKernel(MLModule):
    def __init__(self,
                 trainable: bool = False,
                 num_support_vectors: int = None,
                 num_features: int = None,
                 num_classes: int = 1,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.trainable = trainable
        self.num_support_vectors = num_support_vectors
        self.num_features = num_features
        self.num_classes = num_classes
        self.device = device
        self.dtype = dtype

        if self.trainable:
            if num_support_vectors is None or num_features is None:
                raise ValueError("trainable=True requires num_support_vectors and num_features to be specified")
            # Create parametric support vectors: (num_classes, num_support_vectors, num_features)
            self.fn_support_vectors = nn.Parameter(
                torch.randn(num_classes, num_support_vectors, num_features, device=device, dtype=dtype))
        else:
            self.register_buffer("fn_support_vectors", None)  # Placeholder

    @property
    def theta(self):
        """
        Returns the (log-transformed) non-fixed hyperparameters.
        """
        return torch.tensor([], device=self.device, dtype=self.dtype)

    @theta.setter
    def theta(self, theta):
        """
        Sets the (log-transformed) non-fixed hyperparameters.
        """
        pass

    @property
    def bounds(self):
        """
        Returns the log-transformed bounds on the theta.
        """
        return torch.tensor([], device=self.device, dtype=self.dtype)

    @property
    def hyperparameters(self):
        """
        Returns a list of all hyperparameter specifications.
        """
        return []

    def _get_vectors(self, xi: torch.Tensor, xj: torch.Tensor = None):
        """
        Helper to get the correct xi and xj tensors based on trainable mode.
        If trainable=True and xj is None, use self.fn_support_vectors.
        Also handles broadcasting for (C, M, D) support vectors.
        """
        if self.trainable and xj is None:
            xj = self.fn_support_vectors
            xi = xi.unsqueeze(0)
        
        if xj is not None:
             # Standardize to 3D if either is 3D
             if xi.ndim == 2 and xj.ndim == 3:
                 xi = xi.unsqueeze(0)
             elif xi.ndim == 3 and xj.ndim == 2:
                 xj = xj.unsqueeze(0)
             
             # Handle batch broadcasting if needed
             if xi.ndim == 3 and xj.ndim == 3:
                 if xi.size(0) == 1 and xj.size(0) > 1:
                     xi = xi.expand(xj.size(0), -1, -1)
                 elif xj.size(0) == 1 and xi.size(0) > 1:
                     xj = xj.expand(xi.size(0), -1, -1)
        
        return xi, xj

    def _post_process_kernel(self, K: torch.Tensor, xi: torch.Tensor, xj: torch.Tensor) -> torch.Tensor:
        """
        Squeezes the batch dimension if it was added solely for broadcasting
        and not present in original inputs.
        """
        if K.ndim == 3 and K.size(0) == 1:
            return K.squeeze(0)
        return K

    def forward(self, xi: torch.Tensor, xj: torch.Tensor = None):
        raise NotImplementedError


@register_kernel("linear")
class LinearKernel(BaseKernel):
    def __init__(self,
                 trainable: bool = False,
                 num_support_vectors: int = None,
                 num_features: int = None,
                 num_classes: int = 1,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__(trainable, num_support_vectors, num_features, num_classes, device, dtype, *args, **kwargs)

    def forward(self, xi: torch.Tensor, xj: torch.Tensor = None):
        xi, xj = self._get_vectors(xi, xj)
        if xj is None:  # Standard case (trainable=False, xj=None -> self-kernel on xi)
            xj = xi

        if xj.ndim == 3 and xi.ndim == 3:
            # xi: (C, N, D), xj: (C, M, D) -> (C, N, M)
            return torch.matmul(xi, xj.transpose(-2, -1))
        elif xj.ndim == 3:
            # xi: (N, D), xj: (C, M, D) -> unsqueeze xi -> (1, N, D)
            return torch.matmul(xi.unsqueeze(0), xj.transpose(-2, -1))
        # Default 2D
        return xi @ xj.T


@register_kernel("poly")
class PolyKernel(BaseKernel):
    def __init__(self,
                 degree: float,
                 gamma: float,
                 bias: float,
                 trainable: bool = False,
                 num_support_vectors: int = None,
                 num_features: int = None,
                 num_classes: int = 1,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__(trainable, num_support_vectors, num_features, num_classes, device, dtype, *args, **kwargs)
        self.d = degree
        self.r = bias
        self.gamma = gamma

    @property
    def theta(self):
        return torch.tensor([math.log(self.gamma), math.log(self.r)], device=self.device, dtype=self.dtype)

    @theta.setter
    def theta(self, theta):
        self.gamma = math.exp(theta[0])
        self.r = math.exp(theta[1])

    @property
    def bounds(self):
        return torch.tensor([[-1e5, 1e5], [-1e5, 1e5]], device=self.device, dtype=self.dtype)

    def forward(self, xi: torch.Tensor, xj: torch.Tensor = None):
        xi, xj = self._get_vectors(xi, xj)
        if xj is None: xj = xi

        if xj.ndim == 3:
            if xi.ndim == 2: xi = xi.unsqueeze(0)
            dot = torch.matmul(xi, xj.transpose(-2, -1))
        else:
            dot = xi @ xj.T

        return (self.gamma * dot + self.r) ** self.d


@register_kernel("rbf")
class RBFKernel(BaseKernel):
    def __init__(self,
                 gamma: float,
                 trainable: bool = False,
                 num_support_vectors: int = None,
                 num_features: int = None,
                 num_classes: int = 1,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__(trainable, num_support_vectors, num_features, num_classes, device, dtype, *args, **kwargs)
        self.gamma = gamma

    @property
    def theta(self):
        return torch.tensor([math.log(self.gamma)], device=self.device, dtype=self.dtype)

    @theta.setter
    def theta(self, theta):
        if isinstance(theta, torch.Tensor):
            self.gamma = torch.exp(theta[0]).item()
        else:
            self.gamma = math.exp(theta[0])

    @property
    def bounds(self):
        # Gamma must be positive
        return torch.tensor([[-1e5, 1e5]], device=self.device, dtype=self.dtype)

    def forward(self, xi: torch.Tensor, xj: torch.Tensor = None):
        xi_v, xj_v = self._get_vectors(xi, xj)
        if xj_v is None: xj_v = xi_v

        dist_sq = torch.cdist(xi_v, xj_v, p=2) ** 2
        res = torch.exp(-self.gamma * dist_sq)
        return self._post_process_kernel(res, xi, xj)


# Alias: "gaussian" -> RBF (used by QDPC, etc.)
KernelRegistry._registry["gaussian"] = RBFKernel


@register_kernel("sigmoid")
class SigmoidKernel(BaseKernel):
    def __init__(self,
                 gamma: float,
                 bias: float,
                 trainable: bool = False,
                 num_support_vectors: int = None,
                 num_features: int = None,
                 num_classes: int = 1,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__(trainable, num_support_vectors, num_features, num_classes, device, dtype, *args, **kwargs)
        self.gamma = gamma
        self.r = bias

    @property
    def theta(self):
        return torch.tensor([math.log(self.gamma), math.log(self.r)], device=self.device, dtype=self.dtype)

    @theta.setter
    def theta(self, theta):
        self.gamma = math.exp(theta[0])
        self.r = math.exp(theta[1])

    @property
    def bounds(self):
        return torch.tensor([[-1e5, 1e5], [-1e5, 1e5]], device=self.device, dtype=self.dtype)

    def forward(self, xi: torch.Tensor, xj: torch.Tensor = None):
        xi, xj = self._get_vectors(xi, xj)
        if xj is None: xj = xi

        if xj.ndim == 3:
            if xi.ndim == 2: xi = xi.unsqueeze(0)
            dot = torch.matmul(xi, xj.transpose(-2, -1))
        else:
            dot = xi @ xj.T

        return torch.tanh(self.gamma * dot + self.r)


@register_kernel("const")
class ConstantKernel(BaseKernel):
    def __init__(self,
                 constant_val: float,
                 trainable: bool = False,
                 num_support_vectors: int = None,
                 num_features: int = None,
                 num_classes: int = 1,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__(trainable, num_support_vectors, num_features, num_classes, device, dtype, *args, **kwargs)
        self.const_val = constant_val

    @property
    def theta(self):
        return torch.tensor([math.log(self.const_val)], device=self.device, dtype=self.dtype)

    @theta.setter
    def theta(self, theta):
        if isinstance(theta, torch.Tensor):
            self.const_val = torch.exp(theta[0]).item()
        else:
            self.const_val = math.exp(theta[0])

    @property
    def bounds(self):
        return torch.tensor([[-1e5, 1e5]], device=self.device, dtype=self.dtype)

    def forward(self, xi: torch.Tensor, xj: torch.Tensor = None):
        xi, xj = self._get_vectors(xi, xj)
        if xj is None: xj = xi

        N = xi.size(-2)
        M = xj.size(-2)

        if xj.ndim == 3:  # (C, M, D)
            C = xj.size(0)
            return torch.full((C, N, M), self.const_val, device=xi.device, dtype=self.dtype)

        return torch.full((N, M), self.const_val, device=xi.device, dtype=self.dtype)


@register_kernel("white")
class WhiteKernel(BaseKernel):
    def __init__(self,
                 noise_level: float,
                 trainable: bool = False,
                 num_support_vectors: int = None,
                 num_features: int = None,
                 num_classes: int = 1,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__(trainable, num_support_vectors, num_features, num_classes, device, dtype, *args, **kwargs)
        self.noise_level = noise_level

    @property
    def theta(self):
        return torch.tensor([math.log(self.noise_level)], device=self.device, dtype=self.dtype)

    @theta.setter
    def theta(self, theta):
        if isinstance(theta, torch.Tensor):
            self.noise_level = torch.exp(theta[0]).item()
        else:
            self.noise_level = math.exp(theta[0])

    @property
    def bounds(self):
        return torch.tensor([[-1e5, 1e5]], device=self.device, dtype=self.dtype)

    def forward(self, xi: torch.Tensor, xj: torch.Tensor = None):
        xi, xj = self._get_vectors(xi, xj)
        if xj is None: xj = xi

        is_self = (xi is xj)

        if xj.ndim == 3:
            C, M, _ = xj.size()
            N = xi.size(-2)
            return torch.zeros((C, N, M), device=xi.device, dtype=self.dtype)

        N = xi.size(0)
        M = xj.size(0)
        if N == M and is_self:
            return torch.eye(N, device=xi.device, dtype=self.dtype) * self.noise_level
        return torch.zeros((N, M), device=xi.device, dtype=self.dtype)


@register_kernel("sum")
class SumKernel(BaseKernel):
    def __init__(self,
                 k1: MLModule,
                 k2: MLModule,
                 trainable: bool = False,
                 num_support_vectors: int = None,
                 num_features: int = None,
                 num_classes: int = 1,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__(trainable, num_support_vectors, num_features, num_classes, device, dtype, *args, **kwargs)
        if isinstance(k1, nn.Module):
            self.k1 = k1
        else:
            self.k1 = k1(trainable=trainable, num_support_vectors=num_support_vectors,
                         num_features=num_features, num_classes=num_classes,
                         device=device, dtype=dtype, *args, **kwargs)

        if isinstance(k2, nn.Module):
            self.k2 = k2
        else:
            self.k2 = k2(trainable=trainable, num_support_vectors=num_support_vectors,
                         num_features=num_features, num_classes=num_classes,
                         device=device, dtype=dtype, *args, **kwargs)

    @property
    def theta(self):
        return torch.cat([self.k1.theta, self.k2.theta])

    @theta.setter
    def theta(self, theta):
        k1_dims = self.k1.theta.numel()
        self.k1.theta = theta[:k1_dims]
        self.k2.theta = theta[k1_dims:]

    @property
    def bounds(self):
        return torch.cat([self.k1.bounds, self.k2.bounds])

    def forward(self, xi: torch.Tensor, xj: torch.Tensor = None):
        return self.k1(xi, xj) + self.k2(xi, xj)


@register_kernel("prod")
class ProductKernel(BaseKernel):
    def __init__(self,
                 k1: MLModule,
                 k2: MLModule,
                 trainable: bool = False,
                 num_support_vectors: int = None,
                 num_features: int = None,
                 num_classes: int = 1,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__(trainable, num_support_vectors, num_features, num_classes, device, dtype, *args, **kwargs)
        if isinstance(k1, nn.Module):
            self.k1 = k1
        else:
            self.k1 = k1(trainable=trainable, num_support_vectors=num_support_vectors,
                         num_features=num_features, num_classes=num_classes,
                         device=device, dtype=dtype, *args, **kwargs)

        if isinstance(k2, nn.Module):
            self.k2 = k2
        else:
            self.k2 = k2(trainable=trainable, num_support_vectors=num_support_vectors,
                         num_features=num_features, num_classes=num_classes,
                         device=device, dtype=dtype, *args, **kwargs)

    @property
    def theta(self):
        return torch.cat([self.k1.theta, self.k2.theta])

    @theta.setter
    def theta(self, theta):
        k1_dims = self.k1.theta.numel()
        self.k1.theta = theta[:k1_dims]
        self.k2.theta = theta[k1_dims:]

    @property
    def bounds(self):
        return torch.cat([self.k1.bounds, self.k2.bounds])

    def forward(self, xi: torch.Tensor, xj: torch.Tensor = None):
        return self.k1(xi, xj) * self.k2(xi, xj)


@register_kernel("laplacian")
class LaplacianKernel(BaseKernel):
    def __init__(self,
                 sigma: float,
                 trainable: bool = False,
                 num_support_vectors: int = None,
                 num_features: int = None,
                 num_classes: int = 1,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__(trainable, num_support_vectors, num_features, num_classes, device, dtype, *args, **kwargs)
        self.sigma = abs(sigma)

    def forward(self, xi: torch.Tensor, xj: torch.Tensor = None):
        xi_v, xj_v = self._get_vectors(xi, xj)
        if xj_v is None: xj_v = xi_v

        dist = torch.cdist(xi_v, xj_v, p=1)
        res = torch.exp(-dist / (self.sigma + 1e-6))
        return self._post_process_kernel(res, xi, xj)


@register_kernel("hist_intersection")
class HistogramIntersectionKernel(BaseKernel):
    def __init__(self,
                 trainable: bool = False,
                 num_support_vectors: int = None,
                 num_features: int = None,
                 num_classes: int = 1,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__(trainable, num_support_vectors, num_features, num_classes, device, dtype, *args, **kwargs)

    def forward(self, xi: torch.Tensor, xj: torch.Tensor = None):
        xi, xj = self._get_vectors(xi, xj)
        if xj is None: xj = xi

        xi = xi.unsqueeze(-2)
        xj = xj.unsqueeze(-3)

        intersection = torch.min(xi, xj).sum(dim=-1)
        return intersection


@register_kernel("chi_square")
class ChiSquareKernel(BaseKernel):
    def __init__(self,
                 gamma: float,
                 trainable: bool = False,
                 num_support_vectors: int = None,
                 num_features: int = None,
                 num_classes: int = 1,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__(trainable, num_support_vectors, num_features, num_classes, device, dtype, *args, **kwargs)
        self.gamma = gamma

    def forward(self, xi: torch.Tensor, xj: torch.Tensor = None):
        xi, xj = self._get_vectors(xi, xj)
        if xj is None: xj = xi

        xi = xi.unsqueeze(-2)
        xj = xj.unsqueeze(-3)

        num = (xi - xj).pow(2)
        denom = (xi + xj) + 1e-8
        chi_dist = torch.sum(num / denom, dim=-1)
        return torch.exp(-self.gamma * chi_dist)


@register_kernel("anova")
class AnovaKernel(BaseKernel):
    def __init__(self,
                 degree: int,
                 gamma: float,
                 trainable: bool = False,
                 num_support_vectors: int = None,
                 num_features: int = None,
                 num_classes: int = 1,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__(trainable, num_support_vectors, num_features, num_classes, device, dtype, *args, **kwargs)
        self.d = degree
        self.gamma = gamma

    def anova(self, xi: torch.Tensor, xj: torch.Tensor):
        # xi: (..., N, D), xj: (..., M, D)
        xi = xi.unsqueeze(-2)
        xj = xj.unsqueeze(-3)

        h = torch.exp(-self.gamma * (xi - xj).pow(2))  # (..., N, M, D)

        # Dynamic programming for ANOVA kernel
        # dp[k] stores sum of products of k distinct features
        shape = list(h.shape[:-1])  # (..., N, M)
        dp = torch.zeros([self.d + 1] + shape, device=xi.device, dtype=self.dtype)
        dp[0] = 1.0

        D = xi.size(-1)
        for i in range(D):
            h_feature = h[..., i]  # (..., N, M)
            for d in range(min(i + 1, self.d), 0, -1):
                dp[d] += h_feature * dp[d - 1]

        return dp[self.d]

    def forward(self, xi: torch.Tensor, xj: torch.Tensor = None):
        xi, xj = self._get_vectors(xi, xj)
        if xj is None: xj = xi
        return self.anova(xi, xj)

import torch
import torch.nn as nn
import math
from typing import Optional, Callable, Union, Any, List, Tuple, Literal
from .....models.utils import MLTransform
from torch.func import vmap
import joblib


__all__ = [
    "GaussianRandomProjection",
    "SparseRandomProjection",
]


def _johnson_lindenstrauss_min_dim(n_samples: int, eps: float) -> int:
    """Minimum components required by the Johnson-Lindenstrauss lemma.

    n_components >= 4 * log(n_samples) / (eps^2/2 - eps^3/3)
    """
    if eps <= 0 or eps >= 1:
        raise ValueError(f"eps must be in (0, 1), got {eps}")
    denom = eps ** 2 / 2 - eps ** 3 / 3
    if denom <= 0:
        raise ValueError(f"eps={eps} yields a non-positive denominator; use a smaller value.")
    return max(1, int(math.ceil(4 * math.log(max(n_samples, 2)) / denom)))


class GaussianRandomProjection(MLTransform):
    def __init__(self,
                 n_components: Union[Literal["auto"], int] = "auto",
                 eps: float = 0.1,
                 compute_inverse_components: bool = False,
                 random_state: Union[int, torch.Generator] = None,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args: Any,
                 **kwargs: Any,
                 ) -> None:
        super().__init__()
        self.n_components = n_components
        self.eps = eps
        self.compute_inverse_components = compute_inverse_components
        self.random_state = random_state
        self.device = device
        self.dtype = dtype
        self.kwargs = dict(kwargs)
        self.kwargs["_args"] = args

        self.n_components_ = None
        self.components_ = None
        self.inverse_components_ = None
        self.n_features_in_ = None

    def _get_generator(self) -> Optional[torch.Generator]:
        """Return a torch Generator for reproducible randomness."""
        if self.random_state is None:
            return None
        if isinstance(self.random_state, torch.Generator):
            return self.random_state
        g = torch.Generator(device=self.device)
        g.manual_seed(int(self.random_state))
        return g

    def fit(
        self, data_or_X: Union[torch.Tensor, Any], y: Any = None, **kwargs: Any
    ) -> "GaussianRandomProjection":
        X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        if self.n_components == "auto":
            n_comp = _johnson_lindenstrauss_min_dim(n_samples, self.eps)
        else:
            n_comp = int(self.n_components)
        n_comp = max(1, min(n_comp, n_features))
        self.n_components_ = n_comp

        gen = self._get_generator()
        if gen is not None:
            components = torch.randn(
                n_comp, n_features, device=self.device, dtype=self.dtype, generator=gen
            )
        else:
            components = torch.randn(n_comp, n_features, device=self.device, dtype=self.dtype)

        self.components_ = components / math.sqrt(n_comp)

        if self.compute_inverse_components:
            self.inverse_components_ = torch.linalg.pinv(self.components_)

        return self

    def transform(self, X: Union[torch.Tensor, Any]) -> torch.Tensor:
        if self.components_ is None:
            raise RuntimeError("GaussianRandomProjection instance is not fitted yet.")
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        return X @ self.components_.T

    def inverse_transform(self, X: Union[torch.Tensor, Any]) -> torch.Tensor:
        if self.inverse_components_ is None:
            raise RuntimeError(
                "inverse_components_ is not available. "
                "Set compute_inverse_components=True when creating the estimator."
            )
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        return X @ self.inverse_components_.T

    def fit_transform(
        self, data_or_X: Union[torch.Tensor, Any], y: Any = None, **kwargs: Any
    ) -> torch.Tensor:
        """Fit to data, then transform it."""
        self.fit(data_or_X, y=y)
        return self.transform(data_or_X)

    def forward(self, X: Union[torch.Tensor, Any]) -> torch.Tensor:
        return self.transform(X)


class SparseRandomProjection(MLTransform):
    def __init__(self,
                 n_components: Union[Literal["auto"], int] = "auto",
                 density: Union[Literal["auto"], float] = "auto",
                 eps: float = 0.1,
                 dense_output: bool = False,
                 compute_inverse_components: bool = False,
                 random_state: Union[int, torch.Generator] = None,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args: Any,
                 **kwargs: Any,
                 ) -> None:
        super().__init__()
        self.n_components = n_components
        self.density = density
        self.eps = eps
        self.dense_output = dense_output
        self.compute_inverse_components = compute_inverse_components
        self.random_state = random_state
        self.device = device
        self.dtype = dtype
        self.kwargs = dict(kwargs)
        self.kwargs["_args"] = args

        self.n_components_ = None
        self.components_ = None
        self.inverse_components_ = None
        self.density_ = None
        self.n_features_in_ = None

    def _get_generator(self) -> Optional[torch.Generator]:
        """Return a torch Generator for reproducible randomness."""
        if self.random_state is None:
            return None
        if isinstance(self.random_state, torch.Generator):
            return self.random_state
        g = torch.Generator(device=self.device)
        g.manual_seed(int(self.random_state))
        return g

    def _make_sparse_matrix(
        self, n_components: int, n_features: int, density: float, gen: Optional[torch.Generator]
    ) -> torch.Tensor:
        scale = math.sqrt(1.0 / density) / math.sqrt(n_components)
        if gen is not None:
            u = torch.rand(n_components, n_features, device=self.device, dtype=self.dtype, generator=gen)
            sign_r = torch.randint(0, 2, (n_components, n_features), device=self.device, generator=gen)
        else:
            u = torch.rand(n_components, n_features, device=self.device, dtype=self.dtype)
            sign_r = torch.randint(0, 2, (n_components, n_features), device=self.device)
        signs = sign_r.to(self.dtype) * 2 - 1
        mat = torch.where(u < density, signs * scale, torch.zeros_like(u))
        return mat

    def fit(
        self, data_or_X: Union[torch.Tensor, Any], y: Any = None, **kwargs: Any
    ) -> "SparseRandomProjection":
        X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        if self.n_components == "auto":
            n_comp = _johnson_lindenstrauss_min_dim(n_samples, self.eps)
        else:
            n_comp = int(self.n_components)
        n_comp = max(1, min(n_comp, n_features))
        self.n_components_ = n_comp

        if self.density == "auto":
            density = min(1.0, 1.0 / math.sqrt(n_features))
        else:
            density = float(self.density)
            density = max(min(density, 1.0), 1.0 / n_features)
        self.density_ = density

        gen = self._get_generator()
        self.components_ = self._make_sparse_matrix(n_comp, n_features, density, gen)

        if self.compute_inverse_components:
            self.inverse_components_ = torch.linalg.pinv(self.components_)

        return self

    def transform(self, X: Union[torch.Tensor, Any]) -> torch.Tensor:
        if self.components_ is None:
            raise RuntimeError("SparseRandomProjection instance is not fitted yet.")
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        return X @ self.components_.T

    def inverse_transform(self, X: Union[torch.Tensor, Any]) -> torch.Tensor:
        if self.inverse_components_ is None:
            raise RuntimeError(
                "inverse_components_ is not available. "
                "Set compute_inverse_components=True when creating the estimator."
            )
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        return X @ self.inverse_components_.T

    def fit_transform(
        self, data_or_X: Union[torch.Tensor, Any], y: Any = None, **kwargs: Any
    ) -> torch.Tensor:
        """Fit to data, then transform it."""
        self.fit(data_or_X, y=y)
        return self.transform(data_or_X)

    def forward(self, X: Union[torch.Tensor, Any]) -> torch.Tensor:
        return self.transform(X)

import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Union, Any, List, Tuple, Dict, Literal, Iterable
from ....models.utils import MLModule
import numpy as np
from ..regression.svm.kernels import (
    get_kernel_class,
    RBFKernel,
    PolyKernel,
    SigmoidKernel,
    LinearKernel,
    LaplacianKernel,
    ChiSquareKernel,
    HistogramIntersectionKernel,
)
from torch.func import vmap
import joblib


__all__ = [
    "AdditiveChi2Sampler",
    "SkewedChi2Sampler",
    "Nystroem",
    "RBFSampler",
    "PolynomialCountSketch",
]

# ---------------------------------------------------------------------------
# Default sample intervals for AdditiveChi2Sampler (from Vedaldi & Zisserman)
# ---------------------------------------------------------------------------
_ADDITIVE_CHI2_SAMPLE_INTERVALS: Dict[int, float] = {1: 0.4, 2: 0.5, 3: 0.6}


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _make_generator(
    random_state: Union[int, torch.Generator, None],
    device: torch.device,
) -> Optional[torch.Generator]:
    """Build a deterministic torch.Generator from an int seed, or return as-is."""
    if random_state is None:
        return None
    if isinstance(random_state, torch.Generator):
        return random_state
    g = torch.Generator(device=device)
    g.manual_seed(int(random_state))
    return g


def _to_float_tensor(X: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Convert X (numpy, pandas, list, tensor) to a 2-D float tensor."""
    if not isinstance(X, torch.Tensor) and hasattr(X, "values"):   # pandas DataFrame / Series
        X = X.values
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X.astype(np.float64))
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float64)
    if X.dim() == 1:
        X = X.unsqueeze(0)
    return X.to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# AdditiveChi2Sampler
# ---------------------------------------------------------------------------

class AdditiveChi2Sampler(MLModule):
    def __init__(self,
                 sample_steps: int = 2,
                 sample_interval: float = None,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.sample_steps = sample_steps
        self.sample_interval = sample_interval
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.args = args
        self.kwargs = kwargs
        # Fitted attributes
        self.sample_interval_: Optional[float] = None
        self.n_features_in_: Optional[int] = None
        self.fit_status = False

    # ------------------------------------------------------------------
    def fit(
        self,
        X: Any,
        y: Optional[Any] = None,
        **kwargs,
    ) -> "AdditiveChi2Sampler":
        X = _to_float_tensor(X, self.device, self.dtype)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        if self.sample_interval is not None:
            self.sample_interval_ = float(self.sample_interval)
        else:
            if self.sample_steps not in _ADDITIVE_CHI2_SAMPLE_INTERVALS:
                raise ValueError(
                    f"If sample_steps is not in {{1, 2, 3}}, you must provide "
                    f"sample_interval explicitly. Got sample_steps={self.sample_steps}."
                )
            self.sample_interval_ = _ADDITIVE_CHI2_SAMPLE_INTERVALS[self.sample_steps]

        self.fit_status = True
        return self

    def transform(
        self,
        X: Any,
        **kwargs,
    ) -> torch.Tensor:
        if not self.fit_status:
            raise RuntimeError(
                "AdditiveChi2Sampler is not fitted. Call fit() first."
            )
        X = _to_float_tensor(X, self.device, self.dtype)
        n_samples, n_features = X.shape
        if n_features != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {n_features}."
            )

        X = X.clamp(min=0.0)                          # must be non-negative
        log_X = torch.log(X + 1e-10)                  # (n, d)
        s = self.sample_steps
        Δ = self.sample_interval_
        stride = 2 * s - 1

        X_out = torch.zeros(n_samples, n_features * stride, device=self.device, dtype=self.dtype)

        # j = 0  (one real component per feature)
        w0 = math.sqrt(2.0 * Δ / math.pi)
        X_out[:, ::stride] = w0 * X              # columns 0, stride, 2*stride, …

        # j = 1 … sample_steps-1  (two components: cos + sin, per feature)
        for j in range(1, s):
            wj = 2.0 * math.sqrt(Δ / (math.cosh(math.pi * j * Δ) * math.pi))
            cos_feat = wj * X * torch.cos(j * Δ * log_X)   # (n, d)
            sin_feat = wj * X * torch.sin(j * Δ * log_X)   # (n, d)
            X_out[:, 2 * j - 1 :: stride] = cos_feat       # columns 2j-1, 2j-1+stride, …
            X_out[:, 2 * j     :: stride] = sin_feat       # columns 2j,   2j+stride, …

        return X_out

    def fit_transform(
        self,
        X: Any,
        y: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Fit and transform X."""
        return self.fit(X, y, **kwargs).transform(X)

    def forward(
        self,
        X: Any,
        y: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        if not self.fit_status:
            return self.fit_transform(X, y, **kwargs)
        return self.transform(X, **kwargs)


# ---------------------------------------------------------------------------
# SkewedChi2Sampler
# ---------------------------------------------------------------------------

class SkewedChi2Sampler(MLModule):
    def __init__(self,
                 skewedness: float = 1.0,
                 n_components: int = 100,
                 random_state: Union[int, torch.Generator] = None,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.skewedness = skewedness
        self.n_components = n_components
        self.random_state = random_state
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.args = args
        self.kwargs = kwargs
        # Fitted attributes
        self.random_weights_: Optional[torch.Tensor] = None
        self.random_offset_: Optional[torch.Tensor] = None
        self.n_features_in_: Optional[int] = None
        self.fit_status = False

    def _sample_secant_hyperbolic(
        self,
        shape: Tuple[int, ...],
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        if generator is not None:
            u = torch.rand(shape, device=self.device, dtype=self.dtype, generator=generator)
        else:
            u = torch.rand(shape, device=self.device, dtype=self.dtype)
        # Clamp to avoid ±inf at u=0 or u=1
        u = u.clamp(1e-6, 1.0 - 1e-6)
        return torch.tan(math.pi * (u - 0.5))

    def fit(
        self,
        X: Any,
        y: Optional[Any] = None,
        **kwargs,
    ) -> "SkewedChi2Sampler":
        X = _to_float_tensor(X, self.device, self.dtype)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        g = _make_generator(self.random_state, self.device)

        # Sample random projection directions from secant-hyperbolic distribution
        # scaled by 1/skewedness (to match the kernel parameter)
        raw_weights = self._sample_secant_hyperbolic(
            (n_features, self.n_components), g
        )
        self.random_weights_ = raw_weights / max(abs(self.skewedness), 1e-8)

        # Uniform offset in [0, 2π]
        if g is not None:
            self.random_offset_ = (
                torch.rand(self.n_components, device=self.device, dtype=self.dtype, generator=g)
                * 2.0 * math.pi
            )
        else:
            self.random_offset_ = (
                torch.rand(self.n_components, device=self.device, dtype=self.dtype)
                * 2.0 * math.pi
            )

        self.fit_status = True
        return self

    def transform(
        self,
        X: Any,
        **kwargs,
    ) -> torch.Tensor:
        if not self.fit_status:
            raise RuntimeError(
                "SkewedChi2Sampler is not fitted. Call fit() first."
            )
        X = _to_float_tensor(X, self.device, self.dtype)
        n_samples, n_features = X.shape
        if n_features != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {n_features}."
            )

        min_val = X.min().item()
        if min_val <= -self.skewedness:
            raise ValueError(
                f"All X values must be strictly greater than -skewedness "
                f"(= {-self.skewedness:.4g}). Got min={min_val:.4g}."
            )

        log_X = torch.log(X + self.skewedness)                    # (n, d)
        projection = log_X @ self.random_weights_ + self.random_offset_  # (n, D)
        scale = math.sqrt(2.0 / self.n_components)
        return scale * torch.cos(projection)

    def fit_transform(
        self,
        X: Any,
        y: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Fit and transform X."""
        return self.fit(X, y, **kwargs).transform(X)

    def forward(
        self,
        X: Any,
        y: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        if not self.fit_status:
            return self.fit_transform(X, y, **kwargs)
        return self.transform(X, **kwargs)


# ---------------------------------------------------------------------------
# Nystroem
# ---------------------------------------------------------------------------

class Nystroem(MLModule):
    def __init__(self,
                 kernel: Union[str, MLModule, Callable] = "rbf",
                 gamma: float = None,
                 coef0: float = None,
                 degree: float = None,
                 kernel_params: dict = None,
                 n_components: int = 100,
                 random_state: Union[int, torch.Generator] = None,
                 n_jobs: int = None,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.kernel_params = kernel_params if kernel_params is not None else {}
        self.n_components = n_components
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.args = args
        self.kwargs = kwargs
        # Fitted attributes
        self.components_: Optional[torch.Tensor] = None
        self.component_indices_: Optional[torch.Tensor] = None
        self.normalization_: Optional[torch.Tensor] = None
        self.n_features_in_: Optional[int] = None
        self.fit_status = False

    # ------------------------------------------------------------------
    # Kernel computation
    # ------------------------------------------------------------------

    def _build_kernel_fn(self) -> Callable:
        kernel = self.kernel
        gamma = self.gamma if self.gamma is not None else 1.0
        degree = self.degree if self.degree is not None else 3.0
        coef0 = self.coef0 if self.coef0 is not None else 1.0

        if callable(kernel) and not isinstance(kernel, str):
            # User-supplied callable or MLModule kernel
            def _user_kernel(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
                params = dict(self.kernel_params)
                return kernel(X, Y, **params)
            return _user_kernel

        k_name = str(kernel).lower()

        # Map scikit-learn / common names to registry keys
        _name_map = {
            "polynomial": "poly",
            "additive_chi2": "chi_square",
            "chi2": "chi_square",
            "cosine": None,    # handled separately
        }
        k_name = _name_map.get(k_name, k_name)

        if k_name == "rbf":
            kern = RBFKernel(gamma=gamma)
        elif k_name == "poly":
            kern = PolyKernel(degree=degree, gamma=gamma, bias=coef0)
        elif k_name == "sigmoid":
            kern = SigmoidKernel(gamma=gamma, bias=coef0)
        elif k_name == "linear":
            kern = LinearKernel()
        elif k_name == "laplacian":
            sigma = 1.0 / max(gamma, 1e-8)
            kern = LaplacianKernel(sigma=sigma)
        elif k_name == "chi_square":
            kern = ChiSquareKernel(gamma=gamma)
        elif k_name == "hist_intersection":
            kern = HistogramIntersectionKernel()
        elif k_name is None:
            # Cosine kernel: K(x,y) = x^T y / (||x|| ||y||)
            def _cosine(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
                Xn = F.normalize(X, p=2, dim=1)
                Yn = F.normalize(Y, p=2, dim=1)
                return Xn @ Yn.T
            return _cosine
        else:
            # Try the KernelRegistry
            klass = get_kernel_class(k_name)
            if klass is not None:
                try:
                    kern = klass(gamma=gamma, **self.kernel_params)
                except TypeError:
                    kern = klass(**self.kernel_params)
            else:
                raise ValueError(
                    f"Unknown kernel '{kernel}'. Available: rbf, poly, sigmoid, "
                    f"linear, laplacian, chi2, hist_intersection, cosine, or any "
                    f"registered kernel name."
                )

        kern = kern.to(self.device)

        def _registered_kernel(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return kern(X, Y)

        return _registered_kernel

    # ------------------------------------------------------------------

    def fit(
        self,
        X: Any,
        y: Optional[Any] = None,
        **kwargs,
    ) -> "Nystroem":
        X = _to_float_tensor(X, self.device, self.dtype)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        g = _make_generator(self.random_state, self.device)

        # Sample n_components basis points (without replacement when possible)
        n_comp = min(self.n_components, n_samples)
        if g is not None:
            perm = torch.randperm(n_samples, device=self.device, generator=g)
        else:
            perm = torch.randperm(n_samples, device=self.device)
        idx = perm[:n_comp]

        self.component_indices_ = idx
        self.components_ = X[idx]                              # (n_comp, d)

        # Compute kernel matrix K_mm on the basis points
        kernel_fn = self._build_kernel_fn()
        K_mm = kernel_fn(self.components_, self.components_)  # (n_comp, n_comp)
        K_mm = K_mm.to(dtype=torch.float64)                   # higher precision for SVD

        # Symmetrize (numerical safety)
        K_mm = 0.5 * (K_mm + K_mm.T)

        # SVD: K_mm = U S Vt  (for symmetric PSD: U ≈ Vt.T)
        U, S, Vh = torch.linalg.svd(K_mm, full_matrices=False)

        # K_mm^(-1/2) = Vh.T @ diag(1/sqrt(S)) @ Vh
        S_inv_sqrt = torch.clamp(S, min=1e-12).pow(-0.5)     # (n_comp,)
        self.normalization_ = (Vh.T * S_inv_sqrt) @ Vh        # (n_comp, n_comp)
        self.normalization_ = self.normalization_.to(dtype=self.dtype)

        self.fit_status = True
        return self

    def transform(
        self,
        X: Any,
        **kwargs,
    ) -> torch.Tensor:
        if not self.fit_status:
            raise RuntimeError("Nystroem is not fitted. Call fit() first.")
        X = _to_float_tensor(X, self.device, self.dtype)
        n_samples, n_features = X.shape
        if n_features != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {n_features}."
            )
        kernel_fn = self._build_kernel_fn()
        K_nm = kernel_fn(X, self.components_)              # (n_samples, n_comp)
        return K_nm.to(dtype=self.dtype) @ self.normalization_

    def fit_transform(
        self,
        X: Any,
        y: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Fit to X, then transform X."""
        return self.fit(X, y, **kwargs).transform(X)

    def forward(
        self,
        X: Any,
        y: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        if not self.fit_status:
            return self.fit_transform(X, y, **kwargs)
        return self.transform(X, **kwargs)


# ---------------------------------------------------------------------------
# RBFSampler
# ---------------------------------------------------------------------------

class RBFSampler(MLModule):
    def __init__(self,
                 gamma: Union[float, Literal["scale"]] = None,
                 n_components: int = 100,
                 random_state: Union[int, torch.Generator] = None,
                 n_jobs: int = None,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.args = args
        self.kwargs = kwargs
        # Fitted attributes
        self.random_weights_: Optional[torch.Tensor] = None
        self.random_offset_: Optional[torch.Tensor] = None
        self.n_features_in_: Optional[int] = None
        self._gamma_fitted: Optional[float] = None
        self.fit_status = False

    def _resolve_gamma(self, X: torch.Tensor) -> float:
        """Resolve gamma from self.gamma and (if 'scale') from X."""
        g = self.gamma
        if g is None or (isinstance(g, float) and g == 1.0):
            return 1.0
        if isinstance(g, str) and g.lower() == "scale":
            n_features = X.shape[1]
            var = X.var().item()
            return 1.0 / max(n_features * var, 1e-8)
        return float(g)

    def fit(
        self,
        X: Any,
        y: Optional[Any] = None,
        **kwargs,
    ) -> "RBFSampler":
        X = _to_float_tensor(X, self.device, self.dtype)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        g = _make_generator(self.random_state, self.device)
        gamma = self._resolve_gamma(X)
        self._gamma_fitted = gamma

        # Draw random frequency vectors: W ~ N(0, 2γ * I)
        if g is not None:
            W = torch.randn(n_features, self.n_components, device=self.device, dtype=self.dtype, generator=g)
            b = torch.rand(self.n_components, device=self.device, dtype=self.dtype, generator=g) * 2.0 * math.pi
        else:
            W = torch.randn(n_features, self.n_components, device=self.device, dtype=self.dtype)
            b = torch.rand(self.n_components, device=self.device, dtype=self.dtype) * 2.0 * math.pi

        self.random_weights_ = math.sqrt(2.0 * gamma) * W  # absorb sqrt(2γ) into weights
        self.random_offset_ = b

        self.fit_status = True
        return self

    def transform(
        self,
        X: Any,
        **kwargs,
    ) -> torch.Tensor:
        if not self.fit_status:
            raise RuntimeError("RBFSampler is not fitted. Call fit() first.")
        X = _to_float_tensor(X, self.device, self.dtype)
        n_samples, n_features = X.shape
        if n_features != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {n_features}."
            )
        projection = X @ self.random_weights_ + self.random_offset_   # (n, D)
        scale = math.sqrt(2.0 / self.n_components)
        return scale * torch.cos(projection)

    def fit_transform(
        self,
        X: Any,
        y: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Fit and transform X."""
        return self.fit(X, y, **kwargs).transform(X)

    def forward(
        self,
        X: Any,
        y: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        if not self.fit_status:
            return self.fit_transform(X, y, **kwargs)
        return self.transform(X, **kwargs)


# ---------------------------------------------------------------------------
# PolynomialCountSketch
# ---------------------------------------------------------------------------

class PolynomialCountSketch(MLModule):
    def __init__(self,
                 gamma: float = None,
                 coef0: float = None,
                 degree: float = None,
                 n_components: int = 100,
                 random_state: Union[int, torch.Generator] = None,
                 n_jobs: int = None,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.gamma = gamma if gamma is not None else 1.0
        self.coef0 = coef0 if coef0 is not None else 0.0
        self.degree = int(degree) if degree is not None else 2
        self.n_components = n_components
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.args = args
        self.kwargs = kwargs
        # Fitted attributes
        self.indexHash_: Optional[torch.Tensor] = None
        self.bitHash_: Optional[torch.Tensor] = None
        self.n_features_in_: Optional[int] = None
        # Augmented feature count (n_features + 1 when coef0 != 0)
        self._n_aug_features: Optional[int] = None
        self.fit_status = False

    # ------------------------------------------------------------------
    # Count Sketch
    # ------------------------------------------------------------------

    def _count_sketch(
        self,
        X: torch.Tensor,
        index_hash: torch.Tensor,
        bit_hash: torch.Tensor,
    ) -> torch.Tensor:
        n_samples, n_features = X.shape
        sketch = torch.zeros(n_samples, self.n_components, device=self.device, dtype=self.dtype)
        signed_X = X * bit_hash.unsqueeze(0)           # (n, d) * (1, d) → (n, d)
        sketch.scatter_add_(1, index_hash.unsqueeze(0).expand(n_samples, -1), signed_X)
        return sketch

    def _fft_convolve(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        fa = torch.fft.rfft(a, n=self.n_components)
        fb = torch.fft.rfft(b, n=self.n_components)
        return torch.fft.irfft(fa * fb, n=self.n_components)

    def fit(
        self,
        X: Any,
        y: Optional[Any] = None,
        **kwargs,
    ) -> "PolynomialCountSketch":
        X = _to_float_tensor(X, self.device, self.dtype)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        g = _make_generator(self.random_state, self.device)

        # If coef0 != 0, we augment X with sqrt(coef0) as an extra feature
        n_aug = n_features + (1 if abs(self.coef0) > 1e-12 else 0)
        self._n_aug_features = n_aug

        # indexHash_: (degree, n_aug)  — random bucket indices in [0, n_components)
        # bitHash_:   (degree, n_aug)  — random signs {-1, +1}
        if g is not None:
            rand_idx = torch.randint(
                0, self.n_components,
                (self.degree, n_aug),
                device=self.device, generator=g,
            )
            rand_bits = (
                torch.randint(0, 2, (self.degree, n_aug), device=self.device, generator=g)
                .float() * 2.0 - 1.0
            )
        else:
            rand_idx = torch.randint(0, self.n_components, (self.degree, n_aug), device=self.device)
            rand_bits = torch.randint(0, 2, (self.degree, n_aug), device=self.device).float() * 2.0 - 1.0

        self.indexHash_ = rand_idx
        self.bitHash_ = rand_bits.to(dtype=self.dtype)

        self.fit_status = True
        return self

    def _augment_X(self, X: torch.Tensor) -> torch.Tensor:
        X_scaled = X * math.sqrt(self.gamma)
        if abs(self.coef0) > 1e-12:
            bias = torch.full(
                (X.shape[0], 1), math.sqrt(self.coef0),
                device=self.device, dtype=self.dtype,
            )
            X_scaled = torch.cat([X_scaled, bias], dim=1)
        return X_scaled

    def transform(
        self,
        X: Any,
        **kwargs,
    ) -> torch.Tensor:
        if not self.fit_status:
            raise RuntimeError("PolynomialCountSketch is not fitted. Call fit() first.")
        X = _to_float_tensor(X, self.device, self.dtype)
        n_samples, n_features = X.shape
        if n_features != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {n_features}."
            )

        X_aug = self._augment_X(X)    # (n, n_aug)

        # Degree-0 sketch: uniform ones (identity for convolution)
        sketch = torch.ones(n_samples, self.n_components, device=self.device, dtype=self.dtype)

        for d in range(self.degree):
            sk_d = self._count_sketch(X_aug, self.indexHash_[d], self.bitHash_[d])
            sketch = self._fft_convolve(sketch, sk_d)

        return sketch

    def fit_transform(
        self,
        X: Any,
        y: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Fit and transform X."""
        return self.fit(X, y, **kwargs).transform(X)

    def forward(
        self,
        X: Any,
        y: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        if not self.fit_status:
            return self.fit_transform(X, y, **kwargs)
        return self.transform(X, **kwargs)

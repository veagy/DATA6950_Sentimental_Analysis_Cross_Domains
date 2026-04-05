import warnings
import math
import torch
import torch.nn as nn
from typing import Optional, Callable, Union, Any, List, Tuple, Dict, Literal
from .....models.utils import MLModule
import pandas as pd  # pyright: ignore[reportMissingImports]
import numpy as np
from ...regression.knn.knn import BallTree, KDTree
from torch.func import vmap
import joblib


__all__ = [
    "KernelDensity",
]


class KernelDensity(MLModule):
    def __init__(self,
                 bandwidth: Union[Literal["scott", "silverman"], float]=1.0,
                 algorithm: Union[Literal["kd_tree", "ball_tree", "auto"],
                    Callable, nn.Module]='auto',
                 kernel: Union[Literal["gaussian", "tophat", "epanechnikov",
                    "exponential", "linear", "cosine"],
                    str, Callable, nn.Module]='gaussian',
                 metric: Union[str, Callable, nn.Module]='euclidean',
                 atol: float=0,
                 rtol: float=0,
                 breadth_first: bool=True,
                 leaf_size: int=40,
                 metric_params: dict=None,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.bandwidth = bandwidth
        self.algorithm = algorithm
        self.kernel = kernel
        self.metric = metric
        self.atol = atol
        self.rtol = rtol
        self.breadth_first = breadth_first
        self.leaf_size = leaf_size
        self.metric_params = metric_params if metric_params is not None else {}
        self.device = device
        self.dtype = dtype

        # Fitted attributes
        self.n_features_in_: Optional[int] = None
        self.tree_: Optional[Any] = None
        self.bandwidth_: Optional[float] = None
        self._X_fit: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_algorithm(self, n_samples: int, n_features: int) -> str:
        """Choose the tree algorithm based on the algorithm parameter."""
        alg = self.algorithm
        if callable(alg) or isinstance(alg, nn.Module):
            return 'callable'
        alg_str = str(alg).lower()
        if alg_str in ('kd_tree', 'ball_tree'):
            return alg_str
        # 'auto': use kd_tree for low-dim, ball_tree for high-dim
        return 'kd_tree' if n_features <= 20 else 'ball_tree'

    def _get_metric_fn(self) -> Optional[Callable]:
        """Return a callable distance function from self.metric."""
        metric = self.metric
        if callable(metric) and not isinstance(metric, str):
            params = self.metric_params or {}
            return lambda xi, xj: metric(xi, xj, **params)
        if isinstance(metric, str):
            m = metric.lower()
            p = (self.metric_params or {}).get('p', 2)
            if m in ('euclidean', 'l2'):
                return lambda xi, xj: torch.cdist(xi, xj, p=2)
            elif m in ('manhattan', 'l1', 'cityblock'):
                return lambda xi, xj: torch.cdist(xi, xj, p=1)
            elif m == 'minkowski':
                return lambda xi, xj: torch.cdist(xi, xj, p=p)
            elif m == 'chebyshev':
                return lambda xi, xj: torch.cdist(xi, xj, p=float('inf'))
            else:
                return lambda xi, xj: torch.cdist(xi, xj, p=2)
        return None

    def _compute_distances(self, X_query: torch.Tensor,
                           X_train: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distances (n_query, n_train) using self.metric."""
        metric = self.metric
        if callable(metric) and not isinstance(metric, str):
            params = self.metric_params or {}
            D = metric(X_query, X_train, **params)
            return D.to(dtype=self.dtype)
        if isinstance(metric, str):
            m = metric.lower()
            p = (self.metric_params or {}).get('p', 2)
            if m in ('euclidean', 'l2'):
                return torch.cdist(X_query, X_train, p=2).to(self.dtype)
            elif m in ('manhattan', 'l1', 'cityblock'):
                return torch.cdist(X_query, X_train, p=1).to(self.dtype)
            elif m == 'minkowski':
                return torch.cdist(X_query, X_train, p=p).to(self.dtype)
            elif m == 'chebyshev':
                return torch.cdist(X_query, X_train, p=float('inf')).to(self.dtype)
            elif m == 'cosine':
                xq_n = torch.nn.functional.normalize(X_query, p=2, dim=-1)
                xt_n = torch.nn.functional.normalize(X_train, p=2, dim=-1)
                return (1.0 - xq_n @ xt_n.T).to(self.dtype)
            else:
                return torch.cdist(X_query, X_train, p=2).to(self.dtype)
        return torch.cdist(X_query, X_train, p=2).to(self.dtype)

    def _log_kernel(self, D: torch.Tensor, h: float, d: int) -> torch.Tensor:
        """Compute per-pair log kernel values.

        Parameters
        ----------
        D : torch.Tensor, shape (n_query, n_train)
            Pairwise distances.
        h : float
            Bandwidth.
        d : int
            Number of features (dimensionality).

        Returns
        -------
        log_k : torch.Tensor, shape (n_query, n_train)
            Log-kernel values. -inf for pairs outside kernel support.
        """
        kernel = self.kernel
        kernel_str = kernel if isinstance(kernel, str) else None

        # Log of d-ball volume: log(pi^(d/2) / Gamma(d/2+1))
        log_ball_vol = 0.5 * d * math.log(math.pi) - math.lgamma(0.5 * d + 1)

        if kernel_str == 'gaussian' or kernel_str is None and not callable(kernel):
            # log K_h(u) = -(d/2)*log(2π) - d*log(h) - 0.5*(D/h)^2
            log_norm = -0.5 * d * math.log(2.0 * math.pi) - d * math.log(h)
            return log_norm - 0.5 * (D / h) ** 2

        elif kernel_str == 'tophat':
            # K_h(u) = 1/(vol_d * h^d) if D <= h, else 0
            log_norm = -log_ball_vol - d * math.log(h)
            result = torch.full_like(D, float('-inf'))
            mask = D <= h
            result[mask] = log_norm
            return result

        elif kernel_str == 'epanechnikov':
            # K_h(u) = (d+2)/(2*vol_d*h^d) * (1 - (D/h)^2), D <= h
            log_norm = (math.log(d + 2) - math.log(2.0)
                        - log_ball_vol - d * math.log(h))
            result = torch.full_like(D, float('-inf'))
            mask = D <= h
            if mask.any():
                u2 = (D[mask] / h) ** 2
                inner = (1.0 - u2).clamp(min=1e-12)
                result[mask] = log_norm + torch.log(inner)
            return result

        elif kernel_str == 'exponential':
            # Multivariate exponential kernel:
            # K_h(u) ∝ exp(-||u||/h), normalized over R^d
            # Normalization: integral of exp(-r/h) r^(d-1) over r in [0,∞) × S_{d-1}
            # = h^d * Gamma(d) * 2*pi^(d/2) / Gamma(d/2)
            # = h^d * d * ball_vol
            log_norm = -math.log(d) - log_ball_vol - d * math.log(h)
            return log_norm - D / h

        elif kernel_str == 'linear':
            # K_h(u) = (d+1)/(vol_d*h^d) * (1 - D/h), D <= h
            log_norm = (math.log(d + 1) - log_ball_vol - d * math.log(h))
            result = torch.full_like(D, float('-inf'))
            mask = D <= h
            if mask.any():
                u = D[mask] / h
                inner = (1.0 - u).clamp(min=1e-12)
                result[mask] = log_norm + torch.log(inner)
            return result

        elif kernel_str == 'cosine':
            # K_h(u) ∝ cos(π*D/(2*h)), D <= h
            # Normalization in d dims involves the integral of cos(π*r/(2h))*r^(d-1)
            # Approximate normalization: use d-sphere surface and integrate radially
            # integral_0^h cos(πr/(2h)) * (d * vol_d / h^d) * r^(d-1) dr
            # Use numerical approach: log_norm = log(pi/(4*h)) for 1-d approx
            # For d dims, multiply by surface area factor
            log_norm = -log_ball_vol - d * math.log(h) + math.log(math.pi / 4.0)
            result = torch.full_like(D, float('-inf'))
            mask = D <= h
            if mask.any():
                u = D[mask] / h
                cos_val = torch.cos(math.pi / 2.0 * u).clamp(min=1e-12)
                result[mask] = log_norm + torch.log(cos_val)
            return result

        elif callable(kernel):
            # Custom kernel callable: kernel(D, h, d) -> log kernel values
            try:
                return kernel(D, h, d)
            except TypeError:
                return kernel(D)

        elif isinstance(kernel, nn.Module):
            return kernel(D)

        else:
            # Default to Gaussian
            log_norm = -0.5 * d * math.log(2.0 * math.pi) - d * math.log(h)
            return log_norm - 0.5 * (D / h) ** 2

    def _sample_kernel_noise(self, n_samples: int, d: int, h: float,
                              gen: Optional[torch.Generator]) -> torch.Tensor:
        """Draw noise samples from the kernel distribution.

        Returns
        -------
        noise : torch.Tensor, shape (n_samples, d)
        """
        kernel = self.kernel
        kernel_str = kernel if isinstance(kernel, str) else None
        dev = self.device

        def _randn():
            if gen is not None:
                return torch.randn(n_samples, d, generator=gen,
                                   device=dev, dtype=self.dtype)
            return torch.randn(n_samples, d, device=dev, dtype=self.dtype)

        def _rand():
            if gen is not None:
                return torch.rand(n_samples, device=dev,
                                  dtype=self.dtype, generator=gen)
            return torch.rand(n_samples, device=dev, dtype=self.dtype)

        if kernel_str == 'gaussian' or kernel_str is None:
            return _randn() * h

        elif kernel_str == 'tophat':
            # Uniform in d-ball of radius h via Muller / Marsaglia method
            z = _randn()
            norms = z.norm(dim=1, keepdim=True).clamp(min=1e-12)
            # Radial CDF inverse: r = h * u^(1/d)
            u = _rand().unsqueeze(1)
            r = h * u.pow(1.0 / d)
            return z / norms * r

        elif kernel_str == 'epanechnikov':
            # Acceptance-rejection: uniform in d-ball, accept ∝ (1 - ||u/h||^2)
            batch = max(n_samples * 4, 1000)
            collected: List[torch.Tensor] = []
            total = 0
            while total < n_samples:
                if gen is not None:
                    z = torch.rand(batch, d, generator=gen, device=dev,
                                   dtype=self.dtype) * 2 - 1
                    u_accept = torch.rand(batch, generator=gen, device=dev,
                                         dtype=self.dtype)
                else:
                    z = torch.rand(batch, d, device=dev, dtype=self.dtype) * 2 - 1
                    u_accept = torch.rand(batch, device=dev, dtype=self.dtype)
                r2 = (z ** 2).sum(dim=1)
                in_ball = r2 <= 1.0
                prob = (1.0 - r2[in_ball]).clamp(min=0.0)
                if prob.numel() == 0:
                    continue
                accept = u_accept[in_ball] <= prob
                pts = z[in_ball][accept] * h
                collected.append(pts)
                total += pts.shape[0]
            return torch.cat(collected, dim=0)[:n_samples]

        elif kernel_str == 'exponential':
            # Exponential radial profile in d dims:
            # sample direction uniform on S^(d-1), radius from Gamma(d, h)
            z = _randn()
            direction = z / z.norm(dim=1, keepdim=True).clamp(min=1e-12)
            # Gamma(d, h): use sum of d exponentials
            if gen is not None:
                exp_samples = torch.zeros(n_samples, device=dev, dtype=self.dtype)
                for _ in range(d):
                    u = torch.rand(n_samples, generator=gen, device=dev,
                                   dtype=self.dtype).clamp(min=1e-12)
                    exp_samples -= h * u.log()
            else:
                u = torch.rand(n_samples, d, device=dev,
                               dtype=self.dtype).clamp(min=1e-12)
                exp_samples = -h * u.log().sum(dim=1)
            r = exp_samples.unsqueeze(1)
            return direction * r

        elif kernel_str == 'linear':
            # Triangular radial profile: f(r) ∝ (1 - r/h) * r^(d-1), r in [0, h]
            # Use acceptance-rejection with uniform proposal on d-ball
            batch = max(n_samples * 4, 1000)
            collected: List[torch.Tensor] = []
            total = 0
            while total < n_samples:
                if gen is not None:
                    z = torch.rand(batch, d, generator=gen, device=dev,
                                   dtype=self.dtype) * 2 - 1
                    u_accept = torch.rand(batch, generator=gen, device=dev,
                                         dtype=self.dtype)
                else:
                    z = torch.rand(batch, d, device=dev, dtype=self.dtype) * 2 - 1
                    u_accept = torch.rand(batch, device=dev, dtype=self.dtype)
                r = z.norm(dim=1)
                in_ball = r <= 1.0
                prob = (1.0 - r[in_ball]).clamp(min=0.0)
                if prob.numel() == 0:
                    continue
                accept = u_accept[in_ball] <= prob
                pts = z[in_ball][accept] * h
                collected.append(pts)
                total += pts.shape[0]
            return torch.cat(collected, dim=0)[:n_samples]

        elif kernel_str == 'cosine':
            # Cosine radial profile: f(r) ∝ cos(πr/(2h)), r in [0, h]
            # Acceptance-rejection with uniform on d-ball
            batch = max(n_samples * 4, 1000)
            collected: List[torch.Tensor] = []
            total = 0
            while total < n_samples:
                if gen is not None:
                    z = torch.rand(batch, d, generator=gen, device=dev,
                                   dtype=self.dtype) * 2 - 1
                    u_accept = torch.rand(batch, generator=gen, device=dev,
                                         dtype=self.dtype)
                else:
                    z = torch.rand(batch, d, device=dev, dtype=self.dtype) * 2 - 1
                    u_accept = torch.rand(batch, device=dev, dtype=self.dtype)
                r = z.norm(dim=1)
                in_ball = r <= 1.0
                cos_p = torch.cos(math.pi / 2.0 * r[in_ball]).clamp(min=0.0)
                if cos_p.numel() == 0:
                    continue
                accept = u_accept[in_ball] <= cos_p
                pts = z[in_ball][accept] * h
                collected.append(pts)
                total += pts.shape[0]
            return torch.cat(collected, dim=0)[:n_samples]

        else:
            # Default to Gaussian for custom/unknown kernels
            return _randn() * h

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, data_or_X, y=None, **kwargs) -> "KernelDensity":
        """Fit the Kernel Density model on the training data.

        Parameters
        ----------
        data_or_X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored

        Returns
        -------
        self : KernelDensity
            Fitted estimator.
        """
        if isinstance(data_or_X, (pd.DataFrame, pd.Series)):
            data_or_X = data_or_X.values
        if isinstance(data_or_X, np.ndarray):
            data_or_X = torch.from_numpy(data_or_X)
        X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(1)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self._X_fit = X

        # Resolve bandwidth
        if isinstance(self.bandwidth, str):
            bw = self.bandwidth.lower()
            if bw == 'scott':
                # Scott's rule: h = n^(-1/(d+4))
                self.bandwidth_ = float(n_samples ** (-1.0 / (n_features + 4)))
            elif bw == 'silverman':
                # Silverman's rule: h = (n*(d+2)/4)^(-1/(d+4))
                self.bandwidth_ = float(
                    (n_samples * (n_features + 2) / 4.0) ** (-1.0 / (n_features + 4))
                )
            else:
                raise ValueError(
                    f"Unknown bandwidth estimation method '{self.bandwidth}'. "
                    "Valid string values are 'scott' and 'silverman'."
                )
        else:
            self.bandwidth_ = float(self.bandwidth)

        if self.bandwidth_ <= 0:
            raise ValueError(f"bandwidth must be positive, got {self.bandwidth_}")

        # Build spatial index
        resolved_alg = self._resolve_algorithm(n_samples, n_features)
        metric_fn = self._get_metric_fn()
        if resolved_alg == 'kd_tree':
            self.tree_ = KDTree(
                X, leaf_size=self.leaf_size, metric=metric_fn,
                device=str(self.device), dtype=self.dtype
            )
        elif resolved_alg == 'ball_tree':
            self.tree_ = BallTree(
                X, leaf_size=self.leaf_size, metric=metric_fn,
                device=str(self.device), dtype=self.dtype
            )
        else:
            # callable algorithm or fallback (brute force)
            self.tree_ = None

        self.fit_status = True
        return self

    def score_samples(self, X) -> torch.Tensor:
        """Compute the log-probability density of each sample under the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            An array of points to query. Last dimension should match dimension
            of training data (n_features).

        Returns
        -------
        density : torch.Tensor of shape (n_samples,)
            Log-density of each sample in X. These are normalized to be
            probability densities, so values will be low for high-dimensional
            data.
        """
        if not self.fit_status:
            raise RuntimeError("KernelDensity is not fitted yet. Call fit() first.")

        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)

        n_train = self._X_fit.shape[0]
        d = self.n_features_in_
        h = self.bandwidth_

        # Compute pairwise distances (n_query, n_train)
        D = self._compute_distances(X, self._X_fit)

        # Apply tolerance thresholding (atol/rtol based pruning)
        # Points with D > h*(1 + rtol) + atol contribute negligibly for
        # tophat/epanechnikov/linear/cosine kernels; for Gaussian, always include all.
        # For efficiency: filter when using kernels with compact support.
        kernel_str = self.kernel if isinstance(self.kernel, str) else 'gaussian'
        if kernel_str in ('tophat', 'epanechnikov', 'linear', 'cosine') and (
                self.atol > 0 or self.rtol > 0):
            cutoff = h * (1.0 + self.rtol) + self.atol
            D = D.clamp(max=cutoff + 1e-12)

        # Log kernel values: (n_query, n_train)
        log_k = self._log_kernel(D, h, d)

        # log p(x) = logsumexp(log_k, dim=1) - log(n_train)
        log_dens = torch.logsumexp(log_k, dim=1) - math.log(n_train)
        return log_dens

    def score(self, X, y=None) -> float:
        """Compute the total log-probability under the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            An array of points to query.
        y : ignored

        Returns
        -------
        log_likelihood : float
            Total log-likelihood of the data in X. This is normalized to be a
            probability density, so the result is low for high-dimensional data.
        """
        return float(self.score_samples(X).sum().item())

    def sample(self, n_samples: int = 1,
               random_state: Union[int, torch.Generator, None] = None
               ) -> torch.Tensor:
        """Generate random samples from the fitted kernel density estimate.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.
        random_state : int, torch.Generator, or None
            Seed or generator for reproducibility.

        Returns
        -------
        X : torch.Tensor of shape (n_samples, n_features)
            Generated samples from the fitted kernel density estimate.
        """
        if not self.fit_status:
            raise RuntimeError("KernelDensity is not fitted yet. Call fit() first.")

        # Set up random generator
        gen: Optional[torch.Generator] = None
        if random_state is not None:
            if isinstance(random_state, torch.Generator):
                gen = random_state
            else:
                gen = torch.Generator(device=self.device)
                gen.manual_seed(int(random_state))

        n_train = self._X_fit.shape[0]
        d = self.n_features_in_

        # Draw random training-point indices (resampling step)
        if gen is not None:
            idx = torch.randint(0, n_train, (n_samples,),
                                generator=gen, device=self.device)
        else:
            idx = torch.randint(0, n_train, (n_samples,), device=self.device)

        centers = self._X_fit[idx]  # (n_samples, d)

        # Sample noise from the kernel distribution
        noise = self._sample_kernel_noise(n_samples, d, self.bandwidth_, gen)
        noise = noise.to(device=self.device, dtype=self.dtype)

        return centers + noise

    def forward(self, X) -> torch.Tensor:
        """Return log-density for X (same as score_samples)."""
        return self.score_samples(X)

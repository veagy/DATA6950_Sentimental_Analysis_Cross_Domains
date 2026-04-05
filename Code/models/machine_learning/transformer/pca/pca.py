import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import lgamma, log
from typing import Optional, Callable, Union, Any, List, Tuple, Dict, Literal
from .....models.utils import MLTransform
from .....models.machine_learning.regression.svm.kernels import get_kernel_class
from torch.func import vmap
import joblib

__all__ = [
    "PCA",
    "KernelPCA",
    "SparsePCA",
    "MiniBatchSparsePCA",
    "IncrementalPCA",
]


class PCA(MLTransform):
    def __init__(
        self,
        n_components: Optional[Union[int, float, Literal["mle"]]] = None,
        copy: bool = True,
        whiten: bool = False,
        svd_solver: Union[
            Literal["auto", "full", "covariance_eigh", "arpack", "randomized"],
            Callable[..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
            nn.Module,
        ] = "auto",
        tol: float = 0.0,
        iterated_power: Union[int, Literal["auto"]] = "auto",
        n_oversamples: int = 10,
        power_iteration_normalizer: Union[
            Literal["auto", "QR", "LU", "none"],
            Callable[..., Any],
            nn.Module,
        ] = "auto",
        random_state: Optional[Union[int, torch.Generator]] = None,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float,
        svd_solver_kwargs: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.svd_solver = svd_solver
        self.whiten = whiten
        self.copy = copy
        self.n_components = n_components
        self.tol = tol
        self.iterated_power = iterated_power if isinstance(iterated_power, str) else abs(int(iterated_power))
        self.n_oversamples = n_oversamples
        self.power_iteration_normalizer = power_iteration_normalizer
        # Additional kwargs for custom svd_solver (callable or nn.Module)
        solver_kw = svd_solver_kwargs or kwargs.pop("svd_solver_kwargs", None)
        self._svd_solver_kwargs = dict(solver_kw) if solver_kw is not None else dict(kwargs)
        if random_state is not None:
            torch.manual_seed(random_state)
            self.random_state = torch.Generator().manual_seed(random_state)
        else:
            self.random_state = None
        self.device = device
        self.dtype = dtype
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.n_components_ = None
        self.n_samples_ = None
        self.noise_variance_ = None
        self.n_features_in_ = None

    def _init_module(self, X: torch.Tensor) -> None:
        """
        Initializes the PCA parameters based on the input data shape and
        the provided arguments.
        """
        # 1. Capture basic input attributes
        self.n_samples_, self.n_features_in_ = X.shape[-2], X.shape[-1]

        # 2. Handle custom svd_solver (callable or nn.Module) vs string
        is_custom_solver = callable(self.svd_solver) or isinstance(self.svd_solver, nn.Module)
        if is_custom_solver:
            if self.n_components == 'mle':
                raise ValueError(
                    "n_components='mle' is only supported by built-in svd_solver='full'"
                )
            if isinstance(self.n_components, float) and 0 < self.n_components < 1:
                raise ValueError(
                    "n_components in (0, 1) (variance ratio) is only supported by svd_solver='full'"
                )
        # 2b. Handle 'auto' solver logic for MLE
        elif self.n_components == 'mle':
            if self.svd_solver == 'auto':
                self.svd_solver = 'full'
            if self.svd_solver != 'full':
                raise ValueError("n_components='mle' is only supported by svd_solver='full'")

        # 3. Determine the actual number of components (n_components_)
        if self.n_components is None:
            # Docstring: "Hence, the None case results in: n_components == min(n_samples, n_features) - 1"
            # This -1 applies specifically to arpack. For others, use min(n_samples, n_features).
            # Resolution happens after solver is known (step 4).
            self.n_components_ = None  # Resolved after solver selection
        elif self.n_components == 'mle':
            if self.n_samples_ < self.n_features_in_:
                raise ValueError(
                    "n_components='mle' is only supported if n_samples >= n_features"
                )
            self.n_components_ = 'mle'
        elif isinstance(self.n_components, float) and 0 < self.n_components < 1:
            if self.svd_solver != 'full':
                raise ValueError(
                    "n_components between 0 and 1 is only supported by svd_solver='full'"
                )
            self.n_components_ = self.n_components
        else:
            self.n_components_ = int(self.n_components)

        # 4. Resolve 'auto' SVD solver based on data shape and n_components_
        if is_custom_solver:
            actual_solver = self.svd_solver
        elif self.svd_solver == 'auto':
            # Policy: fewer than 1000 features and more than 10x samples -> covariance_eigh
            if self.n_features_in_ < 1000 and self.n_samples_ > 10 * self.n_features_in_:
                actual_solver = "covariance_eigh"
            # Policy: Data > 500x500 and n_components < 80% of min dimension -> randomized
            elif (self.n_samples_ > 500 and self.n_features_in_ > 500 and
                  not isinstance(self.n_components_, (str, float)) and  # Ensure it's an int for comparison
                  self.n_components_ < 0.8 * min(self.n_samples_, self.n_features_in_)):
                actual_solver = "randomized"
            else:
                actual_solver = "full"
        else:
            actual_solver = self.svd_solver

        # 5. Resolve n_components_ when it was None (depends on solver)
        if self.n_components_ is None:
            if isinstance(actual_solver, str) and actual_solver == 'arpack':
                self.n_components_ = min(self.n_samples_, self.n_features_in_) - 1
            else:
                self.n_components_ = min(self.n_samples_, self.n_features_in_)

        # 6. Final validation for specific solvers (skip for custom callable/nn.Module)
        if isinstance(actual_solver, str) and actual_solver == 'arpack':
            if not (0 < self.n_components_ < min(self.n_samples_, self.n_features_in_)):
                raise ValueError(
                    "arpack requires 0 < n_components < min(X.shape)"
                )

        # 7. Initialize the internal solver function
        self.solver_fn = self._calc_svd_solver(actual_solver, solver_kwargs={
            'tol': self.tol,
            'iterated_power': self.iterated_power,
            'n_oversamples': self.n_oversamples,
            'power_iteration_normalizer': self.power_iteration_normalizer,
            'random_state': self.random_state,
            'device': self.device,
            'dtype': self.dtype,
            'svd_solver_kwargs': self._svd_solver_kwargs,
        })

        # 8. Initialize mean_ (Per-feature empirical mean)
        self.mean_ = torch.mean(X, dim=-2)

    def _determine_components(
        self,
        n_components: Optional[Union[int, float, str]],
        n_samples: int,
        n_features: int,
    ) -> Union[int, float, str]:
        if n_components is None:
            # Arpack requires strictly less than min; others use min
            if self.svd_solver == 'arpack':
                return min(n_samples, n_features) - 1
            return min(n_samples, n_features)
        if isinstance(n_components, str) and n_components.lower() == "mle":
            if self.svd_solver != 'full':
                raise ValueError("MLE estimation requires svd_solver='full'")
            return "mle"
        if isinstance(n_components, float) and 0 < n_components < 1:
            if self.svd_solver != 'full':
                raise ValueError("Variance-based selection requires svd_solver='full'")
            return n_components
        if isinstance(n_components, (int, float)):
            n_comp_int = int(n_components)
            if self.svd_solver == 'arpack':
                limit = min(n_samples, n_features)
                if not (n_comp_int < limit):
                    raise ValueError(
                        "Arpack n_components must be strictly less than min(samples, features)"
                    )
            return n_comp_int
        return n_components

    def _calc_svd_solver(
        self,
        actual_solver: Union[
            Literal["auto", "full", "covariance_eigh", "arpack", "randomized"],
            Callable[..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
            nn.Module,
        ],
        solver_kwargs: Dict[str, Any],
    ) -> Callable[[torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # Custom callable or nn.Module: pass additional args from kwargs
        if callable(actual_solver) or isinstance(actual_solver, nn.Module):
            custom_kwargs = solver_kwargs.get("svd_solver_kwargs") or {}

            def solver_custom(X_centered, n_comp):
                out = actual_solver(X_centered, n_comp, **custom_kwargs)
                if isinstance(out, (list, tuple)) and len(out) >= 3:
                    U, S, Vh = out[0], out[1], out[2]
                else:
                    raise ValueError(
                        "Custom svd_solver must return (U, S, Vh) with U, S, Vh tensors"
                    )
                # Truncate if solver returned more than n_comp
                if U.shape[1] > n_comp or S.shape[0] > n_comp or Vh.shape[0] > n_comp:
                    return U[:, :n_comp], S[:n_comp], Vh[:n_comp, :]
                return U, S, Vh
            return solver_custom

        tol = solver_kwargs.get("tol", 0.0)
        iterated_power = solver_kwargs.get("iterated_power", "auto")
        n_oversamples = solver_kwargs.get("n_oversamples", 10)
        power_iteration_normalizer = solver_kwargs.get("power_iteration_normalizer", "auto")
        random_state = solver_kwargs.get("random_state")
        device = solver_kwargs.get("device", "cpu")
        dtype = solver_kwargs.get("dtype", torch.float64)

        def _get_n_iter():
            if iterated_power == "auto":
                return 2
            return int(iterated_power)

        def solver_full(X_centered, n_comp):
            # "Run exact full SVD calling the standard LAPACK solver via torch.linalg.svd"
            U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
            return U[:, :n_comp], S[:n_comp], Vh[:n_comp, :]

        def solver_covariance_eigh(X_centered, n_comp):
            # "Precompute the covariance matrix... run classical eigenvalue decomposition"
            n_s = X_centered.size(-2)
            cov = (X_centered.T @ X_centered) / (n_s - 1)
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            eigenvalues = torch.flip(eigenvalues, dims=(0,))
            eigenvectors = torch.flip(eigenvectors, dims=(1,))
            eigenvalues = torch.clamp(eigenvalues, min=0.0)
            S = torch.sqrt(eigenvalues[:n_comp] * (n_s - 1))
            Vh = eigenvectors[:, :n_comp].T
            U = (X_centered @ eigenvectors[:, :n_comp]) / S.clamp(min=1e-12)
            return U, S, Vh

        def solver_randomized(X_centered, n_comp):
            # "Run randomized SVD by the method of Halko et al."
            # n_oversamples: additional random vectors for conditioning (sklearn param)
            n_iter = _get_n_iter()
            q = min(n_comp + n_oversamples, min(X_centered.shape[0], X_centered.shape[1]))
            q = max(q, n_comp)
            # power_iteration_normalizer: PyTorch pca_lowrank uses QR internally
            U, S, V = torch.pca_lowrank(
                X_centered,
                q=q,
                center=False,
                niter=n_iter,
            )
            U, S, Vh = U[:, :n_comp], S[:n_comp], V[:, :n_comp].T
            return U, S, Vh

        def solver_arpack(X_centered, n_comp):
            # "Run SVD truncated to n_components calling ARPACK solver"
            # PyTorch has no ARPACK; approximate with pca_lowrank. tol not exposed in PyTorch.
            n_iter = _get_n_iter()
            n_iter = max(n_iter, 10)
            U, S, V = torch.pca_lowrank(
                X_centered,
                q=n_comp,
                center=False,
                niter=n_iter,
            )
            return U, S, V.T

        # Mapping the string from _init_module to the internal functions
        solver_mapping = {
            "full": solver_full,
            "covariance_eigh": solver_covariance_eigh,
            "randomized": solver_randomized,
            "arpack": solver_arpack
        }

        selected_fn = solver_mapping.get(actual_solver, solver_full)

        # Return a closure that only needs the data and component count
        return selected_fn

    def fit(
        self,
        data_or_X: Union[torch.Tensor, Any],
        **kwargs: Any,
    ) -> "PCA":
        """
        Fit the model with X.
        """
        # 1. Initialize metadata and resolve solvers/components
        X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
        self._init_module(X)

        # 2. Centering the data
        # "The input data is centered but not scaled for each feature"
        if self.copy:
            X = X.clone()

        self.mean_ = torch.mean(X, dim=-2)
        X_centered = X - self.mean_

        # 3. Handle MLE / Variance-based selection (if 'full' solver is used)
        # For MLE or float variance ratios, we initially compute the full SVD
        # to determine the optimal number of components.
        if self.n_components_ == 'mle' or isinstance(self.n_components_, float):
            # We need the full singular values to make the decision
            U, S, Vh = self.solver_fn(X_centered, min(self.n_samples_, self.n_features_in_))

            if self.n_components_ == 'mle':
                self.n_components_ = self._infer_dimension_mle(S, self.n_samples_)
            else:
                # Variance percentage selection: smallest k such that cumsum > n_components
                explained_var = (S ** 2) / (self.n_samples_ - 1)
                total_var = explained_var.sum().clamp(min=1e-12)
                ratio_cumsum = torch.cumsum(explained_var / total_var, dim=0)
                thresh = torch.tensor(
                    self.n_components_,
                    device=ratio_cumsum.device,
                    dtype=ratio_cumsum.dtype
                )
                self.n_components_ = (
                    torch.searchsorted(ratio_cumsum, thresh, side="right").item() + 1
                )

            # Slice the pre-computed SVD to the discovered n_components_
            self.components_ = Vh[:self.n_components_]
            self.singular_values_ = S[:self.n_components_]
        else:
            # Standard integer n_components
            U, S, Vh = self.solver_fn(X_centered, self.n_components_)
            self.components_ = Vh
            self.singular_values_ = S

        # 4. Calculate Variance Attributes
        # Variance = (Singular_Values^2) / (n_samples - 1)
        self.explained_variance_ = (self.singular_values_ ** 2) / (self.n_samples_ - 1)

        # Total variance of the centered data
        total_var = torch.var(X_centered, dim=-2, correction=1).sum()
        self.explained_variance_ratio_ = self.explained_variance_ / total_var.clamp(min=1e-12)

        # 5. Noise Variance (Probabilistic PCA model)
        # "Average of (min(n_features, n_samples) - n_components) smallest eigenvalues"
        min_dim = min(self.n_features_in_, self.n_samples_)
        if self.n_components_ < min_dim:
            remaining_var = total_var - self.explained_variance_.sum()
            self.noise_variance_ = (remaining_var / (min_dim - self.n_components_)).item()
        else:
            self.noise_variance_ = 0.0

        return self

    def _assess_dimension(
        self,
        spectrum: torch.Tensor,
        rank: int,
        n_samples: int,
    ) -> float:
        n_features = spectrum.shape[0]
        if not 1 <= rank < n_features:
            raise ValueError(
                "the tested rank should be in [1, n_features - 1]"
            )
        eps = 1e-15
        if spectrum[rank - 1].item() < eps:
            return float('-inf')
        pu = -rank * log(2.0)
        for i in range(1, rank + 1):
            pu += (
                lgamma((n_features - i + 1) / 2.0)
                - log(math.pi) * (n_features - i + 1) / 2.0
            )
        pl = -spectrum[:rank].log().sum().item() * n_samples / 2.0
        v = max(eps, spectrum[rank:].sum().item() / (n_features - rank))
        pv = -log(v) * n_samples * (n_features - rank) / 2.0
        m = n_features * rank - rank * (rank + 1.0) / 2.0
        pp = log(2.0 * math.pi) * (m + rank) / 2.0
        pa = 0.0
        spectrum_ = spectrum.clone()
        spectrum_[rank:n_features] = v
        for i in range(rank):
            for j in range(i + 1, spectrum.shape[0]):
                diff = (spectrum[i].item() - spectrum[j].item()) * (
                    1.0 / spectrum_[j].item() - 1.0 / spectrum_[i].item()
                )
                if diff > 0:
                    pa += log(diff) + log(n_samples)
        ll = pu + pl + pv + pp - pa / 2.0 - rank * log(n_samples) / 2.0
        return ll

    def _infer_dimension_mle(
        self,
        singular_values: torch.Tensor,
        n_samples: int,
    ) -> int:
        spectrum = (singular_values ** 2) / (n_samples - 1)
        n_features = spectrum.shape[0]
        ll = torch.full((n_features,), float('-inf'), device=spectrum.device, dtype=spectrum.dtype)
        ll[0] = float('-inf')
        for rank in range(1, n_features):
            ll[rank] = self._assess_dimension(spectrum, rank, n_samples)
        return int(torch.argmax(ll).item())

    def fit_transform(
        self,
        data_or_X: Union[torch.Tensor, Any],
        **kwargs: Any,
    ) -> torch.Tensor:
        self.fit(data_or_X, **kwargs)
        X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
        if self.copy and isinstance(data_or_X, torch.Tensor):
            X = X.clone()
        return self.transform(X)

    def transform(
        self,
        X: Union[torch.Tensor, Any],
    ) -> torch.Tensor:
        if self.components_ is None:
            raise RuntimeError("PCA instance is not fitted yet. Call 'fit' before 'transform'.")

        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)

        # 1. Center the data using the mean found during fit
        X_centered = X - self.mean_

        # 2. Project data onto the principal components
        # X shape: (n_samples, n_features)
        # components_ shape: (n_components, n_features)
        # Result shape: (n_samples, n_components)
        X_transformed = torch.matmul(X_centered, self.components_.T)

        # 3. Handle Whitening if enabled
        # "multiplied by the square root of n_samples and then divided by the singular values"
        if self.whiten:
            scaling = math.sqrt(self.n_samples_ - 1) / self.singular_values_.clamp(min=1e-12)
            X_transformed = X_transformed * scaling

        return X_transformed

    def inverse_transform(
        self,
        X: Union[torch.Tensor, Any],
    ) -> torch.Tensor:
        """
        Transform data back to its original space.
        In other_decomposition words, return an input X_original whose transform would be X.
        """
        if self.components_ is None:
            raise RuntimeError("PCA instance is not fitted yet.")

        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)

        # 1. If whitened, we must reverse the scaling first
        if self.whiten:
            scaling = self.singular_values_ / math.sqrt(self.n_samples_ - 1)
            X = X * scaling

        # 2. Project back to original feature space
        # X shape: (n_samples, n_components)
        # components_ shape: (n_components, n_features)
        X_original = torch.matmul(X, self.components_)

        # 3. Add the mean back
        X_original = X_original + self.mean_

        return X_original


class KernelPCA(MLTransform):
    def __init__(
        self,
        n_components: Optional[int] = None,
        kernel: Union[
            Literal["linear", "poly", "rbf", "sigmoid", "cosine", "precomputed"],
            Callable[..., torch.Tensor],
            nn.Module,
        ] = "linear",
        gamma: Optional[Union[int, float]] = None,
        degree: Union[int, float] = 3,
        coef0: float = 1,
        kernel_params: Optional[Dict[str, Any]] = None,
        alpha: float = 1.0,
        fit_inverse_transform: bool = False,
        eigen_solver: Union[
            Literal["auto", "dense", "arpack", "randomized"],
            Callable[..., Tuple[torch.Tensor, torch.Tensor]],
            nn.Module,
        ] = "auto",
        tol: float = 0.0,
        max_iter: Optional[int] = None,
        iterated_power: Union[int, Literal["auto"]] = "auto",
        remove_zero_eig: bool = False,
        random_state: Optional[Union[int, torch.Generator]] = None,
        copy_X: bool = True,
        n_jobs: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float,
        eigen_solver_kwargs: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.n_comp = n_components
        self.device = device
        self.dtype = dtype
        self.kernel_name = kernel if isinstance(kernel, str) else None
        kernel_params = kernel_params or {}
        self.kernel_params = {
            "gamma": gamma,
            "degree": degree,
            "coef0": coef0,
            **kernel_params,
        }
        if isinstance(kernel, str):
            if kernel == "precomputed":
                self.kernel = None
            elif kernel == "cosine":
                def _cosine_kernel(xi: torch.Tensor, xj: torch.Tensor) -> torch.Tensor:
                    xi_n = F.normalize(xi.to(torch.float32), p=2, dim=-1)
                    xj_n = F.normalize(xj.to(torch.float32), p=2, dim=-1)
                    return (xi_n @ xj_n.T).to(self.dtype)
                self.kernel = _cosine_kernel
            else:
                kcls = get_kernel_class(kernel)
                if kcls is None:
                    raise ValueError(f"Unknown kernel '{kernel}'. Available: linear, poly, rbf, sigmoid, etc.")
                kparams = self._get_kernel_init_params(kernel, gamma, degree, coef0, n_features=1)
                kparams.update(device=self.device, dtype=self.dtype)
                self.kernel = kcls(**kparams)
        elif callable(kernel):
            kp = {k: v for k, v in self.kernel_params.items() if v is not None}
            self.kernel = lambda xi, xj, _kp=kp: kernel(xi, xj, **_kp)
        elif isinstance(kernel, nn.Module):
            self.kernel = kernel
        else:
            self.kernel = None
        self.alpha = alpha
        self.fit_inverse_transform = fit_inverse_transform
        self._eigen_solver_kwargs = eigen_solver_kwargs or kwargs.get("eigen_solver_kwargs") or dict(kwargs)
        self.eigen_solver, self.eigen_solver_type = self._calc_eigen_solver(
            eigen_solver, self._eigen_solver_kwargs
        )
        self.tol = tol
        self.max_iter = max_iter
        self.iterated_power = iterated_power if isinstance(iterated_power, str) else max(0, int(iterated_power))
        self.remove_zero_eig = remove_zero_eig
        if random_state is not None and isinstance(random_state, int):
            torch.manual_seed(random_state)
            self.random_state = torch.Generator().manual_seed(random_state)
        else:
            self.random_state = random_state
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.eigenvalues_ = None
        self.eigenvectors_ = None
        self.dual_coef_ = None
        self.X_transformed_fit_ = None
        self.X_fit_ = None
        self.n_features_in_ = None
        self.gamma_ = gamma
        self._K_fit_col_mean_ = None
        self._K_fit_mean_ = None

    def _get_kernel_init_params(
        self,
        kernel: str,
        gamma: Optional[float],
        degree: Union[int, float],
        coef0: float,
        n_features: int = 1,
    ) -> Dict[str, Any]:
        """Map kernel name to init params for get_kernel_class instances."""
        g = gamma if gamma is not None else 1.0 / n_features
        if kernel == "linear":
            return {}
        if kernel == "poly":
            return {"degree": degree, "gamma": g, "bias": coef0}
        if kernel == "rbf":
            return {"gamma": g}
        if kernel == "sigmoid":
            return {"gamma": g, "bias": coef0}
        return {}

    def _init_module(self, X: torch.Tensor) -> None:
        """
        Initializes the KernelPCA parameters based on the data shape.
        """
        self.n_samples_ = X.shape[-2]
        self.n_features_in_ = X.shape[-1]

        # Handle gamma=None: "If gamma is None, then it is set to 1/n_features"
        if self.kernel_params.get("gamma") is None and self.kernel_name not in ("precomputed", "linear", "cosine"):
            self.gamma_ = 1.0 / self.n_features_in_
            if self.kernel is not None and hasattr(self.kernel, "gamma"):
                self.kernel.gamma = self.gamma_
        else:
            self.gamma_ = self.kernel_params.get("gamma")

        # Resolve eigen_solver: "if components < 10 and samples > 200, use arpack"
        if self.eigen_solver_type == "auto":
            if self.n_comp is not None and self.n_comp < 10 and self.n_samples_ > 200:
                self.actual_solver_ = "arpack"
            else:
                self.actual_solver_ = "dense"
        elif self.eigen_solver_type == "custom":
            self.actual_solver_ = "custom"
        else:
            self.actual_solver_ = self.eigen_solver_type

        # Determine effective n_components
        if self.n_comp is None:
            self.n_components_ = self.n_samples_
        else:
            self.n_components_ = min(self.n_comp, self.n_samples_)

        # Final validation for ARPACK
        if self.actual_solver_ == "arpack":
            if not (0 < self.n_components_ < self.n_samples_):
                raise ValueError("arpack requires 0 < n_components < n_samples")

    def _calc_eigen_solver(
        self,
        solver_type: Union[str, Callable, nn.Module],
        solver_kwargs: Dict[str, Any],
    ) -> Tuple[Dict[str, Callable], Union[str, Callable, nn.Module]]:
        """
        Internal mapping to eigen decomposition functions.
        Returns (solver_mapping, actual_solver_type).
        """
        is_custom = callable(solver_type) or isinstance(solver_type, nn.Module)
        if is_custom:
            custom_kw = solver_kwargs

            def solver_custom(K: torch.Tensor, n_comp: int):
                out = solver_type(K, n_comp, **custom_kw)
                if isinstance(out, (tuple, list)) and len(out) >= 2:
                    evals, evecs = out[0], out[1]
                else:
                    raise ValueError("Custom eigen_solver must return (eigenvalues, eigenvectors)")
                evals, evecs = evals[:n_comp], evecs[:, :n_comp]
                return evals, evecs

            return {"custom": solver_custom}, "custom"
        self.eigen_solver_type = solver_type

        def solver_dense(K: torch.Tensor, n_comp: int):
            eigenvalues, eigenvectors = torch.linalg.eigh(K)
            eigenvalues = torch.flip(eigenvalues, dims=(0,))
            eigenvectors = torch.flip(eigenvectors, dims=(1,))
            return eigenvalues[:n_comp], eigenvectors[:, :n_comp]

        def solver_randomized(K: torch.Tensor, n_comp: int):
            n_iter = self.iterated_power
            if n_iter == "auto":
                n_iter = 7 if n_comp < 0.1 * min(K.shape) else 4
            else:
                n_iter = int(n_iter)
            U, S, _ = torch.pca_lowrank(K, q=n_comp, center=False, niter=n_iter)
            return S, U

        def solver_arpack(K: torch.Tensor, n_comp: int):
            try:
                eigenvalues, eigenvectors = torch.lobpcg(K, k=n_comp, largest=True)
            except Exception:
                eigenvalues, eigenvectors = torch.linalg.eigh(K)
                eigenvalues = torch.flip(eigenvalues, dims=(0,))
                eigenvectors = torch.flip(eigenvectors, dims=(1,))
                eigenvalues, eigenvectors = eigenvalues[:n_comp], eigenvectors[:, :n_comp]
            return eigenvalues, eigenvectors

        solver_mapping = {
            "dense": solver_dense,
            "arpack": solver_arpack,
            "randomized": solver_randomized,
        }
        return solver_mapping, solver_type

    def fit(
        self,
        data_or_X: Union[torch.Tensor, Any],
        **kwargs: Any,
    ) -> "KernelPCA":
        """
        Fit the model from data in X.
        """
        X = torch.as_tensor(data_or_X, dtype=self.dtype, device=self.device)
        if self.kernel_name == "precomputed":
            if X.shape[0] != X.shape[1]:
                raise ValueError("Precomputed kernel must be a square matrix (n_samples, n_samples)")
            K = X
            self.n_samples_ = K.shape[0]
            self.n_features_in_ = K.shape[0]
            self.X_fit_ = None
            self.actual_solver_ = self.eigen_solver_type if self.eigen_solver_type != "auto" else "dense"
            if self.actual_solver_ == "auto":
                self.actual_solver_ = "dense"
            self.n_components_ = self.n_samples_ if self.n_comp is None else min(self.n_comp, self.n_samples_)
            self.gamma_ = None
        else:
            self._init_module(X)
            self.X_fit_ = X.clone() if self.copy_X else X
            K = self.kernel(self.X_fit_, self.X_fit_)

        n = self.n_samples_
        one_n = torch.ones((n, n), device=self.device, dtype=self.dtype) / n
        K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
        self._K_fit_col_mean_ = K.sum(dim=0) / n
        self._K_fit_mean_ = K.sum() / (n * n)

        solver = self.eigen_solver.get(self.actual_solver_)
        if solver is None:
            solver = self.eigen_solver.get("dense")
        all_eigvals, all_eigvecs = solver(K_centered, self.n_components_)
        all_eigvals = all_eigvals.to(self.device)
        all_eigvecs = all_eigvecs.to(self.device)

        if self.remove_zero_eig or self.n_comp is None:
            tol = max(self.tol, float(torch.finfo(all_eigvals.dtype).eps) * n)
            mask = all_eigvals > tol
            all_eigvals = all_eigvals[mask]
            all_eigvecs = all_eigvecs[:, mask]
            self.n_components_ = len(all_eigvals)

        self.eigenvalues_ = all_eigvals[: self.n_components_]
        self.eigenvectors_ = all_eigvecs[:, : self.n_components_]
        self.eigenvectors_ = self.eigenvectors_ / torch.sqrt(
            self.eigenvalues_.clamp(min=1e-12)
        )

        if self.fit_inverse_transform and self.kernel_name != "precomputed" and self.X_fit_ is not None:
            self.X_transformed_fit_ = torch.matmul(K_centered, self.eigenvectors_)
            self._fit_inverse_transform(self.X_transformed_fit_, self.X_fit_)

        return self

    def _fit_inverse_transform(self, X_transformed, X_original):
        """
        Learns the inverse transform (pre-image) using Ridge Regression.
        Maps X_transformed (n_samples, n_components) -> X_original (n_samples, n_features).
        """
        # (X_trans^T * X_trans + alpha * I) * dual_coef = X_trans^T * X_orig
        n_comp = X_transformed.shape[1]
        I = torch.eye(n_comp, device=self.device, dtype=self.dtype)

        lhs = X_transformed.T @ X_transformed + self.alpha * I
        rhs = X_transformed.T @ X_original

        # Solve for dual_coef_
        self.dual_coef_ = torch.linalg.solve(lhs, rhs)

    def transform(
        self,
        X: Union[torch.Tensor, Any],
    ) -> torch.Tensor:
        """
        Project new data X into the kernel principal components space.
        For precomputed kernel, X must be the Gram matrix between new and fit data.
        """
        if self.eigenvectors_ is None:
            raise RuntimeError("KernelPCA instance is not fitted yet.")

        if self.kernel_name == "precomputed":
            K_new = torch.as_tensor(X, device=self.device, dtype=self.dtype)
            if K_new.shape[1] != self.n_samples_:
                raise ValueError(
                    "Precomputed kernel for transform must have shape (n_samples_new, n_samples_fit)"
                )
        else:
            X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
            K_new = self.kernel(X, self.X_fit_)

        n_fit = self.n_samples_
        K_new_row_mean = K_new.sum(dim=1, keepdim=True) / n_fit
        K_new_centered = (
            K_new
            - K_new_row_mean
            - self._K_fit_col_mean_.unsqueeze(0)
            + self._K_fit_mean_
        )
        return torch.matmul(K_new_centered, self.eigenvectors_)

    def fit_transform(
        self,
        data_or_X: Union[torch.Tensor, Any],
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Fit the model from data in X and transform X.
        """
        self.fit(data_or_X, **kwargs)
        if self.kernel_name == "precomputed":
            return self.transform(data_or_X)
        return self.transform(data_or_X)

    def inverse_transform(
        self,
        X_transformed: Union[torch.Tensor, Any],
    ) -> torch.Tensor:
        """
        Transform data back to its original space using the learned
        pre-image (Ridge Regression).
        """
        if not self.fit_inverse_transform:
            raise RuntimeError(
                "inverse_transform is only available if fit_inverse_transform=True during fit."
            )
        if self.dual_coef_ is None:
            raise RuntimeError("KernelPCA instance is not fitted yet.")
        X_transformed = torch.as_tensor(X_transformed, device=self.device, dtype=self.dtype)
        return torch.matmul(X_transformed, self.dual_coef_)


class SparsePCA(MLTransform):
    def __init__(
        self,
        n_components: Optional[int] = None,
        alpha: float = 1.0,
        ridge_alpha: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-8,
        method: Union[
            Literal["lars", "cd"],
            Callable[..., torch.Tensor],
            nn.Module,
        ] = "lars",
        n_jobs: Optional[int] = None,
        U_init: Optional[Union[List, Tuple, torch.Tensor]] = None,
        V_init: Optional[Union[List, Tuple, torch.Tensor]] = None,
        verbose: Union[int, bool] = False,
        random_state: Optional[Union[torch.Generator, int]] = None,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float,
        method_solver_kwargs: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.n_components = n_components
        self.alpha = alpha
        self.ridge_alpha = ridge_alpha
        self.max_iter = max_iter
        self.tol = tol
        self.method = method
        self.n_jobs = n_jobs
        self.U_init = U_init
        self.V_init = V_init
        self.verbose = int(verbose) if isinstance(verbose, bool) else verbose
        self.device = device
        self.dtype = dtype
        self._method_solver_kwargs = method_solver_kwargs or kwargs.get("method_solver_kwargs") or {}
        if random_state is not None and isinstance(random_state, int):
            torch.manual_seed(random_state)
            self.random_state = torch.Generator().manual_seed(random_state)
        else:
            self.random_state = random_state
        self.components_ = None
        self.error_ = None
        self.n_components_ = None
        self.n_iter_ = None
        self.mean_ = None
        self.n_features_in_ = None

    def _soft_threshold(self, x: torch.Tensor, lam: float) -> torch.Tensor:
        """Soft thresholding for L1 penalty: sign(x) * max(|x| - lam, 0)."""
        return torch.sign(x) * torch.clamp(torch.abs(x) - lam, min=0.0)

    def _lasso_cd(
        self,
        X: torch.Tensor,
        V: torch.Tensor,
        alpha: float,
        max_iter_cd: int = 100,
    ) -> torch.Tensor:
        """
        Coordinate descent for Lasso: min 0.5||X - U V||^2 + alpha ||U||_1.
        X: (n_samples, n_features), V: (n_components, n_features).
        Returns U: (n_samples, n_components).
        """
        n_samples, _ = X.shape
        n_comp = V.shape[0]
        U = torch.zeros((n_samples, n_comp), device=self.device, dtype=self.dtype)
        V_norms_sq = (V * V).sum(dim=1).clamp(min=1e-12)
        for _ in range(max_iter_cd):
            U_old = U.clone()
            for j in range(n_comp):
                r = X - U @ V + U[:, j : j + 1] @ V[j : j + 1, :]
                rho = (r * V[j : j + 1, :]).sum(dim=1)
                U[:, j] = self._soft_threshold(rho / V_norms_sq[j], alpha / V_norms_sq[j])
            if (U - U_old).abs().max() < self.tol * 10:
                break
        return U

    def _lasso_ista(
        self,
        X: torch.Tensor,
        V: torch.Tensor,
        alpha: float,
        max_iter_ista: int = 100,
    ) -> torch.Tensor:
        """
        Iterative Soft-Thresholding Algorithm for Lasso.
        min ||X - U V||^2 + alpha ||U||_1.
        """
        n_samples, _ = X.shape
        n_comp = V.shape[0]
        step = 1.0 / (torch.linalg.norm(V, ord=2) ** 2 + 1e-8)
        U = torch.zeros((n_samples, n_comp), device=self.device, dtype=self.dtype)
        for _ in range(max_iter_ista):
            grad = (U @ V - X) @ V.T
            U = self._soft_threshold(U - step * grad, step * alpha)
        return U

    def _dict_learning(
        self,
        X: torch.Tensor,
        n_comp: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[float], int]:
        """
        Sparse dictionary learning: min 0.5||X - U V||^2 + alpha||U||_1.
        Returns (U, V, errors, n_iter).
        """
        n_samples, n_features = X.shape
        if self.U_init is not None and self.V_init is not None:
            U = torch.as_tensor(self.U_init, device=self.device, dtype=self.dtype)
            V = torch.as_tensor(self.V_init, device=self.device, dtype=self.dtype)
            if U.shape != (n_samples, n_comp) or V.shape != (n_comp, n_features):
                U = torch.randn(n_samples, n_comp, device=self.device, dtype=self.dtype) * 0.01
                V = torch.randn(n_comp, n_features, device=self.device, dtype=self.dtype) * 0.01
        else:
            if self.random_state is not None and isinstance(self.random_state, torch.Generator):
                U = torch.randn(n_samples, n_comp, device=self.device, dtype=self.dtype, generator=self.random_state) * 0.01
                V = torch.randn(n_comp, n_features, device=self.device, dtype=self.dtype, generator=self.random_state) * 0.01
            else:
                U = torch.randn(n_samples, n_comp, device=self.device, dtype=self.dtype) * 0.01
                V = torch.randn(n_comp, n_features, device=self.device, dtype=self.dtype) * 0.01
        errors = []
        is_custom = callable(self.method) or isinstance(self.method, nn.Module)
        for it in range(self.max_iter):
            if is_custom:
                U = torch.as_tensor(
                    self.method(X, V, self.alpha, **self._method_solver_kwargs),
                    device=self.device,
                    dtype=self.dtype,
                )
            elif self.method == "cd":
                U = self._lasso_cd(X, V, self.alpha)
            else:
                U = self._lasso_ista(X, V, self.alpha)
            UtU = U.T @ U + 1e-8 * torch.eye(n_comp, device=self.device, dtype=self.dtype)
            V = torch.linalg.solve(UtU, U.T @ X)
            norms = torch.linalg.norm(V, dim=1, keepdim=True).clamp(min=1e-12)
            V = V / norms
            err = (X - U @ V).pow(2).sum().item()
            errors.append(err)
            if it > 0 and abs(errors[-1] - errors[-2]) < self.tol:
                break
        return U, V, errors, it + 1

    def fit(
        self,
        data_or_X: Union[torch.Tensor, Any],
        **kwargs: Any,
    ) -> "SparsePCA":
        """
        Fit the model from data in X.
        """
        X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
        self.mean_ = X.mean(dim=0)
        X_centered = X - self.mean_
        n_samples, n_features = X_centered.shape
        n_comp = self.n_components if self.n_components is not None else n_features
        n_comp = min(n_comp, n_features, n_samples)
        U, V, errors, n_iter = self._dict_learning(X_centered, n_comp)
        self.components_ = V
        self.n_components_ = n_comp
        self.n_iter_ = n_iter
        self.error_ = torch.tensor(errors, device=self.device, dtype=self.dtype)
        self.n_features_in_ = n_features
        return self

    def transform(
        self,
        X: Union[torch.Tensor, Any],
    ) -> torch.Tensor:
        """
        Least Squares projection of the data onto the sparse components.
        Uses ridge regression for stability.
        """
        if self.components_ is None:
            raise RuntimeError("SparsePCA instance is not fitted yet.")
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        X_centered = X - self.mean_
        V = self.components_
        gram = V @ V.T + self.ridge_alpha * torch.eye(
            V.shape[0], device=self.device, dtype=self.dtype
        )
        U = torch.linalg.solve(gram, V @ X_centered.T).T
        return U

    def inverse_transform(
        self,
        X: Union[torch.Tensor, Any],
    ) -> torch.Tensor:
        """
        Transform data from the latent space to the original space.
        """
        if self.components_ is None:
            raise RuntimeError("SparsePCA instance is not fitted yet.")
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        return X @ self.components_ + self.mean_

    def fit_transform(
        self,
        data_or_X: Union[torch.Tensor, Any],
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Fit the model and transform X.
        """
        self.fit(data_or_X, **kwargs)
        return self.transform(data_or_X)


class MiniBatchSparsePCA(SparsePCA):
    def __init__(
        self,
        n_components: Optional[int] = None,
        alpha: float = 1.0,
        ridge_alpha: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-3,
        method: Union[
            Literal["lars", "cd"],
            Callable[..., torch.Tensor],
            nn.Module,
        ] = "lars",
        callback: Optional[Union[Callable[..., None], nn.Module]] = None,
        batch_size: int = 3,
        shuffle: bool = True,
        max_no_improvement: Optional[int] = 10,
        n_jobs: Optional[int] = None,
        U_init: Optional[Union[List, Tuple, torch.Tensor]] = None,
        V_init: Optional[Union[List, Tuple, torch.Tensor]] = None,
        verbose: Union[int, bool] = False,
        random_state: Optional[Union[torch.Generator, int]] = None,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float,
        method_solver_kwargs: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            n_components=n_components,
            alpha=alpha,
            ridge_alpha=ridge_alpha,
            max_iter=max_iter,
            tol=tol,
            method=method,
            n_jobs=n_jobs,
            U_init=U_init,
            V_init=V_init,
            verbose=verbose,
            random_state=random_state,
            device=device,
            dtype=dtype,
            method_solver_kwargs=method_solver_kwargs,
            *args,
            **kwargs,
        )
        self.callback = callback
        self.batch_size = max(1, batch_size)
        self.shuffle = shuffle
        self.max_no_improvement = max_no_improvement

    def _dict_learning_minibatch(
        self,
        X: torch.Tensor,
        n_comp: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[float], int]:
        """
        Mini-batch sparse dictionary learning.
        Iterates over batches of samples, updating dictionary per batch.
        """
        n_samples, n_features = X.shape
        if self.V_init is not None:
            V = torch.as_tensor(self.V_init, device=self.device, dtype=self.dtype)
            if V.shape != (n_comp, n_features):
                V = torch.randn(n_comp, n_features, device=self.device, dtype=self.dtype) * 0.01
        elif self.random_state is not None and isinstance(self.random_state, torch.Generator):
            V = torch.randn(n_comp, n_features, device=self.device, dtype=self.dtype, generator=self.random_state) * 0.01
        else:
            V = torch.randn(n_comp, n_features, device=self.device, dtype=self.dtype) * 0.01
        norms = torch.linalg.norm(V, dim=1, keepdim=True).clamp(min=1e-12)
        V = V / norms
        errors = []
        is_custom = callable(self.method) or isinstance(self.method, nn.Module)
        batch_size = min(self.batch_size, n_samples)
        n_batches = max(1, (n_samples + batch_size - 1) // batch_size)
        best_cost = float("inf")
        no_improvement_count = 0
        smoothed_cost = float("inf")
        total_iter = 0
        dict_change = float("inf")
        for epoch in range(self.max_iter):
            if self.shuffle:
                if self.random_state is not None and isinstance(self.random_state, torch.Generator):
                    order = torch.randperm(n_samples, device=self.device, generator=self.random_state)
                else:
                    order = torch.randperm(n_samples, device=self.device)
            else:
                order = torch.arange(n_samples, device=self.device)
            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, n_samples)
                batch_ind = order[start:end]
                X_batch = X[batch_ind]
                if is_custom:
                    U_batch = torch.as_tensor(
                        self.method(X_batch, V, self.alpha, **self._method_solver_kwargs),
                        device=self.device,
                        dtype=self.dtype,
                    )
                elif self.method == "cd":
                    U_batch = self._lasso_cd(X_batch, V, self.alpha)
                else:
                    U_batch = self._lasso_ista(X_batch, V, self.alpha)
                V_old = V.clone()
                UtU = U_batch.T @ U_batch + 1e-8 * torch.eye(n_comp, device=self.device, dtype=self.dtype)
                V = torch.linalg.solve(UtU, U_batch.T @ X_batch)
                norms = torch.linalg.norm(V, dim=1, keepdim=True).clamp(min=1e-12)
                V = V / norms
                err = (X_batch - U_batch @ V).pow(2).sum().item()
                errors.append(err)
                total_iter += 1
                smoothed_cost = 0.9 * smoothed_cost + 0.1 * err if total_iter > 1 else err
                if smoothed_cost < best_cost:
                    best_cost = smoothed_cost
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                if total_iter % 5 == 0 and self.callback is not None:
                    try:
                        self.callback(self, total_iter, err)
                    except TypeError:
                        self.callback(self)
                dict_change = (V - V_old).abs().max().item()
                if self.tol > 0 and dict_change < self.tol:
                    break
                if self.max_no_improvement is not None and no_improvement_count >= self.max_no_improvement:
                    break
            if self.tol > 0 and dict_change < self.tol:
                break
            if self.max_no_improvement is not None and no_improvement_count >= self.max_no_improvement:
                break
        return torch.empty(0), V, errors, total_iter

    def fit(
        self,
        data_or_X: Union[torch.Tensor, Any],
        **kwargs: Any,
    ) -> "MiniBatchSparsePCA":
        """
        Fit the model from data in X using mini-batches.
        """
        X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
        self.mean_ = X.mean(dim=0)
        X_centered = X - self.mean_
        n_samples, n_features = X_centered.shape
        n_comp = self.n_components if self.n_components is not None else n_features
        n_comp = min(n_comp, n_features, n_samples)
        _, V, errors, n_iter = self._dict_learning_minibatch(X_centered, n_comp)
        self.components_ = V
        self.n_components_ = n_comp
        self.n_iter_ = n_iter
        self.error_ = torch.tensor(errors, device=self.device, dtype=self.dtype)
        self.n_features_in_ = n_features
        return self


class IncrementalPCA(MLTransform):
    def __init__(
        self,
        n_components: Optional[int] = None,
        whiten: bool = False,
        copy: bool = True,
        batch_size: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float,
        method_solver_kwargs: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.n_components = n_components
        self.whiten = whiten
        self.copy = copy
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self._method_solver_kwargs = method_solver_kwargs or {}
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.var_ = None
        self.noise_variance_ = None
        self.n_components_ = None
        self.n_samples_seen_ = 0
        self.batch_size_ = None
        self.n_features_in_ = None

    def _incremental_mean_var(
        self,
        X: torch.Tensor,
        last_mean: torch.Tensor,
        last_var: torch.Tensor,
        last_count: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Welford's parallel algorithm for incremental mean and variance."""
        n_new = X.shape[0]
        n_total = last_count + n_new
        if n_total == 0:
            return last_mean, last_var, 0
        mean_new = X.mean(dim=0)
        var_new = X.var(dim=0, correction=0) if n_new > 1 else torch.zeros_like(mean_new)
        if last_count == 0:
            return mean_new, var_new.clamp(min=0), n_total
        mean_total = (last_count * last_mean + n_new * mean_new) / n_total
        var_total = (
            last_count * (last_var + last_mean.pow(2))
            + n_new * (var_new + mean_new.pow(2))
        ) / n_total - mean_total.pow(2)
        return mean_total, var_total.clamp(min=0), n_total

    def partial_fit(
        self,
        data_or_X: Union[torch.Tensor, Any],
        **kwargs: Any,
    ) -> "IncrementalPCA":
        """
        Incremental fit with X. All of X is processed as a single batch.
        """
        X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
        if self.copy and isinstance(data_or_X, torch.Tensor):
            X = X.clone()
        n_samples, n_features = X.shape
        first_pass = self.components_ is None
        if first_pass:
            self.n_samples_seen_ = 0
            self.mean_ = torch.zeros(n_features, device=self.device, dtype=self.dtype)
            self.var_ = torch.zeros(n_features, device=self.device, dtype=self.dtype)
        if self.n_components is None:
            if first_pass:
                self.n_components_ = min(n_samples, n_features)
            else:
                self.n_components_ = self.components_.shape[0]
        else:
            if self.n_components > n_features:
                raise ValueError(
                    f"n_components={self.n_components} must be <= n_features={n_features}"
                )
            if first_pass and self.n_components > n_samples:
                raise ValueError(
                    f"n_components={self.n_components} must be <= batch samples {n_samples} "
                    "for the first partial_fit call."
                )
            self.n_components_ = self.n_components
        if self.components_ is not None and self.components_.shape[0] != self.n_components_:
            raise ValueError(
                f"Number of components changed from {self.components_.shape[0]} to "
                f"{self.n_components_}. Set n_components to a fixed value."
            )
        col_mean, col_var, n_total = self._incremental_mean_var(
            X, self.mean_, self.var_, self.n_samples_seen_
        )
        n_total = int(n_total)
        if first_pass:
            X_centered = X - col_mean
        else:
            col_batch_mean = X.mean(dim=0)
            X_centered = X - col_batch_mean
            scale = math.sqrt((self.n_samples_seen_ / n_total) * n_samples)
            mean_correction = scale * (self.mean_ - col_batch_mean)
            upper = self.singular_values_.reshape(-1, 1) * self.components_
            X_centered = torch.cat([upper, X_centered, mean_correction.unsqueeze(0)], dim=0)
        U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
        S = S[: self.n_components_]
        Vh = Vh[: self.n_components_, :]
        total_var = (col_var * n_total).sum().clamp(min=1e-12)
        explained_var = S.pow(2) / (n_total - 1) if n_total > 1 else S.pow(2)
        explained_var_ratio = S.pow(2) / total_var
        self.n_samples_seen_ = n_total
        self.components_ = Vh
        self.singular_values_ = S
        self.mean_ = col_mean
        self.var_ = col_var
        self.explained_variance_ = explained_var
        self.explained_variance_ratio_ = explained_var_ratio
        self.n_features_in_ = n_features
        min_dim = min(n_features, n_total)
        if self.n_components_ < min_dim:
            explained_sum_sq = S.pow(2).sum()
            n_discarded = min_dim - self.n_components_
            discarded_var = (total_var - explained_sum_sq).clamp(min=0)
            self.noise_variance_ = (discarded_var / (n_discarded * max(1, n_total - 1))).item()
        else:
            self.noise_variance_ = 0.0
        return self

    def fit(
        self,
        data_or_X: Union[torch.Tensor, Any],
        **kwargs: Any,
    ) -> "IncrementalPCA":
        """
        Fit the model with X, using minibatches of size batch_size.
        """
        X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
        n_samples, n_features = X.shape
        self.batch_size_ = self.batch_size if self.batch_size is not None else 5 * n_features
        self.batch_size_ = max(
            self.n_components or 0,
            min(self.batch_size_, n_samples),
        )
        self.components_ = None
        self.n_samples_seen_ = 0
        self.mean_ = None
        self.var_ = None
        for start in range(0, n_samples, self.batch_size_):
            end = min(start + self.batch_size_, n_samples)
            self.partial_fit(X[start:end], **kwargs)
        return self

    def transform(
        self,
        X: Union[torch.Tensor, Any],
    ) -> torch.Tensor:
        """
        Apply dimensionality reduction to X.
        X is projected on the first principal components previously extracted.
        """
        if self.components_ is None:
            raise RuntimeError("IncrementalPCA instance is not fitted yet.")
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        X_centered = X - self.mean_
        out = X_centered @ self.components_.T
        if self.whiten:
            denom = max(1, self.n_samples_seen_ - 1)
            scale = math.sqrt(denom) / self.singular_values_.clamp(min=1e-12)
            out = out * scale
        return out

    def inverse_transform(
        self,
        X: Union[torch.Tensor, Any],
    ) -> torch.Tensor:
        """
        Transform data back to its original space.
        """
        if self.components_ is None:
            raise RuntimeError("IncrementalPCA instance is not fitted yet.")
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if self.whiten:
            denom = max(1, self.n_samples_seen_ - 1)
            scale = self.singular_values_ / math.sqrt(denom)
            X = X * scale
        return X @ self.components_ + self.mean_

    def fit_transform(
        self,
        data_or_X: Union[torch.Tensor, Any],
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Fit the model with X and apply dimensionality reduction on X.
        """
        self.fit(data_or_X, **kwargs)
        return self.transform(data_or_X)



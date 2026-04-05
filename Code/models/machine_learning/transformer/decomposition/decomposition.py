import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import itertools
from typing import Optional, Callable, Union, Any, List, Tuple, Dict, Literal
from .....models.utils import MLTransform
from .....models.machine_learning.regression.knn.knn import BallTree, KDTree
from .....models.machine_learning.regression.linear_model.linear_models import (
    Lasso,
    OrthogonalMatchingPursuit,
)
from .....models.machine_learning.regression.linear_model.lars.lars import Lars, LassoLars
from torch.func import vmap
import joblib

__all__ = [
    "FastICA",
    "NMF",
    "MiniBatchNMF",
    "TruncatedSVD",
    "FactorAnalysis",
    "LatentDirichletAllocation",
    "DictionaryLearning",
    "MiniBatchDictionaryLearning",
    "SparseCoder",
    "NeighborhoodComponentsAnalysis",
]


def _gen_batches(n: int, batch_size: int):
    """Generate batch slice indices. Compatible with sklearn's gen_batches."""
    start = 0
    while start < n:
        end = min(start + batch_size, n)
        yield slice(start, end)
        start = end


def _logcosh_g(x: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """G function for neg-entropy: logcosh with derivative."""
    x = x * alpha
    gx = torch.tanh(x)
    g_x = (alpha * (1 - gx.pow(2))).mean(dim=-1)
    return gx, g_x


def _exp_g(x: torch.Tensor, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """G function for neg-entropy: exp with derivative."""
    exp = torch.exp(-(x.pow(2)) / 2)
    gx = x * exp
    g_x = ((1 - x.pow(2)) * exp).mean(dim=-1)
    return gx, g_x


def _cube_g(x: torch.Tensor, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """G function for neg-entropy: cube with derivative."""
    return x.pow(3), (3 * x.pow(2)).mean(dim=-1)


class FastICA(MLTransform):
    def __init__(
            self,
            n_components: Optional[int] = None,
            algorithm: Union[Literal["parallel", "deflation"], Callable, nn.Module] = "parallel",
            whiten: Union[bool, Literal["arbitrary-variance", "unit-variance"]] = "unit-variance",
            fun: Union[str, Literal["logcosh", "exp", "cube"], Callable, nn.Module] = "logcosh",
            func_args: Optional[dict] = None,
            max_iter: int = 200,
            tol: float = 1e-4,
            w_init: Optional[Union[List, Tuple, torch.Tensor]] = None,
            whiten_solver: Union[Literal["eigh", "svd"], Callable, nn.Module] = "svd",
            random_state: Optional[Union[int, torch.Generator]] = None,
            device: Union[str, torch.device] = "cpu",
            dtype: torch.dtype = torch.float,
            whiten_solver_kwargs: Optional[Dict[str, Any]] = None,
            algorithm_kwargs: Optional[Dict[str, Any]] = None,
            *args: Any,
            **kwargs: Any,
    ) -> None:
        super().__init__()
        self.n_components = n_components
        self.algorithm = algorithm if algorithm is not None else "parallel"
        self.whiten = whiten
        self.fun = fun
        self.func_args = func_args or {}
        self.max_iter = max_iter
        self.tol = tol
        self.w_init = w_init
        self.whiten_solver = whiten_solver
        self.random_state = random_state
        self.device = device
        self.dtype = dtype
        self._whiten_solver_kwargs = whiten_solver_kwargs or {}
        self._algorithm_kwargs = algorithm_kwargs or {}
        self.components_ = None
        self.mixing_ = None
        self.mean_ = None
        self.n_features_in_ = None
        self.n_iter_ = 0
        self.whitening_ = None
        if fun == "logcosh":
            fa = dict(self.func_args)
            if "alpha" not in fa:
                fa["alpha"] = 1.0
            fun_callable = lambda x, **kw: _logcosh_g(x, alpha=kw.get("alpha", 1.0))
            self.func = self._resolve_funcs(fun_callable, **fa)
        elif fun == "exp":
            self.func = self._resolve_funcs(_exp_g, **self.func_args)
        elif fun == "cube":
            self.func = self._resolve_funcs(_cube_g, **self.func_args)
        else:
            self.func = self._resolve_funcs(fun, **self.func_args)

    def _gs_decorrelation(
            self,
            w: torch.Tensor,
            W: torch.Tensor,
            j: int,
    ) -> None:
        """Orthonormalize w wrt the first j rows of W."""
        if j > 0:
            w.sub_(w @ W[:j].T @ W[:j])

    def _sym_decorrelation(self, W: torch.Tensor) -> torch.Tensor:
        """Symmetric decorrelation: W <- (W * W.T)^{-1/2} * W."""
        s, u = torch.linalg.eigh(W @ W.T)
        s = s.clamp(min=torch.finfo(W.dtype).tiny)
        return (u * (1.0 / s.sqrt()).unsqueeze(0)) @ u.T @ W

    def _run_g(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Call self.func and return (gx, g_x)."""
        out = self.func(x)
        if isinstance(out, (list, tuple)) and len(out) == 2:
            return out[0], out[1]
        raise ValueError("fun must return (value, derivative) tuple")

    def _ica_def(
            self,
            X: torch.Tensor,
            tol: float,
            max_iter: int,
            w_init: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """Deflationary FastICA."""
        n_components = w_init.shape[0]
        W = torch.zeros(n_components, n_components, device=self.device, dtype=self.dtype)
        n_iters = []
        for j in range(n_components):
            w = w_init[j].clone()
            w = w / w.norm()
            for i in range(max_iter):
                wtx = w @ X
                gwtx, g_wtx = self._run_g(wtx)
                w1 = (X * gwtx.unsqueeze(0)).mean(dim=1) - g_wtx.mean() * w
                self._gs_decorrelation(w1, W, j)
                w1 = w1 / w1.norm()
                lim = (w1 * w).sum().abs().abs() - 1
                if lim.abs() < tol:
                    break
                w = w1
            n_iters.append(i + 1)
            W[j] = w
        return W, max(n_iters)

    def _ica_par(
            self,
            X: torch.Tensor,
            tol: float,
            max_iter: int,
            w_init: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """Parallel FastICA."""
        W = self._sym_decorrelation(w_init.clone())
        p_ = float(X.shape[1])
        for ii in range(max_iter):
            gwtx, g_wtx = self._run_g(W @ X)
            W1 = self._sym_decorrelation(
                (gwtx @ X.T) / p_ - g_wtx.unsqueeze(1) * W
            )
            lim = (W1 * W).sum(dim=1).abs().abs() - 1
            lim = lim.abs().max().item()
            W = W1
            if lim < tol:
                return W, ii + 1
        return W, max_iter

    def fit(
            self,
            data_or_X: Union[torch.Tensor, Any],
            **kwargs: Any,
    ) -> "FastICA":
        """Fit the model to X."""
        X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        n_samples, n_features = X.shape
        if self.fun == "logcosh":
            alpha = self.func_args.get("alpha", 1.0)
            if not 1 <= alpha <= 2:
                raise ValueError("alpha must be in [1, 2] for logcosh")
        do_whiten = self.whiten in ("unit-variance", "arbitrary-variance") or self.whiten is True
        if not do_whiten:
            n_components = n_features  # whiten=False: use all features
        else:
            n_components = self.n_components
            if n_components is None:
                n_components = min(n_samples, n_features)
            n_components = min(n_components, n_samples, n_features)
        self.n_features_in_ = n_features

        if do_whiten:
            X_mean = X.mean(dim=0)
            X_centered = X - X_mean
            if self.whiten_solver == "eigh" or (
                    isinstance(self.whiten_solver, str)
                    and self.whiten_solver.lower() == "eigh"
            ):
                cov = X_centered.T @ X_centered / max(1, n_samples - 1)
                d, u = torch.linalg.eigh(cov)
                eps = torch.finfo(X.dtype).eps * 10
                d = d.clamp(min=eps)
                idx = d.argsort(descending=True)
                d, u = d[idx], u[:, idx]
                K = (u / d.sqrt()).T[:n_components]
            else:
                U, S, _ = torch.linalg.svd(X_centered.T, full_matrices=False)
                K = (U / S.unsqueeze(0)).T[:n_components]
            X1 = K @ X_centered.T
            X1 = X1 * math.sqrt(n_samples)
        else:
            X_mean = None
            X1 = X.T.clone().to(device=self.device, dtype=self.dtype)

        if self.w_init is not None:
            w_init = torch.as_tensor(
                self.w_init, device=self.device, dtype=self.dtype
            )
            if w_init.shape != (n_components, n_components):
                raise ValueError(
                    f"w_init shape {w_init.shape} != ({n_components}, {n_components})"
                )
        else:
            g = self.random_state
            if isinstance(g, int):
                g = torch.Generator(device=self.device).manual_seed(g)
            w_init = torch.randn(
                n_components, n_components,
                device=self.device, dtype=self.dtype, generator=g
            )

        if self.algorithm == "deflation" or (
                isinstance(self.algorithm, str) and self.algorithm.lower() == "deflation"
        ):
            W, n_iter = self._ica_def(X1, self.tol, self.max_iter, w_init)
        else:
            W, n_iter = self._ica_par(X1, self.tol, self.max_iter, w_init)

        self.n_iter_ = n_iter

        if do_whiten:
            if self.whiten == "unit-variance":
                S = W @ K @ X_centered.T
                S_std = S.std(dim=1, keepdim=True).clamp(min=1e-12)
                S = S / S_std
                W = W / S_std.T
            self.components_ = W @ K
            self.mean_ = X_mean
            self.whitening_ = K
        else:
            self.components_ = W
            self.mean_ = None
            self.whitening_ = None

        self.mixing_ = torch.linalg.pinv(self.components_)
        return self

    def transform(self, X: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Recover the sources from X."""
        if self.components_ is None:
            raise RuntimeError("FastICA instance is not fitted yet.")
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if self.mean_ is not None:
            X = X - self.mean_
        return X @ self.components_.T

    def inverse_transform(self, X: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Transform sources back to mixed data."""
        if self.mixing_ is None:
            raise RuntimeError("FastICA instance is not fitted yet.")
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        out = X @ self.mixing_.T
        if self.mean_ is not None:
            out = out + self.mean_
        return out

    def fit_transform(
            self,
            data_or_X: Union[torch.Tensor, Any],
            **kwargs: Any,
    ) -> torch.Tensor:
        """Fit the model and recover the sources from X."""
        self.fit(data_or_X, **kwargs)
        return self.transform(data_or_X)


class NMF(MLTransform):
    def __init__(self,
                 n_components: Union[Literal["auto"], int, None] = "auto",
                 init: Union[
                     Literal["random", "nndsvd", "nndsvda", "nndsvdar", "custom"], Callable, nn.Module, None] = None,
                 solver: Union[Literal["cd", "mu"], Callable, nn.Module] = "cd",
                 beta_loss: Union[float, Literal[
                     "frobenius", "kullback-leibler", "itakura-saito"], Callable, nn.Module] = "frobenius",
                 tol: float = 1e-4,
                 max_iter: int = 200,
                 random_state: Optional[Union[int, torch.Generator]] = None,
                 alpha_W: float = 0.0,
                 alpha_H: Union[float, Literal["same"]] = "same",
                 l1_ratio: float = 0.0,
                 verbose: int = 0,
                 shuffle: bool = False,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args: Any,
                 **kwargs: Any,
                 ) -> None:
        super().__init__()
        self.n_components = n_components
        self.init = init
        self.solver = solver if isinstance(solver, str) else solver
        self.beta_loss = beta_loss
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha_W = alpha_W
        self.alpha_H = alpha_W if alpha_H == "same" else alpha_H
        self.l1_ratio = l1_ratio
        self.verbose = verbose
        self.shuffle = shuffle
        self.device = device
        self.dtype = dtype

        self.components_ = None
        self.n_components_ = None
        self.reconstruction_err_ = None
        self.n_iter_ = 0
        self.n_features_in_ = None
        self.W = None
        self.H = None
        self.kwargs = dict(kwargs)
        self.kwargs["_args"] = args
        self._n_samples = None
        self._beta_val = None

    def _init_module(self, X: torch.Tensor):
        n_samples = X.shape[-2]
        n_features = X.shape[-1]
        self.n_features_in_ = n_features
        self._n_samples = n_samples
        self._alpha_W = float(self.alpha_W)
        self._alpha_H = float(self.alpha_W if self.alpha_H == "same" else self.alpha_H)
        if isinstance(self.n_components, str) and self.n_components == "auto":
            k = n_features
        elif self.n_components is None:
            k = n_features
        else:
            k = self.n_components
        init_actual = self.init
        if init_actual is None:
            init_actual = "nndsvda" if k <= min(n_samples, n_features) else "random"
        is_init_str = isinstance(init_actual, str)
        if is_init_str and init_actual not in ("random", "nndsvd", "nndsvda", "nndsvdar", "custom") and k > min(
                n_samples, n_features):
            raise ValueError(
                f"init='{init_actual}' can only be used when n_components <= min(n_samples, n_features)"
            )
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        if isinstance(init_actual, type) and issubclass(init_actual, nn.Module):
            init_kw = self.kwargs.get("init_kwargs", {})
            init_mod = init_actual(device=self.device, dtype=self.dtype, **init_kw)
            out = init_mod(X, k, device=self.device, dtype=self.dtype, random_state=self.random_state)
            self.W, self.H = (out[0], out[1]) if isinstance(out, (tuple, list)) and len(out) >= 2 else (out, out)
        elif isinstance(init_actual, nn.Module):
            init_kw = self.kwargs.get("init_kwargs", {})
            out = init_actual(X, k, device=self.device, dtype=self.dtype, random_state=self.random_state, **init_kw)
            self.W, self.H = (out[0], out[1]) if isinstance(out, (tuple, list)) and len(out) >= 2 else (out, out)
        elif callable(init_actual):
            init_kw = self.kwargs.get("init_kwargs", {})
            out = init_actual(X, k, device=self.device, dtype=self.dtype, random_state=self.random_state, **init_kw)
            self.W, self.H = (out[0], out[1]) if isinstance(out, (tuple, list)) and len(out) >= 2 else (out, out)
        elif init_actual == "random":
            self._calc_rand_init(X, k)
        elif init_actual == "nndsvd":
            self._calc_nndsvd(X, k)
        elif init_actual == "nndsvda":
            self._calc_nndsvda(X, k)
        elif init_actual == "nndsvdar":
            self._calc_nndsvdar(X, k)
        else:
            self.W, self.H = self.kwargs.get("W", None), self.kwargs.get("H", None)
            scale = torch.sqrt(X.mean() / k)
            gen = torch.Generator().manual_seed(
                self.random_state) if self.random_state is not None else torch.Generator()
            if self.W is None:
                self.W = torch.randn((n_samples, k), generator=gen, device=self.device, dtype=self.dtype) * scale
            if self.H is None:
                self.H = torch.randn((k, n_features), generator=gen, device=self.device, dtype=self.dtype) * scale
        self.W = nn.Parameter(self.W)
        self.H = nn.Parameter(self.H)
        beta_loss_map = {
            "frobenius": 2,
            "kullback-leibler": 1,
            "itakura-saito": 0
        }
        if isinstance(self.beta_loss, str):
            self._beta_val = beta_loss_map.get(self.beta_loss, 2)
            beta_loss = lambda x: self._calc_beta_loss(x, self._beta_val)
        elif isinstance(self.beta_loss, nn.Module):
            self._beta_val = None
            _bl_mod = self.beta_loss
            beta_loss = lambda x: _bl_mod(x, self.W, self.H)
        elif callable(self.beta_loss):
            self._beta_val = None
            beta_loss = lambda x: self.beta_loss(x, self.W, self.H, **self.kwargs.get("beta_loss_kwargs", {}))
        else:
            self._beta_val = float(self.beta_loss) if isinstance(self.beta_loss, (int, float)) else 2
            beta_loss = lambda x: self._calc_beta_loss(x, self._beta_val)
        self.beta_loss = beta_loss
        return self

    def _calc_loss(self, X: torch.Tensor):
        term = self._alpha_W * self.l1_ratio * self.n_features_in_ * torch.norm(self.W.view(-1), p=1)
        term += self._alpha_H * self.l1_ratio * self._n_samples * torch.norm(self.H.view(-1), p=1)
        term += 0.5 * self._alpha_W * (1 - self.l1_ratio) * self.n_features_in_ * torch.linalg.norm(self.W, ord="fro")
        term += 0.5 * self._alpha_H * (1 - self.l1_ratio) * self._n_samples * torch.linalg.norm(self.H, ord="fro")
        return term + self.beta_loss(X)

    def _calc_nndsvd(self, X: torch.Tensor, k: int):
        m = X.size(-2)
        n = X.size(-1)

        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        U = U[:, :k]
        S = S[:k]
        V = Vh[:k, :].t()

        W = torch.zeros((m, k), device=self.device, dtype=self.dtype)
        H = torch.zeros((k, n), device=self.device, dtype=self.dtype)

        W[:, 0] = torch.sqrt(S[0]) * torch.abs(U[:, 0])
        H[0, :] = torch.sqrt(S[0]) * torch.abs(V[:, 0])

        for j in range(1, k):
            u = U[:, j]
            v = V[:, j]

            u_p = torch.clamp(u, min=0.0)
            u_n = torch.clamp(u, max=0.0).abs()
            v_p = torch.clamp(v, min=0.0)
            v_n = torch.clamp(v, max=0.0).abs()

            n_p = u_p.norm() * v_p.norm()
            n_n = u_n.norm() * v_n.norm()

            eps = 1e-10
            if n_p > n_n:
                u_nrm = u_p.norm().clamp(min=eps)
                v_nrm = v_p.norm().clamp(min=eps)
                lbd = torch.sqrt(S[j] * n_p)
                W[:, j] = lbd * (u_p / u_nrm)
                H[j, :] = lbd * (v_p / v_nrm)
            else:
                u_nrm = u_n.norm().clamp(min=eps)
                v_nrm = v_n.norm().clamp(min=eps)
                lbd = torch.sqrt(S[j] * n_n)
                W[:, j] = lbd * (u_n / u_nrm)
                H[j, :] = lbd * (v_n / v_nrm)

        eps = 1e-6
        W[W < eps] = 0
        H[H < eps] = 0
        self.W, self.H = W, H

    def _calc_nndsvda(self, X: torch.Tensor, k: int):
        m = X.size(-2)
        n = X.size(-1)

        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        U = U[:, :k]
        S = S[:k]
        V = Vh[:k, :].t()

        W = torch.zeros((m, k), device=self.device, dtype=self.dtype)
        H = torch.zeros((k, n), device=self.device, dtype=self.dtype)

        W[:, 0] = torch.sqrt(S[0]) * torch.abs(U[:, 0])
        H[0, :] = torch.sqrt(S[0]) * torch.abs(V[:, 0])

        for j in range(1, k):
            u = U[:, j]
            v = V[:, j]

            u_p = torch.clamp(u, min=0.0)
            v_p = torch.clamp(v, min=0.0)
            u_n = torch.clamp(u, max=0.0).neg()
            v_n = torch.clamp(v, max=0.0).neg()

            norm_p = u_p.norm() * v_p.norm()
            norm_n = u_n.norm() * v_n.norm()

            if norm_p > norm_n:
                W[:, j] = torch.sqrt(S[j] * norm_p) * (u_p / u_p.norm())
                H[j, :] = torch.sqrt(S[j] * norm_p) * (v_p / v_p.norm())
            else:
                W[:, j] = torch.sqrt(S[j] * norm_n) * (u_n / u_n.norm())
                H[j, :] = torch.sqrt(S[j] * norm_n) * (v_n / v_n.norm())

        avg = X.mean()
        W[W == 0] = avg
        H[H == 0] = avg
        self.W, self.H = W, H

    def _calc_nndsvdar(self, X: torch.Tensor, k: int):
        m = X.size(-2)
        n = X.size(-1)

        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        U = U[:, :k]
        S = S[:k]
        V = Vh[:k, :].t()

        W = torch.zeros((m, k), device=self.device, dtype=self.dtype)
        H = torch.zeros((k, n), device=self.device, dtype=self.dtype)

        W[:, 0] = torch.sqrt(S[0]) * torch.abs(U[:, 0])
        H[0, :] = torch.sqrt(S[0]) * torch.abs(V[:, 0])

        for j in range(1, k):
            u = U[:, j]
            v = V[:, j]

            u_p = torch.clamp(u, min=0.0)
            v_p = torch.clamp(v, min=0.0)
            u_n = torch.clamp(u, max=0.0).neg()
            v_n = torch.clamp(v, max=0.0).neg()

            norm_p = u_p.norm() * v_p.norm()
            norm_n = u_n.norm() * v_n.norm()

            if norm_p > norm_n:
                W[:, j] = torch.sqrt(S[j] * norm_p) * (u_p / u_p.norm())
                H[j, :] = torch.sqrt(S[j] * norm_p) * (v_p / v_p.norm())
            else:
                W[:, j] = torch.sqrt(S[j] * norm_n) * (u_n / u_n.norm())
                H[j, :] = torch.sqrt(S[j] * norm_n) * (v_n / v_n.norm())
        scale = X.mean() / 100.0
        gen = torch.Generator().manual_seed(self.random_state) if self.random_state is not None else torch.Generator()

        W_rand = torch.rand((*W.shape,), generator=gen, device=self.device, dtype=self.dtype) * scale
        H_rand = torch.rand((*H.shape,), generator=gen, device=self.device, dtype=self.dtype) * scale

        W = torch.where(W == 0, W_rand, W)
        H = torch.where(H == 0, H_rand, H)
        self.W, self.H = W, H

    def _calc_rand_init(self, X: torch.Tensor, k: int):
        m = X.size(-2)
        n = X.size(-1)

        gen = torch.Generator().manual_seed(self.random_state) if self.random_state is not None else torch.Generator()
        scale = torch.sqrt(X.mean() / k)

        W = torch.randn((m, k), generator=gen, device=self.device, dtype=self.dtype).abs() * scale
        H = torch.randn((k, n), generator=gen, device=self.device, dtype=self.dtype).abs() * scale
        self.W, self.H = W, H

    def _calc_beta_loss(
            self,
            X: torch.Tensor,
            beta: Union[int, float] = 2,
            eps: float = 1e-9,
            W: Optional[torch.Tensor] = None,
            H: Optional[torch.Tensor] = None,
    ):
        """Compute beta-divergence between X and W@H. If W,H omitted, use self.W, self.H."""
        W_use = W if W is not None else (self.W.data if isinstance(self.W, nn.Parameter) else self.W)
        H_use = H if H is not None else (self.H.data if isinstance(self.H, nn.Parameter) else self.H)
        X_hat = W_use @ H_use
        X_hat = torch.clamp(X_hat, min=eps)
        X = torch.clamp(X, min=eps)

        if isinstance(beta, float):
            beta = int(beta)

        if beta == 0:
            res = (X / X_hat) - torch.log(X / X_hat) - 1
            return torch.sum(res)
        elif beta == 1:
            res = (X * torch.log(X / X_hat)) - X + X_hat
            return torch.sum(res)
        else:
            return 0.5 * torch.linalg.norm((X - X_hat), ord="fro")

    def _update_cd(self, X: torch.Tensor) -> float:
        """One iteration of coordinate descent (Frobenius only). Returns reconstruction error."""
        W, H = self.W.data, self.H.data
        n_samples, n_features = X.shape
        k = W.shape[1]
        l1_W = self._alpha_W * self.l1_ratio * self.n_features_in_
        l2_W = self._alpha_W * (1 - self.l1_ratio) * self.n_features_in_
        l1_H = self._alpha_H * self.l1_ratio * self._n_samples
        l2_H = self._alpha_H * (1 - self.l1_ratio) * self._n_samples

        HHt = H @ H.T
        XHt = X @ H.T
        for j in range(k):
            if l2_W > 0:
                HHt_j = HHt[j, j] + l2_W
            else:
                HHt_j = HHt[j, j].clamp(min=1e-12)
            numer = XHt[:, j] - l1_W
            W[:, j] = (numer / HHt_j).clamp(min=0)

        WWt = W.T @ W
        WtX = W.T @ X
        for j in range(k):
            if l2_H > 0:
                WWt_j = WWt[j, j] + l2_H
            else:
                WWt_j = WWt[j, j].clamp(min=1e-12)
            numer = WtX[j, :] - l1_H
            H[j, :] = (numer / WWt_j).clamp(min=0)

        rec_err = 0.5 * (X - W @ H).pow(2).sum().item()
        return rec_err

    def _update_mu(self, X: torch.Tensor) -> float:
        """One iteration of multiplicative update. Supports beta_loss. Returns reconstruction error."""
        W, H = self.W.data, self.H.data
        eps = 1e-9
        beta = self._beta_val if self._beta_val is not None else 2

        if beta == 2:
            W_denom = W @ H @ H.T + self._alpha_W * (1 - self.l1_ratio) * self.n_features_in_
            W_denom = W_denom.clamp(min=eps)
            W_numer = X @ H.T - self._alpha_W * self.l1_ratio * self.n_features_in_
            W_numer = W_numer.clamp(min=0)
            W.mul_(W_numer / W_denom)

            H_denom = W.T @ W @ H + self._alpha_H * (1 - self.l1_ratio) * self._n_samples
            H_denom = H_denom.clamp(min=eps)
            H_numer = W.T @ X - self._alpha_H * self.l1_ratio * self._n_samples
            H_numer = H_numer.clamp(min=0)
            H.mul_(H_numer / H_denom)
        else:
            X_hat = (W @ H).clamp(min=eps)
            X_clamp = X.clamp(min=eps)
            if beta == 1:
                W_numer = (X / X_hat) @ H.T
                W_denom = H.sum(dim=1).unsqueeze(0).expand(W.shape[0], -1)
                W.mul_(W_numer / W_denom.clamp(min=eps))
                X_hat = (W @ H).clamp(min=eps)
                H_numer = W.T @ (X / X_hat)
                H_denom = W.sum(dim=0).unsqueeze(1).expand(-1, H.shape[1])
                H.mul_(H_numer / H_denom.clamp(min=eps))
            elif beta == 0:
                W_numer = (X / X_hat.pow(2)) @ H.T
                W_denom = (1.0 / X_hat) @ H.T
                W.mul_(W_numer / W_denom.clamp(min=eps))
                X_hat = (W @ H).clamp(min=eps)
                H_numer = W.T @ (X / X_hat.pow(2))
                H_denom = W.T @ (1.0 / X_hat)
                H.mul_(H_numer / H_denom.clamp(min=eps))
            else:
                W_numer = (X * X_hat.pow(beta - 2)) @ H.T
                W_denom = X_hat.pow(beta - 1) @ H.T
                W.mul_(W_numer / W_denom.clamp(min=eps))
                X_hat = (W @ H).clamp(min=eps)
                H_numer = W.T @ (X * X_hat.pow(beta - 2))
                H_denom = W.T @ X_hat.pow(beta - 1)
                H.mul_(H_numer / H_denom.clamp(min=eps))

        return self._calc_beta_loss(X, beta).item()

    def _run_solver(self, X: torch.Tensor) -> float:
        """Run one solver iteration. Dispatches to cd, mu, or custom Callable/nn.Module."""
        solver = self.solver
        if isinstance(solver, str):
            if solver == "cd":
                return self._update_cd(X)
            return self._update_mu(X)
        if isinstance(solver, type) and issubclass(solver, nn.Module):
            if not hasattr(self, "_solver_module"):
                solver_kw = self.kwargs.get("solver_kwargs", {})
                self._solver_module = solver(device=self.device, dtype=self.dtype, **solver_kw)
            return float(self._solver_module(self, X))
        if isinstance(solver, nn.Module):
            if not hasattr(self, "_solver_module") or self._solver_module is not solver:
                self._solver_module = solver
            return float(self._solver_module(self, X))
        if callable(solver):
            solver_kw = self.kwargs.get("solver_kwargs", {})
            out = solver(self, X, **solver_kw)
            return float(out) if not isinstance(out, (tuple, list)) else float(out[0])
        return self._update_cd(X)

    def fit(self, data_or_X, **kwargs):
        X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        if (X < 0).any():
            raise ValueError("NMF input must be non-negative.")
        self.kwargs.update(kwargs)
        self._init_module(X)
        W, H = self.W.data, self.H.data
        prev_err = float("inf")
        for it in range(self.max_iter):
            rec_err = self._run_solver(X)
            if abs(prev_err - rec_err) < self.tol:
                self.n_iter_ = it + 1
                break
            prev_err = rec_err
        else:
            self.n_iter_ = self.max_iter
        self.components_ = self.H.data.clone()
        self.n_components_ = self.components_.shape[0]
        beta_val = self._beta_val if self._beta_val is not None else 2
        self.reconstruction_err_ = self._calc_beta_loss(X, beta_val).item()
        return self

    def transform(self, X: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Transform X to the space of the learned components (solve for W given H)."""
        if self.components_ is None:
            raise RuntimeError("NMF instance is not fitted yet.")
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        H = self.components_
        Wt = torch.linalg.lstsq(H.T, X.T, rcond=None).solution
        W = Wt.T.clamp(min=0)
        return W

    def inverse_transform(self, X: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Transform data back to its original space (W @ H)."""
        if self.components_ is None:
            raise RuntimeError("NMF instance is not fitted yet.")
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        return X @ self.components_

    def fit_transform(
            self,
            data_or_X: Union[torch.Tensor, Any],
            **kwargs: Any,
    ) -> torch.Tensor:
        """Fit the model and transform X (return W)."""
        self.fit(data_or_X, **kwargs)
        return self.W.data.clone()


class MiniBatchNMF(NMF):
    def __init__(self,
                 n_components: Union[Literal["auto"], int, None] = "auto",
                 init: Union[
                     Literal["random", "nndsvd", "nndsvda", "nndsvdar", "custom"], Callable, nn.Module, None] = None,
                 solver: Union[Literal["cd", "mu"], Callable, nn.Module] = "cd",
                 beta_loss: Union[
                     float, Literal["frobenius", "kullback-leibler", "itakura-saito"], Callable] = "frobenius",
                 tol: float = 1e-4,
                 batch_size: int = 1024,
                 max_iter: int = 200,
                 max_no_improvement: int = 10,
                 random_state: Optional[Union[int, torch.Generator]] = None,
                 alpha_W: float = 0.0,
                 alpha_H: Union[float, Literal["same"]] = "same",
                 l1_ratio: float = 0.0,
                 forget_factor: float = 0.7,
                 fresh_restarts=False,
                 fresh_restarts_max_iter=30,
                 transform_max_iter=None,
                 verbose: int = 0,
                 shuffle: bool = False,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args: Any,
                 **kwargs: Any,
                 ) -> None:
        super().__init__(
            n_components=n_components,
            init=init,
            solver=solver,
            beta_loss=beta_loss,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state,
            alpha_W=alpha_W,
            alpha_H=alpha_H,
            l1_ratio=l1_ratio,
            verbose=verbose,
            shuffle=shuffle,
            device=device,
            dtype=dtype,
            *args, **kwargs
        )
        self.batch_size = batch_size
        self.max_no_improvement = max_no_improvement
        self.forget_factor = forget_factor
        self.fresh_restarts = fresh_restarts
        self.fresh_restarts_max_iter = fresh_restarts_max_iter
        self.transform_max_iter = transform_max_iter
        self.n_steps_ = 0

    def _check_params(self, X: torch.Tensor) -> "MiniBatchNMF":
        """Validate and set MiniBatchNMF-specific parameters. Called at fit time."""
        n_samples = X.shape[0]
        self._batch_size = min(self.batch_size, n_samples)
        self._rho = (self.forget_factor ** (self._batch_size / n_samples))
        beta_val = self._beta_val if self._beta_val is not None else 2
        if beta_val < 1:
            self._gamma = 1.0 / (2.0 - beta_val)
        elif beta_val > 2:
            self._gamma = 1.0 / (beta_val - 1.0)
        else:
            self._gamma = 1.0
        self._transform_max_iter = (
            self.max_iter if self.transform_max_iter is None else self.transform_max_iter
        )
        self._n_components = self.W.shape[1] if hasattr(self, "W") and self.W is not None else None
        return self

    def _compute_regularization(self, X_batch: torch.Tensor, n_samples_total: int) -> Tuple[float, float, float, float]:
        """Compute scaled regularization terms for a mini-batch.
        X_batch is the batch (batch_size x n_features). Regularization for W is scaled by batch,
        for H by full n_samples (H is global).
        """
        batch_size = X_batch.shape[0]
        n_features = X_batch.shape[1]
        alpha_W = self._alpha_W
        alpha_H = self._alpha_H
        l1_reg_W = n_features * alpha_W * self.l1_ratio
        l1_reg_H = n_samples_total * alpha_H * self.l1_ratio
        l2_reg_W = n_features * alpha_W * (1.0 - self.l1_ratio)
        l2_reg_H = n_samples_total * alpha_H * (1.0 - self.l1_ratio)
        return l1_reg_W, l1_reg_H, l2_reg_W, l2_reg_H

    def _solve_W(self, X: torch.Tensor, H: torch.Tensor, max_iter: int) -> torch.Tensor:
        """Minimize the objective w.r.t. W with H fixed. Used for fresh_restarts and transform."""
        eps = 1e-9
        beta = self._beta_val if self._beta_val is not None else 2
        k = H.shape[0]
        avg = torch.sqrt(X.mean() / k)
        W = torch.full((X.shape[0], k), avg, device=self.device, dtype=self.dtype)
        l1_reg_W, _, l2_reg_W, _ = self._compute_regularization(X, X.shape[0])
        gamma = self._gamma
        W_buffer = W.clone()
        for _ in range(max_iter):
            X_hat = (W @ H).clamp(min=eps)
            if beta == 2:
                W_numer = X @ H.T - l1_reg_W
                W_numer = W_numer.clamp(min=0)
                W_denom = W @ H @ H.T + l2_reg_W
                W_denom = W_denom.clamp(min=eps)
                delta = W_numer / W_denom
            elif beta == 1:
                W_numer = (X / X_hat) @ H.T
                W_denom = H.sum(dim=1).unsqueeze(0).expand(W.shape[0], -1)
                delta = W_numer / W_denom.clamp(min=eps)
            elif beta == 0:
                W_numer = (X / X_hat.pow(2)) @ H.T
                W_denom = (1.0 / X_hat) @ H.T
                delta = W_numer / W_denom.clamp(min=eps)
            else:
                W_numer = (X * X_hat.pow(beta - 2)) @ H.T
                W_denom = X_hat.pow(beta - 1) @ H.T
                delta = W_numer / W_denom.clamp(min=eps)
            if gamma != 1:
                delta = delta ** gamma
            W = W * delta
            if self.tol > 0:
                W_norm = torch.norm(W)
                if W_norm > 1e-12:
                    W_diff = torch.norm(W - W_buffer) / W_norm
                    if W_diff <= self.tol:
                        break
            W_buffer.copy_(W)
        if beta < 1:
            W[W < 1e-10] = 0.0
        return W

    def _minibatch_step(
            self,
            X_batch: torch.Tensor,
            W_batch: torch.Tensor,
            H: torch.Tensor,
            update_H: bool,
            A: Optional[torch.Tensor],
            B: Optional[torch.Tensor],
            n_samples_total: int,
    ) -> Tuple[float, torch.Tensor]:
        """Perform the update of W and H for one minibatch. Returns (batch_cost, W_new)."""
        eps = 1e-9
        beta = self._beta_val if self._beta_val is not None else 2
        gamma = self._gamma
        l1_reg_W, l1_reg_H, l2_reg_W, l2_reg_H = self._compute_regularization(
            X_batch, n_samples_total
        )
        if self.fresh_restarts or W_batch is None or W_batch.numel() == 0:
            W_new = self._solve_W(X_batch, H, self.fresh_restarts_max_iter)
        else:
            X_hat = (W_batch @ H).clamp(min=eps)
            X_clamp = X_batch.clamp(min=eps)
            if beta == 2:
                W_numer = X_batch @ H.T - l1_reg_W
                W_numer = W_numer.clamp(min=0)
                W_denom = W_batch @ H @ H.T + l2_reg_W
                W_denom = W_denom.clamp(min=eps)
                delta = W_numer / W_denom
            elif beta == 1:
                W_numer = (X_batch / X_hat) @ H.T
                W_denom = H.sum(dim=1).unsqueeze(0).expand(W_batch.shape[0], -1)
                delta = W_numer / W_denom.clamp(min=eps)
            elif beta == 0:
                W_numer = (X_batch / X_hat.pow(2)) @ H.T
                W_denom = (1.0 / X_hat) @ H.T
                delta = W_numer / W_denom.clamp(min=eps)
            else:
                W_numer = (X_batch * X_hat.pow(beta - 2)) @ H.T
                W_denom = X_hat.pow(beta - 1) @ H.T
                delta = W_numer / W_denom.clamp(min=eps)
            if gamma != 1:
                delta = delta ** gamma
            W_new = W_batch * delta
        if beta < 1:
            W_new[W_new < 1e-10] = 0.0
        batch_cost = (
                             self._calc_beta_loss(X_batch, beta, W=W_new, H=H)
                             + l1_reg_W * W_new.sum()
                             + l1_reg_H * H.sum()
                             + l2_reg_W * (W_new ** 2).sum()
                             + l2_reg_H * (H ** 2).sum()
                     ) / X_batch.shape[0]
        batch_cost = batch_cost.item()
        if update_H and A is not None and B is not None:
            X_hat = (W_new @ H).clamp(min=eps)
            X_clamp = X_batch.clamp(min=eps)
            if beta == 2:
                numer = W_new.T @ X_batch
                denom = W_new.T @ W_new @ H
            elif beta == 1:
                numer = W_new.T @ (X_batch / X_hat)
                denom = W_new.sum(dim=0).unsqueeze(1).expand(-1, H.shape[1])
            elif beta == 0:
                numer = W_new.T @ (X_batch / X_hat.pow(2))
                denom = W_new.T @ (1.0 / X_hat)
            else:
                numer = W_new.T @ (X_batch * X_hat.pow(beta - 2))
                denom = W_new.T @ X_hat.pow(beta - 1)
            denom = denom + l1_reg_H
            if l2_reg_H > 0:
                denom = denom + l2_reg_H * H
            denom = denom.clamp(min=eps)
            if gamma != 1:
                H_old = H.clone()
                H.pow_(1.0 / gamma)
                numer = numer * H
                A.mul_(self._rho)
                B.mul_(self._rho)
                A.add_(numer)
                B.add_(denom)
                H.copy_(A / B)
                H.pow_(gamma)
            else:
                A.mul_(self._rho)
                B.mul_(self._rho)
                A.add_(numer)
                B.add_(denom)
                H.copy_(A / B)
            if beta <= 1:
                H[H < 1e-10] = 0.0
        return batch_cost, W_new

    def _minibatch_convergence(
            self,
            X_batch: torch.Tensor,
            batch_cost: float,
            H: torch.Tensor,
            H_buffer: torch.Tensor,
            n_samples: int,
            step: int,
            n_steps: int,
    ) -> bool:
        """Check early stopping based on H change and smoothed cost."""
        batch_size = X_batch.shape[0]
        step_display = step + 1
        if step_display == 1:
            if self.verbose:
                print(f"Minibatch step {step_display}/{n_steps}: mean batch cost: {batch_cost}")
            return False
        if self._ewa_cost is None:
            self._ewa_cost = batch_cost
        else:
            alpha = min(batch_size / (n_samples + 1), 1.0)
            self._ewa_cost = self._ewa_cost * (1 - alpha) + batch_cost * alpha
        if self.verbose:
            print(
                f"Minibatch step {step_display}/{n_steps}: mean batch cost: "
                f"{batch_cost}, ewa cost: {self._ewa_cost}"
            )
        H_norm = torch.norm(H)
        if H_norm > 1e-12 and self.tol > 0:
            H_diff = torch.norm(H - H_buffer) / H_norm
            if H_diff <= self.tol:
                if self.verbose:
                    print(f"Converged (small H change) at step {step_display}/{n_steps}")
                return True
        if self._ewa_cost_min is None or self._ewa_cost < self._ewa_cost_min:
            self._no_improvement = 0
            self._ewa_cost_min = self._ewa_cost
        else:
            self._no_improvement += 1
        if (
                self.max_no_improvement is not None
                and self._no_improvement >= self.max_no_improvement
        ):
            if self.verbose:
                print(
                    "Converged (lack of improvement in objective function) "
                    f"at step {step_display}/{n_steps}"
                )
            return True
        return False

    def fit(self, data_or_X, y=None, W=None, H=None, **kwargs) -> "MiniBatchNMF":
        """Learn a MiniBatchNMF model for the data X."""
        X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        if (X < 0).any():
            raise ValueError("NMF input must be non-negative.")
        if X.min() == 0 and (self._beta_val is not None and self._beta_val <= 0):
            raise ValueError(
                "When beta_loss <= 0 and X contains zeros, "
                "the solver may diverge. Please add small values to X, or use a positive beta_loss."
            )
        self.kwargs.update(kwargs)
        self._init_module(X)
        self._check_params(X)
        W_mat = self.W.data
        H_mat = self.H.data
        n_samples = X.shape[0]
        H_buffer = H_mat.clone()
        self._components_numerator = H_mat.clone()
        self._components_denominator = torch.ones_like(H_mat, device=self.device, dtype=self.dtype)
        self._ewa_cost = None
        self._ewa_cost_min = None
        self._no_improvement = 0
        batches = list(_gen_batches(n_samples, self._batch_size))
        if self.shuffle and self.random_state is not None:
            g = torch.Generator(device=self.device).manual_seed(self.random_state)
            perm = torch.randperm(n_samples, device=self.device, generator=g)
            batches = [perm[b] for b in batches]
        else:
            batches = [torch.arange(b.start, b.stop, device=self.device) for b in batches]
        n_steps_per_iter = len(batches)
        n_steps = self.max_iter * n_steps_per_iter
        batches_cycle = itertools.cycle(batches)
        n_steps_done = 0
        for i in range(n_steps):
            batch_idx = next(batches_cycle)
            if isinstance(batch_idx, slice):
                X_batch = X[batch_idx]
                W_batch = W_mat[batch_idx]
            else:
                X_batch = X[batch_idx]
                W_batch = W_mat[batch_idx]
            batch_cost, W_new = self._minibatch_step(
                X_batch, W_batch, H_mat, True,
                self._components_numerator, self._components_denominator, n_samples
            )
            W_mat[batch_idx] = W_new
            if self._minibatch_convergence(X_batch, batch_cost, H_mat, H_buffer, n_samples, i, n_steps):
                n_steps_done = i + 1
                break
            H_buffer.copy_(H_mat)
            n_steps_done = i + 1
        if self.fresh_restarts:
            self.W.data = self._solve_W(X, H_mat, self._transform_max_iter)
        self.n_steps_ = n_steps_done
        self.n_iter_ = int(torch.ceil(torch.tensor(n_steps_done / max(n_steps_per_iter, 1), dtype=torch.float)).item())
        self.components_ = H_mat.clone()
        self.n_components_ = self.components_.shape[0]
        beta_val = self._beta_val if self._beta_val is not None else 2
        self.reconstruction_err_ = self._calc_beta_loss(X, beta_val).item()
        return self

    def transform(self, X: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Transform X according to the fitted MiniBatchNMF model.
        Uses iterative multiplicative updates with transform_max_iter (or max_iter if None).
        """
        if self.components_ is None:
            raise RuntimeError("MiniBatchNMF instance is not fitted yet.")
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        return self._solve_W(X, self.components_, self._transform_max_iter)

    def fit_transform(
            self,
            data_or_X: Union[torch.Tensor, Any],
            **kwargs: Any,
    ) -> torch.Tensor:
        """Fit the model and transform X (return W)."""
        self.fit(data_or_X, **kwargs)
        return self.W.data.clone()


class TruncatedSVD(MLTransform):
    def __init__(self,
                 n_components: int = 2,
                 algorithm: Union[Literal["arpack", "randomized"], Callable, nn.Module] = "randomized",
                 n_iter: int = 5,
                 n_oversamples: int = 10,
                 power_iteration_normalizer: Union[Literal["auto", "QR", "LU", "none"], Callable, nn.Module] = "auto",
                 random_state: Optional[Union[int, torch.Generator]] = None,
                 tol: float = 0.0,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args: Any,
                 **kwargs: Any,
                 ) -> None:
        super().__init__()
        self.n_components = n_components
        self.algorithm = algorithm if isinstance(algorithm, str) else algorithm
        self.n_iter = n_iter
        self.n_oversamples = n_oversamples
        self.power_iteration_normalizer = power_iteration_normalizer
        self.random_state = random_state
        self.tol = tol
        self.device = device
        self.dtype = dtype
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.n_features_in_ = None
        self.kwargs = dict(kwargs)
        self.kwargs["_args"] = args

    def _get_random_generator(self) -> Optional[torch.Generator]:
        """Return a torch Generator for reproducible randomness."""
        if self.random_state is None:
            return None
        if isinstance(self.random_state, torch.Generator):
            return self.random_state
        g = torch.Generator(device=self.device)
        g.manual_seed(int(self.random_state))
        return g

    def _randomized_svd(
            self,
            X: torch.Tensor,
            n_components: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomized SVD (Halko et al.). Supports power_iteration_normalizer:
        'auto'/'QR' -> torch.pca_lowrank; 'LU'/'none' -> custom implementation.
        """
        n_samples, n_features = X.shape
        k = min(n_components, min(n_samples, n_features))
        q = min(n_components + self.n_oversamples, min(n_samples, n_features))
        q = max(q, k)
        norm_mode = (
            self.power_iteration_normalizer.lower()
            if isinstance(self.power_iteration_normalizer, str)
            else "auto"
        )
        if norm_mode in ("auto", "qr") or (
                callable(self.power_iteration_normalizer) or isinstance(self.power_iteration_normalizer, nn.Module)):
            U, S, V = torch.pca_lowrank(X, q=q, center=False, niter=self.n_iter)
            U, S, Vh = U[:, :k], S[:k], V[:, :k].T
            return U, S, Vh
        return self._randomized_svd_custom(X, k, q, norm_mode)

    def _randomized_svd_custom(
            self,
            X: torch.Tensor,
            k: int,
            q: int,
            norm_mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Custom randomized SVD with LU/none power iteration normalizer."""
        n_samples, n_features = X.shape
        gen = self._get_random_generator()
        eps = 1e-12
        if n_samples >= n_features:
            Omega = torch.randn(n_features, q, device=self.device, dtype=self.dtype, generator=gen)
            Y = X @ Omega
            for _ in range(self.n_iter):
                Y = X.T @ (X @ Y)
                if norm_mode == "lu":
                    Y, _ = torch.linalg.lu(Y)
                elif norm_mode == "qr":
                    Y, _ = torch.linalg.qr(Y)
            Q, _ = torch.linalg.qr(Y)
            B = Q.T @ X
        else:
            Omega = torch.randn(n_samples, q, device=self.device, dtype=self.dtype, generator=gen)
            Y = X.T @ (X @ Omega.T).T
            for _ in range(self.n_iter):
                Y = X @ (X.T @ Y)
                if norm_mode == "lu":
                    Y, _ = torch.linalg.lu(Y)
                elif norm_mode == "qr":
                    Y, _ = torch.linalg.qr(Y)
            Q, _ = torch.linalg.qr(Y)
            B = (X @ Q).T
        U_b, S, Vh_b = torch.linalg.svd(B, full_matrices=False)
        U = (X @ Vh_b.T) / S.unsqueeze(0).clamp(min=eps) if n_samples >= n_features else (Q @ U_b)
        Vh = Vh_b[:k] if n_samples >= n_features else (X.T @ U / S.unsqueeze(0).clamp(min=eps))[:k]
        S = S[:k]
        if n_samples >= n_features:
            Vh = Vh_b[:k]
        else:
            Vh = (X.T @ (X @ Q @ U_b[:, :k]) / S[:k].unsqueeze(0).clamp(min=eps)).T
        return U[:, :k], S, Vh

    def _arpack_svd(
            self,
            X: torch.Tensor,
            n_components: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ARPACK-style truncated SVD from scratch using power iteration on the Gram matrix.
        PyTorch has no ARPACK; we implement power iteration + deflation on X.T@X or X@X.T.
        """
        n_samples, n_features = X.shape
        k = min(n_components, min(n_samples, n_features) - 1)
        k = max(1, k)
        eps = 1e-12
        tol = self.tol if self.tol > 0 else eps
        gen = self._get_random_generator()
        if n_features <= n_samples:
            G = X.T @ X
            n_g = n_features
            transposed = False
        else:
            G = X @ X.T
            n_g = n_samples
            transposed = True
        S_sq = []
        V_list = []
        G_work = G.clone()
        for j in range(k):
            v = torch.randn(n_g, device=self.device, dtype=self.dtype, generator=gen)
            v = v / v.norm()
            for _ in range(max(self.n_iter, 10)):
                v_new = G_work @ v
                v_new = v_new / (v_new.norm().clamp(min=eps))
                if (v_new - v).norm() < tol:
                    break
                v = v_new
            lam = (v @ G_work @ v).item()
            lam = max(lam, eps)
            s = math.sqrt(lam)
            S_sq.append(s)
            V_list.append(v.clone())
            G_work = G_work - lam * torch.outer(v, v)
        S = torch.tensor(S_sq, device=self.device, dtype=self.dtype)
        V_stack = torch.stack(V_list, dim=1)
        if transposed:
            # V_stack columns are eigenvectors of X@X.T = left singular vectors U
            U = V_stack  # (n_samples, k)
            # Vh = (k, n_features): V = X.T @ U / S, so Vh = V.T = (X.T @ U).T / S = U.T @ X / S
            S_clamp = S.clamp(min=eps).unsqueeze(1)  # (k, 1) for broadcasting
            Vh = (U.T @ X) / S_clamp  # (k, n_samples) @ (n_samples, n_features) -> (k, n_features)
            return U, S, Vh
        else:
            # V_stack columns are eigenvectors of X.T@X = right singular vectors (rows of Vh)
            Vh = V_stack.T
            U = (X @ V_stack) / S.clamp(min=eps)
            return U, S, Vh

    def fit(self, data_or_X: Union[torch.Tensor, Any], y: Any = None, **kwargs: Any) -> "TruncatedSVD":
        """Fit model on training data X."""
        X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        n_comp = min(self.n_components, min(n_samples, n_features))
        if isinstance(self.algorithm, str) and self.algorithm == "arpack":
            if n_comp >= min(n_samples, n_features):
                n_comp = max(1, min(n_samples, n_features) - 1)
            if n_comp < 1:
                raise ValueError(
                    "arpack requires n_components strictly less than min(n_samples, n_features)"
                )
            U, S, Vh = self._arpack_svd(X, n_comp)
        elif isinstance(self.algorithm, str) and self.algorithm == "randomized":
            if self.n_components > n_features:
                raise ValueError(
                    f"n_components({self.n_components}) must be <= n_features({n_features})"
                )
            U, S, Vh = self._randomized_svd(X, n_comp)
        elif callable(self.algorithm) or isinstance(self.algorithm, nn.Module):
            solver_kw = self.kwargs.get("algorithm_kwargs", {})
            out = self.algorithm(X, n_comp, device=self.device, dtype=self.dtype, **solver_kw)
            U, S, Vh = out[0], out[1], out[2]
        else:
            U, S, Vh = self._randomized_svd(X, n_comp)
        self.components_ = Vh
        self.singular_values_ = S
        X_transformed = X @ self.components_.T
        self.explained_variance_ = torch.var(X_transformed, dim=0, unbiased=False)
        full_var = torch.var(X, dim=0, unbiased=False).sum().clamp(min=1e-12)
        self.explained_variance_ratio_ = self.explained_variance_ / full_var
        return self

    def transform(self, X: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Perform dimensionality reduction on X."""
        if self.components_ is None:
            raise RuntimeError("TruncatedSVD instance is not fitted yet.")
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        return X @ self.components_.T

    def fit_transform(
            self,
            data_or_X: Union[torch.Tensor, Any],
            y: Any = None,
            **kwargs: Any,
    ) -> torch.Tensor:
        """Fit model to X and perform dimensionality reduction on X."""
        self.fit(data_or_X, y=y, **kwargs)
        return (torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype) @ self.components_.T)

    def inverse_transform(self, X: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Transform X back to its original space."""
        if self.components_ is None:
            raise RuntimeError("TruncatedSVD instance is not fitted yet.")
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        return X @ self.components_


class FactorAnalysis(MLTransform):
    def __init__(self,
                 n_components: int = None,
                 tol: float = 1e-2,
                 copy: bool = True,
                 max_iter: int = 1000,
                 noise_variance_init: Union[list, tuple, torch.Tensor] = None,
                 svd_method: Union[Literal["lapack", "la_pack", "randomized"], Callable, nn.Module] = "randomized",
                 iterated_power: int = 3,
                 rotation: Union[Literal["varimax", "quartimax"], Callable] = None,
                 random_state: Optional[Union[int, torch.Generator]] = 0,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args: Any,
                 **kwargs: Any,
                 ) -> None:
        super().__init__()
        self.n_components = n_components
        self.tol = tol
        self.copy = copy
        self.max_iter = max_iter
        self.noise_variance_init = noise_variance_init
        _svd = svd_method if isinstance(svd_method, str) else svd_method
        self.svd_method = "lapack" if _svd in ("lapack", "la_pack") else _svd
        self.iterated_power = iterated_power
        self.rotation = rotation if isinstance(rotation, str) or rotation is None else rotation
        self.random_state = random_state
        self.device = device
        self.dtype = dtype
        self.kwargs = dict(kwargs)
        self.kwargs["_args"] = args

        self.components_ = None
        self.loglike_ = []
        self.noise_variance_ = None
        self.n_iter_ = 0
        self.mean_ = None
        self.n_features_in_ = None

    def _get_random_generator(self) -> Optional[torch.Generator]:
        """Return a torch Generator for reproducible randomness."""
        if self.random_state is None:
            return None
        if isinstance(self.random_state, torch.Generator):
            return self.random_state
        g = torch.Generator(device=self.device)
        g.manual_seed(int(self.random_state))
        return g

    def _svd_lapack(self, X: torch.Tensor, n_components: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full SVD via torch.linalg.svd (LAPACK-style)."""
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        k = min(n_components, S.shape[0])
        return U[:, :k], S[:k], Vh[:k]

    def _svd_randomized(
            self, X: torch.Tensor, n_components: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomized SVD via torch.pca_lowrank (or custom if generator not supported)."""
        n_samples, n_features = X.shape
        k = min(n_components, min(n_samples, n_features))
        q = min(k + 10, min(n_samples, n_features))
        q = max(q, k)
        gen = self._get_random_generator()
        try:
            U, S, V = torch.pca_lowrank(
                X, q=q, center=False, niter=self.iterated_power, generator=gen
            )
        except TypeError:
            if gen is not None:
                torch.manual_seed(int(self.random_state) if isinstance(self.random_state, int) else 0)
            U, S, V = torch.pca_lowrank(
                X, q=q, center=False, niter=self.iterated_power
            )
        Vh = V[:, :k].T
        return U[:, :k], S[:k], Vh

    def _log_likelihood(
            self, X_centered: torch.Tensor, L: torch.Tensor, psi: torch.Tensor
    ) -> float:
        """Compute log-likelihood: -n/2 * (p*log(2*pi) + log|C| + tr(C^{-1}*S))."""
        n, p = X_centered.shape
        C = L @ L.T + torch.diag(psi)
        sign, logdet = torch.linalg.slogdet(C)
        if sign <= 0:
            return float("-inf")
        Cinv = torch.linalg.inv(C)
        S = (X_centered.T @ X_centered) / n
        tr_term = (Cinv * S).sum().item()
        return (-n / 2.0) * (p * math.log(2 * math.pi) + logdet.item() + tr_term)

    def _fit_em(
            self,
            X_centered: torch.Tensor,
            n_components: int,
            psi_init: torch.Tensor,
            L_init: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
        """EM algorithm for Factor Analysis."""
        n, p = X_centered.shape
        eps = 1e-12
        L = L_init.clone()
        psi = psi_init.clone().clamp(min=eps)
        loglike_list = []

        for it in range(self.max_iter):
            C = L @ L.T + torch.diag(psi)
            C = C + eps * torch.eye(p, device=C.device, dtype=C.dtype)
            Cinv = torch.linalg.inv(C)
            beta = L.T @ Cinv

            sum_xf = torch.zeros(p, n_components, device=self.device, dtype=self.dtype)
            sum_ff = torch.zeros(n_components, n_components, device=self.device, dtype=self.dtype)

            for i in range(n):
                xi = X_centered[i: i + 1].T
                Ez = beta @ xi
                Ezz = torch.eye(n_components, device=self.device, dtype=self.dtype) - beta @ L + beta @ (
                        xi @ xi.T) @ beta.T
                sum_xf += xi @ Ez.T
                sum_ff += Ezz

            L_new = (sum_xf @ torch.linalg.inv(
                sum_ff + eps * torch.eye(n_components, device=self.device, dtype=self.dtype)))
            S = (X_centered.T @ X_centered) / n
            psi_new = torch.diag(S - L_new @ (sum_xf / n).T)
            psi_new = psi_new.clamp(min=eps)

            loglike = self._log_likelihood(X_centered, L_new, psi_new)
            loglike_list.append(loglike)

            if it > 0 and loglike - loglike_list[-2] < self.tol:
                L = L_new
                psi = psi_new
                break
            L = L_new
            psi = psi_new

        return L, psi, loglike_list

    def _varimax(self, L: torch.Tensor) -> torch.Tensor:
        """Varimax rotation: maximize variance of squared loadings per factor.
        L is (p, k). Rotates columns (factors) via L @ R, R (k, k) orthogonal.
        Uses iterative 2x2 Givens rotations (Kaiser 1958).
        """
        p, k = L.shape
        if k < 2:
            return L
        eps = 1e-12
        L_rot = L.clone()
        for _ in range(20):
            for j1 in range(k):
                for j2 in range(j1 + 1, k):
                    u, v = L_rot[:, j1], L_rot[:, j2]
                    u2, v2 = u ** 2, v ** 2
                    A = p * (2 * (u * v * (u2 - v2)).sum()) - 2 * (u * v).sum() * ((u2 - v2).sum())
                    B = p * ((u2 - v2) ** 2 - 4 * u2 * v2).sum() - ((u2 - v2).sum()) ** 2 + 4 * ((u * v).sum()) ** 2
                    denom = B.abs().clamp(min=eps)
                    tan4t = A / denom
                    t = torch.atan(tan4t.clamp(-1e12, 1e12)) / 4
                    c, s = torch.cos(t).item(), torch.sin(t).item()
                    R2 = torch.tensor([[c, -s], [s, c]], device=L.device, dtype=L.dtype)
                    L_rot[:, [j1, j2]] = L_rot[:, [j1, j2]] @ R2.T
        return L_rot

    def _quartimax(self, L: torch.Tensor) -> torch.Tensor:
        """Quartimax rotation: maximize sum of fourth powers of loadings.
        L is (p, k). Rotates columns via L @ R, R (k, k) orthogonal.
        """
        p, k = L.shape
        if k < 2:
            return L
        L_rot = L.clone()
        for _ in range(20):
            for j1 in range(k):
                for j2 in range(j1 + 1, k):
                    u, v = L_rot[:, j1], L_rot[:, j2]
                    u2, v2 = u ** 2, v ** 2
                    A = 4 * (u * v * (u2 - v2)).sum()
                    B = ((u2 - v2) ** 2 - 4 * u2 * v2).sum()
                    denom = B.abs().clamp(min=1e-12)
                    tan4t = A / denom
                    t = torch.atan(tan4t.clamp(-1e12, 1e12)) / 4
                    c, s = torch.cos(t).item(), torch.sin(t).item()
                    R2 = torch.tensor([[c, -s], [s, c]], device=L.device, dtype=L.dtype)
                    L_rot[:, [j1, j2]] = L_rot[:, [j1, j2]] @ R2.T
        return L_rot

    def fit(self, data_or_X: Union[torch.Tensor, Any], y: Any = None, **kwargs: Any) -> "FactorAnalysis":
        """Fit the Factor Analysis model to X using EM."""
        X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        if self.copy:
            X = X.clone()
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        n_comp = self.n_components
        if n_comp is None:
            n_comp = n_features
        n_comp = min(n_comp, min(n_samples, n_features))

        self.mean_ = X.mean(dim=0)
        X_centered = X - self.mean_

        psi_init = torch.ones(n_features, device=self.device, dtype=self.dtype)
        if self.noise_variance_init is not None:
            psi_init = torch.as_tensor(
                self.noise_variance_init, device=self.device, dtype=self.dtype
            )
            if psi_init.dim() == 0:
                psi_init = psi_init.unsqueeze(0)
            if psi_init.shape[0] != n_features:
                raise ValueError(
                    f"noise_variance_init shape {psi_init.shape[0]} != n_features {n_features}"
                )

        svd_m = self.svd_method
        if isinstance(svd_m, str):
            if svd_m in ("lapack", "la_pack"):
                U, S, Vh = self._svd_lapack(X_centered, n_comp)
            else:
                U, S, Vh = self._svd_randomized(X_centered, n_comp)
        elif callable(svd_m) or isinstance(svd_m, nn.Module):
            solver_kw = self.kwargs.get("svd_method_kwargs", {})
            out = svd_m(X_centered, n_comp, device=self.device, dtype=self.dtype, **solver_kw)
            U, S, Vh = out[0], out[1], out[2]
        else:
            U, S, Vh = self._svd_randomized(X_centered, n_comp)

        # L (p, k): loading matrix. From SVD X = U S Vh, loadings are Vh.T scaled by sqrt(max(0, S^2/n - psi_avg))
        Vh = Vh[:n_comp]
        S = S[:n_comp]
        psi_avg = psi_init.mean().clamp(min=1e-6)
        scale = torch.sqrt(torch.clamp(S ** 2 / n_samples - psi_avg, min=1e-8)).unsqueeze(0)
        L_init = (Vh.T * scale).to(self.device).to(self.dtype)

        L, psi, loglike_list = self._fit_em(X_centered, n_comp, psi_init, L_init)

        if self.rotation is not None:
            rot = self.rotation.lower() if isinstance(self.rotation, str) else self.rotation
            if rot == "varimax":
                L = self._varimax(L)
            elif rot == "quartimax":
                L = self._quartimax(L)
            elif callable(self.rotation):
                L = self.rotation(L)

        self.components_ = L
        self.noise_variance_ = psi
        self.loglike_ = loglike_list
        self.n_iter_ = len(loglike_list)
        return self

    def transform(self, X: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Apply dimensionality reduction to X."""
        if self.components_ is None:
            raise RuntimeError("FactorAnalysis instance is not fitted yet.")
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        X_centered = X - self.mean_
        L = self.components_
        psi = self.noise_variance_
        C = L @ L.T + torch.diag(psi)
        Cinv = torch.linalg.inv(C)
        beta = L.T @ Cinv
        return (X_centered @ beta.T)

    def fit_transform(
            self,
            data_or_X: Union[torch.Tensor, Any],
            y: Any = None,
            **kwargs: Any,
    ) -> torch.Tensor:
        """Fit the model and transform X."""
        self.fit(data_or_X, y=y, **kwargs)
        return self.transform(torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype))

    def get_covariance(self) -> torch.Tensor:
        """Compute the estimated covariance matrix: L L^T + Psi."""
        if self.components_ is None:
            raise RuntimeError("FactorAnalysis instance is not fitted yet.")
        return self.components_ @ self.components_.T + torch.diag(self.noise_variance_)


class LatentDirichletAllocation(MLTransform):
    def __init__(self,
                 n_components: int = 10,
                 doc_topic_prior: float = None,
                 topic_word_prior: float = None,
                 learning_method: Union[Literal["batch", "online"], Callable, nn.Module] = 'batch',
                 learning_decay: float = 0.7,
                 learning_offset: float = 10.0,
                 max_iter: int = 10,
                 batch_size: int = 128,
                 evaluate_every: int = -1,
                 total_samples: int = 1e6,
                 perp_tol: float = 0.1,
                 mean_change_tol: float = 0.001,
                 max_doc_update_iter: int = 100,
                 n_jobs: int = None,
                 verbose: int = 0,
                 random_state: Union[int, torch.Generator] = None,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args: Any,
                 **kwargs: Any,
                 ) -> None:
        super().__init__()
        self.n_components = n_components
        self.doc_topic_prior = doc_topic_prior
        self.topic_word_prior = topic_word_prior
        _lm = learning_method if isinstance(learning_method, str) else learning_method
        self.learning_method = _lm if _lm in ("batch", "online") else "batch"
        self.learning_decay = learning_decay
        self.learning_offset = learning_offset
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.evaluate_every = evaluate_every
        self.total_samples = int(total_samples)
        self.perp_tol = perp_tol
        self.mean_change_tol = mean_change_tol
        self.max_doc_update_iter = max_doc_update_iter
        self.n_jobs = n_jobs if n_jobs is not None else 1
        self.verbose = verbose
        self.random_state = random_state
        self.device = device
        self.dtype = dtype
        self.kwargs = dict(kwargs)
        self.kwargs["_args"] = args

        self.components_ = None
        self.exp_dirichlet_component_ = None
        self.n_batch_iter_ = 0
        self.n_features_in_ = None
        self.n_iter_ = 0
        self.bound_ = None
        self.doc_topic_prior_ = None
        self.topic_word_prior_ = None
        self.random_state_ = random_state

    def _get_random_generator(self) -> Optional[torch.Generator]:
        """Return a torch Generator for reproducible randomness."""
        if self.random_state is None:
            return None
        if isinstance(self.random_state, torch.Generator):
            return self.random_state
        g = torch.Generator(device=self.device)
        g.manual_seed(int(self.random_state))
        return g

    def _init_latent_vars(self, n_features: int) -> None:
        """Initialize components_ (lambda) with prior + small random positive values."""
        gen = self._get_random_generator()
        eps = 1e-10
        lam = torch.ones(
            self.n_components, n_features, device=self.device, dtype=self.dtype
        ) * (self.topic_word_prior_ + eps)
        if gen is not None:
            lam = lam + torch.rand(self.n_components, n_features, device=self.device, dtype=self.dtype,
                                   generator=gen) * 0.01
        else:
            lam = lam + torch.rand(self.n_components, n_features, device=self.device, dtype=self.dtype) * 0.01
        self.components_ = lam.clamp(min=eps)
        self._update_exp_dirichlet_component_()

    def _update_exp_dirichlet_component_(self) -> None:
        """Update exp_dirichlet_component_ = exp(E[log(beta)])."""
        if self.components_ is None:
            return
        lam = self.components_
        psi_lam = torch.digamma(lam)
        psi_sum = torch.digamma(lam.sum(dim=1, keepdim=True))
        self.exp_dirichlet_component_ = torch.exp((psi_lam - psi_sum).clamp(max=20))

    def _e_step(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """E-step: update gamma and phi. Returns (gamma, sufficient_stats)."""
        n_docs, n_features = X.shape
        alpha = self.doc_topic_prior_
        exp_beta = self.exp_dirichlet_component_
        eps = 1e-10
        gamma = torch.zeros(n_docs, self.n_components, device=self.device, dtype=self.dtype)
        suff_stats = torch.zeros(self.n_components, n_features, device=self.device, dtype=self.dtype)
        for d in range(n_docs):
            x_d = X[d: d + 1]
            n_d = x_d.sum().item()
            if n_d < eps:
                gamma[d] = alpha
                continue
            gamma_d = alpha + torch.ones(self.n_components, device=self.device, dtype=self.dtype) * (
                    n_d / self.n_components)
            for _ in range(self.max_doc_update_iter):
                gamma_d_old = gamma_d.clone()
                psi_gamma = torch.digamma(gamma_d)
                psi_sum = torch.digamma(gamma_d.sum())
                log_phi = (torch.log(exp_beta + eps) + (psi_gamma - psi_sum).unsqueeze(1))
                log_phi = log_phi.clamp(max=50)
                phi = F.softmax(log_phi, dim=0)
                gamma_d = alpha + (x_d * phi).sum(dim=1)
                if (gamma_d - gamma_d_old).abs().max() < self.mean_change_tol:
                    break
            gamma[d] = gamma_d
            suff_stats += phi * x_d
        return gamma, suff_stats

    def _m_step_batch(self, X: torch.Tensor, suff_stats: torch.Tensor) -> None:
        """M-step for batch."""
        eta = self.topic_word_prior_
        self.components_ = eta + suff_stats
        self.components_ = self.components_.clamp(min=1e-10)
        self._update_exp_dirichlet_component_()

    def _m_step_online(self, X: torch.Tensor, suff_stats: torch.Tensor, batch_idx: int) -> None:
        """M-step for online."""
        eta = self.topic_word_prior_
        rho = (batch_idx + self.learning_offset) ** (-self.learning_decay)
        batch_suff = eta + suff_stats
        if self.components_ is None:
            self.components_ = batch_suff.clone()
        else:
            self.components_ = (1 - rho) * self.components_ + rho * batch_suff
        self.components_ = self.components_.clamp(min=1e-10)
        self._update_exp_dirichlet_component_()

    def _approx_bound(self, X: torch.Tensor, gamma: torch.Tensor) -> float:
        """Approximate variational bound."""
        n_docs, n_features = X.shape
        alpha = self.doc_topic_prior_
        eta = self.topic_word_prior_
        lam = self.components_
        eps = 1e-10
        bound = 0.0
        for d in range(n_docs):
            x_d = X[d]
            n_d = x_d.sum().item()
            if n_d < eps:
                continue
            gamma_d = gamma[d]
            psi_gamma = torch.digamma(gamma_d)
            psi_sum = torch.digamma(gamma_d.sum())
            log_phi = torch.log(self.exp_dirichlet_component_ + eps) + (psi_gamma - psi_sum).unsqueeze(1)
            phi = F.softmax(log_phi.clamp(max=50), dim=0)
            bound += (
                    (alpha - gamma_d).dot(psi_gamma - psi_sum)
                    + (x_d * (phi * (psi_gamma - psi_sum).unsqueeze(1) + torch.log(
                self.exp_dirichlet_component_ + eps))).sum()
            )
        bound -= n_docs * (math.lgamma(alpha * self.n_components) - self.n_components * math.lgamma(alpha))
        bound -= (torch.lgamma(lam).sum() - torch.lgamma(
            lam.sum(dim=1)).sum() - self.n_components * n_features * math.lgamma(eta)).item()
        return bound

    def fit(self, data_or_X: Union[torch.Tensor, Any], y: Any = None, **kwargs: Any) -> "LatentDirichletAllocation":
        """Fit the LDA model with variational Bayes."""
        X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        if (X < 0).any():
            raise ValueError("LatentDirichletAllocation requires non-negative input.")
        n_docs, n_features = X.shape
        self.n_features_in_ = n_features
        self.doc_topic_prior_ = self.doc_topic_prior if self.doc_topic_prior is not None else 1.0 / self.n_components
        self.topic_word_prior_ = self.topic_word_prior if self.topic_word_prior is not None else 1.0 / self.n_components
        self._init_latent_vars(n_features)
        total_words = X.sum().item()
        batch_iter = 0
        lm = self.learning_method
        for epoch in range(self.max_iter):
            if lm == "batch":
                gamma, suff_stats = self._e_step(X)
                self._m_step_batch(X, suff_stats)
                batch_iter += 1
            else:
                for batch_slice in _gen_batches(n_docs, self.batch_size):
                    X_batch = X[batch_slice]
                    gamma, suff_stats = self._e_step(X_batch)
                    self._m_step_online(X_batch, suff_stats, batch_iter)
                    batch_iter += 1
            if self.evaluate_every > 0 and (epoch + 1) % self.evaluate_every == 0:
                gamma_full, _ = self._e_step(X)
                bound = self._approx_bound(X, gamma_full)
                perp = math.exp(-bound / max(total_words, 1e-10))
                if self.verbose:
                    print(f"Epoch {epoch + 1} perplexity {perp:.4f}")
                if epoch > 0 and abs(perp - getattr(self, "_last_perp", perp)) < self.perp_tol:
                    break
                self._last_perp = perp
        gamma_final, _ = self._e_step(X)
        self.bound_ = self._approx_bound(X, gamma_final)
        self.n_batch_iter_ = batch_iter
        self.n_iter_ = self.max_iter
        return self

    def partial_fit(self, data_or_X: Union[torch.Tensor, Any], y: Any = None,
                    **kwargs: Any) -> "LatentDirichletAllocation":
        """Update the model with a mini-batch (online learning)."""
        X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        if (X < 0).any():
            raise ValueError("LatentDirichletAllocation requires non-negative input.")
        n_docs, n_features = X.shape
        if self.n_features_in_ is None:
            self.n_features_in_ = n_features
            self.doc_topic_prior_ = self.doc_topic_prior if self.doc_topic_prior is not None else 1.0 / self.n_components
            self.topic_word_prior_ = self.topic_word_prior if self.topic_word_prior is not None else 1.0 / self.n_components
            self._init_latent_vars(n_features)
        batch_idx = getattr(self, "_partial_fit_batch_idx", 0)
        gamma, suff_stats = self._e_step(X)
        self._m_step_online(X, suff_stats, batch_idx)
        self._partial_fit_batch_idx = batch_idx + 1
        return self

    def transform(self, X: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Transform X to document-topic distribution."""
        if self.components_ is None:
            raise RuntimeError("LatentDirichletAllocation instance is not fitted yet.")
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        gamma, _ = self._e_step(X)
        return gamma / gamma.sum(dim=1, keepdim=True)

    def fit_transform(self, data_or_X: Union[torch.Tensor, Any], y: Any = None, **kwargs: Any) -> torch.Tensor:
        """Fit and transform X to document-topic distribution."""
        self.fit(data_or_X, y=y, **kwargs)
        return self.transform(torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype))


class DictionaryLearning(MLTransform):
    def __init__(self,
                 n_components: int = None,
                 alpha: float = 1,
                 max_iter: int = 1000,
                 tol: float = 1e-08,
                 fit_algorithm: Union[Literal["lars", "cd"], Callable, nn.Module]='lars',
                 transform_algorithm: Union[Literal["lars", "lasso_lars", "lasso_cd", "omp", "threshold"], Callable, nn.Module]='omp',
                 transform_n_nonzero_coefs: int = None,
                 transform_alpha: float = None,
                 n_jobs: int = None,
                 code_init: Union[list, tuple, torch.Tensor]=None,
                 dict_init: Union[list, tuple, torch.Tensor]=None,
                 callback: Union[Callable, nn.Module]=None,
                 verbose: bool = False,
                 split_sign: bool = False,
                 random_state: Union[int, torch.Generator]=None,
                 positive_code: bool = False,
                 positive_dict: bool = False,
                 transform_max_iter: int = 1000,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args: Any,
                 **kwargs: Any):
        super().__init__()
        self.n_components = n_components
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_algorithm = fit_algorithm if isinstance(fit_algorithm, str) else fit_algorithm
        self.transform_algorithm = transform_algorithm if isinstance(transform_algorithm, str) else transform_algorithm
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs
        self.transform_alpha = transform_alpha
        self.n_jobs = n_jobs if n_jobs is not None else 1
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
        self.kwargs = dict(kwargs)
        self.kwargs["_args"] = args

        self.components_ = None
        self.error_ = []
        self.n_features_in_ = None
        self.n_iter_ = 0

    def _get_random_generator(self) -> Optional[torch.Generator]:
        """Return a torch Generator for reproducible randomness."""
        if self.random_state is None:
            return None
        if isinstance(self.random_state, torch.Generator):
            return self.random_state
        g = torch.Generator(device=self.device)
        g.manual_seed(int(self.random_state))
        return g

    def _sparse_code_omp(
            self,
            X: torch.Tensor,
            D: torch.Tensor,
            n_nonzero: int,
            positive: bool = False,
    ) -> torch.Tensor:
        """Orthogonal Matching Pursuit: greedy sparse coding."""
        n_samples, n_features = X.shape
        n_atoms = D.shape[0]
        code = torch.zeros(n_samples, n_atoms, device=X.device, dtype=X.dtype)
        for i in range(n_samples):
            x = X[i]
            r = x.clone()
            active = torch.zeros(n_atoms, dtype=torch.bool, device=X.device)
            for _ in range(min(n_nonzero, n_atoms)):
                proj = D @ r
                if positive:
                    proj = proj.clamp(min=0)
                proj[active] = -float("inf")
                j = proj.abs().argmax().item()
                active[j] = True
                D_active = D[active]
                c_active = torch.linalg.lstsq(
                    D_active.T,
                    x.unsqueeze(1),
                    rcond=None
                ).solution.squeeze(1)
                if positive:
                    c_active = c_active.clamp(min=0)
                code[i, active] = c_active
                r = x - (D_active.T @ c_active)
                if r.norm() < 1e-12:
                    break
        return code

    def _sparse_code_threshold(
            self,
            X: torch.Tensor,
            D: torch.Tensor,
            alpha: float,
            positive: bool = False,
    ) -> torch.Tensor:
        """Threshold: code = D @ X.T, then soft-threshold by alpha."""
        code = X @ D.T
        if positive:
            code = code.clamp(min=0)
        code = torch.sign(code) * (code.abs() - alpha).clamp(min=0)
        return code

    def _sparse_code_cd(
            self,
            X: torch.Tensor,
            D: torch.Tensor,
            alpha: float,
            max_iter: int,
            positive: bool = False,
    ) -> torch.Tensor:
        """Coordinate descent for Lasso: min 0.5||x - Dc||^2 + alpha||c||_1."""
        n_samples, n_features = X.shape
        n_atoms = D.shape[0]
        DtD = D @ D.T
        DtX = D @ X.T
        eps = 1e-12
        code = torch.zeros(n_samples, n_atoms, device=X.device, dtype=X.dtype)
        for _ in range(max_iter):
            for j in range(n_atoms):
                residual = DtX[j] - (code @ DtD[:, j]) + code[:, j] * DtD[j, j]
                d_jj = DtD[j, j].clamp(min=eps)
                if d_jj > eps:
                    thresh = alpha / d_jj
                    c_new = residual / d_jj
                    if positive:
                        c_new = c_new.clamp(min=0)
                    else:
                        c_new = torch.sign(c_new) * (c_new.abs() - thresh).clamp(min=0)
                    code[:, j] = c_new
        return code

    def _sparse_code_lars(
            self,
            X: torch.Tensor,
            D: torch.Tensor,
            alpha: float,
            positive: bool = False,
    ) -> torch.Tensor:
        """Lasso via LARS: uses LassoLars from regression module."""
        return self._sparse_code_via_lassolars(X, D, alpha, positive)

    def _sparse_code_via_lasso(
            self,
            X: torch.Tensor,
            D: torch.Tensor,
            alpha: float,
            max_iter: int,
            positive: bool = False,
    ) -> torch.Tensor:
        """Sparse coding via Lasso (coordinate descent) from regression module."""
        n_samples, n_features = X.shape
        n_atoms = D.shape[0]
        code = torch.zeros(n_samples, n_atoms, device=X.device, dtype=X.dtype)
        Dt = D.T
        lasso = Lasso(
            alpha=alpha,
            fit_intercept=False,
            max_iter=max_iter,
            tol=self.tol,
            positive=positive,
            device=str(X.device),
            dtype=X.dtype,
        )
        for i in range(n_samples):
            lasso.fit(Dt, X[i])
            coef = lasso.weight.squeeze() if lasso.weight.ndim > 1 else lasso.weight
            if coef.ndim == 0:
                coef = coef.unsqueeze(0)
            code[i] = coef[:n_atoms]
        return code

    def _sparse_code_via_lassolars(
            self,
            X: torch.Tensor,
            D: torch.Tensor,
            alpha: float,
            positive: bool = False,
    ) -> torch.Tensor:
        """Sparse coding via LassoLars from regression module."""
        n_samples, n_features = X.shape
        n_atoms = D.shape[0]
        code = torch.zeros(n_samples, n_atoms, device=X.device, dtype=X.dtype)
        Dt = D.T
        lassolars = LassoLars(
            alpha=alpha,
            fit_intercept=False,
            positive=positive,
            device=str(X.device),
            dtype=X.dtype,
        )
        for i in range(n_samples):
            lassolars.fit(Dt, X[i])
            coef = lassolars.weight.squeeze() if lassolars.weight.ndim > 1 else lassolars.weight
            if coef.ndim == 0:
                coef = coef.unsqueeze(0)
            code[i] = coef[:n_atoms]
        return code

    def _sparse_code_via_lars(
            self,
            X: torch.Tensor,
            D: torch.Tensor,
            n_nonzero: Optional[int] = None,
            positive: bool = False,
    ) -> torch.Tensor:
        """Sparse coding via Lars from regression module (n_nonzero_coefs)."""
        n_samples, n_features = X.shape
        n_atoms = D.shape[0]
        n_nz = n_nonzero if n_nonzero is not None else max(1, n_features // 10)
        code = torch.zeros(n_samples, n_atoms, device=X.device, dtype=X.dtype)
        Dt = D.T
        lars = Lars(
            fit_intercept=False,
            n_nonzero_coefs=min(n_nz, n_atoms),
            positive=positive,
            device=str(X.device),
            dtype=X.dtype,
        )
        for i in range(n_samples):
            lars.fit(Dt, X[i])
            coef = lars.weight.squeeze() if lars.weight.ndim > 1 else lars.weight
            if coef.ndim == 0:
                coef = coef.unsqueeze(0)
            code[i] = coef[:n_atoms]
        return code

    def _sparse_code_via_omp(
            self,
            X: torch.Tensor,
            D: torch.Tensor,
            n_nonzero: int,
            positive: bool = False,
    ) -> torch.Tensor:
        """Sparse coding via OrthogonalMatchingPursuit from regression module."""
        n_samples, n_features = X.shape
        n_atoms = D.shape[0]
        code = torch.zeros(n_samples, n_atoms, device=X.device, dtype=X.dtype)
        Dt = D.T
        omp = OrthogonalMatchingPursuit(
            n_nonzero_coefs=min(n_nonzero, n_atoms),
            fit_intercept=False,
            positive=positive,
            device=str(X.device),
            dtype=X.dtype,
        )
        for i in range(n_samples):
            omp.fit(Dt, X[i])
            coef = omp.weight.squeeze() if omp.weight.ndim > 1 else omp.weight
            if coef.ndim == 0:
                coef = coef.unsqueeze(0)
            code[i] = coef[:n_atoms]
        return code

    def _fit_sparse_code(
            self,
            X: torch.Tensor,
            D: torch.Tensor,
            alpha: float,
            n_nonzero: Optional[int],
            algorithm: str,
            max_iter: int,
    ) -> torch.Tensor:
        """Dispatch sparse coding during fit."""
        n_samples, n_features = X.shape
        n_atoms = D.shape[0]
        transform_alpha = self.transform_alpha if self.transform_alpha is not None else alpha
        n_nz = n_nonzero if n_nonzero is not None else max(1, n_features // 10)

        if callable(algorithm) or isinstance(algorithm, nn.Module):
            solver_kw = self.kwargs.get("fit_algorithm_kwargs", {})
            return algorithm(
                X, D, alpha=alpha, n_nonzero=n_nz, max_iter=max_iter,
                device=self.device, dtype=self.dtype, **solver_kw
            )
        if algorithm in ("lars", "cd"):
            if algorithm == "lars":
                return self._sparse_code_via_lassolars(X, D, alpha, self.positive_code)
            return self._sparse_code_via_lasso(X, D, alpha, max_iter, self.positive_code)
        raise ValueError(f"fit_algorithm must be 'lars', 'cd', or a Callable/nn.Module, got {algorithm}")

    def _transform_sparse_code(
            self,
            X: torch.Tensor,
            D: torch.Tensor,
            algorithm: str,
    ) -> torch.Tensor:
        """Dispatch sparse coding for transform."""
        n_features = D.shape[1]
        alpha = self.alpha
        transform_alpha = self.transform_alpha if self.transform_alpha is not None else alpha
        n_nz = self.transform_n_nonzero_coefs
        if n_nz is None:
            n_nz = max(1, n_features // 10)
        max_iter = self.transform_max_iter

        algo = (
            self.transform_algorithm.lower()
            if isinstance(self.transform_algorithm, str)
            else "omp"
        )
        if callable(self.transform_algorithm) or isinstance(self.transform_algorithm, nn.Module):
            solver_kw = self.kwargs.get("transform_algorithm_kwargs", {})
            return self.transform_algorithm(
                X, D, alpha=transform_alpha, n_nonzero=n_nz,
                device=self.device, dtype=self.dtype, **solver_kw
            )
        if algo == "omp":
            return self._sparse_code_via_omp(X, D, n_nz, self.positive_code)
        if algo == "threshold":
            return self._sparse_code_threshold(X, D, transform_alpha, self.positive_code)
        if algo in ("lasso_cd", "lasso_lars"):
            if algo == "lasso_cd":
                return self._sparse_code_via_lasso(X, D, transform_alpha, max_iter, self.positive_code)
            return self._sparse_code_via_lassolars(X, D, transform_alpha, self.positive_code)
        if algo == "lars":
            return self._sparse_code_via_lars(X, D, n_nz, self.positive_code)
        return self._sparse_code_via_omp(X, D, n_nz, self.positive_code)

    def _dict_update(
            self,
            X: torch.Tensor,
            code: torch.Tensor,
            positive: bool = False,
    ) -> torch.Tensor:
        """Update dictionary: V = (X.T @ U) with column normalization ||V_k||_2 <= 1."""
        D = X.T @ code
        if positive:
            D = D.clamp(min=0)
        norms = D.norm(dim=0, keepdim=True).clamp(min=1e-12)
        D = D / norms
        return D.T

    def fit(self, data_or_X: Union[torch.Tensor, Any], y: Any = None, **kwargs: Any) -> "DictionaryLearning":
        """Fit the dictionary from the data X."""
        X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        fit_alg = (
            self.fit_algorithm.lower()
            if isinstance(self.fit_algorithm, str)
            else self.fit_algorithm
        )
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        n_comp = self.n_components if self.n_components is not None else n_features
        n_comp = min(n_comp, n_features, n_samples)

        gen = self._get_random_generator()
        if self.dict_init is not None and self.code_init is not None:
            D = torch.as_tensor(self.dict_init, device=self.device, dtype=self.dtype)
            code = torch.as_tensor(self.code_init, device=self.device, dtype=self.dtype)
            if D.shape != (n_comp, n_features) or code.shape != (n_samples, n_comp):
                raise ValueError(
                    f"dict_init shape {D.shape} must be ({n_comp}, {n_features}), "
                    f"code_init shape {code.shape} must be ({n_samples}, {n_comp})"
                )
        elif self.dict_init is not None:
            D = torch.as_tensor(self.dict_init, device=self.device, dtype=self.dtype)
            if D.shape[0] != n_comp or D.shape[1] != n_features:
                D = D[:n_comp, :n_features].clone()
            code = self._fit_sparse_code(
                X, D, self.alpha, self.transform_n_nonzero_coefs,
                fit_alg, self.transform_max_iter
            )
        else:
            D = torch.randn(n_comp, n_features, device=self.device, dtype=self.dtype, generator=gen)
            D = D / D.norm(dim=1, keepdim=True).clamp(min=1e-12)
            code = self._fit_sparse_code(
                X, D, self.alpha, self.transform_n_nonzero_coefs,
                fit_alg, self.transform_max_iter
            )

        self.error_ = []
        for it in range(self.max_iter):
            code = self._fit_sparse_code(
                X, D, self.alpha, self.transform_n_nonzero_coefs,
                fit_alg, self.transform_max_iter
            )
            D = self._dict_update(X, code, self.positive_dict)
            err = (X - (code @ D)).norm().item()
            self.error_.append(err)
            if self.callback is not None and (it + 1) % 5 == 0:
                self.callback(self, it + 1)
            if self.verbose and (it + 1) % 10 == 0:
                print(f"DictionaryLearning iter {it + 1}/{self.max_iter}, error={err:.6f}")
            if it > 0 and abs(self.error_[-1] - self.error_[-2]) < self.tol:
                break
        self.components_ = D
        self.n_iter_ = len(self.error_)
        return self

    def transform(self, X: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Encode the data as a sparse combination of the dictionary atoms."""
        if self.components_ is None:
            raise RuntimeError("DictionaryLearning instance is not fitted yet.")
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        code = self._transform_sparse_code(X, self.components_)
        if self.split_sign:
            code = torch.cat([code.clamp(max=0).abs(), code.clamp(min=0)], dim=1)
        return code

    def fit_transform(
            self,
            data_or_X: Union[torch.Tensor, Any],
            y: Any = None,
            **kwargs: Any,
    ) -> torch.Tensor:
        """Fit the model and transform X."""
        self.fit(data_or_X, y=y, **kwargs)
        return self.transform(torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype))

    def inverse_transform(self, X: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Transform data back to original space (code @ dictionary)."""
        if self.components_ is None:
            raise RuntimeError("DictionaryLearning instance is not fitted yet.")
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        if self.split_sign:
            n = X.shape[1] // 2
            X = X[:, :n] - X[:, n:]
        return X @ self.components_


class MiniBatchDictionaryLearning(DictionaryLearning):
    def __init__(self,
                 n_components: int = None,
                 alpha: float = 1,
                 max_iter: int = 1000,
                 tol: float = 1e-08,
                 fit_algorithm: Union[Literal["lars", "cd"], Callable, nn.Module]='lars',
                 transform_algorithm: Union[Literal["lars", "lasso_lars", "lasso_cd", "omp", "threshold"], Callable, nn.Module]='omp',
                 transform_n_nonzero_coefs: int = None,
                 transform_alpha: float = None,
                 batch_size: int = 256,
                 shuffle=True,
                 n_jobs: int = None,
                 code_init: Union[list, tuple, torch.Tensor]=None,
                 dict_init: Union[list, tuple, torch.Tensor]=None,
                 callback: Union[Callable, nn.Module]=None,
                 verbose: bool = False,
                 split_sign: bool = False,
                 random_state: Union[int, torch.Generator]=None,
                 positive_code: bool = False,
                 positive_dict: bool = False,
                 transform_max_iter: int = 1000,
                 max_no_improvement=10,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args: Any,
                 **kwargs: Any):
        super().__init__(
            n_components=n_components,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            fit_algorithm=fit_algorithm,
            transform_algorithm=transform_algorithm,
            transform_n_nonzero_coefs=transform_n_nonzero_coefs,
            transform_alpha=transform_alpha,
            n_jobs=n_jobs,
            code_init=code_init,
            dict_init=dict_init,
            callback=callback,
            verbose=verbose,
            split_sign=split_sign,
            random_state=random_state,
            positive_code=positive_code,
            positive_dict=positive_dict,
            transform_max_iter=transform_max_iter,
            device=device,
            dtype=dtype,
            *args, **kwargs
        )
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_no_improvement = max_no_improvement if max_no_improvement is not None else 10
        self.n_steps_ = 0

    def fit(self, data_or_X: Union[torch.Tensor, Any], y: Any = None, **kwargs: Any) -> "MiniBatchDictionaryLearning":
        """Fit the dictionary from the data X using mini-batch updates."""
        X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        fit_alg = (
            self.fit_algorithm.lower()
            if isinstance(self.fit_algorithm, str)
            else self.fit_algorithm
        )
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        n_comp = self.n_components if self.n_components is not None else n_features
        n_comp = min(n_comp, n_features, n_samples)
        batch_size = min(self.batch_size, n_samples)

        gen = self._get_random_generator()
        if self.dict_init is not None:
            D = torch.as_tensor(self.dict_init, device=self.device, dtype=self.dtype)
            if D.shape[0] != n_comp or D.shape[1] != n_features:
                D = D[:n_comp, :n_features].clone()
            D = D / D.norm(dim=1, keepdim=True).clamp(min=1e-12)
        else:
            D = torch.randn(n_comp, n_features, device=self.device, dtype=self.dtype, generator=gen)
            D = D / D.norm(dim=1, keepdim=True).clamp(min=1e-12)

        self.error_ = []
        n_steps = 0
        n_iter = 0
        best_cost = float("inf")
        no_improvement = 0
        D_prev = D.clone()
        stop = False

        for epoch in range(self.max_iter):
            if stop:
                break
            if self.shuffle and gen is not None:
                perm = torch.randperm(n_samples, device=self.device, generator=gen)
                X_perm = X[perm]
            else:
                X_perm = X

            for batch_slice in _gen_batches(n_samples, batch_size):
                X_batch = X_perm[batch_slice]
                code = self._fit_sparse_code(
                    X_batch, D, self.alpha, self.transform_n_nonzero_coefs,
                    fit_alg, self.transform_max_iter
                )
                rho = batch_size / (n_steps * batch_size / max(1, n_samples) + batch_size)
                rho = min(rho, 1.0)
                D_batch = self._dict_update(X_batch, code, self.positive_dict)
                D = (1 - rho) * D + rho * D_batch
                D = D / D.norm(dim=1, keepdim=True).clamp(min=1e-12)
                n_steps += 1
                err = (X_batch - (code @ D)).norm().item()
                self.error_.append(err)
                if self.callback is not None and n_steps % 5 == 0:
                    self.callback(self, n_steps)
                if self.verbose and n_steps % 100 == 0:
                    print(f"MiniBatch step {n_steps}, error={err:.6f}")

                if self.tol > 0 and D_prev is not None:
                    d_diff = (D - D_prev).norm().item()
                    if d_diff < self.tol:
                        if self.verbose:
                            print(f"Converged (tol) at step {n_steps}")
                        stop = True
                        break
                D_prev = D.clone()

                cost = err
                if cost < best_cost:
                    best_cost = cost
                    no_improvement = 0
                else:
                    no_improvement += 1
                if self.max_no_improvement is not None and no_improvement >= self.max_no_improvement:
                    if self.verbose:
                        print(f"Converged (max_no_improvement) at step {n_steps}")
                    stop = True
                    break
            n_iter += 1

        self.components_ = D
        self.n_iter_ = n_iter
        self.n_steps_ = n_steps
        return self


class SparseCoder(MLTransform):
    def __init__(self,
                 dictionary: Union[list, tuple, torch.Tensor, None] = None,
                 transform_algorithm: Union[Literal["lasso_lars",
                    "ridge_lars", "elasticnet_lars", "lasso_cd",
                    "ridge_cd", "elasticnet_cd", "lasso", "ridge",
                    "elasticnet", "omp", "threshold"], Callable,
                    nn.Module] = "omp",
                 transform_n_nonzero_coefs: int = None,
                 transform_alpha: float = None,
                 split_sign: bool = False,
                 n_jobs: int = None,
                 positive_code: bool = False,
                 transform_max_iter: int = 1000,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args: Any,
                 **kwargs: Any):
        super().__init__()
        if dictionary is None:
            dev = torch.device(device) if isinstance(device, str) else device
            dictionary = torch.eye(2, 2, device=dev, dtype=dtype)
        _dict = torch.as_tensor(dictionary, dtype=dtype)
        self.dictionary = _dict
        self.transform_algorithm = transform_algorithm if isinstance(transform_algorithm, str) else transform_algorithm
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs
        self.transform_alpha = transform_alpha if transform_alpha is not None else 1.0
        self.split_sign = split_sign
        self.n_jobs = n_jobs if n_jobs is not None else 1
        self.positive_code = positive_code
        self.transform_max_iter = transform_max_iter
        self.device = device
        self.dtype = dtype
        self.kwargs = dict(kwargs)
        self.kwargs["_args"] = args

        self.components_ = _dict.to(device=device, dtype=dtype)
        self.n_components_ = self.components_.shape[0]
        self.n_features_in_ = self.components_.shape[1]

    def fit(
        self, data_or_X: Union[torch.Tensor, Any], y: Any = None, **kwargs: Any
    ) -> "SparseCoder":
        """SparseCoder uses a fixed dictionary; fit just records data shape."""
        X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        self.n_features_in_ = X.shape[1]
        return self

    def _sc_omp(self, X: torch.Tensor, D: torch.Tensor, n_nonzero: int) -> torch.Tensor:
        """Orthogonal Matching Pursuit sparse coding."""
        n_samples, _ = X.shape
        n_atoms = D.shape[0]
        code = torch.zeros(n_samples, n_atoms, device=X.device, dtype=X.dtype)
        for i in range(n_samples):
            x = X[i]
            r = x.clone()
            active = torch.zeros(n_atoms, dtype=torch.bool, device=X.device)
            for _ in range(min(n_nonzero, n_atoms)):
                proj = D @ r
                if self.positive_code:
                    proj = proj.clamp(min=0)
                proj_masked = proj.clone()
                proj_masked[active] = -float("inf")
                j = proj_masked.abs().argmax().item()
                active[j] = True
                D_active = D[active]
                c_active = torch.linalg.lstsq(D_active.T, x.unsqueeze(1), rcond=None).solution.squeeze(1)
                if self.positive_code:
                    c_active = c_active.clamp(min=0)
                code[i, active] = c_active
                r = x - D_active.T @ c_active
                if r.norm() < 1e-12:
                    break
        return code

    def _sc_threshold(self, X: torch.Tensor, D: torch.Tensor, alpha: float) -> torch.Tensor:
        """Threshold sparse coding."""
        code = X @ D.T
        if self.positive_code:
            code = code.clamp(min=0)
        return torch.sign(code) * (code.abs() - alpha).clamp(min=0)

    def _sc_cd(self, X: torch.Tensor, D: torch.Tensor, alpha: float) -> torch.Tensor:
        """Coordinate descent Lasso sparse coding."""
        n_samples, _ = X.shape
        n_atoms = D.shape[0]
        DtD = D @ D.T
        DtX = D @ X.T
        eps = 1e-12
        code = torch.zeros(n_samples, n_atoms, device=X.device, dtype=X.dtype)
        for _ in range(self.transform_max_iter):
            for j in range(n_atoms):
                residual = DtX[j] - (code @ DtD[:, j]) + code[:, j] * DtD[j, j]
                d_jj = DtD[j, j].clamp(min=eps)
                c_new = residual / d_jj
                if self.positive_code:
                    c_new = c_new.clamp(min=0)
                else:
                    thresh = alpha / d_jj
                    c_new = torch.sign(c_new) * (c_new.abs() - thresh).clamp(min=0)
                code[:, j] = c_new
        return code

    def _sc_lasso_lars(self, X: torch.Tensor, D: torch.Tensor, alpha: float) -> torch.Tensor:
        """LassoLars sparse coding."""
        n_samples, _ = X.shape
        n_atoms = D.shape[0]
        code = torch.zeros(n_samples, n_atoms, device=X.device, dtype=X.dtype)
        Dt = D.T
        lassolars = LassoLars(
            alpha=alpha,
            fit_intercept=False,
            positive=self.positive_code,
            device=str(X.device),
            dtype=X.dtype,
        )
        for i in range(n_samples):
            lassolars.fit(Dt, X[i])
            coef = lassolars.weight.squeeze() if lassolars.weight.ndim > 1 else lassolars.weight
            if coef.ndim == 0:
                coef = coef.unsqueeze(0)
            code[i] = coef[:n_atoms]
        return code

    def _sc_lars(self, X: torch.Tensor, D: torch.Tensor, n_nonzero: int) -> torch.Tensor:
        """Lars sparse coding."""
        n_samples, n_features = X.shape
        n_atoms = D.shape[0]
        code = torch.zeros(n_samples, n_atoms, device=X.device, dtype=X.dtype)
        Dt = D.T
        lars = Lars(
            fit_intercept=False,
            n_nonzero_coefs=min(n_nonzero, n_atoms),
            positive=self.positive_code,
            device=str(X.device),
            dtype=X.dtype,
        )
        for i in range(n_samples):
            lars.fit(Dt, X[i])
            coef = lars.weight.squeeze() if lars.weight.ndim > 1 else lars.weight
            if coef.ndim == 0:
                coef = coef.unsqueeze(0)
            code[i] = coef[:n_atoms]
        return code

    def transform(self, X: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Encode X as a sparse combination of dictionary atoms."""
        if self.components_ is None:
            raise RuntimeError("SparseCoder is not initialised with a dictionary.")
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        D = self.components_
        n_features = D.shape[1]
        alpha = self.transform_alpha if self.transform_alpha is not None else 1.0
        n_nz = self.transform_n_nonzero_coefs if self.transform_n_nonzero_coefs is not None else max(1, n_features // 10)

        algo = self.transform_algorithm
        if callable(algo) or isinstance(algo, nn.Module):
            code = algo(X, D, alpha=alpha, n_nonzero=n_nz, device=self.device, dtype=self.dtype)
        elif isinstance(algo, str):
            a = algo.lower()
            if a == "omp":
                code = self._sc_omp(X, D, n_nz)
            elif a == "threshold":
                code = self._sc_threshold(X, D, alpha)
            elif a in ("lasso_cd", "ridge_cd", "elasticnet_cd"):
                code = self._sc_cd(X, D, alpha)
            elif a in ("lasso_lars", "ridge_lars", "elasticnet_lars", "lasso"):
                code = self._sc_lasso_lars(X, D, alpha)
            elif a == "lars":
                code = self._sc_lars(X, D, n_nz)
            else:
                code = self._sc_omp(X, D, n_nz)
        else:
            code = self._sc_omp(X, D, n_nz)

        if self.split_sign:
            code = torch.cat([code.clamp(max=0).abs(), code.clamp(min=0)], dim=1)
        return code

    def fit_transform(
        self, data_or_X: Union[torch.Tensor, Any], y: Any = None, **kwargs: Any
    ) -> torch.Tensor:
        """Fit and encode X."""
        self.fit(data_or_X, y=y)
        return self.transform(data_or_X)


class NeighborhoodComponentsAnalysis(MLTransform):
    def __init__(self,
                 n_components: int = None,
                 init: Union[Literal["auto", "pca", "lda", "identity",
                    "random"], list, tuple, torch.Tensor] = 'auto',
                 warm_start: bool = False,
                 max_iter: int = 50,
                 tol: float = 1e-5,
                 callback: Union[Callable, nn.Module] = None,
                 verbose: int = 0,
                 random_state: Union[int, torch.Generator] = None,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args: Any,
                 **kwargs: Any):
        super().__init__()
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
        self.kwargs = dict(kwargs)
        self.kwargs["_args"] = args

        self.components_ = None
        self.n_features_in_ = None
        self.n_iter_ = 0
        self.random_state_ = random_state

    def _get_generator(self) -> Optional[torch.Generator]:
        if self.random_state is None:
            return None
        if isinstance(self.random_state, torch.Generator):
            return self.random_state
        g = torch.Generator(device=self.device)
        g.manual_seed(int(self.random_state))
        return g

    def _get_init_A(
        self, X: torch.Tensor, y: torch.Tensor, n_comp: int, n_features: int
    ) -> torch.Tensor:
        """Compute initial transformation matrix A of shape (n_comp, n_features)."""
        init = self.init
        gen = self._get_generator()

        if isinstance(init, (list, tuple, torch.Tensor)):
            return torch.as_tensor(init, device=self.device, dtype=self.dtype)

        if isinstance(init, str):
            if init == "identity":
                A = torch.eye(n_features, device=self.device, dtype=self.dtype)
                return A[:n_comp]

            if init == "random":
                if gen is not None:
                    return torch.randn(n_comp, n_features, device=self.device, dtype=self.dtype, generator=gen)
                return torch.randn(n_comp, n_features, device=self.device, dtype=self.dtype)

            if init == "pca":
                try:
                    from .....models.machine_learning.transformer.pca import PCA as _PCA
                    pca = _PCA(n_components=n_comp, device=self.device, dtype=self.dtype)
                    pca.fit(X)
                    return pca.components_
                except Exception:
                    pass
                A = torch.eye(n_features, device=self.device, dtype=self.dtype)
                return A[:n_comp]

            if init == "lda":
                classes = torch.unique(y)
                n_classes = len(classes)
                n_comp_lda = min(n_comp, n_classes - 1)
                if n_comp_lda <= 0:
                    A = torch.eye(n_features, device=self.device, dtype=self.dtype)
                    return A[:n_comp]
                try:
                    overall_mean = X.mean(0)
                    S_w = torch.zeros(n_features, n_features, device=self.device, dtype=self.dtype)
                    S_b = torch.zeros(n_features, n_features, device=self.device, dtype=self.dtype)
                    for c in classes:
                        Xc = X[y == c]
                        mc = Xc.mean(0)
                        diff = Xc - mc
                        S_w = S_w + diff.T @ diff
                        diff_b = (mc - overall_mean).unsqueeze(1)
                        S_b = S_b + Xc.shape[0] * (diff_b @ diff_b.T)
                    reg = 1e-6 * torch.eye(n_features, device=self.device, dtype=self.dtype)
                    S_w_inv = torch.linalg.inv(S_w + reg)
                    M = S_w_inv @ S_b
                    eigvals, eigvecs = torch.linalg.eigh(M)
                    idx = torch.argsort(eigvals, descending=True)
                    A = eigvecs[:, idx[:n_comp_lda]].T
                    if n_comp_lda < n_comp:
                        if gen is not None:
                            pad = torch.randn(n_comp - n_comp_lda, n_features, device=self.device, dtype=self.dtype, generator=gen)
                        else:
                            pad = torch.randn(n_comp - n_comp_lda, n_features, device=self.device, dtype=self.dtype)
                        A = torch.cat([A, pad], dim=0)
                    return A
                except Exception:
                    pass
                A = torch.eye(n_features, device=self.device, dtype=self.dtype)
                return A[:n_comp]

            if init == "auto":
                classes = torch.unique(y)
                n_classes = len(classes)
                if n_comp <= min(n_features, n_classes - 1):
                    return self._get_init_A(X, y, n_comp, n_features)
                try:
                    from .....models.machine_learning.transformer.pca import PCA as _PCA
                    pca = _PCA(n_components=n_comp, device=self.device, dtype=self.dtype)
                    pca.fit(X)
                    return pca.components_
                except Exception:
                    pass
                A = torch.eye(n_features, device=self.device, dtype=self.dtype)
                return A[:n_comp]

        if gen is not None:
            return torch.randn(n_comp, n_features, device=self.device, dtype=self.dtype, generator=gen)
        return torch.randn(n_comp, n_features, device=self.device, dtype=self.dtype)

    @staticmethod
    def _nca_loss(A: torch.Tensor, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """NCA objective: sum_i sum_{j: y_j==y_i, j!=i} p_ij  (to maximise)."""
        n = X.shape[0]
        XA = X @ A.T
        sq_dists = torch.cdist(XA, XA, p=2).pow(2)
        mask_diag = torch.eye(n, dtype=torch.bool, device=X.device)
        sq_dists_inf = sq_dists.masked_fill(mask_diag, float("inf"))
        log_p = -sq_dists_inf - torch.logsumexp(-sq_dists_inf, dim=1, keepdim=True)
        p = torch.exp(log_p)
        p = p.masked_fill(mask_diag, 0.0)
        same = (y.unsqueeze(0) == y.unsqueeze(1)).float().masked_fill(mask_diag, 0.0)
        return (p * same).sum()

    def fit(
        self,
        data_or_X: Union[torch.Tensor, Any],
        y: Union[torch.Tensor, Any, None] = None,
        **kwargs: Any,
    ) -> "NeighborhoodComponentsAnalysis":
        """Fit NCA model by optimising the stochastic k-NN objective."""
        if y is None:
            raise ValueError(
                "NeighborhoodComponentsAnalysis requires y (class labels) for fit. "
                "NCA is a supervised dimensionality reduction method."
            )
        X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        y_t = torch.as_tensor(y, device=self.device)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        n_comp = n_features if self.n_components is None else min(int(self.n_components), n_features)

        if self.warm_start and self.components_ is not None:
            A_init = self.components_.clone()
        else:
            A_init = self._get_init_A(X, y_t, n_comp, n_features)

        A = nn.Parameter(A_init.clone())
        optimizer = torch.optim.Adam([A], lr=1e-2)

        best_loss = -float("inf")
        n_iter_no_change = 0
        for i in range(self.max_iter):
            optimizer.zero_grad()
            loss = self._nca_loss(A, X, y_t)
            (-loss).backward()
            optimizer.step()

            loss_val = loss.item()
            grad_norm = A.grad.norm().item() if A.grad is not None else 0.0

            if self.callback is not None:
                self.callback(A.detach().clone().ravel(), i + 1)

            if self.verbose:
                print(f"NCA iter {i + 1}/{self.max_iter}: loss={loss_val:.6f}, |grad|={grad_norm:.2e}")

            if grad_norm < self.tol:
                self.n_iter_ = i + 1
                break
            self.n_iter_ = i + 1

        self.components_ = A.detach().clone()
        return self

    def transform(self, X: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Project X into the learned metric space."""
        if self.components_ is None:
            raise RuntimeError("NeighborhoodComponentsAnalysis instance is not fitted yet.")
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        return X @ self.components_.T

    def fit_transform(
        self,
        data_or_X: Union[torch.Tensor, Any],
        y: Union[torch.Tensor, Any],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Fit NCA and return the transformed X."""
        self.fit(data_or_X, y, **kwargs)
        return self.transform(data_or_X)


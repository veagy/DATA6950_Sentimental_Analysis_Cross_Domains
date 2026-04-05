import copy
import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Callable, Union, Any, List, Tuple, Dict, Literal
from .....models.utils import MLModule
from .....models.machine_learning.cross_validation.search_cv import GridSearchCV
from torch.func import vmap
import joblib


__all__ = [
    "EmpiricalCovariance",
    "EllipticEnvelope",
    "LedoitWolf",
    "MinCovDet",
    "OAS",
    "ShrunkCovariance",
    "GraphicalLasso",
    "GraphicalLassoCV",
    "GraphicalRidge",
    "GraphicalRidgeCV",
    "GraphicalElasticNet",
    "GraphicalElasticNetCV",
]

_FLOAT64_EPS = 2.220446049250313e-16


# ─────────────────────────────────────────────────────────────────────────────
# Private utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _eps_float(eps) -> float:
    """Convert eps (possibly a 0-d tensor) to a Python float."""
    if isinstance(eps, torch.Tensor):
        return eps.item()
    return float(eps)


def _to_tensor(X, dtype, device) -> torch.Tensor:
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=dtype, device=device)
    return X.to(dtype=dtype, device=device)


def _soft_threshold(x: torch.Tensor, threshold) -> torch.Tensor:
    """Element-wise soft-thresholding: sign(x) * max(|x| − t, 0)."""
    if not isinstance(threshold, torch.Tensor):
        threshold = torch.tensor(threshold, dtype=x.dtype, device=x.device)
    return x.sign() * (x.abs() - threshold).clamp(min=0.0)


def _pinv_safe(A: torch.Tensor, eps: float = _FLOAT64_EPS) -> torch.Tensor:
    """Regularised matrix inverse; falls back to pseudo-inverse on failure."""
    p = A.shape[0]
    reg = torch.eye(p, dtype=A.dtype, device=A.device)
    try:
        return torch.linalg.inv(A + eps * reg)
    except Exception:
        return torch.linalg.pinv(A)


def _empirical_cov(X: torch.Tensor,
                   assume_centered: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (location, empirical_covariance) for the n×p matrix X."""
    n = X.shape[0]
    loc = (torch.zeros(X.shape[1], dtype=X.dtype, device=X.device)
           if assume_centered else X.mean(0))
    Xc = X - loc
    cov = (Xc.T @ Xc) / n
    return loc, cov


def _log_likelihood_score(X: torch.Tensor,
                           loc: torch.Tensor,
                           prec: torch.Tensor) -> float:
    """Mean Gaussian log-likelihood of the rows of X under N(loc, prec⁻¹)."""
    p = prec.shape[0]
    Xc = X - loc
    sign, logdet = torch.linalg.slogdet(prec)
    if sign.item() <= 0:
        return float('-inf')
    maha = (Xc @ prec * Xc).sum(dim=1)
    return (0.5 * (logdet - p * math.log(2.0 * math.pi) - maha)).mean().item()


def _mahalanobis(X: torch.Tensor,
                 loc: torch.Tensor,
                 prec: torch.Tensor) -> torch.Tensor:
    """Squared Mahalanobis distances of every row of X from loc."""
    Xc = X - loc
    return (Xc @ prec * Xc).sum(dim=1)


def _shrink_cov(emp_cov: torch.Tensor, shrinkage: float) -> torch.Tensor:
    """Apply convex shrinkage towards a scaled identity target."""
    p = emp_cov.shape[0]
    mu = torch.trace(emp_cov).item() / p
    eye = torch.eye(p, dtype=emp_cov.dtype, device=emp_cov.device)
    return (1.0 - shrinkage) * emp_cov + shrinkage * mu * eye


# ── Chi-2 / normal utilities (needed by MCD and EllipticEnvelope) ────────────

def _normal_ppf(p_val: float) -> float:
    """Rational approximation of the standard-normal PPF (BSM algorithm)."""
    p_val = max(min(p_val, 1.0 - 1e-15), 1e-15)
    a = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637]
    b = [-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833]
    c = [0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
         0.0276438810333863, 0.0038405729373609, 0.0003951896511349,
         0.0000321767881768, 0.0000002888167364, 0.0000003960315187]
    y = p_val - 0.5
    if abs(y) < 0.42:
        r = y * y
        x = y * (((a[3] * r + a[2]) * r + a[1]) * r + a[0]) / (
            (((b[3] * r + b[2]) * r + b[1]) * r + b[0]) * r + 1.0)
    else:
        r = math.log(-math.log(p_val if y < 0 else 1.0 - p_val))
        x = c[0] + r * (c[1] + r * (c[2] + r * (c[3] + r * (c[4] + r * (
            c[5] + r * (c[6] + r * (c[7] + r * c[8])))))))
        if y < 0:
            x = -x
    return x


def _chi2_cdf(x: float, df: float) -> float:
    """Chi-squared CDF via the regularised lower incomplete gamma function."""
    if x <= 0.0:
        return 0.0
    return float(torch.special.gammainc(
        torch.tensor(df / 2.0, dtype=torch.float64),
        torch.tensor(x / 2.0, dtype=torch.float64)))


def _chi2_ppf(p_val: float, df: float) -> float:
    """Chi-squared PPF via Newton-Raphson with Wilson–Hilferty initialisation."""
    h = 2.0 / (9.0 * df)
    z = _normal_ppf(p_val)
    x = max(df * (1.0 - h + z * math.sqrt(h)) ** 3, 1e-10)
    for _ in range(30):
        cdf_val = _chi2_cdf(x, df)
        log_pdf = ((df / 2.0 - 1.0) * math.log(max(x, 1e-300))
                   - x / 2.0 - math.lgamma(df / 2.0) - (df / 2.0) * math.log(2.0))
        pdf_val = math.exp(max(log_pdf, -700))
        if pdf_val < 1e-15:
            break
        x -= (cdf_val - p_val) / (pdf_val + 1e-300)
        x = max(x, 1e-10)
    return x


def _mcd_consistency(h: int, n: int, p: int) -> float:
    """Consistency correction factor c(h, n, p) for the MCD estimator."""
    alpha = (n - h) / n
    if alpha < 1e-10:
        return 1.0
    try:
        q = _chi2_ppf(1.0 - alpha, float(p))
        denom = (1.0 - alpha) / _chi2_cdf(q, float(p + 2))
        return 1.0 / denom if abs(denom) > 1e-10 else 1.0
    except Exception:
        return 1.0


# ── FastMCD helpers ───────────────────────────────────────────────────────────

def _c_step(X: torch.Tensor, h: int, loc: torch.Tensor,
            cov: torch.Tensor, reg: float = 1e-6
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One C-step: recompute location/covariance from the h best-fitting rows."""
    p = X.shape[1]
    prec = _pinv_safe(cov + reg * torch.eye(p, dtype=cov.dtype, device=cov.device))
    Xc = X - loc
    dist = (Xc @ prec * Xc).sum(dim=1)
    idx = torch.argsort(dist)[:h]
    sub = X[idx]
    loc_new = sub.mean(0)
    Xsub = sub - loc_new
    cov_new = (Xsub.T @ Xsub) / float(h)
    return loc_new, cov_new, idx


def _fast_mcd(X: torch.Tensor, h: int,
              random_state: Optional[Union[int, torch.Generator]] = None,
              n_trials: int = 10) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simplified FastMCD: find the h-subset with minimum covariance determinant."""
    n, p = X.shape
    device, dtype = X.device, X.dtype

    gen = torch.Generator(device=device)
    if isinstance(random_state, int):
        gen.manual_seed(random_state)
    elif isinstance(random_state, torch.Generator):
        gen = random_state

    loc0 = X.mean(0)
    Xc0 = X - loc0
    cov0 = (Xc0.T @ Xc0) / n + 1e-6 * torch.eye(p, dtype=dtype, device=device)
    best_loc, best_cov = loc0, cov0
    best_idx = torch.arange(n, device=device)
    best_det = float('inf')

    for _ in range(n_trials):
        perm = torch.randperm(n, generator=gen, device=device)
        init_n = min(p + 1, n)
        sub0 = X[perm[:init_n]]
        loc = sub0.mean(0)
        Xs = sub0 - loc
        cov = (Xs.T @ Xs) / float(init_n) + 1e-6 * torch.eye(p, dtype=dtype, device=device)

        loc_new, cov_new, idx = loc, cov, perm[:init_n]
        for _ in range(50):
            loc_prev = loc_new
            loc_new, cov_new, idx = _c_step(X, h, loc_new, cov_new)
            if (loc_new - loc_prev).abs().max().item() < 1e-8:
                break

        try:
            _, logdet = torch.linalg.slogdet(cov_new + 1e-10 * torch.eye(p, dtype=dtype, device=device))
            det_val = logdet.item()
        except Exception:
            det_val = float('inf')

        if det_val < best_det:
            best_det = det_val
            best_loc, best_cov, best_idx = loc_new, cov_new, idx

    for _ in range(50):
        loc_prev = best_loc
        best_loc, best_cov, best_idx = _c_step(X, h, best_loc, best_cov)
        if (best_loc - loc_prev).abs().max().item() < 1e-10:
            break

    return best_loc, best_cov, best_idx


# ── Coordinate-descent inner solvers for graphical models ────────────────────

def _inner_lasso_cd(W11: torch.Tensor, s12: torch.Tensor, alpha: float,
                    tol: float = 1e-4, max_iter: int = 500) -> torch.Tensor:
    """
    Coordinate descent for the Lasso subproblem inside GraphicalLasso:
        min_β  ½ β' W₁₁ β − s₁₂' β + α ‖β‖₁
    """
    p = W11.shape[0]
    beta = torch.zeros(p, dtype=W11.dtype, device=W11.device)
    for _ in range(max_iter):
        beta_prev = beta.clone()
        for k in range(p):
            rho = s12[k] - torch.dot(W11[k], beta) + W11[k, k] * beta[k]
            dkk = W11[k, k].item()
            if abs(dkk) > 1e-15:
                beta[k] = _soft_threshold(rho.unsqueeze(0), alpha).squeeze() / dkk
            else:
                beta[k] = beta[k].zero_()
        if (beta - beta_prev).abs().max().item() < tol:
            break
    return beta


def _inner_enet_cd(W11: torch.Tensor, s12: torch.Tensor,
                   alpha: float, l1_ratio: float,
                   tol: float = 1e-4, max_iter: int = 500) -> torch.Tensor:
    """
    Coordinate descent for the Elastic-Net subproblem inside GraphicalElasticNet:
        min_β  ½ β' W₁₁ β − s₁₂' β + α·l1·‖β‖₁ + α·(1−l1)/2·‖β‖₂²
    """
    p = W11.shape[0]
    beta = torch.zeros(p, dtype=W11.dtype, device=W11.device)
    a1 = float(alpha) * float(l1_ratio)
    a2 = float(alpha) * (1.0 - float(l1_ratio))
    for _ in range(max_iter):
        beta_prev = beta.clone()
        for k in range(p):
            rho = s12[k] - torch.dot(W11[k], beta) + W11[k, k] * beta[k]
            dkk = W11[k, k].item() + a2
            if abs(dkk) > 1e-15:
                beta[k] = _soft_threshold(rho.unsqueeze(0), a1).squeeze() / dkk
            else:
                beta[k] = beta[k].zero_()
        if (beta - beta_prev).abs().max().item() < tol:
            break
    return beta


# ── Core graphical-model fitting routines ────────────────────────────────────

def _graphical_lasso_fit(
        S: torch.Tensor,
        alpha: float,
        mode: str = 'cd',
        tol: float = 1e-4,
        enet_tol: float = 1e-4,
        max_iter: int = 100,
        verbose: bool = False,
        eps: float = _FLOAT64_EPS,
) -> Tuple[torch.Tensor, torch.Tensor, int, List[Tuple[float, float]]]:
    """
    Block coordinate descent for the Graphical Lasso objective:
        min_Θ { −log det Θ + Tr(S Θ) + α ‖Θ‖_{1,off} }

    Returns
    -------
    covariance_hat, precision_hat, n_iter, costs
    """
    alpha = float(alpha)
    tol = float(tol)
    enet_tol = float(enet_tol)
    eps = _eps_float(eps)

    p = S.shape[0]
    dtype, device = S.dtype, S.device
    eye = torch.eye(p, dtype=dtype, device=device)

    W = S.clone() + alpha * eye
    W_prev = W.clone()
    costs: List[Tuple[float, float]] = []

    inner_iters = max_iter * 10 if mode == 'lars' else max_iter

    it = 0
    for it in range(max_iter):
        for j in range(p):
            mask = torch.ones(p, dtype=torch.bool, device=device)
            mask[j] = False
            W11 = W[mask][:, mask]
            s12 = S[mask, j]
            beta = _inner_lasso_cd(W11, s12, alpha, enet_tol, inner_iters)
            w12 = W11 @ beta
            W[mask, j] = w12
            W[j, mask] = w12

        Theta = _pinv_safe(W, eps)
        sign, logdet = torch.linalg.slogdet(Theta)
        if sign.item() > 0:
            off = Theta - torch.diag(torch.diag(Theta))
            obj = (-logdet + torch.trace(S @ Theta) + alpha * off.abs().sum()).item()
        else:
            obj = float('inf')

        dual_gap = (W - W_prev).abs().max().item()
        costs.append((obj, dual_gap))

        if verbose:
            print(f"  [GraphicalLasso] iter={it + 1:3d}  obj={obj:.6f}  gap={dual_gap:.2e}")

        if dual_gap < tol:
            break
        W_prev = W.clone()

    return W.clone(), _pinv_safe(W, eps), it + 1, costs


def _graphical_enet_fit(
        S: torch.Tensor,
        alpha: float,
        l1_ratio: float = 0.5,
        mode: str = 'cd',
        tol: float = 1e-4,
        enet_tol: float = 1e-4,
        max_iter: int = 100,
        verbose: bool = False,
        eps: float = _FLOAT64_EPS,
) -> Tuple[torch.Tensor, torch.Tensor, int, List[Tuple[float, float]]]:
    """
    Block coordinate descent for the Graphical Elastic-Net objective:
        min_Θ { −log det Θ + Tr(S Θ) + α·l1·‖Θ‖_{1,off} + α·(1−l1)/2·‖Θ‖_F² }

    Returns
    -------
    covariance_hat, precision_hat, n_iter, costs
    """
    alpha = float(alpha)
    l1_ratio = float(l1_ratio)
    tol = float(tol)
    enet_tol = float(enet_tol)
    eps = _eps_float(eps)
    a1 = alpha * l1_ratio
    a2 = alpha * (1.0 - l1_ratio)

    p = S.shape[0]
    dtype, device = S.dtype, S.device
    eye = torch.eye(p, dtype=dtype, device=device)

    W = S.clone() + (a1 + a2) * eye
    W_prev = W.clone()
    costs: List[Tuple[float, float]] = []

    it = 0
    for it in range(max_iter):
        for j in range(p):
            mask = torch.ones(p, dtype=torch.bool, device=device)
            mask[j] = False
            W11 = W[mask][:, mask]
            s12 = S[mask, j]
            beta = _inner_enet_cd(W11, s12, alpha, l1_ratio, enet_tol, max_iter)
            w12 = W11 @ beta
            W[mask, j] = w12
            W[j, mask] = w12

        Theta = _pinv_safe(W, eps)
        sign, logdet = torch.linalg.slogdet(Theta)
        if sign.item() > 0:
            off = Theta - torch.diag(torch.diag(Theta))
            obj = (-logdet + torch.trace(S @ Theta)
                   + a1 * off.abs().sum() + 0.5 * a2 * Theta.pow(2).sum()).item()
        else:
            obj = float('inf')

        dual_gap = (W - W_prev).abs().max().item()
        costs.append((obj, dual_gap))

        if verbose:
            print(f"  [GraphicalElasticNet] iter={it + 1:3d}  obj={obj:.6f}  gap={dual_gap:.2e}")

        if dual_gap < tol:
            break
        W_prev = W.clone()

    return W.clone(), _pinv_safe(W, eps), it + 1, costs


# ── Cross-validation helpers ─────────────────────────────────────────────────

def _data_driven_alpha_grid(S: torch.Tensor, n_alphas: int) -> List[float]:
    """Log-spaced alpha grid derived from the off-diagonal of S."""
    off = S.clone()
    off.fill_diagonal_(0.0)
    alpha_max = max(off.abs().max().item(), 1e-6)
    alpha_min = alpha_max * 1e-2
    return torch.logspace(math.log10(alpha_min), math.log10(alpha_max), n_alphas).tolist()


def _resolve_alphas(alphas, S: torch.Tensor) -> List[float]:
    if isinstance(alphas, int):
        return _data_driven_alpha_grid(S, alphas)
    if isinstance(alphas, torch.Tensor):
        return alphas.flatten().tolist()
    if isinstance(alphas, (list, tuple)):
        return [float(a) for a in alphas]
    return [float(alphas)]


def _refine_grid(alpha_grid: List[float], best_alpha: float, n_new: int) -> List[float]:
    """Return a finer log-spaced grid centred on best_alpha."""
    srt = sorted(alpha_grid)
    if len(srt) < 2:
        return srt
    pos = min(range(len(srt)), key=lambda i: abs(srt[i] - best_alpha))
    left = srt[max(0, pos - 1)]
    right = srt[min(len(srt) - 1, pos + 1)]
    if right <= left:
        return srt
    return torch.logspace(math.log10(left), math.log10(right), max(3, n_new)).tolist()


def _run_grid_cv(base_est: MLModule,
                 param_grid: Dict[str, List],
                 X: torch.Tensor,
                 cv, cv_config, verbose, device, dtype):
    """Run GridSearchCV and return (best_params, cv_results_)."""
    search = GridSearchCV(
        estimator=base_est,
        param_grid=param_grid,
        scoring=None,
        cv=cv,
        cv_config=cv_config,
        verbose=verbose,
        device=device,
        dtype=dtype,
    )
    search.fit(X)
    return search.best_params_, search.cv_results_


def _merge_cv_results(base: Dict, extra: Dict) -> Dict:
    merged = dict(base)
    for k, v in extra.items():
        if k in merged and isinstance(merged[k], list):
            merged[k] = merged[k] + v
        else:
            merged[k] = v
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Covariance estimator classes
# ─────────────────────────────────────────────────────────────────────────────

class EmpiricalCovariance(MLModule):
    def __init__(self,
                 store_precision: bool = True,
                 assume_centered: bool = False,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.store_precision = store_precision
        self.assume_centered = assume_centered
        self.device = device
        self.dtype = dtype

        self.location_: Optional[torch.Tensor] = None
        self.covariance_: Optional[torch.Tensor] = None
        self.precision_: Optional[torch.Tensor] = None
        self.n_features_in_: Optional[int] = None

    def _t(self, X) -> torch.Tensor:
        return _to_tensor(X, self.dtype, self.device)

    def fit(self, X: torch.Tensor, y=None) -> 'EmpiricalCovariance':
        X = self._t(X)
        n, p = X.shape
        self.n_features_in_ = p
        self.location_, self.covariance_ = _empirical_cov(X, self.assume_centered)
        if self.store_precision:
            self.precision_ = _pinv_safe(self.covariance_)
        else:
            self.precision_ = None
        return self

    def get_precision(self) -> torch.Tensor:
        """Return the precision matrix (inverse of covariance)."""
        if self.precision_ is not None:
            return self.precision_
        return _pinv_safe(self.covariance_)

    def score(self, X_test: torch.Tensor, y=None) -> float:
        """Compute the mean log-likelihood of X_test under the estimated model."""
        X_test = self._t(X_test)
        return _log_likelihood_score(X_test, self.location_, self.get_precision())

    def error_norm(self,
                   comp_cov: torch.Tensor,
                   norm: str = 'frobenius',
                   scaling: bool = True,
                   squared: bool = True) -> float:
        comp_cov = self._t(comp_cov)
        diff = self.covariance_ - comp_cov
        if norm == 'frobenius':
            err = diff.pow(2).sum()
            if scaling:
                err = err / diff.numel()
        elif norm == 'spectral':
            err = torch.linalg.norm(diff, ord=2).pow(2)
            if scaling:
                err = err / diff.shape[0]
        else:
            err = diff.abs().sum()
            if scaling:
                err = err / diff.numel()
        if not squared:
            err = err.sqrt()
        return err.item()

    def mahalanobis(self, X: torch.Tensor) -> torch.Tensor:
        """Compute the squared Mahalanobis distances of observations in X."""
        X = self._t(X)
        return _mahalanobis(X, self.location_, self.get_precision())

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.mahalanobis(X)


class ShrunkCovariance(MLModule):
    def __init__(self,
                 store_precision: bool = True,
                 assume_centered: bool = False,
                 shrinkage: float = 0.1,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.store_precision = store_precision
        self.assume_centered = assume_centered
        self.shrinkage = shrinkage
        self.device = device
        self.dtype = dtype

        self.covariance_: Optional[torch.Tensor] = None
        self.location_: Optional[torch.Tensor] = None
        self.precision_: Optional[torch.Tensor] = None
        self.n_features_in_: Optional[int] = None

    def _t(self, X) -> torch.Tensor:
        return _to_tensor(X, self.dtype, self.device)

    def fit(self, X: torch.Tensor, y=None) -> 'ShrunkCovariance':
        X = self._t(X)
        n, p = X.shape
        self.n_features_in_ = p
        self.location_, emp_cov = _empirical_cov(X, self.assume_centered)
        self.covariance_ = _shrink_cov(emp_cov, float(self.shrinkage))
        if self.store_precision:
            self.precision_ = _pinv_safe(self.covariance_)
        else:
            self.precision_ = None
        return self

    def get_precision(self) -> torch.Tensor:
        if self.precision_ is not None:
            return self.precision_
        return _pinv_safe(self.covariance_)

    def score(self, X_test: torch.Tensor, y=None) -> float:
        """Mean log-likelihood of X_test under the shrunk model."""
        X_test = self._t(X_test)
        return _log_likelihood_score(X_test, self.location_, self.get_precision())

    def error_norm(self,
                   comp_cov: torch.Tensor,
                   norm: str = 'frobenius',
                   scaling: bool = True,
                   squared: bool = True) -> float:
        """Frobenius / spectral error norm between self.covariance_ and comp_cov."""
        comp_cov = self._t(comp_cov)
        diff = self.covariance_ - comp_cov
        if norm == 'frobenius':
            err = diff.pow(2).sum()
            if scaling:
                err = err / diff.numel()
        elif norm == 'spectral':
            err = torch.linalg.norm(diff, ord=2).pow(2)
            if scaling:
                err = err / diff.shape[0]
        else:
            err = diff.abs().sum()
            if scaling:
                err = err / diff.numel()
        if not squared:
            err = err.sqrt()
        return err.item()

    def mahalanobis(self, X: torch.Tensor) -> torch.Tensor:
        """Squared Mahalanobis distances under the shrunk precision."""
        X = self._t(X)
        return _mahalanobis(X, self.location_, self.get_precision())

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.mahalanobis(X)


class LedoitWolf(MLModule):
    def __init__(self,
                 store_precision: bool = True,
                 assume_centered: bool = False,
                 block_size: int = 1000,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.store_precision = store_precision
        self.assume_centered = assume_centered
        self.block_size = block_size
        self.device = device
        self.dtype = dtype

        self.covariance_: Optional[torch.Tensor] = None
        self.location_: Optional[torch.Tensor] = None
        self.precision_: Optional[torch.Tensor] = None
        self.shrinkage_: Optional[float] = None
        self.n_features_in_: Optional[int] = None

    def _t(self, X) -> torch.Tensor:
        return _to_tensor(X, self.dtype, self.device)

    def _lw_shrinkage(self, Xc: torch.Tensor, S: torch.Tensor) -> float:
        """
        Ledoit-Wolf analytical shrinkage coefficient.

        φ̂ = (∑ₜ ‖xₜ‖⁴/n − tr(S²)) / n
        δ  = tr(S²) − tr(S)²/p
        ρ* = clamp(φ̂ / δ, 0, 1)
        """
        n, p = Xc.shape
        tr_S2 = torch.trace(S @ S).item()
        tr_S = torch.trace(S).item()

        # φ̂: use block computation to save memory (block_size controls chunk width)
        block_sz = min(self.block_size, p)
        norm4_sum = 0.0
        for j in range(0, p, block_sz):
            chunk = Xc[:, j: j + block_sz]  # (n, b)
            norm4_sum += (chunk * chunk).sum(dim=1).pow(2).sum().item() / n

        phi_hat = (norm4_sum / n) - tr_S2 / n

        delta = tr_S2 - tr_S ** 2 / p
        if abs(delta) < 1e-12:
            return 0.0
        rho = phi_hat / delta
        return float(max(0.0, min(1.0, rho)))

    def fit(self, X: torch.Tensor, y=None) -> 'LedoitWolf':
        X = self._t(X)
        n, p = X.shape
        self.n_features_in_ = p
        self.location_, emp_cov = _empirical_cov(X, self.assume_centered)
        Xc = X - self.location_
        rho = self._lw_shrinkage(Xc, emp_cov)
        self.shrinkage_ = rho
        self.covariance_ = _shrink_cov(emp_cov, rho)
        if self.store_precision:
            self.precision_ = _pinv_safe(self.covariance_)
        else:
            self.precision_ = None
        return self

    def get_precision(self) -> torch.Tensor:
        if self.precision_ is not None:
            return self.precision_
        return _pinv_safe(self.covariance_)

    def score(self, X_test: torch.Tensor, y=None) -> float:
        """Mean log-likelihood of X_test under the LW-shrunk model."""
        X_test = self._t(X_test)
        return _log_likelihood_score(X_test, self.location_, self.get_precision())

    def error_norm(self,
                   comp_cov: torch.Tensor,
                   norm: str = 'frobenius',
                   scaling: bool = True,
                   squared: bool = True) -> float:
        comp_cov = self._t(comp_cov)
        diff = self.covariance_ - comp_cov
        if norm == 'frobenius':
            err = diff.pow(2).sum()
            if scaling:
                err = err / diff.numel()
        elif norm == 'spectral':
            err = torch.linalg.norm(diff, ord=2).pow(2)
            if scaling:
                err = err / diff.shape[0]
        else:
            err = diff.abs().sum()
            if scaling:
                err = err / diff.numel()
        if not squared:
            err = err.sqrt()
        return err.item()

    def mahalanobis(self, X: torch.Tensor) -> torch.Tensor:
        X = self._t(X)
        return _mahalanobis(X, self.location_, self.get_precision())

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.mahalanobis(X)


class OAS(MLModule):
    def __init__(self,
                 store_precision: bool = True,
                 assume_centered: bool = False,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.store_precision = store_precision
        self.assume_centered = assume_centered
        self.device = device
        self.dtype = dtype

        self.covariance_: Optional[torch.Tensor] = None
        self.location_: Optional[torch.Tensor] = None
        self.precision_: Optional[torch.Tensor] = None
        self.shrinkage_: Optional[float] = None
        self.n_features_in_: Optional[int] = None

    def _t(self, X) -> torch.Tensor:
        return _to_tensor(X, self.dtype, self.device)

    def _oas_shrinkage(self, S: torch.Tensor, n: int, p: int) -> float:
        """
        Oracle Approximating Shrinkage coefficient (Chen et al., 2010):

            ρ* = [(1 − 2/p)·tr(S²) + tr(S)²] / [(n+1−2/p)·(tr(S²) − tr(S)²/p)]
        """
        tr_S = torch.trace(S).item()
        tr_S2 = torch.trace(S @ S).item()
        num = (1.0 - 2.0 / p) * tr_S2 + tr_S ** 2
        den = (n + 1.0 - 2.0 / p) * (tr_S2 - tr_S ** 2 / p)
        if abs(den) < 1e-12:
            return 0.0
        return float(max(0.0, min(1.0, num / den)))

    def fit(self, X: torch.Tensor, y=None) -> 'OAS':
        X = self._t(X)
        n, p = X.shape
        self.n_features_in_ = p
        self.location_, emp_cov = _empirical_cov(X, self.assume_centered)
        rho = self._oas_shrinkage(emp_cov, n, p)
        self.shrinkage_ = rho
        self.covariance_ = _shrink_cov(emp_cov, rho)
        if self.store_precision:
            self.precision_ = _pinv_safe(self.covariance_)
        else:
            self.precision_ = None
        return self

    def get_precision(self) -> torch.Tensor:
        if self.precision_ is not None:
            return self.precision_
        return _pinv_safe(self.covariance_)

    def score(self, X_test: torch.Tensor, y=None) -> float:
        """Mean log-likelihood of X_test under the OAS-shrunk model."""
        X_test = self._t(X_test)
        return _log_likelihood_score(X_test, self.location_, self.get_precision())

    def error_norm(self,
                   comp_cov: torch.Tensor,
                   norm: str = 'frobenius',
                   scaling: bool = True,
                   squared: bool = True) -> float:
        comp_cov = self._t(comp_cov)
        diff = self.covariance_ - comp_cov
        if norm == 'frobenius':
            err = diff.pow(2).sum()
            if scaling:
                err = err / diff.numel()
        elif norm == 'spectral':
            err = torch.linalg.norm(diff, ord=2).pow(2)
            if scaling:
                err = err / diff.shape[0]
        else:
            err = diff.abs().sum()
            if scaling:
                err = err / diff.numel()
        if not squared:
            err = err.sqrt()
        return err.item()

    def mahalanobis(self, X: torch.Tensor) -> torch.Tensor:
        X = self._t(X)
        return _mahalanobis(X, self.location_, self.get_precision())

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.mahalanobis(X)


class MinCovDet(MLModule):
    def __init__(self,
                 store_precision: bool = True,
                 assume_centered: bool = False,
                 support_fraction: float = None,
                 random_state: Union[int, torch.Generator] = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.store_precision = store_precision
        self.assume_centered = assume_centered
        self.support_fraction = support_fraction
        self.random_state = random_state
        self.device = device
        self.dtype = dtype

        self.raw_location_: Optional[torch.Tensor] = None
        self.raw_covariance_: Optional[torch.Tensor] = None
        self.raw_support_: Optional[torch.Tensor] = None
        self.location_: Optional[torch.Tensor] = None
        self.covariance_: Optional[torch.Tensor] = None
        self.precision_: Optional[torch.Tensor] = None
        self.support_: Optional[torch.Tensor] = None
        self.dist_: Optional[torch.Tensor] = None
        self.n_features_in_: Optional[int] = None

    def _t(self, X) -> torch.Tensor:
        return _to_tensor(X, self.dtype, self.device)

    def fit(self, X: torch.Tensor, y=None) -> 'MinCovDet':
        X = self._t(X)
        n, p = X.shape
        self.n_features_in_ = p

        if self.support_fraction is not None:
            h = max(int(n * float(self.support_fraction)), p + 1)
        else:
            h = int((n + p + 1) / 2)
        h = min(h, n)

        raw_loc, raw_cov, support_idx = _fast_mcd(X, h, self.random_state)

        c = _mcd_consistency(h, n, p)
        raw_cov_corrected = raw_cov * c

        self.raw_location_ = raw_loc
        self.raw_covariance_ = raw_cov_corrected

        raw_mask = torch.zeros(n, dtype=torch.bool, device=X.device)
        raw_mask[support_idx] = True
        self.raw_support_ = raw_mask

        loc_rw, cov_rw, support_rw = self.reweight_covariance(X)
        self.location_ = loc_rw
        self.covariance_ = cov_rw
        self.support_ = support_rw

        if self.store_precision:
            self.precision_ = _pinv_safe(self.covariance_)
        else:
            self.precision_ = None

        prec = self.get_precision()
        self.dist_ = _mahalanobis(X, self.location_, prec)
        return self

    def correct_covariance(self, data: torch.Tensor) -> torch.Tensor:
        data = self._t(data)
        n, p = data.shape
        h = int((n + p + 1) / 2)
        c = _mcd_consistency(h, n, p)
        return self.raw_covariance_ * c

    def reweight_covariance(self, data: torch.Tensor
                            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data = self._t(data)
        n, p = data.shape

        raw_prec = _pinv_safe(self.raw_covariance_)
        raw_dist = _mahalanobis(data, self.raw_location_, raw_prec)
        threshold = _chi2_ppf(0.975, float(p))
        mask = raw_dist <= threshold

        if mask.sum().item() < p + 1:
            mask = torch.ones(n, dtype=torch.bool, device=data.device)

        subset = data[mask]
        if self.assume_centered:
            loc_rw = torch.zeros(p, dtype=data.dtype, device=data.device)
        else:
            loc_rw = subset.mean(0)
        Xsub = subset - loc_rw
        cov_rw = (Xsub.T @ Xsub) / subset.shape[0]
        return loc_rw, cov_rw, mask

    def get_precision(self) -> torch.Tensor:
        if self.precision_ is not None:
            return self.precision_
        return _pinv_safe(self.covariance_)

    def mahalanobis(self, X: torch.Tensor) -> torch.Tensor:
        """Squared Mahalanobis distances under the robust precision."""
        X = self._t(X)
        return _mahalanobis(X, self.location_, self.get_precision())

    def score(self, X_test: torch.Tensor, y=None) -> float:
        """Mean log-likelihood of X_test under the robust model."""
        X_test = self._t(X_test)
        return _log_likelihood_score(X_test, self.location_, self.get_precision())

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.mahalanobis(X)


class EllipticEnvelope(MLModule):
    def __init__(self,
                 store_precision: bool = True,
                 assume_centered: bool = False,
                 support_fraction: float = None,
                 contamination: float = 0.1,
                 random_state: Union[int, torch.Generator] = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.store_precision = store_precision
        self.assume_centered = assume_centered
        self.support_fraction = support_fraction
        self.contamination = contamination
        self.random_state = random_state
        self.device = device
        self.dtype = dtype

        self.location_: Optional[torch.Tensor] = None
        self.covariance_: Optional[torch.Tensor] = None
        self.precision_: Optional[torch.Tensor] = None
        self.support_: Optional[torch.Tensor] = None
        self.offset_: Optional[float] = None
        self.raw_location_: Optional[torch.Tensor] = None
        self.raw_covariance_: Optional[torch.Tensor] = None
        self.raw_support_: Optional[torch.Tensor] = None
        self.dist_: Optional[torch.Tensor] = None
        self.n_features_in_: Optional[int] = None

    def _t(self, X) -> torch.Tensor:
        return _to_tensor(X, self.dtype, self.device)

    def fit(self, X: torch.Tensor, y=None) -> 'EllipticEnvelope':
        X = self._t(X)
        n, p = X.shape
        self.n_features_in_ = p

        mcd = MinCovDet(
            store_precision=self.store_precision,
            assume_centered=self.assume_centered,
            support_fraction=self.support_fraction,
            random_state=self.random_state,
            device=self.device,
            dtype=self.dtype,
        )
        mcd.fit(X)

        self.location_ = mcd.location_
        self.covariance_ = mcd.covariance_
        self.precision_ = mcd.precision_
        self.support_ = mcd.support_
        self.raw_location_ = mcd.raw_location_
        self.raw_covariance_ = mcd.raw_covariance_
        self.raw_support_ = mcd.raw_support_
        self.dist_ = mcd.dist_

        # offset: the (1 − contamination) quantile of training Mahalanobis distances
        # so that contamination fraction of training points have decision_function < 0
        contamination = float(self.contamination)
        n_outliers = int(math.ceil(n * contamination))
        dist_sorted, _ = torch.sort(self.dist_)
        cutoff_idx = max(0, n - n_outliers - 1)
        self.offset_ = dist_sorted[cutoff_idx].item()
        return self

    def score_samples(self, X: torch.Tensor) -> torch.Tensor:
        """Return the negative squared Mahalanobis distances (anomaly score)."""
        X = self._t(X)
        prec = self.precision_ if self.precision_ is not None else _pinv_safe(self.covariance_)
        return -_mahalanobis(X, self.location_, prec)

    def decision_function(self, X: torch.Tensor) -> torch.Tensor:
        """Shifted anomaly score: decision_function = score_samples − offset_."""
        return self.score_samples(X) - self.offset_

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Return +1 for inliers, −1 for outliers."""
        X = self._t(X)
        df = self.decision_function(X)
        return torch.where(df >= 0,
                           torch.ones_like(df),
                           -torch.ones_like(df))

    def score(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """Accuracy of outlier predictions against ground-truth labels y ∈ {−1, +1}."""
        X = self._t(X)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=self.dtype, device=self.device)
        y = y.to(device=self.device)
        preds = self.predict(X)
        return (preds == y.float()).float().mean().item()

    def mahalanobis(self, X: torch.Tensor) -> torch.Tensor:
        """Squared Mahalanobis distances under the robust precision."""
        X = self._t(X)
        prec = self.precision_ if self.precision_ is not None else _pinv_safe(self.covariance_)
        return _mahalanobis(X, self.location_, prec)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.decision_function(X)


# ─────────────────────────────────────────────────────────────────────────────
# Graphical model estimators
# ─────────────────────────────────────────────────────────────────────────────

class GraphicalLasso(MLModule):
    def __init__(self,
                 alpha: float = 0.01,
                 mode: Literal["lars", "cd"] = 'cd',
                 covariance: Union[Literal["precomputed", "none"], torch.Tensor] = None,
                 tol: float = 0.0001,
                 enet_tol: float = 0.0001,
                 max_iter: int = 100,
                 verbose: bool = False,
                 eps: float = 2.220446049250313e-16,
                 assume_centered: bool = False,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.alpha = alpha
        self.mode = mode
        self.covariance = covariance
        self.tol = tol
        self.enet_tol = enet_tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.eps = eps
        self.assume_centered = assume_centered
        self.device = device
        self.dtype = dtype

        self.location_: Optional[torch.Tensor] = None
        self.covariance_: Optional[torch.Tensor] = None
        self.precision_: Optional[torch.Tensor] = None
        self.n_iter_: Optional[int] = None
        self.costs_: Optional[List[Tuple[float, float]]] = None
        self.n_features_in_: Optional[int] = None

    def _t(self, X) -> torch.Tensor:
        return _to_tensor(X, self.dtype, self.device)

    def fit(self, X: torch.Tensor, y=None) -> 'GraphicalLasso':
        X = self._t(X)
        n, p = X.shape
        self.n_features_in_ = p

        if self.covariance == "precomputed":
            S = X.clone()
            self.location_ = torch.zeros(p, dtype=X.dtype, device=X.device)
        else:
            self.location_, S = _empirical_cov(X, self.assume_centered)

        self.covariance_, self.precision_, self.n_iter_, self.costs_ = (
            _graphical_lasso_fit(
                S, float(self.alpha), self.mode,
                float(self.tol), float(self.enet_tol),
                self.max_iter, bool(self.verbose), _eps_float(self.eps),
            )
        )
        return self

    def score(self, X_test: torch.Tensor, y=None) -> float:
        """Mean Gaussian log-likelihood of X_test under the estimated model."""
        X_test = self._t(X_test)
        return _log_likelihood_score(X_test, self.location_, self.precision_)

    def score_samples(self, X: torch.Tensor) -> torch.Tensor:
        """Per-sample log-likelihood under the estimated Gaussian model."""
        X = self._t(X)
        p = self.precision_.shape[0]
        Xc = X - self.location_
        _, logdet = torch.linalg.slogdet(self.precision_)
        maha = (Xc @ self.precision_ * Xc).sum(dim=1)
        return 0.5 * (logdet - p * math.log(2.0 * math.pi) - maha)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.score_samples(X)


class GraphicalLassoCV(MLModule):
    def __init__(self,
                 alphas: [float, list, tuple, torch.Tensor] = 0.01,
                 mode: Literal["lars", "cd"] = 'cd',
                 covariance: Union[Literal["precomputed", "none"], torch.Tensor] = None,
                 tol: float = 0.0001,
                 cv: Union[int, str, MLModule] = None,
                 cv_config: dict = None,
                 enet_tol: float = 0.0001,
                 max_iter: int = 100,
                 verbose: bool = False,
                 eps: float = 2.220446049250313e-16,
                 assume_centered: bool = False,
                 n_refinements: int = 4,
                 n_jobs: Optional[int] = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.alphas = alphas
        self.mode = mode
        self.covariance = covariance
        self.tol = tol
        self.cv = cv
        self.cv_config = cv_config
        self.enet_tol = enet_tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.eps = eps
        self.assume_centered = assume_centered
        self.n_refinements = n_refinements
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype

        self.location_: Optional[torch.Tensor] = None
        self.covariance_: Optional[torch.Tensor] = None
        self.precision_: Optional[torch.Tensor] = None
        self.costs_: Optional[List[Tuple[float, float]]] = None
        self.alpha_: Optional[float] = None
        self.cv_results_: Optional[Dict] = None
        self.n_iter_: Optional[int] = None
        self.n_features_in_: Optional[int] = None

    def _t(self, X) -> torch.Tensor:
        return _to_tensor(X, self.dtype, self.device)

    def fit(self, X: torch.Tensor, y=None) -> 'GraphicalLassoCV':
        X = self._t(X)
        n, p = X.shape
        self.n_features_in_ = p
        eps_val = _eps_float(self.eps)

        if self.covariance == "precomputed":
            S = X.clone()
            emp_loc = torch.zeros(p, dtype=X.dtype, device=X.device)
        else:
            emp_loc, S = _empirical_cov(X, self.assume_centered)

        alpha_grid = _resolve_alphas(self.alphas, S)

        base_est = GraphicalLasso(
            alpha=alpha_grid[0],
            mode=self.mode,
            covariance=self.covariance,
            tol=self.tol,
            enet_tol=self.enet_tol,
            max_iter=self.max_iter,
            verbose=False,
            eps=eps_val,
            assume_centered=self.assume_centered,
            device=self.device,
            dtype=self.dtype,
        )

        best_alpha: float = alpha_grid[0]
        all_cv_results: Dict = {}
        n_ref = max(0, int(self.n_refinements)) if isinstance(self.alphas, int) else 0

        for refine_idx in range(n_ref + 1):
            best_params, cv_results = _run_grid_cv(
                base_est, {'alpha': alpha_grid},
                X, self.cv, self.cv_config, self.verbose, self.device, self.dtype,
            )
            all_cv_results = (_merge_cv_results(all_cv_results, cv_results)
                              if all_cv_results else cv_results)
            best_alpha = float(best_params['alpha'])

            if refine_idx < n_ref:
                new_grid = _refine_grid(alpha_grid, best_alpha, len(alpha_grid))
                if set(new_grid) == set(alpha_grid):
                    break
                alpha_grid = new_grid

        self.alpha_ = best_alpha
        self.cv_results_ = all_cv_results

        self.covariance_, self.precision_, self.n_iter_, self.costs_ = (
            _graphical_lasso_fit(
                S, best_alpha, self.mode,
                float(self.tol), float(self.enet_tol),
                self.max_iter, bool(self.verbose), eps_val,
            )
        )
        self.location_ = emp_loc
        return self

    def score(self, X_test: torch.Tensor, y=None) -> float:
        """Mean Gaussian log-likelihood of X_test under the best-alpha model."""
        X_test = self._t(X_test)
        return _log_likelihood_score(X_test, self.location_, self.precision_)

    def score_samples(self, X: torch.Tensor) -> torch.Tensor:
        X = self._t(X)
        p = self.precision_.shape[0]
        Xc = X - self.location_
        _, logdet = torch.linalg.slogdet(self.precision_)
        maha = (Xc @ self.precision_ * Xc).sum(dim=1)
        return 0.5 * (logdet - p * math.log(2.0 * math.pi) - maha)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.score_samples(X)


class GraphicalRidge(MLModule):
    def __init__(self,
                 alpha: float = 0.01,
                 covariance: Union[Literal["precomputed", "none"], torch.Tensor] = None,
                 tol: float = 1e-4,
                 max_iter: int = 100,
                 verbose: bool = False,
                 eps: float = _FLOAT64_EPS,
                 assume_centered: bool = False,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.alpha = alpha
        self.covariance = covariance
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.eps = eps
        self.assume_centered = assume_centered
        self.device = device
        self.dtype = dtype

        self.location_: Optional[torch.Tensor] = None
        self.covariance_: Optional[torch.Tensor] = None
        self.precision_: Optional[torch.Tensor] = None
        self.n_iter_: Optional[int] = None
        self.n_features_in_: Optional[int] = None

    def _t(self, X) -> torch.Tensor:
        return _to_tensor(X, self.dtype, self.device)

    def fit(self, X: torch.Tensor, y=None) -> 'GraphicalRidge':
        X = self._t(X)
        n, p = X.shape
        self.n_features_in_ = p

        if self.covariance == "precomputed":
            S = X.clone()
            self.location_ = torch.zeros(p, dtype=X.dtype, device=X.device)
        else:
            self.location_, S = _empirical_cov(X, self.assume_centered)

        alpha = float(self.alpha)
        eps_val = _eps_float(self.eps)

        # Symmetric eigendecomposition: S = U D U'
        eigvals, eigvecs = torch.linalg.eigh(S)

        # Closed-form solution of  alpha·t² + d·t − 1 = 0  (positive root)
        t = (-eigvals + torch.sqrt(eigvals.pow(2) + 4.0 * alpha)) / (2.0 * alpha)
        t = t.clamp(min=eps_val)

        self.precision_ = eigvecs @ torch.diag(t) @ eigvecs.T
        self.covariance_ = eigvecs @ torch.diag(1.0 / t) @ eigvecs.T
        self.n_iter_ = 1
        return self

    def score(self, X_test: torch.Tensor, y=None) -> float:
        """Mean Gaussian log-likelihood of X_test under the ridge-penalized model."""
        X_test = self._t(X_test)
        return _log_likelihood_score(X_test, self.location_, self.precision_)

    def score_samples(self, X: torch.Tensor) -> torch.Tensor:
        X = self._t(X)
        p = self.precision_.shape[0]
        Xc = X - self.location_
        _, logdet = torch.linalg.slogdet(self.precision_)
        maha = (Xc @ self.precision_ * Xc).sum(dim=1)
        return 0.5 * (logdet - p * math.log(2.0 * math.pi) - maha)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.score_samples(X)


class GraphicalRidgeCV(MLModule):
    def __init__(self,
                 alphas: Union[int, List[float], torch.Tensor] = 4,
                 covariance: Union[Literal["precomputed", "none"], torch.Tensor] = None,
                 tol: float = 1e-4,
                 cv: Union[int, str, MLModule] = None,
                 cv_config: dict = None,
                 max_iter: int = 100,
                 verbose: bool = False,
                 eps: float = _FLOAT64_EPS,
                 assume_centered: bool = False,
                 n_refinements: int = 4,
                 n_jobs: Optional[int] = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.alphas = alphas
        self.covariance = covariance
        self.tol = tol
        self.cv = cv
        self.cv_config = cv_config
        self.max_iter = max_iter
        self.verbose = verbose
        self.eps = eps
        self.assume_centered = assume_centered
        self.n_refinements = n_refinements
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype

        self.location_: Optional[torch.Tensor] = None
        self.covariance_: Optional[torch.Tensor] = None
        self.precision_: Optional[torch.Tensor] = None
        self.alpha_: Optional[float] = None
        self.cv_results_: Optional[Dict] = None
        self.n_iter_: Optional[int] = None
        self.n_features_in_: Optional[int] = None

    def _t(self, X) -> torch.Tensor:
        return _to_tensor(X, self.dtype, self.device)

    def fit(self, X: torch.Tensor, y=None) -> 'GraphicalRidgeCV':
        X = self._t(X)
        n, p = X.shape
        self.n_features_in_ = p
        eps_val = _eps_float(self.eps)

        if self.covariance == "precomputed":
            S = X.clone()
            emp_loc = torch.zeros(p, dtype=X.dtype, device=X.device)
        else:
            emp_loc, S = _empirical_cov(X, self.assume_centered)

        alpha_grid = _resolve_alphas(self.alphas, S)
        base_est = GraphicalRidge(
            alpha=alpha_grid[0],
            covariance=self.covariance,
            tol=self.tol,
            max_iter=self.max_iter,
            verbose=False,
            eps=eps_val,
            assume_centered=self.assume_centered,
            device=self.device,
            dtype=self.dtype,
        )

        best_alpha: float = alpha_grid[0]
        all_cv_results: Dict = {}
        n_ref = max(0, int(self.n_refinements)) if isinstance(self.alphas, int) else 0

        for refine_idx in range(n_ref + 1):
            best_params, cv_results = _run_grid_cv(
                base_est, {'alpha': alpha_grid},
                X, self.cv, self.cv_config, self.verbose, self.device, self.dtype,
            )
            all_cv_results = (_merge_cv_results(all_cv_results, cv_results)
                              if all_cv_results else cv_results)
            best_alpha = float(best_params['alpha'])

            if refine_idx < n_ref:
                new_grid = _refine_grid(alpha_grid, best_alpha, len(alpha_grid))
                if set(new_grid) == set(alpha_grid):
                    break
                alpha_grid = new_grid

        self.alpha_ = best_alpha
        self.cv_results_ = all_cv_results

        best_est = GraphicalRidge(
            alpha=best_alpha,
            covariance=self.covariance,
            tol=self.tol,
            max_iter=self.max_iter,
            verbose=bool(self.verbose),
            eps=eps_val,
            assume_centered=self.assume_centered,
            device=self.device,
            dtype=self.dtype,
        )
        best_est.fit(X)
        self.location_ = emp_loc
        self.covariance_ = best_est.covariance_
        self.precision_ = best_est.precision_
        self.n_iter_ = best_est.n_iter_
        return self

    def score(self, X_test: torch.Tensor, y=None) -> float:
        X_test = self._t(X_test)
        return _log_likelihood_score(X_test, self.location_, self.precision_)

    def score_samples(self, X: torch.Tensor) -> torch.Tensor:
        X = self._t(X)
        p = self.precision_.shape[0]
        Xc = X - self.location_
        _, logdet = torch.linalg.slogdet(self.precision_)
        maha = (Xc @ self.precision_ * Xc).sum(dim=1)
        return 0.5 * (logdet - p * math.log(2.0 * math.pi) - maha)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.score_samples(X)


class GraphicalElasticNet(MLModule):
    def __init__(self,
                 alpha: float = 0.01,
                 l1_ratio: float = 0.5,
                 mode: Literal["lars", "cd"] = 'cd',
                 covariance: Union[Literal["precomputed", "none"], torch.Tensor] = None,
                 tol: float = 1e-4,
                 enet_tol: float = 1e-4,
                 max_iter: int = 100,
                 verbose: bool = False,
                 eps: float = _FLOAT64_EPS,
                 assume_centered: bool = False,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.mode = mode
        self.covariance = covariance
        self.tol = tol
        self.enet_tol = enet_tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.eps = eps
        self.assume_centered = assume_centered
        self.device = device
        self.dtype = dtype

        self.location_: Optional[torch.Tensor] = None
        self.covariance_: Optional[torch.Tensor] = None
        self.precision_: Optional[torch.Tensor] = None
        self.n_iter_: Optional[int] = None
        self.costs_: Optional[List[Tuple[float, float]]] = None
        self.n_features_in_: Optional[int] = None

    def _t(self, X) -> torch.Tensor:
        return _to_tensor(X, self.dtype, self.device)

    def fit(self, X: torch.Tensor, y=None) -> 'GraphicalElasticNet':
        X = self._t(X)
        n, p = X.shape
        self.n_features_in_ = p

        if self.covariance == "precomputed":
            S = X.clone()
            self.location_ = torch.zeros(p, dtype=X.dtype, device=X.device)
        else:
            self.location_, S = _empirical_cov(X, self.assume_centered)

        self.covariance_, self.precision_, self.n_iter_, self.costs_ = (
            _graphical_enet_fit(
                S, float(self.alpha), float(self.l1_ratio), self.mode,
                float(self.tol), float(self.enet_tol),
                self.max_iter, bool(self.verbose), _eps_float(self.eps),
            )
        )
        return self

    def score(self, X_test: torch.Tensor, y=None) -> float:
        """Mean Gaussian log-likelihood of X_test under the elastic-net model."""
        X_test = self._t(X_test)
        return _log_likelihood_score(X_test, self.location_, self.precision_)

    def score_samples(self, X: torch.Tensor) -> torch.Tensor:
        X = self._t(X)
        p = self.precision_.shape[0]
        Xc = X - self.location_
        _, logdet = torch.linalg.slogdet(self.precision_)
        maha = (Xc @ self.precision_ * Xc).sum(dim=1)
        return 0.5 * (logdet - p * math.log(2.0 * math.pi) - maha)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.score_samples(X)


class GraphicalElasticNetCV(MLModule):
    def __init__(self,
                 alphas: Union[int, List[float], torch.Tensor] = 4,
                 l1_ratios: Union[float, List[float], torch.Tensor] = 0.5,
                 mode: Literal["lars", "cd"] = 'cd',
                 covariance: Union[Literal["precomputed", "none"], torch.Tensor] = None,
                 tol: float = 1e-4,
                 enet_tol: float = 1e-4,
                 cv: Union[int, str, MLModule] = None,
                 cv_config: dict = None,
                 max_iter: int = 100,
                 verbose: bool = False,
                 eps: float = _FLOAT64_EPS,
                 assume_centered: bool = False,
                 n_refinements: int = 4,
                 n_jobs: Optional[int] = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.alphas = alphas
        self.l1_ratios = l1_ratios
        self.mode = mode
        self.covariance = covariance
        self.tol = tol
        self.enet_tol = enet_tol
        self.cv = cv
        self.cv_config = cv_config
        self.max_iter = max_iter
        self.verbose = verbose
        self.eps = eps
        self.assume_centered = assume_centered
        self.n_refinements = n_refinements
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype

        self.location_: Optional[torch.Tensor] = None
        self.covariance_: Optional[torch.Tensor] = None
        self.precision_: Optional[torch.Tensor] = None
        self.costs_: Optional[List[Tuple[float, float]]] = None
        self.alpha_: Optional[float] = None
        self.l1_ratio_: Optional[float] = None
        self.cv_results_: Optional[Dict] = None
        self.n_iter_: Optional[int] = None
        self.n_features_in_: Optional[int] = None

    def _t(self, X) -> torch.Tensor:
        return _to_tensor(X, self.dtype, self.device)

    def _get_l1_ratio_grid(self) -> List[float]:
        if isinstance(self.l1_ratios, (list, tuple)):
            return [float(r) for r in self.l1_ratios]
        if isinstance(self.l1_ratios, torch.Tensor):
            return self.l1_ratios.flatten().tolist()
        return [float(self.l1_ratios)]

    def fit(self, X: torch.Tensor, y=None) -> 'GraphicalElasticNetCV':
        X = self._t(X)
        n, p = X.shape
        self.n_features_in_ = p
        eps_val = _eps_float(self.eps)

        if self.covariance == "precomputed":
            S = X.clone()
            emp_loc = torch.zeros(p, dtype=X.dtype, device=X.device)
        else:
            emp_loc, S = _empirical_cov(X, self.assume_centered)

        alpha_grid = _resolve_alphas(self.alphas, S)
        l1_grid = self._get_l1_ratio_grid()

        base_est = GraphicalElasticNet(
            alpha=alpha_grid[0],
            l1_ratio=l1_grid[0],
            mode=self.mode,
            covariance=self.covariance,
            tol=self.tol,
            enet_tol=self.enet_tol,
            max_iter=self.max_iter,
            verbose=False,
            eps=eps_val,
            assume_centered=self.assume_centered,
            device=self.device,
            dtype=self.dtype,
        )

        best_alpha: float = alpha_grid[0]
        best_l1: float = l1_grid[0]
        all_cv_results: Dict = {}
        n_ref = max(0, int(self.n_refinements)) if isinstance(self.alphas, int) else 0

        for refine_idx in range(n_ref + 1):
            param_grid = {'alpha': alpha_grid, 'l1_ratio': l1_grid}
            best_params, cv_results = _run_grid_cv(
                base_est, param_grid,
                X, self.cv, self.cv_config, self.verbose, self.device, self.dtype,
            )
            all_cv_results = (_merge_cv_results(all_cv_results, cv_results)
                              if all_cv_results else cv_results)
            best_alpha = float(best_params['alpha'])
            best_l1 = float(best_params['l1_ratio'])

            if refine_idx < n_ref:
                new_grid = _refine_grid(alpha_grid, best_alpha, len(alpha_grid))
                if set(new_grid) == set(alpha_grid):
                    break
                alpha_grid = new_grid

        self.alpha_ = best_alpha
        self.l1_ratio_ = best_l1
        self.cv_results_ = all_cv_results

        best_est = GraphicalElasticNet(
            alpha=best_alpha,
            l1_ratio=best_l1,
            mode=self.mode,
            covariance=self.covariance,
            tol=self.tol,
            enet_tol=self.enet_tol,
            max_iter=self.max_iter,
            verbose=bool(self.verbose),
            eps=eps_val,
            assume_centered=self.assume_centered,
            device=self.device,
            dtype=self.dtype,
        )
        best_est.fit(X)
        self.location_ = emp_loc
        self.covariance_ = best_est.covariance_
        self.precision_ = best_est.precision_
        self.n_iter_ = best_est.n_iter_
        self.costs_ = best_est.costs_
        return self

    def score(self, X_test: torch.Tensor, y=None) -> float:
        """Mean Gaussian log-likelihood of X_test under the best-param model."""
        X_test = self._t(X_test)
        return _log_likelihood_score(X_test, self.location_, self.precision_)

    def score_samples(self, X: torch.Tensor) -> torch.Tensor:
        X = self._t(X)
        p = self.precision_.shape[0]
        Xc = X - self.location_
        _, logdet = torch.linalg.slogdet(self.precision_)
        maha = (Xc @ self.precision_ * Xc).sum(dim=1)
        return 0.5 * (logdet - p * math.log(2.0 * math.pi) - maha)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.score_samples(X)

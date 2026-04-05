import copy
import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Callable, Union, Any, List, Tuple, Dict, Literal
from ....models.utils import MLModule
from torch.func import vmap
import joblib


def _betainc(a: Union[torch.Tensor, float], b: Union[torch.Tensor, float], x: torch.Tensor) -> torch.Tensor:
    """Regularized incomplete beta function. Fallback to scipy when torch.special.betainc unavailable."""
    device = x.device if isinstance(x, torch.Tensor) else torch.device("cpu")
    dtype = x.dtype if isinstance(x, torch.Tensor) else torch.float
    a = torch.as_tensor(a, device=device, dtype=dtype) if not isinstance(a, torch.Tensor) else a
    b = torch.as_tensor(b, device=device, dtype=dtype) if not isinstance(b, torch.Tensor) else b
    if hasattr(torch.special, "betainc"):
        return torch.special.betainc(a, b, x)
    try:
        import scipy.special
        a_np = a.cpu().numpy() if isinstance(a, torch.Tensor) else a
        b_np = b.cpu().numpy() if isinstance(b, torch.Tensor) else b
        x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
        out = scipy.special.betainc(a_np, b_np, x_np)
        return torch.as_tensor(out, device=device, dtype=dtype)
    except ImportError:
        raise RuntimeError(
            "betainc requires torch.special.betainc (PyTorch 1.10+) or scipy. Install scipy."
            
        )


def _arpack_svd(
    C: torch.Tensor,
    n_components: int,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_features, n_targets = C.shape
    k = min(n_components, min(n_features, n_targets))
    k = max(1, k)
    eps = 1e-12
    if n_features >= n_targets:
        G = C @ C.T
        n_g = n_features
        transposed = False
    else:
        G = C.T @ C
        n_g = n_targets
        transposed = True
    S_sq = []
    V_list = []
    G_work = G.clone()
    for j in range(k):
        v = torch.randn(n_g, device=C.device, dtype=C.dtype)
        v = v / (v.norm().clamp(min=eps))
        for _ in range(max(max_iter, n_g)):
            v_new = G_work @ v
            v_new = v_new / (v_new.norm().clamp(min=eps))
            if (v_new - v).norm() < tol:
                break
            v = v_new
        lam = (v @ G_work @ v).item()
        lam = max(lam, eps)
        s = (lam ** 0.5)
        S_sq.append(s)
        V_list.append(v.clone())
        G_work = G_work - lam * torch.outer(v, v)
    S = torch.tensor(S_sq, device=C.device, dtype=C.dtype)
    V_stack = torch.stack(V_list, dim=1)
    if transposed:
        U = (C @ V_stack) / S.clamp(min=eps)
        Vh = V_stack.T
        return U[:, :k], S, Vh[:k]
    else:
        U = V_stack
        Vh = (V_stack.T @ C) / S.clamp(min=eps).unsqueeze(1)
        return U[:, :k], S, Vh[:k]


__all__ = [
    "RFE",
    "RFECV",
    "SequentialFeatureSelector",
    "VarianceThreshold",
    "_arpack_svd",
    "f_classif",
    "chi2",
    "f_regression",
    "r_regression",
    "mutual_info_classif",
    "mutual_info_regression",
    "GenericUnivariateSelect",
    "SelectFdr",
    "SelectFpr",
    "SelectFwe",
    "SelectKBest",
    "SelectPercentile",
    "SelectFromModel",
]


def _f_classif_impl(X: torch.Tensor, y: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    X = torch.as_tensor(X, dtype=torch.float)
    y = torch.as_tensor(y)
    if hasattr(y, 'dtype') and y.dtype in (torch.float32, torch.float64, torch.float):
        y = y.long()
    if X.dim() == 1:
        X = X.unsqueeze(1)
    if y.dim() > 1:
        y = y.flatten()
    n_samples, n_features = X.shape
    device, dtype = X.device, X.dtype

    classes = torch.unique(y)
    n_classes = len(classes)
    if n_classes < 2:
        return torch.zeros(n_features, device=device, dtype=dtype), torch.ones(n_features, device=device, dtype=dtype)

    grand_mean = X.mean(dim=0)
    ss_total = ((X - grand_mean) ** 2).sum(dim=0)
    ss_between = torch.zeros(n_features, device=device, dtype=dtype)
    ss_within = torch.zeros(n_features, device=device, dtype=dtype)

    for c in classes:
        mask = (y == c)
        n_k = mask.sum().float().clamp(min=1)
        class_mean = X[mask].mean(dim=0)
        ss_between += n_k * (class_mean - grand_mean) ** 2
        ss_within += ((X[mask] - class_mean) ** 2).sum(dim=0)

    df_between = n_classes - 1
    df_within = n_samples - n_classes
    df_within = max(df_within, 1)

    ms_between = ss_between / df_between
    ms_within = ss_within.clamp(min=1e-14) / df_within
    f_stat = ms_between / ms_within
    f_stat = torch.nan_to_num(f_stat, nan=0.0, posinf=0.0, neginf=0.0)

    # p-value: 1 - CDF of F(df_between, df_within)
    x_beta = (df_between * f_stat) / (df_within + df_between * f_stat).clamp(min=1e-14)
    x_beta = x_beta.clamp(0.0, 1.0)
    p_values = 1.0 - _betainc(df_between / 2.0, df_within / 2.0, x_beta)
    p_values = torch.nan_to_num(p_values, nan=1.0, posinf=1.0, neginf=1.0)
    return f_stat, p_values


class FClassif(MLModule):
    """Instantiable score function for ANOVA F-value (f_classif)."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.device = kwargs.get("device", "cpu")
        self.dtype = kwargs.get("dtype", torch.float)

    def __new__(cls, X=None, y=None, *args, **kwargs):
        if X is not None and y is not None:
            return _f_classif_impl(X, y, *args, **kwargs)
        return super().__new__(cls)

    def fit(self, X=None, y=None, **kwargs):
        """No-op fit for API compatibility (e.g. save/load tests)."""
        return self


f_classif = FClassif


def _chi2_impl(X: torch.Tensor, y: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    X = torch.as_tensor(X, dtype=torch.float)
    y = torch.as_tensor(y, dtype=torch.long)
    if X.dim() == 1:
        X = X.unsqueeze(1)
    if y.dim() > 1:
        y = y.flatten()
    if (X < 0).any():
        raise ValueError("Input X must be non-negative for chi2.")
    n_samples, n_features = X.shape
    device, dtype = X.device, X.dtype

    classes = torch.unique(y)
    n_classes = len(classes)
    chi2_stats = torch.zeros(n_features, device=device, dtype=dtype)
    p_values = torch.ones(n_features, device=device, dtype=dtype)

    for j in range(n_features):
        col = X[:, j]
        observed_list = []
        for c in classes:
            mask = (y == c)
            col_c = col[mask]
            unique_vals = torch.unique(col_c)
            for v in unique_vals:
                count = (col_c == v).sum().float()
                observed_list.append((c.item(), v.item(), count))
        if not observed_list:
            continue
        unique_x = torch.unique(col)
        unique_y = classes
        contingency = torch.zeros(len(unique_y), len(unique_x), device=device, dtype=dtype)
        for iy, cy in enumerate(unique_y):
            for ix, vx in enumerate(unique_x):
                mask = (y == cy) & (col == vx)
                contingency[iy, ix] = mask.sum().float()
        row_sums = contingency.sum(dim=1, keepdim=True)
        col_sums = contingency.sum(dim=0, keepdim=True)
        total = contingency.sum()
        if total <= 0:
            continue
        expected = row_sums @ col_sums / total.clamp(min=1e-14)
        diff = contingency - expected
        chi2_val = ((diff ** 2) / expected.clamp(min=1e-14)).sum()
        df = (contingency.shape[0] - 1) * (contingency.shape[1] - 1)
        df = max(df, 1)
        chi2_stats[j] = chi2_val
        p_values[j] = 1.0 - torch.special.gammaincc(df / 2.0, chi2_val / 2.0)
    p_values = torch.nan_to_num(p_values, nan=1.0, posinf=0.0, neginf=1.0)
    return chi2_stats, p_values


class Chi2(MLModule):
    """Instantiable score function for chi-squared (chi2)."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.device = kwargs.get("device", "cpu")
        self.dtype = kwargs.get("dtype", torch.float)

    def __new__(cls, X=None, y=None, *args, **kwargs):
        if X is not None and y is not None:
            return _chi2_impl(X, y, *args, **kwargs)
        return super().__new__(cls)

    def fit(self, X=None, y=None, **kwargs):
        """No-op fit for API compatibility (e.g. save/load tests)."""
        return self


chi2 = Chi2


def _f_regression_impl(X: torch.Tensor, y: torch.Tensor,
                 center: bool = True,
                 force_finite: bool = True,
                 *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    r = _r_regression_impl(X, y, center=center, force_finite=force_finite, *args, **kwargs)
    n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
    n_samples = max(n_samples, 3)
    r = torch.as_tensor(r, dtype=torch.float)
    r_sq = r ** 2
    f_stat = r_sq * (n_samples - 2) / (1 - r_sq).clamp(min=1e-14)
    if force_finite:
        f_stat = torch.nan_to_num(f_stat, nan=0.0, posinf=float('inf'))
        f_stat = torch.clamp(f_stat, 0.0, 1e10)
    df1, df2 = 1.0, max(n_samples - 2, 1.0)
    x_beta = (df1 * f_stat) / (df2 + df1 * f_stat).clamp(min=1e-14)
    x_beta = x_beta.clamp(0.0, 1.0)
    p_values = 1.0 - _betainc(df1 / 2.0, df2 / 2.0, x_beta)
    if force_finite:
        p_values = torch.nan_to_num(p_values, nan=1.0, posinf=0.0, neginf=1.0)
    return f_stat, p_values


class FRegression(MLModule):
    """Instantiable score function for F-regression (f_regression)."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.device = kwargs.get("device", "cpu")
        self.dtype = kwargs.get("dtype", torch.float)

    def __new__(cls, X=None, y=None, *args, **kwargs):
        if X is not None and y is not None:
            return _f_regression_impl(X, y, *args, **kwargs)
        return super().__new__(cls)

    def fit(self, X=None, y=None, **kwargs):
        """No-op fit for API compatibility (e.g. save/load tests)."""
        return self


f_regression = FRegression


def _r_regression_impl(X: torch.Tensor, y: torch.Tensor,
                 center: bool = True,
                 force_finite: bool = True,
                 *args, **kwargs) -> torch.Tensor:
    X = torch.as_tensor(X, dtype=torch.float)
    y = torch.as_tensor(y, dtype=torch.float)
    if X.dim() == 1:
        X = X.unsqueeze(1)
    if y.dim() > 1:
        y = y.flatten()
    n_samples, n_features = X.shape
    device, dtype = X.device, X.dtype

    if center:
        X = X - X.mean(dim=0)
        y = y - y.mean()
    cov_xy = (X.T @ y) / max(n_samples - 1, 1)
    std_x = X.std(dim=0, unbiased=True)
    std_y = y.std(unbiased=True)
    denom = std_x * std_y
    denom = denom.clamp(min=1e-14)
    r = cov_xy / denom
    if force_finite:
        r = torch.nan_to_num(r, nan=0.0, posinf=1.0, neginf=-1.0)
        r = torch.clamp(r, -1.0, 1.0)
    return r


class RRegression(MLModule):
    """Instantiable score function for Pearson r (r_regression)."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.device = kwargs.get("device", "cpu")
        self.dtype = kwargs.get("dtype", torch.float)

    def __new__(cls, X=None, y=None, *args, **kwargs):
        if X is not None and y is not None:
            return _r_regression_impl(X, y, *args, **kwargs)
        return super().__new__(cls)

    def fit(self, X=None, y=None, **kwargs):
        """No-op fit for API compatibility (e.g. save/load tests)."""
        return self


r_regression = RRegression


def _mutual_info_classif_impl(X: torch.Tensor, y: torch.Tensor,
                        discrete_features: Union[Literal["auto"], bool] = "auto",
                        n_neighbors: int = 5,
                        copy: bool = True,
                        random_state: Union[int, torch.Generator] = None,
                        n_jobs: int = None,
                        *args, **kwargs):
    X = torch.as_tensor(X, dtype=torch.float)
    y = torch.as_tensor(y, dtype=torch.long)
    if copy:
        X = X.clone()
    if X.dim() == 1:
        X = X.unsqueeze(1)
    if y.dim() > 1:
        y = y.flatten()
    n_samples, n_features = X.shape
    device, dtype = X.device, X.dtype
    if random_state is not None:
        if isinstance(random_state, int):
            g = torch.Generator(device=device).manual_seed(random_state)
        else:
            g = random_state
    else:
        g = None
    eps = 1e-10
    k = min(n_neighbors, n_samples - 1)
    if k < 1:
        return torch.zeros(n_features, device=device, dtype=dtype)
    mi_out = torch.zeros(n_features, device=device, dtype=dtype)
    for j in range(n_features):
        col = X[:, j].unsqueeze(1)
        if g is not None:
            col = col + torch.randn_like(col, generator=g) * eps
        xy = torch.cat([col, y.unsqueeze(1).float()], dim=1)
        dists = torch.cdist(xy, xy, p=2)
        dists.fill_diagonal_(float('inf'))
        _, idx = torch.topk(dists, k, largest=False, dim=1)
        k_dist = dists.gather(1, idx[:, -1:])
        n_x = torch.zeros(n_samples, device=device, dtype=dtype)
        for i in range(n_samples):
            r = k_dist[i].item()
            if r < eps:
                r = eps
            mask = (torch.norm(xy - xy[i:i+1], dim=1) <= r) & (torch.arange(n_samples, device=device) != i)
            n_x[i] = mask.sum().float()
        digamma_n = torch.special.digamma(torch.tensor(n_samples, device=device, dtype=dtype))
        digamma_k = torch.special.digamma(torch.tensor(k, device=device, dtype=dtype))
        digamma_nx = torch.special.digamma(n_x.clamp(min=1))
        mi_out[j] = (digamma_n + digamma_k - digamma_nx).mean()
    return mi_out


class MutualInfoClassif(MLModule):
    """Instantiable score function for mutual_info_classif."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.device = kwargs.get("device", "cpu")
        self.dtype = kwargs.get("dtype", torch.float)

    def __new__(cls, X=None, y=None, *args, **kwargs):
        if X is not None and y is not None:
            return _mutual_info_classif_impl(X, y, *args, **kwargs)
        return super().__new__(cls)

    def fit(self, X=None, y=None, **kwargs):
        """No-op fit for API compatibility (e.g. save/load tests)."""
        return self


mutual_info_classif = MutualInfoClassif


def _mutual_info_regression_impl(X: torch.Tensor, y: torch.Tensor,
                        discrete_features: Union[Literal["auto"], bool] = "auto",
                        n_neighbors: int = 5,
                        copy: bool = True,
                        random_state: Union[int, torch.Generator] = None,
                        n_jobs: int = None,
                        *args, **kwargs):
    X = torch.as_tensor(X, dtype=torch.float)
    y = torch.as_tensor(y, dtype=torch.float)
    if copy:
        X = X.clone()
    if X.dim() == 1:
        X = X.unsqueeze(1)
    if y.dim() > 1:
        y = y.flatten().float()
    n_samples, n_features = X.shape
    device, dtype = X.device, X.dtype
    if random_state is not None:
        if isinstance(random_state, int):
            g = torch.Generator(device=device).manual_seed(random_state)
        else:
            g = random_state
    else:
        g = None
    eps = 1e-10
    mi_out = torch.zeros(n_features, device=device, dtype=dtype)
    n_neighbors = min(n_neighbors, n_samples - 1)
    if n_neighbors < 1:
        return mi_out
    for j in range(n_features):
        col = X[:, j:j+1]
        if g is not None:
            col = col + torch.randn_like(col, generator=g) * eps
        xy = torch.cat([col, y.unsqueeze(1)], dim=1)
        dists = torch.cdist(xy, xy, p=2)
        dists.fill_diagonal_(float('inf'))
        _, idx = torch.topk(dists, n_neighbors, largest=False, dim=1)
        k_dist = dists.gather(1, idx[:, -1:])
        n_x = (dists <= k_dist).sum(dim=1).float() - 1
        n_y = (dists <= k_dist).sum(dim=1).float() - 1
        digamma_n = torch.special.digamma(torch.tensor(n_samples, device=device, dtype=dtype))
        digamma_k = torch.special.digamma(torch.tensor(n_neighbors, device=device, dtype=dtype))
        digamma_nx = torch.special.digamma(n_x.clamp(min=1))
        digamma_ny = torch.special.digamma(n_y.clamp(min=1))
        mi_out[j] = (digamma_n + digamma_k - digamma_nx - digamma_ny).mean()
    return mi_out


class MutualInfoRegression(MLModule):
    """Instantiable score function for mutual_info_regression."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.device = kwargs.get("device", "cpu")
        self.dtype = kwargs.get("dtype", torch.float)

    def __new__(cls, X=None, y=None, *args, **kwargs):
        if X is not None and y is not None:
            return _mutual_info_regression_impl(X, y, *args, **kwargs)
        return super().__new__(cls)

    def fit(self, X=None, y=None, **kwargs):
        """No-op fit for API compatibility (e.g. save/load tests)."""
        return self


mutual_info_regression = MutualInfoRegression


class RFE(MLModule):
    def __init__(self,
                 estimator: MLModule = None,
                 n_features_to_select: Union[int, float] = None,
                 step: Union[int, float] = 1,
                 verbose: int = 0,
                 importance_getter: Union[Literal["auto"], Callable, nn.Module] = "auto",
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        if estimator is None:
            from ..classification.tree_models.trees import DecisionTreeClassifier
            estimator = DecisionTreeClassifier(device=device, dtype=dtype)
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.verbose = verbose
        self.importance_getter = importance_getter
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.args = args
        self.kwargs = kwargs
        # Fitted attributes (set in fit)
        self.classes_ = None
        self.estimator_ = None
        self.n_features_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None
        self.ranking_ = None
        self.support_ = None
        self.fit_status = False

    def _get_importance_getter(self) -> Callable:
        """Return a callable that extracts feature importance from a fitted estimator."""
        if self.importance_getter == "auto":
            def _auto_getter(est):
                if hasattr(est, "coef_"):
                    coef = getattr(est, "coef_")
                    if callable(coef):
                        coef = coef()
                    coef = torch.as_tensor(coef, device=self.device, dtype=self.dtype)
                    if coef.dim() == 1:
                        return torch.abs(coef)
                    return torch.sum(torch.abs(coef), dim=0)
                if hasattr(est, "feature_importances_"):
                    imp = getattr(est, "feature_importances_")
                    if callable(imp):
                        imp = imp()
                    return torch.as_tensor(imp, device=self.device, dtype=self.dtype)
                raise ValueError(
                    "When importance_getter='auto', the estimator must have "
                    "coef_ or feature_importances_ attribute."
                )
            return _auto_getter
        if isinstance(self.importance_getter, str):
            getter = operator.attrgetter(self.importance_getter)
            def _attr_getter(est):
                val = getter(est)
                if callable(val):
                    val = val()
                val = torch.as_tensor(val, device=self.device, dtype=self.dtype)
                if val.dim() > 1:
                    val = torch.sum(torch.abs(val), dim=0)
                else:
                    val = torch.abs(val)
                return val
            return _attr_getter
        if callable(self.importance_getter):
            def _callable_getter(est):
                val = self.importance_getter(est)
                return torch.as_tensor(val, device=self.device, dtype=self.dtype)
            return _callable_getter
        if isinstance(self.importance_getter, nn.Module) and hasattr(self.importance_getter, '__call__'):
            def _module_getter(est):
                val = self.importance_getter(est)
                return torch.as_tensor(val, device=self.device, dtype=self.dtype)
            return _module_getter
        raise ValueError(
            "importance_getter must be 'auto', a string (attribute path), a callable, or nn.Module."
        )

    def _get_feature_importances(self, estimator: MLModule) -> torch.Tensor:
        """Extract feature importances from the fitted estimator."""
        getter = self._get_importance_getter()
        imp = getter(estimator)
        imp = torch.as_tensor(imp, device=self.device, dtype=self.dtype)
        if imp.dim() > 1:
            imp = torch.sum(torch.abs(imp), dim=0)
        return imp.flatten()

    def fit(
        self,
        X: Union[torch.Tensor, Any],
        y: Optional[Union[torch.Tensor, Any]] = None,
        **kwargs,
    ) -> "RFE":
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        n_samples, n_features = X.shape

        if y is not None:
            y = torch.as_tensor(y, device=self.device, dtype=self.dtype)
            if y.dim() == 1:
                y = y.unsqueeze(1)

        if hasattr(X, 'columns') and hasattr(X.columns, '__iter__'):
            self.feature_names_in_ = torch.tensor(
                list(X.columns), device=self.device
            ) if not all(isinstance(c, str) for c in X.columns) else None
        else:
            self.feature_names_in_ = None

        self.n_features_in_ = n_features

        n_features_to_select = self.n_features_to_select
        if n_features_to_select is None:
            n_features_to_select = max(1, n_features // 2)
        elif isinstance(n_features_to_select, int):
            n_features_to_select = max(1, min(n_features_to_select, n_features))
        elif isinstance(n_features_to_select, float):
            if 0.0 < n_features_to_select < 1.0:
                n_features_to_select = max(1, int(n_features * n_features_to_select))
            elif n_features_to_select == 1.0:
                n_features_to_select = n_features
            else:
                raise ValueError(
                    "n_features_to_select as float must be in (0, 1]."
                )
        else:
            raise ValueError("n_features_to_select must be int, float in (0,1], or None.")

        step = self.step
        if isinstance(step, int) and step >= 1:
            step = int(step)
        elif isinstance(step, float) and 0.0 < step < 1.0:
            step = max(1, int(n_features * step))
        else:
            step = 1

        ranking = torch.ones(n_features, device=self.device, dtype=torch.long)
        support = torch.ones(n_features, device=self.device, dtype=torch.bool)
        n_selected = n_features
        rank_counter = 2  # Eliminated features get 2, 3, 4, ...

        try:
            estimator = copy.deepcopy(self.estimator)
        except Exception:
            estimator = self.estimator
        if hasattr(estimator, 'device'):
            estimator.device = self.device
        if hasattr(estimator, 'dtype'):
            estimator.dtype = self.dtype

        get_importance = self._get_importance_getter()

        while n_selected > n_features_to_select:
            X_subset = X[:, support]
            if y is not None:
                estimator.fit(X_subset, y, **kwargs)
            else:
                estimator.fit(X_subset, **kwargs)

            importances = get_importance(estimator)
            importances = torch.as_tensor(importances, device=self.device, dtype=self.dtype).flatten()
            if importances.shape[0] > n_selected:
                importances = importances[:n_selected]
            elif importances.shape[0] != n_selected:
                raise ValueError(
                    f"importance_getter returned {importances.shape[0]} values "
                    f"but {n_selected} features are currently selected."
                )

            indices = torch.argsort(importances)
            n_remove = min(step, n_selected - n_features_to_select)
            if n_remove <= 0:
                break

            least_important = indices[:n_remove]
            selected_indices = torch.where(support)[0]
            to_remove_global = selected_indices[least_important]

            support[to_remove_global] = False
            n_selected -= n_remove
            ranking[to_remove_global] = rank_counter
            rank_counter += 1

            if self.verbose > 0:
                print(f"RFE iteration: {n_selected} features selected.")

        # Selected features get rank 1
        self.ranking_ = ranking.clone()
        self.ranking_[support] = 1
        self.support_ = support
        self.n_features_ = int(support.sum().item())
        self.estimator_ = estimator
        X_final = X[:, support]
        if y is not None:
            self.estimator_.fit(X_final, y, **kwargs)
        else:
            self.estimator_.fit(X_final, **kwargs)

        if hasattr(self.estimator_, "classes_"):
            self.classes_ = getattr(self.estimator_, "classes_")
        self.fit_status = True
        return self

    def transform(self, X: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Reduce X to the selected features."""
        if not self.fit_status:
            raise RuntimeError("RFE instance is not fitted yet. Call fit() first.")
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        return X[:, self.support_]

    def fit_transform(
        self,
        X: Union[torch.Tensor, Any],
        y: Optional[Union[torch.Tensor, Any]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Fit RFE and transform X to the selected features."""
        return self.fit(X, y, **kwargs).transform(X)

    def get_support(self, indices: bool = False) -> Union[torch.Tensor, List[int]]:
        if not self.fit_status:
            raise RuntimeError("RFE instance is not fitted yet. Call fit() first.")
        if indices:
            return torch.where(self.support_)[0].tolist()
        return self.support_.clone()

    def predict(self, X: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Predict using the estimator on the selected features."""
        if not self.fit_status:
            raise RuntimeError("RFE instance is not fitted yet. Call fit() first.")
        X_transformed = self.transform(X)
        return self.estimator_.predict(X_transformed)

    def predict_proba(self, X: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Predict class probabilities if the estimator supports it."""
        if not self.fit_status or self.estimator_ is None:
            raise RuntimeError("RFE instance is not fitted yet. Call fit() first.")
        X_trans = self.transform(X)
        if hasattr(self.estimator_, "predict_proba"):
            return self.estimator_.predict_proba(X_trans)
        raise AttributeError(
            f"Estimator {type(self.estimator_).__name__} does not have predict_proba."
        )

    def decision_function(self, X: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Decision function if the estimator supports it."""
        if not self.fit_status or self.estimator_ is None:
            raise RuntimeError("RFE instance is not fitted yet. Call fit() first.")
        X_trans = self.transform(X)
        if hasattr(self.estimator_, "decision_function"):
            return self.estimator_.decision_function(X_trans)
        raise AttributeError(
            f"Estimator {type(self.estimator_).__name__} does not have decision_function."
        )

    def forward(self, X: torch.Tensor, y: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if not self.fit_status and y is not None:
            self.fit(X, y, **kwargs)
        elif not self.fit_status:
            self.fit(X, **kwargs)
        return self.predict(X)


class RFECV(RFE):
    def __init__(self,
                 estimator: MLModule = None,
                 n_features_to_select: Union[int, float] = None,
                 min_features_to_select: int = 1,
                 cv: Union[str, int, Callable, MLModule] = None,
                 cv_config: dict = None,
                 scoring: Union[str, Callable, nn.Module] = None,
                 step: Union[int, float] = 1,
                 verbose: int = 0,
                 importance_getter: Union[Literal["auto"], Callable, nn.Module] = "auto",
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(
            estimator=estimator,
            n_features_to_select=n_features_to_select,
            step=step,
            verbose=verbose,
            importance_getter=importance_getter,
            device=device,
            dtype=dtype,
            *args, **kwargs
        )
        self.min_features_to_select = min_features_to_select
        self.cv = cv
        self.cv_config = cv_config if cv_config is not None else {}
        self.scoring = scoring
        self.cv_results_ = None

    def _check_cv(self, cv, y=None):
        """Resolve cv to a splitter object (from splitters)."""
        from ....models.machine_learning.cross_validation.splitters import (
            KFoldCV,
            StratifiedKFold,
            CVSplitManager,
        )
        cv_config = self.cv_config.copy()
        if cv is None:
            # Default 5-fold; use StratifiedKFold if classifier and y has classes
            if y is not None and hasattr(self.estimator, "estimator_type") and self.estimator.estimator_type == "classifier":
                try:
                    y_flat = y.flatten() if y.dim() > 1 else y
                    if y_flat.numel() > 0 and len(torch.unique(y_flat)) >= 2:
                        return StratifiedKFold(n_splits=5, **cv_config)
                except Exception:
                    pass
            return KFoldCV(n_splits=5, **cv_config)
        if isinstance(cv, int):
            cv_config.setdefault("n_splits", cv)
            return KFoldCV(**cv_config)
        if isinstance(cv, str):
            dev = str(self.device) if hasattr(self.device, "type") else self.device
            return CVSplitManager(splitter=cv, cv_config=cv_config, device=dev, dtype=self.dtype)
        if isinstance(cv, MLModule) and hasattr(cv, "split"):
            return cv
        if callable(cv):
            dev = str(self.device) if hasattr(self.device, "type") else self.device
            return CVSplitManager(splitter=cv, cv_config=cv_config, device=dev, dtype=self.dtype)
        return KFoldCV(n_splits=5, **cv_config)

    def _get_scorer(self):
        """Return scorer callable(estimator, X, y) -> float."""
        if self.scoring is None:
            return lambda est, X, y: est.score(X, y) if hasattr(est, "score") else 0.0
        if callable(self.scoring):
            return self.scoring
        if isinstance(self.scoring, nn.Module) and hasattr(self.scoring, "__call__"):
            return lambda est, X, y: self.scoring(est, X, y)
        # String scoring (from search_cv pattern)
        s = str(self.scoring).lower()
        if s in ("r2", "r_squared"):
            def _r2(est, X, y):
                y_pred = est.predict(X)
                if y.dim() == 1:
                    y = y.unsqueeze(1)
                ss_res = torch.sum((y - y_pred) ** 2)
                ss_tot = torch.sum((y - y.mean()) ** 2)
                return (1 - ss_res / ss_tot.clamp(min=1e-10)).item()
            return _r2
        if s in ("mse", "neg_mean_squared_error", "mean_squared_error"):
            return lambda est, X, y: -F.mse_loss(est.predict(X), y).item()
        if s in ("mae", "neg_mean_absolute_error", "mean_absolute_error"):
            return lambda est, X, y: -F.l1_loss(est.predict(X), y).item()
        if s in ("accuracy", "acc"):
            def _acc(est, X, y):
                if X.numel() == 0 or (X.dim() > 1 and X.shape[1] == 0):
                    return 0.0
                y_pred = est.predict(X)
                if y_pred.dim() > 1:
                    if y_pred.shape[1] == 0:
                        return 0.0
                    y_pred = torch.argmax(y_pred, dim=1)
                return (y_pred == y.flatten()).float().mean().item()
            return _acc
        return lambda est, X, y: est.score(X, y) if hasattr(est, "score") else 0.0

    def fit(
        self,
        X: Union[torch.Tensor, Any],
        y: Optional[Union[torch.Tensor, Any]] = None,
        groups: Optional[Union[torch.Tensor, Any]] = None,
        **kwargs,
    ) -> "RFECV":
        """Fit RFECV: cross-validate over n_features, pick best, fit final RFE."""
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        n_samples, n_features = X.shape

        if y is None:
            raise ValueError("RFECV requires y for supervised cross-validation.")
        y = torch.as_tensor(y, device=self.device, dtype=self.dtype)
        if y.dim() == 1:
            y = y.unsqueeze(1)

        if hasattr(X, "columns") and hasattr(X.columns, "__iter__"):
            self.feature_names_in_ = list(X.columns) if all(isinstance(c, str) for c in X.columns) else None
        else:
            self.feature_names_in_ = None
        self.n_features_in_ = n_features

        cv_splitter = self._check_cv(self.cv, y)
        splits = list(cv_splitter.split(X, y, groups))
        n_splits = len(splits)

        min_f = max(1, min(self.min_features_to_select, n_features))
        step = self.step
        if isinstance(step, float) and 0.0 < step < 1.0:
            step = max(1, int(n_features * step))
        step = max(1, int(step))

        n_features_list = []
        n_cur = n_features
        while n_cur >= min_f:
            n_features_list.append(n_cur)
            if n_cur <= min_f:
                break
            n_cur = max(min_f, n_cur - step)
        n_features_list = sorted(set(n_features_list))

        scorer = self._get_scorer()

        cv_results = {
            "n_features": n_features_list,
            "mean_test_score": [],
            "std_test_score": [],
        }
        for k in range(n_splits):
            cv_results[f"split{k}_test_score"] = []

        for n_f in n_features_list:
            fold_scores = []
            for k, (train_idx, test_idx) in enumerate(splits):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                rfe = RFE(
                    estimator=copy.deepcopy(self.estimator),
                    n_features_to_select=n_f,
                    step=self.step,
                    verbose=0,
                    importance_getter=self.importance_getter,
                    device=self.device,
                    dtype=self.dtype,
                )
                rfe.fit(X_train, y_train, **kwargs)

                X_test_trans = rfe.transform(X_test)
                # Guard: empty transform output can cause scorer to fail (e.g. argmax dim=0)
                if X_test_trans.numel() == 0 or (X_test_trans.dim() > 1 and X_test_trans.shape[1] == 0):
                    sc = 0.0
                else:
                    sc = scorer(rfe.estimator_, X_test_trans, y_test)
                if isinstance(sc, torch.Tensor):
                    sc = sc.item()
                fold_scores.append(sc)
                cv_results[f"split{k}_test_score"].append(sc)

            scores_t = torch.tensor(fold_scores, device=self.device, dtype=self.dtype)
            cv_results["mean_test_score"].append(scores_t.mean().item())
            std_val = scores_t.std().item() if len(fold_scores) > 1 else 0.0
            cv_results["std_test_score"].append(std_val)

        self.cv_results_ = cv_results
        scores_tensor = torch.tensor(cv_results["mean_test_score"], device=self.device, dtype=self.dtype)
        if scores_tensor.dim() == 0:
            best_idx = 0
        else:
            best_idx = int(torch.argmax(scores_tensor).item())
        best_n_features = n_features_list[best_idx]

        self.n_features_to_select = best_n_features
        super().fit(X, y, **kwargs)

        if self.verbose > 0:
            print(f"RFECV selected {best_n_features} features (best CV score).")
        return self


class GenericUnivariateSelect(MLModule):
    def __init__(self,
                 score_func: Union[Callable, nn.Module] = None,
                 mode: Literal["percentile", "k_best", "fpr", "fdr", "fwe"] = "percentile",
                 param: Union[int, float, Literal["all"]] = 1e-05,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        if score_func is None:
            score_func = f_classif
        self.score_func = self._resolve_funcs(score_func, *args, **kwargs)
        self.mode = mode
        self.param = param
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.scores_ = None
        self.pvalues_ = None
        self.n_features_in_ = None
        self.support_ = None
        self.fit_status = False

    def _resolve_funcs(self, score_func: Union[Callable, nn.Module], *args, **kwargs) -> Callable:
        """Resolve score_func to a callable returning (scores, pvalues) or (scores,)."""
        if score_func is None:
            def _default(X, y):
                return f_classif(X, y, *args, **kwargs)
            return _default
        if callable(score_func):
            def _wrap(X, y):
                out = score_func(X, y, *args, **kwargs)
                if isinstance(out, tuple):
                    return out
                return out, None
            return _wrap
        if isinstance(score_func, nn.Module) and hasattr(score_func, "__call__"):
            def _wrap_mod(X, y):
                out = score_func(X, y)
                if isinstance(out, tuple):
                    return out
                return out, None
            return _wrap_mod
        raise ValueError("score_func must be callable or nn.Module.")

    def fit(
        self,
        X: Union[torch.Tensor, Any],
        y: Optional[Union[torch.Tensor, Any]] = None,
        **kwargs,
    ) -> "GenericUnivariateSelect":
        """Fit the selector on X and y."""
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        if y is not None:
            y = torch.as_tensor(y, device=self.device, dtype=self.dtype)
            if y.dim() > 1:
                y = y.flatten()
            scores, pvalues = self.score_func(X, y)
        else:
            if self.mode not in ("percentile", "k_best"):
                raise ValueError("y is required for modes 'fpr', 'fdr', 'fwe'.")
            scores = torch.var(X, dim=0)
            pvalues = None

        scores, pvalues = (scores, pvalues) if y is not None else (scores, pvalues)
        scores = torch.as_tensor(scores, device=self.device, dtype=self.dtype)
        self.scores_ = scores
        self.pvalues_ = torch.as_tensor(pvalues, device=self.device, dtype=self.dtype) if pvalues is not None else None

        support = torch.ones(n_features, device=self.device, dtype=torch.bool)
        param = self.param
        mode = self.mode

        if mode == "k_best":
            if param == "all":
                pass
            else:
                k = int(param) if isinstance(param, (int, float)) else n_features
                k = min(max(1, k), n_features)
                _, top_indices = torch.topk(scores, k, largest=True)
                support = torch.zeros(n_features, device=self.device, dtype=torch.bool)
                support[top_indices] = True

        elif mode == "percentile":
            if param == "all":
                pass
            else:
                pct = float(param)
                pct = max(0.0, min(100.0, pct))
                k = max(1, int(n_features * pct / 100.0))
                _, top_indices = torch.topk(scores, k, largest=True)
                support = torch.zeros(n_features, device=self.device, dtype=torch.bool)
                support[top_indices] = True

        elif mode == "fpr":
            if self.pvalues_ is None:
                raise ValueError("score_func must return p-values for mode 'fpr'.")
            alpha = float(param)
            support = self.pvalues_ < alpha

        elif mode == "fdr":
            if self.pvalues_ is None:
                raise ValueError("score_func must return p-values for mode 'fdr'.")
            alpha = float(param)
            pv = self.pvalues_.clone()
            pv_sorted, order = torch.sort(pv)
            n = len(pv)
            crit = alpha * torch.arange(1, n + 1, device=self.device, dtype=self.dtype) / n
            below = pv_sorted <= crit
            if below.any():
                n_reject = below.nonzero(as_tuple=True)[0][-1].item() + 1
                thresh = pv_sorted[n_reject - 1]
                support = pv <= thresh
            else:
                support = torch.zeros(n_features, device=self.device, dtype=torch.bool)

        elif mode == "fwe":
            if self.pvalues_ is None:
                raise ValueError("score_func must return p-values for mode 'fwe'.")
            alpha = float(param)
            n = len(self.pvalues_)
            thresh = alpha / max(n, 1)
            support = self.pvalues_ < thresh

        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.support_ = support
        self.fit_status = True
        return self

    def transform(self, X: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Reduce X to the selected features."""
        if not self.fit_status:
            raise RuntimeError("GenericUnivariateSelect is not fitted. Call fit() first.")
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        return X[:, self.support_]

    def fit_transform(
        self,
        X: Union[torch.Tensor, Any],
        y: Optional[Union[torch.Tensor, Any]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Fit and transform X."""
        return self.fit(X, y, **kwargs).transform(X)

    def get_support(self, indices: bool = False) -> Union[torch.Tensor, List[int]]:
        """Get the mask or indices of selected features."""
        if not self.fit_status:
            raise RuntimeError("GenericUnivariateSelect is not fitted. Call fit() first.")
        if indices:
            return torch.where(self.support_)[0].tolist()
        return self.support_.clone()


class SelectFdr(GenericUnivariateSelect):
    def __init__(self,
                 score_func: Union[Callable, nn.Module] = None,
                 alpha: float = 0.05,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(
            score_func=score_func,
            mode="fdr",
            param=alpha,
            device=device,
            dtype=dtype,
            *args, **kwargs
        )


class SelectFpr(GenericUnivariateSelect):
    def __init__(self,
                 score_func: Union[Callable, nn.Module] = None,
                 alpha: float = 0.05,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(
            score_func=score_func,
            mode="fpr",
            param=alpha,
            device=device,
            dtype=dtype,
            *args, **kwargs
        )


class SelectFwe(GenericUnivariateSelect):
    def __init__(self,
                 score_func: Union[Callable, nn.Module] = None,
                 alpha: float = 0.05,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(
            score_func=score_func,
            mode="fwe",
            param=alpha,
            device=device,
            dtype=dtype,
            *args, **kwargs
        )


class SelectKBest(GenericUnivariateSelect):
    def __init__(self,
                 score_func: Union[Callable, nn.Module] = None,
                 k: Union[int, Literal["all"]] = 10,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(
            score_func=score_func,
            mode="k_best",
            param=k,
            device=device,
            dtype=dtype,
            *args, **kwargs
        )


class SelectPercentile(GenericUnivariateSelect):
    def __init__(self,
                 score_func: Union[Callable, nn.Module] = None,
                 percentile: int = 10,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(
            score_func=score_func,
            mode="percentile",
            param=percentile,
            device=device,
            dtype=dtype,
            *args, **kwargs
        )


class SelectFromModel(MLModule):
    def __init__(self,
                 estimator: MLModule = None,
                 threshold: Union[str, float] = None,
                 prefit: bool = False,
                 norm_order: Union[int, float] = 1,
                 max_features: Union[int, Callable] = None,
                 importance_getter: Union[Literal["auto"], Callable, nn.Module] = "auto",
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        if estimator is None:
            from ..classification.tree_models.trees import DecisionTreeClassifier
            estimator = DecisionTreeClassifier(device=device, dtype=dtype)
        self.estimator = estimator
        self.threshold = threshold
        self.prefit = prefit
        self.norm_order = norm_order
        self.max_features = max_features
        self.importance_getter = importance_getter
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.args = args
        self.kwargs = kwargs
        # Fitted attributes (set in fit)
        self.estimator_ = None
        self.n_features_in_ = None
        self.max_features_ = None
        self.threshold_ = None
        self.support_ = None
        self.fit_status = False

    def _get_importance_getter(self) -> Callable:
        """Return a callable that extracts feature importance from a fitted estimator."""
        if self.importance_getter == "auto":
            def _auto_getter(est):
                if hasattr(est, "coef_"):
                    coef = getattr(est, "coef_")
                    if callable(coef):
                        coef = coef()
                    coef = torch.as_tensor(coef, device=self.device, dtype=self.dtype)
                    if coef.dim() == 1:
                        return torch.abs(coef)
                    if self.norm_order == 1:
                        return torch.sum(torch.abs(coef), dim=0)
                    if self.norm_order == float('inf') or self.norm_order == math.inf:
                        return torch.max(torch.abs(coef), dim=0)[0]
                    if self.norm_order == float('-inf') or self.norm_order == -math.inf:
                        return torch.min(torch.abs(coef), dim=0)[0]
                    return torch.sum(torch.abs(coef) ** self.norm_order, dim=0) ** (1.0 / self.norm_order)
                if hasattr(est, "feature_importances_"):
                    imp = getattr(est, "feature_importances_")
                    if callable(imp):
                        imp = imp()
                    return torch.as_tensor(imp, device=self.device, dtype=self.dtype)
                raise ValueError(
                    "When importance_getter='auto', the estimator must have "
                    "coef_ or feature_importances_ attribute."
                )
            return _auto_getter
        if isinstance(self.importance_getter, str):
            getter = operator.attrgetter(self.importance_getter)
            def _attr_getter(est):
                val = getter(est)
                if callable(val):
                    val = val()
                val = torch.as_tensor(val, device=self.device, dtype=self.dtype)
                if val.dim() > 1:
                    if self.norm_order == 1:
                        val = torch.sum(torch.abs(val), dim=0)
                    elif self.norm_order == float('inf') or self.norm_order == math.inf:
                        val = torch.max(torch.abs(val), dim=0)[0]
                    elif self.norm_order == float('-inf') or self.norm_order == -math.inf:
                        val = torch.min(torch.abs(val), dim=0)[0]
                    else:
                        val = torch.sum(torch.abs(val) ** self.norm_order, dim=0) ** (1.0 / self.norm_order)
                else:
                    val = torch.abs(val)
                return val
            return _attr_getter
        if callable(self.importance_getter):
            def _callable_getter(est):
                val = self.importance_getter(est)
                return torch.as_tensor(val, device=self.device, dtype=self.dtype)
            return _callable_getter
        if isinstance(self.importance_getter, nn.Module) and hasattr(self.importance_getter, '__call__'):
            def _module_getter(est):
                val = self.importance_getter(est)
                return torch.as_tensor(val, device=self.device, dtype=self.dtype)
            return _module_getter
        raise ValueError(
            "importance_getter must be 'auto', a string (attribute path), a callable, or nn.Module."
        )

    def _get_feature_importances(self, estimator: MLModule) -> torch.Tensor:
        """Extract feature importances from the fitted estimator."""
        getter = self._get_importance_getter()
        imp = getter(estimator)
        imp = torch.as_tensor(imp, device=self.device, dtype=self.dtype)
        if imp.dim() > 1:
            if self.norm_order == 1:
                imp = torch.sum(torch.abs(imp), dim=0)
            elif self.norm_order == float('inf') or self.norm_order == math.inf:
                imp = torch.max(torch.abs(imp), dim=0)[0]
            elif self.norm_order == float('-inf') or self.norm_order == -math.inf:
                imp = torch.min(torch.abs(imp), dim=0)[0]
            else:
                imp = torch.sum(torch.abs(imp) ** self.norm_order, dim=0) ** (1.0 / self.norm_order)
        else:
            imp = torch.abs(imp)
        return imp.flatten()

    def _parse_threshold(self, threshold_spec: Union[str, float, None], importances: torch.Tensor) -> float:
        """Parse threshold string (e.g. 'mean', 'median', '1.25*mean') to float."""
        if threshold_spec is None:
            return float(importances.mean().item())
        if isinstance(threshold_spec, (int, float)):
            return float(threshold_spec)
        s = str(threshold_spec).strip().lower()
        if s == "median":
            return float(importances.median().item())
        if s == "mean":
            return float(importances.mean().item())
        import re
        m = re.match(r"([\d.]+)\s*\*\s*(mean|median)", s)
        if m:
            factor = float(m.group(1))
            agg = m.group(2)
            if agg == "mean":
                base = importances.mean().item()
            else:
                base = importances.median().item()
            return factor * base
        try:
            return float(threshold_spec)
        except (ValueError, TypeError):
            return float(importances.mean().item())

    def _check_estimator_has_l1(self, estimator: MLModule) -> bool:
        """Check if estimator uses L1 penalty (for default threshold)."""
        for attr in ("penalty", "l1_ratio", "alpha"):
            if hasattr(estimator, attr):
                val = getattr(estimator, attr)
                if isinstance(val, str) and "l1" in val.lower():
                    return True
                if attr == "penalty" and val == "l1":
                    return True
        return False

    def fit(
        self,
        X: Union[torch.Tensor, Any],
        y: Optional[Union[torch.Tensor, Any]] = None,
        **kwargs,
    ) -> "SelectFromModel":
        """Fit the SelectFromModel meta-transformer."""
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        n_samples, n_features = X.shape

        if y is not None:
            y = torch.as_tensor(y, device=self.device, dtype=self.dtype)
            if y.dim() == 1:
                y = y.unsqueeze(1)

        if hasattr(X, 'columns') and hasattr(X.columns, '__iter__'):
            self.feature_names_in_ = list(X.columns) if all(isinstance(c, str) for c in X.columns) else None
        else:
            self.feature_names_in_ = None
        self.n_features_in_ = n_features

        if self.prefit:
            try:
                self.estimator_ = copy.deepcopy(self.estimator)
            except Exception:
                self.estimator_ = self.estimator
        else:
            try:
                self.estimator_ = copy.deepcopy(self.estimator)
            except Exception:
                self.estimator_ = self.estimator
            if hasattr(self.estimator_, 'device'):
                self.estimator_.device = self.device
            if hasattr(self.estimator_, 'dtype'):
                self.estimator_.dtype = self.dtype
            if y is not None:
                self.estimator_.fit(X, y, **{**self.kwargs, **kwargs})
            else:
                self.estimator_.fit(X, **{**self.kwargs, **kwargs})

        importances = self._get_feature_importances(self.estimator_)

        thresh_spec = self.threshold
        if thresh_spec is None:
            if self._check_estimator_has_l1(self.estimator_):
                thresh_spec = 1e-5
            else:
                thresh_spec = "mean"
        self.threshold_ = self._parse_threshold(thresh_spec, importances)

        max_f = self.max_features
        if max_f is not None:
            if callable(max_f):
                self.max_features_ = int(max_f(X))
            else:
                self.max_features_ = int(max_f)
            self.max_features_ = max(1, min(self.max_features_, n_features))
        else:
            self.max_features_ = None

        support = importances >= self.threshold_
        if self.max_features_ is not None:
            n_selected = int(support.sum().item())
            if n_selected > self.max_features_:
                top_k_indices = torch.argsort(importances, descending=True)[:self.max_features_]
                support = torch.zeros(n_features, device=self.device, dtype=torch.bool)
                support[top_k_indices] = True

        self.support_ = support
        self.fit_status = True
        return self

    def partial_fit(
        self,
        X: Union[torch.Tensor, Any],
        y: Optional[Union[torch.Tensor, Any]] = None,
        **kwargs,
    ) -> "SelectFromModel":
        """Update the model with a partial fit (only when prefit=False)."""
        if self.prefit:
            raise ValueError("partial_fit is not supported when prefit=True.")
        if not self.fit_status:
            return self.fit(X, y, **kwargs)
        if not hasattr(self.estimator_, "partial_fit"):
            raise ValueError(
                "partial_fit is not supported when the estimator does not have partial_fit."
            )
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        if y is not None:
            y = torch.as_tensor(y, device=self.device, dtype=self.dtype)
            if y.dim() == 1:
                y = y.unsqueeze(1)
        if y is not None:
            self.estimator_.partial_fit(X, y, **{**self.kwargs, **kwargs})
        else:
            self.estimator_.partial_fit(X, **{**self.kwargs, **kwargs})
        importances = self._get_feature_importances(self.estimator_)
        thresh_spec = self.threshold or "mean"
        self.threshold_ = self._parse_threshold(thresh_spec, importances)
        self.support_ = importances >= self.threshold_
        if self.max_features_ is not None:
            n_features = importances.shape[0]
            n_selected = int(self.support_.sum().item())
            if n_selected > self.max_features_:
                top_k_indices = torch.argsort(importances, descending=True)[:self.max_features_]
                self.support_ = torch.zeros(n_features, device=self.device, dtype=torch.bool)
                self.support_[top_k_indices] = True
        return self

    def transform(self, X: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Reduce X to the selected features."""
        if not self.fit_status:
            raise RuntimeError("SelectFromModel is not fitted. Call fit() first.")
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        return X[:, self.support_]

    def fit_transform(
        self,
        X: Union[torch.Tensor, Any],
        y: Optional[Union[torch.Tensor, Any]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Fit and transform X."""
        return self.fit(X, y, **kwargs).transform(X)

    def get_support(self, indices: bool = False) -> Union[torch.Tensor, List[int]]:
        """Get the mask or indices of selected features."""
        if not self.fit_status:
            raise RuntimeError("SelectFromModel is not fitted. Call fit() first.")
        if indices:
            return torch.where(self.support_)[0].tolist()
        return self.support_.clone()

    def predict(self, X: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Predict using the estimator on the selected features."""
        if not self.fit_status:
            raise RuntimeError("SelectFromModel is not fitted. Call fit() first.")
        X_trans = self.transform(X)
        return self.estimator_.predict(X_trans)

    def predict_proba(self, X: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Predict class probabilities if the estimator supports it."""
        if not self.fit_status or self.estimator_ is None:
            raise RuntimeError("SelectFromModel is not fitted. Call fit() first.")
        X_trans = self.transform(X)
        if hasattr(self.estimator_, "predict_proba"):
            return self.estimator_.predict_proba(X_trans)
        raise AttributeError(
            f"Estimator {type(self.estimator_).__name__} does not have predict_proba."
        )

    def decision_function(self, X: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Decision function if the estimator supports it."""
        if not self.fit_status or self.estimator_ is None:
            raise RuntimeError("SelectFromModel is not fitted. Call fit() first.")
        X_trans = self.transform(X)
        if hasattr(self.estimator_, "decision_function"):
            return self.estimator_.decision_function(X_trans)
        raise AttributeError(
            f"Estimator {type(self.estimator_).__name__} does not have decision_function."
        )

    def forward(self, X: torch.Tensor, y: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if not self.fit_status and y is not None:
            self.fit(X, y, **kwargs)
        elif not self.fit_status:
            self.fit(X, **kwargs)
        return self.predict(X)


class SequentialFeatureSelector(MLModule):
    def __init__(self,
                 estimator: MLModule = None,
                 n_features_to_select: Union[Literal["auto"], int, float] = "auto",
                 tol: float = None,
                 direction: Literal["forward", "backward"] = "forward",
                 scoring: Union[str, Callable, nn.Module] = None,
                 cv: Union[int, str, MLModule, Callable] = None,
                 cv_config: dict = None,
                 n_jobs: int = None,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        if estimator is None:
            from ..classification.tree_models.trees import DecisionTreeClassifier
            estimator = DecisionTreeClassifier(device=device, dtype=dtype)
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.tol = tol
        self.direction = direction
        self.scoring = scoring
        self.cv = cv
        self.cv_config = cv_config if cv_config is not None else {}
        self.n_jobs = n_jobs
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.args = args
        self.kwargs = kwargs
        # Fitted attributes (set in fit)
        self.estimator_ = None
        self.n_features_in_ = None
        self.n_features_to_select_ = None
        self.support_ = None
        self.fit_status = False

    def _check_cv(self, cv, y=None):
        """Resolve cv to a splitter object (from splitters)."""
        from ....models.machine_learning.cross_validation.splitters import (
            KFoldCV,
            StratifiedKFold,
            CVSplitManager,
        )
        cv_config = self.cv_config.copy()
        if cv is None:
            cv_config.setdefault("n_splits", 5)
            if y is not None and hasattr(self.estimator, "estimator_type") and getattr(self.estimator, "estimator_type", None) == "classifier":
                try:
                    y_flat = y.flatten() if y.dim() > 1 else y
                    if y_flat.numel() > 0 and len(torch.unique(y_flat)) >= 2:
                        return StratifiedKFold(**cv_config)
                except Exception:
                    pass
            return KFoldCV(**cv_config)
        if isinstance(cv, int):
            cv_config = {**cv_config, "n_splits": cv}
            return KFoldCV(**cv_config)
        if isinstance(cv, str):
            dev = str(self.device) if hasattr(self.device, "type") else self.device
            return CVSplitManager(splitter=cv, cv_config=cv_config, device=dev, dtype=self.dtype)
        if isinstance(cv, MLModule) and hasattr(cv, "split"):
            return cv
        if callable(cv):
            dev = str(self.device) if hasattr(self.device, "type") else self.device
            return CVSplitManager(splitter=cv, cv_config=cv_config, device=dev, dtype=self.dtype)
        return KFoldCV(n_splits=5, **cv_config)

    def _get_scorer(self):
        """Return scorer callable(estimator, X, y) -> float. Handles y=None for unsupervised."""
        if self.scoring is None:
            def _default(est, X, y):
                if y is not None:
                    return est.score(X, y) if hasattr(est, "score") else 0.0
                return est.score(X) if hasattr(est, "score") else 0.0
            return _default
        if callable(self.scoring):
            return self.scoring
        if isinstance(self.scoring, nn.Module) and hasattr(self.scoring, "__call__"):
            return lambda est, X, y: self.scoring(est, X, y)
        s = str(self.scoring).lower()
        if s in ("r2", "r_squared"):
            def _r2(est, X, y):
                if y is None:
                    return est.score(X) if hasattr(est, "score") else 0.0
                y_pred = est.predict(X)
                if y.dim() == 1:
                    y = y.unsqueeze(1)
                ss_res = torch.sum((y - y_pred) ** 2)
                ss_tot = torch.sum((y - y.mean()) ** 2)
                return (1 - ss_res / ss_tot.clamp(min=1e-10)).item()
            return _r2
        if s in ("mse", "neg_mean_squared_error", "mean_squared_error"):
            def _mse(est, X, y):
                if y is None:
                    return est.score(X) if hasattr(est, "score") else 0.0
                return -F.mse_loss(est.predict(X), y).item()
            return _mse
        if s in ("mae", "neg_mean_absolute_error", "mean_absolute_error"):
            def _mae(est, X, y):
                if y is None:
                    return est.score(X) if hasattr(est, "score") else 0.0
                return -F.l1_loss(est.predict(X), y).item()
            return _mae
        if s in ("accuracy", "acc"):
            def _acc(est, X, y):
                if y is None:
                    return est.score(X) if hasattr(est, "score") else 0.0
                y_pred = est.predict(X)
                if y_pred.dim() > 1:
                    y_pred = torch.argmax(y_pred, dim=1)
                return (y_pred == y.flatten()).float().mean().item()
            return _acc
        def _fallback(est, X, y):
            if y is not None:
                return est.score(X, y) if hasattr(est, "score") else 0.0
            return est.score(X) if hasattr(est, "score") else 0.0
        return _fallback

    def _cross_val_score(self, X: torch.Tensor, y: Optional[torch.Tensor], feature_indices: List[int], groups=None, **fit_kwargs) -> float:
        """Compute mean CV score for a given subset of features using cross_validate from functional."""
        if len(feature_indices) == 0:
            return float('-inf') if self.direction == "forward" else 0.0
        X_sub = X[:, feature_indices]
        cv_splitter = self._check_cv(self.cv, y)
        scorer = self._get_scorer()
        est = copy.deepcopy(self.estimator)
        if hasattr(est, 'device'):
            est.device = self.device
        if hasattr(est, 'dtype'):
            est.dtype = self.dtype
        from ....models.machine_learning.cross_validation.functional import cross_validate
        params = {**self.kwargs, **fit_kwargs}
        result = cross_validate(
            est, X_sub, y,
            groups=groups,
            scoring=scorer,
            cv=cv_splitter,
            n_jobs=self.n_jobs,
            params=params if params else None,
            error_score=float('-inf'),
        )
        scores = result['test_score']
        if not scores:
            return float('-inf')
        return sum(scores) / len(scores)

    def fit(
        self,
        X: Union[torch.Tensor, Any],
        y: Optional[Union[torch.Tensor, Any]] = None,
        groups: Optional[Union[torch.Tensor, Any]] = None,
        **kwargs,
    ) -> "SequentialFeatureSelector":
        """Fit the SequentialFeatureSelector."""
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        n_samples, n_features = X.shape

        if y is not None:
            y = torch.as_tensor(y, device=self.device, dtype=self.dtype)
            if y.dim() == 1:
                y = y.unsqueeze(1)

        if hasattr(X, "columns") and hasattr(X.columns, "__iter__"):
            self.feature_names_in_ = list(X.columns) if all(isinstance(c, str) for c in X.columns) else None
        else:
            self.feature_names_in_ = None
        self.n_features_in_ = n_features

        n_sel = self.n_features_to_select
        tol = self.tol
        if n_sel == "auto":
            if tol is not None:
                n_sel = None
            else:
                n_sel = max(1, n_features // 2)

        if n_sel is not None:
            if isinstance(n_sel, int):
                n_target = max(1, min(n_sel, n_features))
            elif isinstance(n_sel, float):
                if 0.0 < n_sel <= 1.0:
                    n_target = max(1, int(n_features * n_sel))
                else:
                    raise ValueError("n_features_to_select as float must be in (0, 1].")
            else:
                n_target = max(1, n_features // 2)
        else:
            n_target = None

        scorer = self._get_scorer()
        cv_splitter = self._check_cv(self.cv, y)
        splits = list(cv_splitter.split(X, y, groups))

        if self.direction == "forward":
            selected = []
            prev_score = float('-inf')
            while True:
                best_score = float('-inf')
                best_j = -1
                for j in range(n_features):
                    if j in selected:
                        continue
                    cand = selected + [j]
                    score = self._cross_val_score(X, y, cand, groups, **kwargs)
                    if score > best_score:
                        best_score = score
                        best_j = j
                if best_j < 0:
                    break
                if n_target is not None and len(selected) >= n_target:
                    break
                if tol is not None and n_sel == "auto":
                    if best_score - prev_score < tol:
                        break
                selected.append(best_j)
                prev_score = best_score
                if n_target is not None and len(selected) >= n_target:
                    break
        else:
            selected = list(range(n_features))
            prev_score = self._cross_val_score(X, y, selected, groups, **kwargs)
            while True:
                best_score = float('-inf')
                best_j = -1
                for j in selected:
                    cand = [s for s in selected if s != j]
                    if len(cand) == 0:
                        continue
                    score = self._cross_val_score(X, y, cand, groups, **kwargs)
                    if score > best_score:
                        best_score = score
                        best_j = j
                if best_j < 0:
                    break
                if n_target is not None and len(selected) <= n_target:
                    break
                if tol is not None and n_sel == "auto":
                    if best_score - prev_score < tol:
                        break
                selected.remove(best_j)
                prev_score = best_score
                if n_target is not None and len(selected) <= n_target:
                    break

        support = torch.zeros(n_features, device=self.device, dtype=torch.bool)
        support[selected] = True
        self.support_ = support
        self.n_features_to_select_ = len(selected)

        try:
            self.estimator_ = copy.deepcopy(self.estimator)
        except Exception:
            self.estimator_ = self.estimator
        if hasattr(self.estimator_, 'device'):
            self.estimator_.device = self.device
        if hasattr(self.estimator_, 'dtype'):
            self.estimator_.dtype = self.dtype
        X_final = X[:, self.support_]
        if y is not None:
            self.estimator_.fit(X_final, y, **{**self.kwargs, **kwargs})
        else:
            self.estimator_.fit(X_final, **{**self.kwargs, **kwargs})

        self.fit_status = True
        return self

    def transform(self, X: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Reduce X to the selected features."""
        if not self.fit_status:
            raise RuntimeError("SequentialFeatureSelector is not fitted. Call fit() first.")
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        return X[:, self.support_]

    def fit_transform(
        self,
        X: Union[torch.Tensor, Any],
        y: Optional[Union[torch.Tensor, Any]] = None,
        groups: Optional[Union[torch.Tensor, Any]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Fit and transform X."""
        return self.fit(X, y, groups=groups, **kwargs).transform(X)

    def get_support(self, indices: bool = False) -> Union[torch.Tensor, List[int]]:
        """Get the mask or indices of selected features."""
        if not self.fit_status:
            raise RuntimeError("SequentialFeatureSelector is not fitted. Call fit() first.")
        if indices:
            return torch.where(self.support_)[0].tolist()
        return self.support_.clone()

    def predict(self, X: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Predict using the estimator on the selected features."""
        if not self.fit_status:
            raise RuntimeError("SequentialFeatureSelector is not fitted. Call fit() first.")
        X_trans = self.transform(X)
        return self.estimator_.predict(X_trans)

    def predict_proba(self, X: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Predict class probabilities if the estimator supports it."""
        if not self.fit_status or self.estimator_ is None:
            raise RuntimeError("SequentialFeatureSelector is not fitted. Call fit() first.")
        X_trans = self.transform(X)
        if hasattr(self.estimator_, "predict_proba"):
            return self.estimator_.predict_proba(X_trans)
        raise AttributeError(
            f"Estimator {type(self.estimator_).__name__} does not have predict_proba."
        )

    def decision_function(self, X: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Decision function if the estimator supports it."""
        if not self.fit_status or self.estimator_ is None:
            raise RuntimeError("SequentialFeatureSelector is not fitted. Call fit() first.")
        X_trans = self.transform(X)
        if hasattr(self.estimator_, "decision_function"):
            return self.estimator_.decision_function(X_trans)
        raise AttributeError(
            f"Estimator {type(self.estimator_).__name__} does not have decision_function."
        )

    def forward(self, X: torch.Tensor, y: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if not self.fit_status and y is not None:
            self.fit(X, y, **kwargs)
        elif not self.fit_status:
            self.fit(X, **kwargs)
        return self.predict(X)


class VarianceThreshold(MLModule):
    def __init__(self,
                 threshold: float = 0.0,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.threshold = threshold
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.args = args
        self.kwargs = kwargs
        # Fitted attributes (set in fit)
        self.variances_ = None
        self.n_features_in_ = None
        self.support_ = None
        self.fit_status = False

    def fit(
        self,
        X: Union[torch.Tensor, Any],
        y: Optional[Union[torch.Tensor, Any]] = None,
        **kwargs,
    ) -> "VarianceThreshold":
        """Fit the VarianceThreshold by computing feature variances."""
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        n_samples, n_features = X.shape

        if hasattr(X, "columns") and hasattr(X.columns, "__iter__"):
            self.feature_names_in_ = list(X.columns) if all(isinstance(c, str) for c in X.columns) else None
        else:
            self.feature_names_in_ = None
        self.n_features_in_ = n_features

        # Compute variance per feature (column); use unbiased estimator (ddof=1) for sample variance
        # sklearn uses default ddof=0 (population variance); we match sklearn for consistency
        self.variances_ = torch.var(X, dim=0, unbiased=False)
        self.support_ = self.variances_ >= self.threshold
        self.fit_status = True
        return self

    def transform(self, X: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Reduce X to the selected features (variance >= threshold)."""
        if not self.fit_status:
            raise RuntimeError("VarianceThreshold is not fitted. Call fit() first.")
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but VarianceThreshold is expecting "
                f"{self.n_features_in_} features as input."
            )
        return X[:, self.support_]

    def fit_transform(
        self,
        X: Union[torch.Tensor, Any],
        y: Optional[Union[torch.Tensor, Any]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Fit and transform X."""
        return self.fit(X, y, **kwargs).transform(X)

    def get_support(self, indices: bool = False) -> Union[torch.Tensor, List[int]]:
        """Get the mask or indices of selected features."""
        if not self.fit_status:
            raise RuntimeError("VarianceThreshold is not fitted. Call fit() first.")
        if indices:
            return torch.where(self.support_)[0].tolist()
        return self.support_.clone()

    def inverse_transform(self, X: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Reverse the transform operation (restore to original feature space)."""
        if not self.fit_status:
            raise RuntimeError("VarianceThreshold is not fitted. Call fit() first.")
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        n_selected = int(self.support_.sum().item())
        if X.shape[1] != n_selected:
            raise ValueError(
                f"X has {X.shape[1]} features, but VarianceThreshold selected {n_selected}."
            )
        X_full = torch.zeros(X.shape[0], self.n_features_in_, device=self.device, dtype=self.dtype)
        X_full[:, self.support_] = X
        return X_full

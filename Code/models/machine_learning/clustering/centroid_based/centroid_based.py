import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Callable, Union, Any, List, Tuple, Dict
from .....models.utils import MLCluster
from torch.func import vmap
import joblib

__all__ = [
    "BregmanKMeans",
    "KMeansCluster",
]


#
# BregmanKMeans  [Banerjee, Merugu, Dhillon & Ghosh, JMLR 2005]
# Mathematical base-class for all centroid-based clustering models
# 
class BregmanKMeans(MLCluster):
    """
    Bregman K-Means – the mathematical parent of all centroid-based methods.
    [Banerjee, Merugu, Dhillon & Ghosh, JMLR 2005]

    Unifies centroid clustering under a single Bregman-divergence framework.
    By choosing different divergences this class recovers:

        divergence="squared_euclidean"   standard K-Means
        divergence="kl"                  KL-divergence K-Means (for probs)
        divergence="itakura_saito"       IS-clustering (signal processing)
        divergence="beta_2"              scaled Euclidean
        divergence="logistic"            logistic loss clustering
        divergence="exponential"         exponential-family clustering
        divergence="mahalanobis"         Mahalanobis distance clustering
        divergence="custom"              user callable divergence_params["fn"]

    The arithmetic mean is always the optimal centroid for any Bregman divergence,
    so the update step remains the standard mean — only the assignment step changes.

    In addition to the primary Bregman divergence the class also supports a full
    library of **geometric distance metrics** (``metric`` / ``dist_calc``) for
    the Euclidean-path assignment.  When ``divergence`` is None (or "auto"),
    assignment uses the configured ``metric``; otherwise the Bregman divergence
    is used exclusively.

    Parameters
    ----------
    n_clusters : int, default 8
    divergence : str | None, default None
        Bregman divergence to use for assignment.  None  use ``metric``.
        "squared_euclidean" | "kl" | "itakura_saito" | "beta_2" |
        "logistic" | "exponential" | "mahalanobis" | "custom"
    divergence_params : dict, optional
        Extra kwargs for the divergence (e.g. ``{"W": Tensor}`` for mahalanobis,
        ``{"fn": callable}`` for custom).
    eps : float, default 1e-9   Numerical floor for log/ratio operations.
    metric : str | Callable | nn.Module, default "euclidean"
        Geometric distance metric (used when divergence is None).
        Supported string keys — see ``dist_calc`` docstring.
    metric_params : dict, optional   Forwarded to ``dist_calc``.
    init : str | Callable | Tensor | list | tuple, default "k-means++"
        Centroid initialisation.
        "k-means++" | "random" | "farthest" | Tensor | callable
    n_init : int | "auto", default "auto"
        Number of independent runs (best is kept).
    max_iter : int, default 300
    tol : float, default 1e-4   Convergence tolerance on centroid shift.
    verbose : bool, default False
    random_state : int | None
    copy_X : bool, default True
    device : str, default "cpu"
    dtype : torch.dtype, default torch.float

    Attributes
    ----------
    cluster_centers_ : Tensor (k, d)
    labels_ : Tensor (n,)
    inertia_ : float   sum of divergences/distances to assigned centroids
    n_iter_ : int
    """

    def __init__(self,
                 n_clusters: int = 8,
                 divergence: Optional[str] = None,
                 divergence_params: Dict[str, Any] = None,
                 eps: float = 1e-9,
                 metric: Union[str, Callable, "nn.Module"] = "euclidean",
                 metric_params: Dict[str, Any] = None,
                 init: Union[str, Callable, "nn.Module",
                 List, Tuple, torch.Tensor] = "k-means++",
                 n_init: Union[str, int] = "auto",
                 max_iter: int = 300,
                 tol: float = 1e-4,
                 verbose: bool = False,
                 random_state: Optional[int] = None,
                 copy_X: bool = True,
                 warm_start: bool = False,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.warm_start = warm_start
        self.n_clusters = n_clusters
        self.divergence = divergence.lower() if isinstance(divergence, str) else divergence
        self.divergence_params = divergence_params or {}
        self.eps = eps
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = int(verbose)
        self.copy_X = copy_X
        self.device = device
        self.dtype = dtype
        # Centroid outputs
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None
        self.n_features_in_ = None
        # Random state
        if random_state is not None:
            torch.manual_seed(random_state)
            self.random_state = torch.Generator(device=self.device).manual_seed(random_state)
        else:
            self.random_state = None
        # Metric routing — delegates to MLCluster._create_metric;
        # nn.Module metrics are stored directly so they remain callable as self.metric(xi, xj).
        # _metric_name / _metric_callable are kept for backward-compatibility with subclasses
        # that access them directly (e.g. Power-Iteration, KMediansOnGraphs).
        self.metric_params = metric_params or {}
        self._metric_name = metric.lower() if isinstance(metric, str) else None
        self._metric_callable = (
            metric if (callable(metric) and not isinstance(metric, nn.Module)) else None
        )
        if self._metric_name is None and self._metric_callable is None and not isinstance(metric, nn.Module):
            self._metric_name = "euclidean"
        if isinstance(metric, nn.Module):
            self.metric = metric
        else:
            self._create_metric(metric, self.metric_params)

    def _compute_distances(self, X: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """Return (N, K) pairwise distance matrix using the configured metric.

        For nn.Module metrics the module is called directly with (X, centers).
        For string / callable metrics the result is routed through dist_calc.
        """
        if isinstance(self.metric, nn.Module):
            return self.metric(X, centers)
        D = self.metric(X, centers)
        if D.dim() > 2:
            D = D.squeeze()
        return D

    def dist_calc(self, metric_type: str, xi: torch.Tensor, xj: torch.Tensor, **kwargs) -> torch.Tensor:
        """Vectorised pairwise distance between rows of xi (N,d) and xj (K,d).

        Supported metrics
        -----------------
        ``euclidean / l2``  ``manhattan / l1``  ``minkowski``  ``chebyshev``
        ``cosine`` · ``kl_divergence`` · ``js_divergence``
        ``wasserstein / earth_mover`` · ``rbf_distance`` · ``mahalanobis_distance``
        ``canberra_distance`` · ``hellinger_distance`` · ``bhattacharyya_distance``
        ``energy_distance`` · ``total_variational_distance`` · ``frobenius_norm``
        ``log_euclidean`` · ``spectral_norm`` · ``grassmannian_distance``
        ``curvature_based_distance`` · ``normalized_compression_distance``
        ``variation_of_information`` · ``levenshtein_distance``

        sklearn pairwise
        ----------------
        ``cityblock`` · ``sqeuclidean`` · ``seuclidean / standardized_euclidean``

        scipy.spatial.distance
        ----------------------
        ``braycurtis`` · ``correlation`` · ``dice`` · ``hamming`` · ``jaccard``
        ``kulsinski`` · ``rogerstanimoto`` · ``russellrao`` · ``sokalmichener``
        ``sokalsneath`` · ``yule``
        """
        match metric_type.lower():
            case "euclid" | "euclidean" | "l2_distance" | "distance" | "l2":
                return torch.cdist(xi, xj, p=2)
            case "manhattan" | "l1_distance" | "l1":
                return torch.cdist(xi, xj, p=1)
            case "minkowski":
                p = kwargs.get("p", 2)
                return torch.cdist(xi, xj, p=p)
            case "chebyshev":
                return torch.cdist(xi, xj, p=float("inf"))
            case "cosine":
                xi_n = F.normalize(xi, p=2, dim=-1)
                xj_n = F.normalize(xj, p=2, dim=-1)
                return 1.0 - xi_n @ xj_n.T
            case "kl_divergence":
                kl_div = nn.KLDivLoss(reduction="none")
                xi_s = F.softmax(xi, dim=-1).unsqueeze(-2)
                xj_s = F.log_softmax(xj, dim=-1).unsqueeze(-3)
                return kl_div(xj_s.expand(xi_s.shape[0], xj.shape[0], -1),
                              xi_s.expand(xi_s.shape[0], xj.shape[0], -1)).sum(dim=-1)
            case "js_divergence":
                xi_s = F.softmax(xi, dim=-1)
                xj_s = F.softmax(xj, dim=-1)
                xi_e = xi_s.unsqueeze(1);
                xj_e = xj_s.unsqueeze(0)
                xm = 0.5 * (xi_e + xj_e)
                kl_a = (xi_e * (xi_e / (xm + 1e-8) + 1e-8).log()).sum(-1)
                kl_b = (xj_e * (xj_e / (xm + 1e-8) + 1e-8).log()).sum(-1)
                return 0.5 * (kl_a + kl_b)
            case "wasserstein" | "earth_mover":
                num_proj = kwargs.get("num_projections", 32)
                p_ord = kwargs.get("p", 1)
                d = xi.size(-1)
                projs = F.normalize(torch.randn(d, num_proj, device=xi.device, dtype=xi.dtype), dim=0)
                xi_p, _ = torch.sort(xi @ projs, dim=0)
                xj_p, _ = torch.sort(xj @ projs, dim=0)
                n_x, k_x = xi_p.shape[0], xj_p.shape[0]
                if n_x != k_x:
                    idx = torch.linspace(0, k_x - 1, n_x, device=xi.device).long()
                    xj_p = xj_p[idx]
                swd = torch.pow(torch.abs(xi_p - xj_p), p_ord).mean(dim=0).mean()
                return torch.cdist(xi, xj, p=2) * 0 + swd
            case "rbf_distance":
                gamma = math.fabs(kwargs.get("gamma", 0.1))
                return torch.exp(-gamma * torch.cdist(xi, xj, p=2) ** 2)
            case "mahalanobis_distance":
                d = xi.size(-1)
                xc = xi - xi.mean(dim=0, keepdim=True)
                cov = (xc.T @ xc) / max(xi.size(0) - 1, 1)
                eps = 1e-5 * torch.eye(d, device=xi.device, dtype=xi.dtype)
                inv_cov = torch.linalg.inv(cov + eps)
                diff = xi.unsqueeze(1) - xj.unsqueeze(0)
                return torch.sqrt((diff @ inv_cov * diff).sum(dim=-1).clamp(min=1e-8))
            case "canberra_distance":
                xi_e = xi.unsqueeze(1);
                xj_e = xj.unsqueeze(0)
                num = torch.abs(xi_e - xj_e)
                den = torch.abs(xi_e) + torch.abs(xj_e)
                return (num / (den + 1e-8)).sum(dim=-1)
            case "hellinger_distance":
                xi_s = F.softmax(xi, dim=-1).unsqueeze(1)
                xj_s = F.softmax(xj, dim=-1).unsqueeze(0)
                return torch.sqrt(0.5 * ((xi_s.sqrt() - xj_s.sqrt()) ** 2).sum(dim=-1).clamp(min=0))
            case "bhattacharyya_distance":
                xi_s = F.softmax(xi, dim=-1).unsqueeze(1)
                xj_s = F.softmax(xj, dim=-1).unsqueeze(0)
                bc = (xi_s * xj_s).sqrt().sum(dim=-1).clamp(min=1e-8)
                return -bc.log()
            case "energy_distance":
                xi_e = xi.unsqueeze(1);
                xj_e = xj.unsqueeze(0)
                return torch.sqrt(((xi_e - xj_e) ** 2).sum(dim=-1))
            case "total_variational_distance":
                xi_s = F.softmax(xi, dim=-1).unsqueeze(1)
                xj_s = F.softmax(xj, dim=-1).unsqueeze(0)
                return 0.5 * (xi_s - xj_s).abs().sum(dim=-1)
            case "frobenius_norm":
                xi_e = xi.unsqueeze(1);
                xj_e = xj.unsqueeze(0)
                return torch.sqrt(((xi_e - xj_e) ** 2).sum(dim=-1))
            case "log_euclidean":
                xi_e = xi.unsqueeze(1);
                xj_e = xj.unsqueeze(0)
                return torch.sqrt(((xi_e.log() - xj_e.log()) ** 2).sum(dim=-1))
            case "spectral_norm":
                return torch.cdist(xi, xj, p=2)
            case "grassmannian_distance" | "curvature_based_distance":
                return torch.cdist(xi, xj, p=2)
            case "normalized_compression_distance":
                return torch.cdist(xi, xj, p=2)
            case "variation_of_information":
                xi_s = F.softmax(xi, dim=-1).unsqueeze(1)
                xj_s = F.softmax(xj, dim=-1).unsqueeze(0)
                xm = 0.5 * (xi_s + xj_s) + 1e-9
                return (-(xi_s * xm.log()).sum(-1) - (xj_s * xm.log()).sum(-1)
                        + (xi_s * xi_s.log()).sum(-1) + (xj_s * xj_s.log()).sum(-1))
            case "levenshtein_distance":
                return torch.cdist(xi, xj, p=1)

            # ── sklearn pairwise metrics ─────────────────────────────────
            case "cityblock":
                # Same as manhattan / L1
                return torch.cdist(xi, xj, p=1)
            case "sqeuclidean" | "squared_euclidean_sklearn":
                # Squared L2 (no sqrt) — sklearn sqeuclidean convention
                return torch.cdist(xi, xj, p=2) ** 2
            case "seuclidean" | "standardized_euclidean":
                # Standardized Euclidean: scale each feature by its std from xi
                std = xi.std(dim=0).clamp(min=1e-9)  # (d,)
                xi_s = xi / std;
                xj_s = xj / std
                return torch.cdist(xi_s, xj_s, p=2)

            # ── scipy.spatial.distance metrics ───────────────────────────
            case "braycurtis":
                # d(u,v) = sum|ui-vi| / sum|ui+vi|   (ecology dissimilarity)
                xi_e = xi.unsqueeze(1);
                xj_e = xj.unsqueeze(0)
                num = (xi_e - xj_e).abs().sum(-1)
                den = (xi_e.abs() + xj_e.abs()).sum(-1).clamp(min=1e-9)
                return num / den

            case "correlation":
                # Correlation distance: 1 - Pearson(u, v)
                xi_c = xi - xi.mean(dim=-1, keepdim=True)
                xj_c = xj - xj.mean(dim=-1, keepdim=True)
                xi_n = F.normalize(xi_c, p=2, dim=-1)
                xj_n = F.normalize(xj_c, p=2, dim=-1)
                return 1.0 - xi_n @ xj_n.T

            case "dice":
                # Dice dissimilarity for binary vectors: 1 - 2*|u AND v| / (|u|+|v|)
                xi_b = (xi > 0).float();
                xj_b = (xj > 0).float()
                xi_e = xi_b.unsqueeze(1);
                xj_e = xj_b.unsqueeze(0)
                tf = (xi_e * xj_e).sum(-1)  # true-true count
                num = xi_b.sum(-1, keepdim=True) + xj_b.sum(-1)  # (n,k)
                return 1.0 - 2.0 * tf / (num.view(xi.shape[0], xj.shape[0]).clamp(min=1e-9))

            case "hamming":
                # Fraction of positions that differ
                xi_b = (xi > 0).float();
                xj_b = (xj > 0).float()
                xi_e = xi_b.unsqueeze(1);
                xj_e = xj_b.unsqueeze(0)
                d = float(xi.shape[-1])
                return (xi_e - xj_e).abs().sum(-1) / d

            case "jaccard":
                # Jaccard dissimilarity for binary vectors
                xi_b = (xi > 0).float();
                xj_b = (xj > 0).float()
                xi_e = xi_b.unsqueeze(1);
                xj_e = xj_b.unsqueeze(0)
                intersect = (xi_e * xj_e).sum(-1)
                union = ((xi_e + xj_e) > 0).float().sum(-1).clamp(min=1e-9)
                return 1.0 - intersect / union

            case "kulsinski":
                # Kulsinski dissimilarity (deprecated in scipy ≥1.9, kept for compatibility)
                # d = (CTF + CFT - TT + n) / (CFT + CTF + n)
                # where TT = both 1, n = dimension
                xi_b = (xi > 0).float();
                xj_b = (xj > 0).float()
                xi_e = xi_b.unsqueeze(1);
                xj_e = xj_b.unsqueeze(0)
                n = float(xi.shape[-1])
                tt = (xi_e * xj_e).sum(-1)
                tf = (xi_e * (1 - xj_e)).sum(-1)
                ft = ((1 - xi_e) * xj_e).sum(-1)
                return (tf + ft - tt + n) / (tf + ft + n).clamp(min=1e-9)

            case "rogerstanimoto":
                # d = 2*(TF+FT) / (TT + FF + 2*(TF+FT))
                xi_b = (xi > 0).float();
                xj_b = (xj > 0).float()
                xi_e = xi_b.unsqueeze(1);
                xj_e = xj_b.unsqueeze(0)
                n = float(xi.shape[-1])
                tt = (xi_e * xj_e).sum(-1)
                ff = ((1 - xi_e) * (1 - xj_e)).sum(-1)
                tf = (xi_e * (1 - xj_e)).sum(-1)
                ft = ((1 - xi_e) * xj_e).sum(-1)
                r = tf + ft
                return 2.0 * r / (tt + ff + 2.0 * r + 1e-9)

            case "russellrao":
                # d = (n - TT) / n
                xi_b = (xi > 0).float();
                xj_b = (xj > 0).float()
                xi_e = xi_b.unsqueeze(1);
                xj_e = xj_b.unsqueeze(0)
                n = float(xi.shape[-1])
                tt = (xi_e * xj_e).sum(-1)
                return (n - tt) / n

            case "sokalmichener":
                # Same as simple matching coefficient: d = 2*(TF+FT) / (TT+FF+2*(TF+FT))
                xi_b = (xi > 0).float();
                xj_b = (xj > 0).float()
                xi_e = xi_b.unsqueeze(1);
                xj_e = xj_b.unsqueeze(0)
                tf = (xi_e * (1 - xj_e)).sum(-1)
                ft = ((1 - xi_e) * xj_e).sum(-1)
                tt = (xi_e * xj_e).sum(-1)
                ff = ((1 - xi_e) * (1 - xj_e)).sum(-1)
                r = tf + ft
                return 2.0 * r / (tt + ff + 2.0 * r + 1e-9)

            case "sokalsneath":
                # d = 2*(TF+FT) / (TT + 2*(TF+FT))
                xi_b = (xi > 0).float();
                xj_b = (xj > 0).float()
                xi_e = xi_b.unsqueeze(1);
                xj_e = xj_b.unsqueeze(0)
                tt = (xi_e * xj_e).sum(-1)
                tf = (xi_e * (1 - xj_e)).sum(-1)
                ft = ((1 - xi_e) * xj_e).sum(-1)
                r = tf + ft
                return 2.0 * r / (tt + 2.0 * r + 1e-9)

            case "yule":
                # Yule dissimilarity: 2*TF*FT / (TT*FF + TF*FT)
                xi_b = (xi > 0).float();
                xj_b = (xj > 0).float()
                xi_e = xi_b.unsqueeze(1);
                xj_e = xj_b.unsqueeze(0)
                tt = (xi_e * xj_e).sum(-1)
                ff = ((1 - xi_e) * (1 - xj_e)).sum(-1)
                tf = (xi_e * (1 - xj_e)).sum(-1)
                ft = ((1 - xi_e) * xj_e).sum(-1)
                num = 2.0 * tf * ft
                den = (tt * ff + tf * ft).clamp(min=1e-9)
                return num / den

            case _:
                return torch.cdist(xi, xj, p=2)

    # 
    # Bregman divergence  d_phi(x, y)  — (n, k) matrix
    # 

    def _bregman_div(self, X: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Compute (n, k) Bregman divergence matrix d_phi(X[i], C[j]).

        Divergences
        -----------
        "squared_euclidean" : ||x-y||^2               (= standard K-Means)
        "kl"                : sum xi*log(xi/yi)-xi+yi  (generalised KL)
        "itakura_saito"     : sum (x/y - log(x/y) - 1)
        "beta_2"            : ||x-y||^2 / 2
        "logistic"          : sum [x*log(x/y)+(1-x)*log((1-x)/(1-y))]
        "exponential"       : sum exp(x)*(x-y) - exp(x) + exp(y)
        "mahalanobis"       : (x-y)^T W (x-y)  [W from divergence_params]
        "custom"            : divergence_params["fn"](x_expanded, c_expanded)
        """
        div = self.divergence
        if div is None or div == "auto":
            return self._compute_distances(X, C)

        ep = self.eps
        x = X.unsqueeze(1)  # (n, 1, d)
        c = C.unsqueeze(0)  # (1, k, d)

        if div == "squared_euclidean":
            return ((x - c) ** 2).sum(-1)
        if div == "kl":
            xc = x.clamp(min=ep);
            cc = c.clamp(min=ep)
            return (xc * (xc / cc).log() - xc + cc).sum(-1)
        if div == "itakura_saito":
            xc = x.clamp(min=ep);
            cc = c.clamp(min=ep)
            r = xc / cc
            return (r - r.log() - 1).sum(-1)
        if div == "beta_2":
            return ((x - c) ** 2).sum(-1) * 0.5
        if div == "logistic":
            xc = x.clamp(ep, 1 - ep);
            cc = c.clamp(ep, 1 - ep)
            return (xc * (xc / cc).log()
                    + (1 - xc) * ((1 - xc) / (1 - cc)).log()).sum(-1)
        if div == "exponential":
            return (x.exp() * (x - c) - x.exp() + c.exp()).sum(-1)
        if div == "mahalanobis":
            W = self.divergence_params.get("W")
            if W is None:
                return ((x - c) ** 2).sum(-1)
            diff = (x - c);
            return (diff @ W * diff).sum(-1)
        if div == "custom":
            fn = self.divergence_params.get("fn")
            if fn is None:
                raise ValueError("custom divergence requires divergence_params['fn']")
            return fn(x, c)
        raise ValueError(f"Unknown divergence '{self.divergence}'. "
                         f"Valid: squared_euclidean, kl, itakura_saito, beta_2, "
                         f"logistic, exponential, mahalanobis, custom")

    def _assignment_matrix(self, X: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """Return (n, k) matrix — Bregman div if divergence is set, else metric dist."""
        if self.divergence and self.divergence not in ("auto", "none"):
            return self._bregman_div(X, C)
        return self._compute_distances(X, C)

    # 
    # Initialisation
    # 

    def _init_module(self, X: torch.Tensor):
        """Initialise ``self.cluster_centers_`` using the configured ``init``."""
        n, d = X.shape
        k = self.n_clusters
        if isinstance(self.init, torch.Tensor):
            self.cluster_centers_ = self.init.to(device=self.device, dtype=self.dtype)[:k]
            return
        if isinstance(self.init, (list, tuple)):
            self.cluster_centers_ = torch.tensor(self.init, device=self.device, dtype=self.dtype)
            return
        if callable(self.init) and not isinstance(self.init, str):
            self.cluster_centers_ = torch.as_tensor(
                self.init(X.cpu().numpy(), k), device=self.device, dtype=self.dtype)
            return
        mode = str(self.init).lower()
        if mode == "random":
            idx = torch.randperm(n, generator=self.random_state,
                                 device=self.device)[:k]
            self.cluster_centers_ = X[idx].clone()
        elif mode == "farthest":
            start = torch.randint(0, n, (1,), generator=self.random_state,
                                  device=self.device).item()
            sel = [start]
            min_d = self._assignment_matrix(X, X[[start]])[:, 0]
            for _ in range(1, k):
                far = min_d.argmax().item();
                sel.append(far)
                new_d = self._assignment_matrix(X, X[[far]])[:, 0]
                min_d = torch.minimum(min_d, new_d)
            self.cluster_centers_ = X[torch.tensor(sel, device=self.device)].clone()
        else:  # "k-means++"
            start = torch.randint(0, n, (1,), generator=self.random_state,
                                  device=self.device).item()
            sel = [start]
            min_d = self._assignment_matrix(X, X[[start]])[:, 0]
            for _ in range(1, k):
                probs = min_d / min_d.sum().clamp(min=1e-9)
                nxt = torch.multinomial(probs, 1, generator=self.random_state).item()
                sel.append(nxt)
                new_d = self._assignment_matrix(X, X[[nxt]])[:, 0]
                min_d = torch.minimum(min_d, new_d)
            self.cluster_centers_ = X[torch.tensor(sel, device=self.device)].clone()

    # 
    # Fit
    # 

    def fit(self, data_or_X,
            sample_weight: Optional[torch.Tensor] = None,
            **kwargs) -> "BregmanKMeans":
        warm_start = kwargs.get('warm_start', getattr(self, 'warm_start', False))
        X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
        if self.copy_X: X = X.clone()
        n, d = X.shape
        self.n_clusters = min(self.n_clusters, n, d)
        n_runs = (1 if (warm_start and self.cluster_centers_ is not None and
                        self.cluster_centers_.shape[0] == self.n_clusters and
                        self.cluster_centers_.shape[1] == d)
                  else (1 if isinstance(self.init, (list, tuple, torch.Tensor))
                        else (self.n_init if isinstance(self.n_init, int) else 10)))
        best_iner = float('inf');
        best_cen = best_lbl = None;
        best_ni = 0
        for _ in range(n_runs):
            if warm_start and self.cluster_centers_ is not None and self.cluster_centers_.shape[0] == self.n_clusters and self.cluster_centers_.shape[1] == d:
                centers = self.cluster_centers_.clone()
            else:
                self._init_module(X)
                centers = self.cluster_centers_.clone()
            for i in range(self.max_iter):
                D = self._assignment_matrix(X, centers)
                labels = D.argmin(dim=1)
                one_hot = torch.nn.functional.one_hot(labels, self.n_clusters).float()
                counts = one_hot.sum(0, keepdim=True).clamp(min=1)
                new_c = (X.unsqueeze(1) * one_hot.unsqueeze(-1)).sum(0) / counts.T
                shift = (new_c - centers).norm(dim=1).max().item()
                centers = new_c
                if self.verbose:
                    print(f"[BregmanKMeans] iter {i}: shift={shift:.6f}")
                if shift <= self.tol: break
            D = self._assignment_matrix(X, centers)
            labels = D.argmin(dim=1)
            iner = D[torch.arange(X.shape[0], device=self.device), labels].sum().item()
            if iner < best_iner:
                best_iner, best_cen, best_lbl, best_ni = iner, centers.clone(), labels.clone(), i + 1
        self.cluster_centers_ = best_cen
        self.labels_ = best_lbl
        self.inertia_ = best_iner
        self.n_iter_ = best_ni
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        return self._assignment_matrix(X, self.cluster_centers_).argmin(dim=1)

    def transform(self, X: torch.Tensor, **kwargs) -> torch.Tensor:
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        return self._assignment_matrix(X, self.cluster_centers_)

    def fit_predict(self, data_or_X, **kwargs):
        self.fit(data_or_X, **kwargs);
        return self.labels_

    def fit_transform(self, data_or_X, **kwargs):
        self.fit(data_or_X, **kwargs);
        return self.transform(data_or_X)


class KMeansCluster(BregmanKMeans):
    def __init__(self,
                 n_clusters: int = 8,
                 init: Union[str, Callable, nn.Module, list, tuple, torch.Tensor] = "k-means++",
                 n_init: Union[str, int] = "auto",
                 max_iter: int = 300,
                 tol: float = 1e-4,
                 verbose: bool = False,
                 random_state: int = None,
                 copy_X: bool = True,
                 metric: Union[str, Callable, nn.Module] = "euclidean",
                 metric_params: dict = None,
                 algorithm: Union[str, Callable, nn.Module] = "lloyd",
                 n_yinyang_groups: int = 10,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = int(verbose)
        self.device = device
        self.dtype = dtype
        if random_state is not None:
            torch.manual_seed(random_state)
            self.random_state = torch.Generator(device=self.device).manual_seed(random_state)
        else:
            self.random_state = None
        self.copy_X = copy_X
        if isinstance(algorithm, str):
            self.algorithm = algorithm.lower()
        else:
            self.algorithm = algorithm
        self.n_yinyang_groups = n_yinyang_groups
        # Metric routing — delegates to MLCluster._create_metric (inherited via BregmanKMeans).
        # _metric_name / _metric_callable preserved for backward-compatibility with subclasses.
        self.metric_params = metric_params if metric_params is not None else {}
        self._metric_name = metric.lower() if isinstance(metric, str) else None
        self._metric_callable = (
            metric if (callable(metric) and not isinstance(metric, nn.Module)) else None
        )
        if self._metric_name is None and self._metric_callable is None and not isinstance(metric, nn.Module):
            self._metric_name = "euclidean"
        if isinstance(metric, nn.Module):
            self.metric = metric
        else:
            self._create_metric(metric, self.metric_params)
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None
        self.n_features_in_ = None

    # ── Distance utilities ────────────────────────────────────────────────────
    # _metric_from_name / _metric_from_callable are inherited from BregmanKMeans;
    # self.metric is now a lambda (str/callable) or nn.Module created by _create_metric.

    def _compute_distances(self, X: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """Return (N, K) pairwise distance matrix using the configured metric.

        For nn.Module metrics the module is called directly with (X, centers).
        For string / callable metrics the result is routed through dist_calc.
        The returned matrix always has shape (N, K).
        """
        if isinstance(self.metric, nn.Module):
            return self.metric(X, centers)
        D = self.metric(X, centers)  # dispatches to _metric_from_name/callable
        # Ensure 2-D (N, K) — some metric implementations return (N, K) already
        if D.dim() > 2:
            D = D.squeeze()
        return D

    def dist_calc(self, metric_type: str, xi: torch.Tensor, xj: torch.Tensor, **kwargs) -> torch.Tensor:
        """Vectorised pairwise distance between rows of xi (N,d) and xj (K,d).

        Supported metrics
        -----------------
        ``euclidean / l2`` · ``manhattan / l1`` · ``minkowski`` · ``chebyshev``
        ``cosine`` · ``kl_divergence`` · ``js_divergence``
        ``wasserstein / earth_mover`` · ``sliced_wasserstein_distance``
        ``rbf_distance`` · ``mahalanobis_distance`` · ``canberra_distance``
        ``hellinger_distance`` · ``bhattacharyya_distance``
        ``energy_distance`` · ``total_variational_distance``
        ``frobenius_norm`` · ``log_euclidean`` · ``spectral_norm``
        ``grassmannian_distance`` · ``curvature_based_distance``
        ``normalized_compression_distance`` · ``variation_of_information``
        ``levenshtein_distance``
        """
        match metric_type.lower():
            case "euclid" | "euclidean" | "l2_distance" | "distance" | "l2":
                return torch.cdist(xi, xj, p=2)
            case "manhattan" | "l1_distance" | "l1":
                return torch.cdist(xi, xj, p=1)
            case "minkowski":
                p = kwargs.get("p", 2)
                return torch.cdist(xi, xj, p=p)
            case "chebyshev":
                return torch.cdist(xi, xj, p=float("inf"))
            case "cosine":
                xi_n = F.normalize(xi, p=2, dim=-1)
                xj_n = F.normalize(xj, p=2, dim=-1)
                return 1.0 - xi_n @ xj_n.T
            case "kl_divergence":
                kl_div = nn.KLDivLoss(reduction="none")
                xi_s = F.softmax(xi, dim=-1).unsqueeze(-2)  # (N, 1, d)
                xj_s = F.log_softmax(xj, dim=-1).unsqueeze(-3)  # (1, K, d)
                return kl_div(xj_s.expand(xi_s.shape[0], xj.shape[0], -1),
                              xi_s.expand(xi_s.shape[0], xj.shape[0], -1)).sum(dim=-1)
            case "js_divergence":
                xi_s = F.softmax(xi, dim=-1)  # (N, d)
                xj_s = F.softmax(xj, dim=-1)  # (K, d)
                xi_e = xi_s.unsqueeze(1)  # (N, 1, d)
                xj_e = xj_s.unsqueeze(0)  # (1, K, d)
                xm = 0.5 * (xi_e + xj_e)  # (N, K, d)
                kl_a = (xi_e * (xi_e / (xm + 1e-8) + 1e-8).log()).sum(-1)
                kl_b = (xj_e * (xj_e / (xm + 1e-8) + 1e-8).log()).sum(-1)
                return 0.5 * (kl_a + kl_b)
            case "wasserstein" | "earth_mover":
                # 1-D sliced approximation (tractable for arbitrary d)
                num_proj = kwargs.get("num_projections", 32)
                p_ord = kwargs.get("p", 1)
                d = xi.size(-1)
                projs = F.normalize(torch.randn(d, num_proj, device=xi.device, dtype=xi.dtype), dim=0)
                xi_p, _ = torch.sort(xi @ projs, dim=0)  # (N, P)
                xj_p, _ = torch.sort(xj @ projs, dim=0)  # (K, P)
                # Use uniform re-sampling to align sizes
                n, k = xi_p.shape[0], xj_p.shape[0]
                if n != k:
                    idx = torch.linspace(0, k - 1, n, device=xi.device).long()
                    xj_p = xj_p[idx]
                swd = torch.pow(torch.abs(xi_p - xj_p), p_ord).mean(dim=0).mean()
                return torch.cdist(xi, xj, p=2) * 0 + swd  # broadcast scalar → (N,K)
            case "rbf_distance":
                gamma = math.fabs(kwargs.get("gamma", 0.1))
                return torch.exp(-gamma * torch.cdist(xi, xj, p=2) ** 2)
            case "mahalanobis_distance":
                d = xi.size(-1)
                # Compute shared inverse covariance from xi
                xc = xi - xi.mean(dim=0, keepdim=True)
                cov = (xc.T @ xc) / max(xi.size(0) - 1, 1)
                eps = 1e-5 * torch.eye(d, device=xi.device, dtype=xi.dtype)
                inv_cov = torch.linalg.inv(cov + eps)  # (d, d)
                diff = xi.unsqueeze(1) - xj.unsqueeze(0)  # (N, K, d)
                return torch.sqrt((diff @ inv_cov * diff).sum(dim=-1).clamp(min=1e-8))
            case "canberra_distance":
                xi_e = xi.unsqueeze(1)  # (N, 1, d)
                xj_e = xj.unsqueeze(0)  # (1, K, d)
                num = torch.abs(xi_e - xj_e)
                den = torch.abs(xi_e) + torch.abs(xj_e)
                return (num / (den + 1e-8)).sum(dim=-1)  # (N, K)
            case "hellinger_distance":
                xi_s = F.softmax(xi, dim=-1).unsqueeze(1)  # (N, 1, d)
                xj_s = F.softmax(xj, dim=-1).unsqueeze(0)  # (1, K, d)
                return torch.sqrt(torch.clamp(1.0 - torch.sqrt(xi_s * xj_s + 1e-8).sum(-1), min=0))
            case "bhattacharyya_distance":
                xi_s = F.softmax(xi, dim=-1).unsqueeze(1)
                xj_s = F.softmax(xj, dim=-1).unsqueeze(0)
                bc = torch.sqrt(xi_s * xj_s + 1e-8).sum(-1)  # (N, K)
                return -torch.log(bc.clamp(min=1e-8))
            case "energy_distance":
                dist_ij = torch.cdist(xi, xj, p=2)  # (N, K)
                dist_ii = torch.cdist(xi, xi, p=2).mean(dim=1, keepdim=True)  # (N, 1)
                dist_jj = torch.cdist(xj, xj, p=2).mean(dim=1, keepdim=True).T  # (1, K)
                return 2 * dist_ij - dist_ii - dist_jj
            case "total_variational_distance":
                xi_s = F.softmax(xi, dim=-1).unsqueeze(1)
                xj_s = F.softmax(xj, dim=-1).unsqueeze(0)
                return 0.5 * torch.abs(xi_s - xj_s).sum(dim=-1)
            case "frobenius_norm":
                diff = xi.unsqueeze(1) - xj.unsqueeze(0)  # (N, K, d)
                return diff.norm(dim=-1)  # (N, K)
            case "log_euclidean":
                # Log-Euclidean: add log-space correction to L2
                xi_log = torch.sign(xi) * (xi.abs() + 1e-8).log()
                xj_log = torch.sign(xj) * (xj.abs() + 1e-8).log()
                return torch.cdist(xi_log, xj_log, p=2)
            case "spectral_norm":
                # Spectral approximation: use squared singular values as features
                xi_sv = torch.linalg.svdvals(xi)  # (min(N,d),)
                xj_sv = torch.linalg.svdvals(xj)
                # Pad to equal length
                d = max(xi_sv.shape[0], xj_sv.shape[0])
                xi_sv = F.pad(xi_sv, (0, d - xi_sv.shape[0]))
                xj_sv = F.pad(xj_sv, (0, d - xj_sv.shape[0]))
                return (xi_sv - xj_sv).norm().expand(xi.shape[0], xj.shape[0])
            case "sliced_wasserstein_distance":
                num_proj = kwargs.get("num_projections", 64)
                p_ord = kwargs.get("p", 2)
                d = xi.size(-1)
                projs = F.normalize(torch.randn(d, num_proj, device=xi.device, dtype=xi.dtype), dim=0)
                xi_p, _ = torch.sort(xi @ projs, dim=0)  # (N, P)
                xj_r = torch.sort(xj @ projs, dim=0)[0]  # (K, P)
                # Compute per-pair SWD
                D = torch.zeros(xi.shape[0], xj.shape[0], device=xi.device, dtype=xi.dtype)
                for ki in range(xj.shape[0]):
                    # Interpolate xj[ki] projection to N samples
                    xj_interp = F.interpolate(xj_r[ki:ki + 1].unsqueeze(0), size=xi.shape[0],
                                              mode='linear', align_corners=False).squeeze()
                    D[:, ki] = torch.pow(torch.abs(xi_p - xj_interp.unsqueeze(-1)), p_ord).mean(0).mean()
                return D
            case "grassmannian_distance":
                Q_xi, _ = torch.linalg.qr(xi)
                Q_xj, _ = torch.linalg.qr(xj)
                P_xi = Q_xi @ Q_xi.T
                P_xj = Q_xj @ Q_xj.T
                dist_g = torch.linalg.norm(P_xi - P_xj, ord='fro')
                return torch.cdist(xi, xj, p=2) + dist_g
            case "curvature_based_distance":
                sigma = kwargs.get("sigma", 1.0)

                def local_curv(Z):
                    D2 = torch.cdist(Z, Z, p=2) ** 2
                    W = torch.exp(-D2 / (2 * sigma ** 2))
                    W = W / (W.sum(dim=1, keepdim=True) + 1e-8)
                    return (Z - W @ Z).norm(dim=1)

                ci = local_curv(xi).unsqueeze(1)  # (N, 1)
                cj = local_curv(xj).unsqueeze(0)  # (1, K)
                return torch.abs(ci - cj) + torch.cdist(xi, xj, p=2)
            case "normalized_compression_distance":
                def approx_entropy(Z):
                    cov = (Z.T @ Z) / max(Z.shape[0] - 1, 1)
                    eps = 1e-6 * torch.eye(Z.shape[1], device=Z.device, dtype=Z.dtype)
                    _, ld = torch.linalg.slogdet(cov + eps)
                    return ld

                h_i = approx_entropy(xi)
                h_j = approx_entropy(xj)
                h_u = approx_entropy(torch.cat([xi, xj], 0))
                ncd = (h_u - torch.minimum(h_i, h_j)) / (torch.maximum(h_i, h_j) + 1e-8)
                return torch.cdist(xi, xj, p=2) * ncd
            case "variation_of_information":
                def approx_h(Z):
                    cov = (Z.T @ Z) / max(Z.shape[0] - 1, 1)
                    eps = 1e-6 * torch.eye(Z.shape[1], device=Z.device, dtype=Z.dtype)
                    _, ld = torch.linalg.slogdet(cov + eps)
                    return 0.5 * ld

                vi = 2 * approx_h(torch.cat([xi, xj], 0)) - approx_h(xi) - approx_h(xj)
                return torch.cdist(xi, xj, p=2) * vi.abs()
            case "levenshtein_distance":
                # Soft Levenshtein via substitution cost = L2 distance
                sub_cost = torch.cdist(xi, xj, p=2)
                return sub_cost
            case _:
                # Fallback: L2
                return torch.cdist(xi, xj, p=2)

    def _greedy_kmeans_plus_plus(self, X: torch.Tensor, n_local_trials: int = None):
        """
        Implements the "greedy k-means++" algorithm as specified in the docstring.
        It makes several trials at each sampling step and chooses the best centroid among them.
        """
        n_samples, n_features = X.shape
        centroids = torch.empty((self.n_clusters, n_features), device=self.device, dtype=self.dtype)

        # Pick the first center uniformly at random
        idx = torch.randint(0, n_samples, (1,), generator=self.random_state, device=self.device).item()
        centroids[0] = X[idx]

        # Initialize distances to the first center
        # Use squared Euclidean distance as standard for k-means++
        closest_dist_sq = torch.norm(X - centroids[0], dim=1).pow(2)

        if n_local_trials is None:
            # Standard heuristic for n_local_trials in greedy k-means++
            n_local_trials = 2 + int(math.log(self.n_clusters))

        for i in range(1, self.n_clusters):
            # Proportional sampling probability
            probs = closest_dist_sq / closest_dist_sq.sum().clamp(min=1e-12)
            probs = torch.nan_to_num(probs, nan=1e-9, posinf=1.0, neginf=1e-9).clamp(min=1e-9)
            probs = probs / probs.sum()
            if torch.any(probs <= 0) or torch.any(torch.isnan(probs)) or torch.any(torch.isinf(probs)):
                probs = torch.ones(n_samples, device=self.device, dtype=self.dtype) / n_samples
            # Sample multiple candidates
            candidate_indices = torch.multinomial(probs, n_local_trials, generator=self.random_state)
            candidates = X[candidate_indices]

            # For each candidate, calculate the potential reduction in total inertia
            # (distance of all points to the newest candidate)
            # We want to pick the candidate that minimizes total inertia.

            # Efficiency optimization: instead of calculating full cdist for each candidate,
            # we just need to see how much each candidate *improves* the current closest_dist_sq

            candidate_dists_sq = torch.cdist(candidates, X, p=2).pow(2)  # (n_local_trials, n_samples)

            # New potential total inertia if we picked each candidate
            new_total_inertias = torch.sum(torch.minimum(closest_dist_sq, candidate_dists_sq), dim=1)

            best_candidate_idx = torch.argmin(new_total_inertias)
            centroids[i] = candidates[best_candidate_idx]

            # Update the global closest_dist_sq
            closest_dist_sq = torch.minimum(closest_dist_sq, candidate_dists_sq[best_candidate_idx])

        return centroids

    def _init_module(self, X: torch.Tensor):
        self.n_features_in_ = X.size(-1)
        n_samples = X.size(-2)
        if isinstance(self.init, str):
            if self.init in ["k-means++", "kmeans++"]:
                self.cluster_centers_ = self._greedy_kmeans_plus_plus(X)
            elif self.init == "random":
                idx = torch.randint(0, n_samples, (self.n_clusters,), generator=self.random_state, device=self.device)
                self.cluster_centers_ = X[idx, :]
        elif isinstance(self.init, (list, tuple, torch.Tensor)):
            self.cluster_centers_ = torch.as_tensor(self.init, device=self.device, dtype=self.dtype)
        elif isinstance(self.init, nn.Module):
            self.cluster_centers_ = self.init(X)
        elif isinstance(self.init, Callable):
            self.cluster_centers_ = self.init(X, self.n_clusters, self.random_state)

        if isinstance(self.n_init, str):
            if self.n_init == "auto":
                if isinstance(self.init, (Callable, nn.Module)) or self.init == "random":
                    self.n_init = 10
                elif isinstance(self.init, (list, tuple, torch.Tensor)) or self.init in ["k-means++", "kmeans++"]:
                    self.n_init = 1
        return self

    def _run_lloyd(self, X: torch.Tensor, sample_weight: Optional[torch.Tensor] = None):
        if self.n_init > 1:
            self._init_module(X)
        centers = self.cluster_centers_.clone()
        nodes = X.shape[0]
        labels = torch.zeros(nodes, device=self.device, dtype=torch.long)
        for i in range(self.max_iter):
            old_centers = centers.clone()
            dist = self._compute_distances(X, centers).pow(2)
            labels = torch.argmin(dist, dim=-1)
            new_centers = torch.zeros_like(centers)
            for k in range(self.n_clusters):
                mask = (labels == k)
                if mask.any():
                    points_in_clusters = X[mask]
                    if sample_weight is not None:
                        weights = sample_weight[mask].unsqueeze(-1)
                        new_centers[k] = (points_in_clusters * weights).sum(dim=0) / weights.sum()
                    else:
                        new_centers[k] = points_in_clusters.mean(dim=0)
                else:
                    idx = torch.randint(0, nodes, (1,), generator=self.random_state, device=self.device).item()
                    new_centers[k] = X[idx]
            centers = new_centers
            center_shift = torch.linalg.norm(centers - old_centers)
            if center_shift <= self.tol:
                if self.verbose:
                    print(f"Converged at iteration {i} with shift {center_shift:.6f}")
                break
        final_dist = self._compute_distances(X, centers).pow(2)
        min_dist, _ = torch.min(final_dist, dim=-1)
        if sample_weight is not None:
            inertia = torch.sum(min_dist * sample_weight)
        else:
            inertia = torch.sum(min_dist)
        return centers, labels, inertia.item(), i + 1

    def _run_elkan(self, X: torch.Tensor, sample_weight: Optional[torch.Tensor] = None):
        n_samples = X.size(0)
        n_clusters = self.n_clusters
        centers = self.cluster_centers_.clone()
        upper_bounds = torch.full((n_samples,), float('inf'), device=self.device, dtype=self.dtype)
        lower_bounds = torch.zeros((n_samples, n_clusters), device=self.device, dtype=self.dtype)
        labels = torch.full((n_samples,), -1, dtype=torch.long, device=self.device)
        dist_matrix = self._compute_distances(X, centers)
        upper_bounds, labels = torch.min(dist_matrix, dim=1)
        lower_bounds = dist_matrix
        for i in range(self.max_iter):
            old_centers = centers.clone()
            centroid_dist = self._compute_distances(centers, centers)
            s = 0.5 * centroid_dist.masked_fill(torch.eye(n_clusters, device=self.device).bool(), float('inf')).min(
                dim=-1).values
            mask = upper_bounds > s[labels]
            if mask.any():
                for j in range(n_clusters):
                    z_mask = mask & (j != labels) & (upper_bounds > lower_bounds[:, j]) & (
                                upper_bounds > 0.5 * centroid_dist[labels, j])
                    if z_mask.any():
                        actual_dist = torch.norm(X[z_mask] - centers[j], dim=-1)
                        lower_bounds[z_mask, j] = actual_dist
                        closer_mask = actual_dist < upper_bounds[z_mask]
                        if closer_mask.any():
                            subset_indices = torch.where(z_mask)[0][closer_mask]
                            labels[subset_indices] = j
                            upper_bounds[subset_indices] = actual_dist[closer_mask]
                new_centers = torch.zeros_like(centers)
                for k in range(n_clusters):
                    cluster_mask = (labels == k)
                    if cluster_mask.any():
                        if sample_weight is not None:
                            w = sample_weight[cluster_mask].unsqueeze(-1)
                            new_centers[k] = (X[cluster_mask] * w).sum(0) / w.sum()
                        else:
                            new_centers[k] = X[cluster_mask].mean(0)
                    else:
                        new_centers[k] = X[torch.randint(0, n_samples, (1,), generator=self.random_state)].squeeze()
                shift = torch.norm(new_centers - old_centers, dim=1)
                lower_bounds = torch.max(lower_bounds - shift, torch.zeros_like(lower_bounds))
                upper_bounds = upper_bounds + shift[labels]
                centers = new_centers
                if shift.max() < self.tol:
                    break
        inertia = torch.sum(upper_bounds ** 2) if sample_weight is None else torch.sum(
            (upper_bounds ** 2) * sample_weight)
        return centers, labels, inertia.item(), i + 1

    def _run_yinyang(self, X: torch.Tensor, sample_weight: Optional[torch.Tensor] = None):
        """
        Yinyang K-Means [Ding et al., ICML 2015].

        Accelerates standard Lloyd iteration by maintaining:
          - One *upper bound* u(x)  ≥ dist(x, assigned centroid)
          - G *group lower bounds* l_g(x)  ≤ min_{c ∈ group_g} dist(x, c)

        A point's assignment is only re-evaluated when the upper bound
        exceeds at least one group lower bound (the "filter" condition).
        This avoids the majority of exact distance computations.

        Groups of centroids are obtained by a preliminary k-means/random
        partition of the centroid set into ``n_yinyang_groups`` super-clusters.

        Algorithm
        ---------
        1. Initialise groups by randomly assigning centroids to G groups.
        2. Compute exact dist(x, c) for all x, c; set labels, u, l_g.
        3. For each iteration:
           a. Compute centroid drift δ(c) = ‖c_new − c_old‖.
           b. Update per-group max-drift Δ_g = max_{c ∈ g} δ(c).
           c. Update upper bound: u(x) += δ(assigned centroid).
           d. Update group lower bounds: l_g(x) -= Δ_g.
           e. **Global filter**: if u(x) ≤ min_g l_g(x), skip entirely.
           f. **Group filter**: for each group g, if u(x) ≤ l_g(x), skip g.
           g. For all remaining (x, c) pairs compute exact distance;
              tighten bounds and update assignment if a closer centroid found.
        4. Recompute centroids from assignments.
        """
        n_samples, n_features = X.shape
        K = self.n_clusters
        G = max(1, min(self.n_yinyang_groups, K))  # number of groups

        centers = self.cluster_centers_.clone()  # (K, d)

        # ── 1. Assign each centroid to one of G groups (round-robin) ──────
        group_of = torch.arange(K, device=self.device, dtype=torch.long) % G  # (K,)

        # ── 2. Initialise: full distance matrix to get labels, u, l ───────
        dist_mat = self._compute_distances(X, centers)  # (N, K)
        u, labels = dist_mat.min(dim=1)  # upper-bound scalar per point

        # lower bounds per group: min dist to any centroid in the group
        l = torch.full((n_samples, G), float('inf'),
                       device=self.device, dtype=self.dtype)
        for g in range(G):
            g_mask = (group_of == g)
            if g_mask.any():
                l[:, g] = dist_mat[:, g_mask].min(dim=1).values

        n_iter = 0
        for i in range(self.max_iter):
            n_iter = i + 1
            old_centers = centers.clone()

            # ── 3a. Recompute centers ─────────────────────────────────────
            new_centers = torch.zeros_like(centers)
            for k in range(K):
                mask = (labels == k)
                if mask.any():
                    if sample_weight is not None:
                        w = sample_weight[mask].unsqueeze(-1)
                        new_centers[k] = (X[mask] * w).sum(0) / w.sum()
                    else:
                        new_centers[k] = X[mask].mean(0)
                else:
                    idx = torch.randint(0, n_samples, (1,),
                                        generator=self.random_state,
                                        device=self.device).item()
                    new_centers[k] = X[idx]

            # ── 3b. Centroid drift ────────────────────────────────────────
            delta = torch.norm(new_centers - old_centers, dim=1)  # (K,)
            centers = new_centers

            if delta.max() <= self.tol:
                if self.verbose:
                    print(f"[Yinyang] converged at iteration {i} "
                          f"(max drift={delta.max():.6f})")
                break

            # ── 3c–d. Bound updates ───────────────────────────────────────
            u = u + delta[labels]  # upper bound

            # per-group max drift  Δ_g = max_{c ∈ g} δ(c)
            delta_g = torch.zeros(G, device=self.device, dtype=self.dtype)
            for g in range(G):
                g_mask = (group_of == g)
                if g_mask.any():
                    delta_g[g] = delta[g_mask].max()

            l = (l - delta_g.unsqueeze(0)).clamp(min=0.0)  # lower bounds

            # ── 3e. Global filter: skip if u(x) ≤ min_g l_g(x) ──────────
            min_l, _ = l.min(dim=1)  # (N,)
            need_check = u > min_l  # (N,) bool

            if not need_check.any():
                continue

            # ── 3f–g. Per-group filter + exact distance for candidates ────
            check_idx = torch.where(need_check)[0]  # indices to (possibly) reassign

            # Exact distances only for checking points → candidate centroids
            # group filter: for group g skip if u(x) <= l_g(x)
            new_dist_xc = self._compute_distances(X[check_idx], centers)  # (|check|, K)

            new_u = new_dist_xc.min(dim=1).values
            new_lbl = new_dist_xc.argmin(dim=1)

            # Update labels
            labels[check_idx] = new_lbl
            u[check_idx] = new_u

            # Re-tighten group lower bounds for checked points
            for g in range(G):
                g_mask_c = (group_of == g)
                if g_mask_c.any():
                    l[check_idx, g] = new_dist_xc[:, g_mask_c].min(dim=1).values

        # ── Final inertia ─────────────────────────────────────────────────
        final_dist = torch.norm(X - centers[labels], dim=1).pow(2)
        if sample_weight is not None:
            inertia = (final_dist * sample_weight).sum().item()
        else:
            inertia = final_dist.sum().item()

        return centers, labels, inertia, n_iter

    def fit(self, data_or_X, **kwargs):
        X = torch.as_tensor(data_or_X.clone() if self.copy_X else data_or_X, dtype=self.dtype, device=self.device)
        self._init_module(X)

        best_inertia = float('inf')
        best_centers = None
        best_labels = None
        best_n_iter = 0
        sample_weight = kwargs.get("sample_weight", None)

        for i in range(self.n_init):
            if isinstance(self.algorithm, str):
                if self.algorithm == "elkan":
                    run_func = self._run_elkan
                elif self.algorithm == "yinyang":
                    run_func = self._run_yinyang
                else:
                    run_func = self._run_lloyd
                centers, labels, inertia, n_iter = run_func(X, sample_weight)
            elif isinstance(self.algorithm, nn.Module):
                centers, labels, inertia, n_iter = self.algorithm(X, sample_weight)
            elif isinstance(self.algorithm, Callable):
                centers, labels, inertia, n_iter = self.algorithm(X, sample_weight)

            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = labels
                best_n_iter = n_iter
        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(device=self.device, dtype=self.dtype)
        dist = torch.cdist(X, self.cluster_centers_, p=2).pow(2)
        return torch.argmin(dist, dim=-1)

    def transform(self, X: torch.Tensor, **kwargs):
        X = X.to(device=self.device, dtype=self.dtype)
        return torch.cdist(X, self.cluster_centers_, p=2)

    def fit_predict(self, X: torch.Tensor, **kwargs):
        self.fit(X, **kwargs)
        return self.labels_

    def fit_transform(self, X: torch.Tensor, **kwargs):
        self.fit(X, **kwargs)
        X = X.to(device=self.device, dtype=self.dtype)
        return torch.cdist(X, self.cluster_centers_, p=2)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Callable, Union, Any, List, Tuple, Dict, Iterable
import warnings
from .....models.utils import MLCluster
from torch.func import vmap
import joblib

__all__ = [
    "ward_tree",
    "AgglomerativeClustering",
    "FeatureAgglomeration",
    "Birch",
]


def ward_tree(X: torch.Tensor,
              connectivity: Union[list, tuple, torch.Tensor] = None,
              n_clusters: int = None,
              return_distances: bool = False):
    import heapq as _hq

    if not isinstance(X, torch.Tensor):
        X = torch.as_tensor(X, dtype=torch.float)
    X = X.float()
    n_samples, n_features = X.shape

    if connectivity is not None:
        conn = torch.as_tensor(connectivity, dtype=torch.float)
        if conn.shape != (n_samples, n_samples):
            raise ValueError(
                f"connectivity must be ({n_samples}, {n_samples}), "
                f"got {tuple(conn.shape)}."
            )
        conn_mask = (conn != 0)
    else:
        conn_mask = None

    parent = list(range(n_samples))
    sizes  = [1] * n_samples

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int):
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    if conn_mask is not None:
        _uf_parent = list(range(n_samples))
        def _uf_find(x):
            while _uf_parent[x] != x:
                _uf_parent[x] = _uf_parent[_uf_parent[x]]
                x = _uf_parent[x]
            return x
        for _i in range(n_samples):
            for _j in range(_i + 1, n_samples):
                if conn_mask[_i, _j].item():
                    _ri, _rj = _uf_find(_i), _uf_find(_j)
                    if _ri != _rj:
                        _uf_parent[_ri] = _rj
        n_connected_components = len({_uf_find(_i) for _i in range(n_samples)})
    else:
        n_connected_components = 1

    centroids: Dict[int, torch.Tensor] = {i: X[i].clone() for i in range(n_samples)}
    cluster_sizes: Dict[int, int] = {i: 1 for i in range(n_samples)}

    def _ward_dist(a: int, b: int) -> float:
        na, nb = cluster_sizes[a], cluster_sizes[b]
        diff = centroids[a] - centroids[b]
        sq = float((diff * diff).sum().item())
        return (na * nb) / (na + nb) * sq

    heap: list = []
    active_set: set = set(range(n_samples))

    if conn_mask is None:
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                _hq.heappush(heap, (_ward_dist(i, j), i, j))
    else:
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if conn_mask[i, j].item():
                    _hq.heappush(heap, (_ward_dist(i, j), i, j))

    children_list: List[List[int]] = []
    distances_list: List[float] = []
    parents_arr: List[int] = [-1] * (2 * n_samples - 1)
    next_node = n_samples
    n_merges_target = (n_samples - n_clusters if n_clusters is not None else n_samples - 1)

    merges_done = 0
    while heap and merges_done < n_merges_target:
        d, a, b = _hq.heappop(heap)
        if a not in active_set or b not in active_set:
            continue
        d_actual = _ward_dist(a, b)
        if abs(d_actual - d) > 1e-9 * (1 + abs(d)):
            _hq.heappush(heap, (d_actual, a, b))
            continue

        na, nb = cluster_sizes[a], cluster_sizes[b]
        nu = na + nb
        new_centroid = (na * centroids[a] + nb * centroids[b]) / nu
        u = next_node
        next_node += 1
        centroids[u] = new_centroid
        cluster_sizes[u] = nu

        children_list.append([a, b])
        distances_list.append(math.sqrt(max(d_actual, 0.0)))
        parents_arr[a] = u
        parents_arr[b] = u

        active_set.discard(a)
        active_set.discard(b)
        active_set.add(u)

        for v in list(active_set):
            if v == u:
                continue
            _hq.heappush(heap, (_ward_dist(u, v), min(u, v), max(u, v)))

        merges_done += 1

    children = (
        torch.tensor(children_list, dtype=torch.long)
        if children_list
        else torch.zeros((0, 2), dtype=torch.long)
    )
    n_leaves = n_samples
    parents = (
        torch.tensor(parents_arr[:next_node], dtype=torch.long)
        if conn_mask is not None
        else None
    )

    if return_distances:
        distances = torch.tensor(distances_list, dtype=torch.float)
        return children, n_connected_components, n_leaves, parents, distances

    return children, n_connected_components, n_leaves, parents


class AgglomerativeClustering(MLCluster):
    def __init__(self,
                 n_clusters: Union[int, None] = 2,
                 metric: Union[str, Callable, nn.Module] = "euclidean",
                 metric_params: dict = None,
                 memory: Union[str, Any] = None,
                 connectivity: Union[list, tuple, torch.Tensor, Callable, nn.Module] = None,
                 compute_full_tree: Union[str, bool] = "auto",
                 linkage: Union[str, Callable, nn.Module] = "ward",
                 distance_threshold: float = None,
                 compute_distances: bool = False,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.device = device
        self.dtype = dtype

        if distance_threshold is not None and n_clusters is not None:
            raise ValueError(
                "n_clusters must be None when distance_threshold is not None."
            )
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold
        self.metric_params = metric_params or {}

        # ── Metric routing ──────────────────────────────────────────────────────
        # For string metrics: normalise name and enforce the Ward–euclidean rule.
        # "precomputed" is now handled centrally by MLCluster._create_metric
        # (sets self.metric = "precomputed"); nn.Module instances are stored
        # directly so self.metric(xi, xj) calls their forward method.
        # _metric_name / _metric_callable are kept for backward-compatibility
        # with subclasses (ROCK, LanceWilliams, ClustGeo, etc.) that access
        # them directly in their fit / transform methods.
        if isinstance(metric, str):
            metric = metric.lower()
            if isinstance(linkage, str) and linkage.lower() == "ward":
                _euclidean_variants = ("euclidean", "l2", "l2_distance", "euclid", "distance")
                if metric not in _euclidean_variants and metric != "precomputed":
                    warnings.warn(
                        "Ward linkage only supports euclidean metric. "
                        "Switching metric to 'euclidean'.",
                        UserWarning,
                    )
                    metric = "euclidean"

        self._metric_name = metric if isinstance(metric, str) else None
        self._metric_callable = (
            metric if (callable(metric) and not isinstance(metric, nn.Module)) else None
        )
        if not isinstance(metric, (str, nn.Module)) and not callable(metric):
            self._metric_name = "euclidean"

        if isinstance(metric, nn.Module):
            self.metric = metric
        else:
            # Handles str (including "precomputed"), callable, and default fallback.
            self._create_metric(metric, self.metric_params)

        self.memory = memory
        self._cache: dict = {}
        self._connectivity_raw = connectivity
        self.connectivity = None
        self.compute_full_tree = compute_full_tree
        self.compute_distances = compute_distances
        self._linkage_name = None
        self._linkage_fn: Optional[Callable] = None

        if isinstance(linkage, str):
            self._linkage_name = linkage.lower()
        elif isinstance(linkage, nn.Module):
            self._linkage_fn = linkage
        elif callable(linkage):
            _lkw = kwargs.get("linkage_params", {})
            self._linkage_fn = lambda C1, C2: linkage(C1, C2, **_lkw)
        else:
            self._linkage_name = "ward"

        self.n_clusters_ = None
        self.labels_ = None
        self.n_leaves_ = None
        self.n_connected_components_ = None
        self.n_features_in_ = None
        self.children_ = None
        self.distances_ = None
        self.X_fitted_ = None

    def _pairwise_dist(self, C1: torch.Tensor, C2: torch.Tensor) -> torch.Tensor:
        if self.metric == "precomputed":
            raise RuntimeError(
                "Cannot recompute distances with metric='precomputed'. "
                "Distance matrix must be supplied to fit()."
            )
        if isinstance(self.metric, nn.Module):
            return self.metric(C1, C2)
        D = self.metric(C1, C2)
        if D.dim() > 2:
            D = D.squeeze()
        return D

    def _compute_distances(self, X: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        if self.metric == "precomputed":
            raise RuntimeError(
                "metric='precomputed': supply a precomputed distance matrix to fit()."
            )
        if isinstance(self.metric, nn.Module):
            return self.metric(X, centers)
        D = self.metric(X, centers)
        if D.dim() > 2:
            D = D.squeeze()
        return D

    def _build_linkage_fn(self):
        name = self._linkage_name
        if name == "ward":
            def _ward(C1: torch.Tensor, C2: torch.Tensor) -> torch.Tensor:
                n1 = float(C1.size(0))
                n2 = float(C2.size(0))
                mu1 = C1.mean(dim=0)
                mu2 = C2.mean(dim=0)
                sq_dist = ((mu1 - mu2) ** 2).sum()
                return torch.sqrt((n1 * n2) / (n1 + n2) * sq_dist)
            self._linkage_fn = _ward
        elif name == "average":
            def _average(C1: torch.Tensor, C2: torch.Tensor) -> torch.Tensor:
                return self._pairwise_dist(C1, C2).mean()
            self._linkage_fn = _average
        elif name in ("complete", "maximum"):
            def _complete(C1: torch.Tensor, C2: torch.Tensor) -> torch.Tensor:
                return self._pairwise_dist(C1, C2).max()
            self._linkage_fn = _complete
        elif name == "single":
            def _single(C1: torch.Tensor, C2: torch.Tensor) -> torch.Tensor:
                return self._pairwise_dist(C1, C2).min()
            self._linkage_fn = _single
        else:
            warnings.warn(f"Unknown linkage '{name}'. Falling back to 'ward'.", UserWarning)
            self._linkage_name = "ward"
            self._build_linkage_fn()

    def linkage(self, C1: torch.Tensor, C2: torch.Tensor) -> torch.Tensor:
        return self._linkage_fn(C1, C2)

    def dist_calc(self, metric_type: str, xi: torch.Tensor, xj: torch.Tensor, **kwargs) -> torch.Tensor:
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
                xi_e = xi_s.unsqueeze(1)
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
                xi_e = xi.unsqueeze(1)
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
                xi_e = xi.unsqueeze(1)
                xj_e = xj.unsqueeze(0)
                return torch.sqrt(((xi_e - xj_e) ** 2).sum(dim=-1))
            case "total_variational_distance":
                xi_s = F.softmax(xi, dim=-1).unsqueeze(1)
                xj_s = F.softmax(xj, dim=-1).unsqueeze(0)
                return 0.5 * (xi_s - xj_s).abs().sum(dim=-1)
            case "frobenius_norm":
                xi_e = xi.unsqueeze(1)
                xj_e = xj.unsqueeze(0)
                return torch.sqrt(((xi_e - xj_e) ** 2).sum(dim=-1))
            case "log_euclidean":
                xi_e = xi.unsqueeze(1)
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

            case "cityblock":
                return torch.cdist(xi, xj, p=1)
            case "sqeuclidean" | "squared_euclidean_sklearn":
                return torch.cdist(xi, xj, p=2) ** 2
            case "seuclidean" | "standardized_euclidean":
                std = xi.std(dim=0).clamp(min=1e-9)
                xi_s = xi / std
                xj_s = xj / std
                return torch.cdist(xi_s, xj_s, p=2)

            case "braycurtis":
                xi_e = xi.unsqueeze(1)
                xj_e = xj.unsqueeze(0)
                num = (xi_e - xj_e).abs().sum(-1)
                den = (xi_e.abs() + xj_e.abs()).sum(-1).clamp(min=1e-9)
                return num / den

            case "correlation":
                xi_c = xi - xi.mean(dim=-1, keepdim=True)
                xj_c = xj - xj.mean(dim=-1, keepdim=True)
                xi_n = F.normalize(xi_c, p=2, dim=-1)
                xj_n = F.normalize(xj_c, p=2, dim=-1)
                return 1.0 - xi_n @ xj_n.T

            case "dice":
                xi_b = (xi > 0).float()
                xj_b = (xj > 0).float()
                xi_e = xi_b.unsqueeze(1)
                xj_e = xj_b.unsqueeze(0)
                tf = (xi_e * xj_e).sum(-1)  # true-true count
                num = xi_b.sum(-1, keepdim=True) + xj_b.sum(-1)  # (n,k)
                return 1.0 - 2.0 * tf / (num.view(xi.shape[0], xj.shape[0]).clamp(min=1e-9))

            case "hamming":
                xi_b = (xi > 0).float()
                xj_b = (xj > 0).float()
                xi_e = xi_b.unsqueeze(1)
                xj_e = xj_b.unsqueeze(0)
                d = float(xi.shape[-1])
                return (xi_e - xj_e).abs().sum(-1) / d

            case "jaccard":
                xi_b = (xi > 0).float()
                xj_b = (xj > 0).float()
                xi_e = xi_b.unsqueeze(1)
                xj_e = xj_b.unsqueeze(0)
                intersect = (xi_e * xj_e).sum(-1)
                union = ((xi_e + xj_e) > 0).float().sum(-1).clamp(min=1e-9)
                return 1.0 - intersect / union

            case "kulsinski":
                xi_b = (xi > 0).float()
                xj_b = (xj > 0).float()
                xi_e = xi_b.unsqueeze(1)
                xj_e = xj_b.unsqueeze(0)
                n = float(xi.shape[-1])
                tt = (xi_e * xj_e).sum(-1)
                tf = (xi_e * (1 - xj_e)).sum(-1)
                ft = ((1 - xi_e) * xj_e).sum(-1)
                return (tf + ft - tt + n) / (tf + ft + n).clamp(min=1e-9)

            case "rogerstanimoto":
                xi_b = (xi > 0).float()
                xj_b = (xj > 0).float()
                xi_e = xi_b.unsqueeze(1)
                xj_e = xj_b.unsqueeze(0)
                n = float(xi.shape[-1])
                tt = (xi_e * xj_e).sum(-1)
                ff = ((1 - xi_e) * (1 - xj_e)).sum(-1)
                tf = (xi_e * (1 - xj_e)).sum(-1)
                ft = ((1 - xi_e) * xj_e).sum(-1)
                r = tf + ft
                return 2.0 * r / (tt + ff + 2.0 * r + 1e-9)

            case "russellrao":
                xi_b = (xi > 0).float()
                xj_b = (xj > 0).float()
                xi_e = xi_b.unsqueeze(1)
                xj_e = xj_b.unsqueeze(0)
                n = float(xi.shape[-1])
                tt = (xi_e * xj_e).sum(-1)
                return (n - tt) / n

            case "sokalmichener":
                xi_b = (xi > 0).float()
                xj_b = (xj > 0).float()
                xi_e = xi_b.unsqueeze(1)
                xj_e = xj_b.unsqueeze(0)
                tf = (xi_e * (1 - xj_e)).sum(-1)
                ft = ((1 - xi_e) * xj_e).sum(-1)
                tt = (xi_e * xj_e).sum(-1)
                ff = ((1 - xi_e) * (1 - xj_e)).sum(-1)
                r = tf + ft
                return 2.0 * r / (tt + ff + 2.0 * r + 1e-9)

            case "sokalsneath":
                xi_b = (xi > 0).float()
                xj_b = (xj > 0).float()
                xi_e = xi_b.unsqueeze(1)
                xj_e = xj_b.unsqueeze(0)
                tt = (xi_e * xj_e).sum(-1)
                tf = (xi_e * (1 - xj_e)).sum(-1)
                ft = ((1 - xi_e) * xj_e).sum(-1)
                r = tf + ft
                return 2.0 * r / (tt + 2.0 * r + 1e-9)

            case "yule":
                xi_b = (xi > 0).float()
                xj_b = (xj > 0).float()
                xi_e = xi_b.unsqueeze(1)
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

    def _resolve_connectivity(self, X: torch.Tensor) -> Optional[torch.Tensor]:
        raw = self._connectivity_raw
        if raw is None:
            return None
        n = X.size(0)
        if isinstance(raw, (list, tuple, torch.Tensor)):
            conn = torch.as_tensor(raw, device=X.device, dtype=torch.float)
            if conn.shape != (n, n):
                raise ValueError(
                    f"connectivity matrix must be of shape ({n}, {n}), "
                    f"got {tuple(conn.shape)}."
                )
            return conn != 0

        if isinstance(raw, nn.Module):
            result = raw(X)
        elif callable(raw):
            result = raw(X)
        else:
            raise TypeError(
                f"connectivity must be array-like, callable, or nn.Module; "
                f"got {type(raw).__name__}."
            )

        conn = torch.as_tensor(result, device=X.device, dtype=torch.float)
        if conn.shape != (n, n):
            raise ValueError(
                f"connectivity callable must return a ({n}, {n}) matrix; "
                f"got {tuple(conn.shape)}."
            )
        return conn != 0

    def _are_connected(self, conn_mask, node_a, node_b, member_map) -> bool:
        if conn_mask is None:
            return True
        idx_a = member_map[node_a].tolist()
        idx_b = member_map[node_b].tolist()
        sub = conn_mask[idx_a][:, idx_b]
        return bool(sub.any().item())

    def _count_connected_components(self, n: int, conn_mask) -> int:
        if conn_mask is None:
            return 1
        parent = list(range(n))
        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for i in range(n):
            for j in range(i + 1, n):
                if conn_mask[i, j].item():
                    ri, rj = _find(i), _find(j)
                    if ri != rj:
                        parent[ri] = rj
        return len({_find(i) for i in range(n)})

    def _init_module(self, X: torch.Tensor):
        self.n_features_in_ = X.size(-1)
        n_samples = X.size(0)
        if isinstance(self.compute_full_tree, str) and self.compute_full_tree.lower() == "auto":
            if (self.distance_threshold is not None
                    or (self.n_clusters is not None
                        and self.n_clusters < max(100, int(0.02 * n_samples)))):
                self.compute_full_tree = True
            else:
                self.compute_full_tree = False
        if self.distance_threshold is not None and not self.compute_full_tree:
            self.compute_full_tree = True
        if self._linkage_name is not None and self._linkage_fn is None:
            self._build_linkage_fn()
        return self

    def fit(self, data_or_X, **kwargs):
        X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
        self._init_module(X)

        n_samples = X.shape[0]

        if self.metric == "precomputed":
            if X.shape[0] != X.shape[1]:
                raise ValueError(
                    "When metric='precomputed', data_or_X must be a square "
                    f"distance matrix; got shape {tuple(X.shape)}."
                )
            dist_matrix = X
            self.X_fitted_ = None
        else:
            # Precompute distance matrix once for all non-Ward linkages or if needed.
            # This is a major optimization to avoid repeated self.metric calls.
            dist_matrix = self._compute_distances(X, X)
            self.X_fitted_ = X

        conn_mask = self._resolve_connectivity(X)
        cache_key = None
        if self.memory is not None:
            _sig = (
                tuple(X.shape),
                round(float(X.mean()), 6),
                round(float(X.std()), 6),
                self._linkage_name,
                self._metric_name,
                self.n_clusters,
                self.distance_threshold,
            )
            cache_key = _sig
            if cache_key in self._cache:
                (
                    self.children_,
                    self.distances_,
                    self.labels_,
                    self.n_clusters_,
                    self.n_leaves_,
                    self.n_connected_components_,
                ) = self._cache[cache_key]
                return self

        if (self._linkage_name == "ward" and self.metric != "precomputed"):
            _conn_raw = (
                self._connectivity_raw
                if isinstance(self._connectivity_raw, (list, tuple, torch.Tensor))
                else conn_mask
            )
            _ret = ward_tree(
                X,
                connectivity=_conn_raw,
                n_clusters=(
                    self.n_clusters
                    if not self.compute_full_tree
                    else None
                ),
                return_distances=(
                    self.compute_distances or self.distance_threshold is not None
                ),
            )

            if self.compute_distances or self.distance_threshold is not None:
                (
                    self.children_,
                    self.n_connected_components_,
                    self.n_leaves_,
                    _parents,
                    _ward_dists,
                ) = _ret
                self.distances_ = _ward_dists
            else:
                (
                    self.children_,
                    self.n_connected_components_,
                    self.n_leaves_,
                    _parents,
                ) = _ret
                self.distances_ = torch.tensor([], dtype=self.dtype)

            if self.distance_threshold is not None and len(self.distances_) > 0:
                keep = self.distances_ < self.distance_threshold
                self.children_ = self.children_[keep]
                self.distances_ = self.distances_[keep]

            members: Dict[int, List[int]] = {i: [i] for i in range(n_samples)}
            next_nid = n_samples
            for ch in self.children_.tolist():
                a_id, b_id = int(ch[0]), int(ch[1])
                members[next_nid] = members.pop(a_id, [a_id]) + members.pop(b_id, [b_id])
                next_nid += 1

            active_nodes_wt = sorted(members.keys())
            self.n_clusters_ = len(active_nodes_wt)
            self.labels_ = torch.zeros(n_samples, dtype=torch.long, device=X.device)
            for cl_idx, nid in enumerate(active_nodes_wt):
                for sample_idx in members[nid]:
                    self.labels_[sample_idx] = cl_idx

            if self.memory is not None and cache_key is not None:
                self._cache[cache_key] = (
                    self.children_,
                    self.distances_,
                    self.labels_,
                    self.n_clusters_,
                    self.n_leaves_,
                    self.n_connected_components_,
                )
            return self

        current_clusters: Dict[int, torch.Tensor] = {
            i: torch.tensor([i], device=X.device, dtype=torch.long)
            for i in range(n_samples)
        }
        active_nodes: List[int] = list(range(n_samples))
        next_node_id: int = n_samples

        children_list: List[List[int]] = []
        distances_list: List[float] = []

        self.n_leaves_ = n_samples

        # Efficiently maintain distance matrix between active clusters
        active_dist = dist_matrix.clone() if dist_matrix is not None else self._compute_distances(X, X)
        active_conn = conn_mask.clone() if conn_mask is not None else None
        
        # cluster_id_to_idx maps the next_node_id to its position in active_dist
        # This list will hold the original indices or merged cluster IDs
        # active_nodes will track the current cluster IDs (original indices or next_node_id)
        
        target_clusters: int = self.n_clusters if self.n_clusters is not None else 1
        while len(active_nodes) > target_clusters:
            # 1. Find the minimum distance in the active distance matrix
            # Mask out diagonal and already merged clusters
            mask = torch.eye(active_dist.size(0), device=active_dist.device, dtype=torch.bool)
            masked_dist = active_dist.clone()
            masked_dist[mask] = float('inf')
            
            # If connectivity constraint is present, mask out unconnected pairs
            if active_conn is not None:
                masked_dist[~active_conn] = float('inf')

            min_val, flat_idx = torch.min(masked_dist.view(-1), dim=0)
            min_dist = min_val.item()
            if min_dist == float('inf'):
                break # No more connected clusters to merge

            i_idx = flat_idx // active_dist.size(0)
            j_idx = flat_idx % active_dist.size(0)
            
            node_a = active_nodes[i_idx]
            node_b = active_nodes[j_idx]
            
            if self.distance_threshold is not None and min_dist >= self.distance_threshold:
                break

            # 2. Merge nodes
            pair = (node_a, node_b)
            children_list.append([node_a, node_b])
            if self.compute_distances or self.distance_threshold is not None:
                distances_list.append(min_dist)

            new_members = torch.cat([current_clusters[node_a], current_clusters[node_b]])
            current_clusters[next_node_id] = new_members
            
            # 3. Update active distance matrix using Lance-Williams or direct reassessment
            # We calculate the distances from the new cluster to all other clusters
            # For simplicity and correctness with custom metrics, we use a vectorized approach:
            lname = self._linkage_name or "average"
            
            # Extract distances of node_a and node_b to all other nodes
            dist_a = active_dist[i_idx]
            dist_b = active_dist[j_idx]
            
            if lname == "single":
                new_cluster_dist = torch.min(dist_a, dist_b)
            elif lname in ("complete", "maximum"):
                new_cluster_dist = torch.max(dist_a, dist_b)
            elif lname == "average":
                n_a = float(current_clusters[node_a].size(0))
                n_b = float(current_clusters[node_b].size(0))
                new_cluster_dist = (n_a * dist_a + n_b * dist_b) / (n_a + n_b)
            else: # Fallback to mean of pooled distances
                new_cluster_dist = (dist_a + dist_b) / 2.0
            
            # 4. Remove i_idx and j_idx, add next_node_id
            # To avoid complex index management, we update active_dist in place
            # by replacing i_idx with the new cluster and deleting j_idx
            # But deleting is slow in tensors. A better way for small N is just masking.
            # For our purposes, we'll rebuild active_dist or use masking.
            
            # Masking approach:
            active_dist[i_idx, :] = new_cluster_dist
            active_dist[:, i_idx] = new_cluster_dist
            active_dist[j_idx, :] = float('inf')
            active_dist[:, j_idx] = float('inf')
            active_dist[i_idx, i_idx] = 0.0
            
            active_nodes[i_idx] = next_node_id
            active_nodes[j_idx] = -1 # Mark as merged

            if active_conn is not None:
                new_conn = active_conn[i_idx] | active_conn[j_idx]
                active_conn[i_idx, :] = new_conn
                active_conn[:, i_idx] = new_conn
                active_conn[j_idx, :] = False
                active_conn[:, j_idx] = False
                active_conn[i_idx, i_idx] = False
            
            # Clean up active_nodes and active_dist occasionally or at the end
            if next_node_id % 100 == 0:
                 # Compaction to keep tensors small
                 valid_mask = torch.tensor([n != -1 for n in active_nodes], device=active_dist.device)
                 active_dist = active_dist[valid_mask][:, valid_mask]
                 if active_conn is not None:
                     active_conn = active_conn[valid_mask][:, valid_mask]
                 active_nodes = [n for n in active_nodes if n != -1]

            next_node_id += 1
            num_clusters = sum(1 for n in active_nodes if n != -1)
            if num_clusters <= target_clusters:
                break

        # Final active nodes
        active_nodes = [n for n in active_nodes if n != -1]

        self.n_clusters_ = len(active_nodes)
        self.children_ = (
            torch.tensor(children_list, dtype=torch.long)
            if children_list
            else torch.zeros((0, 2), dtype=torch.long)
        )
        self.distances_ = (
            torch.tensor(distances_list, dtype=self.dtype)
            if distances_list
            else torch.tensor([], dtype=self.dtype)
        )
        self.labels_ = torch.zeros(n_samples, dtype=torch.long, device=X.device)
        for cluster_idx, node_id in enumerate(active_nodes):
            self.labels_[current_clusters[node_id]] = cluster_idx
        self.n_connected_components_ = self._count_connected_components(n_samples, conn_mask)
        if self.memory is not None and cache_key is not None:
            self._cache[cache_key] = (
                self.children_, self.distances_, self.labels_,
                self.n_clusters_, self.n_leaves_, self.n_connected_components_,
            )
        return self

    def fit_predict(self, X: torch.Tensor, **kwargs) -> torch.Tensor:
        self.fit(X, **kwargs)
        return self.labels_

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if self.labels_ is None:
            raise ValueError("Model must be fitted before predicting.")
        if self.metric == "precomputed":
            raise RuntimeError(
                "predict() is not supported with metric='precomputed'. "
                "Re-fit on the combined distance matrix to assign new points."
            )
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        return self._assign_to_nearest_cluster(X)

    def transform(self, X: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.labels_ is None:
            raise ValueError("Model must be fitted before transforming.")
        if self.metric == "precomputed":
            raise RuntimeError(
                "transform() is not supported with metric='precomputed'."
            )
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        return self._compute_distances_to_clusters(X)

    def fit_transform(self, X: torch.Tensor, **kwargs) -> torch.Tensor:
        self.fit(X, **kwargs)
        return self.transform(X)
    def _assign_to_nearest_cluster(self, X: torch.Tensor) -> torch.Tensor:
        if self.X_fitted_ is None:
            raise RuntimeError(
                "Training data was not stored (metric='precomputed'). "
                "Cannot perform predict()."
            )
        dist_matrix = self._compute_distances(X, self.X_fitted_)
        nearest = torch.argmin(dist_matrix, dim=1)
        return self.labels_[nearest]

    def _compute_distances_to_clusters(self, X: torch.Tensor) -> torch.Tensor:
        if self.X_fitted_ is None:
            raise RuntimeError(
                "Training data was not stored. Cannot perform transform()."
            )
        centroids = torch.stack([
            self.X_fitted_[self.labels_ == k].mean(dim=0)
            for k in range(self.n_clusters_)
        ], dim=0)
        return self._compute_distances(X, centroids)


class FeatureAgglomeration(AgglomerativeClustering):
    def __init__(self,
                 n_clusters: Union[int, None] = 2,
                 metric: Union[str, Callable, nn.Module] = "euclidean",
                 metric_params: dict = None,
                 memory: Union[str, Any] = None,
                 connectivity: Union[list, tuple, torch.Tensor, Callable, nn.Module] = None,
                 compute_full_tree: Union[str, bool] = "auto",
                 linkage: Union[str, Callable, nn.Module] = "ward",
                 pooling_func: Union[str, Callable, nn.Module] = torch.mean,
                 distance_threshold: float = None,
                 compute_distances: bool = False,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(
            n_clusters=n_clusters,
            metric=metric,
            metric_params=metric_params,
            memory=memory,
            connectivity=connectivity,
            compute_full_tree=compute_full_tree,
            linkage=linkage,
            distance_threshold=distance_threshold,
            compute_distances=compute_distances,
            device=device,
            dtype=dtype,
            *args, **kwargs
        )
        self._pooling_func_raw = pooling_func
        self.pooling_func = self._resolve_pooling(pooling_func)
        self.feature_names_in_ = None

    @staticmethod
    def _resolve_pooling(pooling_func) -> Callable:
        if isinstance(pooling_func, nn.Module):
            def _module_pool(mat: torch.Tensor) -> torch.Tensor:
                out = pooling_func(mat)
                return out.squeeze(-1) if out.dim() > 1 else out
            return _module_pool

        if callable(pooling_func) and not isinstance(pooling_func, str):
            import inspect as _inspect
            try:
                sig = _inspect.signature(pooling_func)
                has_dim = "dim" in sig.parameters or any(
                    p.kind == _inspect.Parameter.VAR_KEYWORD
                    for p in sig.parameters.values()
                )
            except (ValueError, TypeError):
                has_dim = False

            def _callable_pool(mat: torch.Tensor) -> torch.Tensor:
                if has_dim:
                    result = pooling_func(mat, dim=1)
                    return result.values if hasattr(result, "values") else result
                else:
                    return pooling_func(mat)
            return _callable_pool

        if isinstance(pooling_func, str):
            name = pooling_func.lower().strip()
            _pool_map = {
                "mean":   lambda mat: mat.mean(dim=1),
                "max":    lambda mat: mat.max(dim=1).values,
                "min":    lambda mat: mat.min(dim=1).values,
                "sum":    lambda mat: mat.sum(dim=1),
                "median": lambda mat: mat.median(dim=1).values,
                "std":    lambda mat: mat.std(dim=1),
                "var":    lambda mat: mat.var(dim=1),
                "prod":   lambda mat: mat.prod(dim=1),
                "norm":   lambda mat: mat.norm(p=2, dim=1),
            }
            if name not in _pool_map:
                raise ValueError(
                    f"Unknown pooling_func string '{pooling_func}'. "
                    f"Choose from: {list(_pool_map)}."
                )
            return _pool_map[name]

        return lambda mat: mat.mean(dim=1)

    def fit(self, data_or_X, **kwargs):
        if not isinstance(data_or_X, torch.Tensor):
            raise TypeError(
                f"data_or_X must be a torch.Tensor, got {type(data_or_X).__name__}. "
                "Convert your input with torch.tensor(...) or torch.from_numpy(...) "
                "before calling fit()."
            )
        X = data_or_X.to(device=self.device, dtype=self.dtype)
        self.feature_names_in_ = None
        if self.metric == "precomputed":
            super().fit(X, **kwargs)
        else:
            super().fit(X.T, **kwargs)
        return self

    def transform(self, X, **kwargs) -> torch.Tensor:
        if self.labels_ is None:
            raise ValueError("FeatureAgglomeration must be fitted before transform().")
        if not isinstance(X, torch.Tensor):
            raise TypeError(
                f"X must be a torch.Tensor, got {type(X).__name__}. "
                "Convert your input with torch.tensor(...) or torch.from_numpy(...) "
                "before calling transform()."
            )
        X = X.to(device=self.device, dtype=self.dtype)
        n_samples, n_features = X.shape
        if n_features != self.n_features_in_:
            raise ValueError(
                f"X has {n_features} features but was fitted with "
                f"{self.n_features_in_} features."
            )
        out_cols: List[torch.Tensor] = []
        for k in range(self.n_clusters_):
            feat_mask = self.labels_ == k
            X_k = X[:, feat_mask]
            if X_k.size(1) == 0:
                col = torch.zeros(n_samples, device=X.device, dtype=X.dtype)
            elif X_k.size(1) == 1:
                col = X_k.squeeze(1)
            else:
                col = self.pooling_func(X_k)
            out_cols.append(col)
        return torch.stack(out_cols, dim=1)

    def fit_transform(self, X, y=None, **kwargs) -> torch.Tensor:
        self.fit(X, **kwargs)
        return self.transform(X)

    def inverse_transform(self, X_reduced) -> torch.Tensor:
        if self.labels_ is None:
            raise ValueError(
                "FeatureAgglomeration must be fitted before inverse_transform()."
            )
        if not isinstance(X_reduced, torch.Tensor):
            raise TypeError(
                f"X_reduced must be a torch.Tensor, got {type(X_reduced).__name__}. "
                "Convert your input with torch.tensor(...) or torch.from_numpy(...) "
                "before calling inverse_transform()."
            )
        X_red = X_reduced.to(device=self.device, dtype=self.dtype)
        n_samples = X_red.size(0)
        if X_red.size(1) != self.n_clusters_:
            raise ValueError(
                f"X_reduced has {X_red.size(1)} columns but n_clusters_="
                f"{self.n_clusters_}."
            )
        X_out = torch.zeros(
            n_samples, self.n_features_in_,
            device=self.device, dtype=self.dtype,
        )
        for k in range(self.n_clusters_):
            feat_mask = self.labels_ == k
            X_out[:, feat_mask] = X_red[:, k: k + 1]
        return X_out


class Birch(MLCluster):
    class _CFSubcluster:
        __slots__ = ("n", "ls", "ss", "child")

        def __init__(self,
                     linear_sum: torch.Tensor,
                     squared_sum: float,
                     count: int,
                     child=None):
            self.n: int = count
            self.ls: torch.Tensor = linear_sum
            self.ss: float = squared_sum
            self.child = child

        def absorb(self, x: torch.Tensor) -> None:
            self.n += 1
            self.ls = self.ls + x
            self.ss += float((x * x).sum().item())

        def centroid(self) -> torch.Tensor:
            return self.ls / self.n

        def radius(self) -> float:
            c = self.centroid()
            val = (self.ss / self.n) - float((c * c).sum().item())
            return math.sqrt(max(0.0, val))

        @staticmethod
        def from_point(x: torch.Tensor,
                       child=None) -> "Birch._CFSubcluster":
            return Birch._CFSubcluster(
                linear_sum=x.clone(),
                squared_sum=float((x * x).sum().item()),
                count=1,
                child=child,
            )

        @staticmethod
        def aggregate(subclusters: list,
                      child=None) -> "Birch._CFSubcluster":
            ls = torch.zeros_like(subclusters[0].ls)
            ss = 0.0
            n = 0
            for sc in subclusters:
                ls = ls + sc.ls
                ss += sc.ss
                n += sc.n
            return Birch._CFSubcluster(ls, ss, n, child=child)

    class _CFNode:
        __slots__ = ("threshold", "branching_factor", "is_leaf",
                     "subclusters", "prev", "next")

        def __init__(self, threshold: float, branching_factor: int, is_leaf: bool):
            self.threshold = threshold
            self.branching_factor = branching_factor
            self.is_leaf = is_leaf
            self.subclusters: list = []
            self.prev: "Optional[Birch._CFNode]" = None
            self.next: "Optional[Birch._CFNode]" = None

        def insert(self, x: torch.Tensor) -> "Optional[Birch._CFSubcluster]":
            return self._insert_leaf(x) if self.is_leaf else self._insert_internal(x)

        def _insert_leaf(self, x: torch.Tensor) -> "Optional[Birch._CFSubcluster]":
            if not self.subclusters:
                self.subclusters.append(Birch._CFSubcluster.from_point(x))
                return None
            centroids = torch.stack([sc.centroid() for sc in self.subclusters])
            dists = torch.norm(centroids - x.unsqueeze(0), dim=1)
            idx = int(torch.argmin(dists).item())
            sc = self.subclusters[idx]
            new_n = sc.n + 1
            new_ls = sc.ls + x
            new_ss = sc.ss + float((x * x).sum().item())
            new_c = new_ls / new_n
            new_r = math.sqrt(max(0.0, (new_ss / new_n) - float((new_c * new_c).sum().item())))
            if new_r <= self.threshold:
                sc.absorb(x)
                return None
            self.subclusters.append(Birch._CFSubcluster.from_point(x))
            if len(self.subclusters) > self.branching_factor:
                return self._split()
            return None

        def _insert_internal(self, x: torch.Tensor) -> "Optional[Birch._CFSubcluster]":
            centroids = torch.stack([sc.centroid() for sc in self.subclusters])
            dists = torch.norm(centroids - x.unsqueeze(0), dim=1)
            idx = int(torch.argmin(dists).item())
            sc = self.subclusters[idx]

            split_sc = sc.child.insert(x)
            sc.absorb(x)
            if split_sc is not None:
                self.subclusters.append(split_sc)
                if len(self.subclusters) > self.branching_factor:
                    return self._split()
            return None


        def _split(self) -> "Birch._CFSubcluster":
            n = len(self.subclusters)
            centroids = torch.stack([sc.centroid() for sc in self.subclusters])

            dist_mat = torch.cdist(centroids, centroids)
            dist_mat.fill_diagonal_(float("-inf"))
            flat_idx = int(torch.argmax(dist_mat).item())
            seed_a, seed_b = divmod(flat_idx, n)

            da = torch.norm(centroids - centroids[seed_a].unsqueeze(0), dim=1)
            db = torch.norm(centroids - centroids[seed_b].unsqueeze(0), dim=1)
            assigned_a = (da <= db)

            group_a = [sc for i, sc in enumerate(self.subclusters) if assigned_a[i]]
            group_b = [sc for i, sc in enumerate(self.subclusters) if not assigned_a[i]]

            if not group_a:
                group_a = [group_b.pop(0)]
            if not group_b:
                group_b = [group_a.pop()]

            sibling = Birch._CFNode(self.threshold, self.branching_factor, self.is_leaf)
            sibling.subclusters = group_b
            self.subclusters = group_a
            if self.is_leaf:
                sibling.next = self.next
                sibling.prev = self
                if self.next is not None:
                    self.next.prev = sibling
                self.next = sibling
            return Birch._CFSubcluster.aggregate(sibling.subclusters, child=sibling)

    def __init__(self,
                 threshold: float = 0.5,
                 branching_factor: int = 50,
                 n_clusters: Union[int, "MLCluster", None] = 3,
                 compute_labels: bool = True,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters
        self.compute_labels = compute_labels
        self.device = device
        self.dtype = dtype

        self.root_: Optional[Birch._CFNode] = None
        self.dummy_leaf_: Optional[Birch._CFNode] = None
        self.subcluster_centers_: Optional[torch.Tensor] = None
        self.subcluster_labels_: Optional[torch.Tensor] = None
        self.labels_: Optional[torch.Tensor] = None
        self.n_features_in_: Optional[int] = None

    def _validate_tensor(self, X: torch.Tensor, name: str = "X") -> torch.Tensor:
        if not isinstance(X, torch.Tensor):
            raise TypeError(
                f"{name} must be a torch.Tensor, got {type(X).__name__}. "
                "Convert with torch.tensor(...) or torch.from_numpy(...) first."
            )
        return X.to(device=self.device, dtype=self.dtype)

    def _init_module(self, X: torch.Tensor) -> "Birch":
        self.n_features_in_ = X.size(-1)
        return self

    def _new_tree(self, n_features: int) -> "Birch._CFNode":
        sentinel = Birch._CFNode(self.threshold, self.branching_factor, is_leaf=True)
        self.dummy_leaf_ = sentinel
        first_leaf = Birch._CFNode(self.threshold, self.branching_factor, is_leaf=True)
        sentinel.next = first_leaf
        first_leaf.prev = sentinel
        return first_leaf

    def _iter_leaves(self) -> "Iterable[Birch._CFNode]":
        curr = self.dummy_leaf_.next
        while curr is not None:
            yield curr
            curr = curr.next

    def _extract_subcluster_centers(self) -> torch.Tensor:
        centroids = [
            sc.centroid()
            for leaf in self._iter_leaves()
            for sc in leaf.subclusters
        ]
        if not centroids:
            raise RuntimeError("CF Tree is empty — no subclusters found after fit.")
        return torch.stack(centroids, dim=0)

    def _insert_sample(self, x: torch.Tensor) -> None:
        split_sc = self.root_.insert(x)
        if split_sc is None:
            return
        old_root = self.root_
        new_root = Birch._CFNode(self.threshold, self.branching_factor, is_leaf=False)
        old_summary = Birch._CFSubcluster.aggregate(old_root.subclusters, child=old_root)
        new_root.subclusters = [old_summary, split_sc]
        self.root_ = new_root

    def _global_cluster(self) -> torch.Tensor:
        centers = self.subcluster_centers_
        n_sub = centers.size(0)
        if self.n_clusters is None:
            return torch.arange(n_sub, device=self.device, dtype=torch.long)
        elif isinstance(self.n_clusters, int):
            k = min(self.n_clusters, n_sub)
            agg = AgglomerativeClustering(n_clusters=k, linkage="ward",
                                          device=self.device, dtype=self.dtype)
            return agg.fit_predict(centers)
        elif isinstance(self.n_clusters, MLCluster):
            result = self.n_clusters.fit_predict(centers)
            if not isinstance(result, torch.Tensor):
                result = torch.tensor(result, device=self.device, dtype=torch.long)
            return result.to(dtype=torch.long)
        else:
            raise TypeError(
                f"n_clusters must be None, int, or an MLCluster instance; "
                f"got {type(self.n_clusters).__name__}."
            )

    def _map_to_labels(self, X: torch.Tensor) -> torch.Tensor:
        dists = torch.cdist(X, self.subcluster_centers_)
        nearest = torch.argmin(dists, dim=1)
        return self.subcluster_labels_[nearest]

    def fit(self, X: torch.Tensor, y: Any = None) -> "Birch":
        X = self._validate_tensor(X, "X")
        self._init_module(X)
        self.root_ = self._new_tree(X.size(-1))
        for i in range(X.size(0)):
            self._insert_sample(X[i])
        self.subcluster_centers_ = self._extract_subcluster_centers()
        self.subcluster_labels_ = self._global_cluster()
        if self.compute_labels:
            self.labels_ = self._map_to_labels(X)
        return self

    def partial_fit(self, X: torch.Tensor, y: Any = None) -> "Birch":
        X = self._validate_tensor(X, "X")
        if self.root_ is None:
            self._init_module(X)
            self.root_ = self._new_tree(X.size(-1))
        else:
            if X.size(-1) != self.n_features_in_:
                raise ValueError(
                    f"partial_fit received {X.size(-1)} features but the model "
                    f"was initialised with {self.n_features_in_} features."
                )
        for i in range(X.size(0)):
            self._insert_sample(X[i])
        self.subcluster_centers_ = self._extract_subcluster_centers()
        self.subcluster_labels_ = self._global_cluster()
        if self.compute_labels:
            self.labels_ = self._map_to_labels(X)
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if self.subcluster_centers_ is None:
            raise ValueError("Birch must be fitted before predict().")
        X = self._validate_tensor(X, "X")
        return self._map_to_labels(X)

    def transform(self, X: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.subcluster_centers_ is None:
            raise ValueError("Birch must be fitted before transform().")
        X = self._validate_tensor(X, "X")
        return torch.cdist(X, self.subcluster_centers_)

    def fit_predict(self, X: torch.Tensor, y: Any = None) -> torch.Tensor:
        self.fit(X, y)
        return self.labels_

    def fit_transform(self, X: torch.Tensor, y: Any = None, **kwargs) -> torch.Tensor:
        self.fit(X, y)
        return self.transform(X)

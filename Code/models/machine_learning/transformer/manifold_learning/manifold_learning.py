import torch
import torch.nn as nn
import math
from typing import Optional, Callable, Union, Any, List, Tuple, Literal
from .....models.utils import MLTransform
from .....models.machine_learning.regression.knn.knn import BallTree, KDTree
from torch.func import vmap
import joblib

__all__ = [
    "TSNE",
    "Isomap",
    "LocallyLinearEmbedding",
    "MDS",
    "ClassicalMDS",
    "SpectralEmbedding",
]

def _binary_search_perplexity(
        distances: torch.Tensor,
        desired_perplexity: float,
        tol: float = 1e-5,
        max_iter: int = 50,
) -> torch.Tensor:
    """
    Binary search for beta=1/(2*sigma^2) so conditional probs have target perplexity.
    distances: (n_samples, n_samples) squared distances. Row i: distances from i.
    Returns conditional P (n_samples, n_samples) with zero diagonal.
    """
    n = distances.shape[0]
    P = torch.zeros(n, n, device=distances.device, dtype=distances.dtype)
    for i in range(n):
        d_i = distances[i].clone()
        d_i[i] = float("inf")
        beta_min, beta_max = 1e-10, 1e10
        beta = 1.0
        for _ in range(max_iter):
            p_i = torch.exp(-d_i * beta)
            p_i[i] = 0.0
            p_sum = p_i.sum()
            if p_sum < 1e-20:
                p_i = torch.zeros_like(p_i)
                p_i[i] = 1.0
                break
            p_i = p_i / p_sum
            ent = -(p_i * (torch.log(p_i + 1e-20))).sum().item()
            perplexity = math.exp(ent)
            diff = perplexity - desired_perplexity
            if abs(diff) < tol:
                break
            if diff > 0:
                beta_min = beta
                beta = (beta + beta_max) / 2 if beta_max < 1e9 else beta * 2
            else:
                beta_max = beta
                beta = (beta_min + beta) / 2 if beta_min > 1e-10 else beta / 2
        P[i] = p_i
    return P


def _joint_probabilities(
        distances: torch.Tensor,
        perplexity: float,
        verbose: int = 0,
) -> torch.Tensor:
    """Compute joint probabilities P from squared distances (condensed or full)."""
    if distances.dim() == 1:
        n = int((1 + math.sqrt(1 + 8 * len(distances))) / 2)
        D = torch.zeros(n, n, device=distances.device, dtype=distances.dtype)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                D[i, j] = distances[idx]
                D[j, i] = distances[idx]
                idx += 1
    else:
        D = distances
    conditional_P = _binary_search_perplexity(D, perplexity)
    P = conditional_P + conditional_P.T
    P = P / P.sum().clamp(min=1e-20)
    P = torch.clamp(P, min=1e-12)
    return P

def _joint_probabilities(
        distances: torch.Tensor,
        perplexity: float,
        verbose: int = 0,
) -> torch.Tensor:
    """Compute joint probabilities P from squared distances (condensed or full)."""
    if distances.dim() == 1:
        n = int((1 + math.sqrt(1 + 8 * len(distances))) / 2)
        D = torch.zeros(n, n, device=distances.device, dtype=distances.dtype)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                D[i, j] = distances[idx]
                D[j, i] = distances[idx]
                idx += 1
    else:
        D = distances
    conditional_P = _binary_search_perplexity(D, perplexity)
    P = conditional_P + conditional_P.T
    P = P / P.sum().clamp(min=1e-20)
    P = torch.clamp(P, min=1e-12)
    return P

class TSNE(MLTransform):
    def __init__(
            self,
            n_components: int = 2,
            perplexity: float = 30.0,
            early_exaggeration: float = 12.0,
            learning_rate: Union[Literal["auto"], float] = "auto",
            max_iter: int = 1000,
            n_iter_without_progress: int = 300,
            min_grad_norm: float = 1e-7,
            metric: Union[str, Callable, nn.Module] = "euclidean",
            metric_params: Optional[dict] = None,
            init: Union[Literal["random", "pca"], tuple, list, torch.Tensor] = "pca",
            verbose: int = 0,
            random_state: Optional[Union[int, torch.Generator]] = None,
            method: Union[Literal["barnes_hut", "exact"], Callable, nn.Module] = "barnes_hut",
            method_params: Optional[dict] = None,
            angle: float = 0.5,
            n_jobs: Optional[int] = None,
            device: Union[str, torch.device] = "cpu",
            dtype: torch.dtype = torch.float,
            *args: Any,
            **kwargs: Any,
    ) -> None:
        super().__init__()
        from .....models.utils import MLCluster
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.metric_params = metric_params
        self.init = init
        self.verbose = verbose
        self.random_state = random_state
        self.method = method if isinstance(method, str) else method
        self._method_params = method_params or {}
        self.angle = angle
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype
        self.metric = MLCluster()._create_metric(metric, metric_params or {})
        self.embedding_ = None
        self.kl_divergence_ = None
        self.n_features_in_ = None
        self.learning_rate_ = None
        self.n_iter_ = 0

    def _get_metric_fn(self):
        """Return the actual metric from MLCluster (precomputed string or callable)."""
        return self.metric.metric if hasattr(self.metric, "metric") else self.metric

    def _kl_divergence(
            self,
            Y: torch.Tensor,
            P: torch.Tensor,
            degrees_of_freedom: int,
            compute_error: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute KL divergence and gradient. Y: (n, n_components), P: (n, n)."""
        n, n_comp = Y.shape
        dof = float(degrees_of_freedom)
        diff = Y.unsqueeze(1) - Y.unsqueeze(0)
        sqdist = diff.pow(2).sum(dim=-1)
        dist = (sqdist / dof + 1.0).pow(-(dof + 1) / 2)
        dist = dist * (1 - torch.eye(n, device=Y.device, dtype=Y.dtype))
        Q = dist / (2.0 * dist.sum().clamp(min=1e-20))
        Q = Q.clamp(min=1e-12)
        if compute_error:
            kl = (P * (torch.log(P.clamp(min=1e-12)) - torch.log(Q))).sum()
        else:
            kl = torch.tensor(float("nan"), device=Y.device, dtype=Y.dtype)
        PQd = (P - Q) * dist
        grad = torch.zeros_like(Y)
        for i in range(n):
            grad[i] = (PQd[i].unsqueeze(1) * diff[i]).sum(dim=0)
        c = 2.0 * (dof + 1) / dof
        grad = grad * c
        return kl, grad

    def fit(
            self,
            data_or_X: Union[torch.Tensor, Any],
            **kwargs: Any,
    ) -> "TSNE":
        """Fit X into an embedded space."""
        X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        n_samples = X.shape[0]
        perplexity = min(self.perplexity, n_samples - 1) if n_samples > 1 else 1.0
        if perplexity < 1:
            raise ValueError(
                f"perplexity ({self.perplexity}) must be less than n_samples ({n_samples})"
            )
        _metric_fn = self._get_metric_fn()
        if _metric_fn == "precomputed":
            if isinstance(self.init, str) and self.init == "pca":
                raise ValueError('init="pca" cannot be used with metric="precomputed".')
            if X.shape[0] != X.shape[1]:
                raise ValueError("X should be a square distance matrix for precomputed metric.")
            distances = X.clone()
        else:
            self.n_features_in_ = X.shape[1]
            distances = _metric_fn(X, X)
            if isinstance(distances, torch.Tensor):
                distances = distances.pow(2)
            else:
                distances = torch.as_tensor(distances, device=self.device, dtype=self.dtype).pow(2)
        distances = distances.clamp(min=0)
        P = _joint_probabilities(distances, perplexity, self.verbose)
        if self.learning_rate == "auto":
            self.learning_rate_ = max(n_samples / self.early_exaggeration / 4, 50.0)
        else:
            self.learning_rate_ = float(self.learning_rate)
        if isinstance(self.init, (tuple, list, torch.Tensor)):
            Y = torch.as_tensor(self.init, device=self.device, dtype=self.dtype)
            if Y.shape != (n_samples, self.n_components):
                raise ValueError(f"init shape {Y.shape} != ({n_samples}, {self.n_components})")
        elif self.init == "pca":
            if _metric_fn == "precomputed":
                raise ValueError('init="pca" cannot be used with metric="precomputed".')
            from .....models.machine_learning.transformer.pca import PCA
            pca = PCA(n_components=self.n_components, device=self.device, dtype=self.dtype)
            Y = pca.fit_transform(X)
            Y = Y / (Y[:, 0].std().clamp(min=1e-12)) * 1e-4
        else:
            g = self.random_state
            if isinstance(g, int):
                g = torch.Generator(device=self.device).manual_seed(g)
            Y = 1e-4 * torch.randn(
                n_samples, self.n_components,
                device=self.device, dtype=self.dtype, generator=g
            )
        dof = max(self.n_components - 1, 1)
        P_exag = P * self.early_exaggeration
        n_iter_check = 50
        n_iter_without_progress = (self.n_iter_without_progress // n_iter_check) * n_iter_check
        exploration_max = 250
        params = Y.clone().ravel()
        update = torch.zeros_like(params)
        gains = torch.ones_like(params)
        best_error = float("inf")
        best_iter = 0
        for stage, (P_cur, max_it, momentum, n_no_progress) in enumerate([
            (P_exag, exploration_max, 0.5, exploration_max),
            (P, self.max_iter, 0.8, n_iter_without_progress),
        ]):
            it_start = 0 if stage == 0 else exploration_max
            for it in range(it_start, max_it):
                check = (it + 1) % n_iter_check == 0 or it == max_it - 1
                kl, grad = self._kl_divergence(
                    params.reshape(n_samples, self.n_components),
                    P_cur, dof, compute_error=check
                )
                grad = grad.ravel()
                inc = (update * grad) < 0
                dec = ~inc
                gains = torch.where(inc, gains + 0.2, gains * 0.8)
                gains = gains.clamp(min=0.01)
                grad = grad * gains
                update = momentum * update - self.learning_rate_ * grad
                params = params + update
                if check:
                    grad_norm = grad.norm().item()
                    err = kl.item() if torch.isfinite(kl) else float("inf")
                    if err < best_error:
                        best_error = err
                        best_iter = it
                    elif it - best_iter > n_no_progress:
                        break
                    if grad_norm <= self.min_grad_norm:
                        break
        self.n_iter_ = it
        self.embedding_ = params.reshape(n_samples, self.n_components)
        self.kl_divergence_ = best_error
        return self

    def transform(self, X: Union[torch.Tensor, Any]) -> torch.Tensor:
        """t-SNE does not support transform of new points. Use fit_transform."""
        if self.embedding_ is None:
            raise RuntimeError("TSNE instance is not fitted yet.")
        raise NotImplementedError(
            "t-SNE does not support transforming new data. Use fit_transform on the full dataset."
        )

    def fit_transform(
            self,
            data_or_X: Union[torch.Tensor, Any],
            **kwargs: Any,
    ) -> torch.Tensor:
        """Fit X into an embedded space and return the transformed output."""
        self.fit(data_or_X, **kwargs)
        return self.embedding_.clone()

class Isomap(MLTransform):
    def __init__(self,
                 n_neighbors: int = 5,
                 radius: float = None,
                 n_components: int = 2,
                 eigen_solver: Union[Literal["auto", "arpack", "dense"], Callable, nn.Module] = "auto",
                 tol: float = 0,
                 max_iter: int = None,
                 path_method: Union[Literal["auto", "FW", "D"], Callable, nn.Module] = "auto",
                 neighbors_algorithm: Union[
                     Literal["auto", "brute", "ball_tree", "kd_tree"], Callable, nn.Module] = "auto",
                 n_jobs: int = None,
                 metric: Union[str, Callable, nn.Module] = "minkowski",
                 p: float = 2,
                 metric_params: dict = None,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args: Any,
                 **kwargs: Any,
                 ) -> None:
        super().__init__()
        metric_params = metric_params or {}
        if "p" not in metric_params:
            metric_params = dict(metric_params)
            metric_params["p"] = p
        from .....models.utils import MLCluster
        self.metric = MLCluster()._create_metric(metric, metric_params).metric
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.n_components = n_components
        self.eigen_solver = eigen_solver if isinstance(eigen_solver, str) else eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.path_method = path_method if isinstance(path_method, str) else path_method
        self.n_jobs = n_jobs if n_jobs is not None else 1
        self.p = p
        self.metric_params = metric_params
        self.device = device
        self.dtype = dtype
        self.leaf_size = kwargs.get("neighbors_leaf_size", 40)
        self.kwargs = dict(kwargs)
        self.kwargs["_args"] = args

        self._neighbors_algorithm = neighbors_algorithm
        if neighbors_algorithm in ["ball_tree", "kd_tree"]:
            if neighbors_algorithm == "ball_tree":
                self.neighbors_algorithm = "ball_tree"
            else:
                self.neighbors_algorithm = "kd_tree"
        elif neighbors_algorithm == "brute":
            self.neighbors_algorithm = "brute"
        else:
            self.neighbors_algorithm = neighbors_algorithm if isinstance(neighbors_algorithm, str) else "auto"

        self.embedding_ = None
        self.dist_matrix_ = None
        self.n_features_in_ = None
        self.nbrs_ = None
        self._eigenvalues_ = None
        self._eigenvectors_ = None
        self.kernel_pca_ = None

    def _build_nbrs(self, X: torch.Tensor) -> None:
        """Build nearest neighbors structure (BallTree, KDTree, or brute)."""
        if getattr(self, "metric", None) == "precomputed":
            self.nbrs_ = None
            return
        n_samples = X.shape[0]
        algo = self.neighbors_algorithm
        if algo == "auto":
            is_standard = isinstance(self.metric, Callable) or (
                    hasattr(self.metric, "__name__") and "minkowski" in str(self.metric).lower()
            )
            algo = "kd_tree" if (is_standard and not X.is_cuda) else "brute"

        if algo == "ball_tree":
            self.nbrs_ = BallTree(
                X, leaf_size=self.leaf_size, metric=self.metric,
                device=self.device, dtype=self.dtype, n_jobs=self.n_jobs
            )
        elif algo == "kd_tree":
            self.nbrs_ = KDTree(
                X, leaf_size=self.leaf_size, metric=self.metric,
                device=self.device, dtype=self.dtype, n_jobs=self.n_jobs
            )
        else:
            self.nbrs_ = None

    def _kneighbors(self, X: torch.Tensor, n_neighbors: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (distances, indices) for k nearest neighbors."""
        if self.nbrs_ is not None:
            dists, inds = self.nbrs_.query(X, k=n_neighbors, return_distance=True)
            return dists, inds
        all_dists = self.metric(X, self._fit_X)
        k = min(n_neighbors, all_dists.shape[1])
        dists, inds = torch.topk(all_dists, k, dim=1, largest=False)
        return dists, inds

    def _radius_neighbors(self, X: torch.Tensor, radius: float) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Return (list of distances, list of indices) for neighbors within radius."""
        if self.nbrs_ is not None and hasattr(self.nbrs_, "query_radius"):
            return self.nbrs_.query_radius(X, radius, return_distance=True, sort_results=True)
        all_dists = self.metric(X, self._fit_X)
        dists_list, inds_list = [], []
        for i in range(X.shape[0]):
            mask = all_dists[i] <= radius
            inds = torch.where(mask)[0]
            dists_list.append(all_dists[i, inds])
            inds_list.append(inds)
        return dists_list, inds_list

    def _shortest_path_fw(self, D: torch.Tensor) -> torch.Tensor:
        """Floyd-Warshall all-pairs shortest path."""
        n = D.shape[0]
        inf = torch.tensor(float("inf"), device=D.device, dtype=D.dtype)
        dist = D.clone()
        dist[dist == 0] = inf
        torch.diagonal(dist).fill_(0)
        for k in range(n):
            d_k = dist[:, k:k + 1] + dist[k:k + 1, :]
            dist = torch.minimum(dist, d_k)
        return dist

    def _shortest_path_dijkstra(self, D: torch.Tensor) -> torch.Tensor:
        """Dijkstra all-pairs shortest path (run from each source)."""
        n = D.shape[0]
        inf = float("inf")
        result = torch.full_like(D, inf)
        result.fill_diagonal_(0)
        for s in range(n):
            dist = torch.full((n,), inf, device=D.device, dtype=D.dtype)
            dist[s] = 0
            visited = torch.zeros(n, dtype=torch.bool, device=D.device)
            for _ in range(n - 1):
                u = (dist + (visited.float() * 1e12)).argmin().item()
                if dist[u].item() == inf:
                    break
                visited[u] = True
                for v in range(n):
                    if not visited[v] and D[u, v].item() < inf:
                        alt = dist[u].item() + D[u, v].item()
                        if alt < dist[v].item():
                            dist[v] = alt
            result[s] = dist
        return result

    def _mds_embedding(self, G: torch.Tensor) -> torch.Tensor:
        """Kernel MDS: center G, eigendecompose, return embedding."""
        n = G.shape[0]
        one_n = torch.ones(n, n, device=G.device, dtype=G.dtype) / n
        G_centered = G - one_n @ G - G @ one_n + one_n @ G @ one_n
        k = min(self.n_components + 1, n)
        solver = self.eigen_solver
        if solver == "auto":
            solver = "dense" if n < 500 else "arpack"

        if solver == "dense" or (callable(solver) or isinstance(solver, nn.Module)):
            eigenvalues, eigenvectors = torch.linalg.eigh(G_centered)
            eigenvalues = torch.flip(eigenvalues, dims=(0,))
            eigenvectors = torch.flip(eigenvectors, dims=(1,))
            eigenvalues = torch.clamp(eigenvalues, min=0)
        else:
            eigenvalues, eigenvectors = self._arpack_eigh(G_centered, k)

        self._eigenvalues_ = eigenvalues[:self.n_components]
        self._eigenvectors_ = eigenvectors[:, :self.n_components]
        emb = self._eigenvectors_ * torch.sqrt(self._eigenvalues_.clamp(min=0))
        return emb

    def _arpack_eigh(self, M: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Power iteration for top-k eigenvalues of symmetric M (ARPACK-style)."""
        n = M.shape[0]
        eps = 1e-12
        tol = self.tol if self.tol > 0 else eps
        max_it = self.max_iter if self.max_iter is not None else 100
        evals, evecs = [], []
        M_work = M.clone()
        for _ in range(k):
            v = torch.randn(n, device=M.device, dtype=M.dtype)
            v = v / v.norm()
            for _ in range(max_it):
                v_new = M_work @ v
                v_new = v_new / (v_new.norm().clamp(min=eps))
                if (v_new - v).norm() < tol:
                    break
                v = v_new
            lam = (v @ M_work @ v).item()
            lam = max(lam, eps)
            evals.append(lam)
            evecs.append(v.clone())
            M_work = M_work - lam * torch.outer(v, v)
        return torch.tensor(evals, device=M.device, dtype=M.dtype), torch.stack(evecs, dim=1)

    def fit(self, data_or_X: Union[torch.Tensor, Any], y: Any = None, **kwargs: Any) -> "Isomap":
        """Compute the embedding vectors for data X."""
        X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        if self.n_neighbors is not None and self.radius is not None:
            raise ValueError("Both n_neighbors and radius provided. Use one.")
        if self.n_neighbors is None and self.radius is None:
            self.n_neighbors = 5

        if self.metric == "precomputed":
            if X.shape[0] != X.shape[1]:
                raise ValueError("Precomputed metric requires a square distance matrix.")
            self._fit_X = X
            self.n_features_in_ = X.shape[0]
            self.dist_matrix_ = X.clone()
            G = -0.5 * (self.dist_matrix_ ** 2)
            self.embedding_ = self._mds_embedding(G)
            self.kernel_pca_ = type("KernelPCAProxy", (), {
                "eigenvalues_": self._eigenvalues_,
                "eigenvectors_": self._eigenvectors_,
            })()
            return self

        self._fit_X = X
        self.n_features_in_ = X.shape[1]
        n_samples = X.shape[0]
        self._build_nbrs(X)

        if self.n_neighbors is not None:
            dists, inds = self._kneighbors(X, self.n_neighbors)
        else:
            d_list, i_list = self._radius_neighbors(X, self.radius)
            dists, inds = d_list, i_list

        inf = float("inf")
        D = torch.full((n_samples, n_samples), inf, device=self.device, dtype=self.dtype)
        torch.diagonal(D).fill_(0)
        if self.n_neighbors is not None:
            for i in range(n_samples):
                for j in range(dists.shape[1]):
                    ni = inds[i, j].item()
                    d = dists[i, j].item()
                    if d < D[i, ni]:
                        D[i, ni] = d
                        D[ni, i] = d
        else:
            for i in range(n_samples):
                for j, (d, ni) in enumerate(zip(d_list[i], i_list[i])):
                    ni = ni.item() if torch.is_tensor(ni) else ni
                    d = d.item() if torch.is_tensor(d) else d
                    if d < D[i, ni]:
                        D[i, ni] = d
                        D[ni, i] = d

        path_m = self.path_method
        if path_m == "auto":
            path_m = "FW" if n_samples < 500 else "D"
        if path_m == "FW":
            self.dist_matrix_ = self._shortest_path_fw(D)
        else:
            self.dist_matrix_ = self._shortest_path_dijkstra(D)

        G = -0.5 * (self.dist_matrix_ ** 2)
        self.embedding_ = self._mds_embedding(G)
        self.kernel_pca_ = type("KernelPCAProxy", (), {
            "eigenvalues_": self._eigenvalues_,
            "eigenvectors_": self._eigenvectors_,
        })()
        return self

    def fit_transform(
            self,
            data_or_X: Union[torch.Tensor, Any],
            y: Any = None,
            **kwargs: Any,
    ) -> torch.Tensor:
        """Fit the model from data in X and transform X."""
        self.fit(data_or_X, y=y, **kwargs)
        return self.embedding_.clone()

    def transform(self, X: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Transform X by projecting onto the embedding."""
        if self.embedding_ is None:
            raise RuntimeError("Isomap instance is not fitted yet.")
        if self.metric == "precomputed":
            raise NotImplementedError("transform is not supported when metric='precomputed'.")
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        n_queries = X.shape[0]
        n_fit = self._fit_X.shape[0]
        if self.n_neighbors is not None:
            dists, inds = self._kneighbors(X, self.n_neighbors)
        else:
            d_list, i_list = self._radius_neighbors(X, self.radius)
        G_X = torch.full((n_queries, n_fit), float("inf"), device=self.device, dtype=self.dtype)
        if self.n_neighbors is not None:
            for i in range(n_queries):
                d_neigh = dists[i]
                i_neigh = inds[i]
                d_geo = self.dist_matrix_[i_neigh]
                d_geo = d_geo + d_neigh.unsqueeze(1)
                G_X[i] = d_geo.min(dim=0).values
        else:
            for i in range(n_queries):
                i_neigh = i_list[i]
                if len(i_neigh) == 0:
                    continue
                d_neigh = d_list[i]
                idx = i_neigh.long() if torch.is_tensor(i_neigh) else torch.tensor(i_neigh, device=self.device)
                d_geo = self.dist_matrix_[idx] + d_neigh.unsqueeze(1)
                G_X[i, :] = d_geo.min(dim=0).values
        G_X = torch.clamp(G_X, max=1e12)
        G_X = -0.5 * (G_X ** 2)
        G_fit = -0.5 * (self.dist_matrix_ ** 2)
        G_X_row_mean = G_X.mean(dim=1, keepdim=True)
        G_fit_col_mean = G_fit.mean(dim=0, keepdim=True)
        G_fit_all_mean = G_fit.mean()
        G_X_centered = G_X - G_X_row_mean - G_fit_col_mean + G_fit_all_mean
        K = self._eigenvectors_.T @ G_X_centered.T
        K = K / (self._eigenvalues_.clamp(min=1e-12).unsqueeze(1))
        return K.T

class LocallyLinearEmbedding(MLTransform):
    def __init__(self,
                 n_neighbors: int = 5,
                 n_components: int = 2,
                 reg: float = 0.001,
                 eigen_solver: Union[Literal["auto", "arpack", "dense"]] = 'auto',
                 tol: float = 1e-06,
                 max_iter: int = 100,
                 method: Union[Literal["standard", "hessian", "modified", "ltsa"], Callable, nn.Module] = 'standard',
                 hessian_tol: float = 0.0001,
                 modified_tol: float = 1e-12,
                 neighbors_algorithm: Union[
                     Literal["auto", "brute", "ball_tree", "kd_tree"], Callable, nn.Module] = 'auto',
                 random_state: Union[int, torch.Generator] = None,
                 n_jobs: int = None,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args: Any,
                 **kwargs: Any,
                 ) -> None:
        super().__init__()
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.reg = reg
        _es = eigen_solver if isinstance(eigen_solver, str) else eigen_solver
        self.eigen_solver = _es if _es in ("auto", "arpack", "dense") else "auto"
        self.tol = tol
        self.max_iter = max_iter
        _m = method if isinstance(method, str) else method
        self.method = _m if _m in ("standard", "hessian", "modified", "ltsa") else "standard"
        self.hessian_tol = hessian_tol
        self.modified_tol = modified_tol
        self._neighbors_algorithm = neighbors_algorithm
        if neighbors_algorithm in ["ball_tree", "kd_tree"]:
            self.neighbors_algorithm = neighbors_algorithm
        elif neighbors_algorithm == "brute":
            self.neighbors_algorithm = "brute"
        else:
            self.neighbors_algorithm = "auto" if not isinstance(neighbors_algorithm, str) else "auto"
        self.random_state = random_state
        self.n_jobs = n_jobs if n_jobs is not None else 1
        self.device = device
        self.dtype = dtype
        self.leaf_size = kwargs.get("neighbors_leaf_size", 40)
        self.kwargs = dict(kwargs)
        self.kwargs["_args"] = args

        from .....models.utils import MLCluster
        self.metric = MLCluster()._create_metric("minkowski", {"p": 2}).metric

        self.embedding_ = None
        self.reconstruction_error_ = None
        self.n_features_in_ = None
        self.nbrs_ = None

    def _build_nbrs(self, X: torch.Tensor) -> None:
        """Build nearest neighbors structure (BallTree, KDTree, or brute)."""
        if X.is_cuda:
            self.nbrs_ = None
            return
        n_samples = X.shape[0]
        algo = self.neighbors_algorithm
        if algo == "auto":
            algo = "kd_tree" if n_samples > 0 and not X.is_cuda else "brute"
        if algo == "ball_tree":
            self.nbrs_ = BallTree(X, leaf_size=self.leaf_size, metric=self.metric,
                                  device=self.device, dtype=self.dtype, n_jobs=self.n_jobs)
        elif algo == "kd_tree":
            self.nbrs_ = KDTree(X, leaf_size=self.leaf_size, metric=self.metric,
                                device=self.device, dtype=self.dtype, n_jobs=self.n_jobs)
        else:
            self.nbrs_ = None

    def _kneighbors(self, X: torch.Tensor, n_neighbors: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (distances, indices) for k nearest neighbors."""
        if self.nbrs_ is not None:
            dists, inds = self.nbrs_.query(X, k=n_neighbors, return_distance=True)
            return dists, inds
        all_dists = self.metric(X, self._fit_X)
        k = min(n_neighbors, all_dists.shape[1])
        dists, inds = torch.topk(all_dists, k, dim=1, largest=False)
        return dists, inds

    def _compute_weights_standard(self, X: torch.Tensor, inds: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction weights for standard LLE."""
        n_samples, n_features = X.shape
        n_neighbors = inds.shape[1]
        W = torch.zeros(n_samples, n_samples, device=self.device, dtype=self.dtype)
        eps = 1e-12
        for i in range(n_samples):
            Xi = X[i]
            neighbors = X[inds[i]]
            Z = neighbors - Xi
            C = Z @ Z.T
            trace_C = C.trace().clamp(min=eps)
            C = C + self.reg * trace_C * torch.eye(n_neighbors, device=self.device, dtype=self.dtype)
            try:
                Cinv = torch.linalg.inv(C)
            except Exception:
                C = C + eps * torch.eye(n_neighbors, device=self.device, dtype=self.dtype)
                Cinv = torch.linalg.inv(C)
            w = Cinv.sum(dim=1)
            w = w / w.sum().clamp(min=eps)
            for j in range(n_neighbors):
                W[i, inds[i, j]] = w[j]
        return W

    def _compute_weights_modified(self, X: torch.Tensor, inds: torch.Tensor) -> torch.Tensor:
        """Compute weights for modified LLE (uses regularization)."""
        return self._compute_weights_standard(X, inds)

    def _compute_weights_ltsa(self, X: torch.Tensor, inds: torch.Tensor) -> torch.Tensor:
        """Compute weights for LTSA - similar to standard but different embedding step."""
        return self._compute_weights_standard(X, inds)

    def _compute_weights_hessian(self, X: torch.Tensor, inds: torch.Tensor) -> torch.Tensor:
        """Compute weights for Hessian eigenmap - requires more neighbors."""
        return self._compute_weights_standard(X, inds)

    def _lle_embedding(self, W: torch.Tensor) -> torch.Tensor:
        """Compute embedding from weight matrix via eigendecomposition of (I-W)'(I-W)."""
        n = W.shape[0]
        I = torch.eye(n, device=self.device, dtype=self.dtype)
        M = (I - W).T @ (I - W)
        M = (M + M.T) / 2
        k = min(self.n_components + 1, n)
        solver = self.eigen_solver
        if solver == "auto":
            solver = "dense" if n < 500 else "arpack"
        if solver == "dense" or (callable(solver) or isinstance(solver, nn.Module)):
            eigenvalues, eigenvectors = torch.linalg.eigh(M)
            idx = torch.argsort(eigenvalues)
            eigenvectors = eigenvectors[:, idx]
            eigenvalues = eigenvalues[idx]
        else:
            eigenvalues, eigenvectors = self._arpack_eigh_smallest(M, k)
        emb = eigenvectors[:, 1:self.n_components + 1]
        self._eigenvalues_ = eigenvalues[1:self.n_components + 1]
        self._eigenvectors_ = emb
        return emb

    def _arpack_eigh_smallest(self, M: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """ARPACK-style: find k smallest eigenvalues via shift-invert (largest of (M+sigma*I)^{-1})."""
        n = M.shape[0]
        eps = 1e-12
        sigma = 1e-4 * M.diag().mean().item() if M.diag().abs().sum() > 0 else 1e-4
        M_shifted = M + sigma * torch.eye(n, device=M.device, dtype=M.dtype)
        try:
            M_inv = torch.linalg.inv(M_shifted)
        except Exception:
            M_inv = torch.linalg.inv(M_shifted + eps * torch.eye(n, device=M.device, dtype=M.dtype))
        M_inv = (M_inv + M_inv.T) / 2
        tol = self.tol if self.tol > 0 else eps
        max_it = self.max_iter
        gen = torch.Generator(device=self.device)
        if self.random_state is not None:
            gen.manual_seed(int(self.random_state))
        evals, evecs = [], []
        M_work = M_inv.clone()
        for _ in range(k):
            v = torch.randn(n, device=M.device, dtype=M.dtype, generator=gen)
            v = v / v.norm()
            for _ in range(max_it):
                v_new = M_work @ v
                v_new = v_new / (v_new.norm().clamp(min=eps))
                if (v_new - v).norm() < tol:
                    break
                v = v_new
            lam = (v @ M_work @ v).item()
            lam = max(lam, eps)
            evals.append(1.0 / lam - sigma)
            evecs.append(v.clone())
            M_work = M_work - lam * torch.outer(v, v)
        return torch.tensor(evals, device=M.device, dtype=M.dtype), torch.stack(evecs, dim=1)

    def fit(self, data_or_X: Union[torch.Tensor, Any], y: Any = None, **kwargs: Any) -> "LocallyLinearEmbedding":
        """Compute the embedding vectors for data X."""
        X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self._fit_X = X
        k = self.n_neighbors
        if self.method == "hessian" and k <= self.n_components * (1 + (self.n_components + 1) / 2):
            raise ValueError(
                f"hessian method requires n_neighbors > n_components*(1+(n_components+1)/2). "
                f"Got n_neighbors={k}, n_components={self.n_components}"
            )
        self._build_nbrs(X)
        _, inds = self._kneighbors(X, k)
        if self.method == "standard":
            W = self._compute_weights_standard(X, inds)
        elif self.method == "modified":
            W = self._compute_weights_modified(X, inds)
        elif self.method == "hessian":
            W = self._compute_weights_hessian(X, inds)
        elif self.method == "ltsa":
            W = self._compute_weights_ltsa(X, inds)
        elif callable(self.method) or isinstance(self.method, nn.Module):
            W = self.method(X, inds, self)
        else:
            W = self._compute_weights_standard(X, inds)
        self.embedding_ = self._lle_embedding(W)
        diff = self._fit_X - (W @ self._fit_X)
        self.reconstruction_error_ = (diff * diff).sum().sqrt().item()
        return self

    def fit_transform(self, data_or_X: Union[torch.Tensor, Any], y: Any = None, **kwargs: Any) -> torch.Tensor:
        """Fit the model from data in X and transform X."""
        self.fit(data_or_X, y=y, **kwargs)
        return self.embedding_.clone()

    def transform(self, X: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Transform X using the barycentric embedding (out-of-sample extension)."""
        if self.embedding_ is None:
            raise RuntimeError("LocallyLinearEmbedding instance is not fitted yet.")
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        n_queries = X.shape[0]
        n_fit = self._fit_X.shape[0]
        _, inds = self._kneighbors(X, self.n_neighbors)
        Y_embed = self.embedding_
        Y_out = torch.zeros(n_queries, self.n_components, device=self.device, dtype=self.dtype)
        eps = 1e-12
        for i in range(n_queries):
            Xi = X[i]
            neighbors = self._fit_X[inds[i]]
            Z = neighbors - Xi
            C = Z @ Z.T
            trace_C = C.trace().clamp(min=eps)
            C = C + self.reg * trace_C * torch.eye(self.n_neighbors, device=self.device, dtype=self.dtype)
            try:
                Cinv = torch.linalg.inv(C)
            except Exception:
                C = C + eps * torch.eye(self.n_neighbors, device=self.device, dtype=self.dtype)
                Cinv = torch.linalg.inv(C)
            w = Cinv.sum(dim=1)
            w = w / w.sum().clamp(min=eps)
            Y_out[i] = (w.unsqueeze(0) @ Y_embed[inds[i]]).squeeze(0)
        return Y_out


class MDS(MLTransform):
    def __init__(self,
                 n_components: int = 2,
                 metric_mds: bool = True,
                 n_init: int = 1,
                 init: Union[Literal["random", "classical_mds"],
                    Callable, nn.Module] = "classical_mds",
                 max_iter: int = 100,
                 verbose: int = 0,
                 eps: float = 1e-6,
                 n_jobs: int = None,
                 random_state: Union[int, torch.Generator] = None,
                 metric: Union[str, Callable, nn.Module] = "euclidean",
                 metric_params: dict = None,
                 normalized_stress: Union[bool, Literal["auto"]] = "auto",
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args: Any,
                 **kwargs: Any,
                 ) -> None:
        super().__init__()
        self.n_components = n_components
        self.metric_mds = metric_mds
        self.n_init = n_init
        self.init = init
        self.max_iter = max_iter
        self.verbose = verbose
        self.eps = eps
        self.n_jobs = n_jobs if n_jobs is not None else 1
        self.random_state = random_state
        self.metric_params = metric_params or {}
        self.normalized_stress = normalized_stress
        self.device = device
        self.dtype = dtype
        self.kwargs = dict(kwargs)
        self.kwargs["_args"] = args

        from .....models.utils import MLCluster
        mp = dict(self.metric_params)
        if "p" not in mp:
            mp["p"] = 2
        self._metric_obj = MLCluster()._create_metric(metric, mp)
        self.metric = self._metric_obj.metric if hasattr(self._metric_obj, "metric") else self._metric_obj

        self.embedding_ = None
        self.stress_ = None
        self.dissimilarity_matrix_ = None
        self.n_features_in_ = None
        self.n_iter_ = 0

    def _get_generator(self) -> Optional[torch.Generator]:
        if self.random_state is None:
            return None
        if isinstance(self.random_state, torch.Generator):
            return self.random_state
        g = torch.Generator(device=self.device)
        g.manual_seed(int(self.random_state))
        return g

    def _compute_dissimilarity(self, X: torch.Tensor) -> torch.Tensor:
        """Compute pairwise dissimilarity matrix from X."""
        if self.metric == "precomputed" or (
            isinstance(self.metric, str) and self.metric == "precomputed"
        ):
            return X.clone()
        D = self.metric(X, X)
        if not isinstance(D, torch.Tensor):
            D = torch.as_tensor(D, device=self.device, dtype=self.dtype)
        return D.to(device=self.device, dtype=self.dtype)

    @staticmethod
    def _isotonic_regress_1d(values: torch.Tensor) -> torch.Tensor:
        """Pool Adjacent Violators for ascending isotonic regression."""
        vals = values.tolist()
        pool = [[v, 1] for v in vals]
        i = 0
        while i < len(pool) - 1:
            if pool[i][0] > pool[i + 1][0]:
                s = pool[i][0] * pool[i][1] + pool[i + 1][0] * pool[i + 1][1]
                c = pool[i][1] + pool[i + 1][1]
                pool[i] = [s / c, c]
                pool.pop(i + 1)
                if i > 0:
                    i -= 1
            else:
                i += 1
        result = []
        for mean, count in pool:
            result.extend([mean] * int(count))
        return torch.tensor(result, device=values.device, dtype=values.dtype)

    def _smacof_single(
        self, D: torch.Tensor, Y_init: torch.Tensor
    ) -> Tuple[torch.Tensor, float, int]:
        """Run a single SMACOF pass. Returns (Y, stress, n_iter)."""
        n = D.shape[0]
        Y = Y_init.clone()
        eps_denom = 1e-12
        stress_old = float("inf")
        n_iter = 0

        for it in range(self.max_iter):
            diff_Y = Y.unsqueeze(1) - Y.unsqueeze(0)
            d_Y = diff_Y.pow(2).sum(dim=-1).sqrt().clamp(min=eps_denom)
            torch.diagonal(d_Y).fill_(0.0)

            if self.metric_mds:
                D_use = D
                raw_stress = ((d_Y - D_use).pow(2).sum() * 0.5).item()
            else:
                # Non-metric: compute disparities via isotonic regression
                triu_idx = torch.triu_indices(n, n, offset=1, device=self.device)
                d_flat = d_Y[triu_idx[0], triu_idx[1]]
                D_flat = D[triu_idx[0], triu_idx[1]]
                sort_order = torch.argsort(D_flat)
                d_sorted = d_flat[sort_order]
                disp_sorted = self._isotonic_regress_1d(d_sorted)
                disparities_flat = torch.zeros_like(D_flat)
                disparities_flat[sort_order] = disp_sorted
                D_use = torch.zeros_like(D)
                D_use[triu_idx[0], triu_idx[1]] = disparities_flat
                D_use[triu_idx[1], triu_idx[0]] = disparities_flat
                raw_stress = ((d_Y - D_use).pow(2).sum() * 0.5).item()

            # B matrix
            with torch.no_grad():
                ratio = D_use / d_Y.clamp(min=eps_denom)
                ratio[d_Y < eps_denom] = 0.0
                torch.diagonal(ratio).fill_(0.0)
                B = -ratio
                B.diagonal().copy_(-B.sum(dim=1))

            Y = (B @ Y) / n
            n_iter = it + 1

            if self.verbose:
                print(f"MDS SMACOF iter {it + 1}: stress={raw_stress:.6f}")

            rel_change = abs(stress_old - raw_stress) / max(1.0, abs(stress_old))
            if rel_change < self.eps and it > 0:
                break
            stress_old = raw_stress

        return Y, raw_stress, n_iter

    def fit(
        self, data_or_X: Union[torch.Tensor, Any], y: Any = None, **kwargs: Any
    ) -> "MDS":
        """Fit the model from data X."""
        X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)

        if self.metric == "precomputed" or (
            isinstance(self.metric, str) and self.metric == "precomputed"
        ):
            D = X
            n = D.shape[0]
            self.n_features_in_ = n
        else:
            self.n_features_in_ = X.shape[1]
            D = self._compute_dissimilarity(X)
            n = D.shape[0]

        self.dissimilarity_matrix_ = D
        gen = self._get_generator()

        init = self.init
        if isinstance(init, str) and init == "classical_mds":
            cmds = ClassicalMDS(
                n_components=self.n_components, device=self.device, dtype=self.dtype
            )
            cmds.fit(D if (self.metric == "precomputed") else X)
            Y_cmds = cmds.embedding_
            if Y_cmds.shape[1] < self.n_components:
                pad = torch.zeros(
                    n,
                    self.n_components - Y_cmds.shape[1],
                    device=self.device,
                    dtype=self.dtype,
                )
                Y_cmds = torch.cat([Y_cmds, pad], dim=1)
            Y_inits = [Y_cmds]
            n_runs = 1
        elif callable(init) or isinstance(init, nn.Module):
            n_runs = self.n_init
            Y_inits = [
                init(X, self.n_components, device=self.device, dtype=self.dtype)
                for _ in range(n_runs)
            ]
        elif isinstance(init, (list, tuple, torch.Tensor)):
            Y_inits = [torch.as_tensor(init, device=self.device, dtype=self.dtype)]
            n_runs = 1
        else:
            n_runs = self.n_init
            Y_inits = []
            for _ in range(n_runs):
                if gen is not None:
                    Y = torch.randn(
                        n,
                        self.n_components,
                        device=self.device,
                        dtype=self.dtype,
                        generator=gen,
                    )
                else:
                    Y = torch.randn(n, self.n_components, device=self.device, dtype=self.dtype)
                Y_inits.append(Y)

        best_Y, best_stress, best_iter = None, float("inf"), 0
        for Y_init in Y_inits:
            Y, stress, n_iter = self._smacof_single(D, Y_init)
            if stress < best_stress:
                best_stress, best_Y, best_iter = stress, Y, n_iter

        norm_stress = self.normalized_stress
        if norm_stress == "auto":
            norm_stress = not self.metric_mds
        if norm_stress:
            diff_Y = best_Y.unsqueeze(1) - best_Y.unsqueeze(0)
            d_sq_sum = diff_Y.pow(2).sum(dim=-1).sum().clamp(min=1e-12).item()
            self.stress_ = math.sqrt(best_stress * 2 / d_sq_sum)
        else:
            self.stress_ = best_stress

        self.embedding_ = best_Y
        self.n_iter_ = best_iter
        return self

    def fit_transform(
        self,
        data_or_X: Union[torch.Tensor, Any],
        y: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Fit the model and return the embedding."""
        self.fit(data_or_X, y=y, **kwargs)
        return self.embedding_.clone()

    def transform(self, X: Union[torch.Tensor, Any]) -> torch.Tensor:
        """MDS does not support out-of-sample transform."""
        raise NotImplementedError(
            "MDS does not support transform on new data. Use fit_transform on the full dataset."
        )


class ClassicalMDS(MLTransform):
    def __init__(self,
                 n_components: int = 2,
                 metric: Union[str, Callable, nn.Module] = "euclidean",
                 metric_params: dict = None,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args: Any,
                 **kwargs: Any,
                 ) -> None:
        super().__init__()
        self.n_components = n_components
        self.metric_params = metric_params or {}
        self.device = device
        self.dtype = dtype
        self.kwargs = dict(kwargs)
        self.kwargs["_args"] = args

        from .....models.utils import MLCluster
        mp = dict(self.metric_params)
        if "p" not in mp:
            mp["p"] = 2
        _mo = MLCluster()._create_metric(metric, mp)
        self.metric = _mo.metric if hasattr(_mo, "metric") else _mo

        self.embedding_ = None
        self.dissimilarity_matrix_ = None
        self.eigenvalues_ = None
        self.n_features_in_ = None

    def _compute_dissimilarity(self, X: torch.Tensor) -> torch.Tensor:
        """Compute pairwise dissimilarity matrix."""
        if self.metric == "precomputed" or (
            isinstance(self.metric, str) and self.metric == "precomputed"
        ):
            return X.clone()
        D = self.metric(X, X)
        if not isinstance(D, torch.Tensor):
            D = torch.as_tensor(D, device=self.device, dtype=self.dtype)
        return D.to(device=self.device, dtype=self.dtype)

    def _double_center(self, D: torch.Tensor) -> torch.Tensor:
        """Double center the squared distance matrix: B = -0.5 * H D^2 H."""
        n = D.shape[0]
        D2 = D.pow(2)
        one_n = torch.ones(n, n, device=D.device, dtype=D.dtype) / n
        B = -0.5 * (D2 - one_n @ D2 - D2 @ one_n + one_n @ D2 @ one_n)
        return (B + B.T) / 2

    def fit(
        self, data_or_X: Union[torch.Tensor, Any], y: Any = None, **kwargs: Any
    ) -> "ClassicalMDS":
        """Fit the model from data X."""
        X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)

        if self.metric == "precomputed" or (
            isinstance(self.metric, str) and self.metric == "precomputed"
        ):
            D = X
            self.n_features_in_ = X.shape[0]
        else:
            self.n_features_in_ = X.shape[1]
            D = self._compute_dissimilarity(X)

        self.dissimilarity_matrix_ = D
        B = self._double_center(D)
        n = B.shape[0]
        k = min(self.n_components, n)

        eigvals, eigvecs = torch.linalg.eigh(B)
        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        eigvals_k = eigvals[:k].clamp(min=0.0)
        eigvecs_k = eigvecs[:, :k]
        self.eigenvalues_ = eigvals_k
        self.embedding_ = eigvecs_k * eigvals_k.sqrt().unsqueeze(0)
        return self

    def fit_transform(
        self,
        data_or_X: Union[torch.Tensor, Any],
        y: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Fit the model and return the embedding."""
        self.fit(data_or_X, y=y, **kwargs)
        return self.embedding_.clone()

    def transform(self, X: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Classical MDS does not support out-of-sample transform."""
        raise NotImplementedError(
            "ClassicalMDS does not support transform on new data. Use fit_transform on the full dataset."
        )


class SpectralEmbedding(MLTransform):
    def __init__(self,
                 n_components: int = 2,
                 affinity: Union[Literal["nearest_neighbors", "rbf",
                    "precomputed", "precomputed_nearest_neighbors"],
                    Callable, nn.Module] = 'nearest_neighbors',
                 gamma: float = None,
                 random_state: Union[int, torch.Generator] = None,
                 eigen_solver: Union[Literal["arpack", "lobpcg", "amg"],
                    Callable, nn.Module] = None,
                 eigen_tol: Union[Literal["auto"], float] = "auto",
                 n_neighbors: int = None,
                 n_jobs: int = None,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args: Any,
                 **kwargs: Any,
                 ) -> None:
        super().__init__()
        self.n_components = n_components
        self.affinity = affinity
        self.gamma = gamma
        self.random_state = random_state
        self.eigen_solver = eigen_solver
        self.eigen_tol = eigen_tol
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs if n_jobs is not None else 1
        self.device = device
        self.dtype = dtype
        self.kwargs = dict(kwargs)
        self.kwargs["_args"] = args

        self.embedding_ = None
        self.affinity_matrix_ = None
        self.n_features_in_ = None
        self.n_neighbors_ = None

    def _get_generator(self) -> Optional[torch.Generator]:
        if self.random_state is None:
            return None
        if isinstance(self.random_state, torch.Generator):
            return self.random_state
        g = torch.Generator(device=self.device)
        g.manual_seed(int(self.random_state))
        return g

    def _build_affinity(self, X: torch.Tensor) -> torch.Tensor:
        """Build the affinity matrix from X."""
        n = X.shape[0]
        aff = self.affinity

        if callable(aff) and not isinstance(aff, str):
            A = aff(X)
            if not isinstance(A, torch.Tensor):
                A = torch.as_tensor(A, device=self.device, dtype=self.dtype)
            return A.to(device=self.device, dtype=self.dtype)

        if isinstance(aff, nn.Module):
            A = aff(X)
            if not isinstance(A, torch.Tensor):
                A = torch.as_tensor(A, device=self.device, dtype=self.dtype)
            return A.to(device=self.device, dtype=self.dtype)

        if aff == "precomputed":
            return X.clone()

        if aff == "precomputed_nearest_neighbors":
            A = X.clone()
            A = (A + A.T) / 2
            return A

        gamma = self.gamma if self.gamma is not None else (1.0 / X.shape[1] if X.dim() > 1 and X.shape[1] > 0 else 1.0)

        if aff == "rbf":
            diffs = X.unsqueeze(1) - X.unsqueeze(0)
            sq_dists = diffs.pow(2).sum(dim=-1)
            A = torch.exp(-gamma * sq_dists)
            return A

        if aff == "nearest_neighbors":
            k = self.n_neighbors_
            diffs = X.unsqueeze(1) - X.unsqueeze(0)
            sq_dists = diffs.pow(2).sum(dim=-1)
            knn_sq, knn_idx = torch.topk(sq_dists, k + 1, dim=1, largest=False)
            knn_sq = knn_sq[:, 1:]
            knn_idx = knn_idx[:, 1:]
            A = torch.zeros(n, n, device=self.device, dtype=self.dtype)
            for i in range(n):
                for jj in range(k):
                    j = knn_idx[i, jj].item()
                    w = math.exp(-gamma * knn_sq[i, jj].item())
                    A[i, j] = w
            A = (A + A.T) / 2
            return A

        raise ValueError(f"Unknown affinity: {aff}")

    def _graph_laplacian(self, A: torch.Tensor) -> torch.Tensor:
        """Compute normalized graph Laplacian: L = I - D^{-1/2} A D^{-1/2}."""
        n = A.shape[0]
        deg = A.sum(dim=1).clamp(min=1e-12)
        D_inv_sqrt = torch.diag(1.0 / deg.sqrt())
        L = torch.eye(n, device=self.device, dtype=self.dtype) - D_inv_sqrt @ A @ D_inv_sqrt
        return (L + L.T) / 2

    def _arpack_eigh_smallest(self, M: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find k smallest eigenvalues of symmetric M via shift-invert power iteration."""
        n = M.shape[0]
        eps = 1e-12
        gen = self._get_generator()
        sigma = max(1e-4, M.abs().mean().item() * 1e-4)
        M_shifted = M + sigma * torch.eye(n, device=M.device, dtype=M.dtype)
        try:
            M_inv = torch.linalg.inv(M_shifted)
        except Exception:
            M_inv = torch.linalg.pinv(M_shifted)
        M_inv = (M_inv + M_inv.T) / 2
        tol_val = 1e-6 if self.eigen_tol == "auto" else float(self.eigen_tol if self.eigen_tol is not None else 1e-6)
        evals, evecs = [], []
        M_work = M_inv.clone()
        for _ in range(k):
            if gen is not None:
                v = torch.randn(n, device=M.device, dtype=M.dtype, generator=gen)
            else:
                v = torch.randn(n, device=M.device, dtype=M.dtype)
            v = v / v.norm().clamp(min=eps)
            for _ in range(200):
                v_new = M_work @ v
                nrm = v_new.norm().clamp(min=eps)
                v_new = v_new / nrm
                if (v_new - v).norm() < tol_val or (v_new + v).norm() < tol_val:
                    break
                v = v_new
            lam_inv = (v @ M_work @ v).item()
            lam_inv = max(lam_inv, eps)
            lam = 1.0 / lam_inv - sigma
            evals.append(lam)
            evecs.append(v.clone())
            M_work = M_work - lam_inv * torch.outer(v, v)
        return (
            torch.tensor(evals, device=M.device, dtype=M.dtype),
            torch.stack(evecs, dim=1),
        )

    def fit(
        self, data_or_X: Union[torch.Tensor, Any], y: Any = None, **kwargs: Any
    ) -> "SpectralEmbedding":
        """Fit the model from data X."""
        X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        n_samples = X.shape[0]

        if self.affinity in ("precomputed", "precomputed_nearest_neighbors"):
            self.n_features_in_ = n_samples
        else:
            self.n_features_in_ = X.shape[1] if X.dim() > 1 else 1

        if self.n_neighbors is None:
            self.n_neighbors_ = max(1, n_samples // 10)
        else:
            self.n_neighbors_ = min(self.n_neighbors, n_samples - 1)

        self.affinity_matrix_ = self._build_affinity(X)
        L = self._graph_laplacian(self.affinity_matrix_)

        k = min(self.n_components + 1, n_samples)
        solver = self.eigen_solver
        if solver is None:
            solver = "dense" if n_samples <= 300 else "arpack"

        if solver == "arpack" or (solver in ("lobpcg", "amg") and n_samples > 300):
            evals, evecs = self._arpack_eigh_smallest(L, k)
            idx = torch.argsort(evals)
            evals, evecs = evals[idx], evecs[:, idx]
        else:
            evals, evecs = torch.linalg.eigh(L)

        self.embedding_ = evecs[:, 1: self.n_components + 1]
        return self

    def fit_transform(
        self,
        data_or_X: Union[torch.Tensor, Any],
        y: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Fit the model and return the embedding."""
        self.fit(data_or_X, y=y, **kwargs)
        return self.embedding_.clone()

    def transform(self, X: Union[torch.Tensor, Any]) -> torch.Tensor:
        """SpectralEmbedding does not support out-of-sample transform."""
        raise NotImplementedError(
            "SpectralEmbedding does not support transform on new data. Use fit_transform on the full dataset."
        )
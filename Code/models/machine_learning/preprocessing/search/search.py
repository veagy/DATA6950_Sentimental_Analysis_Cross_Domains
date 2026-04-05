import warnings
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
    "NearestNeighbors",
]


class NearestNeighbors(MLModule):
    def __init__(self,
                 n_neighbors: int=5,
                 radius: float=1.0,
                 algorithm: Union[Literal["ball_tree", "kd_tree",
                    "brute", "auto"], Callable, nn.Module]='auto',
                 leaf_size: int=30,
                 metric: Union[str, Callable, nn.Module]='minkowski',
                 p: float=2,
                 metric_params: dict=None,
                 n_jobs: int=None,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        if metric_params is None:
            metric_params = {}
        metric_params["p"] = p
        from ....utils.utils import MLCluster
        self.metric = MLCluster()._create_metric(metric, metric_params).metric

        self.n_neighbors = n_neighbors
        self.radius = radius
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self._metric_spec = metric
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype

        # Fitted attributes
        self.effective_metric_: Optional[str] = None
        self.effective_metric_params_: Optional[dict] = None
        self.n_features_in_: Optional[int] = None
        self.n_samples_fit_: Optional[int] = None

        # Internal state
        self._X_fit: Optional[torch.Tensor] = None
        self._tree: Optional[Any] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_algorithm(self, n_samples: int, n_features: int) -> str:
        """Resolve algorithm string: 'ball_tree', 'kd_tree', 'brute', or 'callable'."""
        alg = self.algorithm
        if callable(alg) or isinstance(alg, nn.Module):
            return 'callable'
        alg_str = str(alg).lower()
        if alg_str in ('ball_tree', 'kd_tree', 'brute', 'callable'):
            return alg_str
        # 'auto': choose based on data dimensionality
        if self._metric_spec == 'precomputed':
            return 'brute'
        return 'kd_tree' if n_features <= 15 else 'ball_tree'

    def _get_tree_metric_fn(self) -> Optional[Callable]:
        """Return a callable suitable for BallTree/KDTree from self.metric."""
        metric = self.metric
        if callable(metric) and metric != "precomputed":
            return metric
        return None

    def _compute_distances(self, X_query: torch.Tensor,
                           X_train: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distance matrix (n_query, n_train)."""
        metric = self.metric
        if metric == "precomputed":
            # X_query is already a precomputed distance matrix
            return X_query.to(dtype=self.dtype)
        if callable(metric):
            D = metric(X_query, X_train)
            return D.to(dtype=self.dtype)
        # Fallback: Minkowski with p
        return torch.cdist(
            X_query.to(self.dtype), X_train.to(self.dtype), p=self.p
        )

    def _build_tree(self, X: torch.Tensor, alg: str) -> Optional[Any]:
        """Build a BallTree or KDTree over X."""
        metric_fn = self._get_tree_metric_fn()
        n_jobs = self.n_jobs if self.n_jobs is not None else 1
        if alg == 'ball_tree':
            return BallTree(X, leaf_size=self.leaf_size, metric=metric_fn,
                            device=str(self.device), dtype=self.dtype,
                            n_jobs=n_jobs)
        if alg == 'kd_tree':
            return KDTree(X, leaf_size=self.leaf_size, metric=metric_fn,
                          device=str(self.device), dtype=self.dtype,
                          n_jobs=n_jobs)
        return None  # brute force

    def _knn_brute(self, X_query: torch.Tensor, X_train: torch.Tensor,
                   k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Brute-force k-NN: returns (distances, indices), shape (n_q, k)."""
        D = self._compute_distances(X_query, X_train)  # (n_q, n_train)
        k_eff = min(k, X_train.shape[0])
        dists, idx = torch.topk(D, k=k_eff, dim=1, largest=False, sorted=True)
        return dists, idx

    def _knn_query(self, X_query: torch.Tensor, X_train: torch.Tensor,
                   k: int, tree=None,
                   is_train_query: bool = False
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (distances, indices) of k-nearest neighbours.

        When ``is_train_query=True`` we request k+1 and drop the self-hit
        (first column) to avoid returning a point as its own neighbour.
        """
        extra = 1 if is_train_query else 0
        k_req = k + extra

        if tree is not None:
            dists, idx = tree.query(X_query, k=k_req)
            if dists.dim() == 1:
                dists = dists.unsqueeze(0)
                idx = idx.unsqueeze(0)
        else:
            dists, idx = self._knn_brute(X_query, X_train, k_req)

        if is_train_query:
            # Drop self-hit column (col 0 should be distance 0 to self)
            dists = dists[:, extra:extra + k]
            idx = idx[:, extra:extra + k]
        else:
            dists = dists[:, :k]
            idx = idx[:, :k]

        return dists, idx

    def _radius_brute(self, X_query: torch.Tensor, X_train: torch.Tensor,
                      r: float, sort_results: bool = False,
                      is_train_query: bool = False
                      ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Brute-force radius search. Returns lists of distances and indices."""
        D = self._compute_distances(X_query, X_train)  # (n_q, n_train)
        n_q = X_query.shape[0]
        all_dists = []
        all_idx = []
        for i in range(n_q):
            row = D[i]
            if is_train_query:
                # Exclude exact self-hit (distance == 0 at own index i)
                mask = (row <= r) & ~(
                    torch.arange(row.shape[0], device=row.device) == i)
            else:
                mask = row <= r
            d_i = row[mask]
            idx_i = mask.nonzero(as_tuple=False).squeeze(1)
            if sort_results and d_i.numel() > 0:
                order = d_i.argsort()
                d_i = d_i[order]
                idx_i = idx_i[order]
            all_dists.append(d_i)
            all_idx.append(idx_i)
        return all_dists, all_idx

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, data_or_X, y=None, **kwargs) -> "NearestNeighbors":
        """Fit the nearest neighbors estimator from the training dataset.

        Parameters
        ----------
        data_or_X : array-like of shape (n_samples, n_features) or
                    (n_samples, n_samples) if metric='precomputed'
            Training data.
        y : ignored

        Returns
        -------
        self : NearestNeighbors
            The fitted nearest neighbors estimator.
        """
        if isinstance(data_or_X, (pd.DataFrame, pd.Series)):
            data_or_X = data_or_X.values
        if isinstance(data_or_X, np.ndarray):
            data_or_X = torch.from_numpy(data_or_X)
        X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self.n_samples_fit_ = n_samples
        self._X_fit = X

        # Record effective metric
        if isinstance(self._metric_spec, str):
            self.effective_metric_ = self._metric_spec
        else:
            self.effective_metric_ = 'callable'
        self.effective_metric_params_ = dict(self.metric_params)

        # Build spatial index
        resolved_alg = self._resolve_algorithm(n_samples, n_features)
        if resolved_alg in ('ball_tree', 'kd_tree'):
            self._tree = self._build_tree(X, resolved_alg)
        else:
            self._tree = None

        self.fit_status = True
        return self

    def kneighbors(self,
                   X=None,
                   n_neighbors: Optional[int] = None,
                   return_distance: bool = True
                   ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Find the K-neighbors of a point.

        Returns indices of and distances to the neighbors of each point.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features), default=None
            The query point or points. If not provided, neighbors of each
            indexed point are returned. In this case, the query point is
            not considered its own neighbor.

        n_neighbors : int, default=None
            Number of neighbors required for each sample. The default is the
            value passed to the constructor.

        return_distance : bool, default=True
            Whether or not to return the distances.

        Returns
        -------
        neigh_dist : torch.Tensor of shape (n_queries, n_neighbors)
            Array representing the distances to the points, only present if
            ``return_distance=True``.

        neigh_ind : torch.Tensor of shape (n_queries, n_neighbors)
            Indices of the nearest points in the population matrix.
        """
        if not self.fit_status:
            raise RuntimeError("NearestNeighbors is not fitted yet. Call fit() first.")

        k = n_neighbors if n_neighbors is not None else self.n_neighbors
        k = min(k, self.n_samples_fit_)

        is_train_query = X is None
        if is_train_query:
            X_q = self._X_fit
        else:
            if isinstance(X, (pd.DataFrame, pd.Series)):
                X = X.values
            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X)
            X_q = torch.as_tensor(X, device=self.device, dtype=self.dtype)
            if X_q.dim() == 1:
                X_q = X_q.unsqueeze(0)

        dists, idx = self._knn_query(
            X_q, self._X_fit, k, self._tree,
            is_train_query=is_train_query
        )

        if return_distance:
            return dists, idx
        return idx

    def kneighbors_graph(self,
                         X=None,
                         n_neighbors: Optional[int] = None,
                         mode: Literal["connectivity", "distance"] = 'connectivity'
                         ) -> torch.Tensor:
        """Compute the (weighted) graph of k-Neighbors for points in X.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features), default=None
            The query point or points. If not provided, neighbors of each
            indexed point are returned.

        n_neighbors : int, default=None
            Number of neighbors for each sample. The default is the value
            passed to the constructor.

        mode : {'connectivity', 'distance'}, default='connectivity'
            Type of returned matrix:
            - 'connectivity': returns a binary adjacency matrix (0/1).
            - 'distance': returns a weighted adjacency matrix with distances.

        Returns
        -------
        A : torch.Tensor of shape (n_queries, n_samples_fit)
            Graph where A[i, j] is the connectivity weight between point i
            and its k-th neighbor j.  Zeros indicate no edge.
        """
        if not self.fit_status:
            raise RuntimeError("NearestNeighbors is not fitted yet. Call fit() first.")

        k = n_neighbors if n_neighbors is not None else self.n_neighbors

        is_train_query = X is None
        if is_train_query:
            X_q = self._X_fit
        else:
            if isinstance(X, (pd.DataFrame, pd.Series)):
                X = X.values
            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X)
            X_q = torch.as_tensor(X, device=self.device, dtype=self.dtype)
            if X_q.dim() == 1:
                X_q = X_q.unsqueeze(0)

        n_q = X_q.shape[0]
        n_train = self.n_samples_fit_

        dists, idx = self._knn_query(
            X_q, self._X_fit, k, self._tree,
            is_train_query=is_train_query
        )

        # Build dense adjacency matrix
        A = torch.zeros(n_q, n_train, device=self.device, dtype=self.dtype)

        # Scatter: for each query i, set A[i, idx[i, j]] = 1 or dist
        row_idx = torch.arange(n_q, device=self.device).unsqueeze(1).expand_as(idx)
        if mode == 'connectivity':
            A[row_idx, idx] = 1.0
        elif mode == 'distance':
            A[row_idx, idx] = dists
        else:
            raise ValueError(
                f"Unknown mode '{mode}'. Valid options are 'connectivity' and 'distance'."
            )
        return A

    def radius_neighbors(self,
                         X=None,
                         radius: Optional[float] = None,
                         return_distance: bool = True,
                         sort_results: bool = False
                         ) -> Union[Tuple[List[torch.Tensor], List[torch.Tensor]],
                                    List[torch.Tensor]]:
        """Find the neighbors within a given radius of a point or points.

        Return indices of and distances to the neighbors of each point.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default=None
            The query point or points. If not provided, neighbors of each
            indexed point are returned.  In this case, the query point is
            not considered its own neighbor.

        radius : float, default=None
            Limiting distance of neighbors to return. The default is the
            value passed to the constructor.

        return_distance : bool, default=True
            Whether or not to return the distances.

        sort_results : bool, default=False
            If True, the distances and indices will be sorted in increasing
            order of distances before being returned.

        Returns
        -------
        neigh_dist : list of torch.Tensor of shape (n_neighbors,)
            Array representing the distances to the points, only present if
            ``return_distance=True``.  The array is of objects because the
            number of neighbors of each point may differ.

        neigh_ind : list of torch.Tensor of shape (n_neighbors,)
            An array of arrays of indices of the approximate nearest points
            from the population matrix that lie within a ball of size
            ``radius`` around the query points.
        """
        if not self.fit_status:
            raise RuntimeError("NearestNeighbors is not fitted yet. Call fit() first.")

        r = radius if radius is not None else self.radius

        is_train_query = X is None
        if is_train_query:
            X_q = self._X_fit
        else:
            if isinstance(X, (pd.DataFrame, pd.Series)):
                X = X.values
            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X)
            X_q = torch.as_tensor(X, device=self.device, dtype=self.dtype)
            if X_q.dim() == 1:
                X_q = X_q.unsqueeze(0)

        # Use tree's query_radius if available; fall back to brute force
        if self._tree is not None and hasattr(self._tree, 'query_radius'):
            try:
                tree_result = self._tree.query_radius(
                    X_q, r=r,
                    return_distance=True,
                    sort_results=sort_results
                )
                # BallTree/KDTree returns (distances_list, indices_list) or
                # just indices_list depending on return_distance
                if return_distance:
                    all_dists, all_idx = tree_result
                else:
                    if isinstance(tree_result, tuple):
                        _, all_idx = tree_result
                    else:
                        all_idx = tree_result
                        all_dists = [None] * len(all_idx)

                # When querying training set, exclude self-hits (dist ≈ 0)
                if is_train_query:
                    n_q = X_q.shape[0]
                    filtered_dists, filtered_idx = [], []
                    for i in range(n_q):
                        d_i = all_dists[i]
                        idx_i = all_idx[i]
                        if d_i is not None:
                            mask = d_i > 0
                            d_i = d_i[mask]
                            idx_i = idx_i[mask]
                        filtered_dists.append(d_i)
                        filtered_idx.append(idx_i)
                    all_dists, all_idx = filtered_dists, filtered_idx

                if return_distance:
                    return all_dists, all_idx
                return all_idx
            except Exception:
                pass  # Fall back to brute force

        # Brute force radius search
        all_dists, all_idx = self._radius_brute(
            X_q, self._X_fit, r,
            sort_results=sort_results,
            is_train_query=is_train_query
        )

        if return_distance:
            return all_dists, all_idx
        return all_idx

    def radius_neighbors_graph(self,
                                X=None,
                                radius: Optional[float] = None,
                                mode: Literal["connectivity", "distance"] = 'connectivity',
                                sort_results: bool = False
                                ) -> torch.Tensor:
        """Compute the (weighted) graph of Neighbors for points in X.

        Neighborhoods are restricted to the ball of given radius.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default=None
            The query point or points. If not provided, neighbors of each
            indexed point are returned.

        radius : float, default=None
            Radius of neighborhoods. The default is the value passed to
            the constructor.

        mode : {'connectivity', 'distance'}, default='connectivity'
            Type of returned matrix:
            - 'connectivity': binary adjacency matrix (0 or 1).
            - 'distance': weighted adjacency matrix with distances.

        sort_results : bool, default=False
            If True, in each row the non-zero entries will be sorted by
            increasing distances.

        Returns
        -------
        A : torch.Tensor of shape (n_queries, n_samples_fit)
            Graph where A[i, j] is the connectivity weight between point i
            and point j when j lies within the radius ball around i.
        """
        if not self.fit_status:
            raise RuntimeError("NearestNeighbors is not fitted yet. Call fit() first.")

        r = radius if radius is not None else self.radius

        is_train_query = X is None
        if is_train_query:
            X_q = self._X_fit
        else:
            if isinstance(X, (pd.DataFrame, pd.Series)):
                X = X.values
            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X)
            X_q = torch.as_tensor(X, device=self.device, dtype=self.dtype)
            if X_q.dim() == 1:
                X_q = X_q.unsqueeze(0)

        n_q = X_q.shape[0]
        n_train = self.n_samples_fit_

        result = self.radius_neighbors(
            X=None if is_train_query else X_q,
            radius=r,
            return_distance=True,
            sort_results=sort_results
        )
        all_dists, all_idx = result

        A = torch.zeros(n_q, n_train, device=self.device, dtype=self.dtype)
        for i in range(n_q):
            idx_i = all_idx[i]
            if idx_i.numel() == 0:
                continue
            if mode == 'connectivity':
                A[i, idx_i] = 1.0
            elif mode == 'distance':
                d_i = all_dists[i]
                if d_i is not None and d_i.numel() > 0:
                    A[i, idx_i] = d_i.to(dtype=self.dtype)
                else:
                    A[i, idx_i] = 1.0
            else:
                raise ValueError(
                    f"Unknown mode '{mode}'. Valid options are "
                    "'connectivity' and 'distance'."
                )
        return A

    def predict(self, X) -> torch.Tensor:
        """Return indices of nearest neighbors (wraps :meth:`kneighbors`)."""
        return self.kneighbors(X, return_distance=False)

    def forward(self, X) -> torch.Tensor:
        """Return distances to k-nearest neighbors for each point in X."""
        dists, _ = self.kneighbors(X, return_distance=True)
        return dists

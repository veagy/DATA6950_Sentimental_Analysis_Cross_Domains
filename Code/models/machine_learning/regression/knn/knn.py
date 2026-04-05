import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Union, Any, List, Tuple, Dict, Literal, Optional
import math
from .....models.utils import MLModule, MLRegressor
import warnings


__all__ = [
    "NeighborhoodComponentsAnalysis",
    "BallTree",
    "KDTree",
    "KNeighboursRegression",
    "RadiusNeighborsRegressor"
]


def _nca_objective_and_grad(
    L_flat: torch.Tensor,
    X: torch.Tensor,
    y_indices: torch.Tensor,
    n_components: int,
    n_features: int,
    same_class_mask: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[float, torch.Tensor]:
    n_samples = X.shape[0]
    L = L_flat.view(n_components, n_features)
    L = L.to(device=device, dtype=dtype)
    X = X.to(device=device, dtype=dtype)
    X_proj = X @ L.T
    sq_norms = (X_proj ** 2).sum(dim=1)
    dist_sq = sq_norms.unsqueeze(1) + sq_norms.unsqueeze(0) - 2 * (X_proj @ X_proj.T)
    dist_sq = dist_sq.clamp(min=0)
    exp_neg_d = torch.exp(-dist_sq)
    exp_neg_d.fill_diagonal_(0)
    Z = exp_neg_d.sum(dim=1, keepdim=True).clamp(min=1e-12)
    p = exp_neg_d / Z
    F_val = (p * same_class_mask).sum().item()
    L_param = L.detach().requires_grad_(True)
    X_proj_g = X @ L_param.T
    sq_norms_g = (X_proj_g ** 2).sum(dim=1)
    dist_sq_g = sq_norms_g.unsqueeze(1) + sq_norms_g.unsqueeze(0) - 2 * (X_proj_g @ X_proj_g.T)
    dist_sq_g = dist_sq_g.clamp(min=0)
    exp_neg_d_g = torch.exp(-dist_sq_g).clone()
    exp_neg_d_g.fill_diagonal_(0)
    Z_g = exp_neg_d_g.sum(dim=1, keepdim=True).clamp(min=1e-12)
    p_g = exp_neg_d_g / Z_g
    obj_g = (p_g * same_class_mask).sum()
    obj_g.backward()
    grad = L_param.grad
    if grad is None:
        grad = torch.zeros_like(L_param)
    return -F_val, -grad.flatten().detach()


def _nca_objective_value(
    L_flat: torch.Tensor,
    X: torch.Tensor,
    n_components: int,
    n_features: int,
    same_class_mask: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    with torch.no_grad():
        n_samples = X.shape[0]
        L = L_flat.view(n_components, n_features).to(device=device, dtype=dtype)
        X_t = X.to(device=device, dtype=dtype)
        X_proj = X_t @ L.T
        sq_norms = (X_proj ** 2).sum(dim=1)
        dist_sq = sq_norms.unsqueeze(1) + sq_norms.unsqueeze(0) - 2 * (X_proj @ X_proj.T)
        dist_sq = dist_sq.clamp(min=0)
        exp_neg_d = torch.exp(-dist_sq)
        exp_neg_d.fill_diagonal_(0)
        Z = exp_neg_d.sum(dim=1, keepdim=True).clamp(min=1e-12)
        p = exp_neg_d / Z
        F_val = (p * same_class_mask).sum().item()
    return -F_val


class NeighborhoodComponentsAnalysis(MLRegressor):
    """
    Neighborhood Components Analysis.
    Learns a linear transformation in a supervised fashion to improve
    k-NN classification accuracy in the transformed space.
    """

    def __init__(self,
                 n_components: int = None,
                 init: Union[Literal["auto", "pca", "lda", "identity", "random"], list, tuple, torch.Tensor] = "auto",
                 warm_start: bool = False,
                 max_iter: int = 50,
                 tol: float = 1e-5,
                 callback: Callable = None,
                 verbose: int = 0,
                 random_state: Union[int, torch.Generator] = None,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
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
        self.random_state_ = self._resolve_random_state(random_state)

    def _resolve_random_state(self, random_state):
        return random_state

    def _get_random_generator(self) -> Optional[torch.Generator]:
        if self.random_state is None:
            return None
        if isinstance(self.random_state, torch.Generator):
            return self.random_state
        dev = self.device if isinstance(self.device, torch.device) else torch.device(self.device)
        g = torch.Generator(device=dev)
        g.manual_seed(int(self.random_state))
        return g

    def _init_transformation(self, X, y_indices, n_comp, n_features, n_samples, n_classes):
        init_val = self.init
        gen = self._get_random_generator()
        if isinstance(init_val, (list, tuple)):
            L = torch.as_tensor(init_val, device=self.device, dtype=self.dtype)
        elif hasattr(init_val, "__array__") and not isinstance(init_val, torch.Tensor):
            L = torch.as_tensor(init_val, device=self.device, dtype=self.dtype)
        elif isinstance(init_val, torch.Tensor):
            L = init_val.to(device=self.device, dtype=self.dtype)
        else:
            init_str = str(init_val).lower() if isinstance(init_val, str) else "auto"

        if isinstance(init_val, (list, tuple, torch.Tensor)) or (
            hasattr(init_val, "__array__") and not isinstance(init_val, torch.Tensor)
        ):
            if L.shape[0] != n_comp or L.shape[1] != n_features:
                raise ValueError(f"init array shape {L.shape} must be ({n_comp}, {n_features})")
            return L

        if init_str == "auto":
            init_str = "lda" if n_comp <= min(n_features, n_classes - 1) else (
                "pca" if n_comp < min(n_features, n_samples) else "identity")
        if init_str == "identity":
            L = torch.eye(n_comp, n_features, device=self.device, dtype=self.dtype)
        elif init_str == "random":
            L = torch.randn(n_comp, n_features, device=self.device, dtype=self.dtype, generator=gen)
        elif init_str == "pca":
            X_centered = X - X.mean(dim=0)
            _, _, Vh = torch.linalg.svd(X_centered, full_matrices=False)
            L = Vh[:n_comp].clone()
        elif init_str == "lda":
            L = self._init_lda(X, y_indices, n_comp, n_features, n_samples, n_classes)
        else:
            L = torch.eye(n_comp, n_features, device=self.device, dtype=self.dtype)
        return L

    def _init_lda(self, X, y_indices, n_comp, n_features, n_samples, n_classes):
        eps = 1e-12
        means = torch.zeros(n_classes, n_features, device=self.device, dtype=self.dtype)
        for c in range(n_classes):
            mask = y_indices == c
            if mask.any():
                means[c] = X[mask].mean(dim=0)
        xbar = means.mean(dim=0)
        Sw = torch.zeros(n_features, n_features, device=self.device, dtype=self.dtype)
        for c in range(n_classes):
            mask = y_indices == c
            if mask.any():
                Sw = Sw + (X[mask] - means[c]).T @ (X[mask] - means[c])
        Sw = Sw / max(n_samples - n_classes, 1)
        Sb = (means - xbar).T @ (means - xbar) * n_samples / n_classes
        Sw = Sw + eps * torch.eye(n_features, device=self.device, dtype=self.dtype)
        try:
            L_chol = torch.linalg.cholesky(Sw)
            L_inv = torch.linalg.inv(L_chol)
            W = L_inv @ Sb @ L_inv.T
            evals, evecs = torch.linalg.eigh(W)
            idx = evals.argsort(descending=True)
            evecs = evecs[:, idx]
            n_lda = max(1, min(n_comp, n_classes - 1, (evals > eps).sum().item()))
            scalings = L_inv.T @ evecs[:, :n_lda]
            L = torch.zeros(n_comp, n_features, device=self.device, dtype=self.dtype)
            L[:n_lda] = scalings.T
        except RuntimeError:
            L = torch.eye(n_comp, n_features, device=self.device, dtype=self.dtype)
        return L

    def fit(self, data_or_X, y=None, **kwargs):
        if isinstance(data_or_X, (list, tuple)) and len(data_or_X) == 2 and y is None:
            X, y = data_or_X
        else:
            X = data_or_X
        y = y if y is not None else kwargs.get("y")
        if y is None:
            raise ValueError("NeighborhoodComponentsAnalysis requires y (labels).")
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        y = torch.as_tensor(y, device=self.device, dtype=torch.long)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        if y.ndim > 1:
            y = y[:, 0]
        if y.dim() == 1:
            y = y.squeeze()
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        classes, y_indices = torch.unique(y, return_inverse=True)
        n_classes = len(classes)
        if n_classes < 2:
            raise ValueError("NeighborhoodComponentsAnalysis requires at least 2 classes.")
        n_comp = self.n_components if self.n_components is not None else n_features
        n_comp = min(n_comp, n_features, n_samples)
        same_class_mask = (y_indices.unsqueeze(1) == y_indices.unsqueeze(0)).to(self.dtype)
        same_class_mask.fill_diagonal_(0)
        if self.warm_start and self.components_ is not None:
            L_init = self.components_.clone()
            if L_init.shape[0] != n_comp or L_init.shape[1] != n_features:
                L_init = self._init_transformation(X, y_indices, n_comp, n_features, n_samples, n_classes)
        else:
            L_init = self._init_transformation(X, y_indices, n_comp, n_features, n_samples, n_classes)
        L_flat = L_init.flatten().clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS([L_flat], lr=1.0, max_iter=20, line_search_fn="strong_wolfe")
        prev_obj = float("-inf")
        n_iter = 0

        def closure():
            optimizer.zero_grad()
            neg_f, grad = _nca_objective_and_grad(
                L_flat, X, y_indices, n_comp, n_features, same_class_mask, X.device, self.dtype)
            L_flat.grad = grad.view_as(L_flat).clone()
            return torch.tensor(neg_f, device=X.device, dtype=self.dtype)

        for _ in range(self.max_iter):
            optimizer.step(closure)
            neg_f = _nca_objective_value(L_flat, X, n_comp, n_features, same_class_mask, X.device, self.dtype)
            obj = -neg_f
            n_iter += 1
            if self.callback is not None:
                self.callback(L_flat.detach(), n_iter)
            if self.verbose >= 1:
                print(f"NCA iter {n_iter}/{self.max_iter}, objective={obj:.6f}")
            if obj - prev_obj < self.tol:
                break
            prev_obj = obj
        self.components_ = L_flat.detach().view(n_comp, n_features).clone()
        self.n_iter_ = n_iter
        self.random_state_ = self._resolve_random_state(self.random_state)
        self.fit_status = True
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Return transformed X (same as transform)."""
        return self.transform(X)

    def transform(self, X):
        if self.components_ is None:
            raise RuntimeError("NeighborhoodComponentsAnalysis instance is not fitted yet.")
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        return X @ self.components_.T

    def fit_transform(self, data_or_X, y=None, **kwargs):
        kwargs["y"] = y
        self.fit(data_or_X, y=y, **kwargs)
        return self.transform(torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype))


class BallTree(MLModule):
    def __init__(self,
                 data: torch.Tensor = None,
                 leaf_size: int = 40,
                 metric: Callable = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 n_jobs: int = 1,
                 *args, **kwargs):
        super().__init__()
        if data is None:
            data = torch.zeros(1, 1, device=torch.device(device), dtype=dtype)
        self.data = data.to(device).to(dtype)
        self.leaf_size = leaf_size
        self.metric = metric
        self.device = device
        self.dtype = dtype
        self.n_jobs = n_jobs if n_jobs is not None else 1
        self.indices = torch.arange(data.size(0), device=device)
        self.root = self._build(self.indices)
        self._fit_y = None

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = None
        for arg in args:
            if isinstance(arg, (str, torch.device)):
                device = arg
                break
        if "device" in kwargs:
            device = kwargs["device"]
        if device is not None:
            self.device = str(device)
            if hasattr(self, "data") and isinstance(self.data, torch.Tensor):
                self.data = self.data.to(device)
            if hasattr(self, "indices") and isinstance(self.indices, torch.Tensor):
                self.indices = self.indices.to(device)
            if hasattr(self, "_fit_y") and isinstance(self._fit_y, torch.Tensor):
                self._fit_y = self._fit_y.to(device)
                
            def _move_node(node):
                if node is None: return
                if hasattr(node, 'point') and isinstance(node.point, torch.Tensor):
                    node.point = node.point.to(device)
                if hasattr(node, 'idxs') and isinstance(node.idxs, torch.Tensor):
                    node.idxs = node.idxs.to(device)
                _move_node(getattr(node, 'left', None))
                _move_node(getattr(node, 'right', None))
                
            if hasattr(self, "root"):
                _move_node(self.root)
        return self

    def fit(self, data_or_X, y=None, **kwargs):
        """Build the tree from X. Optionally store y for predict (1-NN regression)."""
        X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        self.data = X
        self.indices = torch.arange(X.size(0), device=self.device)
        self.root = self._build(self.indices)
        self._fit_y = torch.as_tensor(y, device=self.device, dtype=self.dtype) if y is not None else None
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Return 1-NN regression predictions (requires y in fit)."""
        if self._fit_y is None:
            raise RuntimeError("BallTree.predict requires y in fit.")
        X = X.to(self.device).to(self.dtype)
        _, indices = self.query(X, k=1, return_distance=True)
        return self._fit_y[indices.squeeze(-1)]

    # ... properties ...

    def query(self, X: torch.Tensor, k: int = 1, return_distance: bool = True):
        X = X.to(self.device).to(self.dtype)
        B = X.size(0)

        distances = torch.full((B, k), float('inf'), device=self.device, dtype=self.dtype)
        indices = torch.full((B, k), -1, device=self.device, dtype=torch.long)

        if self.device == 'cpu' and self.n_jobs != 1:
            from joblib import Parallel, delayed

            def process_point(i):
                d_buf = torch.full((k,), float('inf'), dtype=self.dtype)
                i_buf = torch.full((k,), -1, dtype=torch.long)
                self._query_point(X[i], k, d_buf, i_buf)
                return d_buf, i_buf

            results = Parallel(n_jobs=self.n_jobs)(delayed(process_point)(i) for i in range(B))

            for i, (d, ind) in enumerate(results):
                distances[i] = d
                indices[i] = ind
        else:
            for i in range(B):
                self._query_point(X[i], k, distances[i], indices[i])

        if return_distance:
            return distances, indices
        else:
            return indices

    def _query_point(self, point, k, dists_buf, inds_buf):
        current_best = []

        def push_best(d, idx):
            if len(current_best) < k:
                current_best.append((d, idx))
                current_best.sort(key=lambda x: x[0])
            elif d < current_best[-1][0]:
                current_best[-1] = (d, idx)
                current_best.sort(key=lambda x: x[0])

        def get_worst_dist():
            if len(current_best) < k:
                return float('inf')
            return current_best[-1][0]

        def search(node):
            if node is None:
                return

            # Triangle inequality pruning
            # dist(query, node.center) - node.radius > current_worst
            dist_to_center = self._dist(point, node.point).item()

            # Pruning requires us to know if the sphere can contain a point closer than current worst
            if dist_to_center - node.radius > get_worst_dist():  # Lower bound of dist to any point in ball
                return

            if node.idxs is not None:
                # Leaf
                leaf_data = self.data[node.idxs]
                d = self._dist(point, leaf_data).squeeze()
                if d.ndim == 0:  # single point
                    push_best(d.item(), node.idxs[0].item())
                else:
                    for j, idx in enumerate(node.idxs):
                        push_best(d[j].item(), idx.item())
                return

            # Internal
            # Visit closer child first
            # We don't have split plane distance like KDTree.
            # Heuristic: visit child with closer center
            dist_left = self._dist(point, node.left.point).item()
            dist_right = self._dist(point, node.right.point).item()

            if dist_left < dist_right:
                search(node.left)
                search(node.right)
            else:
                search(node.right)
                search(node.left)

        search(self.root)

        for j, (d, idx) in enumerate(current_best):
            dists_buf[j] = d
            inds_buf[j] = idx

    def query_radius(self, X: torch.Tensor, r: float, return_distance: bool = True, sort_results: bool = False):
        """Return neighbors within radius r for each point in X."""
        X = X.to(self.device).to(self.dtype)
        B = X.size(0)
        all_distances = []
        all_indices = []

        if self.device == 'cpu' and self.n_jobs != 1:
            from joblib import Parallel, delayed

            def process_point_radius(i):
                d_list = []
                i_list = []
                self._query_point_radius(X[i], r, d_list, i_list)
                d_tensor = torch.tensor(d_list, dtype=self.dtype)
                i_tensor = torch.tensor(i_list, dtype=torch.long)
                if sort_results:
                    d_tensor, perm = torch.sort(d_tensor)
                    i_tensor = i_tensor[perm]
                return d_tensor, i_tensor

            results = Parallel(n_jobs=self.n_jobs)(delayed(process_point_radius)(i) for i in range(B))
            for d, ind in results:
                all_distances.append(d)
                all_indices.append(ind)
        else:
            for i in range(B):
                d_list = []
                i_list = []
                self._query_point_radius(X[i], r, d_list, i_list)
                d_tensor = torch.tensor(d_list, dtype=self.dtype).to(self.device)
                i_tensor = torch.tensor(i_list, dtype=torch.long).to(self.device)
                if sort_results:
                    d_tensor, perm = torch.sort(d_tensor)
                    i_tensor = i_tensor[perm]
                all_distances.append(d_tensor)
                all_indices.append(i_tensor)

        if return_distance:
            return all_distances, all_indices
        else:
            return all_indices

    def _query_point_radius(self, point, r, dists_list, inds_list):
        """Find all points within radius r of point (BallTree triangle-inequality pruning)."""
        def search(node):
            if node is None:
                return
            dist_to_center = self._dist(point, node.point).item()
            if dist_to_center - node.radius > r:
                return
            if node.idxs is not None:
                leaf_data = self.data[node.idxs]
                d = self._dist(point, leaf_data).squeeze()
                if d.ndim == 0:
                    if d.item() <= r:
                        dists_list.append(d.item())
                        inds_list.append(node.idxs[0].item())
                else:
                    mask = d <= r
                    for j in torch.where(mask)[0]:
                        dists_list.append(d[j].item())
                        inds_list.append(node.idxs[j].item())
                return
            search(node.left)
            search(node.right)

        search(self.root)

    def forward(self, X: torch.Tensor, y: torch.Tensor = None, **kwargs) -> torch.Tensor:
        return self.query(X)

    def _dist(self, xi, xj):
        """Compute distance between xi and xj. xi: (d,) or (1,d), xj: (d,) or (n,d)."""
        if self.metric is not None and callable(self.metric):
            if xi.dim() == 1:
                xi = xi.unsqueeze(0)
            if xj.dim() == 1:
                xj = xj.unsqueeze(0)
            d = self.metric(xi, xj)
            return d.squeeze()
        if xi.dim() == 1:
            xi = xi.unsqueeze(0)
        if xj.dim() == 1:
            xj = xj.unsqueeze(0)
        return torch.norm(xi - xj, dim=-1)

    def _build(self, idxs, depth=0):
        """Build BallTree: split by max-variance dimension, use centroid and radius."""
        if len(idxs) <= self.leaf_size:
            pts = self.data[idxs]
            centroid = pts.mean(dim=0)
            if pts.shape[0] == 1:
                radius = 0.0
            else:
                d = self._dist(centroid.unsqueeze(0), pts).squeeze()
                radius = d.max().item()
            return Node(idxs=idxs, point=centroid, radius=radius)
        data_subset = self.data[idxs]
        variances = torch.var(data_subset, dim=0)
        axis = torch.argmax(variances).item()
        sorted_indices = torch.argsort(data_subset[:, axis])
        sorted_idxs = idxs[sorted_indices]
        mid = len(idxs) // 2
        left_idxs = sorted_idxs[:mid]
        right_idxs = sorted_idxs[mid:]
        left_node = self._build(left_idxs, depth + 1)
        right_node = self._build(right_idxs, depth + 1)
        centroid = (left_node.point * len(left_idxs) + right_node.point * len(right_idxs)) / len(idxs)
        left_rad = self._dist(centroid.unsqueeze(0), self.data[left_idxs]).max().item()
        right_rad = self._dist(centroid.unsqueeze(0), self.data[right_idxs]).max().item()
        radius = max(left_rad, right_rad)
        return Node(left=left_node, right=right_node, point=centroid, radius=radius)


class Node:
    def __init__(self, idxs=None, point=None, left=None, right=None, axis=None, split_val=None, radius=None):
        self.idxs = idxs  # indices of data points in this node (for leaf)
        self.point = point  # centroid or representative point (optional)
        self.left = left  # left child
        self.right = right  # right child
        self.axis = axis  # split axis (for KDTree)
        self.split_val = split_val  # split value (for KDTree)
        self.radius = radius  # max distance from center to any point (for BallTree)


class KDTree(MLModule):
    def __init__(self,
                 data: torch.Tensor = None,
                 leaf_size: int = 40,
                 metric: Callable = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 n_jobs: int = 1,
                 *args, **kwargs):
        super().__init__()
        if data is None:
            data = torch.zeros(1, 1, device=torch.device(device), dtype=dtype)
        self.data = data.to(device).to(dtype)
        self.leaf_size = leaf_size
        self.metric = metric
        self.device = device
        self.dtype = dtype
        self.n_jobs = n_jobs if n_jobs is not None else 1
        self.indices = torch.arange(data.size(0), device=device)
        self.root = self._build(self.indices)
        self._fit_y = None

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = None
        for arg in args:
            if isinstance(arg, (str, torch.device)):
                device = arg
                break
        if "device" in kwargs:
            device = kwargs["device"]
        if device is not None:
            self.device = str(device)
            if hasattr(self, "data") and isinstance(self.data, torch.Tensor):
                self.data = self.data.to(device)
            if hasattr(self, "indices") and isinstance(self.indices, torch.Tensor):
                self.indices = self.indices.to(device)
            if hasattr(self, "_fit_y") and isinstance(self._fit_y, torch.Tensor):
                self._fit_y = self._fit_y.to(device)
                
            def _move_node(node):
                if node is None: return
                if hasattr(node, 'point') and isinstance(node.point, torch.Tensor):
                    node.point = node.point.to(device)
                if hasattr(node, 'idxs') and isinstance(node.idxs, torch.Tensor):
                    node.idxs = node.idxs.to(device)
                _move_node(getattr(node, 'left', None))
                _move_node(getattr(node, 'right', None))
                
            if hasattr(self, "root"):
                _move_node(self.root)
        return self

    def fit(self, data_or_X, y=None, **kwargs):
        """Build the tree from X. Optionally store y for predict (1-NN regression)."""
        X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        self.data = X
        self.indices = torch.arange(X.size(0), device=self.device)
        self.root = self._build(self.indices)
        self._fit_y = torch.as_tensor(y, device=self.device, dtype=self.dtype) if y is not None else None
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Return 1-NN regression predictions (requires y in fit)."""
        if self._fit_y is None:
            raise RuntimeError("KDTree.predict requires y in fit.")
        X = X.to(self.device).to(self.dtype)
        _, indices = self.query(X, k=1, return_distance=True)
        return self._fit_y[indices.squeeze(-1)]

    # ... properties ...

    def query(self, X: torch.Tensor, k: int = 1, return_distance: bool = True):
        X = X.to(self.device).to(self.dtype)
        B = X.size(0)

        distances = torch.full((B, k), float('inf'), device=self.device, dtype=self.dtype)
        indices = torch.full((B, k), -1, device=self.device, dtype=torch.long)

        if self.device == 'cpu' and self.n_jobs != 1:
            from joblib import Parallel, delayed

            def process_point(i):
                d_buf = torch.full((k,), float('inf'), dtype=self.dtype)
                i_buf = torch.full((k,), -1, dtype=torch.long)
                self._query_point(X[i], k, d_buf, i_buf)
                return d_buf, i_buf

            results = Parallel(n_jobs=self.n_jobs)(delayed(process_point)(i) for i in range(B))

            for i, (d, ind) in enumerate(results):
                distances[i] = d
                indices[i] = ind
        else:
            for i in range(B):
                self._query_point(X[i], k, distances[i], indices[i])

        if return_distance:
            return distances, indices
        else:
            return indices

    def query_radius(self, X: torch.Tensor, r: float, return_distance: bool = True, sort_results: bool = False):
        X = X.to(self.device).to(self.dtype)
        B = X.size(0)

        all_distances = []
        all_indices = []

        if self.device == 'cpu' and self.n_jobs != 1:
            from joblib import Parallel, delayed

            def process_point_radius(i):
                d_list = []
                i_list = []
                self._query_point_radius(X[i], r, d_list, i_list)
                d_tensor = torch.tensor(d_list, dtype=self.dtype)
                i_tensor = torch.tensor(i_list, dtype=torch.long)
                if sort_results:
                    d_tensor, perm = torch.sort(d_tensor)
                    i_tensor = i_tensor[perm]
                return d_tensor, i_tensor

            results = Parallel(n_jobs=self.n_jobs)(delayed(process_point_radius)(i) for i in range(B))

            for d, ind in results:
                all_distances.append(d)
                all_indices.append(ind)
        else:
            for i in range(B):
                d_list = []
                i_list = []
                self._query_point_radius(X[i], r, d_list, i_list)
                d_tensor = torch.tensor(d_list, dtype=self.dtype).to(self.device)
                i_tensor = torch.tensor(i_list, dtype=torch.long).to(self.device)
                if sort_results:
                    d_tensor, perm = torch.sort(d_tensor)
                    i_tensor = i_tensor[perm]
                all_distances.append(d_tensor)
                all_indices.append(i_tensor)

        # Return object array of arrays (list of tensors here)
        if return_distance:
            return all_distances, all_indices
        else:
            return all_indices

    def _dist(self, xi, xj):
        """Compute distance between xi and xj. xi: (d,) or (1,d), xj: (d,) or (n,d)."""
        if self.metric is not None and callable(self.metric):
            if xi.dim() == 1:
                xi = xi.unsqueeze(0)
            if xj.dim() == 1:
                xj = xj.unsqueeze(0)
            d = self.metric(xi, xj)
            return d.squeeze()
        if xi.dim() == 1:
            xi = xi.unsqueeze(0)
        if xj.dim() == 1:
            xj = xj.unsqueeze(0)
        return torch.norm(xi - xj, dim=-1)

    def _query_point_radius(self, point, r, dists_list, inds_list):
        def search(node):
            if node is None:
                return

            if node.idxs is not None:
                leaf_data = self.data[node.idxs]
                d = self._dist(point, leaf_data)
                if d.ndim == 0:
                    d = d.unsqueeze(0)
                mask = d <= r
                valid_dists = d[mask]
                valid_idxs = node.idxs[mask]

                if valid_dists.numel() > 0:
                    dists_list.extend(valid_dists.tolist())
                    inds_list.extend(valid_idxs.tolist())
                return

            diff = point[node.axis].item() - node.split_val
            
            # Check bounding box distance logic for KDTree
            # For KDTree, "distance to node" is max(0, abs(diff) if pruning)
            # We visit if minimum distance to node's region <= r
            
            close_child = node.left if diff < 0 else node.right
            far_child = node.right if diff < 0 else node.left

            search(close_child)

            if abs(diff) <= r:
                search(far_child)

        search(self.root)

    def _build(self, idxs, depth=0):
        if len(idxs) <= self.leaf_size:
            return Node(idxs=idxs)

        data_subset = self.data[idxs]
        variances = torch.var(data_subset, dim=0)
        axis = torch.argmax(variances).item()

        sorted_indices = torch.argsort(data_subset[:, axis])
        sorted_idxs = idxs[sorted_indices]

        mid = len(idxs) // 2
        split_val = data_subset[sorted_indices[mid], axis].item()

        left_idxs = sorted_idxs[:mid]
        right_idxs = sorted_idxs[mid:]

        return Node(
            left=self._build(left_idxs, depth + 1),
            right=self._build(right_idxs, depth + 1),
            axis=axis,
            split_val=split_val
        )

    def _query_point(self, point, k, dists_buf, inds_buf):
        current_best = []

        def push_best(d, idx):
            if len(current_best) < k:
                current_best.append((d, idx))
                current_best.sort(key=lambda x: x[0])
            elif d < current_best[-1][0]:
                current_best[-1] = (d, idx)
                current_best.sort(key=lambda x: x[0])

        def get_worst_dist():
            if len(current_best) < k:
                return float('inf')
            return current_best[-1][0]

        def search(node):
            if node is None:
                return

            if node.idxs is not None:
                leaf_data = self.data[node.idxs]
                d = self._dist(point, leaf_data).squeeze()
                if d.ndim == 0:
                    push_best(d.item(), node.idxs[0].item())
                else:
                    for j, idx in enumerate(node.idxs):
                        push_best(d[j].item(), idx.item())
                return

            diff = point[node.axis].item() - node.split_val

            close_child = node.left if diff < 0 else node.right
            far_child = node.right if diff < 0 else node.left

            search(close_child)

            if abs(diff) < get_worst_dist():
                search(far_child)

        search(self.root)

        for j, (d, idx) in enumerate(current_best):
            dists_buf[j] = d
            inds_buf[j] = idx

    def forward(self, X: torch.Tensor, y: torch.Tensor = None, **kwargs) -> torch.Tensor:
        return self.query(X)


class KNeighboursRegression(MLRegressor):
    def __init__(self,
                 n_neighbors: int = 5,
                 weights: Union[str, Callable] = "uniform",
                 algorithm: str = "auto",
                 leaf_size: int = 30,
                 p: float = 2,
                 metric: Union[str, Callable, object] = "minkowski",
                 metric_params: dict = None,
                 n_jobs: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        if metric_params is None:
            metric_params = {}
        metric_params["p"] = p
        from .....models.utils import MLCluster
        self.metric = MLCluster()._create_metric(metric, metric_params).metric
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype
        self.in_features = None
        self.out_features = None

    @property
    def effective_metric_(self):
        return self.metric

    @property
    def effective_metric_params(self):
        return {}

    @property
    def n_features_in_(self):
        return self.in_features

    def _init_module_(self, X, y):
        self.in_features = X.size(-1) if X.ndim > 1 else 1
        self.out_features = y.size(-1) if y.ndim > 1 else 1
        return self

    def forward(self, X: torch.Tensor, y: torch.Tensor = None, **kwargs) -> torch.Tensor:
        if not self.fit_status:
            if self.in_features is None:
                self._init_module_(X, y)
            self.fit(X, y)
        return self.predict(X)

    def fit(self, data_or_X, y=None, **kwargs):
        if isinstance(data_or_X, (list, tuple)) and len(data_or_X) == 2 and y is None:
            X, y = data_or_X
        else:
            X = data_or_X

        if y is None:
            raise ValueError("y must be provided for KNeighboursRegression fit")

        if self.in_features is None:
            self._init_module_(X, y)

        # Move to device if needed
        if self.device:
            X = X.to(self.device)
            y = y.to(self.device)

        self._fit_X = X
        self._fit_y = y

        # Decide algorithm
        self._algorithm = self.algorithm
        if self._algorithm == 'auto':
            # Default to brute for GPU or if metric is complex.
            # Use tree if CPU, standard metric, and reasonable size is implicit in user choice usually.
            # Here we default to brute if on CUDA, else brute (since pyTorch is fast on matrix ops).
            # Unless user asks for tree, brute is often faster for batch queries in PyTorch.
            # However, we implement tree support if possible for CPU users.
            metric_name = getattr(self, '_metric_name', None)
            is_standard_metric = (
                        isinstance(metric_name, str) and metric_name.lower() in ["minkowski", "euclidean", "l2",
                                                                                 "manhattan", "l1", "chebyshev",
                                                                                 "l2_distance", "l1_distance"])
            if is_standard_metric and (not X.is_cuda):
                self._algorithm = 'kd_tree'  # Default to KDTree for standard metrics on CPU
            else:
                self._algorithm = 'brute'

        if self._algorithm in ['ball_tree', 'kd_tree']:
            # Use custom implementations
            p = 2

            # Helper to get callable metric if self.metric is not callable
            metric_callable = None
            if isinstance(self.metric, Callable):
                metric_callable = self.metric
            else:
                # If metric is string or object, the trees need a callable.
                # However, our custom trees' _dist method currently mostly relies on self.metric being callable OR None (euclidean).
                # If we passed a string metric, self.metric IS a lambda in __init__.
                # So self.metric is ready to use.
                metric_callable = self.metric

            try:
                if self._algorithm == 'ball_tree':
                    self._tree = BallTree(X, leaf_size=self.leaf_size, metric=metric_callable, p=p, device=self.device,
                                          dtype=self.dtype, n_jobs=self.n_jobs)
                else:
                    self._tree = KDTree(X, leaf_size=self.leaf_size, metric=metric_callable, p=p, device=self.device,
                                        dtype=self.dtype, n_jobs=self.n_jobs)
            except Exception as e:
                warnings.warn(f"Failed to build tree: {e}. Falling back to brute force.")
                self._algorithm = 'brute'

        self.fit_status = True
        return self

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        if X is None:
            if self._fit_X is None:
                raise RuntimeError("Model likely not fitted.")
            X = self._fit_X

        if self.device:
            X = X.to(self.device)

        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        if self._fit_X is None:
            raise RuntimeError("Model handles not fitted yet. Call 'fit' first.")

        # Ensure stored training data is on the same device as X
        if self._fit_X is not None and self._fit_X.device != X.device:
            self._fit_X = self._fit_X.to(X.device)
        if self._fit_y is not None and self._fit_y.device != X.device:
            self._fit_y = self._fit_y.to(X.device)

        if self._algorithm in ['ball_tree', 'kd_tree']:
            dists, inds = self._tree.query(X, k=n_neighbors, return_distance=True)
            dists = dists.to(X.device)
            inds = inds.to(X.device)
        else:
            # Brute force
            # dist_calc returns (B, N_train)
            all_dists = self.metric(X, self._fit_X)

            # topk
            # If metric is similarity, we want largest. But standard is distance -> smallest.
            # We assume distance.
            k = min(n_neighbors, all_dists.size(-1))
            dists, inds = torch.topk(all_dists, k=k, dim=-1, largest=False)

        if return_distance:
            return dists, inds
        else:
            return inds

    def predict(self, X):
        dists, inds = self.kneighbors(X)

        # Gather targets
        y_train = self._fit_y

        # Ensure y_train is on correct device for gathering
        if y_train.device != inds.device:
            y_train = y_train.to(inds.device)

        # y_train shape (N_train, Out)
        # inds shape (B, K)
        # We need to index rows of y_train.
        # Check dimensions
        if y_train.ndim == 1:
            y_train = y_train.unsqueeze(-1)

        # F.embedding(inds, y_train) works if y_train is weight matrix (Rows, Cols)
        # y_neighbors = F.embedding(inds, y_train) 
        # But F.embedding expects Long tensor indices. inds is Long.
        y_neighbors = F.embedding(inds, y_train)  # (B, K, Out)

        epsilon = 1e-8

        if self.weights == 'uniform':
            preds = y_neighbors.mean(dim=1)  # (B, Out)

        elif self.weights == 'distance':
            weights = 1.0 / (dists + epsilon)  # (B, K)

            # Weighted average
            # weights: (B, K) -> (B, K, 1)
            weights = weights.unsqueeze(-1)

            num = (weights * y_neighbors).sum(dim=1)
            den = weights.sum(dim=1)
            preds = num / (den + epsilon)

        elif callable(self.weights):
            weights = self.weights(dists)  # (B, K)
            weights = weights.unsqueeze(-1)
            num = (weights * y_neighbors).sum(dim=1)
            den = weights.sum(dim=1)
            preds = num / (den + epsilon)
        else:
            raise ValueError(f"Unknown weights mode: {self.weights}")

        if self.out_features == 1:
            return preds.squeeze(-1)

        return preds


class RadiusNeighborsRegressor(KNeighboursRegression):
    def __init__(self,
                 radius: float = 10.0,
                 weights: Union[str, Callable] = "uniform",
                 algorithm: str = "auto",
                 leaf_size: int = 30,
                 p: float = 2,
                 metric: Union[str, Callable, object] = "minkowski",
                 metric_params: dict = None,
                 n_jobs: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs,
            device=device,
            dtype=dtype,
            *args, **kwargs
        )
        self.radius = radius

    def radius_neighbors(self, X=None, radius=None, return_distance=True, sort_results=False):
        if X is None:
            if self._fit_X is None:
                raise RuntimeError("Model likely not fitted.")
            X = self._fit_X

        if self.device:
            X = X.to(self.device)

        if radius is None:
            radius = self.radius

        if self._fit_X is None:
            raise RuntimeError("Model handles not fitted yet. Call 'fit' first.")

        if self._algorithm in ['ball_tree', 'kd_tree']:
            # Tree-based search
            if self._algorithm == 'ball_tree' and not hasattr(self._tree, 'query_radius'):
                 # Fallback if BallTree doesn't implement query_radius yet
                 warnings.warn("BallTree does not implement query_radius. Falling back to brute force.")
                 return self._brute_radius_neighbors(X, radius, return_distance, sort_results)
            return self._tree.query_radius(X, r=radius, return_distance=return_distance, sort_results=sort_results)
        else:
            # Brute force
            return self._brute_radius_neighbors(X, radius, return_distance, sort_results)

    def _brute_radius_neighbors(self, X, radius, return_distance=True, sort_results=False):
        # Compute full distance matrix
        # dist_calc returns (B, N_train)
        all_dists = self.metric(X, self._fit_X)
        
        # Filter by radius
        mask = all_dists <= radius
        
        results_dists = []
        results_inds = []
        
        B = X.shape[0]
        for i in range(B):
            row_mask = mask[i]
            d = all_dists[i][row_mask]
            ind = torch.nonzero(row_mask, as_tuple=True)[0]
            
            if sort_results:
                d, perm = torch.sort(d)
                ind = ind[perm]
                
            results_dists.append(d)
            results_inds.append(ind)
            
        if return_distance:
            return results_dists, results_inds
        else:
            return results_inds

    def predict(self, X):
        # Get neighbors within radius
        # results are lists of tensors
        dists_list, inds_list = self.radius_neighbors(X, return_distance=True)

        y_train = self._fit_y
        if y_train.device != X.device:
             y_train = y_train.to(X.device)
             
        if y_train.ndim == 1:
            y_train = y_train.unsqueeze(-1)

        B = len(dists_list)
        Out = y_train.shape[-1]
        
        preds = torch.zeros((B, Out), device=X.device, dtype=y_train.dtype)
        
        epsilon = 1e-8

        for i in range(B):
            dists = dists_list[i]  # (K_i,)
            inds = inds_list[i]    # (K_i,)
            
            if inds.numel() == 0:
                # No neighbors found. 
                # Scikit-learn raises an exception or returns outliers.
                # Here we will raise an exception to be safe, or could return 0/NaN.
                # Let's raise an exception generally or handle gracefully? 
                # User asked to implement based on docstring args... 
                # Usually standard behavior is raise ValueError if no neighbors and no outlier_label
                # For now let's just warn and leave as 0? Or maybe raise exception.
                raise ValueError(f"No neighbors found for query point {i} within radius {self.radius}.")
            
            # y_neighbors for this point
            y_neighbors = y_train[inds] # (K_i, Out)
            
            if self.weights == 'uniform':
                preds[i] = y_neighbors.mean(dim=0)
            
            elif self.weights == 'distance':
                # invalid weights (dist=0) -> 1/0 = inf. 
                # If distance is 0, weight should be high. 
                # Handle zero distance: if 0, that point gets all weight.
                
                zero_dist_mask = dists == 0
                if zero_dist_mask.any():
                    # Point coincides with training point.
                    weights = zero_dist_mask.to(dists.dtype)
                else:
                    weights = 1.0 / (dists + epsilon)
                
                # Expand weights for broadcasting if Out > 1
                w_unsqueezed = weights.unsqueeze(-1) # (K_i, 1)
                
                num = (w_unsqueezed * y_neighbors).sum(dim=0)
                den = weights.sum(dim=0)
                preds[i] = num / (den + epsilon)
                
            elif callable(self.weights):
                weights = self.weights(dists)
                w_unsqueezed = weights.unsqueeze(-1)
                num = (w_unsqueezed * y_neighbors).sum(dim=0)
                den = weights.sum(dim=0)
                preds[i] = num / (den + epsilon)
            else:
                 raise ValueError(f"Unknown weights mode: {self.weights}")

        if self.out_features == 1:
            return preds.squeeze(-1)

        return preds


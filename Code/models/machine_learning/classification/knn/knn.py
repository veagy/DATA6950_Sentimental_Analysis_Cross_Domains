import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, List, Callable
import math
import warnings
from .....models.utils import MLClassifier
from .....models.machine_learning.regression.knn.knn import BallTree, KDTree
from torch.func import vmap
import joblib

__all__ = ["KNeighborsClassifier", "RadiusNeighborsClassifier", "NearestCentroid"]


class KNeighborsClassifier(MLClassifier):
    def __init__(self,
                 n_neighbors: int = 5,
                 weights: Union[str, Callable] = "uniform",
                 algorithm: str = "auto",
                 leaf_size: int = 30,
                 p: float = 2,
                 metric: Union[str, Callable, object] = "minkowski",
                 metric_params: dict = None,
                 class_weights: Union[str, dict] = None,
                 n_jobs: int = None,
                 warm_start: bool = False,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.warm_start = warm_start
        self.classes_ = None
        self.n_classes_ = None
        if metric_params is None:
            metric_params = {}
        metric_params["p"] = p
        from ....utils.utils import MLCluster
        self.metric = MLCluster()._create_metric(metric, metric_params).metric
        self.n_features_in_ = None
        self.n_samples_fit_ = None
        self.outputs_2d_ = None
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype
        self.class_weights = class_weights
        self._fit_X = None
        self._fit_y = None
        self._tree = None
        self._algorithm = None

    def _init_module_(self, X: torch.Tensor, y: torch.Tensor):
        self.n_samples_fit_ = X.shape[0]
        self.n_features_in_ = X.shape[1] if X.ndim > 1 else 1

        if y.ndim == 1:
            self.outputs_2d_ = False
        else:
            self.outputs_2d_ = True

        self.classes_, y_indices = torch.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)

        if self.class_weights is not None:
            if isinstance(self.class_weights, str) and self.class_weights == "balanced":
                # Compute balanced weights: n_samples / (n_classes * torch.bincount(y))
                counts = torch.bincount(y_indices)
                self._computed_class_weights = self.n_samples_fit_ / (self.n_classes_ * counts.to(self.dtype))
            elif isinstance(self.class_weights, dict):
                # Map dict to tensor indexed by class index
                weight_tensor = torch.ones(self.n_classes_, device=self.device, dtype=self.dtype)
                for i, cls in enumerate(self.classes_):
                    cls_val = cls.item()
                    if cls_val in self.class_weights:
                        weight_tensor[i] = self.class_weights[cls_val]
                self._computed_class_weights = weight_tensor
            else:
                self._computed_class_weights = torch.ones(self.n_classes_, device=self.device, dtype=self.dtype)
        else:
            self._computed_class_weights = torch.ones(self.n_classes_, device=self.device, dtype=self.dtype)

        return self

    def fit(self, data_or_X, y=None, **kwargs):
        warm_start = kwargs.get('warm_start', getattr(self, 'warm_start', False))
        X = torch.as_tensor(data_or_X, device=self.device if self.device else None, dtype=self.dtype)
        y = torch.as_tensor(y, device=self.device if self.device else None, dtype=torch.long)

        if warm_start and self._fit_X is not None and self._fit_y is not None:
            prev_X = self._fit_X.to(X.device).to(X.dtype)
            prev_y = self._fit_y.to(y.device).to(y.dtype)
            X = torch.cat([prev_X, X], dim=0)
            y = torch.cat([prev_y, y], dim=0)

        if self.n_features_in_ is None:
            self._init_module_(X, y)

        if self.device and self._computed_class_weights is not None:
            self._computed_class_weights = self._computed_class_weights.to(self.device)

        self._fit_X = X
        self._fit_y = y

        self._algorithm = self.algorithm
        if self._algorithm == 'auto':
            if not X.is_cuda:
                self._algorithm = 'kd_tree'
            else:
                self._algorithm = 'brute'

        if self._algorithm in ['ball_tree', 'kd_tree']:
            p = 2
            try:
                if self._algorithm == 'ball_tree':
                    self._tree = BallTree(X, leaf_size=self.leaf_size, metric=self.metric, p=p,
                                          device=self.device, dtype=self.dtype, n_jobs=self.n_jobs)
                else:
                    self._tree = KDTree(X, leaf_size=self.leaf_size, metric=self.metric, p=p,
                                        device=self.device, dtype=self.dtype, n_jobs=self.n_jobs)
            except Exception as e:
                warnings.warn(f"Failed to build tree: {e}. Falling back to brute force.")
                self._algorithm = 'brute'

        return self

    def kneighbors(self, X: torch.Tensor = None, n_neighbors: int = None, return_distance: bool = True):
        if X is None:
            X = self._fit_X
        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        X = X.to(self.device).to(self.dtype)
        # Ensure stored training data is on the same device (guards against load drift)
        if self._fit_X is not None and self._fit_X.device != X.device:
            self._fit_X = self._fit_X.to(X.device)
        if self._fit_y is not None and self._fit_y.device != X.device:
            self._fit_y = self._fit_y.to(X.device)

        if self._algorithm in ['ball_tree', 'kd_tree'] and self._tree is not None:
            dists, inds = self._tree.query(X, k=n_neighbors, return_distance=True)
        else:
            all_dists = self.metric(X, self._fit_X)
            k = min(n_neighbors, all_dists.size(-1))
            dists, inds = torch.topk(all_dists, k=k, dim=-1, largest=False)

        if return_distance:
            return dists, inds
        return inds

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        dists, inds = self.kneighbors(X)
        B = X.shape[0]

        dev = X.device if X.is_cuda else torch.device(self.device)
        if self._fit_y is not None and self._fit_y.device != dev:
            self._fit_y = self._fit_y.to(dev)
        y_neighbors = self._fit_y[inds]

        # Map y_neighbors to class indices
        if self.classes_ is not None and self.classes_.device != dev:
            self.classes_ = self.classes_.to(dev)
        sorted_classes = self.classes_
        y_neighbor_indices = torch.searchsorted(sorted_classes, y_neighbors)

        # Compute weights
        if self.weights == 'uniform':
            weights = torch.ones_like(dists)
        elif self.weights == 'distance':
            weights = 1.0 / (dists + 1e-8)
        elif callable(self.weights):
            weights = self.weights(dists)
        else:
            raise ValueError(f"Unknown weights mode: {self.weights}")

        # Incorporate class weights
        neighbor_class_weights = self._computed_class_weights[y_neighbor_indices]
        weights = weights * neighbor_class_weights

        # Aggregate weights per class
        probs = torch.zeros((B, self.n_classes_), device=self.device, dtype=self.dtype)

        for i in range(self.n_classes_):
            mask = (y_neighbor_indices == i)
            probs[:, i] = (weights * mask.to(self.dtype)).sum(dim=1)

        # Normalize
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)
        return probs

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        probs = self.predict_proba(X)
        indices = torch.argmax(probs, dim=1)
        return self.classes_[indices]

    def decision_function(self, X: torch.Tensor) -> torch.Tensor:
        """
        Decision function for KNeighborsClassifier.
        Returns the probabilities as scores.
        """
        return self.predict_proba(X)

    def predict_log_proba(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute log probabilities of possible outcomes for samples in X.
        """
        return torch.log(torch.clamp(self.predict_proba(X), min=1e-15))

    def forward(self, X: torch.Tensor, y: torch.Tensor = None, **kwargs) -> torch.Tensor:
        if not self.fit_status:
            self.fit(X, y)
        return self.predict(X)


class RadiusNeighborsClassifier(KNeighborsClassifier):
    def __init__(self,
                 radius: float = 1.0,
                 weights: Union[str, Callable] = "uniform",
                 algorithm: str = "auto",
                 leaf_size: int = 30,
                 p: float = 2,
                 metric: Union[str, Callable, object] = "minkowski",
                 metric_params: dict = None,
                 class_weight: Union[str, dict] = None,
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
            class_weights=class_weight,
            n_jobs=n_jobs,
            device=device,
            dtype=dtype,
            *args, **kwargs
        )
        self.radius = radius

    def radius_neighbors(self, X: torch.Tensor = None, radius: float = None, return_distance: bool = True,
                         sort_results: bool = False):
        if X is None:
            if self._fit_X is None:
                raise RuntimeError("Model handles not fitted yet. Call 'fit' first.")
            X = self._fit_X
        if radius is None:
            radius = self.radius

        X = X.to(self.device).to(self.dtype)
        # Ensure stored data is on the same device
        if self._fit_X is not None and self._fit_X.device != X.device:
            self._fit_X = self._fit_X.to(X.device)
        if self._fit_y is not None and self._fit_y.device != X.device:
            self._fit_y = self._fit_y.to(X.device)
        if self.classes_ is not None and self.classes_.device != X.device:
            self.classes_ = self.classes_.to(X.device)

        if self._algorithm in ['ball_tree', 'kd_tree'] and self._tree is not None:
            if self._algorithm == 'ball_tree' and not hasattr(self._tree, 'query_radius'):
                warnings.warn("BallTree does not implement query_radius. Falling back to brute force.")
                return self._brute_radius_neighbors(X, radius, return_distance, sort_results)
            return self._tree.query_radius(X, r=radius, return_distance=return_distance, sort_results=sort_results)
        else:
            return self._brute_radius_neighbors(X, radius, return_distance, sort_results)

    def _brute_radius_neighbors(self, X, radius, return_distance=True, sort_results=False):
        all_dists = self.metric(X, self._fit_X)
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
        return results_inds

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        dists_list, inds_list = self.radius_neighbors(X, return_distance=True)
        B = len(dists_list)
        probs = torch.zeros((B, self.n_classes_), device=self.device, dtype=self.dtype)

        for i in range(B):
            dists = dists_list[i]
            inds = inds_list[i]

            if inds.numel() == 0:
                raise ValueError(f"No neighbors found for query point {i} within radius {self.radius}.")

            y_neighbors = self._fit_y[inds]
            y_neighbor_indices = torch.searchsorted(self.classes_, y_neighbors)

            if self.weights == 'uniform':
                weights = torch.ones_like(dists)
            elif self.weights == 'distance':
                weights = 1.0 / (dists + 1e-8)
            elif callable(self.weights):
                weights = self.weights(dists)
            else:
                raise ValueError(f"Unknown weights mode: {self.weights}")

            neighbor_class_weights = self._computed_class_weights[y_neighbor_indices]
            weights = weights * neighbor_class_weights

            for c in range(self.n_classes_):
                mask = (y_neighbor_indices == c)
                probs[i, c] = (weights * (y_neighbor_indices == c).to(self.dtype)).sum()

        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)
        return probs

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        probas = self.predict_proba(X)
        indices = torch.argmax(probas, dim=1)
        return self.classes_[indices]

    def decision_function(self, X: torch.Tensor) -> torch.Tensor:
        """
        Decision function for RadiusNeighborsClassifier.
        Returns the probabilities as scores.
        """
        return self.predict_proba(X)

    def predict_log_proba(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute log probabilities of possible outcomes for samples in X.
        """
        return torch.log(torch.clamp(self.predict_proba(X), min=1e-15))

    def forward(self, X: torch.Tensor, y: torch.Tensor = None):
        if not self.fit_status:
            self.fit(X, y)
        return self.predict(X)


class NearestCentroid(MLClassifier):
    def __init__(self,
                 p: float = 2,
                 metric: Union[str, Callable, object] = "minkowski",
                 shrink_threshold: float = None,
                 priors: Union[str, List, Tuple, torch.Tensor] = "uniform",
                 metric_params: dict = None,
                 class_weights: Union[str, dict] = None,
                 n_jobs: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        if metric_params is None:
            metric_params = {}
        metric_params["p"] = p
        from ....utils.utils import MLCluster
        self.metric = MLCluster()._create_metric(metric, metric_params).metric
        self._metric_str = metric if isinstance(metric, str) else ''
        self.class_weights = class_weights
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype
        self.shrink_threshold = shrink_threshold
        self.priors = priors
        self.centroids_ = []
        self.deviations_ = []
        self.within_class_std_dev_ = []
        self.class_prior_ = []
        self.classes_ = None
        self.n_classes_ = None
        self.n_features_in_ = None
        self.fit_status = False

    def _init_module_(self, X: torch.Tensor, y: torch.Tensor):
        self.n_samples_fit_ = X.shape[0]
        self.n_features_in_ = X.shape[1] if X.ndim > 1 else 1

        self.classes_, y_indices = torch.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)

        if self.priors == "uniform":
            self.class_prior_ = torch.full((self.n_classes_,), 1.0 / self.n_classes_, device=self.device,
                                           dtype=self.dtype)
        elif self.priors == "empirical":
            counts = torch.bincount(y_indices, minlength=self.n_classes_)
            self.class_prior_ = counts.to(self.dtype) / self.n_samples_fit_
        elif isinstance(self.priors, (list, tuple, torch.Tensor)):
            self.class_prior_ = torch.as_tensor(self.priors, device=self.device, dtype=self.dtype)
            if self.class_prior_.shape[0] != self.n_classes_:
                raise ValueError("Number of priors must match number of classes.")
        else:
            raise ValueError(f"Unknown priors mode: {self.priors}")

        return self

    def fit(self, data_or_X, y=None, **kwargs):
        X = data_or_X

        if self.n_features_in_ is None:
            self._init_module_(X, y)

        if self.device:
            X = X.to(self.device).to(self.dtype)
            y = y.to(self.device).to(torch.long)
            self.class_prior_ = self.class_prior_.to(self.device)

        _, y_indices = torch.unique(y, return_inverse=True)

        centroids = torch.zeros((self.n_classes_, self.n_features_in_), device=self.device, dtype=self.dtype)

        is_manhattan = False
        if isinstance(self._metric_str, str) and self._metric_str.lower() in ["manhattan", "l1"]:
            is_manhattan = True

        for i in range(self.n_classes_):
            mask = (y_indices == i)
            if not mask.any():
                warnings.warn(f"Class {self.classes_[i]} has no samples. Centroid will be zero.")
                continue

            X_class = X[mask]
            if is_manhattan:
                centroids[i] = torch.median(X_class, dim=0).values
            else:
                centroids[i] = X_class.mean(dim=0)

        if self.shrink_threshold is not None:
            global_centroid = X.mean(dim=0)

            variance = torch.zeros(self.n_features_in_, device=self.device, dtype=self.dtype)
            for i in range(self.n_classes_):
                mask = (y_indices == i)
                if mask.any():
                    variance += torch.sum((X[mask] - centroids[i]) ** 2, dim=0)

            within_class_std = torch.sqrt(variance / (self.n_samples_fit_ - self.n_classes_))
            self.within_class_std_dev_ = within_class_std

            s0 = torch.median(within_class_std)

            counts = torch.bincount(y_indices, minlength=self.n_classes_).to(self.dtype)
            m_k = torch.sqrt(1.0 / counts + 1.0 / self.n_samples_fit_)

            denom = m_k.unsqueeze(1) * (within_class_std.unsqueeze(0) + s0)
            deviations = (centroids - global_centroid.unsqueeze(0)) / (denom + 1e-8)

            shrunken_deviations = torch.sign(deviations) * torch.clamp(torch.abs(deviations) - self.shrink_threshold,
                                                                       min=0.0)
            self.deviations_ = shrunken_deviations

            centroids = global_centroid.unsqueeze(0) + denom * shrunken_deviations

        self.centroids_ = centroids
        self.fit_status = True
        return self

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        if not self.fit_status:
            raise RuntimeError("Model handles not fitted yet. Call 'fit' first.")

        X = X.to(self.device).to(self.dtype)
        dev = X.device
        # Ensure stored tensors are on the same device (guards against load drift)
        if isinstance(self.centroids_, torch.Tensor) and self.centroids_.device != dev:
            self.centroids_ = self.centroids_.to(dev)
        if isinstance(self.classes_, torch.Tensor) and self.classes_.device != dev:
            self.classes_ = self.classes_.to(dev)
        if isinstance(self.class_prior_, torch.Tensor) and self.class_prior_.device != dev:
            self.class_prior_ = self.class_prior_.to(dev)
        if isinstance(self.within_class_std_dev_, torch.Tensor) and self.within_class_std_dev_.device != dev:
            self.within_class_std_dev_ = self.within_class_std_dev_.to(dev)
        dists = self.metric(X, self.centroids_)

        if self.shrink_threshold is not None:
            s0 = torch.median(self.within_class_std_dev_)
            weights = 1.0 / ((self.within_class_std_dev_ + s0) ** 2 + 1e-8)

            B = X.shape[0]
            scores = torch.zeros((B, self.n_classes_), device=self.device, dtype=self.dtype)
            for i in range(self.n_classes_):
                diff_sq = (X - self.centroids_[i]) ** 2
                scores[:, i] = torch.sum(diff_sq * weights, dim=1)

            scores += -2.0 * torch.log(self.class_prior_ + 1e-8).unsqueeze(0)

            probs = F.softmax(-0.5 * scores, dim=1)
        else:
            probs = F.softmax(-dists, dim=1)

        return probs

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        probs = self.predict_proba(X)
        indices = torch.argmax(probs, dim=1)
        return self.classes_[indices]

    def decision_function(self, X: torch.Tensor) -> torch.Tensor:
        """
        Decision function for NearestCentroid.
        Returns the probabilities as scores.
        """
        return self.predict_proba(X)

    def predict_log_proba(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute log probabilities of possible outcomes for samples in X.
        """
        return torch.log(torch.clamp(self.predict_proba(X), min=1e-15))

    def forward(self, X: torch.Tensor, y: torch.Tensor = None, **kwargs) -> torch.Tensor:
        if not self.fit_status:
            self.fit(X, y, **kwargs)
        return self.predict(X)


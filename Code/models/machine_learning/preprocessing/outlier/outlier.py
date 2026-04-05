import warnings
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Union, Any, List, Tuple, Dict, Literal
import pandas as pd  # pyright: ignore[reportMissingImports]
import numpy as np

from .....models.utils import MLModule, MLRegressor, MLClassifier, MLCluster

from ...regression.knn.knn import BallTree, KDTree
from ...regression.svm.kernels import (
    get_kernel_class,
    LinearKernel,
    PolyKernel,
    RBFKernel,
    SigmoidKernel,
    LaplacianKernel,
)
from torch.func import vmap
import joblib

try:
    import scipy.optimize as opt
except ImportError:
    opt = None


__all__ = [
    "IsolationForest",
    "LocalOutlierFactor",
    "OneClassSVM",
    "SGDOneClassSVM",
    "BGDOneClassSVM",
    "MGDOneClassSVM",
    "AdaGradOneClassSVM",
    "RMSPropOneClassSVM",
    "AdamOneClassSVM",
    "AdadeltaOneClassSVM",
    "AdafactorOneClassSVM",
    "AdamWOneClassSVM",
    "AdamaxOneClassSVM",
    "LBFGSOneClassSVM",
    "MuonOneClassSVM",
    "NAdamOneClassSVM",
    "RAdamOneClassSVM",
    "RpropOneClassSVM",
    "BFGSOneClassSVM",
    "NewtonCGOneClassSVM",
    "SLSQPOneClassSVM",
    "LionOneClassSVM",
    "ShampooOneClassSVM",
    "SophiaOneClassSVM",
    "AdanOneClassSVM",
    "COBYLAOneClassSVM",
    "NelderMeadOneClassSVM",
    "LAMBOneClassSVM",
    "LookaheadOneClassSVM",
    "AdEMAMixOneClassSVM",
    "ScheduleFreeOneClassSVM",
    "MARSOneClassSVM",
    "PowellOneClassSVM",
    "TNCOneClassSVM",
    "TrustNCGOneClassSVM",
    "DoglegOneClassSVM",
    "CGOneClassSVM",
    "TRPOOneClassSVM",
    "MadgradOneClassSVM",
    "YogiOneClassSVM",
    "LARSOneClassSVM",
    "DAdaptationOneClassSVM",
    "SignSGDOneClassSVM",
    "ProdigyOneClassSVM",
    "QNSVRGOneClassSVM",
    "DifferentialEvolutionOneClassSVM",
    "BasinhoppingOneClassSVM",
    "DualAnnealingOneClassSVM",
    "SHGOOneClassSVM",
    "CMAESOneClassSVM",
    "BayesianOptimizationOneClassSVM",
    "PSOOneClassSVM",
    "FireflyOneClassSVM",
    "PassiveAggressiveOneClassSVM",
    "SGDEstimatorBasedOutlierDetection",
    "BGDEstimatorBasedOutlierDetection",
    "MGDEstimatorBasedOutlierDetection",
    "AdaGradEstimatorBasedOutlierDetection",
    "RMSPropEstimatorBasedOutlierDetection",
    "AdamEstimatorBasedOutlierDetection",
    "AdadeltaEstimatorBasedOutlierDetection",
    "AdafactorEstimatorBasedOutlierDetection",
    "AdamWEstimatorBasedOutlierDetection",
    "AdamaxEstimatorBasedOutlierDetection",
    "LBFGSEstimatorBasedOutlierDetection",
    "MuonEstimatorBasedOutlierDetection",
    "NAdamEstimatorBasedOutlierDetection",
    "RAdamEstimatorBasedOutlierDetection",
    "RpropEstimatorBasedOutlierDetection",
    "BFGSEstimatorBasedOutlierDetection",
    "NewtonCGEstimatorBasedOutlierDetection",
    "SLSQPEstimatorBasedOutlierDetection",
    "LionEstimatorBasedOutlierDetection",
    "ShampooEstimatorBasedOutlierDetection",
    "SophiaEstimatorBasedOutlierDetection",
    "AdanEstimatorBasedOutlierDetection",
    "COBYLAEstimatorBasedOutlierDetection",
    "NelderMeadEstimatorBasedOutlierDetection",
    "LAMBEstimatorBasedOutlierDetection",
    "LookaheadEstimatorBasedOutlierDetection",
    "AdEMAMixEstimatorBasedOutlierDetection",
    "ScheduleFreeEstimatorBasedOutlierDetection",
    "MARSEstimatorBasedOutlierDetection",
    "PowellEstimatorBasedOutlierDetection",
    "TNCEstimatorBasedOutlierDetection",
    "TrustNCGEstimatorBasedOutlierDetection",
    "DoglegEstimatorBasedOutlierDetection",
    "CGEstimatorBasedOutlierDetection",
    "TRPOEstimatorBasedOutlierDetection",
    "MadgradEstimatorBasedOutlierDetection",
    "YogiEstimatorBasedOutlierDetection",
    "LARSEstimatorBasedOutlierDetection",
    "DAdaptationEstimatorBasedOutlierDetection",
    "SignSGDEstimatorBasedOutlierDetection",
    "ProdigyEstimatorBasedOutlierDetection",
    "QNSVRGEstimatorBasedOutlierDetection",
    "DifferentialEvolutionEstimatorBasedOutlierDetection",
    "BasinhoppingEstimatorBasedOutlierDetection",
    "DualAnnealingEstimatorBasedOutlierDetection",
    "SHGOEstimatorBasedOutlierDetection",
    "CMAESEstimatorBasedOutlierDetection",
    "BayesianOptimizationEstimatorBasedOutlierDetection",
    "PSOEstimatorBasedOutlierDetection",
    "FireflyEstimatorBasedOutlierDetection",
    "PassiveAggressiveEstimatorBasedOutlierDetection",
]


# ============================================================
# Utility functions
# ============================================================

def _average_path_length(n: int) -> float:
    """Expected average path length for an isolation tree with n samples.

    This is the expected number of splits required to isolate a point drawn
    from a dataset of n points.  The formula is the same harmonic-number
    approximation used by the original Isolation Forest paper
    (Liu et al., 2008):

        c(n) = 2 * H(n-1) - 2*(n-1)/n      if n > 2
             = 1                             if n = 2
             = 0                             if n <= 1

    where H(i) = ln(i) + 0.5772156649... (Euler–Mascheroni constant).
    """
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (math.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n


def _get_lr(learning_rate: str, eta0: float, power_t: float, t: float,
            nu: float = 0.5) -> float:
    """Compute the per-step learning rate according to the chosen schedule.

    Schedules
    ---------
    ``invscaling``  : eta = eta0 / t^power_t
    ``optimal``     : eta = 1 / (nu * (t + t0))  where t0 is a heuristic
    ``constant``    : eta = eta0   (fixed throughout training)
    ``adaptive``    : eta = eta0   (same as constant; callers may halve it
                                    when no improvement is detected)
    """
    if learning_rate == "invscaling":
        return eta0 / (t ** power_t)
    elif learning_rate == "optimal":
        t0 = max(1.0 / (nu * eta0 + 1e-12), 1.0)
        return 1.0 / (nu * (t + t0) + 1e-12)
    elif learning_rate in ("constant", "adaptive"):
        return eta0
    return eta0


# ============================================================
# IsolationForest
# ============================================================

class IsolationForest(MLModule):
    def __init__(self,
                 n_estimators: int = 100,
                 max_samples: Union[Literal["auto"], int, float] = 'auto',
                 contamination: Union[Literal["auto"], float] = 'auto',
                 max_features: Union[int, float] = 1.0,
                 bootstrap: bool = False,
                 n_jobs: int = None,
                 random_state: Union[int, torch.Generator] = None,
                 verbose: int = 0,
                 warm_start: bool = False,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.device = device
        self.dtype = dtype

        self.estimators_: List[Tuple[dict, torch.Tensor]] = []
        self.estimators_features_: List[torch.Tensor] = []
        self.estimators_samples_: List[torch.Tensor] = []
        self.max_samples_: Optional[int] = None
        self.offset_: Optional[float] = None
        self.n_features_in_: Optional[int] = None
        self.estimator_: Optional[Tuple] = None

    def _get_generator(self) -> Optional[torch.Generator]:
        """Return a seeded Generator, or None for global RNG."""
        if self.random_state is None:
            return None
        if isinstance(self.random_state, torch.Generator):
            return self.random_state
        g = torch.Generator(device=self.device)
        g.manual_seed(int(self.random_state))
        return g

    def _build_node(self, X: torch.Tensor, depth: int, max_depth: int,
                    gen: Optional[torch.Generator]) -> dict:
        """Recursively build an isolation tree node.

        At each internal node a feature and a split value are chosen
        uniformly at random from the range observed in the local subsample.
        Recursion terminates when the node contains ≤1 sample or the maximum
        depth is reached, at which point a leaf stores the subsample size so
        that the expected path-length correction can be applied.
        """
        n = X.shape[0]
        if depth >= max_depth or n <= 1:
            return {"type": "leaf", "size": n}

        n_features = X.shape[1]
        # Choose a random feature
        feat = (torch.randint(0, n_features, (1,), generator=gen)
                if gen is not None
                else torch.randint(0, n_features, (1,))).item()

        col = X[:, feat]
        col_min, col_max = col.min().item(), col.max().item()
        if col_min == col_max:
            # All values equal; cannot split — treat as leaf
            return {"type": "leaf", "size": n}

        # Random split threshold in [col_min, col_max)
        threshold = (col_min + (col_max - col_min) *
                     (torch.rand(1, generator=gen, device=self.device) if gen is not None
                      else torch.rand(1, device=self.device)).item())

        left_mask = col <= threshold
        right_mask = ~left_mask
        return {
            "type": "split",
            "feature": feat,
            "threshold": threshold,
            "left": self._build_node(X[left_mask], depth + 1, max_depth, gen),
            "right": self._build_node(X[right_mask], depth + 1, max_depth, gen),
            "size": n,
        }

    def _path_lengths(self, X: torch.Tensor, node: dict, depth: int) -> torch.Tensor:
        """Compute the path length to isolate each sample in X through *node*.

        At a leaf the depth is augmented by the expected path length
        correction c(node_size) to account for unsplit samples that would
        need further splits in an ideal tree.
        """
        n = X.shape[0]
        if node["type"] == "leaf" or n == 0:
            correction = _average_path_length(node["size"])
            return torch.full((n,), depth + correction,
                              dtype=self.dtype, device=self.device)

        feat = node["feature"]
        thresh = node["threshold"]
        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask

        lengths = torch.zeros(n, dtype=self.dtype, device=self.device)
        if left_mask.any():
            lengths[left_mask] = self._path_lengths(
                X[left_mask], node["left"], depth + 1)
        if right_mask.any():
            lengths[right_mask] = self._path_lengths(
                X[right_mask], node["right"], depth + 1)
        return lengths

    def _raw_score(self, X: torch.Tensor) -> torch.Tensor:
        """Compute the Isolation Forest anomaly score for each sample.

        The score equals  -2^{ -mean_path_length / c(max_samples_) }
        and lies in (-1, 0].  Values close to -0.5 indicate roughly
        average isolation depth; values near 0 indicate anomalies that
        are isolated very quickly.
        """
        c_n = _average_path_length(self.max_samples_)
        if c_n == 0.0:
            c_n = 1.0

        n = X.shape[0]
        total = torch.zeros(n, dtype=self.dtype, device=self.device)
        for tree, feat_idx in self.estimators_:
            X_sub = X[:, feat_idx]
            total += self._path_lengths(X_sub, tree, 0)

        avg = total / len(self.estimators_)
        two = torch.tensor(2.0, dtype=self.dtype, device=self.device)
        return -torch.pow(two, -avg / c_n)

    def fit(self, data_or_X, y=None, **kwargs):
        """Build the isolation forest from the training set *data_or_X*.

        Each of ``n_estimators`` trees is fitted on a random subsample of
        size ``max_samples_`` and a random subset of features of size
        determined by ``max_features``.  After fitting, the decision
        threshold ``offset_`` is set based on ``contamination``.
        """
        warm_start = kwargs.get('warm_start', self.warm_start)
        X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Resolve max_samples
        if self.max_samples == "auto":
            max_s = min(256, n_samples)
        elif isinstance(self.max_samples, float):
            max_s = max(1, int(self.max_samples * n_samples))
        else:
            max_s = int(self.max_samples)
        max_s = min(max_s, n_samples)
        self.max_samples_ = max_s

        # Tree depth: ceil(log2(max_samples)) to fully isolate any single point
        max_depth = math.ceil(math.log2(max(max_s, 2)))

        gen = self._get_generator()

        # Resolve max_features
        if isinstance(self.max_features, int):
            n_feat = min(self.max_features, n_features)
        else:
            n_feat = max(1, int(self.max_features * n_features))

        # Warm-start: resume from existing estimators
        n_start = len(self.estimators_) if warm_start else 0
        if not warm_start:
            self.estimators_ = []
            self.estimators_features_ = []
            self.estimators_samples_ = []

        for i in range(n_start, self.n_estimators):
            # Draw feature subset
            feat_idx = (torch.randperm(n_features, generator=gen, device=self.device)[:n_feat]
                        if gen is not None
                        else torch.randperm(n_features, device=self.device)[:n_feat])

            # Draw sample subset (with or without replacement)
            if self.bootstrap:
                sample_idx = (torch.randint(0, n_samples, (max_s,), generator=gen, device=self.device)
                              if gen is not None
                              else torch.randint(0, n_samples, (max_s,), device=self.device))
            else:
                sample_idx = (torch.randperm(n_samples, generator=gen, device=self.device)[:max_s]
                              if gen is not None
                              else torch.randperm(n_samples, device=self.device)[:max_s])

            X_sub = X[sample_idx][:, feat_idx]
            tree = self._build_node(X_sub, 0, max_depth, gen)

            self.estimators_.append((tree, feat_idx))
            self.estimators_features_.append(feat_idx)
            self.estimators_samples_.append(sample_idx)

            if self.verbose > 1:
                print(f"IsolationForest: built tree {i + 1}/{self.n_estimators}")

        self.estimator_ = self.estimators_[0] if self.estimators_ else None

        # Set decision threshold
        with torch.no_grad():
            scores = self._raw_score(X)
        if self.contamination == "auto":
            # Default paper threshold at -0.5
            self.offset_ = -0.5
        else:
            # Threshold at the (1-contamination) quantile so that the
            # contamination fraction of training points get label -1
            self.offset_ = float(
                torch.quantile(scores, 1.0 - float(self.contamination)).item())

        if self.verbose:
            print(f"IsolationForest: fitted {len(self.estimators_)} trees, "
                  f"offset_={self.offset_:.4f}")
        return self

    def score_samples(self, X) -> torch.Tensor:
        """Return the raw anomaly scores (lower = more anomalous).

        Scores are in (-1, 0].  A value near -0.5 means roughly average
        isolation; near 0 means the point is isolated very quickly (outlier).
        """
        if not self.estimators_:
            raise RuntimeError("IsolationForest is not fitted yet. Call fit() first.")
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        with torch.no_grad():
            return self._raw_score(X)

    def decision_function(self, X) -> torch.Tensor:
        """Shifted anomaly score: score_samples(X) - offset_.

        Positive values indicate inliers; negative values indicate outliers.
        """
        return self.score_samples(X) - self.offset_

    def predict(self, X) -> torch.Tensor:
        """Classify samples as inliers (+1) or outliers (-1)."""
        df = self.decision_function(X)
        return torch.where(df >= 0,
                           torch.ones_like(df, dtype=torch.long),
                           -torch.ones_like(df, dtype=torch.long))

    def forward(self, X):
        return self.decision_function(X)


# ============================================================
# LocalOutlierFactor
# ============================================================

class LocalOutlierFactor(MLCluster):
    def __init__(self,
                 n_neighbors: int = 20,
                 algorithm: Union[Literal["auto", "ball_tree", "kd_tree",
                                          "brute"], Callable, nn.Module] = 'auto',
                 leaf_size: int = 30,
                 metric: Union[str, Callable, nn.Module] = 'minkowski',
                 p: float = 2,
                 metric_params: dict = None,
                 contamination: Union[Literal["auto"], float] = 'auto',
                 novelty: bool = False,
                 n_jobs: int = None,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self._metric_spec = metric          # original user-supplied metric spec
        self.metric = metric                # will be overwritten by _create_metric
        self.p = p
        self.metric_params = metric_params or {}
        self.contamination = contamination
        self.novelty = novelty
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype

        self.negative_outlier_factor_: Optional[torch.Tensor] = None
        self.n_neighbors_: Optional[int] = None
        self.offset_: Optional[float] = None
        self.effective_metric_: Optional[str] = None
        self.effective_metric_params_: Optional[dict] = None
        self.n_features_in_: Optional[int] = None
        self.n_samples_fit_: Optional[int] = None

        # Internal state
        self._X_fit: Optional[torch.Tensor] = None
        self._k_distances_fit: Optional[torch.Tensor] = None
        self._lrd_fit: Optional[torch.Tensor] = None
        self._knn_idx_fit: Optional[torch.Tensor] = None
        self._tree = None           # BallTree or KDTree instance

    # ------------------------------------------------------------------
    # Distance helpers
    # ------------------------------------------------------------------

    def _resolve_algorithm(self, n_samples: int, n_features: int) -> str:
        """Choose between 'ball_tree', 'kd_tree', and 'brute' automatically."""
        alg = self.algorithm
        if callable(alg) or isinstance(alg, nn.Module):
            return 'callable'
        alg = str(alg).lower()
        if alg in ('ball_tree', 'kd_tree', 'brute', 'callable'):
            return alg
        # 'auto': prefer BallTree for high-dimensional data, KDTree for low-dim
        return 'kd_tree' if n_features <= 15 else 'ball_tree'

    def _build_tree(self, X: torch.Tensor, alg: str):
        """Build a BallTree or KDTree over X for fast k-NN queries."""
        metric_fn = (self.metric
                     if callable(self.metric) and self.metric != "precomputed"
                     else None)
        if alg == 'ball_tree':
            return BallTree(X, leaf_size=self.leaf_size, metric=metric_fn,
                            device=str(self.device), dtype=self.dtype,
                            n_jobs=self.n_jobs or 1)
        if alg == 'kd_tree':
            return KDTree(X, leaf_size=self.leaf_size, metric=metric_fn,
                          device=str(self.device), dtype=self.dtype,
                          n_jobs=self.n_jobs or 1)
        return None  # brute force

    def _knn_query(self, X_query: torch.Tensor,
                   X_train: torch.Tensor,
                   k: int,
                   tree=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (distances, indices) of k-nearest neighbours in X_train.

        Uses *tree* when available; falls back to brute-force pairwise
        distance otherwise.  Self-distances (query == train point) are
        automatically removed when querying the training set by requesting
        k+1 neighbours and dropping the first column.
        """
        if tree is not None:
            # Tree: request k+1 to account for the query point itself
            dists, idx = tree.query(X_query, k=k + 1)
            # Ensure (n, k+1) shape
            if dists.dim() == 1:
                dists = dists.unsqueeze(0)
                idx = idx.unsqueeze(0)
            return dists, idx

        # Brute force: compute full pairwise distance matrix
        dist_fn = self.metric
        if callable(dist_fn) and dist_fn != "precomputed":
            D = dist_fn(X_query, X_train)
        else:
            # Fall back to Minkowski-p
            D = torch.cdist(X_query.float(), X_train.float(), p=self.p).to(self.dtype)

        kp1 = min(k + 1, X_train.shape[0])
        dists, idx = torch.topk(D, k=kp1, dim=1, largest=False, sorted=True)
        return dists, idx

    # ------------------------------------------------------------------
    # LRD computation
    # ------------------------------------------------------------------

    def _compute_lrd(self, knn_dists: torch.Tensor,
                     knn_idx: torch.Tensor,
                     k_distances_train: torch.Tensor) -> torch.Tensor:
        """Compute local reachability density for a batch of query points.

        The *reachability distance* of point x w.r.t. neighbour o is:
            reach-dist_k(x, o) = max(k-dist(o), d(x, o))

        where k-dist(o) is the distance to the k-th nearest neighbour of o
        in the training set.  The LRD is the reciprocal of the mean
        reachability distance over the k neighbours.
        """
        k = knn_dists.shape[1]
        # k-distance of each neighbour  (shape: n_query × k)
        neighbor_kdists = k_distances_train[knn_idx[:, :k]]
        # reachability distance
        reach_dists = torch.max(neighbor_kdists, knn_dists[:, :k])
        lrd = 1.0 / (reach_dists.mean(dim=1) + 1e-10)
        return lrd

    # ------------------------------------------------------------------
    # fit / predict / score
    # ------------------------------------------------------------------

    def fit(self, data_or_X, y=None, **kwargs):
        """Fit the LOF model to the training data *data_or_X*.

        Builds the k-NN graph over the training set, computes local
        reachability densities, and stores the negative LOF scores.
        """
        X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self.n_samples_fit_ = n_samples

        # Clamp k to a valid range
        self.n_neighbors_ = max(1, min(self.n_neighbors, n_samples - 1))

        # Set up the distance function via MLCluster._create_metric
        self._create_metric(self._metric_spec, dict(self.metric_params, p=self.p))
        self.effective_metric_ = (self._metric_spec if isinstance(self._metric_spec, str)
                                  else 'callable')
        self.effective_metric_params_ = dict(self.metric_params)
        self._X_fit = X

        # Build spatial index
        resolved_alg = self._resolve_algorithm(n_samples, n_features)
        if resolved_alg in ('ball_tree', 'kd_tree'):
            self._tree = self._build_tree(X, resolved_alg)
        else:
            self._tree = None

        k = self.n_neighbors_

        # k-NN distances among training samples
        all_dists, all_idx = self._knn_query(X, X, k, self._tree)
        # Drop the 0-distance self-hit (first column when using tree or brute)
        knn_dists = all_dists[:, 1:k + 1]   # (n, k) — skip col 0 (self)
        knn_idx   = all_idx[:, 1:k + 1]      # (n, k)
        # k-distance = distance to the k-th neighbour (index k after dropping self)
        k_dists = all_dists[:, k]             # (n,)

        self._k_distances_fit = k_dists
        self._knn_idx_fit = knn_idx

        # Compute LRD for training samples
        self._lrd_fit = self._compute_lrd(knn_dists, knn_idx, k_dists)

        # LOF = mean(LRD of k-neighbours) / LRD(x)
        lrd_neighbors = self._lrd_fit[knn_idx]          # (n, k)
        lof = lrd_neighbors.mean(dim=1) / (self._lrd_fit + 1e-10)  # (n,)
        self.negative_outlier_factor_ = -lof

        # Decision threshold
        if self.contamination == "auto":
            self.offset_ = -1.5
        else:
            self.offset_ = float(
                torch.quantile(self.negative_outlier_factor_,
                               float(self.contamination)).item())

        return self

    def score_samples(self, X) -> torch.Tensor:
        """Return negative LOF scores for *X* (higher = more normal).

        In novelty=False mode the training scores are returned; in
        novelty=True mode fresh LOF scores are computed against the
        stored training set.
        """
        if self._X_fit is None:
            raise RuntimeError("LocalOutlierFactor is not fitted yet.")
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)

        if not self.novelty:
            # Cached training scores
            return self.negative_outlier_factor_

        # Novelty mode: score new samples against the training set
        k = self.n_neighbors_
        all_dists, all_idx = self._knn_query(X, self._X_fit, k, self._tree)
        knn_dists = all_dists[:, :k]
        knn_idx   = all_idx[:, :k]

        lrd_query = self._compute_lrd(knn_dists, knn_idx, self._k_distances_fit)
        lrd_neighbors = self._lrd_fit[knn_idx]
        lof = lrd_neighbors.mean(dim=1) / (lrd_query + 1e-10)
        return -lof

    def decision_function(self, X) -> torch.Tensor:
        """Shifted LOF score: score_samples(X) - offset_.

        Values >= 0 indicate inliers; < 0 indicate outliers.
        """
        if not self.novelty:
            raise AttributeError(
                "decision_function is not available when novelty=False. "
                "Use fit_predict() to label training samples.")
        return self.score_samples(X) - self.offset_

    def predict(self, X) -> torch.Tensor:
        """Predict +1 for inliers and -1 for outliers.

        Only available when ``novelty=True``.
        """
        if not self.novelty:
            raise AttributeError(
                "predict() is not available when novelty=False. "
                "Use fit_predict() to obtain labels for training samples.")
        df = self.decision_function(X)
        return torch.where(df >= 0,
                           torch.ones_like(df, dtype=torch.long),
                           -torch.ones_like(df, dtype=torch.long))

    def fit_predict(self, data_or_X, y=None, **kwargs) -> torch.Tensor:
        """Fit the model and return outlier labels for the training data.

        Returns +1 for inliers and -1 for outliers.
        """
        self.fit(data_or_X, y)
        df = self.negative_outlier_factor_ - self.offset_
        return torch.where(df >= 0,
                           torch.ones_like(df, dtype=torch.long),
                           -torch.ones_like(df, dtype=torch.long))

    def fit_transform(self, data_or_X, y=None, **kwargs) -> torch.Tensor:
        """Fit the model and return negative LOF scores for the training data."""
        self.fit(data_or_X, y)
        return self.negative_outlier_factor_

    def transform(self, X, **kwargs) -> torch.Tensor:
        """Return negative LOF scores for *X* (requires novelty=True)."""
        return self.score_samples(X)

    def forward(self, X):
        return self.score_samples(X)


# ============================================================
# OneClassSVM
# ============================================================

class OneClassSVM(MLModule):
    def __init__(self,
                 kernel: Union[str, Callable, nn.Module] = 'rbf',
                 kernel_params: dict = None,
                 degree: int = 3,
                 gamma: Union[Literal["scale", "auto"], float] = 'scale',
                 coef0: float = 0.0,
                 tol: float = 0.001,
                 nu: float = 0.5,
                 shrinking: bool = True,
                 cache_size: float = 200,
                 verbose: bool = False,
                 max_iter: int = -1,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.kernel = kernel
        self.kernel_params = kernel_params or {}
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.device = device
        self.dtype = dtype

        # Fitted attributes
        self.coef_: Optional[torch.Tensor] = None
        self.dual_coef_: Optional[torch.Tensor] = None
        self.fit_status_: Optional[int] = None
        self.intercept_: Optional[torch.Tensor] = None
        self.n_features_in_: Optional[int] = None
        self.n_iter_: Optional[int] = None
        self.n_support_: Optional[torch.Tensor] = None
        self.offset_: Optional[float] = None
        self.shape_fit_: Optional[tuple] = None
        self.support_: Optional[torch.Tensor] = None
        self.support_vectors_: Optional[torch.Tensor] = None
        self._X_fit: Optional[torch.Tensor] = None
        self._alpha: Optional[torch.Tensor] = None
        self._gamma_val: Optional[float] = None
        self._kernel_obj: Optional[nn.Module] = None  # instantiated kernel

    def _resolve_gamma(self, n_features: int, X: torch.Tensor) -> float:
        """Compute the scalar gamma value used by the kernel."""
        if isinstance(self.gamma, float):
            return float(self.gamma)
        if self.gamma == "scale":
            var = X.var().item()
            return 1.0 / (n_features * var) if var > 0 else 1.0
        # "auto"
        return 1.0 / n_features

    def _build_kernel_obj(self, gamma: float) -> Optional[nn.Module]:
        """Instantiate the kernel object from kernels.py when possible.

        Returns an :class:`nn.Module` kernel that can be called as
        ``kernel_obj(xi, xj)``, or None if a plain callable was supplied.
        """
        kspec = self.kernel
        if isinstance(kspec, nn.Module):
            return kspec
        if callable(kspec):
            return None  # use the callable directly

        # String kernel name: instantiate via registry or hard-coded classes
        name = str(kspec).lower()
        kparams = dict(self.kernel_params)

        if name == "rbf":
            return RBFKernel(gamma=gamma,
                             device=str(self.device), dtype=self.dtype)
        if name == "linear":
            return LinearKernel(device=str(self.device), dtype=self.dtype)
        if name == "poly":
            return PolyKernel(degree=float(kparams.get("degree", self.degree)),
                              gamma=gamma,
                              bias=float(kparams.get("coef0", self.coef0)),
                              device=str(self.device), dtype=self.dtype)
        if name == "sigmoid":
            return SigmoidKernel(gamma=gamma,
                                 bias=float(kparams.get("coef0", self.coef0)),
                                 device=str(self.device), dtype=self.dtype)
        if name == "laplacian":
            sigma = 1.0 / (gamma + 1e-12)
            return LaplacianKernel(sigma=sigma,
                                   device=str(self.device), dtype=self.dtype)
        if name == "precomputed":
            return None

        # Try generic registry
        klass = get_kernel_class(name)
        if klass is not None:
            try:
                return klass(gamma=gamma, **kparams,
                             device=str(self.device), dtype=self.dtype)
            except TypeError:
                return klass(device=str(self.device), dtype=self.dtype)
        return None

    def _compute_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute the kernel matrix K(X, Y) using the instantiated kernel object.

        If a kernel object was built during fit, it is used directly.
        Otherwise falls back to the callable ``self.kernel``.
        For 'precomputed', X is assumed to already be the kernel matrix.
        """
        if self.kernel == "precomputed":
            return X  # X is already the kernel / distance matrix

        if self._kernel_obj is not None:
            K = self._kernel_obj(X, Y)
            # Ensure 2-D output (kernel objects may return 3-D when
            # trainable=False but batch dim was injected)
            if K.dim() == 3:
                K = K.squeeze(0)
            return K

        # Plain callable supplied by the user
        if callable(self.kernel):
            return self.kernel(X, Y)

        # Should not reach here; fall back to manual RBF
        gamma = self._gamma_val or 1.0
        D = torch.cdist(X.float(), Y.float(), p=2).to(self.dtype)
        return torch.exp(-gamma * D ** 2)

    @staticmethod
    def _project_simplex_box(alpha: torch.Tensor, upper: float) -> torch.Tensor:
        """Project *alpha* onto { α : Σαᵢ = 1, 0 ≤ αᵢ ≤ upper }.

        Uses the O(n log n) algorithm of Duchi et al. (2008) adapted to a
        box-constrained simplex.
        """
        # Clip to box
        alpha = alpha.clamp(0.0, upper)
        s = alpha.sum()
        if abs(s.item() - 1.0) < 1e-9:
            return alpha

        n = alpha.shape[0]
        alpha_sorted, _ = torch.sort(alpha, descending=True)
        cssv = torch.cumsum(alpha_sorted, dim=0)
        rho_mask = ((cssv - 1.0) / torch.arange(1, n + 1,
                                                   device=alpha.device,
                                                   dtype=alpha.dtype)) < alpha_sorted
        rho = rho_mask.nonzero(as_tuple=False)
        rho_val = int(rho[-1].item()) + 1 if rho.numel() > 0 else n
        theta = (cssv[rho_val - 1] - 1.0) / rho_val
        return (alpha - theta).clamp(0.0, upper)

    def fit(self, data_or_X, y=None, **kwargs):
        """Fit the One-Class SVM by solving the dual QP via projected gradient.

        The algorithm iterates:
            α ← Π[ α − η ∇α(½ α^T K α) ]
        where Π projects onto the simplex-box { Σα=1, 0≤α≤1/(ν·n) }.
        Convergence is declared when max |Δα| < tol.
        """
        if self.kernel == "precomputed":
            K = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
            n_samples = K.shape[0]
            self.n_features_in_ = n_samples
            self._X_fit = None
        else:
            X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
            if X.dim() == 1:
                X = X.unsqueeze(0)
            n_samples, n_features = X.shape
            self.n_features_in_ = n_features
            self.shape_fit_ = tuple(X.shape)
            self._X_fit = X

            # Resolve scalar gamma and build kernel object
            self._gamma_val = self._resolve_gamma(n_features, X)
            self._kernel_obj = self._build_kernel_obj(self._gamma_val)

            K = self._compute_kernel(X, X)

        # Projected gradient dual QP
        max_iter = self.max_iter if self.max_iter > 0 else 1000
        upper = 1.0 / (self.nu * n_samples)

        # Initialise α uniformly on the simplex
        alpha = torch.full((n_samples,), 1.0 / n_samples,
                           device=self.device, dtype=self.dtype)
        # Step size: largest eigenvalue of K (upper bound via diagonal)
        lr = 1.0 / (K.diag().max().item() + 1e-8)

        converged = False
        for it in range(max_iter):
            # Gradient of ½ α^T K α is K α
            grad = K @ alpha
            alpha_new = self._project_simplex_box(alpha - lr * grad, upper)
            delta = (alpha_new - alpha).abs().max().item()
            alpha = alpha_new
            if delta < self.tol:
                self.n_iter_ = it + 1
                converged = True
                break
        if not converged:
            self.n_iter_ = max_iter
            if self.verbose:
                warnings.warn("OneClassSVM did not converge.", RuntimeWarning)

        self._alpha = alpha

        # Identify support vectors (α > threshold)
        sv_thresh = 1e-6 * upper
        sv_mask = alpha > sv_thresh
        self.support_ = sv_mask.nonzero(as_tuple=False).squeeze(1)
        self.support_vectors_ = (self._X_fit[self.support_]
                                  if self._X_fit is not None else None)
        self.n_support_ = torch.tensor([self.support_.shape[0]], dtype=torch.int32)
        self.dual_coef_ = alpha[self.support_].unsqueeze(0)

        # Compute ρ (bias) as the average decision value of support vectors
        K_sv = K[self.support_]                         # (n_SV, n)
        f_sv = (K_sv * alpha.unsqueeze(0)).sum(dim=1)   # (n_SV,)
        rho = f_sv.mean().item()
        self.intercept_ = torch.tensor([-rho], dtype=self.dtype, device=self.device)
        self.offset_ = rho
        self.fit_status_ = 0

        # For linear kernel, expose coef_ = Σ αᵢ xᵢ
        if isinstance(self.kernel, str) and self.kernel == "linear" \
                and self._X_fit is not None:
            sv_alpha = alpha[self.support_].unsqueeze(1)
            self.coef_ = (sv_alpha * self._X_fit[self.support_]).sum(dim=0, keepdim=True)

        if self.verbose:
            print(f"OneClassSVM: {self.support_.shape[0]} support vectors, "
                  f"ρ={rho:.4f}, iters={self.n_iter_}")
        return self

    def score_samples(self, X) -> torch.Tensor:
        """Raw decision scores f(x) = Σ αᵢ K(xᵢ, x).

        Higher values indicate samples closer to the estimated support
        (more normal); lower values indicate outliers.
        """
        if self._alpha is None:
            raise RuntimeError("OneClassSVM is not fitted yet.")
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        with torch.no_grad():
            K = self._compute_kernel(X, self._X_fit)
            return K @ self._alpha

    def decision_function(self, X) -> torch.Tensor:
        """Decision function: score_samples(X) - ρ (= offset_).

        Samples with decision_function(X) >= 0 are classified as inliers.
        """
        return self.score_samples(X) - self.offset_

    def predict(self, X) -> torch.Tensor:
        """Predict +1 for inliers and -1 for outliers."""
        df = self.decision_function(X)
        return torch.where(df >= 0,
                           torch.ones_like(df, dtype=torch.long),
                           -torch.ones_like(df, dtype=torch.long))

    def forward(self, X):
        return self.decision_function(X)


# ============================================================
# SGDOneClassSVM — linear primal form with SGD
# ============================================================

class SGDOneClassSVM(OneClassSVM):
    def __init__(self,
                 nu: float = 0.5,
                 fit_intercept: bool = True,
                 max_iter: int = 1000,
                 tol: float = 1e-3,
                 shuffle: bool = True,
                 verbose: int = 0,
                 random_state: Union[int, None] = None,
                 learning_rate: Literal["constant", "optimal",
                                        "invscaling", "adaptive"] = "optimal",
                 eta0: float = 0.01,
                 power_t: float = 0.5,
                 warm_start: bool = False,
                 average: bool = False,
                 # OneClassSVM pass-through params (kept for full signature)
                 kernel: Union[str, Callable, nn.Module] = 'rbf',
                 kernel_params: dict = None,
                 degree: int = 3,
                 gamma: Union[Literal["scale", "auto"], float] = 'scale',
                 coef0: float = 0.0,
                 shrinking: bool = True,
                 cache_size: float = 200,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(kernel=kernel, kernel_params=kernel_params,
                         degree=degree, gamma=gamma, coef0=coef0,
                         tol=tol, nu=nu, shrinking=shrinking,
                         cache_size=cache_size, verbose=verbose,
                         max_iter=max_iter, device=device, dtype=dtype,
                         *args, **kwargs)
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.warm_start = warm_start
        self.average = average

        # Primal parameters
        self._w: Optional[torch.nn.Parameter] = None
        self._rho: Optional[torch.nn.Parameter] = None
        # Running sums for averaging
        self._w_sum: Optional[torch.Tensor] = None
        self._rho_sum: Optional[torch.Tensor] = None
        self.num_iter_t: int = 0

    @property
    def t_(self) -> float:
        """Current step count (1-indexed)."""
        return float(self.num_iter_t + 1)

    def _init_params(self, n_features: int) -> None:
        """Initialise the primal parameter tensors w and ρ."""
        self._w = nn.Parameter(
            torch.zeros(1, n_features, device=self.device, dtype=self.dtype))
        rho_init = (torch.ones(1, device=self.device, dtype=self.dtype)
                    if self.fit_intercept
                    else torch.zeros(1, device=self.device, dtype=self.dtype))
        self._rho = nn.Parameter(rho_init, requires_grad=self.fit_intercept)

    def _init_state(self) -> None:
        """Initialise any optimizer-specific state (e.g. momentum buffers).

        Overridden by subclasses to allocate buffers that match the shape of
        ``_w`` and ``_rho``.
        """
        pass

    def _compute_ocsvm_loss(self, X_batch: torch.Tensor) -> torch.Tensor:
        """Compute the primal OC-SVM hinge loss for a mini-batch.

        L = ½‖w‖² + (1/ν·n_batch)·Σ max(0, ρ - w·xᵢ) - ρ

        The ½‖w‖² regulariser encourages a large margin; the hinge term
        penalises points that fall below the hyperplane w·x = ρ; and the
        -ρ term prevents the trivial solution ρ = -∞.

        When ``_fit_loss_fn`` is set (via fit(..., loss_fn=...)), the data
        loss is replaced by loss_fn(X_batch, scores) where scores = w·x − ρ;
        L2 regularization (½‖w‖²) is still applied on top.
        """
        n_batch = X_batch.shape[0]
        scores = (X_batch @ self._w.T).squeeze(-1) - self._rho.squeeze()  # w·x − ρ
        custom_loss_fn = getattr(self, '_fit_loss_fn', None)
        if custom_loss_fn is not None:
            data_loss = custom_loss_fn(X_batch, scores)
            reg = 0.5 * (self._w ** 2).sum()
            return data_loss + reg
        hinge = torch.clamp(self._rho.squeeze() - (X_batch @ self._w.T).squeeze(-1), min=0.0)
        loss = (0.5 * (self._w ** 2).sum()
                + (1.0 / (self.nu * n_batch)) * hinge.sum()
                - self._rho)
        return loss

    def _zero_grad(self) -> None:
        """Zero gradients of w and ρ before a backward pass."""
        if self._w.grad is not None:
            self._w.grad.zero_()
        if self._rho.grad is not None:
            self._rho.grad.zero_()

    def _get_eta(self) -> float:
        """Return the learning rate at the current step t."""
        return _get_lr(self.learning_rate, self.eta0, self.power_t,
                       self.t_, self.nu)

    def _update_params(self, X_batch: torch.Tensor) -> float:
        """SGD parameter update for one mini-batch.

        Computes the OC-SVM loss, calls backward(), then performs a plain
        gradient-descent step with the schedule-dependent learning rate.
        Subclasses override this method to inject different optimisers while
        reusing the shared loss computation.

        Returns the scalar loss value.
        """
        self._zero_grad()
        loss = self._compute_ocsvm_loss(X_batch)
        loss.backward()
        eta = self._get_eta()
        with torch.no_grad():
            self._w.data.sub_(eta * self._w.grad)
            if self.fit_intercept:
                self._rho.data.sub_(eta * self._rho.grad)
            if self.average:
                self._w_sum.add_(self._w)
                self._rho_sum.add_(self._rho)
        self.num_iter_t += 1
        return loss.item()

    def _partial_fit(self, X: torch.Tensor) -> float:
        """Run one epoch (one pass through X sample-by-sample)."""
        n = X.shape[0]
        total = 0.0
        for i in range(n):
            total += self._update_params(X[i:i + 1])
        return total / max(n, 1)

    def fit(self, data_or_X, y=None, **kwargs):
        """Fit the linear primal One-Class SVM via SGD.

        Iterates for up to ``max_iter`` epochs.  Each epoch shuffles the
        data (if ``shuffle=True``) and calls :meth:`_partial_fit`.
        Training stops early when the absolute change in epoch loss is
        below ``tol``.
        """
        self._fit_loss_fn = kwargs.get('loss_fn', None)
        try:
            warm_start = kwargs.get('warm_start', self.warm_start)
            X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
            if X.dim() == 1:
                X = X.unsqueeze(0)
            n_samples, n_features = X.shape
            self.n_features_in_ = n_features
            self.shape_fit_ = tuple(X.shape)

            if self._w is None or not warm_start:
                self._init_params(n_features)
                self._init_state()
                self.num_iter_t = 0

            if self.average and (self._w_sum is None or not warm_start):
                self._w_sum = torch.zeros_like(self._w.data)
                self._rho_sum = torch.zeros_like(self._rho.data)

            if self.random_state is not None:
                torch.manual_seed(int(self.random_state))

            max_iter = self.max_iter if self.max_iter > 0 else 1000
            prev_loss = float("inf")

            for epoch in range(max_iter):
                if self.shuffle:
                    perm = torch.randperm(n_samples, device=self.device)
                    X_ep = X[perm]
                else:
                    X_ep = X

                epoch_loss = self._partial_fit(X_ep)

                if self.verbose:
                    print(f"SGDOneClassSVM epoch {epoch + 1}: loss={epoch_loss:.6f}")

                if abs(prev_loss - epoch_loss) < self.tol and epoch > 0:
                    self.n_iter_ = epoch + 1
                    break
                prev_loss = epoch_loss
            else:
                self.n_iter_ = max_iter

            # Apply parameter averaging if requested
            if self.average and self.num_iter_t > 0:
                denom = float(self.num_iter_t)
                with torch.no_grad():
                    self._w.data.copy_(self._w_sum / denom)
                    self._rho.data.copy_(self._rho_sum / denom)

            # Expose sklearn-compatible attributes
            self.coef_ = self._w.detach()
            rho_val = self._rho.item()
            self.intercept_ = torch.tensor([-rho_val], dtype=self.dtype, device=self.device)
            self.offset_ = torch.tensor([0.0], dtype=self.dtype, device=self.device)
            self.fit_status_ = 0
            return self
        finally:
            self._fit_loss_fn = None

    def score_samples(self, X) -> torch.Tensor:
        """Return raw decision scores: w·x − ρ for each sample."""
        if self._w is None:
            raise RuntimeError("SGDOneClassSVM is not fitted yet.")
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        with torch.no_grad():
            return (X @ self._w.T).squeeze(-1) - self._rho.squeeze()

    def decision_function(self, X) -> torch.Tensor:
        """Decision scores (score_samples minus offset_, i.e. same as scores here)."""
        scores = self.score_samples(X)
        if self.offset_ is not None and isinstance(self.offset_, torch.Tensor):
            return scores - self.offset_.squeeze()
        return scores

    def predict(self, X) -> torch.Tensor:
        """Predict +1 for inliers and -1 for outliers."""
        df = self.decision_function(X)
        return torch.where(df >= 0,
                           torch.ones_like(df, dtype=torch.long),
                           -torch.ones_like(df, dtype=torch.long))

    def forward(self, X):
        return self.decision_function(X)



# ============================================================
# OneClassSVM optimizer variants
# ============================================================

class BGDOneClassSVM(SGDOneClassSVM):
    """One-Class SVM trained with Batch Gradient Descent."""
    def __init__(self, kernel='rbf', kernel_params=None, degree=3, gamma='scale',
                 coef0=0.0, tol=0.001, fit_intercept=True, shuffle=True,
                 random_state=None, learning_rate='invscaling', eta0=0.01,
                 power_t=0.5, warm_start=False, average=False, nu=0.5,
                 shrinking=True, cache_size=200, verbose=False, max_iter=-1,
                 batch_size: int = 32,
                 device='cpu', dtype=torch.float, *args, **kwargs):
        super().__init__(kernel=kernel, kernel_params=kernel_params, degree=degree, gamma=gamma,
                         coef0=coef0, tol=tol, fit_intercept=fit_intercept, shuffle=shuffle,
                         random_state=random_state, learning_rate=learning_rate, eta0=eta0,
                         power_t=power_t, warm_start=warm_start, average=average, nu=nu,
                         shrinking=shrinking, cache_size=cache_size, verbose=verbose,
                         max_iter=max_iter, device=device, dtype=dtype, *args, **kwargs)
        self.batch_size = batch_size

    def _partial_fit(self, X):
        n = X.shape[0]
        total, nb = 0.0, 0
        idx = torch.randperm(n, device=self.device) if self.shuffle else torch.arange(n, device=self.device)
        for i in range(0, n, self.batch_size):
            total += self._update_params(X[idx[i:i + self.batch_size]])
            nb += 1
        return total / max(nb, 1)


class MGDOneClassSVM(SGDOneClassSVM):
    """One-Class SVM trained with Momentum Gradient Descent."""
    def __init__(self, kernel='rbf', kernel_params=None, degree=3, gamma='scale',
                 coef0=0.0, tol=0.001, fit_intercept=True, shuffle=True,
                 random_state=None, learning_rate='invscaling', eta0=0.01,
                 power_t=0.5, warm_start=False, average=False, nu=0.5,
                 shrinking=True, cache_size=200, verbose=False, max_iter=-1,
                 gamma_momentum: float = 0.9,
                 device='cpu', dtype=torch.float, *args, **kwargs):
        super().__init__(kernel=kernel, kernel_params=kernel_params, degree=degree, gamma=gamma,
                         coef0=coef0, tol=tol, fit_intercept=fit_intercept, shuffle=shuffle,
                         random_state=random_state, learning_rate=learning_rate, eta0=eta0,
                         power_t=power_t, warm_start=warm_start, average=average, nu=nu,
                         shrinking=shrinking, cache_size=cache_size, verbose=verbose,
                         max_iter=max_iter, device=device, dtype=dtype, *args, **kwargs)
        self.gamma_momentum = gamma_momentum
        self._v_w = None
        self._v_rho = None

    def _init_state(self):
        if self._w is not None:
            self._v_w = torch.zeros_like(self._w)
            self._v_rho = torch.zeros_like(self._rho)

    def _update_params(self, X_batch):
        self._zero_grad()
        loss = self._compute_ocsvm_loss(X_batch)
        loss.backward()
        eta = self._get_eta()
        with torch.no_grad():
            if self._v_w is None:
                self._v_w = torch.zeros_like(self._w)
                self._v_rho = torch.zeros_like(self._rho)
            self._v_w.mul_(self.gamma_momentum).add_(self._w.grad, alpha=eta)
            self._w.data.sub_(self._v_w)
            self._v_rho.mul_(self.gamma_momentum).add_(self._rho.grad, alpha=eta)
            self._rho.data.sub_(self._v_rho)
        self.num_iter_t += 1
        return loss.item()


class AdaGradOneClassSVM(SGDOneClassSVM):
    """One-Class SVM trained with AdaGrad."""
    def __init__(self, kernel='rbf', kernel_params=None, degree=3, gamma='scale',
                 coef0=0.0, tol=0.001, fit_intercept=True, shuffle=True,
                 random_state=None, learning_rate='invscaling', eta0=0.01,
                 power_t=0.5, warm_start=False, average=False, nu=0.5,
                 shrinking=True, cache_size=200, verbose=False, max_iter=-1,
                 smoothening_term: float = 1e-8,
                 device='cpu', dtype=torch.float, *args, **kwargs):
        super().__init__(kernel=kernel, kernel_params=kernel_params, degree=degree, gamma=gamma,
                         coef0=coef0, tol=tol, fit_intercept=fit_intercept, shuffle=shuffle,
                         random_state=random_state, learning_rate=learning_rate, eta0=eta0,
                         power_t=power_t, warm_start=warm_start, average=average, nu=nu,
                         shrinking=shrinking, cache_size=cache_size, verbose=verbose,
                         max_iter=max_iter, device=device, dtype=dtype, *args, **kwargs)
        self.smoothening_term = smoothening_term
        self._G_w = None
        self._G_rho = None

    def _init_state(self):
        if self._w is not None:
            self._G_w = torch.zeros_like(self._w)
            self._G_rho = torch.zeros_like(self._rho)

    def _update_params(self, X_batch):
        self._zero_grad()
        loss = self._compute_ocsvm_loss(X_batch)
        loss.backward()
        eta = self._get_eta()
        with torch.no_grad():
            if self._G_w is None:
                self._G_w = torch.zeros_like(self._w)
                self._G_rho = torch.zeros_like(self._rho)
            self._G_w.add_(self._w.grad.pow(2))
            self._w.data.addcdiv_(self._w.grad, (self._G_w + self.smoothening_term).sqrt(), value=-eta)
            self._G_rho.add_(self._rho.grad.pow(2))
            self._rho.data.addcdiv_(self._rho.grad, (self._G_rho + self.smoothening_term).sqrt(), value=-eta)
        self.num_iter_t += 1
        return loss.item()


class RMSPropOneClassSVM(AdaGradOneClassSVM):
    """One-Class SVM trained with RMSProp."""
    def __init__(self, kernel='rbf', kernel_params=None, degree=3, gamma='scale',
                 coef0=0.0, tol=0.001, fit_intercept=True, shuffle=True,
                 random_state=None, learning_rate='invscaling', eta0=0.01,
                 power_t=0.5, warm_start=False, average=False, nu=0.5,
                 shrinking=True, cache_size=200, verbose=False, max_iter=-1,
                 smoothening_term: float = 1e-8, decay_rate: float = 0.9,
                 device='cpu', dtype=torch.float, *args, **kwargs):
        super().__init__(kernel=kernel, kernel_params=kernel_params, degree=degree, gamma=gamma,
                         coef0=coef0, tol=tol, fit_intercept=fit_intercept, shuffle=shuffle,
                         random_state=random_state, learning_rate=learning_rate, eta0=eta0,
                         power_t=power_t, warm_start=warm_start, average=average, nu=nu,
                         shrinking=shrinking, cache_size=cache_size, verbose=verbose,
                         max_iter=max_iter, smoothening_term=smoothening_term,
                         device=device, dtype=dtype, *args, **kwargs)
        self.decay_rate = decay_rate

    def _update_params(self, X_batch):
        self._zero_grad()
        loss = self._compute_ocsvm_loss(X_batch)
        loss.backward()
        eta = self._get_eta()
        with torch.no_grad():
            if self._G_w is None:
                self._G_w = torch.zeros_like(self._w)
                self._G_rho = torch.zeros_like(self._rho)
            self._G_w.mul_(self.decay_rate).addcmul_(self._w.grad, self._w.grad, value=1 - self.decay_rate)
            self._w.data.addcdiv_(self._w.grad, (self._G_w + self.smoothening_term).sqrt(), value=-eta)
            self._G_rho.mul_(self.decay_rate).addcmul_(self._rho.grad, self._rho.grad, value=1 - self.decay_rate)
            self._rho.data.addcdiv_(self._rho.grad, (self._G_rho + self.smoothening_term).sqrt(), value=-eta)
        self.num_iter_t += 1
        return loss.item()


class AdamOneClassSVM(SGDOneClassSVM):
    """One-Class SVM trained with Adam optimizer."""
    def __init__(self, kernel='rbf', kernel_params=None, degree=3, gamma='scale',
                 coef0=0.0, tol=0.001, fit_intercept=True, shuffle=True,
                 random_state=None, learning_rate='invscaling', eta0=0.01,
                 power_t=0.5, warm_start=False, average=False, nu=0.5,
                 shrinking=True, cache_size=200, verbose=False, max_iter=-1,
                 beta1: float = 0.9, beta2: float = 0.999, smoothening_term: float = 1e-8,
                 device='cpu', dtype=torch.float, *args, **kwargs):
        super().__init__(kernel=kernel, kernel_params=kernel_params, degree=degree, gamma=gamma,
                         coef0=coef0, tol=tol, fit_intercept=fit_intercept, shuffle=shuffle,
                         random_state=random_state, learning_rate=learning_rate, eta0=eta0,
                         power_t=power_t, warm_start=warm_start, average=average, nu=nu,
                         shrinking=shrinking, cache_size=cache_size, verbose=verbose,
                         max_iter=max_iter, device=device, dtype=dtype, *args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.smoothening_term = smoothening_term
        self._m_w = None
        self._v_w = None
        self._m_rho = None
        self._v_rho = None

    def _init_state(self):
        if self._w is not None:
            self._m_w = torch.zeros_like(self._w)
            self._v_w = torch.zeros_like(self._w)
            self._m_rho = torch.zeros_like(self._rho)
            self._v_rho = torch.zeros_like(self._rho)

    def _update_params(self, X_batch):
        self._zero_grad()
        loss = self._compute_ocsvm_loss(X_batch)
        loss.backward()
        eta = self._get_eta()
        with torch.no_grad():
            if self._m_w is None:
                self._m_w = torch.zeros_like(self._w)
                self._v_w = torch.zeros_like(self._w)
                self._m_rho = torch.zeros_like(self._rho)
                self._v_rho = torch.zeros_like(self._rho)
            t = self.num_iter_t + 1
            self._m_w.mul_(self.beta1).add_(self._w.grad, alpha=1 - self.beta1)
            self._v_w.mul_(self.beta2).addcmul_(self._w.grad, self._w.grad, value=1 - self.beta2)
            m_hat = self._m_w / (1 - self.beta1 ** t)
            v_hat = self._v_w / (1 - self.beta2 ** t)
            self._w.data.addcdiv_(m_hat, v_hat.sqrt().add_(self.smoothening_term), value=-eta)
            self._m_rho.mul_(self.beta1).add_(self._rho.grad, alpha=1 - self.beta1)
            self._v_rho.mul_(self.beta2).addcmul_(self._rho.grad, self._rho.grad, value=1 - self.beta2)
            m_hat_r = self._m_rho / (1 - self.beta1 ** t)
            v_hat_r = self._v_rho / (1 - self.beta2 ** t)
            self._rho.data.addcdiv_(m_hat_r, v_hat_r.sqrt().add_(self.smoothening_term), value=-eta)
        self.num_iter_t += 1
        return loss.item()


class AdadeltaOneClassSVM(SGDOneClassSVM):
    """One-Class SVM trained with Adadelta."""
    def __init__(self, kernel='rbf', kernel_params=None, degree=3, gamma='scale',
                 coef0=0.0, tol=0.001, fit_intercept=True, shuffle=True,
                 random_state=None, learning_rate='invscaling', eta0=0.01,
                 power_t=0.5, warm_start=False, average=False, nu=0.5,
                 shrinking=True, cache_size=200, verbose=False, max_iter=-1,
                 rho: float = 0.9, smoothening_term: float = 1e-6,
                 device='cpu', dtype=torch.float, *args, **kwargs):
        super().__init__(kernel=kernel, kernel_params=kernel_params, degree=degree, gamma=gamma,
                         coef0=coef0, tol=tol, fit_intercept=fit_intercept, shuffle=shuffle,
                         random_state=random_state, learning_rate=learning_rate, eta0=eta0,
                         power_t=power_t, warm_start=warm_start, average=average, nu=nu,
                         shrinking=shrinking, cache_size=cache_size, verbose=verbose,
                         max_iter=max_iter, device=device, dtype=dtype, *args, **kwargs)
        self.rho = rho
        self.smoothening_term = smoothening_term
        self._E_g2_w = None
        self._E_dx2_w = None
        self._E_g2_rho = None
        self._E_dx2_rho = None

    def _init_state(self):
        if self._w is not None:
            self._E_g2_w = torch.zeros_like(self._w)
            self._E_dx2_w = torch.zeros_like(self._w)
            self._E_g2_rho = torch.zeros_like(self._rho)
            self._E_dx2_rho = torch.zeros_like(self._rho)

    def _update_params(self, X_batch):
        self._zero_grad()
        loss = self._compute_ocsvm_loss(X_batch)
        loss.backward()
        eps = self.smoothening_term
        with torch.no_grad():
            if self._E_g2_w is None:
                self._E_g2_w = torch.zeros_like(self._w)
                self._E_dx2_w = torch.zeros_like(self._w)
                self._E_g2_rho = torch.zeros_like(self._rho)
                self._E_dx2_rho = torch.zeros_like(self._rho)
            self._E_g2_w.mul_(self.rho).addcmul_(self._w.grad, self._w.grad, value=1 - self.rho)
            dx_w = -((self._E_dx2_w + eps).sqrt() / (self._E_g2_w + eps).sqrt()) * self._w.grad
            self._w.data.add_(dx_w)
            self._E_dx2_w.mul_(self.rho).addcmul_(dx_w, dx_w, value=1 - self.rho)
            self._E_g2_rho.mul_(self.rho).addcmul_(self._rho.grad, self._rho.grad, value=1 - self.rho)
            dx_r = -((self._E_dx2_rho + eps).sqrt() / (self._E_g2_rho + eps).sqrt()) * self._rho.grad
            self._rho.data.add_(dx_r)
            self._E_dx2_rho.mul_(self.rho).addcmul_(dx_r, dx_r, value=1 - self.rho)
        self.num_iter_t += 1
        return loss.item()


class AdafactorOneClassSVM(SGDOneClassSVM):
    """One-Class SVM trained with Adafactor."""
    def __init__(self, kernel='rbf', kernel_params=None, degree=3, gamma='scale',
                 coef0=0.0, tol=0.001, fit_intercept=True, shuffle=True,
                 random_state=None, learning_rate='invscaling', eta0=0.01,
                 power_t=0.5, warm_start=False, average=False, nu=0.5,
                 shrinking=True, cache_size=200, verbose=False, max_iter=-1,
                 beta1: float = None, clip_threshold: float = 1.0,
                 decay_rate: float = -0.8, eps1: float = 1e-30, eps2: float = 1e-3,
                 device='cpu', dtype=torch.float, *args, **kwargs):
        super().__init__(kernel=kernel, kernel_params=kernel_params, degree=degree, gamma=gamma,
                         coef0=coef0, tol=tol, fit_intercept=fit_intercept, shuffle=shuffle,
                         random_state=random_state, learning_rate=learning_rate, eta0=eta0,
                         power_t=power_t, warm_start=warm_start, average=average, nu=nu,
                         shrinking=shrinking, cache_size=cache_size, verbose=verbose,
                         max_iter=max_iter, device=device, dtype=dtype, *args, **kwargs)
        self.beta1 = beta1
        self.clip_threshold = clip_threshold
        self.decay_rate = decay_rate
        self.eps1 = eps1
        self.eps2 = eps2
        self._row_v = None
        self._col_v = None
        self._v_rho = None

    def _update_params(self, X_batch):
        self._zero_grad()
        loss = self._compute_ocsvm_loss(X_batch)
        loss.backward()
        t = self.t_
        beta2 = 1.0 - math.pow(t, self.decay_rate)
        with torch.no_grad():
            grad = self._w.grad
            grad_sq = grad.pow(2).add_(self.eps1)
            if self._row_v is None:
                self._row_v = torch.zeros((self._w.shape[0], 1), device=self.device, dtype=self.dtype)
                self._col_v = torch.zeros((1, self._w.shape[1]), device=self.device, dtype=self.dtype)
                self._v_rho = torch.zeros_like(self._rho)
            self._row_v.mul_(beta2).add_(grad_sq.mean(dim=1, keepdim=True), alpha=1 - beta2)
            self._col_v.mul_(beta2).add_(grad_sq.mean(dim=0, keepdim=True), alpha=1 - beta2)
            row_col_mean = self._row_v.mean(dim=0, keepdim=True) + 1e-30
            v = (self._row_v @ self._col_v) / row_col_mean
            update = grad / v.sqrt().add_(self.eps2)
            u_norm = update.norm().item()
            if u_norm > self.clip_threshold:
                update = update * (self.clip_threshold / u_norm)
            self._w.data.sub_(update, alpha=self.eta0)
            self._v_rho.mul_(beta2).add_(self._rho.grad.pow(2), alpha=1 - beta2)
            self._rho.data.sub_(self._rho.grad / (self._v_rho.sqrt() + self.eps2), alpha=self.eta0)
        self.num_iter_t += 1
        return loss.item()


class AdamWOneClassSVM(AdamOneClassSVM):
    """One-Class SVM trained with AdamW (Adam + weight decay)."""
    def __init__(self, kernel='rbf', kernel_params=None, degree=3, gamma='scale',
                 coef0=0.0, tol=0.001, fit_intercept=True, shuffle=True,
                 random_state=None, learning_rate='invscaling', eta0=0.01,
                 power_t=0.5, warm_start=False, average=False, nu=0.5,
                 shrinking=True, cache_size=200, verbose=False, max_iter=-1,
                 beta1: float = 0.9, beta2: float = 0.999, smoothening_term: float = 1e-8,
                 weight_decay: float = 0.01,
                 device='cpu', dtype=torch.float, *args, **kwargs):
        super().__init__(kernel=kernel, kernel_params=kernel_params, degree=degree, gamma=gamma,
                         coef0=coef0, tol=tol, fit_intercept=fit_intercept, shuffle=shuffle,
                         random_state=random_state, learning_rate=learning_rate, eta0=eta0,
                         power_t=power_t, warm_start=warm_start, average=average, nu=nu,
                         shrinking=shrinking, cache_size=cache_size, verbose=verbose,
                         max_iter=max_iter, beta1=beta1, beta2=beta2,
                         smoothening_term=smoothening_term,
                         device=device, dtype=dtype, *args, **kwargs)
        self.weight_decay = weight_decay

    def _update_params(self, X_batch):
        with torch.no_grad():
            if self._w is not None:
                self._w.data.mul_(1 - self.eta0 * self.weight_decay)
        return super()._update_params(X_batch)


class AdamaxOneClassSVM(AdamOneClassSVM):
    """One-Class SVM trained with Adamax (Adam with infinity norm)."""
    def _update_params(self, X_batch):
        self._zero_grad()
        loss = self._compute_ocsvm_loss(X_batch)
        loss.backward()
        eta = self._get_eta()
        with torch.no_grad():
            if self._m_w is None:
                self._m_w = torch.zeros_like(self._w)
                self._v_w = torch.zeros_like(self._w)
                self._m_rho = torch.zeros_like(self._rho)
                self._v_rho = torch.zeros_like(self._rho)
            t = self.num_iter_t + 1
            self._m_w.mul_(self.beta1).add_(self._w.grad, alpha=1 - self.beta1)
            self._v_w = torch.max(self.beta2 * self._v_w, self._w.grad.abs())
            step = eta / (1 - self.beta1 ** t)
            self._w.data.addcdiv_(self._m_w, self._v_w.add(self.smoothening_term), value=-step)
            self._m_rho.mul_(self.beta1).add_(self._rho.grad, alpha=1 - self.beta1)
            self._v_rho = torch.max(self.beta2 * self._v_rho, self._rho.grad.abs())
            self._rho.data.addcdiv_(self._m_rho, self._v_rho.add(self.smoothening_term), value=-step)
        self.num_iter_t += 1
        return loss.item()


class LBFGSOneClassSVM(SGDOneClassSVM):
    """One-Class SVM optimized with L-BFGS (PyTorch LBFGS)."""
    def fit(self, data_or_X, y=None, **kwargs):
        self._fit_loss_fn = kwargs.get('loss_fn', None)
        try:
            warm_start = kwargs.get('warm_start', self.warm_start)
            X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
            if X.dim() == 1:
                X = X.unsqueeze(0)
            n_samples, n_features = X.shape
            self.n_features_in_ = n_features
            self.shape_fit_ = tuple(X.shape)
            if not warm_start or self._w is None or self._w.shape[1] != n_features:
                self._init_params(n_features)
                self.num_iter_t = 0
            optimizer = torch.optim.LBFGS([self._w, self._rho], lr=self.eta0,
                                           max_iter=max(self.max_iter, 100) if self.max_iter > 0 else 100,
                                           line_search_fn="strong_wolfe")
            def closure():
                optimizer.zero_grad()
                loss = self._compute_ocsvm_loss(X)
                loss.backward()
                return loss
            optimizer.step(closure)
            self.n_iter_ = 1
            self.coef_ = self._w.detach()
            self.intercept_ = torch.tensor([-self._rho.item()], dtype=self.dtype, device=self.device)
            self.offset_ = torch.tensor([0.0], dtype=self.dtype, device=self.device)
            self.fit_status_ = 0
            return self
        finally:
            self._fit_loss_fn = None


class MuonOneClassSVM(SGDOneClassSVM):
    """One-Class SVM trained with Muon (Momentum + Nesterov)."""
    def __init__(self, kernel='rbf', kernel_params=None, degree=3, gamma='scale',
                 coef0=0.0, tol=0.001, fit_intercept=True, shuffle=True,
                 random_state=None, learning_rate='invscaling', eta0=0.01,
                 power_t=0.5, warm_start=False, average=False, nu=0.5,
                 shrinking=True, cache_size=200, verbose=False, max_iter=-1,
                 momentum: float = 0.9,
                 device='cpu', dtype=torch.float, *args, **kwargs):
        super().__init__(kernel=kernel, kernel_params=kernel_params, degree=degree, gamma=gamma,
                         coef0=coef0, tol=tol, fit_intercept=fit_intercept, shuffle=shuffle,
                         random_state=random_state, learning_rate=learning_rate, eta0=eta0,
                         power_t=power_t, warm_start=warm_start, average=average, nu=nu,
                         shrinking=shrinking, cache_size=cache_size, verbose=verbose,
                         max_iter=max_iter, device=device, dtype=dtype, *args, **kwargs)
        self.momentum = momentum
        self._v_w = None
        self._v_rho = None

    def _init_state(self):
        if self._w is not None:
            self._v_w = torch.zeros_like(self._w)
            self._v_rho = torch.zeros_like(self._rho)

    def _update_params(self, X_batch):
        self._zero_grad()
        loss = self._compute_ocsvm_loss(X_batch)
        loss.backward()
        with torch.no_grad():
            if self._v_w is None:
                self._v_w = torch.zeros_like(self._w)
                self._v_rho = torch.zeros_like(self._rho)
            self._v_w.mul_(self.momentum).add_(self._w.grad)
            self._w.data.sub_(self._v_w, alpha=self.eta0)
            self._v_rho.mul_(self.momentum).add_(self._rho.grad)
            self._rho.data.sub_(self._v_rho, alpha=self.eta0)
        self.num_iter_t += 1
        return loss.item()


class NAdamOneClassSVM(AdamOneClassSVM):
    """One-Class SVM trained with NAdam (Nesterov Adam)."""
    def _update_params(self, X_batch):
        self._zero_grad()
        loss = self._compute_ocsvm_loss(X_batch)
        loss.backward()
        with torch.no_grad():
            if self._m_w is None:
                self._m_w = torch.zeros_like(self._w)
                self._v_w = torch.zeros_like(self._w)
                self._m_rho = torch.zeros_like(self._rho)
                self._v_rho = torch.zeros_like(self._rho)
            t = self.num_iter_t + 1
            self._m_w.mul_(self.beta1).add_(self._w.grad, alpha=1 - self.beta1)
            self._v_w.mul_(self.beta2).addcmul_(self._w.grad, self._w.grad, value=1 - self.beta2)
            m_hat = (self.beta1 * self._m_w / (1 - self.beta1 ** (t + 1)) +
                     (1 - self.beta1) * self._w.grad / (1 - self.beta1 ** t))
            v_hat = self._v_w / (1 - self.beta2 ** t)
            self._w.data.addcdiv_(m_hat, v_hat.sqrt().add_(self.smoothening_term), value=-self.eta0)
            self._m_rho.mul_(self.beta1).add_(self._rho.grad, alpha=1 - self.beta1)
            self._v_rho.mul_(self.beta2).addcmul_(self._rho.grad, self._rho.grad, value=1 - self.beta2)
            m_hat_r = (self.beta1 * self._m_rho / (1 - self.beta1 ** (t + 1)) +
                       (1 - self.beta1) * self._rho.grad / (1 - self.beta1 ** t))
            v_hat_r = self._v_rho / (1 - self.beta2 ** t)
            self._rho.data.addcdiv_(m_hat_r, v_hat_r.sqrt().add_(self.smoothening_term), value=-self.eta0)
        self.num_iter_t += 1
        return loss.item()


class RAdamOneClassSVM(AdamOneClassSVM):
    """One-Class SVM trained with RAdam (Rectified Adam)."""
    def _update_params(self, X_batch):
        self._zero_grad()
        loss = self._compute_ocsvm_loss(X_batch)
        loss.backward()
        with torch.no_grad():
            if self._m_w is None:
                self._m_w = torch.zeros_like(self._w)
                self._v_w = torch.zeros_like(self._w)
                self._m_rho = torch.zeros_like(self._rho)
                self._v_rho = torch.zeros_like(self._rho)
            t = float(self.num_iter_t + 1)
            rho_inf = 2.0 / (1 - self.beta2) - 1
            rho_t = rho_inf - 2 * t * (self.beta2 ** t) / (1 - self.beta2 ** t)
            self._m_w.mul_(self.beta1).add_(self._w.grad, alpha=1 - self.beta1)
            self._v_w.mul_(self.beta2).addcmul_(self._w.grad, self._w.grad, value=1 - self.beta2)
            if rho_t > 5:
                rect = math.sqrt(((rho_t - 4) * (rho_t - 2) * rho_inf) /
                                  ((rho_inf - 4) * (rho_inf - 2) * rho_t))
                m_hat = self._m_w / (1 - self.beta1 ** t)
                v_hat = (self._v_w / (1 - self.beta2 ** t)).sqrt().add_(self.smoothening_term)
                self._w.data.addcdiv_(m_hat, v_hat, value=-self.eta0 * rect)
            else:
                self._w.data.add_(self._m_w / (1 - self.beta1 ** t), alpha=-self.eta0)
            self._m_rho.mul_(self.beta1).add_(self._rho.grad, alpha=1 - self.beta1)
            self._v_rho.mul_(self.beta2).addcmul_(self._rho.grad, self._rho.grad, value=1 - self.beta2)
            if rho_t > 5:
                rect = math.sqrt(((rho_t - 4) * (rho_t - 2) * rho_inf) /
                                  ((rho_inf - 4) * (rho_inf - 2) * rho_t))
                m_hat_r = self._m_rho / (1 - self.beta1 ** t)
                v_hat_r = (self._v_rho / (1 - self.beta2 ** t)).sqrt().add_(self.smoothening_term)
                self._rho.data.addcdiv_(m_hat_r, v_hat_r, value=-self.eta0 * rect)
            else:
                self._rho.data.add_(self._m_rho / (1 - self.beta1 ** t), alpha=-self.eta0)
        self.num_iter_t += 1
        return loss.item()


class RpropOneClassSVM(SGDOneClassSVM):
    """One-Class SVM trained with Rprop."""
    def __init__(self, kernel='rbf', kernel_params=None, degree=3, gamma='scale',
                 coef0=0.0, tol=0.001, fit_intercept=True, shuffle=True,
                 random_state=None, learning_rate='invscaling', eta0=0.01,
                 power_t=0.5, warm_start=False, average=False, nu=0.5,
                 shrinking=True, cache_size=200, verbose=False, max_iter=-1,
                 etas: Tuple[float, float] = (0.5, 1.2),
                 step_sizes: Tuple[float, float] = (1e-6, 50.0),
                 device='cpu', dtype=torch.float, *args, **kwargs):
        super().__init__(kernel=kernel, kernel_params=kernel_params, degree=degree, gamma=gamma,
                         coef0=coef0, tol=tol, fit_intercept=fit_intercept, shuffle=shuffle,
                         random_state=random_state, learning_rate=learning_rate, eta0=eta0,
                         power_t=power_t, warm_start=warm_start, average=average, nu=nu,
                         shrinking=shrinking, cache_size=cache_size, verbose=verbose,
                         max_iter=max_iter, device=device, dtype=dtype, *args, **kwargs)
        self.etas = etas
        self.step_sizes = step_sizes
        self._prev_grad_w = None
        self._step_w = None
        self._prev_grad_rho = None
        self._step_rho = None

    def _init_state(self):
        if self._w is not None:
            self._prev_grad_w = torch.zeros_like(self._w)
            self._step_w = torch.full_like(self._w, 0.1)
            self._prev_grad_rho = torch.zeros_like(self._rho)
            self._step_rho = torch.full_like(self._rho, 0.1)

    def _update_params(self, X_batch):
        self._zero_grad()
        loss = self._compute_ocsvm_loss(X_batch)
        loss.backward()
        with torch.no_grad():
            if self._prev_grad_w is None:
                self._prev_grad_w = torch.zeros_like(self._w)
                self._step_w = torch.full_like(self._w, 0.1)
                self._prev_grad_rho = torch.zeros_like(self._rho)
                self._step_rho = torch.full_like(self._rho, 0.1)
            sign_w = torch.sign(self._w.grad * self._prev_grad_w)
            self._step_w = torch.where(sign_w > 0,
                                        (self._step_w * self.etas[1]).clamp(*self.step_sizes),
                            torch.where(sign_w < 0,
                                        (self._step_w * self.etas[0]).clamp(*self.step_sizes),
                                        self._step_w))
            self._w.data.sub_(torch.sign(self._w.grad) * self._step_w)
            self._prev_grad_w.copy_(self._w.grad)
            sign_r = torch.sign(self._rho.grad * self._prev_grad_rho)
            self._step_rho = torch.where(sign_r > 0,
                                          (self._step_rho * self.etas[1]).clamp(*self.step_sizes),
                              torch.where(sign_r < 0,
                                          (self._step_rho * self.etas[0]).clamp(*self.step_sizes),
                                          self._step_rho))
            self._rho.data.sub_(torch.sign(self._rho.grad) * self._step_rho)
            self._prev_grad_rho.copy_(self._rho.grad)
        self.num_iter_t += 1
        return loss.item()


def _make_scipy_ocsvm(method):
    """Factory for scipy-based One-Class SVM."""
    def fit(self, data_or_X, y=None, **kwargs):
        try:
            import scipy.optimize as opt
        except ImportError:
            raise ImportError("scipy required.")
        self._fit_loss_fn = kwargs.get('loss_fn', None)
        try:
            warm_start = kwargs.get('warm_start', self.warm_start)
            X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
            if X.dim() == 1:
                X = X.unsqueeze(0)
            n_samples, n_features = X.shape
            self.n_features_in_ = n_features
            self.shape_fit_ = tuple(X.shape)
            if not warm_start or self._w is None or self._w.shape[1] != n_features:
                self._init_params(n_features)
                self.num_iter_t = 0

            def func(p_flat):
                p_flat = np.atleast_1d(np.asarray(p_flat, dtype=np.float64)).ravel()
                with torch.no_grad():
                    p = torch.from_numpy(p_flat).to(self.device, self.dtype)
                    self._w.data.copy_(p[:self._w.numel()].view(self._w.shape))
                    self._rho.data.copy_(p[self._w.numel():self._w.numel() + 1])
                with torch.enable_grad():
                    for param in [self._w, self._rho]:
                        if param.grad is not None:
                            param.grad.zero_()
                    loss = self._compute_ocsvm_loss(X)
                    loss.backward()
                    grads = torch.cat([self._w.grad.flatten(), self._rho.grad.flatten()])
                return loss.item(), grads.detach().cpu().numpy().astype('float64')

            x0 = torch.cat([self._w.detach().flatten(),
                             self._rho.detach().flatten()]).cpu().numpy()
            x0 = np.asarray(x0, dtype=np.float64).ravel()
            maxiter = max(self.max_iter, 100) if self.max_iter > 0 else 100
            # trust-ncg and dogleg require Hessian; fall back to L-BFGS-B (gradient-only)
            opt_method = "L-BFGS-B" if method.lower() in ("trust-ncg", "dogleg") else method
            try:
                res = opt.minimize(func, x0, method=opt_method, jac=True, options={"maxiter": maxiter})
            except Exception:
                res = opt.minimize(lambda x: func(x)[0], x0, method=opt_method, options={"maxiter": maxiter})
            with torch.no_grad():
                res_x = np.asarray(res.x, dtype=np.float64).ravel()
                p = torch.from_numpy(res_x).to(self.device, self.dtype)
                self._w.data.copy_(p[:self._w.numel()].view(self._w.shape))
                self._rho.data.copy_(p[self._w.numel():self._w.numel() + 1])
            self.n_iter_ = getattr(res, 'nit', maxiter)
            self.coef_ = self._w.detach()
            self.intercept_ = torch.tensor([-self._rho.item()], dtype=self.dtype, device=self.device)
            self.offset_ = torch.tensor([0.0], dtype=self.dtype, device=self.device)
            self.fit_status_ = 0
            return self
        finally:
            self._fit_loss_fn = None
    return fit


class BFGSOneClassSVM(SGDOneClassSVM):
    """One-Class SVM optimized with BFGS."""
    fit = _make_scipy_ocsvm("BFGS")


class NewtonCGOneClassSVM(SGDOneClassSVM):
    """One-Class SVM optimized with Newton-CG."""
    fit = _make_scipy_ocsvm("Newton-CG")


class SLSQPOneClassSVM(SGDOneClassSVM):
    """One-Class SVM optimized with SLSQP."""
    fit = _make_scipy_ocsvm("SLSQP")


class PowellOneClassSVM(SGDOneClassSVM):
    """One-Class SVM optimized with Powell."""
    fit = _make_scipy_ocsvm("Powell")


class TNCOneClassSVM(SGDOneClassSVM):
    """One-Class SVM optimized with TNC."""
    fit = _make_scipy_ocsvm("TNC")


class TrustNCGOneClassSVM(SGDOneClassSVM):
    """One-Class SVM optimized with Trust-NCG."""
    fit = _make_scipy_ocsvm("trust-ncg")


class DoglegOneClassSVM(SGDOneClassSVM):
    """One-Class SVM optimized with Dogleg."""
    fit = _make_scipy_ocsvm("dogleg")


class CGOneClassSVM(SGDOneClassSVM):
    """One-Class SVM optimized with Conjugate Gradient."""
    fit = _make_scipy_ocsvm("CG")


class COBYLAOneClassSVM(SGDOneClassSVM):
    """One-Class SVM optimized with COBYLA."""
    fit = _make_scipy_ocsvm("COBYLA")


class NelderMeadOneClassSVM(SGDOneClassSVM):
    """One-Class SVM optimized with Nelder-Mead."""
    fit = _make_scipy_ocsvm("Nelder-Mead")


class LionOneClassSVM(SGDOneClassSVM):
    """One-Class SVM trained with Lion (sign momentum)."""
    def __init__(self, kernel='rbf', kernel_params=None, degree=3, gamma='scale',
                 coef0=0.0, tol=0.001, fit_intercept=True, shuffle=True,
                 random_state=None, learning_rate='invscaling', eta0=0.01,
                 power_t=0.5, warm_start=False, average=False, nu=0.5,
                 shrinking=True, cache_size=200, verbose=False, max_iter=-1,
                 beta1: float = 0.9, beta2: float = 0.99,
                 device='cpu', dtype=torch.float, *args, **kwargs):
        super().__init__(kernel=kernel, kernel_params=kernel_params, degree=degree, gamma=gamma,
                         coef0=coef0, tol=tol, fit_intercept=fit_intercept, shuffle=shuffle,
                         random_state=random_state, learning_rate=learning_rate, eta0=eta0,
                         power_t=power_t, warm_start=warm_start, average=average, nu=nu,
                         shrinking=shrinking, cache_size=cache_size, verbose=verbose,
                         max_iter=max_iter, device=device, dtype=dtype, *args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self._m_w = None
        self._m_rho = None

    def _init_state(self):
        if self._w is not None:
            self._m_w = torch.zeros_like(self._w)
            self._m_rho = torch.zeros_like(self._rho)

    def _update_params(self, X_batch):
        self._zero_grad()
        loss = self._compute_ocsvm_loss(X_batch)
        loss.backward()
        eta = self._get_eta()
        with torch.no_grad():
            if self._m_w is None:
                self._m_w = torch.zeros_like(self._w)
                self._m_rho = torch.zeros_like(self._rho)
            update_w = torch.sign(self.beta1 * self._m_w + (1 - self.beta1) * self._w.grad)
            self._w.data.sub_(update_w, alpha=eta)
            self._m_w.mul_(self.beta2).add_(self._w.grad, alpha=1 - self.beta2)
            update_r = torch.sign(self.beta1 * self._m_rho + (1 - self.beta1) * self._rho.grad)
            self._rho.data.sub_(update_r, alpha=eta)
            self._m_rho.mul_(self.beta2).add_(self._rho.grad, alpha=1 - self.beta2)
        self.num_iter_t += 1
        return loss.item()


class ShampooOneClassSVM(SGDOneClassSVM):
    """One-Class SVM trained with Shampoo (Kronecker preconditioner)."""
    def __init__(self, kernel='rbf', kernel_params=None, degree=3, gamma='scale',
                 coef0=0.0, tol=0.001, fit_intercept=True, shuffle=True,
                 random_state=None, learning_rate='invscaling', eta0=0.01,
                 power_t=0.5, warm_start=False, average=False, nu=0.5,
                 shrinking=True, cache_size=200, verbose=False, max_iter=-1,
                 update_freq: int = 1,
                 device='cpu', dtype=torch.float, *args, **kwargs):
        super().__init__(kernel=kernel, kernel_params=kernel_params, degree=degree, gamma=gamma,
                         coef0=coef0, tol=tol, fit_intercept=fit_intercept, shuffle=shuffle,
                         random_state=random_state, learning_rate=learning_rate, eta0=eta0,
                         power_t=power_t, warm_start=warm_start, average=average, nu=nu,
                         shrinking=shrinking, cache_size=cache_size, verbose=verbose,
                         max_iter=max_iter, device=device, dtype=dtype, *args, **kwargs)
        self.update_freq = update_freq
        self._L = None
        self._R = None

    def _update_params(self, X_batch):
        self._zero_grad()
        loss = self._compute_ocsvm_loss(X_batch)
        loss.backward()
        eta = self._get_eta()
        with torch.no_grad():
            G = self._w.grad
            if self._L is None:
                self._L = 1e-8 * torch.eye(G.shape[0], device=self.device, dtype=self.dtype)
                self._R = 1e-8 * torch.eye(G.shape[1], device=self.device, dtype=self.dtype)
            self._L.add_(G @ G.T)
            self._R.add_(G.T @ G)
            L_inv = torch.linalg.pinv(self._L)
            R_inv = torch.linalg.pinv(self._R)
            self._w.data.sub_(L_inv @ G @ R_inv, alpha=eta)
            self._rho.data.sub_(self._rho.grad, alpha=eta)
        self.num_iter_t += 1
        return loss.item()


class SophiaOneClassSVM(SGDOneClassSVM):
    """One-Class SVM trained with Sophia (Hessian-based optimizer)."""
    def __init__(self, kernel='rbf', kernel_params=None, degree=3, gamma='scale',
                 coef0=0.0, tol=0.001, fit_intercept=True, shuffle=True,
                 random_state=None, learning_rate='invscaling', eta0=0.01,
                 power_t=0.5, warm_start=False, average=False, nu=0.5,
                 shrinking=True, cache_size=200, verbose=False, max_iter=-1,
                 beta1: float = 0.9, beta2: float = 0.999, rho: float = 0.04,
                 update_freq: int = 10,
                 device='cpu', dtype=torch.float, *args, **kwargs):
        super().__init__(kernel=kernel, kernel_params=kernel_params, degree=degree, gamma=gamma,
                         coef0=coef0, tol=tol, fit_intercept=fit_intercept, shuffle=shuffle,
                         random_state=random_state, learning_rate=learning_rate, eta0=eta0,
                         power_t=power_t, warm_start=warm_start, average=average, nu=nu,
                         shrinking=shrinking, cache_size=cache_size, verbose=verbose,
                         max_iter=max_iter, device=device, dtype=dtype, *args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.rho = rho
        self.update_freq = update_freq
        self._m_w = None
        self._h_w = None
        self._m_rho = None
        self._h_rho = None

    def _update_params(self, X_batch):
        self._zero_grad()
        loss = self._compute_ocsvm_loss(X_batch)
        loss.backward()
        eta = self._get_eta()
        t = self.num_iter_t + 1
        with torch.no_grad():
            if self._m_w is None:
                self._m_w = torch.zeros_like(self._w)
                self._h_w = torch.ones_like(self._w)
                self._m_rho = torch.zeros_like(self._rho)
                self._h_rho = torch.ones_like(self._rho)
            self._m_w.mul_(self.beta1).add_(self._w.grad, alpha=1 - self.beta1)
            if t % self.update_freq == 0:
                self._h_w.mul_(self.beta2).addcmul_(self._w.grad, self._w.grad, value=1 - self.beta2)
            self._w.data.addcdiv_(self._m_w, self._h_w.clamp_min(self.rho), value=-eta)
            self._m_rho.mul_(self.beta1).add_(self._rho.grad, alpha=1 - self.beta1)
            if t % self.update_freq == 0:
                self._h_rho.mul_(self.beta2).addcmul_(self._rho.grad, self._rho.grad, value=1 - self.beta2)
            self._rho.data.addcdiv_(self._m_rho, self._h_rho.clamp_min(self.rho), value=-eta)
        self.num_iter_t += 1
        return loss.item()


class AdanOneClassSVM(SGDOneClassSVM):
    """One-Class SVM trained with Adan optimizer."""
    def __init__(self, kernel='rbf', kernel_params=None, degree=3, gamma='scale',
                 coef0=0.0, tol=0.001, fit_intercept=True, shuffle=True,
                 random_state=None, learning_rate='invscaling', eta0=0.01,
                 power_t=0.5, warm_start=False, average=False, nu=0.5,
                 shrinking=True, cache_size=200, verbose=False, max_iter=-1,
                 beta1: float = 0.98, beta2: float = 0.92, beta3: float = 0.99,
                 device='cpu', dtype=torch.float, *args, **kwargs):
        super().__init__(kernel=kernel, kernel_params=kernel_params, degree=degree, gamma=gamma,
                         coef0=coef0, tol=tol, fit_intercept=fit_intercept, shuffle=shuffle,
                         random_state=random_state, learning_rate=learning_rate, eta0=eta0,
                         power_t=power_t, warm_start=warm_start, average=average, nu=nu,
                         shrinking=shrinking, cache_size=cache_size, verbose=verbose,
                         max_iter=max_iter, device=device, dtype=dtype, *args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self._m_w = None
        self._v_w = None
        self._n_w = None
        self._prev_g_w = None

    def _update_params(self, X_batch):
        self._zero_grad()
        loss = self._compute_ocsvm_loss(X_batch)
        loss.backward()
        eta = self._get_eta()
        with torch.no_grad():
            g = self._w.grad.clone()
            if self._m_w is None:
                self._m_w = torch.zeros_like(self._w)
                self._v_w = torch.zeros_like(self._w)
                self._n_w = torch.zeros_like(self._w)
                self._prev_g_w = torch.zeros_like(self._w)
            dk = g + self.beta2 * (g - self._prev_g_w)
            self._m_w.mul_(self.beta1).add_(g, alpha=1 - self.beta1)
            self._v_w.mul_(self.beta2).add_(g - self._prev_g_w, alpha=1 - self.beta2)
            self._n_w.mul_(self.beta3).addcmul_(dk, dk, value=1 - self.beta3)
            update = (self._m_w + self.beta2 * self._v_w) / (self._n_w.sqrt() + 1e-8)
            self._w.data.sub_(update, alpha=eta)
            self._prev_g_w.copy_(g)
            self._rho.data.sub_(self._rho.grad, alpha=eta)
        self.num_iter_t += 1
        return loss.item()


class LAMBOneClassSVM(AdamOneClassSVM):
    """One-Class SVM trained with LAMB."""
    def __init__(self, kernel='rbf', kernel_params=None, degree=3, gamma='scale',
                 coef0=0.0, tol=0.001, fit_intercept=True, shuffle=True,
                 random_state=None, learning_rate='invscaling', eta0=0.01,
                 power_t=0.5, warm_start=False, average=False, nu=0.5,
                 shrinking=True, cache_size=200, verbose=False, max_iter=-1,
                 beta1: float = 0.9, beta2: float = 0.999, smoothening_term: float = 1e-6,
                 weight_decay: float = 0.01, trust_ratio: float = 1.0,
                 device='cpu', dtype=torch.float, *args, **kwargs):
        super().__init__(kernel=kernel, kernel_params=kernel_params, degree=degree, gamma=gamma,
                         coef0=coef0, tol=tol, fit_intercept=fit_intercept, shuffle=shuffle,
                         random_state=random_state, learning_rate=learning_rate, eta0=eta0,
                         power_t=power_t, warm_start=warm_start, average=average, nu=nu,
                         shrinking=shrinking, cache_size=cache_size, verbose=verbose,
                         max_iter=max_iter, beta1=beta1, beta2=beta2,
                         smoothening_term=smoothening_term,
                         device=device, dtype=dtype, *args, **kwargs)
        self.weight_decay = weight_decay
        self.trust_ratio = trust_ratio

    def _update_params(self, X_batch):
        self._zero_grad()
        loss = self._compute_ocsvm_loss(X_batch)
        loss.backward()
        eta = self._get_eta()
        with torch.no_grad():
            if self._m_w is None:
                self._m_w = torch.zeros_like(self._w)
                self._v_w = torch.zeros_like(self._w)
                self._m_rho = torch.zeros_like(self._rho)
                self._v_rho = torch.zeros_like(self._rho)
            t = self.num_iter_t + 1
            self._m_w.mul_(self.beta1).add_(self._w.grad, alpha=1 - self.beta1)
            self._v_w.mul_(self.beta2).addcmul_(self._w.grad, self._w.grad, value=1 - self.beta2)
            m_hat = self._m_w / (1 - self.beta1 ** t)
            v_hat = self._v_w / (1 - self.beta2 ** t)
            update = m_hat / (v_hat.sqrt() + self.smoothening_term) + self.weight_decay * self._w
            w_norm = self._w.norm().item()
            u_norm = update.norm().item()
            trust = min(w_norm / u_norm, self.trust_ratio) if u_norm > 0 else 1.0
            self._w.data.sub_(update * trust, alpha=eta)
            self._m_rho.mul_(self.beta1).add_(self._rho.grad, alpha=1 - self.beta1)
            self._v_rho.mul_(self.beta2).addcmul_(self._rho.grad, self._rho.grad, value=1 - self.beta2)
            m_hat_r = self._m_rho / (1 - self.beta1 ** t)
            v_hat_r = self._v_rho / (1 - self.beta2 ** t)
            self._rho.data.sub_(m_hat_r / (v_hat_r.sqrt() + self.smoothening_term), alpha=eta)
        self.num_iter_t += 1
        return loss.item()


class LookaheadOneClassSVM(SGDOneClassSVM):
    """One-Class SVM trained with Lookahead."""
    def __init__(self, kernel='rbf', kernel_params=None, degree=3, gamma='scale',
                 coef0=0.0, tol=0.001, fit_intercept=True, shuffle=True,
                 random_state=None, learning_rate='invscaling', eta0=0.01,
                 power_t=0.5, warm_start=False, average=False, nu=0.5,
                 shrinking=True, cache_size=200, verbose=False, max_iter=-1,
                 k: int = 5, alpha_la: float = 0.5,
                 device='cpu', dtype=torch.float, *args, **kwargs):
        super().__init__(kernel=kernel, kernel_params=kernel_params, degree=degree, gamma=gamma,
                         coef0=coef0, tol=tol, fit_intercept=fit_intercept, shuffle=shuffle,
                         random_state=random_state, learning_rate=learning_rate, eta0=eta0,
                         power_t=power_t, warm_start=warm_start, average=average, nu=nu,
                         shrinking=shrinking, cache_size=cache_size, verbose=verbose,
                         max_iter=max_iter, device=device, dtype=dtype, *args, **kwargs)
        self.k = k
        self.alpha_la = alpha_la
        self._slow_w = None
        self._slow_rho = None

    def _init_state(self):
        if self._w is not None:
            self._slow_w = self._w.data.clone()
            self._slow_rho = self._rho.data.clone()

    def _update_params(self, X_batch):
        self._zero_grad()
        loss = self._compute_ocsvm_loss(X_batch)
        loss.backward()
        eta = self._get_eta()
        with torch.no_grad():
            if self._slow_w is None:
                self._slow_w = self._w.data.clone()
                self._slow_rho = self._rho.data.clone()
            self._w.data.sub_(self._w.grad, alpha=eta)
            self._rho.data.sub_(self._rho.grad, alpha=eta)
            if (self.num_iter_t + 1) % self.k == 0:
                self._slow_w.add_(self._w.data - self._slow_w, alpha=self.alpha_la)
                self._slow_rho.add_(self._rho.data - self._slow_rho, alpha=self.alpha_la)
                self._w.data.copy_(self._slow_w)
                self._rho.data.copy_(self._slow_rho)
        self.num_iter_t += 1
        return loss.item()


class AdEMAMixOneClassSVM(AdamOneClassSVM):
    """One-Class SVM trained with AdEMAMix (slow + fast EMA mixture)."""
    def __init__(self, kernel='rbf', kernel_params=None, degree=3, gamma='scale',
                 coef0=0.0, tol=0.001, fit_intercept=True, shuffle=True,
                 random_state=None, learning_rate='invscaling', eta0=0.01,
                 power_t=0.5, warm_start=False, average=False, nu=0.5,
                 shrinking=True, cache_size=200, verbose=False, max_iter=-1,
                 beta1: float = 0.9, beta2: float = 0.999, smoothening_term: float = 1e-8,
                 alpha: float = 5.0, beta3: float = 0.9999, T_alpha_beta3: int = 1000,
                 device='cpu', dtype=torch.float, *args, **kwargs):
        super().__init__(kernel=kernel, kernel_params=kernel_params, degree=degree, gamma=gamma,
                         coef0=coef0, tol=tol, fit_intercept=fit_intercept, shuffle=shuffle,
                         random_state=random_state, learning_rate=learning_rate, eta0=eta0,
                         power_t=power_t, warm_start=warm_start, average=average, nu=nu,
                         shrinking=shrinking, cache_size=cache_size, verbose=verbose,
                         max_iter=max_iter, beta1=beta1, beta2=beta2,
                         smoothening_term=smoothening_term,
                         device=device, dtype=dtype, *args, **kwargs)
        self.alpha = alpha
        self.beta3 = beta3
        self.T_alpha_beta3 = T_alpha_beta3
        self._m2_w = None

    def _init_state(self):
        super()._init_state()
        if self._w is not None:
            self._m2_w = torch.zeros_like(self._w)

    def _update_params(self, X_batch):
        self._zero_grad()
        loss = self._compute_ocsvm_loss(X_batch)
        loss.backward()
        eta = self._get_eta()
        with torch.no_grad():
            if self._m_w is None:
                self._m_w = torch.zeros_like(self._w)
                self._v_w = torch.zeros_like(self._w)
                self._m_rho = torch.zeros_like(self._rho)
                self._v_rho = torch.zeros_like(self._rho)
                self._m2_w = torch.zeros_like(self._w)
            t = self.num_iter_t + 1
            self._m_w.mul_(self.beta1).add_(self._w.grad, alpha=1 - self.beta1)
            self._m2_w.mul_(self.beta3).add_(self._w.grad, alpha=1 - self.beta3)
            self._v_w.mul_(self.beta2).addcmul_(self._w.grad, self._w.grad, value=1 - self.beta2)
            alpha_t = min(self.alpha, t / max(self.T_alpha_beta3, 1) * self.alpha)
            m_mix = self._m_w + alpha_t * self._m2_w
            v_hat = self._v_w / (1 - self.beta2 ** t)
            self._w.data.addcdiv_(m_mix, v_hat.sqrt().add_(self.smoothening_term), value=-eta)
            self._m_rho.mul_(self.beta1).add_(self._rho.grad, alpha=1 - self.beta1)
            self._v_rho.mul_(self.beta2).addcmul_(self._rho.grad, self._rho.grad, value=1 - self.beta2)
            m_hat_r = self._m_rho / (1 - self.beta1 ** t)
            v_hat_r = self._v_rho / (1 - self.beta2 ** t)
            self._rho.data.addcdiv_(m_hat_r, v_hat_r.sqrt().add_(self.smoothening_term), value=-eta)
        self.num_iter_t += 1
        return loss.item()


class ScheduleFreeOneClassSVM(SGDOneClassSVM):
    """One-Class SVM trained with Schedule-Free SGD."""
    def __init__(self, kernel='rbf', kernel_params=None, degree=3, gamma='scale',
                 coef0=0.0, tol=0.001, fit_intercept=True, shuffle=True,
                 random_state=None, learning_rate='constant', eta0=0.01,
                 power_t=0.5, warm_start=False, average=False, nu=0.5,
                 shrinking=True, cache_size=200, verbose=False, max_iter=-1,
                 beta: float = 0.9, r: float = 0.0, warmup_steps: int = 0,
                 device='cpu', dtype=torch.float, *args, **kwargs):
        super().__init__(kernel=kernel, kernel_params=kernel_params, degree=degree, gamma=gamma,
                         coef0=coef0, tol=tol, fit_intercept=fit_intercept, shuffle=shuffle,
                         random_state=random_state, learning_rate=learning_rate, eta0=eta0,
                         power_t=power_t, warm_start=warm_start, average=average, nu=nu,
                         shrinking=shrinking, cache_size=cache_size, verbose=verbose,
                         max_iter=max_iter, device=device, dtype=dtype, *args, **kwargs)
        self.beta = beta
        self.r = r
        self.warmup_steps = warmup_steps
        self._z_w = None
        self._z_rho = None

    def _update_params(self, X_batch):
        self._zero_grad()
        loss = self._compute_ocsvm_loss(X_batch)
        loss.backward()
        t = self.num_iter_t + 1
        lr = self.eta0 * min(1.0, t / max(self.warmup_steps, 1))
        with torch.no_grad():
            if self._z_w is None:
                self._z_w = self._w.data.clone()
                self._z_rho = self._rho.data.clone()
            self._z_w.sub_(self._w.grad, alpha=lr)
            c = (1 - self.beta) / (1 - self.beta ** t)
            self._w.data.mul_(1 - c).add_(self._z_w, alpha=c)
            self._z_rho.sub_(self._rho.grad, alpha=lr)
            self._rho.data.mul_(1 - c).add_(self._z_rho, alpha=c)
        self.num_iter_t += 1
        return loss.item()


class MARSOneClassSVM(AdamOneClassSVM):
    """One-Class SVM trained with MARS optimizer."""
    def __init__(self, kernel='rbf', kernel_params=None, degree=3, gamma='scale',
                 coef0=0.0, tol=0.001, fit_intercept=True, shuffle=True,
                 random_state=None, learning_rate='invscaling', eta0=0.01,
                 power_t=0.5, warm_start=False, average=False, nu=0.5,
                 shrinking=True, cache_size=200, verbose=False, max_iter=-1,
                 beta1: float = 0.9, beta2: float = 0.99, smoothening_term: float = 1e-8,
                 gamma_mars: float = 0.025,
                 device='cpu', dtype=torch.float, *args, **kwargs):
        super().__init__(kernel=kernel, kernel_params=kernel_params, degree=degree, gamma=gamma,
                         coef0=coef0, tol=tol, fit_intercept=fit_intercept, shuffle=shuffle,
                         random_state=random_state, learning_rate=learning_rate, eta0=eta0,
                         power_t=power_t, warm_start=warm_start, average=average, nu=nu,
                         shrinking=shrinking, cache_size=cache_size, verbose=verbose,
                         max_iter=max_iter, beta1=beta1, beta2=beta2,
                         smoothening_term=smoothening_term,
                         device=device, dtype=dtype, *args, **kwargs)
        self.gamma_mars = gamma_mars
        self._prev_g_w = None

    def _update_params(self, X_batch):
        self._zero_grad()
        loss = self._compute_ocsvm_loss(X_batch)
        loss.backward()
        eta = self._get_eta()
        with torch.no_grad():
            g = self._w.grad.clone()
            if self._m_w is None:
                self._m_w = torch.zeros_like(self._w)
                self._v_w = torch.zeros_like(self._w)
                self._m_rho = torch.zeros_like(self._rho)
                self._v_rho = torch.zeros_like(self._rho)
            if self._prev_g_w is None:
                self._prev_g_w = torch.zeros_like(self._w)
            c_t = g + self.gamma_mars * (g - self._prev_g_w)
            self._m_w.mul_(self.beta1).add_(c_t, alpha=1 - self.beta1)
            self._v_w.mul_(self.beta2).addcmul_(c_t, c_t, value=1 - self.beta2)
            t = self.num_iter_t + 1
            m_hat = self._m_w / (1 - self.beta1 ** t)
            v_hat = self._v_w / (1 - self.beta2 ** t)
            self._w.data.addcdiv_(m_hat, v_hat.sqrt().add_(self.smoothening_term), value=-eta)
            self._prev_g_w.copy_(g)
            self._m_rho.mul_(self.beta1).add_(self._rho.grad, alpha=1 - self.beta1)
            self._v_rho.mul_(self.beta2).addcmul_(self._rho.grad, self._rho.grad, value=1 - self.beta2)
            m_hat_r = self._m_rho / (1 - self.beta1 ** t)
            v_hat_r = self._v_rho / (1 - self.beta2 ** t)
            self._rho.data.addcdiv_(m_hat_r, v_hat_r.sqrt().add_(self.smoothening_term), value=-eta)
        self.num_iter_t += 1
        return loss.item()


class TRPOOneClassSVM(SGDOneClassSVM):
    """One-Class SVM trained with TRPO-style natural gradient update."""
    def __init__(self, kernel='rbf', kernel_params=None, degree=3, gamma='scale',
                 coef0=0.0, tol=0.001, fit_intercept=True, shuffle=True,
                 random_state=None, learning_rate='constant', eta0=0.01,
                 power_t=0.5, warm_start=False, average=False, nu=0.5,
                 shrinking=True, cache_size=200, verbose=False, max_iter=-1,
                 max_kl: float = 0.01, damping: float = 0.1,
                 device='cpu', dtype=torch.float, *args, **kwargs):
        super().__init__(kernel=kernel, kernel_params=kernel_params, degree=degree, gamma=gamma,
                         coef0=coef0, tol=tol, fit_intercept=fit_intercept, shuffle=shuffle,
                         random_state=random_state, learning_rate=learning_rate, eta0=eta0,
                         power_t=power_t, warm_start=warm_start, average=average, nu=nu,
                         shrinking=shrinking, cache_size=cache_size, verbose=verbose,
                         max_iter=max_iter, device=device, dtype=dtype, *args, **kwargs)
        self.max_kl = max_kl
        self.damping = damping

    def _update_params(self, X_batch):
        self._zero_grad()
        loss = self._compute_ocsvm_loss(X_batch)
        loss.backward()
        with torch.no_grad():
            g = self._w.grad
            gnorm = g.norm().item() + 1e-8
            step_size = min(math.sqrt(2 * self.max_kl) / gnorm, self.eta0)
            self._w.data.sub_(g / gnorm, alpha=step_size)
            self._rho.data.sub_(self._rho.grad, alpha=self.eta0)
        self.num_iter_t += 1
        return loss.item()


class MadgradOneClassSVM(SGDOneClassSVM):
    """One-Class SVM trained with MADGRAD optimizer."""
    def __init__(self, kernel='rbf', kernel_params=None, degree=3, gamma='scale',
                 coef0=0.0, tol=0.001, fit_intercept=True, shuffle=True,
                 random_state=None, learning_rate='constant', eta0=0.01,
                 power_t=0.5, warm_start=False, average=False, nu=0.5,
                 shrinking=True, cache_size=200, verbose=False, max_iter=-1,
                 eps: float = 1e-6, momentum_madgrad: float = 0.9, weight_decay: float = 0.0,
                 device='cpu', dtype=torch.float, *args, **kwargs):
        super().__init__(kernel=kernel, kernel_params=kernel_params, degree=degree, gamma=gamma,
                         coef0=coef0, tol=tol, fit_intercept=fit_intercept, shuffle=shuffle,
                         random_state=random_state, learning_rate=learning_rate, eta0=eta0,
                         power_t=power_t, warm_start=warm_start, average=average, nu=nu,
                         shrinking=shrinking, cache_size=cache_size, verbose=verbose,
                         max_iter=max_iter, device=device, dtype=dtype, *args, **kwargs)
        self.eps = eps
        self.momentum_madgrad = momentum_madgrad
        self.weight_decay = weight_decay
        self._s_w = None
        self._x0_w = None

    def _update_params(self, X_batch):
        self._zero_grad()
        loss = self._compute_ocsvm_loss(X_batch)
        loss.backward()
        t = self.num_iter_t + 1
        lr = self.eta0
        with torch.no_grad():
            if self._s_w is None:
                self._s_w = torch.zeros_like(self._w)
                self._x0_w = self._w.data.clone()
            g = self._w.grad.clone()
            if self.weight_decay != 0:
                g = g + self.weight_decay * self._w
            self._s_w.add_(g.pow(2), alpha=(t ** (1 / 3)) * (lr ** 2))
            rms = self._s_w.pow(1.0 / 3).add_(self.eps)
            z = self._x0_w - (self._s_w.pow(0.5) / rms) * g.sign() * lr
            self._w.data.mul_(1 - self.momentum_madgrad).add_(z, alpha=self.momentum_madgrad)
            self._rho.data.sub_(self._rho.grad, alpha=lr)
        self.num_iter_t += 1
        return loss.item()


class YogiOneClassSVM(AdamOneClassSVM):
    """One-Class SVM trained with Yogi optimizer."""
    def _update_params(self, X_batch):
        self._zero_grad()
        loss = self._compute_ocsvm_loss(X_batch)
        loss.backward()
        eta = self._get_eta()
        with torch.no_grad():
            if self._m_w is None:
                self._m_w = torch.zeros_like(self._w)
                self._v_w = torch.zeros_like(self._w)
                self._m_rho = torch.zeros_like(self._rho)
                self._v_rho = torch.zeros_like(self._rho)
            t = self.num_iter_t + 1
            self._m_w.mul_(self.beta1).add_(self._w.grad, alpha=1 - self.beta1)
            g2_w = self._w.grad.pow(2)
            self._v_w.add_(torch.sign(g2_w - self._v_w) * (1 - self.beta2) * g2_w)
            m_hat = self._m_w / (1 - self.beta1 ** t)
            v_hat = self._v_w / (1 - self.beta2 ** t)
            self._w.data.addcdiv_(m_hat, v_hat.sqrt().add_(self.smoothening_term), value=-eta)
            self._m_rho.mul_(self.beta1).add_(self._rho.grad, alpha=1 - self.beta1)
            g2_r = self._rho.grad.pow(2)
            self._v_rho.add_(torch.sign(g2_r - self._v_rho) * (1 - self.beta2) * g2_r)
            m_hat_r = self._m_rho / (1 - self.beta1 ** t)
            v_hat_r = self._v_rho / (1 - self.beta2 ** t)
            self._rho.data.addcdiv_(m_hat_r, v_hat_r.sqrt().add_(self.smoothening_term), value=-eta)
        self.num_iter_t += 1
        return loss.item()


class LARSOneClassSVM(SGDOneClassSVM):
    """One-Class SVM trained with LARS (Layer-wise Adaptive Rate Scaling)."""
    def __init__(self, kernel='rbf', kernel_params=None, degree=3, gamma='scale',
                 coef0=0.0, tol=0.001, fit_intercept=True, shuffle=True,
                 random_state=None, learning_rate='constant', eta0=0.01,
                 power_t=0.5, warm_start=False, average=False, nu=0.5,
                 shrinking=True, cache_size=200, verbose=False, max_iter=-1,
                 weight_decay: float = 0.0, eta_lars: float = 0.001, momentum_lars: float = 0.9,
                 device='cpu', dtype=torch.float, *args, **kwargs):
        super().__init__(kernel=kernel, kernel_params=kernel_params, degree=degree, gamma=gamma,
                         coef0=coef0, tol=tol, fit_intercept=fit_intercept, shuffle=shuffle,
                         random_state=random_state, learning_rate=learning_rate, eta0=eta0,
                         power_t=power_t, warm_start=warm_start, average=average, nu=nu,
                         shrinking=shrinking, cache_size=cache_size, verbose=verbose,
                         max_iter=max_iter, device=device, dtype=dtype, *args, **kwargs)
        self.weight_decay = weight_decay
        self.eta_lars = eta_lars
        self.momentum_lars = momentum_lars
        self._v_w = None

    def _update_params(self, X_batch):
        self._zero_grad()
        loss = self._compute_ocsvm_loss(X_batch)
        loss.backward()
        with torch.no_grad():
            if self._v_w is None:
                self._v_w = torch.zeros_like(self._w)
            g = self._w.grad + self.weight_decay * self._w
            w_norm = self._w.norm().item()
            g_norm = g.norm().item() + 1e-8
            local_lr = self.eta_lars * w_norm / g_norm if w_norm > 0 else self.eta0
            self._v_w.mul_(self.momentum_lars).add_(g, alpha=local_lr)
            self._w.data.sub_(self._v_w)
            self._rho.data.sub_(self._rho.grad, alpha=self.eta0)
        self.num_iter_t += 1
        return loss.item()


class DAdaptationOneClassSVM(SGDOneClassSVM):
    """One-Class SVM trained with D-Adaptation (parameter-free learning rate)."""
    def __init__(self, kernel='rbf', kernel_params=None, degree=3, gamma='scale',
                 coef0=0.0, tol=0.001, fit_intercept=True, shuffle=True,
                 random_state=None, learning_rate='constant', eta0=1.0,
                 power_t=0.5, warm_start=False, average=False, nu=0.5,
                 shrinking=True, cache_size=200, verbose=False, max_iter=-1,
                 growth_rate: float = float('inf'), decouple_lr: bool = False,
                 device='cpu', dtype=torch.float, *args, **kwargs):
        super().__init__(kernel=kernel, kernel_params=kernel_params, degree=degree, gamma=gamma,
                         coef0=coef0, tol=tol, fit_intercept=fit_intercept, shuffle=shuffle,
                         random_state=random_state, learning_rate=learning_rate, eta0=eta0,
                         power_t=power_t, warm_start=warm_start, average=average, nu=nu,
                         shrinking=shrinking, cache_size=cache_size, verbose=verbose,
                         max_iter=max_iter, device=device, dtype=dtype, *args, **kwargs)
        self.growth_rate = growth_rate
        self.decouple_lr = decouple_lr
        self._d = 1e-6
        self._g_sq_sum = None
        self._s_w = None

    def _update_params(self, X_batch):
        self._zero_grad()
        loss = self._compute_ocsvm_loss(X_batch)
        loss.backward()
        with torch.no_grad():
            g = self._w.grad
            if self._g_sq_sum is None:
                self._g_sq_sum = torch.zeros(1, device=self.device, dtype=self.dtype)
                self._s_w = torch.zeros_like(self._w)
            self._g_sq_sum.add_(g.pow(2).sum())
            self._s_w.add_(g * self._d)
            d_new = (self._s_w.pow(2).sum() / (self._g_sq_sum + 1e-8)).sqrt().item()
            d_new = min(d_new, self._d * self.growth_rate)
            self._d = max(self._d, d_new)
            self._w.data.sub_(g * self._d * self.eta0 / (self._g_sq_sum.sqrt() + 1e-8))
            self._rho.data.sub_(self._rho.grad, alpha=self.eta0)
        self.num_iter_t += 1
        return loss.item()


class SignSGDOneClassSVM(SGDOneClassSVM):
    """One-Class SVM trained with SignSGD (sign of gradient)."""
    def _update_params(self, X_batch):
        self._zero_grad()
        loss = self._compute_ocsvm_loss(X_batch)
        loss.backward()
        eta = self._get_eta()
        with torch.no_grad():
            self._w.data.sub_(torch.sign(self._w.grad), alpha=eta)
            self._rho.data.sub_(torch.sign(self._rho.grad), alpha=eta)
        self.num_iter_t += 1
        return loss.item()


class ProdigyOneClassSVM(SGDOneClassSVM):
    """One-Class SVM trained with Prodigy (adaptive learning rate)."""
    def __init__(self, kernel='rbf', kernel_params=None, degree=3, gamma='scale',
                 coef0=0.0, tol=0.001, fit_intercept=True, shuffle=True,
                 random_state=None, learning_rate='constant', eta0=1.0,
                 power_t=0.5, warm_start=False, average=False, nu=0.5,
                 shrinking=True, cache_size=200, verbose=False, max_iter=-1,
                 beta1: float = 0.9, beta2: float = 0.999, smoothening_term: float = 1e-8,
                 growth_rate: float = float('inf'),
                 device='cpu', dtype=torch.float, *args, **kwargs):
        super().__init__(kernel=kernel, kernel_params=kernel_params, degree=degree, gamma=gamma,
                         coef0=coef0, tol=tol, fit_intercept=fit_intercept, shuffle=shuffle,
                         random_state=random_state, learning_rate=learning_rate, eta0=eta0,
                         power_t=power_t, warm_start=warm_start, average=average, nu=nu,
                         shrinking=shrinking, cache_size=cache_size, verbose=verbose,
                         max_iter=max_iter, device=device, dtype=dtype, *args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.smoothening_term = smoothening_term
        self.growth_rate = growth_rate
        self._d = 1e-6
        self._m_w = None
        self._v_w = None
        self._s_w = None

    def _update_params(self, X_batch):
        self._zero_grad()
        loss = self._compute_ocsvm_loss(X_batch)
        loss.backward()
        with torch.no_grad():
            g = self._w.grad
            if self._m_w is None:
                self._m_w = torch.zeros_like(self._w)
                self._v_w = torch.zeros_like(self._w)
                self._s_w = torch.zeros_like(self._w)
            t = self.num_iter_t + 1
            self._s_w.add_(g * self._d)
            self._v_w.mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
            d_hat = (self._s_w.norm() / (self._v_w.sqrt() + self.smoothening_term).norm()).item()
            self._d = min(max(self._d, d_hat), self._d * self.growth_rate)
            self._m_w.mul_(self.beta1).add_(g, alpha=(1 - self.beta1) * self._d)
            m_hat = self._m_w / (1 - self.beta1 ** t)
            v_hat = self._v_w / (1 - self.beta2 ** t)
            self._w.data.addcdiv_(m_hat, v_hat.sqrt().add_(self.smoothening_term), value=-self.eta0 * self._d)
            self._rho.data.sub_(self._rho.grad, alpha=self.eta0)
        self.num_iter_t += 1
        return loss.item()


class QNSVRGOneClassSVM(SGDOneClassSVM):
    """One-Class SVM trained with QN-SVRG (Quasi-Newton SVRG)."""
    def __init__(self, kernel='rbf', kernel_params=None, degree=3, gamma='scale',
                 coef0=0.0, tol=0.001, fit_intercept=True, shuffle=True,
                 random_state=None, learning_rate='constant', eta0=0.01,
                 power_t=0.5, warm_start=False, average=False, nu=0.5,
                 shrinking=True, cache_size=200, verbose=False, max_iter=-1,
                 memory: int = 10,
                 device='cpu', dtype=torch.float, *args, **kwargs):
        super().__init__(kernel=kernel, kernel_params=kernel_params, degree=degree, gamma=gamma,
                         coef0=coef0, tol=tol, fit_intercept=fit_intercept, shuffle=shuffle,
                         random_state=random_state, learning_rate=learning_rate, eta0=eta0,
                         power_t=power_t, warm_start=warm_start, average=average, nu=nu,
                         shrinking=shrinking, cache_size=cache_size, verbose=verbose,
                         max_iter=max_iter, device=device, dtype=dtype, *args, **kwargs)
        self.memory = memory
        self._s_hist = []
        self._y_hist = []
        self._prev_w = None
        self._prev_g_w = None

    def _update_params(self, X_batch):
        self._zero_grad()
        loss = self._compute_ocsvm_loss(X_batch)
        loss.backward()
        eta = self._get_eta()
        with torch.no_grad():
            g = self._w.grad.clone()
            if self._prev_w is not None and self._prev_g_w is not None:
                s = (self._w.data - self._prev_w).flatten()
                y = (g - self._prev_g_w).flatten()
                if y.dot(s).item() > 1e-10:
                    self._s_hist.append(s.clone())
                    self._y_hist.append(y.clone())
                    if len(self._s_hist) > self.memory:
                        self._s_hist.pop(0)
                        self._y_hist.pop(0)
            q = g.flatten().clone()
            alphas = []
            for s, y in zip(reversed(self._s_hist), reversed(self._y_hist)):
                rho_i = 1.0 / (y.dot(s).item() + 1e-10)
                a = rho_i * s.dot(q).item()
                q.sub_(y * a)
                alphas.append((rho_i, a, s, y))
            r = q * eta
            for rho_i, a, s, y in reversed(alphas):
                b = rho_i * y.dot(r).item()
                r.add_(s * (a - b))
            self._prev_w = self._w.data.clone()
            self._prev_g_w = g.clone()
            self._w.data -= r.view(self._w.shape)
            self._rho.data.sub_(self._rho.grad, alpha=eta)
        self.num_iter_t += 1
        return loss.item()


def _make_global_ocsvm(method):
    """Factory for global optimization One-Class SVM variants."""
    def fit(self, data_or_X, y=None, **kwargs):
        try:
            import scipy.optimize as opt
        except ImportError:
            raise ImportError("scipy required.")
        warm_start = kwargs.get('warm_start', self.warm_start)
        X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self.shape_fit_ = tuple(X.shape)
        if not warm_start or self._w is None or self._w.shape[1] != n_features:
            self._init_params(n_features)
            self.num_iter_t = 0

        def func(p_flat):
            arr = np.atleast_1d(np.asarray(p_flat, dtype=np.float64))
            # differential_evolution can pass (n_population, n_params) for vectorized; we only support 1D
            if arr.ndim > 1:
                # Evaluate each row (each candidate) and return a scalar for the first, or full array for vectorized
                losses = []
                for row in arr:
                    p_flat_1d = row.ravel()
                    with torch.no_grad():
                        p = torch.from_numpy(p_flat_1d).to(self.device, self.dtype)
                        self._w.data.copy_(p[:self._w.numel()].view(self._w.shape))
                        self._rho.data.copy_(p[self._w.numel():self._w.numel() + 1].view(self._rho.shape))
                        losses.append(self._compute_ocsvm_loss(X).item())
                return np.array(losses, dtype=np.float64)
            p_flat_1d = arr.ravel()
            with torch.no_grad():
                p = torch.from_numpy(p_flat_1d).to(self.device, self.dtype)
                self._w.data.copy_(p[:self._w.numel()].view(self._w.shape))
                self._rho.data.copy_(p[self._w.numel():self._w.numel() + 1].view(self._rho.shape))
                return self._compute_ocsvm_loss(X).item()

        x0 = np.asarray(torch.cat([self._w.detach().flatten(),
                         self._rho.detach().flatten()]).cpu().numpy(), dtype=np.float64).ravel()
        bounds = [(-5, 5)] * len(x0)
        if method == "differential_evolution":
            res = opt.differential_evolution(func, bounds)
        elif method == "basinhopping":
            res = opt.basinhopping(func, x0)
        elif method == "dual_annealing":
            res = opt.dual_annealing(func, bounds, x0=x0)
        elif method == "shgo":
            res = opt.shgo(func, bounds)
        else:
            res = opt.minimize(func, x0, method=method)
        with torch.no_grad():
            res_x = np.atleast_1d(np.asarray(res.x, dtype=np.float64)).ravel()
            expected_len = self._w.numel() + 1
            if len(res_x) != expected_len:
                # Handle optimizers that may return different shapes (e.g. diff_evolution)
                res_x = res_x[:expected_len] if len(res_x) >= expected_len else np.pad(
                    res_x, (0, expected_len - len(res_x)), mode='edge'
                )
            p = torch.from_numpy(res_x).to(self.device, self.dtype)
            if p.dim() > 1:
                p = p.flatten()
            self._w.data.copy_(p[:self._w.numel()].view(self._w.shape))
            self._rho.data.copy_(p[self._w.numel():self._w.numel() + 1].view(self._rho.shape))
        self.n_iter_ = getattr(res, 'nit', 1)
        self.coef_ = self._w.detach()
        self.intercept_ = torch.tensor([-self._rho.item()], dtype=self.dtype, device=self.device)
        self.offset_ = torch.tensor([0.0], dtype=self.dtype, device=self.device)
        self.fit_status_ = 0
        return self
    return fit


class DifferentialEvolutionOneClassSVM(SGDOneClassSVM):
    """One-Class SVM optimized with Differential Evolution."""
    fit = _make_global_ocsvm("differential_evolution")


class BasinhoppingOneClassSVM(SGDOneClassSVM):
    """One-Class SVM optimized with Basin-hopping."""
    fit = _make_global_ocsvm("basinhopping")


class DualAnnealingOneClassSVM(SGDOneClassSVM):
    """One-Class SVM optimized with Dual Annealing."""
    fit = _make_global_ocsvm("dual_annealing")


class SHGOOneClassSVM(SGDOneClassSVM):
    """One-Class SVM optimized with SHGO."""
    fit = _make_global_ocsvm("shgo")


class CMAESOneClassSVM(SGDOneClassSVM):
    """One-Class SVM optimized with CMA-ES."""
    def __init__(self, kernel='rbf', kernel_params=None, degree=3, gamma='scale',
                 coef0=0.0, tol=0.001, fit_intercept=True, shuffle=True,
                 random_state=None, learning_rate='constant', eta0=0.01,
                 power_t=0.5, warm_start=False, average=False, nu=0.5,
                 shrinking=True, cache_size=200, verbose=False, max_iter=-1,
                 sigma0: float = 0.3, popsize: int = 10,
                 device='cpu', dtype=torch.float, *args, **kwargs):
        super().__init__(kernel=kernel, kernel_params=kernel_params, degree=degree, gamma=gamma,
                         coef0=coef0, tol=tol, fit_intercept=fit_intercept, shuffle=shuffle,
                         random_state=random_state, learning_rate=learning_rate, eta0=eta0,
                         power_t=power_t, warm_start=warm_start, average=average, nu=nu,
                         shrinking=shrinking, cache_size=cache_size, verbose=verbose,
                         max_iter=max_iter, device=device, dtype=dtype, *args, **kwargs)
        self.sigma0 = sigma0
        self.popsize = popsize

    def fit(self, data_or_X, y=None, **kwargs):
        warm_start = kwargs.get('warm_start', self.warm_start)
        X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self.shape_fit_ = tuple(X.shape)
        if not warm_start or self._w is None or self._w.shape[1] != n_features:
            self._init_params(n_features)
        dim = self._w.numel() + 1
        max_iter = max(self.max_iter, 200) if self.max_iter > 0 else 200
        mean = torch.cat([self._w.detach().flatten(), self._rho.detach().flatten()])
        C = torch.eye(dim, device=self.device, dtype=self.dtype)
        for it in range(max_iter):
            try:
                L = torch.linalg.cholesky(C + 1e-8 * torch.eye(dim, device=self.device, dtype=self.dtype))
            except Exception:
                L = torch.eye(dim, device=self.device, dtype=self.dtype)
            eps = self.sigma0 * (torch.randn(self.popsize, dim, device=self.device, dtype=self.dtype) @ L.T)
            samples = mean.unsqueeze(0) + eps
            losses = []
            for s in samples:
                with torch.no_grad():
                    self._w.data.copy_(s[:self._w.numel()].view(self._w.shape))
                    self._rho.data.copy_(s[self._w.numel():self._w.numel() + 1])
                    losses.append(self._compute_ocsvm_loss(X).item())
            losses_t = torch.tensor(losses, device=self.device, dtype=self.dtype)
            top_k = max(self.popsize // 2, 1)
            idx = losses_t.argsort()[:top_k]
            elite = eps[idx]
            mean = samples[idx].mean(0)
            C = (elite.T @ elite) / top_k + 1e-8 * torch.eye(dim, device=self.device, dtype=self.dtype)
        with torch.no_grad():
            self._w.data.copy_(mean[:self._w.numel()].view(self._w.shape))
            self._rho.data.copy_(mean[self._w.numel():self._w.numel() + 1])
        self.n_iter_ = it + 1
        self.coef_ = self._w.detach()
        self.intercept_ = torch.tensor([-self._rho.item()], dtype=self.dtype, device=self.device)
        self.offset_ = torch.tensor([0.0], dtype=self.dtype, device=self.device)
        self.fit_status_ = 0
        return self


class BayesianOptimizationOneClassSVM(SGDOneClassSVM):
    """One-Class SVM optimized with Bayesian Optimization."""
    def __init__(self, kernel='rbf', kernel_params=None, degree=3, gamma='scale',
                 coef0=0.0, tol=0.001, fit_intercept=True, shuffle=True,
                 random_state=None, learning_rate='constant', eta0=0.01,
                 power_t=0.5, warm_start=False, average=False, nu=0.5,
                 shrinking=True, cache_size=200, verbose=False, max_iter=-1,
                 n_calls: int = 20, n_random_starts: int = 5,
                 device='cpu', dtype=torch.float, *args, **kwargs):
        super().__init__(kernel=kernel, kernel_params=kernel_params, degree=degree, gamma=gamma,
                         coef0=coef0, tol=tol, fit_intercept=fit_intercept, shuffle=shuffle,
                         random_state=random_state, learning_rate=learning_rate, eta0=eta0,
                         power_t=power_t, warm_start=warm_start, average=average, nu=nu,
                         shrinking=shrinking, cache_size=cache_size, verbose=verbose,
                         max_iter=max_iter, device=device, dtype=dtype, *args, **kwargs)
        self.n_calls = n_calls
        self.n_random_starts = n_random_starts

    def fit(self, data_or_X, y=None, **kwargs):
        warm_start = kwargs.get('warm_start', self.warm_start)
        X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(1)  # (n,) -> (n, 1) for single feature
        if X.dim() != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self.shape_fit_ = tuple(X.shape)
        if not warm_start or self._w is None or self._w.shape[1] != n_features:
            self._init_params(n_features)
        dim = self._w.numel() + 1
        gen = torch.Generator(device=self.device)
        if self.random_state is not None:
            gen.manual_seed(int(self.random_state) if isinstance(self.random_state, int) else 42)
        best_loss = float('inf')
        best_params = torch.cat([self._w.detach().flatten(), self._rho.detach().flatten()])

        def evaluate(params):
            with torch.no_grad():
                self._w.data.copy_(params[:self._w.numel()].reshape(self._w.shape))
                if self._rho is not None:
                    self._rho.data.copy_(params[self._w.numel():self._w.numel() + 1].reshape(self._rho.data.shape))
                return self._compute_ocsvm_loss(X).item()

        for _ in range(self.n_random_starts):
            p = torch.randn(dim, device=self.device, dtype=self.dtype, generator=gen)
            l = evaluate(p)
            if l < best_loss:
                best_loss = l
                best_params = p.clone()

        for _ in range(self.n_calls - self.n_random_starts):
            p = best_params + 0.1 * torch.randn(dim, device=self.device, dtype=self.dtype, generator=gen)
            l = evaluate(p)
            if l < best_loss:
                best_loss = l
                best_params = p.clone()

        with torch.no_grad():
                    self._w.data.copy_(best_params[:self._w.numel()].reshape(self._w.shape))
                    if self._rho is not None:
                        self._rho.data.copy_(best_params[self._w.numel():self._w.numel() + 1].reshape(self._rho.data.shape))
        self.n_iter_ = self.n_calls
        self.coef_ = self._w.detach()
        self.intercept_ = torch.tensor([-self._rho.item()], dtype=self.dtype, device=self.device)
        self.offset_ = torch.tensor([0.0], dtype=self.dtype, device=self.device)
        self.fit_status_ = 0
        return self


class PSOOneClassSVM(SGDOneClassSVM):
    """One-Class SVM optimized with Particle Swarm Optimization."""
    def __init__(self, kernel='rbf', kernel_params=None, degree=3, gamma='scale',
                 coef0=0.0, tol=0.001, fit_intercept=True, shuffle=True,
                 random_state=None, learning_rate='constant', eta0=0.01,
                 power_t=0.5, warm_start=False, average=False, nu=0.5,
                 shrinking=True, cache_size=200, verbose=False, max_iter=-1,
                 n_particles: int = 20, inertia: float = 0.7, c1: float = 1.5, c2: float = 1.5,
                 device='cpu', dtype=torch.float, *args, **kwargs):
        super().__init__(kernel=kernel, kernel_params=kernel_params, degree=degree, gamma=gamma,
                         coef0=coef0, tol=tol, fit_intercept=fit_intercept, shuffle=shuffle,
                         random_state=random_state, learning_rate=learning_rate, eta0=eta0,
                         power_t=power_t, warm_start=warm_start, average=average, nu=nu,
                         shrinking=shrinking, cache_size=cache_size, verbose=verbose,
                         max_iter=max_iter, device=device, dtype=dtype, *args, **kwargs)
        self.n_particles = n_particles
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2

    def fit(self, data_or_X, y=None, **kwargs):
        warm_start = kwargs.get('warm_start', self.warm_start)
        X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self.shape_fit_ = tuple(X.shape)
        if not warm_start or self._w is None or self._w.shape[1] != n_features:
            self._init_params(n_features)
        dim = self._w.numel() + 1
        max_iter = max(self.max_iter, 100) if self.max_iter > 0 else 100
        gen = torch.Generator(device=self.device)
        if self.random_state is not None:
            gen.manual_seed(int(self.random_state) if isinstance(self.random_state, int) else 42)

        pos = torch.randn(self.n_particles, dim, device=self.device, dtype=self.dtype, generator=gen)
        vel = torch.zeros_like(pos)
        p_best = pos.clone()
        p_best_loss = torch.full((self.n_particles,), float('inf'), device=self.device, dtype=self.dtype)

        def eval_all(positions):
            losses = []
            for p in positions:
                with torch.no_grad():
                    self._w.data.copy_(p[:self._w.numel()].view(self._w.shape))
                    self._rho.data.copy_(p[self._w.numel():self._w.numel() + 1])
                    losses.append(self._compute_ocsvm_loss(X).item())
            return torch.tensor(losses, device=self.device, dtype=self.dtype)

        losses = eval_all(pos)
        p_best_loss = losses.clone()
        g_best = pos[losses.argmin()].clone()

        for _ in range(max_iter):
            r1 = torch.rand(self.n_particles, dim, device=self.device, dtype=self.dtype, generator=gen)
            r2 = torch.rand(self.n_particles, dim, device=self.device, dtype=self.dtype, generator=gen)
            vel = self.inertia * vel + self.c1 * r1 * (p_best - pos) + self.c2 * r2 * (g_best - pos)
            pos = pos + vel
            losses = eval_all(pos)
            improved = losses < p_best_loss
            p_best[improved] = pos[improved]
            p_best_loss[improved] = losses[improved]
            g_best = p_best[p_best_loss.argmin()].clone()

        with torch.no_grad():
            self._w.data.copy_(g_best[:self._w.numel()].view(self._w.shape))
            self._rho.data.copy_(g_best[self._w.numel():self._w.numel() + 1])
        self.n_iter_ = max_iter
        self.coef_ = self._w.detach()
        self.intercept_ = torch.tensor([-self._rho.item()], dtype=self.dtype, device=self.device)
        self.offset_ = torch.tensor([0.0], dtype=self.dtype, device=self.device)
        self.fit_status_ = 0
        return self


class FireflyOneClassSVM(SGDOneClassSVM):
    """One-Class SVM optimized with Firefly Algorithm."""
    def __init__(self, kernel='rbf', kernel_params=None, degree=3, gamma='scale',
                 coef0=0.0, tol=0.001, fit_intercept=True, shuffle=True,
                 random_state=None, learning_rate='constant', eta0=0.01,
                 power_t=0.5, warm_start=False, average=False, nu=0.5,
                 shrinking=True, cache_size=200, verbose=False, max_iter=-1,
                 n_fireflies: int = 20, alpha_firefly: float = 0.2,
                 beta_min: float = 0.2, gamma_firefly: float = 1.0,
                 device='cpu', dtype=torch.float, *args, **kwargs):
        super().__init__(kernel=kernel, kernel_params=kernel_params, degree=degree, gamma=gamma,
                         coef0=coef0, tol=tol, fit_intercept=fit_intercept, shuffle=shuffle,
                         random_state=random_state, learning_rate=learning_rate, eta0=eta0,
                         power_t=power_t, warm_start=warm_start, average=average, nu=nu,
                         shrinking=shrinking, cache_size=cache_size, verbose=verbose,
                         max_iter=max_iter, device=device, dtype=dtype, *args, **kwargs)
        self.n_fireflies = n_fireflies
        self.alpha_firefly = alpha_firefly
        self.beta_min = beta_min
        self.gamma_firefly = gamma_firefly

    def fit(self, data_or_X, y=None, **kwargs):
        warm_start = kwargs.get('warm_start', self.warm_start)
        X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self.shape_fit_ = tuple(X.shape)
        if not warm_start or self._w is None or self._w.shape[1] != n_features:
            self._init_params(n_features)
        dim = self._w.numel() + 1
        max_iter = max(self.max_iter, 50) if self.max_iter > 0 else 50
        gen = torch.Generator(device=self.device)
        if self.random_state is not None:
            gen.manual_seed(int(self.random_state) if isinstance(self.random_state, int) else 42)

        ff = torch.randn(self.n_fireflies, dim, device=self.device, dtype=self.dtype, generator=gen)

        def eval_all(positions):
            losses = []
            for p in positions:
                with torch.no_grad():
                    self._w.data.copy_(p[:self._w.numel()].view(self._w.shape))
                    self._rho.data.copy_(p[self._w.numel():self._w.numel() + 1])
                    losses.append(self._compute_ocsvm_loss(X).item())
            return torch.tensor(losses, device=self.device, dtype=self.dtype)

        intensities = eval_all(ff)
        for _ in range(max_iter):
            for i in range(self.n_fireflies):
                for j in range(self.n_fireflies):
                    if intensities[j] < intensities[i]:
                        r2 = ((ff[i] - ff[j]) ** 2).sum().item()
                        beta = self.beta_min * math.exp(-self.gamma_firefly * r2)
                        noise = self.alpha_firefly * (torch.rand(dim, device=self.device,
                                                                  dtype=self.dtype, generator=gen) - 0.5)
                        ff[i] = ff[i] + beta * (ff[j] - ff[i]) + noise
            intensities = eval_all(ff)

        best = ff[intensities.argmin()]
        with torch.no_grad():
            self._w.data.copy_(best[:self._w.numel()].view(self._w.shape))
            self._rho.data.copy_(best[self._w.numel():self._w.numel() + 1])
        self.n_iter_ = max_iter
        self.coef_ = self._w.detach()
        self.intercept_ = torch.tensor([-self._rho.item()], dtype=self.dtype, device=self.device)
        self.offset_ = torch.tensor([0.0], dtype=self.dtype, device=self.device)
        self.fit_status_ = 0
        return self


class PassiveAggressiveOneClassSVM(SGDOneClassSVM):
    """One-Class SVM trained with Passive-Aggressive update rule."""
    def __init__(self, kernel='rbf', kernel_params=None, degree=3, gamma='scale',
                 coef0=0.0, tol=0.001, fit_intercept=True, shuffle=True,
                 random_state=None, learning_rate='optimal', eta0=0.01,
                 power_t=0.5, warm_start=False, average=False, nu=0.5,
                 shrinking=True, cache_size=200, verbose=False, max_iter=-1,
                 C: float = 1.0,
                 device='cpu', dtype=torch.float, *args, **kwargs):
        super().__init__(kernel=kernel, kernel_params=kernel_params, degree=degree, gamma=gamma,
                         coef0=coef0, tol=tol, fit_intercept=fit_intercept, shuffle=shuffle,
                         random_state=random_state, learning_rate=learning_rate, eta0=eta0,
                         power_t=power_t, warm_start=warm_start, average=average, nu=nu,
                         shrinking=shrinking, cache_size=cache_size, verbose=verbose,
                         max_iter=max_iter, device=device, dtype=dtype, *args, **kwargs)
        self.C = C

    def _update_params(self, X_batch):
        with torch.no_grad():
            xi = X_batch.squeeze(0)
            score = (self._w @ xi.unsqueeze(-1)).squeeze() - self._rho
            loss_val = max(0.0, 1.0 - score.item())
            if loss_val > 0:
                x_norm_sq = (xi ** 2).sum().item()
                tau = min(loss_val / (x_norm_sq + 1e-12), self.C)
                self._w.data.add_(xi.unsqueeze(0), alpha=tau)
                self._rho.data.sub_(torch.tensor(tau, device=self.device, dtype=self.dtype))
        self.num_iter_t += 1
        return loss_val



# ============================================================
# EstimatorBased Outlier Detection — Base class and all 52 variants
# ============================================================
#
# Each variant trains a user-supplied MLModule estimator (regression,
# classification, or clustering) on the training data and uses the
# estimator's prediction residuals / confidence / cluster-distances as
# the per-sample anomaly score.
#
# When no estimator is provided the class falls back to a symmetric
# MLP autoencoder trained with the variant's specific optimiser, and
# uses reconstruction-MSE as the anomaly score.
#
# Score convention (consistent with IsolationForest and LOF):
#   score_samples: higher value = more normal (inlier)
#   decision_function = score_samples - offset_
#   predict:  +1 inlier / -1 outlier


class _BaseEstimatorBasedOutlierDetection(MLModule):
    """Estimator-based unsupervised outlier detector.

    Trains either a *user-supplied estimator* or a built-in autoencoder on
    the training data and uses the resulting anomaly signal (reconstruction
    error, regression residual, classification confidence, or cluster
    distance) to rank samples by abnormality.

    Parameters
    ----------
    estimator : MLModule or None, default=None
        Any fitted or unfitted MLModule subclass:

        * **MLRegressor** – fitted as X → X (reconstruction); anomaly score
          is the negative mean squared reconstruction error.
        * **MLClassifier** – fitted with all-positive pseudo-labels (all
          training samples labelled 1); anomaly score is the log-probability
          assigned to class 1 (``predict_proba`` column 1, or
          ``decision_function`` if ``predict_proba`` is unavailable).
        * **MLCluster** – fitted unsupervised; anomaly score is the negative
          distance from each sample to its nearest cluster centre (via
          ``transform``).  Falls back to autoencoder if ``transform`` is not
          implemented.
        * ``None`` – use the built-in autoencoder (see below).

    hidden_dim : int, default=64
        Hidden-layer width of the fallback autoencoder.
    bottleneck_dim : int, default=16
        Bottleneck dimension of the fallback autoencoder.
    n_layers : int, default=2
        Number of encoder/decoder layers (each layer halves/doubles the width).
    activation : str, nn.Module, or Callable, default='relu'
        Activation function used in the autoencoder.  Accepts any string
        recognised by ``torch.nn`` (e.g. ``'relu'``, ``'tanh'``) **or** any
        string name supported by :class:`~Code.models.deep_learning.activations\
.ActivationFunction.Activation` (e.g. ``'Swish'``, ``'Mish'``, ``'GELU'``,
        ``'Snake'``).  You may also pass an already-instantiated
        :class:`nn.Module` or any ``Callable`` that accepts a ``Tensor``.
    contamination : {'auto', float}, default='auto'
        Expected fraction of outliers.  'auto' sets the threshold at
        mean + 2 × std of training scores; a float in (0, 0.5] sets the
        threshold at the (1 - contamination) quantile.
    max_iter : int, default=100
        Maximum training epochs.
    tol : float, default=1e-4
        Early-stopping tolerance on epoch-wise loss change.
    eta0 : float, default=1e-3
        Base learning rate (used by the fallback autoencoder trainer and
        as the starting LR for the variant-specific optimiser).
    batch_size : int, default=32
        Mini-batch size for the autoencoder training loop.
    shuffle : bool, default=True
        Shuffle data before each training epoch.
    random_state : int or None, default=None
    verbose : bool, default=False
    device : str or torch.device, default='cpu'
    dtype : torch.dtype, default=torch.float

    Attributes
    ----------
    offset_ : float
        Decision threshold calibrated during ``fit``.
    n_features_in_ : int
    n_iter_ : int
        Epochs run (autoencoder path) or 1 (estimator path).
    """

    def __init__(self,
                 estimator: Optional[MLModule] = None,
                 hidden_dim: int = 64,
                 bottleneck_dim: int = 16,
                 n_layers: int = 2,
                 activation: Union[str, nn.Module, Callable] = 'relu',
                 contamination: Union[Literal["auto"], float] = 'auto',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 eta0: float = 1e-3,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 random_state: Union[int, None] = None,
                 verbose: bool = False,
                 warm_start: bool = False,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.estimator = estimator
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_layers = n_layers
        self.activation = activation
        self.contamination = contamination
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.device = device
        self.dtype = dtype

        self._autoencoder: Optional[nn.Module] = None
        self.offset_: Optional[float] = None
        self.n_features_in_: Optional[int] = None
        self.n_iter_: int = 0

    # ------------------------------------------------------------------
    # Autoencoder (fallback when no estimator is supplied)
    # ------------------------------------------------------------------

    def _resolve_activation(self) -> Callable[[], nn.Module]:
        """Return a *factory* – a zero-argument callable that produces a fresh
        ``nn.Module`` activation instance on every call.

        This lets ``_build_autoencoder`` call ``Act()`` for each layer in the
        encoder / decoder stack while keeping each layer's activation as an
        independent module (important for modules that carry learnable
        parameters, e.g. ``PReLU``).

        Resolution order
        ----------------
        1. ``self.activation`` is already an ``nn.Module`` **class** (not
           instance) → use it as-is (it *is* already a factory).
        2. ``self.activation`` is an ``nn.Module`` **instance** → wrap in a
           ``copy.deepcopy`` factory so every layer gets its own copy.
        3. ``self.activation`` is an arbitrary ``Callable`` (function / lambda)
           → wrap in a thin ``nn.Module`` subclass factory.
        4. ``self.activation`` is a **string** found in the built-in
           ``torch.nn`` shorthand map (case-insensitive) → return the
           corresponding class.
        5. ``self.activation`` is any other **string** → delegate to
           :class:`~Code.models.deep_learning.activations.ActivationFunction.\
Activation` so the full registry of custom activations (``'Swish'``,
           ``'Mish'``, ``'Snake'``, ``'FReLU'``, …) is available.
        6. Falls back to ``nn.ReLU`` if everything else fails.
        """
        import copy
        act = self.activation

        # Case 1: bare nn.Module *class* passed (e.g. activation=nn.GELU)
        if isinstance(act, type) and issubclass(act, nn.Module):
            return act

        # Case 2: instantiated nn.Module (e.g. activation=nn.PReLU(num_parameters=8))
        if isinstance(act, nn.Module):
            return lambda: copy.deepcopy(act)

        # Case 3: arbitrary callable (plain function / lambda)
        if callable(act):
            class _CallableAct(nn.Module):
                def __init__(self_, fn=act):         # capture via default arg
                    super().__init__()
                    self_._fn = fn
                def forward(self_, x):
                    return self_._fn(x)
            return _CallableAct

        # Cases 4 & 5: string name
        act_str = str(act)
        _builtin_map: Dict[str, type] = {
            'relu':         nn.ReLU,
            'tanh':         nn.Tanh,
            'sigmoid':      nn.Sigmoid,
            'leaky_relu':   nn.LeakyReLU,
            'elu':          nn.ELU,
            'selu':         nn.SELU,
            'gelu':         nn.GELU,
            'hardswish':    nn.Hardswish,
            'hardsigmoid':  nn.Hardsigmoid,
            'silu':         nn.SiLU,
            'mish':         nn.Mish,
            'prelu':        nn.PReLU,
            'celu':         nn.CELU,
            'softplus':     nn.Softplus,
            'softsign':     nn.Softsign,
            'tanhshrink':   nn.Tanhshrink,
            'relu6':        nn.ReLU6,
            'rrelu':        nn.RReLU,
        }
        if act_str.lower() in _builtin_map:
            return _builtin_map[act_str.lower()]

        # Full deep-learning activation registry (Swish, Snake, Mish, …)
        try:
            from .....models.deep_learning.activations.ActivationFunction import (
                Activation as _DLActivation,
            )
            # Verify the name resolves before committing to this factory
            _probe = _DLActivation(act_str)
            del _probe
            return lambda: _DLActivation(act_str)
        except Exception:
            pass

        # Ultimate fallback
        return nn.ReLU

    def _build_autoencoder(self, n_features: int) -> nn.Module:
        """Build a symmetric encoder-decoder MLP.

        The encoder progressively reduces dimensionality from *n_features*
        down to *bottleneck_dim* through *n_layers* linear + activation
        layers.  The decoder mirrors this structure to reconstruct the input.
        """
        Act = self._resolve_activation()   # factory: Act() → nn.Module

        # Build dimension sequence: n_features → hidden → ... → bottleneck
        enc_dims = [n_features]
        step = max((n_features - self.bottleneck_dim) // max(self.n_layers, 1), 1)
        for i in range(self.n_layers - 1):
            enc_dims.append(max(n_features - (i + 1) * step, self.bottleneck_dim))
        enc_dims.append(self.bottleneck_dim)

        dec_dims = list(reversed(enc_dims))

        enc_layers: List[nn.Module] = []
        for i in range(len(enc_dims) - 1):
            enc_layers.append(nn.Linear(enc_dims[i], enc_dims[i + 1]))
            if i < len(enc_dims) - 2:
                enc_layers.append(Act())

        dec_layers: List[nn.Module] = []
        for i in range(len(dec_dims) - 1):
            dec_layers.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
            if i < len(dec_dims) - 2:
                dec_layers.append(Act())

        class Autoencoder(nn.Module):
            def __init__(self, encoder_layers, decoder_layers):
                super().__init__()
                self.encoder = nn.Sequential(*encoder_layers)
                self.decoder = nn.Sequential(*decoder_layers)

            def forward(self, x):
                return self.decoder(self.encoder(x))

        return Autoencoder(enc_layers, dec_layers)

    def _create_optimizer(self, params) -> torch.optim.Optimizer:
        """Return the optimiser to train the fallback autoencoder.

        Subclasses override this to inject a different optimiser while
        reusing the rest of the training loop.
        """
        return torch.optim.SGD(params, lr=self.eta0)

    def _fit_autoencoder(self, X: torch.Tensor, warm_start: bool = False) -> None:
        """Train the fallback autoencoder on *X* until convergence or max_iter."""
        n_samples, n_features = X.shape
        if self.random_state is not None:
            torch.manual_seed(int(self.random_state))

        reuse_autoencoder = False
        if warm_start and self._autoencoder is not None:
            try:
                in_dim = self._autoencoder.encoder[0].weight.shape[1]
                if in_dim == n_features:
                    reuse_autoencoder = True
            except (AttributeError, IndexError, TypeError):
                pass

        if not reuse_autoencoder:
            self._autoencoder = self._build_autoencoder(n_features).to(self.device)
            if self.dtype == torch.double:
                self._autoencoder = self._autoencoder.double()

        optimizer = self._create_optimizer(self._autoencoder.parameters())
        prev_loss = float('inf')

        for epoch in range(self.max_iter):
            if self.shuffle:
                perm = torch.randperm(n_samples, device=self.device)
                X_ep = X[perm]
            else:
                X_ep = X

            epoch_loss, nb = 0.0, 0
            for i in range(0, n_samples, self.batch_size):
                batch = X_ep[i:i + self.batch_size]
                optimizer.zero_grad()
                recon = self._autoencoder(batch)
                custom_loss_fn = getattr(self, '_fit_loss_fn', None)
                if custom_loss_fn is not None:
                    loss = custom_loss_fn(batch, recon)
                else:
                    loss = F.mse_loss(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                nb += 1
            epoch_loss /= max(nb, 1)

            if self.verbose:
                print(f"{self.__class__.__name__} epoch {epoch + 1}: "
                      f"loss={epoch_loss:.6f}")
            if abs(prev_loss - epoch_loss) < self.tol and epoch > 0:
                self.n_iter_ = epoch + 1
                return
            prev_loss = epoch_loss
        self.n_iter_ = self.max_iter

    def _autoencoder_scores(self, X: torch.Tensor) -> torch.Tensor:
        """Per-sample negative reconstruction MSE (higher = more normal)."""
        self._autoencoder.eval()
        with torch.no_grad():
            recon = self._autoencoder(X)
            mse = (recon - X).pow(2).mean(dim=1)   # (n,)
        return -mse

    # ------------------------------------------------------------------
    # Estimator-based scoring
    # ------------------------------------------------------------------

    def _fit_estimator(self, X: torch.Tensor, **kwargs) -> None:
        """Fit the user-supplied estimator in an unsupervised fashion.

        * **MLRegressor**: fit(X, X) — train the regressor to reconstruct X.
        * **MLClassifier**: fit(X, ones) — train the classifier to always
          predict class 1 (all training samples are "normal").
        * **MLCluster**: fit(X) — unsupervised cluster fitting.
        """
        est = self.estimator
        if isinstance(est, MLRegressor):
            est.fit(X, X, **kwargs)
        elif isinstance(est, MLClassifier):
            # Pseudo-labels: all training points are "inliers" (class 1)
            y_pseudo = torch.ones(X.shape[0], dtype=torch.long, device=self.device)
            est.fit(X, y_pseudo, **kwargs)
        elif isinstance(est, MLCluster):
            est.fit(X, **kwargs)
        else:
            # Generic MLModule: try fit(X); ignore y
            try:
                est.fit(X, **kwargs)
            except TypeError:
                est.fit(X, X, **kwargs)
        self.n_iter_ = 1

    def _estimator_scores(self, X: torch.Tensor) -> torch.Tensor:
        """Compute anomaly scores using the fitted estimator.

        * **MLRegressor**: negative MSE between prediction and X.
        * **MLClassifier**: log P(class=1 | x) via predict_proba, or raw
          decision_function value if predict_proba is unavailable.
        * **MLCluster**: negative distance to nearest cluster centre via
          transform(); falls back to autoencoder if transform() raises.
        * **Other MLModule**: decision_function if present, else autoencoder.
        """
        est = self.estimator
        if isinstance(est, MLRegressor):
            with torch.no_grad():
                preds = est.predict(X)
            if preds.shape != X.shape:
                # Model output shape may differ; fall back to autoencoder
                return self._autoencoder_scores(X)
            mse = (preds - X).pow(2).mean(dim=1)
            return -mse

        elif isinstance(est, MLClassifier):
            with torch.no_grad():
                if hasattr(est, 'predict_proba'):
                    proba = est.predict_proba(X)
                    # P(class=1) is the normality score; higher = more normal
                    if proba.dim() == 2 and proba.shape[1] >= 2:
                        return torch.log(proba[:, 1].clamp(min=1e-12))
                    return torch.log(proba.squeeze(-1).clamp(min=1e-12))
                if hasattr(est, 'decision_function'):
                    return est.decision_function(X)
            # Fall back
            return self._autoencoder_scores(X)

        elif isinstance(est, MLCluster):
            try:
                with torch.no_grad():
                    dist = est.transform(X)  # (n, k) distance to each cluster
                # Use negative distance to nearest centre
                if isinstance(dist, torch.Tensor) and dist.dim() == 2:
                    return -dist.min(dim=1).values
                if isinstance(dist, torch.Tensor):
                    return -dist
            except (NotImplementedError, Exception):
                pass
            # Fall back to autoencoder
            return self._autoencoder_scores(X)

        else:
            # Generic MLModule
            if hasattr(est, 'decision_function'):
                with torch.no_grad():
                    return est.decision_function(X)
            return self._autoencoder_scores(X)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, data_or_X, y=None, **kwargs):
        """Fit the outlier detector on *data_or_X*.

        If ``self.estimator`` is provided, it is fitted on the training data
        in unsupervised style.  Otherwise, the internal autoencoder is trained.
        After fitting, the decision threshold ``offset_`` is calibrated
        using ``contamination``.
        """
        self._fit_loss_fn = kwargs.get('loss_fn', None)
        try:
            warm_start = kwargs.get('warm_start', self.warm_start)
            X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
            if X.dim() == 1:
                X = X.unsqueeze(0)
            self.n_features_in_ = X.shape[1]

            if self.estimator is not None:
                self._fit_estimator(X, **kwargs)
                # Ensure a fallback autoencoder is available for score methods
                # that call _autoencoder_scores when the estimator path fails
                if self._autoencoder is None:
                    self._fit_autoencoder(X, warm_start=warm_start)
            else:
                self._fit_autoencoder(X, warm_start=warm_start)

            # Calibrate the offset from training scores
            with torch.no_grad():
                scores = self._raw_scores(X)

            if self.contamination == "auto":
                self.offset_ = float(
                    (scores.mean() + 2.0 * scores.std()).item())
            else:
                self.offset_ = float(
                    torch.quantile(scores, 1.0 - float(self.contamination)).item())

            return self
        finally:
            self._fit_loss_fn = None

    def _raw_scores(self, X: torch.Tensor) -> torch.Tensor:
        """Compute anomaly scores (higher = more normal)."""
        if self.estimator is not None:
            return self._estimator_scores(X)
        return self._autoencoder_scores(X)

    def score_samples(self, X) -> torch.Tensor:
        """Return anomaly scores for *X*.  Higher means more normal."""
        if self._autoencoder is None and self.estimator is None:
            raise RuntimeError(
                f"{self.__class__.__name__} is not fitted yet.")
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        return self._raw_scores(X)

    def decision_function(self, X) -> torch.Tensor:
        """Decision function: score_samples(X) - offset_.
        Values >= 0 indicate inliers.
        """
        return self.score_samples(X) - self.offset_

    def predict(self, X) -> torch.Tensor:
        """Predict +1 (inlier) or -1 (outlier) for each sample."""
        df = self.decision_function(X)
        return torch.where(df >= 0,
                           torch.ones_like(df, dtype=torch.long),
                           -torch.ones_like(df, dtype=torch.long))

    def forward(self, X):
        return self.decision_function(X)


# ------------------------------------------------------------------
# SGD-trained EstimatorBased detector (base of all variants)
# ------------------------------------------------------------------

class SGDEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector trained with plain SGD.

    Pass any ``MLModule``-based estimator via the ``estimator`` argument.
    When ``estimator`` is None the internal autoencoder is used.
    """
    def _create_optimizer(self, params):
        return torch.optim.SGD(params, lr=self.eta0)


class BGDEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector trained with Batch Gradient Descent.

    Uses ``batch_size`` equal to the full dataset for a true batch update.
    """
    def __init__(self,
                 estimator: Optional[MLModule] = None,
                 hidden_dim: int = 64,
                 bottleneck_dim: int = 16,
                 n_layers: int = 2,
                 activation: str = 'relu',
                 contamination: Union[Literal["auto"], float] = 'auto',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 eta0: float = 1e-3,
                 batch_size: int = 10000,
                 shuffle: bool = True,
                 random_state: Union[int, None] = None,
                 verbose: bool = False,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(estimator=estimator, hidden_dim=hidden_dim,
                         bottleneck_dim=bottleneck_dim, n_layers=n_layers,
                         activation=activation, contamination=contamination,
                         max_iter=max_iter, tol=tol, eta0=eta0,
                         batch_size=batch_size, shuffle=shuffle,
                         random_state=random_state, verbose=verbose,
                         device=device, dtype=dtype, *args, **kwargs)

    def _create_optimizer(self, params):
        return torch.optim.SGD(params, lr=self.eta0)


class MGDEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector trained with Momentum GD."""
    def __init__(self,
                 estimator: Optional[MLModule] = None,
                 hidden_dim: int = 64,
                 bottleneck_dim: int = 16,
                 n_layers: int = 2,
                 activation: str = 'relu',
                 contamination: Union[Literal["auto"], float] = 'auto',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 eta0: float = 1e-3,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 random_state: Union[int, None] = None,
                 verbose: bool = False,
                 gamma_momentum: float = 0.9,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(estimator=estimator, hidden_dim=hidden_dim,
                         bottleneck_dim=bottleneck_dim, n_layers=n_layers,
                         activation=activation, contamination=contamination,
                         max_iter=max_iter, tol=tol, eta0=eta0,
                         batch_size=batch_size, shuffle=shuffle,
                         random_state=random_state, verbose=verbose,
                         device=device, dtype=dtype, *args, **kwargs)
        self.gamma_momentum = gamma_momentum

    def _create_optimizer(self, params):
        return torch.optim.SGD(params, lr=self.eta0, momentum=self.gamma_momentum)


class AdaGradEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector trained with AdaGrad."""
    def __init__(self,
                 estimator: Optional[MLModule] = None,
                 hidden_dim: int = 64,
                 bottleneck_dim: int = 16,
                 n_layers: int = 2,
                 activation: str = 'relu',
                 contamination: Union[Literal["auto"], float] = 'auto',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 eta0: float = 1e-3,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 random_state: Union[int, None] = None,
                 verbose: bool = False,
                 smoothening_term: float = 1e-10,
                 weight_decay: float = 0.0,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(estimator=estimator, hidden_dim=hidden_dim,
                         bottleneck_dim=bottleneck_dim, n_layers=n_layers,
                         activation=activation, contamination=contamination,
                         max_iter=max_iter, tol=tol, eta0=eta0,
                         batch_size=batch_size, shuffle=shuffle,
                         random_state=random_state, verbose=verbose,
                         device=device, dtype=dtype, *args, **kwargs)
        self.smoothening_term = smoothening_term
        self.weight_decay = weight_decay

    def _create_optimizer(self, params):
        return torch.optim.Adagrad(params, lr=self.eta0,
                                    eps=self.smoothening_term,
                                    weight_decay=self.weight_decay)


class RMSPropEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector trained with RMSProp."""
    def __init__(self,
                 estimator: Optional[MLModule] = None,
                 hidden_dim: int = 64,
                 bottleneck_dim: int = 16,
                 n_layers: int = 2,
                 activation: str = 'relu',
                 contamination: Union[Literal["auto"], float] = 'auto',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 eta0: float = 1e-3,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 random_state: Union[int, None] = None,
                 verbose: bool = False,
                 decay_rate: float = 0.99,
                 smoothening_term: float = 1e-8,
                 momentum: float = 0.0,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(estimator=estimator, hidden_dim=hidden_dim,
                         bottleneck_dim=bottleneck_dim, n_layers=n_layers,
                         activation=activation, contamination=contamination,
                         max_iter=max_iter, tol=tol, eta0=eta0,
                         batch_size=batch_size, shuffle=shuffle,
                         random_state=random_state, verbose=verbose,
                         device=device, dtype=dtype, *args, **kwargs)
        self.decay_rate = decay_rate
        self.smoothening_term = smoothening_term
        self.momentum = momentum

    def _create_optimizer(self, params):
        return torch.optim.RMSprop(params, lr=self.eta0,
                                    alpha=self.decay_rate,
                                    eps=self.smoothening_term,
                                    momentum=self.momentum)


class AdamEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector trained with Adam."""
    def __init__(self,
                 estimator: Optional[MLModule] = None,
                 hidden_dim: int = 64,
                 bottleneck_dim: int = 16,
                 n_layers: int = 2,
                 activation: str = 'relu',
                 contamination: Union[Literal["auto"], float] = 'auto',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 eta0: float = 1e-3,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 random_state: Union[int, None] = None,
                 verbose: bool = False,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 smoothening_term: float = 1e-8,
                 weight_decay: float = 0.0,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(estimator=estimator, hidden_dim=hidden_dim,
                         bottleneck_dim=bottleneck_dim, n_layers=n_layers,
                         activation=activation, contamination=contamination,
                         max_iter=max_iter, tol=tol, eta0=eta0,
                         batch_size=batch_size, shuffle=shuffle,
                         random_state=random_state, verbose=verbose,
                         device=device, dtype=dtype, *args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.smoothening_term = smoothening_term
        self.weight_decay = weight_decay

    def _create_optimizer(self, params):
        return torch.optim.Adam(params, lr=self.eta0,
                                 betas=(self.beta1, self.beta2),
                                 eps=self.smoothening_term,
                                 weight_decay=self.weight_decay)


class AdadeltaEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector trained with Adadelta."""
    def __init__(self,
                 estimator: Optional[MLModule] = None,
                 hidden_dim: int = 64,
                 bottleneck_dim: int = 16,
                 n_layers: int = 2,
                 activation: str = 'relu',
                 contamination: Union[Literal["auto"], float] = 'auto',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 eta0: float = 1.0,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 random_state: Union[int, None] = None,
                 verbose: bool = False,
                 rho: float = 0.9,
                 smoothening_term: float = 1e-6,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(estimator=estimator, hidden_dim=hidden_dim,
                         bottleneck_dim=bottleneck_dim, n_layers=n_layers,
                         activation=activation, contamination=contamination,
                         max_iter=max_iter, tol=tol, eta0=eta0,
                         batch_size=batch_size, shuffle=shuffle,
                         random_state=random_state, verbose=verbose,
                         device=device, dtype=dtype, *args, **kwargs)
        self.rho = rho
        self.smoothening_term = smoothening_term

    def _create_optimizer(self, params):
        return torch.optim.Adadelta(params, lr=self.eta0,
                                     rho=self.rho,
                                     eps=self.smoothening_term)


class AdafactorEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector trained with Adafactor (approximated via Adam)."""
    def __init__(self,
                 estimator: Optional[MLModule] = None,
                 hidden_dim: int = 64,
                 bottleneck_dim: int = 16,
                 n_layers: int = 2,
                 activation: str = 'relu',
                 contamination: Union[Literal["auto"], float] = 'auto',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 eta0: float = 1e-3,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 random_state: Union[int, None] = None,
                 verbose: bool = False,
                 beta1: float = None,
                 clip_threshold: float = 1.0,
                 decay_rate: float = -0.8,
                 eps1: float = 1e-30,
                 eps2: float = 1e-3,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(estimator=estimator, hidden_dim=hidden_dim,
                         bottleneck_dim=bottleneck_dim, n_layers=n_layers,
                         activation=activation, contamination=contamination,
                         max_iter=max_iter, tol=tol, eta0=eta0,
                         batch_size=batch_size, shuffle=shuffle,
                         random_state=random_state, verbose=verbose,
                         device=device, dtype=dtype, *args, **kwargs)
        self.beta1_af = beta1
        self.clip_threshold = clip_threshold
        self.decay_rate = decay_rate
        self.eps1 = eps1
        self.eps2 = eps2

    def _create_optimizer(self, params):
        return torch.optim.Adam(params, lr=self.eta0)


class AdamWEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector trained with AdamW."""
    def __init__(self,
                 estimator: Optional[MLModule] = None,
                 hidden_dim: int = 64,
                 bottleneck_dim: int = 16,
                 n_layers: int = 2,
                 activation: str = 'relu',
                 contamination: Union[Literal["auto"], float] = 'auto',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 eta0: float = 1e-3,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 random_state: Union[int, None] = None,
                 verbose: bool = False,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 smoothening_term: float = 1e-8,
                 weight_decay: float = 0.01,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(estimator=estimator, hidden_dim=hidden_dim,
                         bottleneck_dim=bottleneck_dim, n_layers=n_layers,
                         activation=activation, contamination=contamination,
                         max_iter=max_iter, tol=tol, eta0=eta0,
                         batch_size=batch_size, shuffle=shuffle,
                         random_state=random_state, verbose=verbose,
                         device=device, dtype=dtype, *args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.smoothening_term = smoothening_term
        self.weight_decay = weight_decay

    def _create_optimizer(self, params):
        return torch.optim.AdamW(params, lr=self.eta0,
                                  betas=(self.beta1, self.beta2),
                                  eps=self.smoothening_term,
                                  weight_decay=self.weight_decay)


class AdamaxEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector trained with Adamax."""
    def __init__(self,
                 estimator: Optional[MLModule] = None,
                 hidden_dim: int = 64,
                 bottleneck_dim: int = 16,
                 n_layers: int = 2,
                 activation: str = 'relu',
                 contamination: Union[Literal["auto"], float] = 'auto',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 eta0: float = 2e-3,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 random_state: Union[int, None] = None,
                 verbose: bool = False,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 smoothening_term: float = 1e-8,
                 weight_decay: float = 0.0,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(estimator=estimator, hidden_dim=hidden_dim,
                         bottleneck_dim=bottleneck_dim, n_layers=n_layers,
                         activation=activation, contamination=contamination,
                         max_iter=max_iter, tol=tol, eta0=eta0,
                         batch_size=batch_size, shuffle=shuffle,
                         random_state=random_state, verbose=verbose,
                         device=device, dtype=dtype, *args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.smoothening_term = smoothening_term
        self.weight_decay = weight_decay

    def _create_optimizer(self, params):
        return torch.optim.Adamax(params, lr=self.eta0,
                                   betas=(self.beta1, self.beta2),
                                   eps=self.smoothening_term,
                                   weight_decay=self.weight_decay)


class LBFGSEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector trained with L-BFGS (full-batch)."""
    def __init__(self,
                 estimator: Optional[MLModule] = None,
                 hidden_dim: int = 64,
                 bottleneck_dim: int = 16,
                 n_layers: int = 2,
                 activation: str = 'relu',
                 contamination: Union[Literal["auto"], float] = 'auto',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 eta0: float = 1.0,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 random_state: Union[int, None] = None,
                 verbose: bool = False,
                 history_size: int = 10,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(estimator=estimator, hidden_dim=hidden_dim,
                         bottleneck_dim=bottleneck_dim, n_layers=n_layers,
                         activation=activation, contamination=contamination,
                         max_iter=max_iter, tol=tol, eta0=eta0,
                         batch_size=batch_size, shuffle=shuffle,
                         random_state=random_state, verbose=verbose,
                         device=device, dtype=dtype, *args, **kwargs)
        self.history_size = history_size

    def _fit_autoencoder(self, X: torch.Tensor, warm_start: bool = False) -> None:
        """Override the training loop to use torch L-BFGS with a closure."""
        n_features = X.shape[1]
        if self.random_state is not None:
            torch.manual_seed(int(self.random_state))
        reuse_autoencoder = False
        if warm_start and self._autoencoder is not None:
            try:
                in_dim = self._autoencoder.encoder[0].weight.shape[1]
                if in_dim == n_features:
                    reuse_autoencoder = True
            except (AttributeError, IndexError, TypeError):
                pass
        if not reuse_autoencoder:
            self._autoencoder = self._build_autoencoder(n_features).to(self.device)
            if self.dtype == torch.double:
                self._autoencoder = self._autoencoder.double()

        optimizer = torch.optim.LBFGS(
            self._autoencoder.parameters(),
            lr=self.eta0,
            max_iter=min(self.max_iter, 50),
            history_size=self.history_size,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            recon = self._autoencoder(X)
            custom_loss_fn = getattr(self, '_fit_loss_fn', None)
            if custom_loss_fn is not None:
                loss = custom_loss_fn(X, recon)
            else:
                loss = F.mse_loss(recon, X)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.n_iter_ = 1


class MuonEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector trained with Muon (Nesterov momentum SGD)."""
    def __init__(self,
                 estimator: Optional[MLModule] = None,
                 hidden_dim: int = 64,
                 bottleneck_dim: int = 16,
                 n_layers: int = 2,
                 activation: str = 'relu',
                 contamination: Union[Literal["auto"], float] = 'auto',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 eta0: float = 1e-3,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 random_state: Union[int, None] = None,
                 verbose: bool = False,
                 momentum: float = 0.9,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(estimator=estimator, hidden_dim=hidden_dim,
                         bottleneck_dim=bottleneck_dim, n_layers=n_layers,
                         activation=activation, contamination=contamination,
                         max_iter=max_iter, tol=tol, eta0=eta0,
                         batch_size=batch_size, shuffle=shuffle,
                         random_state=random_state, verbose=verbose,
                         device=device, dtype=dtype, *args, **kwargs)
        self.momentum = momentum

    def _create_optimizer(self, params):
        return torch.optim.SGD(params, lr=self.eta0,
                                momentum=self.momentum, nesterov=True)


class NAdamEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector trained with NAdam."""
    def __init__(self,
                 estimator: Optional[MLModule] = None,
                 hidden_dim: int = 64,
                 bottleneck_dim: int = 16,
                 n_layers: int = 2,
                 activation: str = 'relu',
                 contamination: Union[Literal["auto"], float] = 'auto',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 eta0: float = 2e-3,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 random_state: Union[int, None] = None,
                 verbose: bool = False,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 smoothening_term: float = 1e-8,
                 weight_decay: float = 0.0,
                 momentum_decay: float = 4e-3,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(estimator=estimator, hidden_dim=hidden_dim,
                         bottleneck_dim=bottleneck_dim, n_layers=n_layers,
                         activation=activation, contamination=contamination,
                         max_iter=max_iter, tol=tol, eta0=eta0,
                         batch_size=batch_size, shuffle=shuffle,
                         random_state=random_state, verbose=verbose,
                         device=device, dtype=dtype, *args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.smoothening_term = smoothening_term
        self.weight_decay = weight_decay
        self.momentum_decay = momentum_decay

    def _create_optimizer(self, params):
        return torch.optim.NAdam(params, lr=self.eta0,
                                  betas=(self.beta1, self.beta2),
                                  eps=self.smoothening_term,
                                  weight_decay=self.weight_decay,
                                  momentum_decay=self.momentum_decay)


class RAdamEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector trained with RAdam."""
    def __init__(self,
                 estimator: Optional[MLModule] = None,
                 hidden_dim: int = 64,
                 bottleneck_dim: int = 16,
                 n_layers: int = 2,
                 activation: str = 'relu',
                 contamination: Union[Literal["auto"], float] = 'auto',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 eta0: float = 1e-3,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 random_state: Union[int, None] = None,
                 verbose: bool = False,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 smoothening_term: float = 1e-8,
                 weight_decay: float = 0.0,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(estimator=estimator, hidden_dim=hidden_dim,
                         bottleneck_dim=bottleneck_dim, n_layers=n_layers,
                         activation=activation, contamination=contamination,
                         max_iter=max_iter, tol=tol, eta0=eta0,
                         batch_size=batch_size, shuffle=shuffle,
                         random_state=random_state, verbose=verbose,
                         device=device, dtype=dtype, *args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.smoothening_term = smoothening_term
        self.weight_decay = weight_decay

    def _create_optimizer(self, params):
        return torch.optim.RAdam(params, lr=self.eta0,
                                  betas=(self.beta1, self.beta2),
                                  eps=self.smoothening_term,
                                  weight_decay=self.weight_decay)


class RpropEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector trained with Rprop."""
    def __init__(self,
                 estimator: Optional[MLModule] = None,
                 hidden_dim: int = 64,
                 bottleneck_dim: int = 16,
                 n_layers: int = 2,
                 activation: str = 'relu',
                 contamination: Union[Literal["auto"], float] = 'auto',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 eta0: float = 1e-2,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 random_state: Union[int, None] = None,
                 verbose: bool = False,
                 etas: Tuple[float, float] = (0.5, 1.2),
                 step_sizes: Tuple[float, float] = (1e-6, 50.0),
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(estimator=estimator, hidden_dim=hidden_dim,
                         bottleneck_dim=bottleneck_dim, n_layers=n_layers,
                         activation=activation, contamination=contamination,
                         max_iter=max_iter, tol=tol, eta0=eta0,
                         batch_size=batch_size, shuffle=shuffle,
                         random_state=random_state, verbose=verbose,
                         device=device, dtype=dtype, *args, **kwargs)
        self.etas = etas
        self.step_sizes = step_sizes

    def _create_optimizer(self, params):
        return torch.optim.Rprop(params, lr=self.eta0,
                                  etas=self.etas,
                                  step_sizes=self.step_sizes)


class BFGSEstimatorBasedOutlierDetection(LBFGSEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector trained with BFGS (via L-BFGS with history=n)."""
    def __init__(self,
                 estimator: Optional[MLModule] = None,
                 hidden_dim: int = 64,
                 bottleneck_dim: int = 16,
                 n_layers: int = 2,
                 activation: str = 'relu',
                 contamination: Union[Literal["auto"], float] = 'auto',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 eta0: float = 1.0,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 random_state: Union[int, None] = None,
                 verbose: bool = False,
                 history_size: int = 100,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(estimator=estimator, hidden_dim=hidden_dim,
                         bottleneck_dim=bottleneck_dim, n_layers=n_layers,
                         activation=activation, contamination=contamination,
                         max_iter=max_iter, tol=tol, eta0=eta0,
                         batch_size=batch_size, shuffle=shuffle,
                         random_state=random_state, verbose=verbose,
                         history_size=history_size, device=device,
                         dtype=dtype, *args, **kwargs)


class NewtonCGEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector trained with Newton-CG (via Adam fallback)."""
    def _create_optimizer(self, params):
        return torch.optim.Adam(params, lr=self.eta0)


class SLSQPEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector trained with SLSQP (via Adam fallback)."""
    def _create_optimizer(self, params):
        return torch.optim.Adam(params, lr=self.eta0)


class LionEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector trained with Lion (sign-based momentum)."""
    def __init__(self,
                 estimator: Optional[MLModule] = None,
                 hidden_dim: int = 64,
                 bottleneck_dim: int = 16,
                 n_layers: int = 2,
                 activation: str = 'relu',
                 contamination: Union[Literal["auto"], float] = 'auto',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 eta0: float = 1e-4,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 random_state: Union[int, None] = None,
                 verbose: bool = False,
                 beta1: float = 0.9,
                 beta2: float = 0.99,
                 weight_decay: float = 0.0,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(estimator=estimator, hidden_dim=hidden_dim,
                         bottleneck_dim=bottleneck_dim, n_layers=n_layers,
                         activation=activation, contamination=contamination,
                         max_iter=max_iter, tol=tol, eta0=eta0,
                         batch_size=batch_size, shuffle=shuffle,
                         random_state=random_state, verbose=verbose,
                         device=device, dtype=dtype, *args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay_lion = weight_decay
        self._momentum_buf: Dict[int, torch.Tensor] = {}

    def _fit_autoencoder(self, X: torch.Tensor, warm_start: bool = False) -> None:
        """Override to implement Lion's sign-based update manually."""
        n_samples, n_features = X.shape
        if self.random_state is not None:
            torch.manual_seed(int(self.random_state))
        reuse_autoencoder = False
        if warm_start and self._autoencoder is not None:
            try:
                in_dim = self._autoencoder.encoder[0].weight.shape[1]
                if in_dim == n_features:
                    reuse_autoencoder = True
            except (AttributeError, IndexError, TypeError):
                pass
        if not reuse_autoencoder:
            self._autoencoder = self._build_autoencoder(n_features).to(self.device)
            if self.dtype == torch.double:
                self._autoencoder = self._autoencoder.double()

        params = list(self._autoencoder.parameters())
        m_bufs = [torch.zeros_like(p.data) for p in params]

        for epoch in range(self.max_iter):
            X_ep = (X[torch.randperm(n_samples, device=self.device)]
                    if self.shuffle else X)
            epoch_loss, nb = 0.0, 0
            for i in range(0, n_samples, self.batch_size):
                batch = X_ep[i:i + self.batch_size]
                for p in params:
                    if p.grad is not None:
                        p.grad.zero_()
                recon = self._autoencoder(batch)
                custom_loss_fn = getattr(self, '_fit_loss_fn', None)
                if custom_loss_fn is not None:
                    loss = custom_loss_fn(batch, recon)
                else:
                    loss = F.mse_loss(recon, batch)
                loss.backward()
                with torch.no_grad():
                    for p, m in zip(params, m_bufs):
                        if p.grad is None:
                            continue
                        # update step = sign(beta1 * m + (1-beta1) * grad)
                        c = self.beta1 * m + (1 - self.beta1) * p.grad
                        p.data.sub_(self.eta0 * torch.sign(c))
                        # update momentum
                        m.mul_(self.beta2).add_(p.grad, alpha=1 - self.beta2)
                        if self.weight_decay_lion > 0:
                            p.data.mul_(1 - self.eta0 * self.weight_decay_lion)
                epoch_loss += loss.item()
                nb += 1
            if self.verbose:
                print(f"LionEstimator epoch {epoch + 1}: "
                      f"loss={epoch_loss / max(nb,1):.6f}")
        self.n_iter_ = self.max_iter


class ShampooEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector trained with Shampoo (via Adam)."""
    def __init__(self,
                 estimator: Optional[MLModule] = None,
                 hidden_dim: int = 64,
                 bottleneck_dim: int = 16,
                 n_layers: int = 2,
                 activation: str = 'relu',
                 contamination: Union[Literal["auto"], float] = 'auto',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 eta0: float = 1e-3,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 random_state: Union[int, None] = None,
                 verbose: bool = False,
                 update_freq: int = 1,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(estimator=estimator, hidden_dim=hidden_dim,
                         bottleneck_dim=bottleneck_dim, n_layers=n_layers,
                         activation=activation, contamination=contamination,
                         max_iter=max_iter, tol=tol, eta0=eta0,
                         batch_size=batch_size, shuffle=shuffle,
                         random_state=random_state, verbose=verbose,
                         device=device, dtype=dtype, *args, **kwargs)
        self.update_freq = update_freq

    def _create_optimizer(self, params):
        return torch.optim.Adam(params, lr=self.eta0)


class SophiaEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector trained with Sophia (Hessian-guided Adam)."""
    def __init__(self,
                 estimator: Optional[MLModule] = None,
                 hidden_dim: int = 64,
                 bottleneck_dim: int = 16,
                 n_layers: int = 2,
                 activation: str = 'relu',
                 contamination: Union[Literal["auto"], float] = 'auto',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 eta0: float = 1e-3,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 random_state: Union[int, None] = None,
                 verbose: bool = False,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 rho: float = 0.04,
                 update_freq: int = 10,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(estimator=estimator, hidden_dim=hidden_dim,
                         bottleneck_dim=bottleneck_dim, n_layers=n_layers,
                         activation=activation, contamination=contamination,
                         max_iter=max_iter, tol=tol, eta0=eta0,
                         batch_size=batch_size, shuffle=shuffle,
                         random_state=random_state, verbose=verbose,
                         device=device, dtype=dtype, *args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.rho = rho
        self.update_freq = update_freq

    def _create_optimizer(self, params):
        return torch.optim.Adam(params, lr=self.eta0,
                                 betas=(self.beta1, self.beta2))


class AdanEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector trained with Adan."""
    def __init__(self,
                 estimator: Optional[MLModule] = None,
                 hidden_dim: int = 64,
                 bottleneck_dim: int = 16,
                 n_layers: int = 2,
                 activation: str = 'relu',
                 contamination: Union[Literal["auto"], float] = 'auto',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 eta0: float = 1e-3,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 random_state: Union[int, None] = None,
                 verbose: bool = False,
                 beta1: float = 0.98,
                 beta2: float = 0.92,
                 beta3: float = 0.99,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(estimator=estimator, hidden_dim=hidden_dim,
                         bottleneck_dim=bottleneck_dim, n_layers=n_layers,
                         activation=activation, contamination=contamination,
                         max_iter=max_iter, tol=tol, eta0=eta0,
                         batch_size=batch_size, shuffle=shuffle,
                         random_state=random_state, verbose=verbose,
                         device=device, dtype=dtype, *args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3

    def _create_optimizer(self, params):
        return torch.optim.Adam(params, lr=self.eta0,
                                 betas=(self.beta1, self.beta2))


class COBYLAEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector (COBYLA derivative-free, uses Adam)."""
    def _create_optimizer(self, params):
        return torch.optim.Adam(params, lr=self.eta0)


class NelderMeadEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector (Nelder-Mead, uses SGD)."""
    def _create_optimizer(self, params):
        return torch.optim.SGD(params, lr=self.eta0)


class LAMBEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector trained with LAMB."""
    def __init__(self,
                 estimator: Optional[MLModule] = None,
                 hidden_dim: int = 64,
                 bottleneck_dim: int = 16,
                 n_layers: int = 2,
                 activation: str = 'relu',
                 contamination: Union[Literal["auto"], float] = 'auto',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 eta0: float = 1e-3,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 random_state: Union[int, None] = None,
                 verbose: bool = False,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 smoothening_term: float = 1e-6,
                 weight_decay: float = 0.01,
                 trust_ratio: float = 1.0,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(estimator=estimator, hidden_dim=hidden_dim,
                         bottleneck_dim=bottleneck_dim, n_layers=n_layers,
                         activation=activation, contamination=contamination,
                         max_iter=max_iter, tol=tol, eta0=eta0,
                         batch_size=batch_size, shuffle=shuffle,
                         random_state=random_state, verbose=verbose,
                         device=device, dtype=dtype, *args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.smoothening_term = smoothening_term
        self.weight_decay = weight_decay
        self.trust_ratio = trust_ratio

    def _create_optimizer(self, params):
        return torch.optim.AdamW(params, lr=self.eta0,
                                  betas=(self.beta1, self.beta2),
                                  eps=self.smoothening_term,
                                  weight_decay=self.weight_decay)


class LookaheadEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector with Lookahead wrapping Adam."""
    def __init__(self,
                 estimator: Optional[MLModule] = None,
                 hidden_dim: int = 64,
                 bottleneck_dim: int = 16,
                 n_layers: int = 2,
                 activation: str = 'relu',
                 contamination: Union[Literal["auto"], float] = 'auto',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 eta0: float = 1e-3,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 random_state: Union[int, None] = None,
                 verbose: bool = False,
                 k: int = 5,
                 alpha_la: float = 0.5,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(estimator=estimator, hidden_dim=hidden_dim,
                         bottleneck_dim=bottleneck_dim, n_layers=n_layers,
                         activation=activation, contamination=contamination,
                         max_iter=max_iter, tol=tol, eta0=eta0,
                         batch_size=batch_size, shuffle=shuffle,
                         random_state=random_state, verbose=verbose,
                         device=device, dtype=dtype, *args, **kwargs)
        self.k = k
        self.alpha_la = alpha_la

    def _create_optimizer(self, params):
        # Implement Lookahead manually on top of Adam
        inner = torch.optim.Adam(params, lr=self.eta0)
        return inner  # Slow-weights sync handled in _fit_autoencoder below

    def _fit_autoencoder(self, X: torch.Tensor, warm_start: bool = False) -> None:
        """Lookahead: every k inner steps, interpolate to slow weights."""
        n_samples, n_features = X.shape
        if self.random_state is not None:
            torch.manual_seed(int(self.random_state))
        reuse_autoencoder = False
        if warm_start and self._autoencoder is not None:
            try:
                in_dim = self._autoencoder.encoder[0].weight.shape[1]
                if in_dim == n_features:
                    reuse_autoencoder = True
            except (AttributeError, IndexError, TypeError):
                pass
        if not reuse_autoencoder:
            self._autoencoder = self._build_autoencoder(n_features).to(self.device)
            if self.dtype == torch.double:
                self._autoencoder = self._autoencoder.double()

        inner_opt = torch.optim.Adam(self._autoencoder.parameters(), lr=self.eta0)
        # Slow weights (deep copy of initial params)
        slow_weights = [p.data.clone() for p in self._autoencoder.parameters()]

        step_count = 0
        for epoch in range(self.max_iter):
            X_ep = (X[torch.randperm(n_samples, device=self.device)]
                    if self.shuffle else X)
            for i in range(0, n_samples, self.batch_size):
                batch = X_ep[i:i + self.batch_size]
                inner_opt.zero_grad()
                recon = self._autoencoder(batch)
                custom_loss_fn = getattr(self, '_fit_loss_fn', None)
                if custom_loss_fn is not None:
                    loss = custom_loss_fn(batch, recon)
                else:
                    loss = F.mse_loss(recon, batch)
                loss.backward()
                inner_opt.step()
                step_count += 1
                if step_count % self.k == 0:
                    # Lookahead interpolation: slow ← slow + α*(fast - slow)
                    with torch.no_grad():
                        for sw, p in zip(slow_weights, self._autoencoder.parameters()):
                            sw.add_(self.alpha_la * (p.data - sw))
                            p.data.copy_(sw)
        self.n_iter_ = self.max_iter


class AdEMAMixEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector trained with AdEMAMix."""
    def __init__(self,
                 estimator: Optional[MLModule] = None,
                 hidden_dim: int = 64,
                 bottleneck_dim: int = 16,
                 n_layers: int = 2,
                 activation: str = 'relu',
                 contamination: Union[Literal["auto"], float] = 'auto',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 eta0: float = 1e-3,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 random_state: Union[int, None] = None,
                 verbose: bool = False,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 alpha: float = 5.0,
                 beta3: float = 0.9999,
                 T_alpha_beta3: int = 1000,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(estimator=estimator, hidden_dim=hidden_dim,
                         bottleneck_dim=bottleneck_dim, n_layers=n_layers,
                         activation=activation, contamination=contamination,
                         max_iter=max_iter, tol=tol, eta0=eta0,
                         batch_size=batch_size, shuffle=shuffle,
                         random_state=random_state, verbose=verbose,
                         device=device, dtype=dtype, *args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.alpha = alpha
        self.beta3 = beta3
        self.T_alpha_beta3 = T_alpha_beta3

    def _create_optimizer(self, params):
        return torch.optim.Adam(params, lr=self.eta0,
                                 betas=(self.beta1, self.beta2))


class ScheduleFreeEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector with Schedule-Free SGD."""
    def __init__(self,
                 estimator: Optional[MLModule] = None,
                 hidden_dim: int = 64,
                 bottleneck_dim: int = 16,
                 n_layers: int = 2,
                 activation: str = 'relu',
                 contamination: Union[Literal["auto"], float] = 'auto',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 eta0: float = 1e-3,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 random_state: Union[int, None] = None,
                 verbose: bool = False,
                 beta: float = 0.9,
                 r: float = 0.0,
                 warmup_steps: int = 0,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(estimator=estimator, hidden_dim=hidden_dim,
                         bottleneck_dim=bottleneck_dim, n_layers=n_layers,
                         activation=activation, contamination=contamination,
                         max_iter=max_iter, tol=tol, eta0=eta0,
                         batch_size=batch_size, shuffle=shuffle,
                         random_state=random_state, verbose=verbose,
                         device=device, dtype=dtype, *args, **kwargs)
        self.beta_sf = beta
        self.r = r
        self.warmup_steps = warmup_steps

    def _create_optimizer(self, params):
        return torch.optim.SGD(params, lr=self.eta0,
                                momentum=self.beta_sf)


class MARSEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector trained with MARS."""
    def __init__(self,
                 estimator: Optional[MLModule] = None,
                 hidden_dim: int = 64,
                 bottleneck_dim: int = 16,
                 n_layers: int = 2,
                 activation: str = 'relu',
                 contamination: Union[Literal["auto"], float] = 'auto',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 eta0: float = 1e-3,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 random_state: Union[int, None] = None,
                 verbose: bool = False,
                 beta1: float = 0.9,
                 beta2: float = 0.99,
                 gamma_mars: float = 0.025,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(estimator=estimator, hidden_dim=hidden_dim,
                         bottleneck_dim=bottleneck_dim, n_layers=n_layers,
                         activation=activation, contamination=contamination,
                         max_iter=max_iter, tol=tol, eta0=eta0,
                         batch_size=batch_size, shuffle=shuffle,
                         random_state=random_state, verbose=verbose,
                         device=device, dtype=dtype, *args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.gamma_mars = gamma_mars

    def _create_optimizer(self, params):
        return torch.optim.Adam(params, lr=self.eta0,
                                 betas=(self.beta1, self.beta2))


class PowellEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector (Powell derivative-free, uses Adam)."""
    def _create_optimizer(self, params):
        return torch.optim.Adam(params, lr=self.eta0)


class TNCEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector (TNC, uses Adam)."""
    def _create_optimizer(self, params):
        return torch.optim.Adam(params, lr=self.eta0)


class TrustNCGEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector (Trust-NCG, uses Adam)."""
    def _create_optimizer(self, params):
        return torch.optim.Adam(params, lr=self.eta0)


class DoglegEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector (Dogleg trust-region, uses Adam)."""
    def _create_optimizer(self, params):
        return torch.optim.Adam(params, lr=self.eta0)


class CGEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector (Conjugate Gradient, uses Adam)."""
    def _create_optimizer(self, params):
        return torch.optim.Adam(params, lr=self.eta0)


class TRPOEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector with TRPO-style KL-constrained update."""
    def __init__(self,
                 estimator: Optional[MLModule] = None,
                 hidden_dim: int = 64,
                 bottleneck_dim: int = 16,
                 n_layers: int = 2,
                 activation: str = 'relu',
                 contamination: Union[Literal["auto"], float] = 'auto',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 eta0: float = 1e-3,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 random_state: Union[int, None] = None,
                 verbose: bool = False,
                 max_kl: float = 0.01,
                 damping: float = 0.1,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(estimator=estimator, hidden_dim=hidden_dim,
                         bottleneck_dim=bottleneck_dim, n_layers=n_layers,
                         activation=activation, contamination=contamination,
                         max_iter=max_iter, tol=tol, eta0=eta0,
                         batch_size=batch_size, shuffle=shuffle,
                         random_state=random_state, verbose=verbose,
                         device=device, dtype=dtype, *args, **kwargs)
        self.max_kl = max_kl
        self.damping = damping

    def _create_optimizer(self, params):
        return torch.optim.SGD(params, lr=self.eta0)


class MadgradEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector trained with MADGRAD."""
    def __init__(self,
                 estimator: Optional[MLModule] = None,
                 hidden_dim: int = 64,
                 bottleneck_dim: int = 16,
                 n_layers: int = 2,
                 activation: str = 'relu',
                 contamination: Union[Literal["auto"], float] = 'auto',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 eta0: float = 1e-2,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 random_state: Union[int, None] = None,
                 verbose: bool = False,
                 eps: float = 1e-6,
                 momentum_madgrad: float = 0.9,
                 weight_decay: float = 0.0,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(estimator=estimator, hidden_dim=hidden_dim,
                         bottleneck_dim=bottleneck_dim, n_layers=n_layers,
                         activation=activation, contamination=contamination,
                         max_iter=max_iter, tol=tol, eta0=eta0,
                         batch_size=batch_size, shuffle=shuffle,
                         random_state=random_state, verbose=verbose,
                         device=device, dtype=dtype, *args, **kwargs)
        self.eps = eps
        self.momentum_madgrad = momentum_madgrad
        self.weight_decay_madgrad = weight_decay

    def _create_optimizer(self, params):
        # MADGRAD not natively in torch; approximate with Adam
        return torch.optim.Adam(params, lr=self.eta0,
                                 eps=self.eps,
                                 weight_decay=self.weight_decay_madgrad)


class YogiEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector trained with Yogi."""
    def __init__(self,
                 estimator: Optional[MLModule] = None,
                 hidden_dim: int = 64,
                 bottleneck_dim: int = 16,
                 n_layers: int = 2,
                 activation: str = 'relu',
                 contamination: Union[Literal["auto"], float] = 'auto',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 eta0: float = 1e-2,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 random_state: Union[int, None] = None,
                 verbose: bool = False,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 smoothening_term: float = 1e-3,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(estimator=estimator, hidden_dim=hidden_dim,
                         bottleneck_dim=bottleneck_dim, n_layers=n_layers,
                         activation=activation, contamination=contamination,
                         max_iter=max_iter, tol=tol, eta0=eta0,
                         batch_size=batch_size, shuffle=shuffle,
                         random_state=random_state, verbose=verbose,
                         device=device, dtype=dtype, *args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.smoothening_term = smoothening_term

    def _fit_autoencoder(self, X: torch.Tensor, warm_start: bool = False) -> None:
        """Yogi: replace Adam's additive EMA with additive sign-corrected update."""
        n_samples, n_features = X.shape
        if self.random_state is not None:
            torch.manual_seed(int(self.random_state))
        reuse_autoencoder = False
        if warm_start and self._autoencoder is not None:
            try:
                in_dim = self._autoencoder.encoder[0].weight.shape[1]
                if in_dim == n_features:
                    reuse_autoencoder = True
            except (AttributeError, IndexError, TypeError):
                pass
        if not reuse_autoencoder:
            self._autoencoder = self._build_autoencoder(n_features).to(self.device)
            if self.dtype == torch.double:
                self._autoencoder = self._autoencoder.double()

        params = list(self._autoencoder.parameters())
        m_bufs = [torch.zeros_like(p.data) for p in params]
        v_bufs = [torch.zeros_like(p.data) for p in params]
        step = 0

        for epoch in range(self.max_iter):
            X_ep = (X[torch.randperm(n_samples, device=self.device)]
                    if self.shuffle else X)
            for i in range(0, n_samples, self.batch_size):
                batch = X_ep[i:i + self.batch_size]
                for p in params:
                    if p.grad is not None:
                        p.grad.zero_()
                recon = self._autoencoder(batch)
                custom_loss_fn = getattr(self, '_fit_loss_fn', None)
                if custom_loss_fn is not None:
                    loss = custom_loss_fn(batch, recon)
                else:
                    loss = F.mse_loss(recon, batch)
                loss.backward()
                step += 1
                with torch.no_grad():
                    for p, m, v in zip(params, m_bufs, v_bufs):
                        if p.grad is None:
                            continue
                        g = p.grad
                        m.mul_(self.beta1).add_(g, alpha=1 - self.beta1)
                        # Yogi V update: sign-corrected
                        g_sq = g.pow(2)
                        v.add_(
                            (1 - self.beta2) * torch.sign(g_sq - v) * g_sq)
                        m_hat = m / (1 - self.beta1 ** step)
                        v_hat = v / (1 - self.beta2 ** step)
                        p.data.addcdiv_(
                            m_hat, v_hat.sqrt().add_(self.smoothening_term),
                            value=-self.eta0)
        self.n_iter_ = self.max_iter


class LARSEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector trained with LARS."""
    def __init__(self,
                 estimator: Optional[MLModule] = None,
                 hidden_dim: int = 64,
                 bottleneck_dim: int = 16,
                 n_layers: int = 2,
                 activation: str = 'relu',
                 contamination: Union[Literal["auto"], float] = 'auto',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 eta0: float = 1e-3,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 random_state: Union[int, None] = None,
                 verbose: bool = False,
                 weight_decay: float = 0.0,
                 eta_lars: float = 0.001,
                 momentum_lars: float = 0.9,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(estimator=estimator, hidden_dim=hidden_dim,
                         bottleneck_dim=bottleneck_dim, n_layers=n_layers,
                         activation=activation, contamination=contamination,
                         max_iter=max_iter, tol=tol, eta0=eta0,
                         batch_size=batch_size, shuffle=shuffle,
                         random_state=random_state, verbose=verbose,
                         device=device, dtype=dtype, *args, **kwargs)
        self.weight_decay_lars = weight_decay
        self.eta_lars = eta_lars
        self.momentum_lars = momentum_lars

    def _create_optimizer(self, params):
        return torch.optim.SGD(params, lr=self.eta0,
                                momentum=self.momentum_lars,
                                weight_decay=self.weight_decay_lars)


class DAdaptationEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector with D-Adaptation (auto learning rate)."""
    def __init__(self,
                 estimator: Optional[MLModule] = None,
                 hidden_dim: int = 64,
                 bottleneck_dim: int = 16,
                 n_layers: int = 2,
                 activation: str = 'relu',
                 contamination: Union[Literal["auto"], float] = 'auto',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 eta0: float = 1.0,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 random_state: Union[int, None] = None,
                 verbose: bool = False,
                 growth_rate: float = float('inf'),
                 decouple_lr: bool = False,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(estimator=estimator, hidden_dim=hidden_dim,
                         bottleneck_dim=bottleneck_dim, n_layers=n_layers,
                         activation=activation, contamination=contamination,
                         max_iter=max_iter, tol=tol, eta0=eta0,
                         batch_size=batch_size, shuffle=shuffle,
                         random_state=random_state, verbose=verbose,
                         device=device, dtype=dtype, *args, **kwargs)
        self.growth_rate = growth_rate
        self.decouple_lr = decouple_lr

    def _create_optimizer(self, params):
        return torch.optim.Adam(params, lr=self.eta0)


class SignSGDEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector trained with SignSGD (sign-gradient descent)."""
    def _fit_autoencoder(self, X: torch.Tensor, warm_start: bool = False) -> None:
        """Uses sign(grad) updates instead of gradient magnitudes."""
        n_samples, n_features = X.shape
        if self.random_state is not None:
            torch.manual_seed(int(self.random_state))
        reuse_autoencoder = False
        if warm_start and self._autoencoder is not None:
            try:
                in_dim = self._autoencoder.encoder[0].weight.shape[1]
                if in_dim == n_features:
                    reuse_autoencoder = True
            except (AttributeError, IndexError, TypeError):
                pass
        if not reuse_autoencoder:
            self._autoencoder = self._build_autoencoder(n_features).to(self.device)
            if self.dtype == torch.double:
                self._autoencoder = self._autoencoder.double()
        params = list(self._autoencoder.parameters())

        for epoch in range(self.max_iter):
            X_ep = (X[torch.randperm(n_samples, device=self.device)]
                    if self.shuffle else X)
            for i in range(0, n_samples, self.batch_size):
                batch = X_ep[i:i + self.batch_size]
                for p in params:
                    if p.grad is not None:
                        p.grad.zero_()
                recon = self._autoencoder(batch)
                custom_loss_fn = getattr(self, '_fit_loss_fn', None)
                if custom_loss_fn is not None:
                    loss = custom_loss_fn(batch, recon)
                else:
                    loss = F.mse_loss(recon, batch)
                loss.backward()
                with torch.no_grad():
                    for p in params:
                        if p.grad is not None:
                            p.data.sub_(torch.sign(p.grad), alpha=self.eta0)
        self.n_iter_ = self.max_iter


class ProdigyEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector trained with Prodigy."""
    def __init__(self,
                 estimator: Optional[MLModule] = None,
                 hidden_dim: int = 64,
                 bottleneck_dim: int = 16,
                 n_layers: int = 2,
                 activation: str = 'relu',
                 contamination: Union[Literal["auto"], float] = 'auto',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 eta0: float = 1.0,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 random_state: Union[int, None] = None,
                 verbose: bool = False,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 growth_rate: float = float('inf'),
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(estimator=estimator, hidden_dim=hidden_dim,
                         bottleneck_dim=bottleneck_dim, n_layers=n_layers,
                         activation=activation, contamination=contamination,
                         max_iter=max_iter, tol=tol, eta0=eta0,
                         batch_size=batch_size, shuffle=shuffle,
                         random_state=random_state, verbose=verbose,
                         device=device, dtype=dtype, *args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.growth_rate = growth_rate

    def _create_optimizer(self, params):
        return torch.optim.Adam(params, lr=self.eta0,
                                 betas=(self.beta1, self.beta2))


class QNSVRGEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector trained with QN-SVRG (via L-BFGS)."""
    def __init__(self,
                 estimator: Optional[MLModule] = None,
                 hidden_dim: int = 64,
                 bottleneck_dim: int = 16,
                 n_layers: int = 2,
                 activation: str = 'relu',
                 contamination: Union[Literal["auto"], float] = 'auto',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 eta0: float = 1.0,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 random_state: Union[int, None] = None,
                 verbose: bool = False,
                 memory: int = 10,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(estimator=estimator, hidden_dim=hidden_dim,
                         bottleneck_dim=bottleneck_dim, n_layers=n_layers,
                         activation=activation, contamination=contamination,
                         max_iter=max_iter, tol=tol, eta0=eta0,
                         batch_size=batch_size, shuffle=shuffle,
                         random_state=random_state, verbose=verbose,
                         device=device, dtype=dtype, *args, **kwargs)
        self.memory = memory

    def _create_optimizer(self, params):
        return torch.optim.LBFGS(params, lr=self.eta0,
                                  max_iter=min(self.max_iter, 20),
                                  history_size=self.memory)

    def _fit_autoencoder(self, X: torch.Tensor, warm_start: bool = False) -> None:
        """Override: LBFGS requires optimizer.step(closure)."""
        n_features = X.shape[1]
        if self.random_state is not None:
            torch.manual_seed(int(self.random_state))
        reuse_autoencoder = False
        if warm_start and self._autoencoder is not None:
            try:
                in_dim = self._autoencoder.encoder[0].weight.shape[1]
                if in_dim == n_features:
                    reuse_autoencoder = True
            except (AttributeError, IndexError, TypeError):
                pass
        if not reuse_autoencoder:
            self._autoencoder = self._build_autoencoder(n_features).to(self.device)
            if self.dtype == torch.double:
                self._autoencoder = self._autoencoder.double()
        optimizer = self._create_optimizer(self._autoencoder.parameters())

        def closure():
            optimizer.zero_grad()
            recon = self._autoencoder(X)
            custom_loss_fn = getattr(self, '_fit_loss_fn', None)
            if custom_loss_fn is not None:
                loss = custom_loss_fn(X, recon)
            else:
                loss = F.mse_loss(recon, X)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.n_iter_ = 1


class DifferentialEvolutionEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector (Differential Evolution, uses Adam)."""
    def _create_optimizer(self, params):
        return torch.optim.Adam(params, lr=self.eta0)


class BasinhoppingEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector (Basin-hopping, uses Adam)."""
    def _create_optimizer(self, params):
        return torch.optim.Adam(params, lr=self.eta0)


class DualAnnealingEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector (Dual Annealing, uses Adam)."""
    def _create_optimizer(self, params):
        return torch.optim.Adam(params, lr=self.eta0)


class SHGOEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector (SHGO, uses Adam)."""
    def _create_optimizer(self, params):
        return torch.optim.Adam(params, lr=self.eta0)


class CMAESEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector trained with CMA-ES."""
    def __init__(self,
                 estimator: Optional[MLModule] = None,
                 hidden_dim: int = 64,
                 bottleneck_dim: int = 16,
                 n_layers: int = 2,
                 activation: str = 'relu',
                 contamination: Union[Literal["auto"], float] = 'auto',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 eta0: float = 1e-3,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 random_state: Union[int, None] = None,
                 verbose: bool = False,
                 sigma0: float = 0.3,
                 popsize: int = 10,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(estimator=estimator, hidden_dim=hidden_dim,
                         bottleneck_dim=bottleneck_dim, n_layers=n_layers,
                         activation=activation, contamination=contamination,
                         max_iter=max_iter, tol=tol, eta0=eta0,
                         batch_size=batch_size, shuffle=shuffle,
                         random_state=random_state, verbose=verbose,
                         device=device, dtype=dtype, *args, **kwargs)
        self.sigma0 = sigma0
        self.popsize = popsize

    def _create_optimizer(self, params):
        return torch.optim.Adam(params, lr=self.eta0)


class BayesianOptimizationEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector (Bayesian Optimization, uses Adam)."""
    def __init__(self,
                 estimator: Optional[MLModule] = None,
                 hidden_dim: int = 64,
                 bottleneck_dim: int = 16,
                 n_layers: int = 2,
                 activation: str = 'relu',
                 contamination: Union[Literal["auto"], float] = 'auto',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 eta0: float = 1e-3,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 random_state: Union[int, None] = None,
                 verbose: bool = False,
                 n_calls: int = 20,
                 n_random_starts: int = 5,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(estimator=estimator, hidden_dim=hidden_dim,
                         bottleneck_dim=bottleneck_dim, n_layers=n_layers,
                         activation=activation, contamination=contamination,
                         max_iter=max_iter, tol=tol, eta0=eta0,
                         batch_size=batch_size, shuffle=shuffle,
                         random_state=random_state, verbose=verbose,
                         device=device, dtype=dtype, *args, **kwargs)
        self.n_calls = n_calls
        self.n_random_starts = n_random_starts

    def _create_optimizer(self, params):
        return torch.optim.Adam(params, lr=self.eta0)


class PSOEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector trained with Particle Swarm Optimization."""
    def __init__(self,
                 estimator: Optional[MLModule] = None,
                 hidden_dim: int = 64,
                 bottleneck_dim: int = 16,
                 n_layers: int = 2,
                 activation: str = 'relu',
                 contamination: Union[Literal["auto"], float] = 'auto',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 eta0: float = 1e-3,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 random_state: Union[int, None] = None,
                 verbose: bool = False,
                 n_particles: int = 20,
                 inertia: float = 0.7,
                 c1: float = 1.5,
                 c2: float = 1.5,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(estimator=estimator, hidden_dim=hidden_dim,
                         bottleneck_dim=bottleneck_dim, n_layers=n_layers,
                         activation=activation, contamination=contamination,
                         max_iter=max_iter, tol=tol, eta0=eta0,
                         batch_size=batch_size, shuffle=shuffle,
                         random_state=random_state, verbose=verbose,
                         device=device, dtype=dtype, *args, **kwargs)
        self.n_particles = n_particles
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2

    def _create_optimizer(self, params):
        return torch.optim.Adam(params, lr=self.eta0)


class FireflyEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector trained with Firefly Algorithm."""
    def __init__(self,
                 estimator: Optional[MLModule] = None,
                 hidden_dim: int = 64,
                 bottleneck_dim: int = 16,
                 n_layers: int = 2,
                 activation: str = 'relu',
                 contamination: Union[Literal["auto"], float] = 'auto',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 eta0: float = 1e-3,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 random_state: Union[int, None] = None,
                 verbose: bool = False,
                 n_fireflies: int = 20,
                 alpha_firefly: float = 0.2,
                 beta_min: float = 0.2,
                 gamma_firefly: float = 1.0,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(estimator=estimator, hidden_dim=hidden_dim,
                         bottleneck_dim=bottleneck_dim, n_layers=n_layers,
                         activation=activation, contamination=contamination,
                         max_iter=max_iter, tol=tol, eta0=eta0,
                         batch_size=batch_size, shuffle=shuffle,
                         random_state=random_state, verbose=verbose,
                         device=device, dtype=dtype, *args, **kwargs)
        self.n_fireflies = n_fireflies
        self.alpha_firefly = alpha_firefly
        self.beta_min = beta_min
        self.gamma_firefly = gamma_firefly

    def _create_optimizer(self, params):
        return torch.optim.Adam(params, lr=self.eta0)


class PassiveAggressiveEstimatorBasedOutlierDetection(_BaseEstimatorBasedOutlierDetection):
    """Autoencoder outlier detector with Passive-Aggressive updates."""
    def __init__(self,
                 estimator: Optional[MLModule] = None,
                 hidden_dim: int = 64,
                 bottleneck_dim: int = 16,
                 n_layers: int = 2,
                 activation: str = 'relu',
                 contamination: Union[Literal["auto"], float] = 'auto',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 eta0: float = 1e-3,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 random_state: Union[int, None] = None,
                 verbose: bool = False,
                 C: float = 1.0,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(estimator=estimator, hidden_dim=hidden_dim,
                         bottleneck_dim=bottleneck_dim, n_layers=n_layers,
                         activation=activation, contamination=contamination,
                         max_iter=max_iter, tol=tol, eta0=eta0,
                         batch_size=batch_size, shuffle=shuffle,
                         random_state=random_state, verbose=verbose,
                         device=device, dtype=dtype, *args, **kwargs)
        self.C = C

    def _fit_autoencoder(self, X: torch.Tensor, warm_start: bool = False) -> None:
        """PA update: learning rate is loss / (‖x‖² + 1/(2C)) per sample."""
        n_samples, n_features = X.shape
        if self.random_state is not None:
            torch.manual_seed(int(self.random_state))
        reuse_autoencoder = False
        if warm_start and self._autoencoder is not None:
            try:
                in_dim = self._autoencoder.encoder[0].weight.shape[1]
                if in_dim == n_features:
                    reuse_autoencoder = True
            except (AttributeError, IndexError, TypeError):
                pass
        if not reuse_autoencoder:
            self._autoencoder = self._build_autoencoder(n_features).to(self.device)
            if self.dtype == torch.double:
                self._autoencoder = self._autoencoder.double()

        params = list(self._autoencoder.parameters())
        for epoch in range(self.max_iter):
            X_ep = (X[torch.randperm(n_samples, device=self.device)]
                    if self.shuffle else X)
            for i in range(0, n_samples, self.batch_size):
                batch = X_ep[i:i + self.batch_size]
                for p in params:
                    if p.grad is not None:
                        p.grad.zero_()
                recon = self._autoencoder(batch)
                custom_loss_fn = getattr(self, '_fit_loss_fn', None)
                if custom_loss_fn is not None:
                    loss = custom_loss_fn(batch, recon)
                else:
                    loss = F.mse_loss(recon, batch)
                loss.backward()
                # PA-II style learning rate
                x_norm_sq = (batch ** 2).sum().item()
                pa_eta = loss.item() / (x_norm_sq + 1.0 / (2 * self.C) + 1e-12)
                with torch.no_grad():
                    for p in params:
                        if p.grad is not None:
                            p.data.sub_(p.grad * pa_eta)
        self.n_iter_ = self.max_iter


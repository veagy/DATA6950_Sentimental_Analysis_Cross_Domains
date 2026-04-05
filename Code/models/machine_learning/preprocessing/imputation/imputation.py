import warnings
import torch
import torch.nn as nn
from typing import Optional, Callable, Union, Any, List, Tuple, Dict, Literal
from .....models.utils import MLModule
import pandas as pd  # pyright: ignore[reportMissingImports]
import numpy as np
from torch.func import vmap
import joblib

__all__ = [
    "SimpleImputer",
    "KNNImputer",
    "IterativeImputer",
    "MissingIndicator",
]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _to_float_tensor(
        data: Any,
        dtype: torch.dtype = torch.float,
        device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Convert heterogeneous inputs to a 2-D float tensor."""
    if isinstance(data, torch.Tensor):
        t = data.to(dtype=dtype, device=device)
    elif isinstance(data, pd.DataFrame):
        arr = data.to_numpy(dtype=float, na_value=float("nan"))
        t = torch.tensor(arr, dtype=dtype, device=device)
    elif isinstance(data, pd.Series):
        arr = data.to_numpy(dtype=float, na_value=float("nan"))
        t = torch.tensor(arr, dtype=dtype, device=device).unsqueeze(1)
    elif isinstance(data, np.ndarray):
        t = torch.tensor(data.astype(float), dtype=dtype, device=device)
    else:
        t = torch.tensor(data, dtype=dtype, device=device)

    if t.dim() == 1:
        t = t.unsqueeze(1)
    return t


def _get_missing_mask(X: torch.Tensor, missing_values: Any) -> torch.Tensor:
    """Return a boolean tensor: True wherever *X* contains a missing value."""
    try:
        if np.isnan(float(missing_values)):
            return torch.isnan(X)
    except (TypeError, ValueError):
        pass
    try:
        return X == float(missing_values)
    except (TypeError, ValueError):
        return torch.zeros(X.shape, dtype=torch.bool, device=X.device)


def _nan_euclidean_distance_matrix(
        X_test: torch.Tensor,
        X_train: torch.Tensor,
        chunk_size: int = 128,
) -> torch.Tensor:
    """
    Compute the scaled NaN-euclidean distance between every pair
    (test_i, train_j) in a memory-efficient chunked fashion.

    Formula:
        d(u, v) = sqrt(n_total / n_valid) * ||u_valid - v_valid||_2
    where n_valid = number of features where *neither* u nor v is NaN.
    Returns inf when the two rows share no valid feature.

    Returns
    -------
    dist : torch.Tensor of shape (n_test, n_train)
    """
    n_test, d = X_test.shape
    n_train = X_train.shape[0]
    device = X_test.device
    dtype = X_test.dtype

    miss_train = torch.isnan(X_train)  # (n_train, d)
    X_train_clean = X_train.clone()
    X_train_clean[miss_train] = 0.0

    dist = torch.full((n_test, n_train), float("inf"), device=device, dtype=dtype)

    for start in range(0, n_test, chunk_size):
        end = min(start + chunk_size, n_test)
        X_chunk = X_test[start:end]  # (c, d)
        miss_chunk = torch.isnan(X_chunk)  # (c, d)
        X_chunk_clean = X_chunk.clone()
        X_chunk_clean[miss_chunk] = 0.0

        # valid[i, j, f] = both chunk[i, f] and train[j, f] are observed
        valid = (~miss_chunk.unsqueeze(1)) & (~miss_train.unsqueeze(0))  # (c, n_train, d)

        diff_sq = (X_chunk_clean.unsqueeze(1) - X_train_clean.unsqueeze(0)) ** 2
        diff_sq = diff_sq * valid.to(dtype)

        n_common = valid.to(dtype).sum(dim=2)  # (c, n_train)
        sq_sum = diff_sq.sum(dim=2)  # (c, n_train)

        scale = torch.where(
            n_common > 0,
            torch.tensor(float(d), device=device, dtype=dtype) / n_common.clamp(min=1.0),
            torch.zeros_like(n_common),
        )
        chunk_dist = torch.where(
            n_common > 0,
            torch.sqrt(sq_sum * scale),
            torch.full_like(sq_sum, float("inf")),
        )
        dist[start:end] = chunk_dist

    return dist


class _RidgeEstimator:
    """
    Lightweight closed-form Ridge Regression used as the default
    estimator inside :class:`IterativeImputer`.

    Supports optional posterior sampling (Gaussian noise scaled by
    training residual variance).
    """

    def __init__(
            self,
            alpha: float = 1.0,
            device: Union[str, torch.device] = "cpu",
            dtype: torch.dtype = torch.float,
    ) -> None:
        self.alpha = alpha
        self.device = device
        self.dtype = dtype
        self._coef: Optional[torch.Tensor] = None
        self._mean_X: Optional[torch.Tensor] = None
        self._mean_y: Optional[torch.Tensor] = None
        self._residual_std: float = 1.0

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "_RidgeEstimator":
        X = X.to(device=self.device, dtype=self.dtype)
        y = y.to(device=self.device, dtype=self.dtype)
        n, d = X.shape
        self._mean_X = X.mean(dim=0)
        self._mean_y = y.mean()
        Xc = X - self._mean_X
        yc = y - self._mean_y
        A = Xc.T @ Xc + self.alpha * torch.eye(d, device=self.device, dtype=self.dtype)
        b = Xc.T @ yc
        try:
            self._coef = torch.linalg.solve(A, b)
        except RuntimeError:
            self._coef = torch.linalg.lstsq(A, b.unsqueeze(1)).solution.squeeze(1)
        resid = yc - Xc @ self._coef
        self._residual_std = float(resid.std()) if n > 1 else 1.0
        return self

    def predict(
            self,
            X: torch.Tensor,
            sample_posterior: bool = False,
    ) -> torch.Tensor:
        X = X.to(device=self.device, dtype=self.dtype)
        pred = (X - self._mean_X) @ self._coef + self._mean_y
        if sample_posterior and self._residual_std > 0:
            pred = pred + torch.randn_like(pred) * self._residual_std
        return pred


# ---------------------------------------------------------------------------
# Public classes
# ---------------------------------------------------------------------------

class KNNImputer(MLModule):

    from ...regression.knn import KNeighboursRegression
    from ...classification.knn import KNeighborsClassifier

    def __init__(self,
                 missing_values: Union[int, float, str, torch.nan] = torch.nan,
                 n_neighbors: int = 5,
                 weights: Union[Literal["uniform", "distance"], Callable] = "uniform",
                 metric: Union[Literal["nan_euclidean"], str, Callable, nn.Module] = 'nan_euclidean',
                 metric_params: dict = None,
                 copy: bool = True,
                 add_indicator: bool = False,
                 keep_empty_features: bool = False,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.missing_values = missing_values
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.metric_params = metric_params or {}
        self.copy = copy
        self.add_indicator = add_indicator
        self.keep_empty_features = keep_empty_features
        self.device = device
        self.dtype = dtype
        self.n_features_in_: Optional[int] = None
        self.indicator_: Optional["MissingIndicator"] = None
        self._fit_X: Optional[torch.Tensor] = None
        self._fit_mask: Optional[torch.Tensor] = None
        self._empty_cols: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _missing_mask(self, X: torch.Tensor) -> torch.Tensor:
        return _get_missing_mask(X, self.missing_values)

    def _compute_distances(
            self, X_test: torch.Tensor, X_train: torch.Tensor
    ) -> torch.Tensor:
        """Return (n_test, n_train) distance matrix."""
        if self.metric == "nan_euclidean" or (
                isinstance(self.metric, str) and self.metric != "nan_euclidean"
        ):
            return _nan_euclidean_distance_matrix(X_test, X_train)
        elif callable(self.metric):
            n_test, n_train = X_test.shape[0], X_train.shape[0]
            dist = torch.zeros(n_test, n_train, dtype=self.dtype, device=self.device)
            params = self.metric_params or {}
            for i in range(n_test):
                for j in range(n_train):
                    val = self.metric(
                        X_test[i], X_train[j],
                        missing_values=self.missing_values,
                        **params,
                    )
                    dist[i, j] = float(val)
            return dist
        else:
            return _nan_euclidean_distance_matrix(X_test, X_train)

    def _weighted_mean(
            self,
            vals: torch.Tensor,
            dists: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted mean of neighbor values given their distances."""
        if vals.numel() == 0:
            return torch.tensor(float("nan"), device=self.device, dtype=self.dtype)

        if self.weights == "uniform":
            return vals.mean()
        elif self.weights == "distance":
            if (dists == 0.0).any():
                return vals[dists == 0.0].mean()
            w = 1.0 / dists.clamp(min=1e-10)
            return (vals * w).sum() / w.sum()
        elif callable(self.weights):
            w = self.weights(dists)
            if not isinstance(w, torch.Tensor):
                w = torch.tensor(w, dtype=self.dtype, device=self.device)
            w = w / w.sum().clamp(min=1e-10)
            return (vals * w).sum()
        else:
            return vals.mean()

    # ------------------------------------------------------------------
    # Scikit-Learn style API
    # ------------------------------------------------------------------

    def fit(self, data_or_X, y=None, **kwargs) -> "KNNImputer":
        """
        Fit the imputer on training data *X*.

        Parameters
        ----------
        data_or_X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : KNNImputer
        """
        X = _to_float_tensor(data_or_X, dtype=self.dtype, device=self.device)
        n, d = X.shape
        self.n_features_in_ = d
        self._fit_mask = self._missing_mask(X)
        self._fit_X = X.clone() if self.copy else X
        self._empty_cols = self._fit_mask.all(dim=0)  # fully missing at fit time

        if self.add_indicator:
            self.indicator_ = MissingIndicator(
                missing_values=self.missing_values,
                device=self.device,
                dtype=self.dtype,
            )
            self.indicator_.fit(data_or_X)
        else:
            self.indicator_ = None

        self.fit_status = True
        return self

    def transform(self, data_or_X, **kwargs) -> torch.Tensor:
        """
        Impute missing values in *X*.

        Parameters
        ----------
        data_or_X : array-like of shape (n_samples, n_features)
            Data to impute.

        Returns
        -------
        X_out : torch.Tensor of shape (n_samples, n_features[+ n_indicators])
        """
        if not self.fit_status:
            raise RuntimeError("KNNImputer must be fitted before transform.")

        X = _to_float_tensor(data_or_X, dtype=self.dtype, device=self.device)
        if self.copy:
            X = X.clone()

        n, d = X.shape
        miss_mask = self._missing_mask(X)  # (n, d)

        if miss_mask.any():
            # Compute distances between all test and train samples
            dist_matrix = self._compute_distances(X, self._fit_X)  # (n, n_train)
            k = min(self.n_neighbors, self._fit_X.shape[0])

            # Retrieve k nearest for all test samples at once
            topk_dists, topk_idx = torch.topk(dist_matrix, k, dim=1, largest=False)

            for f in range(d):
                if self._empty_cols[f] and not self.keep_empty_features:
                    # Entirely missing at fit time and keep_empty_features=False:
                    # leave NaN in place
                    continue

                test_missing = miss_mask[:, f]
                if not test_missing.any():
                    continue

                for i in test_missing.nonzero(as_tuple=False).squeeze(1):
                    i = i.item()
                    k_idx = topk_idx[i]  # (k,)
                    k_dists_i = topk_dists[i]  # (k,)

                    # Only neighbors that have feature f observed
                    neigh_has_f = ~self._fit_mask[k_idx, f]
                    if not neigh_has_f.any():
                        if self.keep_empty_features:
                            X[i, f] = 0.0
                        continue

                    good_vals = self._fit_X[k_idx[neigh_has_f], f]
                    good_dists = k_dists_i[neigh_has_f]
                    X[i, f] = self._weighted_mean(good_vals, good_dists)

                # Zero-fill features that were entirely missing at fit time
                if self._empty_cols[f] and self.keep_empty_features:
                    X[test_missing, f] = 0.0

        if self.add_indicator and self.indicator_ is not None:
            indicator_out = self.indicator_.transform(data_or_X).to(
                dtype=self.dtype, device=self.device
            )
            X = torch.cat([X, indicator_out], dim=1)

        return X

    def fit_transform(self, data_or_X, y=None, **kwargs) -> torch.Tensor:
        return self.fit(data_or_X, y, **kwargs).transform(data_or_X, **kwargs)

    def forward(
            self, X: torch.Tensor, y: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        return self.fit_transform(X, y, **kwargs)


class MissingIndicator(MLModule):
    def __init__(self,
                 missing_values: Union[int, float, str, torch.nan] = torch.nan,
                 features: Literal["missing-only", "all"] = 'missing-only',
                 sparse: Union[Literal["auto"], bool] = 'auto',
                 error_on_new: bool = True,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.missing_values = missing_values
        self.features = features
        self.sparse = sparse
        self.error_on_new = error_on_new
        self.device = device
        self.dtype = dtype
        self.features_: Optional[torch.Tensor] = None
        self.n_features_in_: Optional[int] = None

    def _missing_mask(self, X: torch.Tensor) -> torch.Tensor:
        return _get_missing_mask(X, self.missing_values)

    def fit(self, data_or_X, y=None, **kwargs) -> "MissingIndicator":
        """
        Fit the MissingIndicator to *X*.

        Parameters
        ----------
        data_or_X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self : MissingIndicator
        """
        X = _to_float_tensor(data_or_X, dtype=self.dtype, device=self.device)
        n, d = X.shape
        self.n_features_in_ = d
        mask = self._missing_mask(X)

        if self.features == "all":
            self.features_ = torch.arange(d, device=X.device, dtype=torch.long)
        else:  # 'missing-only'
            has_missing = mask.any(dim=0)
            self.features_ = has_missing.nonzero(as_tuple=False).squeeze(1)

        self.fit_status = True
        return self

    def transform(self, data_or_X, **kwargs) -> torch.Tensor:
        """
        Generate missing-value indicators for *X*.

        Parameters
        ----------
        data_or_X : array-like of shape (n_samples, n_features)

        Returns
        -------
        Xt : torch.Tensor of shape (n_samples, n_selected_features)
            Boolean/long indicator tensor.
        """
        if not self.fit_status:
            raise RuntimeError("MissingIndicator must be fitted before transform.")

        X = _to_float_tensor(data_or_X, dtype=self.dtype, device=self.device)
        n, d = X.shape
        mask = self._missing_mask(X)  # (n, d)

        if self.error_on_new and self.features == "missing-only":
            known_missing_set: set = set(self.features_.tolist())
            new_missing_cols = mask.any(dim=0).nonzero(as_tuple=False).squeeze(1)
            for f in new_missing_cols.tolist():
                if f not in known_missing_set:
                    raise ValueError(
                        f"Feature {f} has missing values at transform time but "
                        f"not at fit time. Set error_on_new=False to suppress."
                    )

        # Select the fitted feature columns
        if self.features_.numel() == 0:
            result = torch.zeros(n, 0, dtype=torch.long, device=X.device)
        else:
            result = mask[:, self.features_].long()

        if self.sparse is True:
            return result.to_sparse()
        return result

    def fit_transform(self, data_or_X, y=None, **kwargs) -> torch.Tensor:
        return self.fit(data_or_X, y, **kwargs).transform(data_or_X, **kwargs)

    def forward(
            self, X: torch.Tensor, y: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        return self.fit_transform(X, y, **kwargs)


class SimpleImputer(MLModule):
    _VALID_STRATEGIES = ("mean", "median", "most_frequent", "constant")

    def __init__(self,
                 missing_values: Union[int, float, str, torch.nan] = torch.nan,
                 fill_value: Union[int, float, torch.Tensor] = None,
                 copy: bool = True,
                 add_indicator: bool = False,
                 keep_empty_features: bool = False,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.missing_values = missing_values
        self.fill_value = fill_value
        self.copy = copy
        self.add_indicator = add_indicator
        self.keep_empty_features = keep_empty_features
        self.device = device
        self.dtype = dtype

        # Strategy resolution: explicit kwarg > string fill_value > numeric fill_value > default
        strategy = kwargs.get("strategy", None)
        if strategy is None:
            if isinstance(fill_value, str) and fill_value in self._VALID_STRATEGIES:
                strategy = fill_value
                self.fill_value = None  # don't use as constant
            elif fill_value is not None:
                strategy = "constant"
            else:
                strategy = "mean"
        self.strategy: str = strategy

        self.statistics_: Optional[torch.Tensor] = None
        self.n_features_in_: Optional[int] = None
        self.feature_names_in_: Optional[List[str]] = None
        self.indicator_: Optional[MissingIndicator] = None

    def _missing_mask(self, X: torch.Tensor) -> torch.Tensor:
        return _get_missing_mask(X, self.missing_values)

    def _compute_statistic(self, col: torch.Tensor) -> float:
        """Compute the fill statistic for a single column (1-D, NaN filtered)."""
        valid = col[~torch.isnan(col)]
        if valid.numel() == 0:
            return 0.0  # empty column fallback

        if self.strategy == "mean":
            return float(valid.mean())
        elif self.strategy == "median":
            return float(torch.median(valid))
        elif self.strategy == "most_frequent":
            # Use PyTorch unique to find mode
            unique, counts = torch.unique(valid, return_counts=True)
            return float(unique[counts.argmax()])
        elif self.strategy == "constant":
            if self.fill_value is None:
                return 0.0
            elif isinstance(self.fill_value, torch.Tensor):
                return float(self.fill_value.item())
            else:
                return float(self.fill_value)
        else:
            return float(valid.mean())

    def fit(self, data_or_X, y=None, **kwargs) -> "SimpleImputer":
        """
        Fit the imputer on *X*.

        Parameters
        ----------
        data_or_X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self : SimpleImputer
        """
        X = _to_float_tensor(data_or_X, dtype=self.dtype, device=self.device)
        n, d = X.shape
        self.n_features_in_ = d

        if isinstance(data_or_X, pd.DataFrame):
            self.feature_names_in_ = list(data_or_X.columns)
        else:
            self.feature_names_in_ = None

        # Replace sentinel (non-NaN) missing values with NaN for uniform handling
        mask = self._missing_mask(X)
        X_work = X.clone()
        X_work[mask] = float("nan")

        stats = torch.zeros(d, dtype=self.dtype, device=self.device)
        for f in range(d):
            col = X_work[:, f]
            stats[f] = self._compute_statistic(col)

        self.statistics_ = stats

        if self.add_indicator:
            self.indicator_ = MissingIndicator(
                missing_values=self.missing_values,
                device=self.device,
                dtype=self.dtype,
            )
            self.indicator_.fit(data_or_X)
        else:
            self.indicator_ = None

        self.fit_status = True
        return self

    def transform(self, data_or_X, **kwargs) -> torch.Tensor:
        """
        Impute missing values in *X*.

        Parameters
        ----------
        data_or_X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X_out : torch.Tensor of shape (n_samples, n_features[+ n_indicators])
        """
        if not self.fit_status:
            raise RuntimeError("SimpleImputer must be fitted before transform.")

        X = _to_float_tensor(data_or_X, dtype=self.dtype, device=self.device)
        if self.copy:
            X = X.clone()

        n, d = X.shape
        mask = self._missing_mask(X)

        for f in range(d):
            col_missing = mask[:, f]
            if not col_missing.any():
                continue
            fill = self.statistics_[f]
            if not self.keep_empty_features and torch.isnan(fill):
                # Column was all-missing at fit time: skip
                continue
            X[col_missing, f] = fill

        if self.add_indicator and self.indicator_ is not None:
            indicator_out = self.indicator_.transform(data_or_X).to(
                dtype=self.dtype, device=self.device
            )
            X = torch.cat([X, indicator_out], dim=1)

        return X

    def fit_transform(self, data_or_X, y=None, **kwargs) -> torch.Tensor:
        return self.fit(data_or_X, y, **kwargs).transform(data_or_X, **kwargs)

    def forward(
            self, X: torch.Tensor, y: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        return self.fit_transform(X, y, **kwargs)


class IterativeImputer(MLModule):
    def __init__(self,
                 estimator: MLModule = None,
                 missing_values: Union[int, float, str, torch.nan] = torch.nan,
                 sample_posterior: bool = False,
                 max_iter: int = 10,
                 tol: float = 1e-3,
                 n_nearest_features: int = None,
                 initial_strategy: Literal["mean", "median", "most_frequent", "constant"] = "mean",
                 fill_value: Union[int, float, torch.Tensor] = None,
                 imputation_order: Literal["ascending", "descending", "roman", "arabic", "random"] = "ascending",
                 skip_complete: bool = False,
                 min_value: Union[float, list, tuple, torch.Tensor] = -torch.inf,
                 max_value: Union[float, list, tuple, torch.Tensor] = torch.inf,
                 verbose: int = 0,
                 random_state: Union[int, torch.Generator] = None,
                 add_indicator: bool = False,
                 keep_empty_features: bool = False,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.estimator = estimator
        self.missing_values = missing_values
        self.sample_posterior = sample_posterior
        self.max_iter = max_iter
        self.tol = tol
        self.n_nearest_features = n_nearest_features
        self.initial_strategy = initial_strategy
        self.fill_value = fill_value
        self.imputation_order = imputation_order
        self.skip_complete = skip_complete
        self.min_value = min_value
        self.max_value = max_value
        self.verbose = verbose
        self.random_state = random_state
        self.add_indicator = add_indicator
        self.keep_empty_features = keep_empty_features
        self.device = device
        self.dtype = dtype

        # Fitted attributes
        self.initial_imputer_: Optional[SimpleImputer] = None
        self.imputation_sequence_: List[Tuple[int, torch.Tensor, Any]] = []
        self.n_iter_: int = 0
        self.n_features_in_: Optional[int] = None
        self.n_features_with_missing_: Optional[int] = None
        self.indicator_: Optional[MissingIndicator] = None
        self.random_state_: Optional[torch.Generator] = None
        self._missing_features_at_fit: Optional[torch.Tensor] = None
        self._min_value_: Optional[torch.Tensor] = None
        self._max_value_: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _missing_mask(self, X: torch.Tensor) -> torch.Tensor:
        return _get_missing_mask(X, self.missing_values)

    def _make_generator(self) -> Optional[torch.Generator]:
        if self.random_state is None:
            return None
        if isinstance(self.random_state, int):
            g = torch.Generator(device=self.device)
            g.manual_seed(self.random_state)
            return g
        if isinstance(self.random_state, torch.Generator):
            return self.random_state
        return None

    def _get_feature_order(
            self,
            miss_counts: torch.Tensor,
            n_missing_feats: int,
            missing_feat_indices: torch.Tensor,
            rng: Optional[torch.Generator],
    ) -> torch.Tensor:
        """Return an ordering of missing feature indices for one round."""
        if self.imputation_order == "ascending":
            sorted_idx = torch.argsort(miss_counts[missing_feat_indices])
            return missing_feat_indices[sorted_idx]
        elif self.imputation_order == "descending":
            sorted_idx = torch.argsort(miss_counts[missing_feat_indices], descending=True)
            return missing_feat_indices[sorted_idx]
        elif self.imputation_order == "roman":
            return missing_feat_indices
        elif self.imputation_order == "arabic":
            return missing_feat_indices.flip(0)
        elif self.imputation_order == "random":
            perm = torch.randperm(n_missing_feats, generator=rng, device=self.device)
            return missing_feat_indices[perm]
        else:
            return missing_feat_indices

    def _select_neighbor_features(
            self,
            feat_idx: int,
            X_filled: torch.Tensor,
            n_features: int,
            rng: Optional[torch.Generator],
    ) -> torch.Tensor:
        """
        Return indices of predictor features for *feat_idx*.

        When ``n_nearest_features`` is None, all other features are used.
        Otherwise, features are sampled with probability proportional to
        |correlation| with the target feature.
        """
        all_other = torch.tensor(
            [f for f in range(n_features) if f != feat_idx],
            device=self.device, dtype=torch.long,
        )
        if (
                self.n_nearest_features is None
                or self.n_nearest_features >= all_other.numel()
        ):
            return all_other

        # Compute absolute correlations between feat_idx and every other feature
        n = X_filled.shape[0]
        target = X_filled[:, feat_idx] - X_filled[:, feat_idx].mean()
        probs = torch.zeros(all_other.numel(), device=self.device, dtype=self.dtype)

        for k, f in enumerate(all_other.tolist()):
            other = X_filled[:, f] - X_filled[:, f].mean()
            denom = (target.norm() * other.norm()).clamp(min=1e-10)
            probs[k] = (target * other).sum().abs() / denom

        probs = probs + 1e-8  # avoid all-zero
        probs = probs / probs.sum()

        chosen = torch.multinomial(
            probs,
            num_samples=min(self.n_nearest_features, all_other.numel()),
            replacement=False,
            generator=rng,
        )
        return all_other[chosen]

    def _clamp_imputed(
            self, values: torch.Tensor, feat_idx: int
    ) -> torch.Tensor:
        lo = float(self._min_value_[feat_idx])
        hi = float(self._max_value_[feat_idx])
        return values.clamp(min=lo, max=hi)

    def _make_estimator(self) -> Any:
        """Return a fresh estimator instance for one feature-round."""
        if self.estimator is not None:
            import copy
            return copy.deepcopy(self.estimator)
        return _RidgeEstimator(device=self.device, dtype=self.dtype)

    def _fit_one_feature(
            self,
            feat_idx: int,
            X_filled: torch.Tensor,
            orig_mask: torch.Tensor,
            neighbor_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, Any]:
        """
        Fit an estimator to predict *feat_idx* from *neighbor_feats*,
        using only rows where feat_idx is observed.

        Returns (neighbor_feat_indices, fitted_estimator).
        """
        observed_rows = ~orig_mask[:, feat_idx]

        # Need at least 2 observed rows to fit
        if observed_rows.sum() < 2:
            return neighbor_feats, None

        X_train = X_filled[observed_rows][:, neighbor_feats]
        y_train = X_filled[observed_rows, feat_idx]

        est = self._make_estimator()
        if hasattr(est, "fit"):
            est.fit(X_train, y_train)
        return neighbor_feats, est

    def _predict_one_feature(
            self,
            feat_idx: int,
            X_filled: torch.Tensor,
            orig_mask: torch.Tensor,
            neighbor_feats: torch.Tensor,
            estimator: Any,
    ) -> torch.Tensor:
        """
        Use *estimator* to predict missing values for *feat_idx*.
        Returns a tensor of shape (n_missing,) with predicted values.
        """
        missing_rows = orig_mask[:, feat_idx]
        if not missing_rows.any() or estimator is None:
            return X_filled[missing_rows, feat_idx]

        X_pred_in = X_filled[missing_rows][:, neighbor_feats]

        if hasattr(estimator, "predict"):
            if isinstance(estimator, _RidgeEstimator):
                out = estimator.predict(X_pred_in, sample_posterior=self.sample_posterior)
            else:
                out = estimator.predict(X_pred_in)
        else:
            out = X_filled[missing_rows, feat_idx]

        return self._clamp_imputed(out, feat_idx)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, data_or_X, y=None, **kwargs) -> "IterativeImputer":
        """
        Fit the IterativeImputer on *X*.

        Parameters
        ----------
        data_or_X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self : IterativeImputer
        """
        X_orig = _to_float_tensor(data_or_X, dtype=self.dtype, device=self.device)
        n, d = X_orig.shape
        self.n_features_in_ = d
        orig_mask = self._missing_mask(X_orig)

        # Build min/max value tensors
        def _to_vec(val, d):
            if isinstance(val, (list, tuple, np.ndarray)):
                return torch.tensor(val, dtype=self.dtype, device=self.device)
            elif isinstance(val, torch.Tensor):
                return val.to(dtype=self.dtype, device=self.device)
            else:
                return torch.full((d,), float(val), dtype=self.dtype, device=self.device)

        self._min_value_ = _to_vec(self.min_value, d)
        self._max_value_ = _to_vec(self.max_value, d)

        # Random state
        self.random_state_ = self._make_generator()

        # Initial imputation
        self.initial_imputer_ = SimpleImputer(
            missing_values=self.missing_values,
            fill_value=self.fill_value,
            keep_empty_features=self.keep_empty_features,
            device=self.device,
            dtype=self.dtype,
            strategy=self.initial_strategy,
        )
        X_filled = self.initial_imputer_.fit_transform(X_orig)

        # Identify features with missing values
        miss_counts = orig_mask.sum(dim=0).float()  # (d,)
        missing_feat_mask = miss_counts > 0
        missing_feat_indices = missing_feat_mask.nonzero(as_tuple=False).squeeze(1)
        n_missing = int(missing_feat_mask.sum().item())
        self.n_features_with_missing_ = n_missing
        self._missing_features_at_fit = missing_feat_indices

        if n_missing == 0:
            self.n_iter_ = 0
            self.imputation_sequence_ = []
            self.fit_status = True
            return self

        imputation_sequence: List[Tuple[int, torch.Tensor, Any]] = []
        n_iter_done = 0

        for iteration in range(self.max_iter):
            X_prev = X_filled.clone()

            feat_order = self._get_feature_order(
                miss_counts, n_missing, missing_feat_indices, self.random_state_
            )

            round_sequence: List[Tuple[int, torch.Tensor, Any]] = []

            for feat_idx_t in feat_order:
                feat_idx = feat_idx_t.item()
                neighbor_feats = self._select_neighbor_features(
                    feat_idx, X_filled, d, self.random_state_
                )
                neigh_feats_fitted, est = self._fit_one_feature(
                    feat_idx, X_filled, orig_mask, neighbor_feats
                )
                predictions = self._predict_one_feature(
                    feat_idx, X_filled, orig_mask, neigh_feats_fitted, est
                )
                X_filled[orig_mask[:, feat_idx], feat_idx] = predictions
                round_sequence.append((feat_idx, neigh_feats_fitted, est))

            imputation_sequence.extend(round_sequence)
            n_iter_done += 1

            # Convergence check (only when not sampling from posterior)
            if not self.sample_posterior:
                observed_vals = X_orig[~orig_mask]
                max_observed = observed_vals.abs().max().clamp(min=1e-10)
                change = (X_filled - X_prev).abs().max() / max_observed
                if self.verbose >= 1:
                    print(f"[IterativeImputer] iter {iteration + 1}: change = {change:.6f}")
                if change < self.tol:
                    if self.verbose >= 1:
                        print(f"[IterativeImputer] converged at iteration {iteration + 1}.")
                    break

        self.imputation_sequence_ = imputation_sequence
        self.n_iter_ = n_iter_done

        if self.add_indicator:
            self.indicator_ = MissingIndicator(
                missing_values=self.missing_values,
                device=self.device,
                dtype=self.dtype,
            )
            self.indicator_.fit(data_or_X)
        else:
            self.indicator_ = None

        self.fit_status = True
        return self

    def transform(self, data_or_X, **kwargs) -> torch.Tensor:
        """
        Impute missing values in *X* using the fitted models.

        Parameters
        ----------
        data_or_X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X_out : torch.Tensor of shape (n_samples, n_features[+ n_indicators])
        """
        if not self.fit_status:
            raise RuntimeError("IterativeImputer must be fitted before transform.")

        X_orig = _to_float_tensor(data_or_X, dtype=self.dtype, device=self.device)
        n, d = X_orig.shape
        orig_mask = self._missing_mask(X_orig)

        X_filled = self.initial_imputer_.transform(X_orig)

        # Features that had NO missing values at fit time
        fit_missing_set: set = (
            set(self._missing_features_at_fit.tolist())
            if self._missing_features_at_fit is not None
            else set()
        )

        # Apply each (feat_idx, neighbor_feats, estimator) in the sequence.
        # If multiple rounds were done during fit, use the LAST round's estimators
        # (the most recent, stored at the tail of imputation_sequence_).
        n_missing = self.n_features_with_missing_ or 0
        if n_missing > 0 and self.imputation_sequence_:
            # Last round: last n_missing_features entries
            last_round = self.imputation_sequence_[-n_missing:]
            for feat_idx, neighbor_feats, est in last_round:
                missing_rows = orig_mask[:, feat_idx]
                if not missing_rows.any():
                    continue
                if self.skip_complete and feat_idx not in fit_missing_set:
                    continue
                if est is None:
                    continue
                predictions = self._predict_one_feature(
                    feat_idx, X_filled, orig_mask, neighbor_feats, est
                )
                X_filled[missing_rows, feat_idx] = predictions

        if self.add_indicator and self.indicator_ is not None:
            indicator_out = self.indicator_.transform(data_or_X).to(
                dtype=self.dtype, device=self.device
            )
            X_filled = torch.cat([X_filled, indicator_out], dim=1)

        return X_filled

    def fit_transform(self, data_or_X, y=None, **kwargs) -> torch.Tensor:
        """
        Fit and transform in a single pass, returning the imputed training data.

        Parameters
        ----------
        data_or_X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        X_out : torch.Tensor of shape (n_samples, n_features[+ n_indicators])
        """
        X_orig = _to_float_tensor(data_or_X, dtype=self.dtype, device=self.device)
        n, d = X_orig.shape
        self.n_features_in_ = d
        orig_mask = self._missing_mask(X_orig)

        def _to_vec(val, d):
            if isinstance(val, (list, tuple, np.ndarray)):
                return torch.tensor(val, dtype=self.dtype, device=self.device)
            elif isinstance(val, torch.Tensor):
                return val.to(dtype=self.dtype, device=self.device)
            else:
                return torch.full((d,), float(val), dtype=self.dtype, device=self.device)

        self._min_value_ = _to_vec(self.min_value, d)
        self._max_value_ = _to_vec(self.max_value, d)
        self.random_state_ = self._make_generator()

        self.initial_imputer_ = SimpleImputer(
            missing_values=self.missing_values,
            fill_value=self.fill_value,
            keep_empty_features=self.keep_empty_features,
            device=self.device,
            dtype=self.dtype,
            strategy=self.initial_strategy,
        )
        X_filled = self.initial_imputer_.fit_transform(X_orig)

        miss_counts = orig_mask.sum(dim=0).float()
        missing_feat_mask = miss_counts > 0
        missing_feat_indices = missing_feat_mask.nonzero(as_tuple=False).squeeze(1)
        n_missing = int(missing_feat_mask.sum().item())
        self.n_features_with_missing_ = n_missing
        self._missing_features_at_fit = missing_feat_indices

        imputation_sequence: List[Tuple[int, torch.Tensor, Any]] = []
        n_iter_done = 0

        for iteration in range(self.max_iter):
            X_prev = X_filled.clone()
            feat_order = self._get_feature_order(
                miss_counts, n_missing, missing_feat_indices, self.random_state_
            )
            round_seq: List[Tuple[int, torch.Tensor, Any]] = []

            for feat_idx_t in feat_order:
                feat_idx = feat_idx_t.item()
                neighbor_feats = self._select_neighbor_features(
                    feat_idx, X_filled, d, self.random_state_
                )
                neigh_fitted, est = self._fit_one_feature(
                    feat_idx, X_filled, orig_mask, neighbor_feats
                )
                predictions = self._predict_one_feature(
                    feat_idx, X_filled, orig_mask, neigh_fitted, est
                )
                X_filled[orig_mask[:, feat_idx], feat_idx] = predictions
                round_seq.append((feat_idx, neigh_fitted, est))

            imputation_sequence.extend(round_seq)
            n_iter_done += 1

            if not self.sample_posterior and n_missing > 0:
                observed_vals = X_orig[~orig_mask]
                max_obs = observed_vals.abs().max().clamp(min=1e-10)
                change = (X_filled - X_prev).abs().max() / max_obs
                if self.verbose >= 1:
                    print(f"[IterativeImputer] iter {iteration + 1}: change = {change:.6f}")
                if change < self.tol:
                    if self.verbose >= 1:
                        print(f"[IterativeImputer] converged at iteration {iteration + 1}.")
                    break

        self.imputation_sequence_ = imputation_sequence
        self.n_iter_ = n_iter_done
        self.fit_status = True

        if self.add_indicator:
            self.indicator_ = MissingIndicator(
                missing_values=self.missing_values,
                device=self.device,
                dtype=self.dtype,
            )
            self.indicator_.fit(data_or_X)
            indicator_out = self.indicator_.transform(data_or_X).to(
                dtype=self.dtype, device=self.device
            )
            X_filled = torch.cat([X_filled, indicator_out], dim=1)
        else:
            self.indicator_ = None

        return X_filled

    def forward(
            self, X: torch.Tensor, y: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        if not self.fit_status:
            return self.fit_transform(X, y, **kwargs)
        return self.transform(X, **kwargs)

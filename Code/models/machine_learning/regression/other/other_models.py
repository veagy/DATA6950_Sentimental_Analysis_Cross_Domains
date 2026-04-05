import torch
from typing import Union

from .....models.utils import MLRegressor
from torch.func import vmap
import joblib

__all__ = ["IsotonicRegression", "DummyRegressor"]


class IsotonicRegression(MLRegressor):
    def __init__(self,
                 y_min: float = None,
                 y_max: float = None,
                 increasing: Union[str, bool] = True,
                 out_of_bounds: str = "nan",
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.y_min = y_min
        self.y_max = y_max
        self.increasing = increasing
        self._increasing = None
        self.out_of_bounds = out_of_bounds
        self.device = device
        self.dtype = dtype
        self._x_min = None
        self._x_max = None
        self._x_thresholds = None
        self._y_thresholds = None

    def _check_inputs(self, X: torch.Tensor, y: torch.Tensor = None):
        if X.ndim > 1 and X.shape[1] > 1:
            import warnings
            warnings.warn("Isotonic regression expects 1D input but got >1 feature. Only the first feature will be used.")
            X = X[:, 0]
        
        X = X.flatten()
            
        if y is not None:
            if y.ndim == 1:
                y = y.unsqueeze(1)
            
        if y is not None and X.shape[0] != y.shape[0]:
            raise ValueError(f"Found input variables with inconsistent numbers of samples: [{X.shape[0]}, {y.shape[0]}]")

        # Ensure correct device and dtype
        X = X.to(self.device, dtype=self.dtype)
        if y is not None:
            y = y.to(self.device, dtype=self.dtype)

        return X, y

    def _spearmanr(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """
        Calculate Spearman's rank correlation coefficient.
        """
        X_rank = torch.argsort(torch.argsort(X.flatten())).float()
        y_rank = torch.argsort(torch.argsort(y.flatten())).float()

        X_rank_centered = X_rank - X_rank.mean()
        y_rank_centered = y_rank - y_rank.mean()

        cov = (X_rank_centered * y_rank_centered).sum()
        std_x = torch.sqrt((X_rank_centered ** 2).sum())
        std_y = torch.sqrt((y_rank_centered ** 2).sum())

        if std_x == 0 or std_y == 0:
            return 0.0

        return cov / (std_x * std_y)

    def _pava(self, y: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        """
        Pool Adjacent Violators Algorithm (PAVA) for isotonic regression.
        Implementation for 1D data.
        """
        y_hat = y.clone().flatten()
        n = len(y_hat)

        if n == 0:
            return y_hat

        if weights is None:
            weights = torch.ones_like(y_hat)
        else:
            weights = weights.flatten()

        block_values = []
        block_weights = []
        block_counts = []

        idx = 0
        while idx < n:
            val = y_hat[idx]
            weight = weights[idx]
            count = 1

            block_values.append(val)
            block_weights.append(weight)
            block_counts.append(count)

            while len(block_values) > 1 and block_values[-1] < block_values[-2]:
                v2, w2, c2 = block_values.pop(), block_weights.pop(), block_counts.pop()
                v1, w1, c1 = block_values.pop(), block_weights.pop(), block_counts.pop()

                new_w = w1 + w2
                new_v = (v1 * w1 + v2 * w2) / new_w
                new_c = c1 + c2

                block_values.append(new_v)
                block_weights.append(new_w)
                block_counts.append(new_c)

            idx += 1

        result = torch.zeros_like(y_hat)
        curr_idx = 0
        for val, count in zip(block_values, block_counts):
            result[curr_idx: curr_idx + count] = val
            curr_idx += count

        return result

    def fit(self, data_or_X, y: torch.Tensor = None, **kwargs):
        """
        Fit the isotonic regression model.
        """
        X, y = self._check_inputs(data_or_X, y)

        if y is None:
            raise ValueError("y cannot be None for fitting.")

        sample_weight = kwargs.get("sample_weight")
        if sample_weight is not None:
            sample_weight = sample_weight.to(self.device, dtype=self.dtype)
            if sample_weight.ndim == 0:
                sample_weight = sample_weight.expand_as(y.flatten())
            elif sample_weight.ndim > 1:
                sample_weight = sample_weight.flatten()

        indices = torch.argsort(X.flatten())
        X_sorted = X.flatten()[indices]
        y_sorted = y[indices]

        if sample_weight is not None:
            weights_sorted = sample_weight[indices]
        else:
            weights_sorted = torch.ones_like(X_sorted)

        unique_X, inverse_indices = torch.unique_consecutive(X_sorted, return_inverse=True)
        n_unique = len(unique_X)
        inverse_indices = inverse_indices.clamp(0, max(0, n_unique - 1))

        if n_unique < len(X_sorted):
            new_weights = torch.zeros_like(unique_X)
            new_weights.index_add_(0, inverse_indices.long(), weights_sorted)

            new_y_weighted = torch.zeros((n_unique, y.shape[1]), device=self.device, dtype=self.dtype)
            for j in range(y.shape[1]):
                new_y_weighted[:, j].index_add_(0, inverse_indices.long(), y_sorted[:, j] * weights_sorted)

            X_sorted = unique_X
            y_sorted = new_y_weighted / (new_weights.unsqueeze(1) + 1e-10)
            weights_sorted = new_weights

        self._increasing = []
        y_isotonic_list = []
        for j in range(y_sorted.shape[1]):
            yj = y_sorted[:, j]
            inc = self.increasing
            if inc == 'auto':
                corr = self._spearmanr(X_sorted, yj)
                inc = corr > 0
            self._increasing.append(inc)

            target_yj = yj if inc else -yj
            iso_j = self._pava(target_yj, weights_sorted)
            if not inc:
                iso_j = -iso_j
            y_isotonic_list.append(iso_j)
            
        y_isotonic = torch.stack(y_isotonic_list, dim=1)

        if self.y_min is not None:
            y_isotonic = torch.clamp(y_isotonic, min=self.y_min)
        if self.y_max is not None:
            y_isotonic = torch.clamp(y_isotonic, max=self.y_max)

        self._x_thresholds = X_sorted
        self._y_thresholds = y_isotonic
        self._x_min = X_sorted[0]
        self._x_max = X_sorted[-1]

        return self

    def predict(self, X: torch.Tensor):
        """
        Predict labels for data X.
        """
        X, _ = self._check_inputs(X)
        y_pred = self._predict(X)
        return y_pred.view(-1) if y_pred.shape[1] == 1 else y_pred

    def transform(self, X: torch.Tensor):
        return self.predict(X)

    def _predict(self, X: torch.Tensor):
        if self._x_thresholds is None:
            raise RuntimeError("Model is not fitted")

        X_flat = X.flatten()
        mask_lower = X_flat < self._x_min
        mask_upper = X_flat > self._x_max
        mask_oob = mask_lower | mask_upper

        if self.out_of_bounds == 'raise':
            if mask_oob.any():
                raise ValueError("X contains values out of bounds.")

        X_clamped = torch.clamp(X_flat, min=self._x_min, max=self._x_max)

        n_t = len(self._x_thresholds)
        if n_t < 2:
            return self._y_thresholds[0].expand(len(X_flat), -1)
            
        indices = torch.bucketize(X_clamped, self._x_thresholds, right=False)
        indices = torch.clamp(indices, min=1, max=n_t - 1)

        idx_right = indices.clamp(max=n_t - 1)
        idx_left = (indices - 1).clamp(min=0)

        x_left = self._x_thresholds[idx_left].unsqueeze(1)
        x_right = self._x_thresholds[idx_right].unsqueeze(1)
        y_left = self._y_thresholds[idx_left]
        y_right = self._y_thresholds[idx_right]

        denominator = x_right - x_left
        numerator = X_clamped.unsqueeze(1) - x_left

        slope = (y_right - y_left) / torch.where(denominator == 0, torch.ones_like(denominator), denominator)
        y_pred = y_left + slope * numerator

        if self.out_of_bounds == 'nan':
            y_pred[mask_oob] = float('nan')

        return y_pred

    def __calc_x_bounds(self, X: torch.Tensor):
        self._x_min = X.min()
        self._x_max = X.max()
        return self

    @property
    def X_min_(self):
        return self._x_min

    @property
    def X_max_(self):
        return self._x_max

    @property
    def x_thresholds_(self):
        return self._x_thresholds

    @property
    def y_thresholds_(self):
        return self._y_thresholds

    @property
    def f_(self):
        return self._predict

    @property
    def increasing_(self):
        return self._increasing

    def forward(self, X: torch.Tensor, y: torch.Tensor = None):
        """
        Forward pass. If y is provided, fits the model. Otherwise predicts.
        """
        self.__calc_x_bounds(X)
        if self._x_thresholds is None:
            self.fit(X, y)
        return self._predict(X)


class DummyRegressor(MLRegressor):
    def __init__(self,
                 strategy: str = "mean",
                 constant: Union[int, float] = None,
                 quantile: float = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.const = None
        self.strategy = strategy
        self.constant = constant
        self.quantile = quantile
        self.device = device
        self.dtype = dtype
        self.in_features = None
        self.out_features = None

    @property
    def constant_(self):
        return self.const

    @property
    def n_features_in_(self):
        return self.in_features

    @property
    def n_outputs_in_(self):
        return self.out_features

    def _init_module(self, X: torch.Tensor, y: torch.Tensor):
        self.in_features = X.size(-1)
        self.out_features = y.size(-1)
        return self

    def fit(self, data_or_X, y=None, **kwargs):
        """
        Fit the dummy regressor.
        """
        X, y = self._check_inputs(data_or_X, y)
        self._init_module(X, y)

        if y is None:
            raise ValueError("y cannot be None for fitting.")

        if self.strategy == "mean":
            self.const = torch.mean(y, dim=0, keepdim=True)
        elif self.strategy == "median":
            # torch.median returns (values, indices) or just values depending on version/args
            # For multi-dim y, we want median per output column.
            # torch.median(input, dim, keepdim=True) returns (values, indices)
            self.const = torch.median(y, dim=0, keepdim=True).values
        elif self.strategy == "quantile":
            if self.quantile is None:
                raise ValueError("Quantile must be provided for 'quantile' strategy.")
            if not (0.0 <= self.quantile <= 1.0):
                raise ValueError("Quantile must be in [0.0, 1.0].")
            self.const = torch.quantile(y, self.quantile, dim=0, keepdim=True)
        elif self.strategy == "constant":
            if self.constant is None:
                raise ValueError("Constant must be provided for 'constant' strategy.")

            # Ensure constant is a tensor of correct shape
            const = torch.tensor(self.constant, device=self.device, dtype=self.dtype)

            if const.ndim == 0:
                const = const.view(1, 1)  # scalar -> (1, 1)
            elif const.ndim == 1:
                # If y has n_outputs, constant should match
                if const.shape[0] != self.out_features:
                    # If constant is scalar-like but provided as 1D array of len 1?
                    if const.shape[0] == 1:
                        const = const.view(1, 1).expand(1, self.out_features)
                    else:
                        # Or if y is 1D?
                        if self.out_features == 1 and const.shape[0] == 1:
                            const = const.view(1, 1)
                        else:
                            raise ValueError(
                                f"Constant shape {const.shape} mismatch with y shape (n_outputs={self.out_features})")
                else:
                    # Matches output features, reshape to (1, n_outputs)
                    const = const.view(1, -1)

            # Now const should be (1, n_outputs) or (1, 1) expandable?
            if const.ndim == 2:
                if const.shape[1] != self.out_features:
                    # Only if it was (1, 1) and out_features > 1, we might expand?
                    if const.shape[1] == 1:
                        const = const.expand(1, self.out_features)
                    else:
                        raise ValueError(f"Constant shape {const.shape} mismatch with n_outputs={self.out_features}")
            else:
                # Should not happen given above logic
                const = const.view(1, -1)

            self.const = const
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return self

    def _predict(self, X: torch.Tensor):
        """
        Perform classification on test vectors X.
        """
        if self.const is None:
            raise RuntimeError("Model is not fitted")

        # X is just used for shape
        # check inputs
        if X.ndim == 1:
            X = X.view(-1, 1)
        X = X.to(self.device, dtype=self.dtype)

        n_samples = X.shape[0]
        # const is (1, n_outputs)
        # result (n_samples, n_outputs)

        y_pred = self.const.expand(n_samples, -1)

        # If output was 1D originally, we might want to flatten?
        # But MLModule usually keeps 2D (N, 1).
        # other_models.py: IsotonicRegression checks input and reshapes 1D input to 2D.
        # So output should generally be 2D if inputs are 2D.

        return y_pred

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict using the fitted constant (mean/median/quantile/constant)."""
        return self._predict(X)

    def forward(self, X: torch.Tensor, y: torch.Tensor = None):
        X, y = self._check_inputs(X, y)
        if self.in_features is None:
            self._init_module(X, y)
            self.fit(X, y)
        return self._predict(X)

    def _check_inputs(self, X: torch.Tensor, y: torch.Tensor = None):
        if X.ndim == 1:
            X = X.view(-1, 1)
        if y is not None and y.ndim == 1:
            y = y.view(-1, 1)

        # Ensure correct device and dtype
        X = X.to(self.device, dtype=self.dtype)
        if y is not None:
            y = y.to(self.device, dtype=self.dtype)

        return X, y

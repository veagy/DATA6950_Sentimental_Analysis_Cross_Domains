import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional, Union, Tuple, List, Literal

class MLModule(torch.nn.Module):
    """Base class to retain sklearn-like fit/transform state."""
    def __init__(self):
        super().__init__()
        self.fit_status = False

    def fit_transform(self, X: torch.Tensor, y: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """Sequential fit and transform executing as a unified step."""
        return self.fit(X, y, **kwargs).transform(X, **kwargs)

    def forward(self, X: torch.Tensor, y: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """Standard PyTorch module forward executing fit_transform."""
        return self.fit_transform(X, y, **kwargs)

class Binarizer(MLModule):
    def __init__(self,
                 threshold: float = 0.0,
                 copy: bool = True,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.threshold = threshold
        self.copy = copy
        self.device = device
        self.dtype = dtype
        self.n_features_in_ = None

    def fit(self, data_or_X, y=None, **kwargs):
        X = torch.as_tensor(data_or_X, dtype=self.dtype, device=self.device)
        self.n_features_in_ = X.size(-1)
        self.fit_status = True
        return self

    def transform(self, X: torch.Tensor, **kwargs):
        X = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        out = torch.where(X >= self.threshold, torch.tensor(1.0, device=X.device, dtype=X.dtype), torch.tensor(0.0, device=X.device, dtype=X.dtype))
        if self.copy:
            return out
        X.copy_(out)
        return X

    def fit_transform(self, X: torch.Tensor, y: torch.Tensor = None, **kwargs):
        return self.fit(X, y, **kwargs).transform(X, **kwargs)

    def forward(self, X: torch.Tensor, y: torch.Tensor = None, **kwargs) -> torch.Tensor:
        return self.fit_transform(X, y, **kwargs)

class MaxAbsScaler(MLModule):
    def __init__(self,
                 copy: bool = True,
                 clip: bool = False,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.copy = copy
        self.clip = clip
        self.device = device
        self.dtype = dtype
        self.scale_ = None
        self.max_abs_ = None
        self.n_features_in_ = None
        self.n_samples_seen_ = None

    def fit(self, data_or_X, y=None, **kwargs):
        X = torch.as_tensor(data_or_X, dtype=self.dtype, device=self.device)
        copy_X = X.clone() if self.copy else X
        self.n_features_in_ = copy_X.size(-1)
        self.max_abs_ = torch.max(torch.abs(copy_X), dim=0).values
        scale = self.max_abs_.clone()
        scale[scale == 0.0] = 1.0
        self.scale_ = scale
        self.n_samples_seen_ = copy_X.size(-2)
        self.fit_status = True
        return self

    def transform(self, X: torch.Tensor, **kwargs):
        copy_X = X.clone() if self.copy else X
        copy_X = X / self.scale_
        if self.clip:
            copy_X = copy_X.clamp(min=-1.0, max=1.0)
        return copy_X

    def inverse_transform(self, X: torch.Tensor, **kwargs):
        copy_X = X.clone() if self.copy else X
        return copy_X * self.scale_

    def fit_transform(self, X: torch.Tensor, y: torch.Tensor = None, **kwargs):
        return self.fit(X, y, **kwargs).transform(X, **kwargs)

    def forward(self, X: torch.Tensor, y: torch.Tensor = None, **kwargs) -> torch.Tensor:
        return self.fit_transform(X, y, **kwargs)

class MinMaxScaler(MLModule):
    def __init__(self,
                 feature_range: Union[Tuple[float, float], List[float], torch.Tensor] = (0, 1),
                 copy: bool = True,
                 clip: bool = False,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        feature_range = torch.as_tensor(feature_range, device=device, dtype=dtype).flatten().tolist()
        if len(feature_range) >= 2:
            self.feature_range = feature_range[:2]
        else:
            self.feature_range = [0.0, 1.0]
        self.copy = copy
        self.clip = clip
        self.device = device
        self.dtype = dtype
        self.min_ = None
        self.scale_ = None
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None
        self.n_features_in_ = None
        self.n_samples_seen_ = None

    def fit(self, data_or_X, y=None, **kwargs):
        X = torch.as_tensor(data_or_X, dtype=self.dtype, device=self.device)
        copy_X = X.clone() if self.copy else X
        self.n_features_in_ = copy_X.shape[1]
        self.n_samples_seen_ = copy_X.shape[0]
        self.data_min_, _ = copy_X.min(dim=0, keepdim=True)
        self.data_max_, _ = copy_X.max(dim=0, keepdim=True)
        self.data_range_ = self.data_max_ - self.data_min_
        safe_range = torch.where(
            self.data_range_ > 0,
            self.data_range_,
            torch.ones_like(self.data_range_)
        )
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / safe_range
        self.min_ = self.feature_range[0] - self.data_min_ * self.scale_
        self.fit_status = True
        return self

    def transform(self, X: torch.Tensor, **kwargs):
        copy_X = X.clone() if self.copy else X
        copy_X = copy_X * self.scale_ + self.min_
        if self.clip:
            copy_X = copy_X.clamp(min=self.feature_range[0], max=self.feature_range[1])
        return copy_X

    def inverse_transform(self, X: torch.Tensor, **kwargs):
        copy_X = X.clone() if self.copy else X
        copy_X = (copy_X - self.min_) / self.scale_
        return copy_X

    def fit_transform(self, X: torch.Tensor, y: torch.Tensor = None, **kwargs):
        return self.fit(X, y, **kwargs).transform(X, **kwargs)

    def forward(self, X: torch.Tensor, y: torch.Tensor = None, **kwargs) -> torch.Tensor:
        return self.fit_transform(X, y, **kwargs)

class Normalizer(MLModule):
    def __init__(self,
                 norm: Union[Literal["l1", "l2", "max"], float]="l2",
                 copy: bool = True,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        if isinstance(norm, str):
            if norm == "l1": self.p = 1
            elif norm == "l2": self.p = 2
            elif norm == "max": self.p = float('inf')
            else: self.p = 2
        elif isinstance(norm, float):
            self.p = norm
        self.copy = copy
        self.device = device
        self.dtype = dtype
        self.n_features_in_ = None

    def fit(self, data_or_X, y=None, **kwargs):
        X = torch.as_tensor(data_or_X, dtype=self.dtype, device=self.device)
        self.n_features_in_ = X.shape[1]
        self.fit_status = True
        return self

    def transform(self, X: torch.Tensor, **kwargs):
        X = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        copy_X = X.clone() if self.copy else X
        norm_X = copy_X.norm(p=self.p, dim=-1, keepdim=True)
        norm_X = torch.where(norm_X > 0, norm_X, torch.ones_like(norm_X))
        return copy_X / norm_X

    def fit_transform(self, X: torch.Tensor, y: torch.Tensor = None, **kwargs):
        return self.fit(X, y, **kwargs).transform(X, **kwargs)

    def forward(self, X: torch.Tensor, y: torch.Tensor = None, **kwargs) -> torch.Tensor:
        return self.fit_transform(X, y, **kwargs)

class StandardScaler(MLModule):
    def __init__(self,
                 copy: bool = True,
                 with_mean: bool = True,
                 with_std: bool = True,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self.device = device
        self.dtype = dtype
        self.scale_ = None
        self.mean_ = None
        self.var_ = None
        self.n_features_in_ = None
        self.n_samples_seen_ = None

    def fit(self, data_or_X, y=None, **kwargs):
        X = torch.as_tensor(data_or_X, dtype=self.dtype, device=self.device)
        copy_X = X.clone() if self.copy else X
        self.n_features_in_ = copy_X.shape[1]
        self.n_samples_seen_ = copy_X.shape[0]
        if self.with_mean:
            self.mean_ = copy_X.mean(dim=0, keepdim=True)
        if self.with_std:
            self.var_ = copy_X.var(dim=0, unbiased=False, keepdim=True)
            self.scale_ = torch.sqrt(self.var_.clone())
            self.scale_[self.scale_ == 0] = 1.0
        self.fit_status = True
        return self

    def transform(self, X: torch.Tensor, **kwargs):
        X = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        copy_X = X.clone() if self.copy else X
        if self.with_mean:
            copy_X = copy_X - self.mean_
        if self.with_std:
            copy_X = copy_X / self.scale_
        return copy_X

    def inverse_transform(self, X: torch.Tensor, **kwargs):
        X = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        copy_X = X.clone() if self.copy else X
        if self.with_std:
            copy_X = copy_X * self.scale_
        if self.with_mean:
            copy_X += self.mean_
        return copy_X

    def fit_transform(self, X: torch.Tensor, y: torch.Tensor = None, **kwargs):
        return self.fit(X, y, **kwargs).transform(X, **kwargs)

    def forward(self, X: torch.Tensor, y: torch.Tensor = None, **kwargs) -> torch.Tensor:
        return self.fit_transform(X, y, **kwargs)

class RobustScaler(MLModule):
    def __init__(self,
                 with_centering: bool = True,
                 with_scaling: bool = True,
                 quantile_range: Tuple[float, float] = (25.0, 75.0),
                 unit_variance: bool = False,
                 copy: bool = True,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.unit_variance = unit_variance
        self.copy = copy
        self.device = device
        self.dtype = dtype
        self.center_ = None
        self.scale_ = None
        self.n_features_in_ = None

    def fit(self, data_or_X, y=None, **kwargs):
        X = torch.as_tensor(data_or_X, dtype=self.dtype, device=self.device)
        self.n_features_in_ = X.shape[1]
        q_min, q_max = self.quantile_range

        q = torch.tensor([q_min / 100.0, 0.5, q_max / 100.0], device=X.device)
        vals = torch.quantile(X, q, dim=0)

        if self.with_centering:
            self.center_ = vals[1]
        else:
            self.center_ = torch.zeros(self.n_features_in_, device=X.device)

        if self.with_scaling:
            q_range = vals[2] - vals[0]
            q_range[q_range == 0] = 1.0

            if self.unit_variance:
                adjust = torch.tensor(1.3489795003921634, device=X.device)
                q_range = q_range / adjust

            self.scale_ = q_range
        else:
            self.scale_ = torch.ones(self.n_features_in_, device=X.device)

        self.fit_status = True
        return self

    def transform(self, data_or_X, **kwargs):
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Scaler must be fitted before transforming.")
        X = torch.as_tensor(data_or_X, dtype=self.dtype, device=self.device)
        return (X - self.center_) / self.scale_

    def fit_transform(self, X: torch.Tensor, y: torch.Tensor = None, **kwargs):
        return self.fit(X, **kwargs).transform(X, **kwargs)

    def inverse_transform(self, data_or_X, **kwargs):
        X = torch.as_tensor(data_or_X, dtype=self.dtype, device=self.device)
        return (X * self.scale_) + self.center_

    def forward(self, X: torch.Tensor, y: torch.Tensor = None, **kwargs) -> torch.Tensor:
        return self.fit_transform(X, y, **kwargs)

class PowerTransformer(MLModule):
    def __init__(self,
                 method: str = 'yeo-johnson',
                 standardize: bool = True,
                 copy: bool = True,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.method = method.lower()
        self.standardize = standardize
        self.copy = copy
        self.device = device
        self.dtype = dtype

        if self.method != 'yeo-johnson':
            raise NotImplementedError("Only 'yeo-johnson' is implemented in this Torch version.")

        self.register_buffer('lambdas_', None)
        self.register_buffer('mean_', None)
        self.register_buffer('std_', None)

    def _yeo_johnson_transform(self, x, lmbda):
        out = torch.zeros_like(x)
        pos = x >= 0
        neg = ~pos

        if torch.any(pos):
            if torch.abs(lmbda) < 1e-6:
                out[pos] = torch.log1p(x[pos])
            else:
                out[pos] = (torch.pow(x[pos] + 1, lmbda) - 1) / lmbda

        if torch.any(neg):
            if torch.abs(lmbda - 2) < 1e-6:
                out[neg] = -torch.log1p(-x[neg])
            else:
                out[neg] = -(torch.pow(-x[neg] + 1, 2 - lmbda) - 1) / (2 - lmbda)
        return out

    def _log_likelihood(self, lmbda, x):
        n_samples = x.shape[0]
        x_trans = self._yeo_johnson_transform(x, lmbda)
        var = torch.var(x_trans, unbiased=False)
        if (var == 0).item():
            return torch.tensor(float("inf"), device=x.device)

        log_like = -n_samples / 2 * torch.log(var)
        log_like += (lmbda - 1) * torch.sum(torch.sign(x) * torch.log1p(torch.abs(x)))
        return -log_like 

    def fit(self, data_or_X, y=None, **kwargs):
        X = torch.as_tensor(data_or_X, dtype=self.dtype, device=self.device)
        n_features = X.shape[1]
        self.n_features_in_ = n_features
        self.lambdas_ = torch.zeros(n_features, device=X.device)
        iterations = kwargs.get("iterations", 100)

        for i in range(n_features):
            feat = X[:, i]
            lmbda = torch.tensor(1.0, requires_grad=True, device=X.device)
            optimizer = optim.LBFGS([lmbda], lr=0.1, max_iter=iterations)

            def closure():
                optimizer.zero_grad()
                loss = self._log_likelihood(lmbda, feat)
                loss.backward()
                return loss

            optimizer.step(closure)
            self.lambdas_[i] = lmbda.detach()

        if self.standardize:
            X_trans = self.transform(X, _skip_std=True)
            self.mean_ = torch.mean(X_trans, dim=0)
            self.std_ = torch.std(X_trans, dim=0)
            self.std_[self.std_ == 0] = 1.0

        self.fit_status = True
        return self

    def transform(self, data_or_X, _skip_std=False):
        X = torch.as_tensor(data_or_X, dtype=self.dtype, device=self.device)
        X_trans = torch.zeros_like(X)
        for i in range(X.shape[1]):
            X_trans[:, i] = self._yeo_johnson_transform(X[:, i], self.lambdas_[i])

        if self.standardize and not _skip_std:
            X_trans = (X_trans - self.mean_) / self.std_

        return X_trans

    def inverse_transform(self, data_or_X, **kwargs):
        X = torch.as_tensor(data_or_X, dtype=self.dtype, device=self.device)
        X_orig = torch.zeros_like(X)
        for i in range(X.shape[1]):
            X_orig[:, i] = self._yeo_johnson_inverse(X[:, i], self.lambdas_[i])
        if self.standardize:
            X_orig = X_orig * self.std_ + self.mean_
        return X_orig

    def _yeo_johnson_inverse(self, x: torch.Tensor, lmbda: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        pos = x >= 0
        neg = ~pos
        eps = 1e-6
        if torch.any(pos):
            if torch.abs(lmbda) < eps:
                out[pos] = torch.expm1(x[pos])
            else:
                out[pos] = torch.pow(x[pos] * lmbda + 1, 1.0 / lmbda) - 1
        if torch.any(neg):
            if torch.abs(lmbda - 2) < eps:
                out[neg] = 1 - torch.exp(-x[neg])
            else:
                out[neg] = 1 - torch.pow(-(2 - lmbda) * x[neg] + 1, 1.0 / (2 - lmbda))
        return out

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class QuantileTransformer(MLModule):
    def __init__(self,
                 n_quantiles: int = 1000,
                 output_distribution: str = 'uniform',
                 subsample: int = 10_000,
                 random_state: Optional[Union[int, torch.Generator]] = None,
                 copy: bool = True,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution.lower()
        self.subsample = subsample
        self.random_state = random_state
        self.copy = copy
        self.device = device
        self.dtype = dtype
        self.n_features_in_ = None
        self.register_buffer('quantiles_', None)
        self.register_buffer('references_', None)

    def _get_quantiles(self, X):
        n_samples, n_features = X.shape
        n_quantiles = min(self.n_quantiles, n_samples)
        references = torch.linspace(0, 1, n_quantiles, device=X.device)
        quantiles = torch.quantile(X, references, dim=0)
        return quantiles, references

    def fit(self, data_or_X, y=None, **kwargs):
        X = torch.as_tensor(data_or_X, dtype=self.dtype, device=self.device)
        self.n_features_in_ = X.shape[1]
        
        if self.subsample is not None and X.shape[0] > self.subsample:
            gen = self.random_state
            if isinstance(gen, int):
                gen = torch.Generator(device=X.device).manual_seed(gen)
            indices = torch.randperm(X.shape[0], generator=gen, device=X.device)[:self.subsample]
            X_subset = X[indices]
        else:
            X_subset = X

        self.quantiles_, self.references_ = self._get_quantiles(X_subset)
        self.fit_status = True
        return self

    def transform(self, data_or_X, **kwargs):
        if self.quantiles_ is None:
            raise RuntimeError("Transformer must be fitted before transforming.")
        X = torch.as_tensor(data_or_X, dtype=self.dtype, device=self.device)
        n_features = X.shape[1]
        X_trans = torch.zeros_like(X)

        for i in range(n_features):
            feature_column = X[:, i]
            interp_unit = self._interpolate(feature_column, self.quantiles_[:, i], self.references_)
            
            if self.output_distribution == 'normal':
                eps = 1e-7
                interp_unit = torch.clamp(interp_unit, eps, 1.0 - eps)
                X_trans[:, i] = torch.erfinv(2.0 * interp_unit - 1.0) * torch.sqrt(
                    torch.tensor(2.0, device=X.device, dtype=X.dtype)
                )
            else:
                X_trans[:, i] = interp_unit

        return X_trans

    def inverse_transform(self, data_or_X, **kwargs):
        if self.quantiles_ is None:
            raise RuntimeError("Transformer must be fitted before inverse transforming.")
        X = torch.as_tensor(data_or_X, dtype=self.dtype, device=self.device)
        n_features = X.shape[1]
        X_orig = torch.zeros_like(X)

        for i in range(n_features):
            feature_column = X[:, i]
            if self.output_distribution == "normal":
                uniform = 0.5 * (1.0 + torch.erf(feature_column / torch.sqrt(torch.tensor(2.0, device=X.device, dtype=X.dtype))))
            else:
                uniform = feature_column
                
            X_orig[:, i] = self._interpolate(uniform, self.references_.flatten(), self.quantiles_[:, i].flatten())
        return X_orig

    def _interpolate(self, x, x_points, y_points):
        x = x.flatten()
        x_points = x_points.flatten()
        y_points = y_points.flatten()

        right_idx = torch.searchsorted(x_points, x, side="right")
        left_idx = (right_idx - 1).clamp(min=0)
        right_idx = right_idx.clamp(max=x_points.shape[0] - 1)

        x_left = x_points[left_idx]
        x_right = x_points[right_idx]
        y_left = y_points[left_idx]
        y_right = y_points[right_idx]

        diff = x_right - x_left
        diff = torch.where(diff == 0, torch.ones_like(diff, dtype=diff.dtype), diff)
        weight = (x - x_left) / diff
        return (y_left + weight * (y_right - y_left)).to(x.dtype)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

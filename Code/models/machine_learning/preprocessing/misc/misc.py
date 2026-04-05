import warnings
import torch
import torch.nn as nn
from itertools import combinations, combinations_with_replacement
from typing import Optional, Callable, Union, Any, List, Tuple, Dict, Literal
from .....models.utils import MLModule
import pandas as pd  # pyright: ignore[reportMissingImports]
import numpy as np
from torch.func import vmap
import joblib

__all__ = [
    "PolynomialFeatures",
    "KernelCenterer",
    "SplineTransformers",
    "SplineTransformer",
    "FunctionTransformer",
]


# ---------------------------------------------------------------------------
# Helpers shared across classes
# ---------------------------------------------------------------------------

def _to_tensor(X: Any, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Convert array-like / DataFrame / Tensor input to a 2-D float Tensor."""
    if isinstance(X, torch.Tensor):
        return X.to(dtype=dtype, device=device)
    if isinstance(X, pd.DataFrame):
        return torch.tensor(X.values, dtype=dtype, device=device)
    if isinstance(X, np.ndarray):
        return torch.tensor(X, dtype=dtype, device=device)
    return torch.tensor(np.array(X), dtype=dtype, device=device)




def _build_powers(n_features: int, min_degree: int, max_degree: int,
                  interaction_only: bool, include_bias: bool) -> List[List[int]]:
    """Return list of exponent rows for polynomial feature generation."""
    powers: List[List[int]] = []
    start_d = max(1, min_degree)

    if include_bias and min_degree == 0:
        powers.append([0] * n_features)

    for d in range(start_d, max_degree + 1):
        it = (combinations(range(n_features), d)
              if interaction_only
              else combinations_with_replacement(range(n_features), d))
        for combo in it:
            row = [0] * n_features
            for idx in combo:
                row[idx] += 1
            powers.append(row)

    return powers


# ---------------------------------------------------------------------------
# B-spline helpers for SplineTransformers
# ---------------------------------------------------------------------------

def _build_knots(col: torch.Tensor, n_knots: int,
                 strategy: str) -> torch.Tensor:
    """Compute n_knots boundary+internal knots for a single feature column."""
    if strategy == "uniform":
        return torch.linspace(float(col.min()), float(col.max()),
                              n_knots, dtype=col.dtype, device=col.device)
    elif strategy == "quantile":
        q = torch.linspace(0.0, 1.0, n_knots, device=col.device)
        return torch.quantile(col.float(), q).to(col.dtype)
    else:
        raise ValueError(f"Unknown knot strategy: {strategy!r}")


def _augment_knots(knots: torch.Tensor, degree: int) -> torch.Tensor:
    """Build augmented knot vector by repeating boundary knots `degree` times."""
    return torch.cat([
        knots[:1].expand(degree),
        knots,
        knots[-1:].expand(degree),
    ])


def _bspline_basis(x: torch.Tensor, aug_knots: torch.Tensor,
                   degree: int) -> torch.Tensor:
    device, dtype = x.device, x.dtype
    n = x.shape[0]
    m = len(aug_knots) - 1  # number of intervals in augmented sequence

    # --- Degree-0 basis (indicator functions) ---
    # B_{i,0}(x) = 1 if aug_knots[i] <= x < aug_knots[i+1], else 0
    k_lo = aug_knots[:-1].unsqueeze(0)   # (1, m)
    k_hi = aug_knots[1:].unsqueeze(0)    # (1, m)
    xc = x.unsqueeze(1)                  # (n, 1)

    B = ((xc >= k_lo) & (xc < k_hi)).to(dtype)  # (n, m)

    # Handle points equal to the last knot (right-closed last interval)
    last_mask = (x == aug_knots[-1])
    if last_mask.any():
        B[last_mask, -1] = 1.0

    # --- Recursion ---
    for d in range(1, degree + 1):
        n_b = B.shape[1] - 1  # new number of basis functions
        i_idx = torch.arange(n_b, device=device)

        t_i   = aug_knots[i_idx]          # (n_b,)
        t_id  = aug_knots[i_idx + d]      # (n_b,)
        t_id1 = aug_knots[i_idx + d + 1]  # (n_b,)
        t_i1  = aug_knots[i_idx + 1]      # (n_b,)

        denom_l = (t_id - t_i).unsqueeze(0)    # (1, n_b)
        denom_r = (t_id1 - t_i1).unsqueeze(0)  # (1, n_b)

        safe_denom_l = denom_l.abs().clamp(min=1e-10)
        safe_denom_r = denom_r.abs().clamp(min=1e-10)

        alpha_l = torch.where(
            denom_l.abs() > 1e-10,
            (xc - t_i.unsqueeze(0)) / safe_denom_l,
            torch.zeros(1, 1, device=device, dtype=dtype),
        )  # (n, n_b)

        alpha_r = torch.where(
            denom_r.abs() > 1e-10,
            (t_id1.unsqueeze(0) - xc) / safe_denom_r,
            torch.zeros(1, 1, device=device, dtype=dtype),
        )  # (n, n_b)

        B = B[:, :-1] * alpha_l + B[:, 1:] * alpha_r  # (n, n_b)

    return B  # (n, n_basis)


def _bspline_basis_periodic(x: torch.Tensor, knots: torch.Tensor,
                             degree: int) -> torch.Tensor:
    period = knots[-1] - knots[0]
    # Wrap x into [knots[0], knots[-1])
    x_wrap = knots[0] + (x - knots[0]) % period

    # Build extended augmented knot vector for periodic splines
    aug = torch.cat([
        knots[-(degree + 1):-1] - period,
        knots,
        knots[1:degree + 1] + period,
    ])

    B_all = _bspline_basis(x_wrap, aug, degree)  # (n, many)
    n_splines = len(knots) - 1
    # The periodic basis has n_splines functions; fold the wrap-around columns
    # back (only the first n_splines columns are independent)
    return B_all[:, :n_splines]


class PolynomialFeatures(MLModule):
    def __init__(self,
                 degree: Union[int, tuple] = 2,
                 interaction_only: bool = False,
                 include_bias: bool = True,
                 order: Literal["C", "F"] = "C",
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args,
                 **kwargs,
                 ):
        super().__init__()
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.order = order
        self.device = (torch.device(device)
                       if isinstance(device, str) else device)
        self.dtype = dtype
        self.args = args
        self.kwargs = kwargs

        # Fitted attributes
        self.powers_: Optional[torch.Tensor] = None
        self.n_features_in_: Optional[int] = None
        self.n_output_features_: Optional[int] = None
        self.fit_status = False

    # ------------------------------------------------------------------
    def _parse_degree(self) -> Tuple[int, int]:
        if isinstance(self.degree, (tuple, list)):
            min_d, max_d = int(self.degree[0]), int(self.degree[1])
        else:
            min_d, max_d = 0, int(self.degree)
        if min_d < 0 or max_d < 0 or min_d > max_d:
            raise ValueError(
                f"degree must satisfy 0 <= min_degree <= max_degree, got {self.degree}"
            )
        return min_d, max_d

    # ------------------------------------------------------------------
    def fit(self, X: Any, y: Any = None, **kwargs) -> "PolynomialFeatures":
        X_t = _to_tensor(X, self.dtype, self.device)
        if X_t.ndim == 1:
            X_t = X_t.unsqueeze(1)

        n_features = X_t.shape[1]
        min_d, max_d = self._parse_degree()

        power_list = _build_powers(
            n_features, min_d, max_d,
            self.interaction_only, self.include_bias,
        )

        self.n_features_in_ = n_features
        self.powers_ = torch.tensor(
            power_list, dtype=torch.long, device=self.device
        )  # (n_out, n_features)
        self.n_output_features_ = self.powers_.shape[0]
        self.fit_status = True
        return self

    # ------------------------------------------------------------------
    def transform(self, X: Any, **kwargs) -> torch.Tensor:
        if not self.fit_status:
            raise RuntimeError("Call fit() before transform().")
        X_t = _to_tensor(X, self.dtype, self.device)
        if X_t.ndim == 1:
            X_t = X_t.unsqueeze(1)

        n_samples, n_features = X_t.shape
        if n_features != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {n_features}."
            )

        n_out = self.n_output_features_
        # Vectorised: (n_samples, n_out, n_features) ** powers
        X_exp = X_t.unsqueeze(1).expand(n_samples, n_out, n_features)
        p_exp = self.powers_.unsqueeze(0).expand(n_samples, n_out, n_features)
        out = (X_exp ** p_exp.to(self.dtype)).prod(dim=2)  # (n_samples, n_out)

        # Memory order: 'F' transposes and re-transposes (same tensor, just
        # signals downstream code; PyTorch doesn't have Fortran-order natively)
        if self.order == "F":
            out = out.t().contiguous().t()

        return out

    # ------------------------------------------------------------------
    def fit_transform(self, X: Any, y: Any = None, **kwargs) -> torch.Tensor:
        return self.fit(X, y, **kwargs).transform(X, **kwargs)

    # ------------------------------------------------------------------
    def get_feature_names_out(
        self, input_features: Optional[List[str]] = None
    ) -> List[str]:
        if not self.fit_status:
            raise RuntimeError("Call fit() before get_feature_names_out().")
        n = self.n_features_in_
        if input_features is None:
            input_features = [f"x{i}" for i in range(n)]

        names: List[str] = []
        for power_row in self.powers_.tolist():
            parts = []
            for feat_name, exp in zip(input_features, power_row):
                if exp == 0:
                    continue
                elif exp == 1:
                    parts.append(feat_name)
                else:
                    parts.append(f"{feat_name}^{exp}")
            names.append(" ".join(parts) if parts else "1")
        return names

    # ------------------------------------------------------------------
    def forward(self, X: Any, **kwargs) -> torch.Tensor:
        if not self.fit_status:
            self.fit(X, **kwargs)
        return self.transform(X, **kwargs)

    # ------------------------------------------------------------------
    def predict(self, X: Any, **kwargs) -> torch.Tensor:
        return self.transform(X, **kwargs)

    # ------------------------------------------------------------------
    def score(self, X: Any, y: Any = None, **kwargs) -> float:
        return 0.0


# ---------------------------------------------------------------------------
# KernelCenterer
# ---------------------------------------------------------------------------

class KernelCenterer(MLModule):
    def __init__(self,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args,
                 **kwargs,
                 ):
        super().__init__()
        self.device = (torch.device(device)
                       if isinstance(device, str) else device)
        self.dtype = dtype
        self.args = args
        self.kwargs = kwargs

        # Fitted attributes
        self.K_fit_rows_: Optional[torch.Tensor] = None  # (n_train,)
        self.K_fit_all_: Optional[torch.Tensor] = None   # scalar
        self.n_features_in_: Optional[int] = None
        self.fit_status = False

    # ------------------------------------------------------------------
    def fit(self, K: Any, y: Any = None, **kwargs) -> "KernelCenterer":
        K_t = _to_tensor(K, self.dtype, self.device)
        if K_t.ndim != 2 or K_t.shape[0] != K_t.shape[1]:
            raise ValueError(
                "Kernel matrix must be a square 2-D matrix, "
                f"got shape {tuple(K_t.shape)}."
            )
        n = K_t.shape[0]
        self.K_fit_rows_ = K_t.mean(dim=0)          # column means  (n_train,)
        self.K_fit_all_ = self.K_fit_rows_.mean()    # overall mean  scalar
        self.n_features_in_ = n
        self.fit_status = True
        return self

    # ------------------------------------------------------------------
    def transform(self, K: Any, copy: bool = True, **kwargs) -> torch.Tensor:
        if not self.fit_status:
            raise RuntimeError("Call fit() before transform().")
        K_t = _to_tensor(K, self.dtype, self.device)
        if copy:
            K_t = K_t.clone()

        # K_t shape: (n_test, n_train)
        # Row means of test kernel matrix
        K_pred_rows = K_t.mean(dim=1, keepdim=True)  # (n_test, 1)

        # Center: K̃ = K - col_means - row_means + overall_mean
        K_t = K_t - self.K_fit_rows_.unsqueeze(0)   # subtract column means
        K_t = K_t - K_pred_rows                     # subtract row means
        K_t = K_t + self.K_fit_all_                 # add overall mean
        return K_t

    # ------------------------------------------------------------------
    def fit_transform(self, K: Any, y: Any = None,
                      copy: bool = True, **kwargs) -> torch.Tensor:
        return self.fit(K, y, **kwargs).transform(K, copy=copy, **kwargs)

    # ------------------------------------------------------------------
    def forward(self, K: Any, **kwargs) -> torch.Tensor:
        if not self.fit_status:
            self.fit(K, **kwargs)
        return self.transform(K, **kwargs)

    # ------------------------------------------------------------------
    def predict(self, K: Any, **kwargs) -> torch.Tensor:
        return self.transform(K, **kwargs)

    # ------------------------------------------------------------------
    def score(self, K: Any, y: Any = None, **kwargs) -> float:
        return 0.0


# ---------------------------------------------------------------------------
# SplineTransformers
# ---------------------------------------------------------------------------

class SplineTransformers(MLModule):
    def __init__(self,
                 n_knots: int = 5,
                 degree: int = 3,
                 knots: Union[Literal["uniform", "quantile"],
                    list, tuple, torch.Tensor] = 'uniform',
                 extrapolation: Literal["error", "constant",
                 "linear", "continue", "periodic"] = 'constant',
                 include_bias: bool = True,
                 order: Literal["C", "F"] = "C",
                 handle_missing: Literal["error", "zeros"] = "error",
                 sparse_output: bool = False,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args,
                 **kwargs,
                 ):
        super().__init__()
        self.n_knots = n_knots
        self.degree = degree
        self.knots = knots
        self.extrapolation = extrapolation
        self.include_bias = include_bias
        self.order = order
        self.handle_missing = handle_missing
        self.sparse_output = sparse_output
        self.device = (torch.device(device)
                       if isinstance(device, str) else device)
        self.dtype = dtype
        self.args = args
        self.kwargs = kwargs

        # Fitted attributes
        self.bsplines_: List[Dict] = []   # one dict per feature with knot info
        self.n_features_in_: Optional[int] = None
        self.n_features_out_: Optional[int] = None
        self.fit_status = False

    # ------------------------------------------------------------------
    def _get_knots_for_feature(self, col: torch.Tensor) -> torch.Tensor:
        """Return the knot vector (not augmented) for a single feature."""
        if isinstance(self.knots, str):
            return _build_knots(col, self.n_knots, self.knots)
        else:
            # array-like: shape (n_knots, n_features) or (n_knots,)
            knot_arr = torch.tensor(
                np.array(self.knots), dtype=self.dtype, device=self.device
            )
            if knot_arr.ndim == 2:
                # caller passes column index externally; handled in fit loop
                raise RuntimeError(
                    "Array-like knots with multiple features must be passed "
                    "as (n_knots, n_features); handled per-feature in fit."
                )
            return knot_arr.to(dtype=self.dtype, device=self.device)

    def _n_splines(self, n_knots_feat: int) -> int:
        if self.extrapolation == "periodic":
            return n_knots_feat - 1
        return n_knots_feat + self.degree - 1

    # ------------------------------------------------------------------
    def fit(self, X: Any, y: Any = None, **kwargs) -> "SplineTransformers":
        X_t = _to_tensor(X, self.dtype, self.device)
        if X_t.ndim == 1:
            X_t = X_t.unsqueeze(1)

        n_samples, n_features = X_t.shape

        # NaN handling during fit
        nan_mask = torch.isnan(X_t)
        if nan_mask.any():
            if self.handle_missing == "error":
                raise ValueError(
                    "Input contains NaN values. Set handle_missing='zeros' "
                    "to suppress this error."
                )
            # Replace NaNs with column means for computing knot positions
            col_means = X_t.nanmean(dim=0)
            X_t = torch.where(nan_mask, col_means.unsqueeze(0).expand_as(X_t), X_t)

        # Validate n_knots
        if isinstance(self.knots, str) and self.n_knots < 2:
            raise ValueError(f"n_knots must be >= 2, got {self.n_knots}.")

        self.bsplines_ = []
        total_out = 0

        # Preprocess array-like knots
        array_knots: Optional[torch.Tensor] = None
        if not isinstance(self.knots, str):
            array_knots = torch.tensor(
                np.array(self.knots), dtype=self.dtype, device=self.device
            )
            if array_knots.ndim == 1:
                # Same knots for every feature
                array_knots = array_knots.unsqueeze(1).expand(-1, n_features)

        for j in range(n_features):
            col = X_t[:, j]
            if array_knots is not None:
                k_vec = array_knots[:, j] if array_knots.ndim == 2 else array_knots
            else:
                k_vec = _build_knots(col, self.n_knots, self.knots)  # type: ignore[arg-type]

            n_k = k_vec.shape[0]
            n_sp = self._n_splines(n_k)
            n_out_j = n_sp if self.include_bias else n_sp - 1
            total_out += n_out_j

            aug_knots = (None if self.extrapolation == "periodic"
                         else _augment_knots(k_vec, self.degree))
            self.bsplines_.append({
                "knots": k_vec,
                "aug_knots": aug_knots,
                "n_splines": n_sp,
                "x_min": float(k_vec[0]),
                "x_max": float(k_vec[-1]),
            })

        self.n_features_in_ = n_features
        self.n_features_out_ = total_out
        self.fit_status = True
        return self

    # ------------------------------------------------------------------
    def _transform_feature(self, col: torch.Tensor, bs: Dict,
                           nan_mask: torch.Tensor) -> torch.Tensor:
        """Evaluate B-splines for one feature column."""
        x_min, x_max = bs["x_min"], bs["x_max"]
        extrap = self.extrapolation
        n = col.shape[0]
        n_sp = bs["n_splines"]
        dev, dt = self.device, self.dtype

        if extrap == "error":
            out_of_range = (col < x_min) | (col > x_max)
            if out_of_range.any():
                raise ValueError(
                    "Values outside the training range and "
                    "extrapolation='error'."
                )

        if extrap == "periodic":
            B = _bspline_basis_periodic(col, bs["knots"], self.degree)
        else:
            aug_knots = bs["aug_knots"]

            if extrap in ("constant", "error"):
                col_c = col.clamp(min=x_min, max=x_max)
            elif extrap == "linear":
                col_c = col  # keep as-is; linear adjustment applied after
            else:  # 'continue'
                col_c = col

            B = _bspline_basis(col_c, aug_knots, self.degree)  # (n, n_sp)

            if extrap == "linear":
                # Linearly extrapolate by extending boundary derivatives
                eps = (x_max - x_min) * 1e-4

                B_lo = _bspline_basis(
                    torch.full((1,), x_min + eps, device=dev, dtype=dt),
                    aug_knots, self.degree,
                )
                B_bound_lo = _bspline_basis(
                    torch.full((1,), x_min, device=dev, dtype=dt),
                    aug_knots, self.degree,
                )
                deriv_lo = (B_lo - B_bound_lo) / eps  # (1, n_sp)

                B_hi = _bspline_basis(
                    torch.full((1,), x_max - eps, device=dev, dtype=dt),
                    aug_knots, self.degree,
                )
                B_bound_hi = _bspline_basis(
                    torch.full((1,), x_max, device=dev, dtype=dt),
                    aug_knots, self.degree,
                )
                deriv_hi = (B_bound_hi - B_hi) / eps  # (1, n_sp)

                lo_mask = col < x_min  # (n,)
                hi_mask = col > x_max  # (n,)

                if lo_mask.any():
                    delta = (col[lo_mask] - x_min).unsqueeze(1)  # (k, 1)
                    B_extrap = B_bound_lo + delta * deriv_lo      # (k, n_sp)
                    B[lo_mask] = B_extrap

                if hi_mask.any():
                    delta = (col[hi_mask] - x_max).unsqueeze(1)  # (k, 1)
                    B_extrap = B_bound_hi + delta * deriv_hi      # (k, n_sp)
                    B[hi_mask] = B_extrap

        # Zero out NaN positions if requested
        if self.handle_missing == "zeros" and nan_mask.any():
            B[nan_mask] = 0.0

        # Drop last column if include_bias=False
        if not self.include_bias:
            B = B[:, :-1]

        return B  # (n, n_sp or n_sp-1)

    # ------------------------------------------------------------------
    def transform(self, X: Any, **kwargs) -> torch.Tensor:
        if not self.fit_status:
            raise RuntimeError("Call fit() before transform().")

        X_t = _to_tensor(X, self.dtype, self.device)
        if X_t.ndim == 1:
            X_t = X_t.unsqueeze(1)

        n_samples, n_features = X_t.shape
        if n_features != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {n_features}."
            )

        nan_mask_all = torch.isnan(X_t)

        parts: List[torch.Tensor] = []
        for j, bs in enumerate(self.bsplines_):
            col = X_t[:, j]
            nan_col = nan_mask_all[:, j]
            if self.handle_missing == "zeros" and nan_col.any():
                col_safe = col.clone()
                col_safe[nan_col] = float(bs["x_min"])
            else:
                col_safe = col
            B_j = self._transform_feature(col_safe, bs, nan_col)
            parts.append(B_j)

        out = torch.cat(parts, dim=1)  # (n_samples, n_features_out_)

        if self.order == "F":
            out = out.t().contiguous().t()

        if self.sparse_output:
            return out.to_sparse()
        return out

    # ------------------------------------------------------------------
    def fit_transform(self, X: Any, y: Any = None, **kwargs) -> torch.Tensor:
        return self.fit(X, y, **kwargs).transform(X, **kwargs)

    # ------------------------------------------------------------------
    def get_feature_names_out(
        self, input_features: Optional[List[str]] = None
    ) -> List[str]:
        if not self.fit_status:
            raise RuntimeError("Call fit() before get_feature_names_out().")
        if input_features is None:
            input_features = [f"x{j}" for j in range(self.n_features_in_)]
        names: List[str] = []
        for j, bs in enumerate(self.bsplines_):
            n_sp = bs["n_splines"]
            n_out_j = n_sp if self.include_bias else n_sp - 1
            feat = input_features[j]
            for sp in range(n_out_j):
                names.append(f"{feat}_sp{sp}")
        return names

    # ------------------------------------------------------------------
    def forward(self, X: Any, **kwargs) -> torch.Tensor:
        if not self.fit_status:
            self.fit(X, **kwargs)
        return self.transform(X, **kwargs)

    # ------------------------------------------------------------------
    def predict(self, X: Any, **kwargs) -> torch.Tensor:
        return self.transform(X, **kwargs)

    # ------------------------------------------------------------------
    def score(self, X: Any, y: Any = None, **kwargs) -> float:
        return 0.0


class FunctionTransformer(MLModule):
    def __init__(self,
                 func: Union[str, Callable, nn.Module] = None,
                 inverse_func: Union[str, Callable, nn.Module] = None,
                 validate: bool = False,
                 accept_sparse: bool = False,
                 check_inverse: bool = False,
                 feature_names_out: Union[Literal["one-to-one"],
                 Callable, nn.Module] = None,
                 kw_args: dict = None,
                 inv_kw_args: dict = None,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args,
                 **kwargs,
                 ):
        super().__init__()
        self.func = func
        self.inverse_func = inverse_func
        self.validate = validate
        self.accept_sparse = accept_sparse
        self.check_inverse = check_inverse
        self.feature_names_out = feature_names_out
        self.kw_args = kw_args or {}
        self.inv_kw_args = inv_kw_args or {}
        self.device = (torch.device(device)
                       if isinstance(device, str) else device)
        self.dtype = dtype
        self.args = args
        self.kwargs = kwargs

        # Fitted attributes
        self.n_features_in_: Optional[int] = None
        self.fit_status = False

        # Resolve func/inverse_func via MLModule._resolve_funcs.
        # None stays None (treated as identity at call time).
        self._func_resolved = self._resolve_funcs(func) if func is not None else None
        self._inv_func_resolved = (self._resolve_funcs(inverse_func)
                                   if inverse_func is not None else None)

    # ------------------------------------------------------------------
    def _validate_input(self, X: Any) -> torch.Tensor:
        """Optionally convert and validate the input."""
        if not self.validate:
            if isinstance(X, torch.Tensor):
                return X.to(device=self.device, dtype=self.dtype)
            return _to_tensor(X, self.dtype, self.device)

        # validate=True: convert and check 2-D
        X_t = _to_tensor(X, self.dtype, self.device)
        if X_t.is_sparse and not self.accept_sparse:
            raise ValueError(
                "Sparse input detected but accept_sparse=False."
            )
        if X_t.ndim != 2:
            raise ValueError(
                f"validate=True requires a 2-D input, got ndim={X_t.ndim}."
            )
        return X_t

    # ------------------------------------------------------------------
    def _apply(self, X: torch.Tensor, fn, extra_kw: dict) -> torch.Tensor:
        if fn is None:
            return X  # identity
        result = fn(X, **extra_kw)
        if not isinstance(result, torch.Tensor):
            result = torch.as_tensor(result, device=self.device, dtype=self.dtype)
        return result

    # ------------------------------------------------------------------
    def fit(self, X: Any, y: Any = None, **kwargs) -> "FunctionTransformer":
        X_t = self._validate_input(X)
        if X_t.ndim >= 2:
            self.n_features_in_ = X_t.shape[1]
        else:
            self.n_features_in_ = 1

        if self.check_inverse and (self._func_resolved is not None
                                   and self._inv_func_resolved is not None):
            X_out = self._apply(X_t, self._func_resolved, self.kw_args)
            X_rec = self._apply(X_out, self._inv_func_resolved, self.inv_kw_args)
            if not torch.allclose(X_t.float(), X_rec.float(), atol=1e-5):
                warnings.warn(
                    "The provided functions are not inverse of each other. "
                    "Set check_inverse=False to suppress this warning.",
                    UserWarning,
                    stacklevel=2,
                )

        self.fit_status = True
        return self

    # ------------------------------------------------------------------
    def transform(self, X: Any, **kwargs) -> torch.Tensor:
        X_t = self._validate_input(X)
        return self._apply(X_t, self._func_resolved, self.kw_args)

    # ------------------------------------------------------------------
    def inverse_transform(self, X: Any, **kwargs) -> torch.Tensor:
        X_t = self._validate_input(X)
        return self._apply(X_t, self._inv_func_resolved, self.inv_kw_args)

    # ------------------------------------------------------------------
    def fit_transform(self, X: Any, y: Any = None, **kwargs) -> torch.Tensor:
        return self.fit(X, y, **kwargs).transform(X, **kwargs)

    # ------------------------------------------------------------------
    def get_feature_names_out(
        self, input_features: Optional[List[str]] = None
    ) -> List[str]:
        fno = self.feature_names_out
        if fno is None:
            raise AttributeError(
                "get_feature_names_out is only defined when feature_names_out "
                "is not None."
            )
        n = self.n_features_in_ or 0
        if input_features is None:
            input_features = [f"x{i}" for i in range(n)]

        if fno == "one-to-one":
            return list(input_features)

        if callable(fno):
            return list(fno(self, input_features))

        raise ValueError(
            f"feature_names_out must be 'one-to-one', callable, or None; "
            f"got {fno!r}."
        )

    # ------------------------------------------------------------------
    def forward(self, X: Any, **kwargs) -> torch.Tensor:
        if not self.fit_status:
            self.fit(X, **kwargs)
        return self.transform(X, **kwargs)

    # ------------------------------------------------------------------
    def predict(self, X: Any, **kwargs) -> torch.Tensor:
        return self.transform(X, **kwargs)

    # ------------------------------------------------------------------
    def score(self, X: Any, y: Any = None, **kwargs) -> float:
        return 0.0


# sklearn-compatible singular alias
SplineTransformer = SplineTransformers

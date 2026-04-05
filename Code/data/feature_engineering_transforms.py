import torch
import math
from typing import Optional, List, Callable, Dict, Any, Union
from collections.abc import Iterable
from ..data.scaling import MLModule # reusing base MLModule created in scaling.py

class PolynomialFeatures(MLModule):
    def __init__(self,
                 degree: Union[int, tuple] = 2,
                 interaction_only: bool = False,
                 include_bias: bool = True,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.device = device
        self.dtype = dtype
        self.n_features_in_ = None
        self.powers_ = None
        self.n_output_features_ = None

    def fit(self, data_or_X, y=None, **kwargs):
        X = torch.as_tensor(data_or_X, dtype=self.dtype, device=self.device)
        self.n_features_in_ = X.shape[1]
        
        min_degree = 0 if self.include_bias else 1
        max_degree = self.degree if isinstance(self.degree, int) else max(self.degree)

        combinations = []
        for d in range(min_degree, max_degree + 1):
            if d == 0:
                combinations.append(torch.zeros(self.n_features_in_, dtype=torch.int))
            else:
                self._generate_combinations(self.n_features_in_, d, self.interaction_only, [], combinations)
                
        self.powers_ = torch.stack(combinations).to(self.device)
        self.n_output_features_ = self.powers_.shape[0]
        self.fit_status = True
        return self

    def _generate_combinations(self, n_features, degree, interaction_only, current_combo, all_combos):
        if degree == 0:
            combo_tensor = torch.zeros(n_features, dtype=torch.int)
            for feat in current_combo:
                combo_tensor[feat] += 1
            all_combos.append(combo_tensor)
            return
            
        start_idx = current_combo[-1] if current_combo else 0
        if interaction_only and current_combo:
            start_idx += 1
            
        for i in range(start_idx, n_features):
            self._generate_combinations(n_features, degree - 1, interaction_only, current_combo + [i], all_combos)

    def transform(self, data_or_X, **kwargs):
        X = torch.as_tensor(data_or_X, dtype=self.dtype, device=self.device)
        if self.powers_ is None:
            raise RuntimeError("Transformer must be fitted first.")
            
        n_samples = X.shape[0]
        out = torch.ones((n_samples, self.n_output_features_), dtype=self.dtype, device=self.device)
        
        for i in range(self.n_output_features_):
            power_counts = self.powers_[i]
            feat_vals = torch.ones(n_samples, dtype=self.dtype, device=self.device)
            # Find indices where power > 0 to calculate terms
            active_dims = torch.nonzero(power_counts, as_tuple=True)[0]
            for dim in active_dims:
                feat_vals *= torch.pow(X[:, dim], power_counts[dim].float())
            out[:, i] = feat_vals
            
        return out


class FunctionTransformer(MLModule):
    def __init__(self,
                 func: Optional[Callable] = None,
                 inverse_func: Optional[Callable] = None,
                 validate: bool = False,
                 accept_sparse: bool = False,
                 check_inverse: bool = True,
                 kw_args: Optional[Dict[str, Any]] = None,
                 inv_kw_args: Optional[Dict[str, Any]] = None,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.func = func
        self.inverse_func = inverse_func
        self.validate = validate
        self.accept_sparse = accept_sparse
        self.check_inverse = check_inverse
        self.kw_args = kw_args if kw_args is not None else {}
        self.inv_kw_args = inv_kw_args if inv_kw_args is not None else {}
        self.device = device
        self.dtype = dtype

    def fit(self, data_or_X, y=None, **kwargs):
        X = torch.as_tensor(data_or_X, dtype=self.dtype, device=self.device)
        
        if self.validate:
            self.n_features_in_ = X.shape[1]
            if self.check_inverse and self.inverse_func is not None and self.func is not None:
                trans_X = self.transform(X)
                inv_X = self.inverse_transform(trans_X)
                if not torch.allclose(X, inv_X, atol=1e-5):
                    raise ValueError("Inverse function is not the actual inverse of the function.")
                    
        self.fit_status = True
        return self

    def transform(self, data_or_X, **kwargs):
        X = torch.as_tensor(data_or_X, dtype=self.dtype, device=self.device)
        if self.func is None:
            return X
        return self.func(X, **self.kw_args)

    def inverse_transform(self, data_or_X, **kwargs):
        X = torch.as_tensor(data_or_X, dtype=self.dtype, device=self.device)
        if self.inverse_func is None:
            return X
        return self.inverse_func(X, **self.inv_kw_args)


class SplineTransformer(MLModule):
    def __init__(self,
                 n_knots: int = 5,
                 degree: int = 3,
                 knots: Union[str, torch.Tensor] = 'uniform',
                 extrapolation: str = 'constant',
                 include_bias: bool = True,
                 order: str = 'C',
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.n_knots = n_knots
        self.degree = degree
        self.knots = knots
        self.extrapolation = extrapolation
        self.include_bias = include_bias
        self.order = order
        self.device = device
        self.dtype = dtype
        
        self.bsplines_ = None
        self.n_features_in_ = None
        self.n_features_out_ = None
        
    def fit(self, data_or_X, y=None, **kwargs):
        X = torch.as_tensor(data_or_X, dtype=self.dtype, device=self.device)
        self.n_features_in_ = X.shape[1]
        
        if isinstance(self.knots, str) and self.knots == 'uniform':
            x_min, _ = torch.min(X, dim=0)
            x_max, _ = torch.max(X, dim=0)
            
            knots_list = []
            for i in range(self.n_features_in_):
                k = torch.linspace(x_min[i].item(), x_max[i].item(), self.n_knots, device=self.device)
                
                # Expand knots by degree on both ends to create B-spline basis
                step = k[1] - k[0] if self.n_knots > 1 else torch.tensor(1.0, device=self.device)
                left_padding = k[0] - step * torch.arange(self.degree, 0, -1, device=self.device)
                right_padding = k[-1] + step * torch.arange(1, self.degree + 1, device=self.device)
                
                full_knots = torch.cat([left_padding, k, right_padding])
                knots_list.append(full_knots)
                
            self.bsplines_ = torch.stack(knots_list)
        else:
            self.bsplines_ = torch.as_tensor(self.knots, dtype=self.dtype, device=self.device)
            
        self.n_features_out_ = self.n_features_in_ * (self.n_knots + self.degree - 1)
        if not self.include_bias:
            self.n_features_out_ -= self.n_features_in_
            
        self.fit_status = True
        return self

    def _basis_recursive(self, x, knots, degree, i):
        if degree == 0:
            return ((x >= knots[i]) & (x < knots[i + 1])).float()
            
        # Add tiny epsilon to prevent div/0
        denom1 = knots[i + degree] - knots[i]
        denom1 = torch.where(denom1 == 0, torch.tensor(1e-8, device=x.device), denom1)
        
        denom2 = knots[i + degree + 1] - knots[i + 1]
        denom2 = torch.where(denom2 == 0, torch.tensor(1e-8, device=x.device), denom2)
        
        term1 = ((x - knots[i]) / denom1) * self._basis_recursive(x, knots, degree - 1, i)
        term2 = ((knots[i + degree + 1] - x) / denom2) * self._basis_recursive(x, knots, degree - 1, i + 1)
        
        return term1 + term2

    def transform(self, data_or_X, **kwargs):
        X = torch.as_tensor(data_or_X, dtype=self.dtype, device=self.device)
        n_samples = X.shape[0]
        
        out_features = []
        for i in range(self.n_features_in_):
            feat = X[:, i]
            knots = self.bsplines_[i]
            
            # Clip extrapolation depending on setting
            if self.extrapolation == 'constant':
                feat = torch.clamp(feat, min=knots[self.degree], max=knots[-(self.degree+1)])
            
            splines_for_feat = []
            start_spline = 1 if not self.include_bias else 0
            n_splines = self.n_knots + self.degree - 1
            
            for j in range(start_spline, n_splines):
                basis = self._basis_recursive(feat, knots, self.degree, j)
                splines_for_feat.append(basis)
                
            out_features.extend(splines_for_feat)
            
        return torch.stack(out_features, dim=1)


class KernelCenterer(MLModule):
    def __init__(self,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.K_fit_rows_ = None
        self.K_fit_all_ = None
        self.n_features_in_ = None

    def fit(self, data_or_K, y=None, **kwargs):
        K = torch.as_tensor(data_or_K, dtype=self.dtype, device=self.device)
        self.n_features_in_ = K.shape[1]
        
        # Mean over rows
        self.K_fit_rows_ = torch.mean(K, dim=0, keepdim=True)
        # Mean over entire matrix
        self.K_fit_all_ = torch.mean(K)
        
        self.fit_status = True
        return self

    def transform(self, data_or_K, **kwargs):
        if self.K_fit_rows_ is None:
            raise RuntimeError("KernelCenterer must be fitted first.")
            
        K = torch.as_tensor(data_or_K, dtype=self.dtype, device=self.device)
        K_pred_rows = torch.mean(K, dim=1, keepdim=True)
        
        # Double centering formula
        K_c = K - self.K_fit_rows_ - K_pred_rows + self.K_fit_all_
        return K_c

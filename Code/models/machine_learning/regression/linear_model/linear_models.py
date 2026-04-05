
try:
    import scipy.optimize as opt
except ImportError:
    opt = None
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Union, Tuple, List, Callable, Iterable
from .....models.utils import MLModule, MLRegressor
import warnings
from torch.func import vmap
import joblib


__all__ = [
    "LinearRegression",
    "Ridge",
    "RidgeCV",
    "Lasso",
    "LassoCV",
    "ElasticNet",
    "ElasticNetCV",
    "HuberRegressor",
    "OrthogonalMatchingPursuit",
    "OrthogonalMatchingPursuitCV",
    "TheilSenRegressor",
    "RANSACRegressor",
    "QuantileRegressor",
]



class LinearRegression(MLRegressor):
    def __init__(self,
                 fit_intercept: bool = True,
                 copy_X: bool = True,
                 tol: float = 1e-6,
                 n_jobs: int = None,
                 positive: bool = False,
                 warm_start: bool = False,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.register_parameter('weight', None)
        self.register_parameter('bias', None)
        self.fit_intercept = fit_intercept
        self.warm_start = warm_start
        self.copy_X = copy_X
        self.tol = tol
        self.n_jobs = n_jobs
        self.positive = positive
        self.device = device
        self.dtype = dtype

    def _init_module(self, X, y):
        in_features = X.size(-1)
        out_features = y.size(-1)
        device = X.device
        dtype = X.dtype
        self.weight = nn.Parameter(
            torch.randn((out_features, in_features), device=device, dtype=dtype) * 0.1
        )
        if self.fit_intercept:
            self.bias = nn.Parameter(
                torch.randn((out_features,), device=device, dtype=dtype) * 0.1
            )
        else:
            self.bias = None
        # Update internal device/dtype tracking
        self.device = str(device)
        self.dtype = dtype
        return self

    @staticmethod
    def rank_(X):
        return torch.linalg.matrix_rank(X)

    @property
    def coef_(self):
        return self.weight

    @property
    def intercept_(self):
        if self.fit_intercept:
            return self.bias
        else:
            return None

    def _nnls(self, X: torch.Tensor, y: torch.Tensor, max_iter: int = 500, tol: float = 1e-6):
        """
        Solves min ||y - Xw||^2 s.t. w >= 0 using Accelerated Projected Gradient Descent.
        Vectorized for multi-target y.
        """
        # Precompute XtX and Xty
        XtX = X.T @ X
        Xty = X.T @ y

        n_features = XtX.shape[0]
        n_targets = y.shape[1]

        # Step size based on Lipschitz constant (max eigenvalue of XtX)
        with torch.no_grad():
            try:
                L = torch.linalg.norm(XtX, ord=2)
            except RuntimeError:
                # Fallback to trace if norm fails
                L = torch.trace(XtX)

            eta = 1.0 / (L + 1e-9)

            w = torch.zeros((n_features, n_targets), device=X.device, dtype=X.dtype)
            y_acc = w.clone()  # For Nesterov acceleration
            t = 1.0

            for i in range(max_iter):
                # Gradient: X^T(Xw - y) = XtX @ w - Xty
                grad = XtX @ y_acc - Xty

                # Projected Gradient Step
                w_new = torch.clamp(y_acc - eta * grad, min=0)

                # Convergence check
                if torch.max(torch.abs(w_new - w)) < tol:
                    w = w_new
                    break

                # Acceleration update
                t_new = (1.0 + math.sqrt(1.0 + 4.0 * t ** 2)) / 2.0
                y_acc = w_new + ((t - 1.0) / t_new) * (w_new - w)

                t = t_new
                w = w_new

        return w

    @staticmethod
    def singular_(X):
        return torch.linalg.svdvals(X)

    @property
    def n_features_in_(self):
        return self.weight.shape[1] if self.weight is not None else 0

    def forward(self, X: torch.Tensor, y: Optional[torch.Tensor] = None):
        """
        Forward pass. Handled via predict() which is called after robust conversion.
        """
        # We don't call super().forward(X) directly because it would call self.predict(X)
        # and self.predict(X) wouldn't have the conversion if called directly.
        # Actually, self(X) -> forward(x) -> predict(x)
        
        # Robust conversion (same as base class)
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X)
        try:
            device = self.weight.device if self.weight is not None else self.device
            dtype = self.weight.dtype if self.weight is not None else self.dtype
        except Exception:
            device = 'cpu'
            dtype = torch.float32
        X = X.to(device=device, dtype=dtype)

        if self.weight is None:
            if y is None:
                raise ValueError(
                    "Model weights are not initialized. Please call fit() first or provide y in forward() for auto-initialization.")
            self._init_module(X, y)
            # Re-init might have changed params, so ensure fit uses current device
            return self.fit(X, y).predict(X)
        return self.predict(X)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X)
        try:
            device = self.weight.device if self.weight is not None else getattr(self, 'device', 'cpu')
            dtype = self.weight.dtype if self.weight is not None else getattr(self, 'dtype', torch.float32)
        except Exception:
            device, dtype = 'cpu', torch.float32
        X = X.to(device=device, dtype=dtype)
        weight = torch.abs(self.weight) if self.positive else self.weight
        return F.linear(X, weight, self.bias)

    def fit(self, data_or_X, y=None, **kwargs):
        # Determine target device and dtype from input data
        target_device = data_or_X.device if isinstance(data_or_X, torch.Tensor) else self.device
        target_dtype = data_or_X.dtype if isinstance(data_or_X, torch.Tensor) else self.dtype

        # Determine target device and dtype from input data
        
        

        self._fit_loss_fn = kwargs.get('loss_fn', None)
        try:
            warm_start = kwargs.get('warm_start', getattr(self, 'warm_start', False))
            sample_weight = kwargs.get("sample_weight", None)
            # 1. Handle copy_X
            X_working = torch.as_tensor(data_or_X, device=target_device, dtype=target_dtype)
            if self.copy_X:
                X_working = X_working.clone()

            # 2. Manage Parallelism (n_jobs)
            prev_threads = torch.get_num_threads()
            if self.n_jobs is not None:
                num_threads = self.n_jobs if self.n_jobs > 0 else torch.get_num_threads()
                torch.set_num_threads(num_threads)

            try:
                # Use device from input if not already initialized
                target_device = X_working.device
                target_dtype = X_working.dtype

                y = torch.as_tensor(y, device=target_device, dtype=target_dtype)

                if y.ndim == 1:
                    y = y.unsqueeze(1)

                if sample_weight is not None:
                    sample_weight = torch.as_tensor(sample_weight, device=target_device, dtype=target_dtype)
                    if sample_weight.ndim == 1:
                        sample_weight = sample_weight.unsqueeze(1)
                    sw_sqrt = torch.sqrt(sample_weight)
                else:
                    sw_sqrt = None

                # Initialize module if weights are None, not warm_start, or shape mismatch
                need_init = self.weight is None or not warm_start
                if not need_init and (self.weight.shape[1] != X_working.shape[1] or self.weight.shape[0] != y.shape[1]):
                    need_init = True
                if need_init:
                    self._init_module(X_working, y)

                if not self.positive:
                    # --- Standard Path: Closed-form Solution ---
                    if self.fit_intercept:
                        ones = torch.ones((X_working.shape[0], 1), device=target_device, dtype=target_dtype)
                        X_hat = torch.cat([X_working, ones], dim=1)
                    else:
                        X_hat = X_working

                    if sw_sqrt is not None:
                        # Apply weights: sqrt(w) * X and sqrt(w) * y
                        X_hat = X_hat * sw_sqrt
                        y_weighted = y * sw_sqrt
                    else:
                        y_weighted = y

                    solution = torch.linalg.lstsq(X_hat, y_weighted, rcond=self.tol).solution
                    if solution.dim() == 1:
                        solution = solution.unsqueeze(1)

                    with torch.no_grad():
                        if self.fit_intercept:
                            self.weight.copy_(solution[:-1, :].T)
                            self.bias.copy_(solution[-1, :])
                        else:
                            self.weight.copy_(solution.T)

                else:
                    # --- Positive Path: Iterative Optimization ---
                    optimizer = torch.optim.LBFGS([self.weight] + ([self.bias] if self.fit_intercept else []),
                                                  lr=1, max_iter=100, tolerance_grad=self.tol)

                    def closure():
                        optimizer.zero_grad()
                        pred = F.linear(X_working, self.weight, self.bias)
                        custom_loss_fn = getattr(self, '_fit_loss_fn', None)
                        if custom_loss_fn is not None:
                            loss = custom_loss_fn(y, pred)
                        elif sw_sqrt is not None:
                            loss = torch.mean(sample_weight * (pred - y) ** 2)
                        else:
                            loss = F.mse_loss(pred, y)
                        loss.backward()
                        return loss

                    optimizer.step(closure)

                    with torch.no_grad():
                        self.weight.clamp_(min=0)  # Strict enforcement of the positive constraint

            finally:
                if self.n_jobs is not None:
                    torch.set_num_threads(prev_threads)

            self.fit_status = True


            # Update internal tracking


            self.device = str(target_device)


            self.dtype = target_dtype
            # Update internal tracking
            self.device = str(target_device)
            self.dtype = target_dtype

            # Update internal tracking

            self.device = str(target_device)

            self.dtype = target_dtype


            return self
        finally:
            self._fit_loss_fn = None


class Ridge(LinearRegression):
    def __init__(self,
                 alpha: Union[float, List[float], Tuple[float], torch.Tensor] = 1.0,
                 fit_intercept: bool = True,
                 copy_X: bool = True,
                 max_iter: int = None,
                 tol: float = 1e-6,
                 solver: str = "auto",
                 n_jobs: int = None,
                 positive: bool = False,
                 random_state: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__(fit_intercept=fit_intercept,
                         copy_X=copy_X,
                         tol=tol,
                         n_jobs=n_jobs,
                         positive=positive,
                         device=device,
                         dtype=dtype,
                         *args, **kwargs)
        self.alpha = alpha
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state
        self.num_iter = None

    @property
    def solver_(self):
        return self.solver

    @property
    def n_iter_(self):
        return self.num_iter

    def fit(self, data_or_X, y=None, **kwargs):
        # Determine target device and dtype from input data
        target_device = data_or_X.device if isinstance(data_or_X, torch.Tensor) else self.device
        target_dtype = data_or_X.dtype if isinstance(data_or_X, torch.Tensor) else self.dtype

        # Determine target device and dtype from input data
        
        

        self._fit_loss_fn = kwargs.get('loss_fn', None)
        try:
            warm_start = kwargs.get('warm_start', getattr(self, 'warm_start', False))
            # 1. Preprocessing and Parameter Setup
            sample_weight = kwargs.get("sample_weight", None)
            X_working = torch.as_tensor(data_or_X, device=target_device, dtype=target_dtype)
            if self.copy_X:
                X_working = X_working.clone()
            
            if y is None:
                raise ValueError("y cannot be None for fitting.")

            # Target device/dtype from input
            target_device = X_working.device
            target_dtype = X_working.dtype
            
            y = torch.as_tensor(y, device=target_device, dtype=target_dtype)

            # Ensure y is at least 2D for consistent matrix operations
            if y.ndim == 1:
                y = y.unsqueeze(1)
                is_1d_y = True
            else:
                is_1d_y = False

            n_samples, n_features = X_working.shape
            n_targets = y.shape[1]

            if sample_weight is not None:
                sample_weight = torch.as_tensor(sample_weight, device=target_device, dtype=target_dtype)
                if sample_weight.ndim == 1:
                    sample_weight = sample_weight.unsqueeze(1)
                sw_sqrt = torch.sqrt(sample_weight)
            else:
                sw_sqrt = None

            # Initialize module if weight is None, not warm_start, or shape mismatch
            need_init = self.weight is None or not warm_start
            if not need_init:
                need_init = self.weight.shape[1] != X_working.shape[1] or self.weight.shape[0] != y.shape[1]
            if need_init:
                self._init_module(X_working, y)

            # Handle Parallelism (n_jobs)
            prev_threads = torch.get_num_threads()
            if self.n_jobs is not None:
                num_threads = self.n_jobs if self.n_jobs > 0 else torch.get_num_threads()
                torch.set_num_threads(num_threads)

            try:
                # Handle Alpha
                alpha = self.alpha
                if isinstance(alpha, (float, int)):
                    alpha = torch.full((n_targets,), float(alpha), device=target_device, dtype=target_dtype)
                elif isinstance(alpha, (list, tuple)):
                    alpha = torch.tensor(alpha, device=target_device, dtype=target_dtype)
                elif isinstance(alpha, torch.Tensor):
                    alpha = alpha.to(device=target_device, dtype=target_dtype).view(-1)

                # Centering to avoid regularizing intercept
                if self.fit_intercept:
                    if sample_weight is not None:
                        sw_sum = sample_weight.sum()
                        X_offset = (X_working * sample_weight).sum(dim=0, keepdim=True) / sw_sum
                        y_offset = (y * sample_weight).sum(dim=0, keepdim=True) / sw_sum
                    else:
                        X_offset = X_working.mean(dim=0, keepdim=True)
                        y_offset = y.mean(dim=0, keepdim=True)
                    X_working = X_working - X_offset
                    y_centered = y - y_offset
                else:
                    X_offset = None
                    y_offset = None
                    y_centered = y

                # Solver Selection
                solver = self.solver
                if solver == "auto":
                    if self.positive:
                        solver = "lbfgs"
                    elif n_samples > 100000:
                        solver = "sag"
                    elif n_features > n_samples:
                        solver = "svd"
                    else:
                        solver = "cholesky"

                # Weighted scaling for linear solvers
                if sw_sqrt is not None and solver in ["svd", "cholesky"]:
                    X_solve = X_working * sw_sqrt
                    y_solve = y_centered * sw_sqrt
                else:
                    X_solve = X_working
                    y_solve = y_centered

                # Execute Solvers
                if solver == "svd":
                    U, S, Vh = torch.linalg.svd(X_solve, full_matrices=False)
                    # w = V (S^2 + alpha)^-1 S U^T y
                    S2 = S ** 2
                    if torch.all(alpha == alpha[0]).item():
                        denom = S2 + alpha[0]
                        w = Vh.T @ (torch.diag(S / denom) @ (U.T @ y_solve))
                    else:
                        # Handle different alphas per target
                        w_list = []
                        for j in range(n_targets):
                            denom = S2 + alpha[j]
                            w_j = Vh.T @ (torch.diag(S / denom) @ (U.T @ y_solve[:, j:j + 1]))
                            w_list.append(w_j)
                        w = torch.cat(w_list, dim=1)

                elif solver == "cholesky":
                    # (X^T X + alpha I) w = X^T y
                    XtX = X_solve.T @ X_solve
                    if torch.all(alpha == alpha[0]).item():
                        A = XtX + alpha[0] * torch.eye(n_features, device=target_device, dtype=target_dtype)
                        w = torch.linalg.solve(A, X_solve.T @ y_solve)
                    else:
                        w_list = []
                        Xty = X_solve.T @ y_solve
                        for j in range(n_targets):
                            A = XtX + alpha[j] * torch.eye(n_features, device=target_device, dtype=target_dtype)
                            w_j = torch.linalg.solve(A, Xty[:, j:j + 1])
                            w_list.append(w_j)
                        w = torch.cat(w_list, dim=1)

                elif solver == "lbfgs":
                    # Iterative optimization
                    optimizer = torch.optim.LBFGS([self.weight] + ([self.bias] if self.fit_intercept else []),
                                                  lr=1, max_iter=self.max_iter or 100, tolerance_grad=self.tol)

                    def closure():
                        optimizer.zero_grad()
                        pred = F.linear(X_working, self.weight, self.bias)
                        custom_loss_fn = getattr(self, '_fit_loss_fn', None)
                        if custom_loss_fn is not None:
                            loss = custom_loss_fn(y_centered, pred)
                        elif sample_weight is not None:
                            loss = torch.mean(sample_weight * (pred - y_centered) ** 2)
                        else:
                            loss = F.mse_loss(pred, y_centered)
                        # Add Ridge penalty (scaled by n_samples since MSE loss is used or mean loss)
                        penalty = torch.sum(alpha * torch.sum(self.weight ** 2, dim=1)) / (n_targets * n_samples)
                        (loss + penalty).backward()
                        return loss + penalty

                    self.num_iter = optimizer.step(closure)
                    if self.positive:
                        with torch.no_grad():
                            self.weight.clamp_(min=0)
                    self.fit_status = True

                    # Update internal tracking

                    self.device = str(target_device)

                    self.dtype = target_dtype
                    # Update internal tracking
                    self.device = str(target_device)
                    self.dtype = target_dtype
                    # Update internal tracking
                    self.device = str(target_device)
                    self.dtype = target_dtype

                    return self

                elif solver in ["sag", "saga", "lsqr", "sparse_cg"]:
                    if self.random_state is not None:
                        torch.manual_seed(self.random_state)

                    optimizer = torch.optim.Adam([self.weight] + ([self.bias] if self.fit_intercept else []), lr=0.01)
                    self.num_iter = 0
                    for _ in range(self.max_iter or 1000):
                        self.num_iter += 1
                        optimizer.zero_grad()
                        pred = F.linear(X_working, self.weight, self.bias)
                        custom_loss_fn = getattr(self, '_fit_loss_fn', None)
                        if custom_loss_fn is not None:
                            loss = custom_loss_fn(y_centered, pred)
                        elif sample_weight is not None:
                            loss = torch.mean(sample_weight * (pred - y_centered) ** 2)
                        else:
                            loss = F.mse_loss(pred, y_centered)
                        penalty = torch.sum(alpha * torch.sum(self.weight ** 2, dim=1)) / (n_targets * n_samples)
                        total_loss = loss + penalty
                        total_loss.backward()
                        optimizer.step()
                        if total_loss.item() < self.tol:
                            break

                    if self.positive:
                        with torch.no_grad():
                            self.weight.clamp_(min=0)
                    return self

                else:
                    raise ValueError(f"Unknown solver: {solver}")

                # Finalize weights and intercept for closed-form solvers
                with torch.no_grad():
                    self.weight.copy_(w.T)
                    if self.fit_intercept:
                        # b = y_mean - x_mean @ w
                        bias_val = (y_offset - X_offset @ w).squeeze(0)
                        if is_1d_y:
                            self.bias.copy_(bias_val.squeeze(-1))
                        else:
                            self.bias.copy_(bias_val)
                    else:
                        if self.bias is not None:
                            self.bias.zero_()
                    # Update internal tracking to reflect final state
                    self.device = str(target_device)
                    self.dtype = target_dtype

            finally:
                if self.n_jobs is not None:
                    torch.set_num_threads(prev_threads)

            return self
        finally:
            self._fit_loss_fn = None


class RidgeCV(MLRegressor):
    def __init__(self,
                 alphas: Union[List[float], Tuple[float], torch.Tensor] = (0.1, 1.0, 10.0),
                 fit_intercept: bool = True,
                 scoring: Union[str, Callable] = None,
                 cv: Union[int, MLModule, Iterable] = None,
                 cv_config: dict = None,
                 gcv_mode: str = "auto",
                 store_cv_results: bool = False,
                 alpha_per_target: bool = False,
                 copy_X: bool = True,
                 max_iter: int = None,
                 tol: float = 1e-6,
                 solver: str = "auto",
                 n_jobs: int = None,
                 positive: bool = False,
                 random_state: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.alpha = None
        self.alphas = alphas
        # Initialize Ridge with the first alpha just for structure, GridSearchCV will override
        initial_alpha = alphas[0] if isinstance(alphas, (list, tuple)) else alphas.ravel()[0].item() if isinstance(
            alphas, torch.Tensor) else 1.0

        self.ridge = Ridge(
            alpha=initial_alpha,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            max_iter=max_iter,
            tol=tol,
            solver=solver,
            n_jobs=n_jobs,
            positive=positive,
            random_state=random_state,
            dtype=dtype,
            device=device,
            *args, **kwargs
        )

        if isinstance(alphas, torch.Tensor):
            self.alphas = self.alphas.tolist()

        self.params_dict = {
            "alpha": self.alphas,
            "fit_intercept": [fit_intercept],  # Grid search expects lists
            "positive": [positive]
        }

        # Handle cv_config
        cv_config = cv_config if cv_config is not None else {}
        # Handle cv_config
        cv_config = cv_config if cv_config is not None else {}
        cv_config["gcv_mode"] = gcv_mode
        self.alpha_per_target = alpha_per_target

        # RidgeCV defaults to LeaveOneOut if cv is None
        if cv is None:
            from ...cross_validation.splitters import LeaveOneOut
            cv = LeaveOneOut(**cv_config)  # Utilizes gcv_mode if implemented in LOO

        self.search_params = {
            "estimator": self.ridge,
            "param_grid": self.params_dict,
            "scoring": scoring,
            "n_jobs": n_jobs,
            "cv": cv,
            "cv_config": cv_config,
            "return_train_score": True,
            "device": device,
            "dtype": dtype,
            "store_cv_values": store_cv_results
        }
        from ...cross_validation.search_cv import GridSearchCV
        self.search = GridSearchCV(**self.search_params)
        self.estimator = None
        self.fit_status = False
        self._cv_results = None
        self._best_score = None
        self._best_index = None
        self.device = device
        self.dtype = dtype

    def forward(self, X: torch.Tensor, y: Optional[torch.Tensor] = None):
        if not self.fit_status:
            self.fit(X, y)
            self.fit_status = True
        return self.estimator(X)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if self.estimator is None:
            raise ValueError("RidgeCV must be fitted before predict().")
        self.estimator.to(self.device)
        return self.estimator.predict(X)

    def fit(self, data_or_X, y=None, **kwargs):
        # Determine target device and dtype from input data
        target_device = data_or_X.device if isinstance(data_or_X, torch.Tensor) else self.device
        target_dtype = data_or_X.dtype if isinstance(data_or_X, torch.Tensor) else self.dtype

        # Determine target device and dtype from input data
        
        
        
        X = torch.as_tensor(data_or_X, device=target_device, dtype=target_dtype)
        if y is not None:
            y = torch.as_tensor(y, device=target_device, dtype=target_dtype)
        # Check for multi-output and alpha_per_target
        if y is not None and y.ndim > 1 and y.shape[1] > 1 and self.alpha_per_target:
            from ...cross_validation.search_cv import GridSearchCV  # Import needed here
            n_targets = y.shape[1]
            best_alphas = []
            best_scores = []
            all_cv_values = []

            # Independent search per target
            for i in range(n_targets):
                y_target = y[:, i:i + 1]  # Keep 2D (n, 1)
                search = GridSearchCV(**self.search_params)
                search.fit(X, y_target, **kwargs)

                # Collect best alpha
                best_alpha = search.best_params_['alpha']
                best_alphas.append(best_alpha)
                best_scores.append(search.best_score_)

                # Collect cv_values if present
                if hasattr(search, 'cv_values_') and search.cv_values_ is not None:
                    all_cv_values.append(search.cv_values_)

            self.alpha = best_alphas  # List of scalars
            self._best_score = best_scores  # List of scores

            # Refit final estimator with vector alphas
            self.estimator = Ridge(alpha=self.alpha, fit_intercept=self.ridge.fit_intercept,
                                   copy_X=self.ridge.copy_X, max_iter=self.ridge.max_iter,
                                   tol=self.ridge.tol, solver=self.ridge.solver,
                                   n_jobs=self.ridge.n_jobs, positive=self.ridge.positive,
                                   random_state=self.ridge.random_state, dtype=self.ridge.dtype,
                                   device=self.ridge.device)

            self.estimator.fit(X, y, **kwargs)  # Ridge.fit handles vector alpha

            # Combine cv_results
            if all_cv_values:
                # Stack: (n_samples, n_candidates) -> (n_samples, n_targets, n_candidates)
                try:
                    self._cv_results = torch.stack(all_cv_values, dim=1)
                except:
                    self._cv_results = all_cv_values

        else:
            self.search.fit(X, y, **kwargs)
            self.estimator = self.search.best_estimator_
            self.alpha = self.search.best_params_['alpha']
            self._best_score = self.search.best_score_
            if hasattr(self.search, 'cv_values_'):
                self._cv_results = self.search.cv_values_

        self.fit_status = True


        # Update internal tracking


        self.device = str(target_device)


        self.dtype = target_dtype
        # Update internal tracking
        self.device = str(target_device)
        self.dtype = target_dtype
        # Update internal tracking
        self.device = str(target_device)
        self.dtype = target_dtype
        # Update internal tracking
        self.device = str(target_device)
        self.dtype = target_dtype
        return self

    @property
    def cv_results_(self):
        if self._cv_results is not None:
            return self._cv_results
        return self.search.cv_results_  # Fallback to dict if no store_cv_values

    @property
    def coef_(self):
        return self.estimator.coef_

    @property
    def intercept_(self):
        return self.estimator.intercept_

    @property
    def alpha_(self):
        return self.alpha

    @property
    def best_score_(self):
        return self._best_score

    @property
    def n_features_in_(self):
        return self.estimator.in_features


class Lasso(LinearRegression):
    def __init__(self,
                 alpha: Union[float, List[float], Tuple[float], torch.Tensor] = 1.0,
                 fit_intercept: bool = True,
                 precompute: Union[bool, List[list], Tuple[tuple], torch.Tensor] = False,
                 copy_X: bool = True,
                 max_iter: int = None,
                 tol: float = 1e-6,
                 warm_start: bool = False,
                 solver: str = "auto",
                 n_jobs: int = None,
                 positive: bool = False,
                 random_state: int = None,
                 selection: str = "cyclic",
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__(fit_intercept=fit_intercept,
                         copy_X=copy_X,
                         tol=tol,
                         n_jobs=n_jobs,
                         positive=positive,
                         device=device,
                         dtype=dtype,
                         *args, **kwargs)
        self.alpha = alpha
        self.num_iter = None
        self.precompute = precompute
        self.max_iter = max_iter
        self.warm_start = warm_start
        self.solver = solver
        self.random_state = random_state
        self.selection = selection

    def _compute_dual_gap(self, X: torch.Tensor, y: torch.Tensor):
        predictions = self.forward(X, y)
        residuals = y - predictions
        mse_part = torch.sum(residuals ** 2) / (2 * X.shape[0])
        l1_part = self.alpha * torch.norm(self.weight, p=1)
        primal_obj = mse_part + l1_part
        theta = residuals / X.shape[0]
        xt_theta = torch.matmul(X.T, theta)
        max_xt_theta = torch.norm(xt_theta, p=float("inf"))
        if max_xt_theta > self.alpha:
            dual_scale = self.alpha / max_xt_theta
            theta = theta * dual_scale
        y_n = y / X.shape[0]
        dual_obj = 0.5 * (torch.sum(y ** 2) / X.shape[0]) - 0.5 * X.shape[0] * torch.sum((y_n - theta) ** 2)
        dual_gap = primal_obj - dual_obj
        return dual_gap

    @property
    def sparse_coef_(self):
        weight = self.weight.clone()
        weight[torch.abs(weight) < self.tol] = 0
        sparse_weight = weight.to_sparse_csr()
        return sparse_weight

    @property
    def sparse_intercept_(self):
        if not self.fit_intercept:
            return
        else:
            bias = self.bias.clone()
            sparse_bias = bias.to_sparse_csr()
            return sparse_bias

    @property
    def n_iter_(self):
        return self.num_iter

    def fit(self, data_or_X, y=None, **kwargs):
        # Determine target device and dtype from input data
        target_device = data_or_X.device if isinstance(data_or_X, torch.Tensor) else self.device
        target_dtype = data_or_X.dtype if isinstance(data_or_X, torch.Tensor) else self.dtype

        # Determine target device and dtype from input data
        
        

        self._fit_loss_fn = kwargs.get('loss_fn', None)
        try:
            # Determine target device and dtype from input data
            
            
            
            # 1. Preprocessing and Parameter Setup
            X_working = torch.as_tensor(data_or_X, device=target_device, dtype=target_dtype)
            if self.copy_X:
                X_working = X_working.clone()

            if y is None:
                raise ValueError("y cannot be None for fitting.")

            # Target device/dtype from input
            target_device = X_working.device
            target_dtype = X_working.dtype

            y = torch.as_tensor(y, device=target_device, dtype=target_dtype)
            sample_weight = kwargs.get("sample_weight", None)
            if sample_weight is not None:
                sample_weight = torch.as_tensor(sample_weight, device=target_device, dtype=target_dtype)
                if sample_weight.ndim == 1:
                    sample_weight = sample_weight.unsqueeze(1)

            # Ensure y is at least 2D for consistent matrix operations
            if y.ndim == 1:
                y = y.unsqueeze(1)
                is_1d_y = True
            else:
                is_1d_y = False

            n_samples, n_features = X_working.shape
            n_targets = y.shape[1]

            # Initialize module if weight is None or if not warm_start
            if self.weight is None or not self.warm_start:
                self._init_module(X_working, y)
                with torch.no_grad():
                    self.weight.zero_()
                    if self.bias is not None:
                        self.bias.zero_()

            # Handle Parallelism (n_jobs)
            prev_threads = torch.get_num_threads()
            if self.n_jobs is not None:
                num_threads = self.n_jobs if self.n_jobs > 0 else torch.get_num_threads()
                torch.set_num_threads(num_threads)

            try:
                # Handle Alpha
                alpha = self.alpha
                if isinstance(alpha, (float, int)):
                    alpha = torch.full((n_targets,), float(alpha), device=target_device, dtype=target_dtype)
                elif isinstance(alpha, (list, tuple)):
                    alpha = torch.tensor(alpha, device=target_device, dtype=target_dtype)
                elif isinstance(alpha, torch.Tensor):
                    alpha = alpha.to(device=target_device, dtype=target_dtype).view(-1)

                # Centering to avoid regularizing intercept
                if self.fit_intercept:
                    if sample_weight is not None:
                        sw_sum = sample_weight.sum(dim=0, keepdim=True)
                        X_offset = (sample_weight * X_working).sum(dim=0, keepdim=True) / sw_sum
                        y_offset = (sample_weight * y).sum(dim=0, keepdim=True) / sw_sum
                    else:
                        X_offset = X_working.mean(dim=0, keepdim=True)
                        y_offset = y.mean(dim=0, keepdim=True)
                    X_working = X_working - X_offset
                    y_centered = y - y_offset
                else:
                    X_offset = None
                    y_offset = None
                    y_centered = y

                # Solver Selection
                solver = self.solver
                if solver == "auto":
                    solver = "cd"  # Default for Lasso

                if solver == "cd":
                    # --- Coordinate Descent Solver ---
                    # Weighting: X_weighted = sqrt(sw) * X, y_weighted = sqrt(sw) * y
                    if sample_weight is not None:
                        sw_sqrt = torch.sqrt(sample_weight)
                        X_weighted = sw_sqrt * X_working
                        y_weighted = sw_sqrt * y_centered
                    else:
                        X_weighted = X_working
                        y_weighted = y_centered

                    # Precompute Gram matrix if requested
                    if self.precompute is True or isinstance(self.precompute, (torch.Tensor, list, tuple)):
                        if isinstance(self.precompute, torch.Tensor):
                            Gram = self.precompute.to(device=target_device, dtype=target_dtype)
                        elif isinstance(self.precompute, (list, tuple)):
                            Gram = torch.tensor(self.precompute, device=target_device, dtype=target_dtype)
                        else:
                            Gram = X_weighted.T @ X_weighted
                    else:
                        Gram = None

                    XtY = X_weighted.T @ y_weighted
                    X_norms_sq = torch.sum(X_weighted ** 2, dim=0)

                    # Weight shape in MLModule is (out_features, in_features) -> (n_targets, n_features)
                    W = self.weight.data.T.clone()

                    self.num_iter = 0
                    max_iter = self.max_iter or 1000

                    features_idx = torch.arange(n_features)
                    if self.random_state is not None:
                        torch.manual_seed(self.random_state)

                    for iteration in range(max_iter):
                        self.num_iter += 1
                        W_old = W.clone()

                        if self.selection == "random":
                            features_idx = features_idx[torch.randperm(n_features)]

                        for j in features_idx:
                            # rho_j = X_j^T (y - Xw + X_j w_j)
                            if Gram is not None:
                                rho_j = XtY[j] - Gram[j] @ W + Gram[j, j] * W[j]
                            else:
                                if iteration == 0 and j == 0:
                                    Gram = X_weighted.T @ X_weighted
                                rho_j = XtY[j] - Gram[j] @ W + Gram[j, j] * W[j]

                            # Soft-thresholding
                            threshold = n_samples * alpha
                            W[j] = torch.sign(rho_j) * torch.clamp(torch.abs(rho_j) - threshold, min=0)
                            if X_norms_sq[j] > 1e-12:
                                W[j] = W[j] / X_norms_sq[j]
                            else:
                                W[j] = 0.0

                            if self.positive:
                                W[j] = torch.clamp(W[j], min=0)

                        # Check convergence
                        if torch.max(torch.abs(W - W_old)) < self.tol:
                            break

                    with torch.no_grad():
                        self.weight.copy_(W.T)

                else:
                    # --- Proximal Gradient Descent for other_decomposition solvers ---
                    optimizer = torch.optim.Adam([self.weight] + ([self.bias] if self.fit_intercept else []), lr=0.01)
                    self.num_iter = 0
                    max_iter = self.max_iter or 1000

                    alpha_expanded = alpha.unsqueeze(1)

                    for _ in range(max_iter):
                        self.num_iter += 1
                        optimizer.zero_grad()
                        pred = F.linear(X_working, self.weight, self.bias)
                        custom_loss_fn = getattr(self, '_fit_loss_fn', None)
                        if custom_loss_fn is not None:
                            loss = custom_loss_fn(y_centered, pred)
                        elif sample_weight is not None:
                            loss = torch.mean(sample_weight * (pred - y_centered) ** 2)
                        else:
                            loss = F.mse_loss(pred, y_centered)
                        loss.backward()
                        optimizer.step()

                        # Proximal step
                        with torch.no_grad():
                            lr = 0.01
                            threshold = lr * alpha_expanded
                            self.weight.data = torch.sign(self.weight.data) * torch.clamp(
                                torch.abs(self.weight.data) - threshold, min=0)

                            if self.positive:
                                self.weight.data.clamp_(min=0)

                        if loss.item() < self.tol:
                            break

                # Finalize intercept
                with torch.no_grad():
                    if self.fit_intercept:
                        # b = y_mean - x_mean @ w
                        w_final = self.weight.T
                        bias_val = (y_offset - X_offset @ w_final).squeeze(0)
                        if is_1d_y:
                            self.bias.copy_(bias_val.squeeze(-1))
                        else:
                            self.bias.copy_(bias_val)
                    else:
                        if self.bias is not None:
                            self.bias.zero_()
                    
                    # Update internal tracking
                    self.device = str(target_device)
                    self.dtype = target_dtype

                self.dual_gap_ = torch.tensor([self.tol], device=target_device, dtype=target_dtype)  # Placeholder

            finally:
                if self.n_jobs is not None:
                    torch.set_num_threads(prev_threads)

            self.fit_status = True


            # Update internal tracking


            self.device = str(target_device)


            self.dtype = target_dtype
            # Update internal tracking
            self.device = str(target_device)
            self.dtype = target_dtype

            # Update internal tracking

            self.device = str(target_device)

            self.dtype = target_dtype


            return self
        finally:
            self._fit_loss_fn = None


class LassoCV(MLRegressor):
    def __init__(self,
                 eps: float = 1e-3,
                 n_alphas: int = 100,
                 alphas: Union[list, tuple, torch.Tensor] = None,
                 fit_intercept: bool = True,
                 precompute: Union[bool, List[list], Tuple[tuple], torch.Tensor] = False,
                 copy_X: bool = True,
                 max_iter: Union[int, List[int]] = 1000,
                 tol: Union[float, List[float]] = 1e-4,
                 warm_start: bool = False,
                 cv: Union[str, int, Iterable, Callable, MLModule] = None,
                 cv_config: dict = None,
                 scoring: Union[str, Callable, Iterable] = None,
                 verbose: Union[bool, int] = False,
                 solver: Union[str, List[str]] = "auto",
                 n_jobs: int = None,
                 positive: bool = False,
                 random_state: int = None,
                 selection: Union[str, List[str]] = "cyclic",
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 store_cv_values: bool = False,
                 alpha_per_target: bool = False,
                 *args, **kwargs):
        super().__init__()
        self.eps = eps
        self.n_alphas = n_alphas
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.precompute = precompute
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.cv = cv
        self.cv_config = cv_config
        self.scoring = scoring
        self.verbose = verbose
        self.solver = solver
        self.n_jobs = n_jobs
        self.positive = positive
        self.random_state = random_state
        self.selection = selection
        self.device = device
        self.dtype = dtype
        self.alpha_per_target = alpha_per_target
        self.store_cv_values = store_cv_values

        # Initialize base Lasso for cloning
        self.lasso = Lasso(
            fit_intercept=fit_intercept,
            precompute=precompute,
            copy_X=copy_X,
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            solver=solver,
            n_jobs=n_jobs,
            positive=positive,
            random_state=random_state,
            selection=selection,
            device=device,
            dtype=dtype,
            *args, **kwargs
        )

        if isinstance(alphas, (list, tuple)):
            self.alphas = torch.tensor(alphas, device=self.device, dtype=self.dtype)
        elif isinstance(alphas, torch.Tensor):
            self.alphas = self.alphas.to(device=self.device, dtype=self.dtype)

        # Params dict for search (alphas placeholders, populated in fit if None)
        self.params_dict = {
            "alpha": self.alphas if self.alphas is not None else [1.0],  # Placeholder
            "fit_intercept": [fit_intercept],
            "max_iter": [max_iter],
            "tol": [tol],
            "solver": [solver],
            "selection": [selection]
        }

        # Structure matches RidgeCV: Search setup in init
        # Handle cv_config
        cv_config = cv_config if cv_config is not None else {}

        # LassoCV defaults to 5-fold if cv is None
        if cv is None:
            from ...cross_validation.splitters import KFoldCV
            cv = KFoldCV(n_splits=5, **cv_config)

        self.search_params = {
            "estimator": self.lasso,
            "param_grid": self.params_dict,
            "scoring": scoring,
            "n_jobs": n_jobs,
            "cv": cv,
            "cv_config": cv_config,
            "return_train_score": True,
            "device": device,
            "dtype": dtype,
            "store_cv_values": store_cv_values,
            "verbose": verbose
        }

        from ...cross_validation.search_cv import GridSearchCV
        self.search = GridSearchCV(**self.search_params)

        self.estimator = None
        self.fit_status = False
        self._mse_path_ = None
        self._alphas_ = None
        self._alpha_ = None

    def fit(self, data_or_X, y=None, **kwargs):
        # Determine target device and dtype from input data
        target_device = data_or_X.device if isinstance(data_or_X, torch.Tensor) else self.device
        target_dtype = data_or_X.dtype if isinstance(data_or_X, torch.Tensor) else self.dtype

        try:
            
            
            
            X = torch.as_tensor(data_or_X, device=target_device, dtype=target_dtype)
            y = torch.as_tensor(y, device=target_device, dtype=target_dtype)

            if y.ndim == 1:
                y = y.unsqueeze(1)

            n_samples, n_features = X.shape

            # Use X_working/y_working for initial calculations (like alpha path)
            X_working = X.clone()
            y_working = y.clone()

            # Generate alphas if None
            if self.alphas is None:
                if self.fit_intercept:
                    X_working -= X_working.mean(dim=0)
                    y_working -= y_working.mean(dim=0)

                # Simple heuristic for alpha_max
                scores = torch.abs(X_working.T @ y_working)
                # Sklearn uses max over targets
                alpha_max = torch.max(scores) / n_samples
                alpha_min = self.eps * alpha_max

                import math
                self._alphas_ = torch.logspace(math.log10(alpha_max.item()), math.log10(alpha_min.item()),
                                               self.n_alphas, device=target_device, dtype=target_dtype)
                self._alphas_ = torch.flip(self._alphas_, dims=[0])
            else:
                self._alphas_ = self.alphas

            # Update params_dict with actual alphas
            # GridSearchCV expects list for iteration
            self.params_dict["alpha"] = self._alphas_.tolist()
            self.search.param_grid = self.params_dict  # Update grid on search obj

            # Handle alpha_per_target
            if y.ndim > 1 and y.shape[1] > 1 and self.alpha_per_target:
                from ...cross_validation.search_cv import GridSearchCV
                n_targets = y.shape[1]
                best_alphas = []
                mse_paths = []

                for i in range(n_targets):
                    y_target = y[:, i:i + 1]
                    # We need a fresh search object or deepcopy for safety
                    # Re-instantiate to ensure clean state or use copy.deepcopy
                    # Here we just re-use params but create new instance
                    search = GridSearchCV(**self.search_params)
                    search.param_grid = self.params_dict  # Ensure alphas are set

                    search.fit(X, y_target, **kwargs)
                    best_alphas.append(search.best_params_['alpha'])

                    # Extract Path
                    n_splits = search.n_splits_
                    mses = []
                    for k in range(n_splits):
                        key = f"split{k}_test_score"
                        if key in search.cv_results_:
                            mses.append(torch.tensor(search.cv_results_[key], device=target_device))
                    if mses:
                        mse_paths.append(torch.stack(mses, dim=1))

                self._alpha_ = best_alphas
                if mse_paths:
                    self._mse_path_ = torch.stack(mse_paths, dim=1)

                # Final Estimator Construction
                # Should use vector alpha specific Lasso logic?
                # Our Lasso supports list/tensor alpha.
                self.estimator = Lasso(
                    alpha=best_alphas,
                    fit_intercept=self.fit_intercept,
                    precompute=self.precompute,
                    copy_X=self.copy_X,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    warm_start=self.warm_start,
                    solver=self.solver,
                    n_jobs=self.n_jobs,
                    positive=self.positive,
                    random_state=self.random_state,
                    selection=self.selection,
                    device=target_device,
                    dtype=target_dtype
                )
                self.estimator.fit(X, y, **kwargs)

            else:
                self.search.fit(X, y, **kwargs)
                self.estimator = self.search.best_estimator_
                self._alpha_ = self.search.best_params_['alpha']

                n_splits = self.search.n_splits_
                mses = []
                for i in range(n_splits):
                    key = f"split{i}_test_score"
                    if key in self.search.cv_results_:
                        mses.append(torch.tensor(self.search.cv_results_[key], device=target_device))

                if mses:
                    self._mse_path_ = torch.stack(mses, dim=1)  # (n_alphas, n_folds)

            self.fit_status = True


            # Update internal tracking


            self.device = str(target_device)


            self.dtype = target_dtype
            # Update internal tracking
            self.device = str(target_device)
            self.dtype = target_dtype
            # Update internal tracking
            self.device = str(target_device)
            self.dtype = target_dtype
            return self

        except Exception as e:
            print(f"DEBUG EXCEPTION IN LassoCV.fit: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def forward(self, X: torch.Tensor, y: torch.Tensor = None):
        if not self.fit_status:
            if y is not None:
                self.fit(X, y)
            else:
                raise RuntimeError("Model not fitted")
        return self.estimator(X)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if self.estimator is None:
            raise ValueError("LassoCV must be fitted before predict().")
        return self.estimator.predict(X)

    # Properties to expose attributes
    @property
    def alpha_(self):
        return self._alpha_

    @property
    def alphas_(self):
        return self._alphas_

    @property
    def mse_path_(self):
        return self._mse_path_

    @property
    def coef_(self):
        return self.estimator.coef_

    @property
    def intercept_(self):
        return self.estimator.intercept_

    @property
    def n_iter_(self):
        return self.estimator.n_iter_

    @property
    def dual_gap_(self):
        return self.estimator.dual_gap_

    @property
    def n_features_in_(self):
        return self.estimator.n_features_in_


class ElasticNet(LinearRegression):
    def __init__(self,
                 alpha: float = 1.0,
                 l1_norm: float = 0.5,
                 fit_intercept: bool = True,
                 precompute: Union[bool, List[list], Tuple[tuple], torch.Tensor] = False,
                 copy_X: bool = True,
                 max_iter: int = None,
                 tol: float = 1e-6,
                 warm_start: bool = False,
                 solver: str = "auto",
                 n_jobs: int = None,
                 positive: bool = False,
                 random_state: int = None,
                 selection: str = "cyclic",
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__(fit_intercept=fit_intercept,
                         copy_X=copy_X,
                         tol=tol,
                         n_jobs=n_jobs,
                         positive=positive,
                         device=device,
                         dtype=dtype,
                         *args, **kwargs)
        self.alpha = alpha
        self.l1_norm = l1_norm
        self.num_iter = None
        self.dual_gap_ = None
        self.precompute = precompute
        self.max_iter = max_iter
        self.warm_start = warm_start
        self.solver = solver
        self.random_state = random_state
        self.selection = selection

    def _compute_dual_gap(self, X: torch.Tensor, y: torch.Tensor):
        predictions = self.forward(X, y)
        residuals = y - predictions
        mse_part = torch.sum(residuals ** 2) / (2 * X.shape[0])
        l1_part = self.alpha * torch.norm(self.weight, p=1)
        primal_obj = mse_part + l1_part
        theta = residuals / X.shape[0]
        xt_theta = torch.matmul(X.T, theta)
        max_xt_theta = torch.norm(xt_theta, p=float("inf"))
        if max_xt_theta > self.alpha:
            dual_scale = self.alpha / max_xt_theta
            theta = theta * dual_scale
        y_n = y / X.shape[0]
        dual_obj = 0.5 * (torch.sum(y ** 2) / X.shape[0]) - 0.5 * X.shape[0] * torch.sum((y_n - theta) ** 2)
        dual_gap = primal_obj - dual_obj
        return dual_gap

    @property
    def sparse_coef_(self):
        weight = self.weight.clone()
        weight[torch.abs(weight) < self.tol] = 0
        sparse_weight = weight.to_sparse_csr()
        return sparse_weight

    @property
    def sparse_intercept_(self):
        if not self.fit_intercept:
            return
        else:
            bias = self.bias.clone()
            sparse_bias = bias.to_sparse_csr()
            return sparse_bias

    @property
    def n_iter_(self):
        return self.num_iter

    def fit(self, data_or_X, y=None, **kwargs):
        # Determine target device and dtype from input data
        target_device = data_or_X.device if isinstance(data_or_X, torch.Tensor) else self.device
        target_dtype = data_or_X.dtype if isinstance(data_or_X, torch.Tensor) else self.dtype

        # Determine target device and dtype from input data
        
        

        self._fit_loss_fn = kwargs.get('loss_fn', None)
        try:
            # Determine target device and dtype from input data
            
            
            
            # 1. Preprocessing and Parameter Setup
            X_working = torch.as_tensor(data_or_X, device=target_device, dtype=target_dtype)
            if self.copy_X:
                X_working = X_working.clone()

            if y is None:
                raise ValueError("y cannot be None for fitting.")

            y = y.to(device=X_working.device, dtype=X_working.dtype)
            sample_weight = kwargs.get("sample_weight", None)
            if sample_weight is not None:
                sample_weight = torch.as_tensor(sample_weight, device=X_working.device, dtype=X_working.dtype)
                if sample_weight.ndim == 1:
                    sample_weight = sample_weight.unsqueeze(1)
            
            # Target device/dtype from input
            target_device = X_working.device
            target_dtype = X_working.dtype

            # Ensure y is at least 2D for consistent matrix operations
            if y.ndim == 1:
                y = y.unsqueeze(1)
                is_1d_y = True
            else:
                is_1d_y = False

            n_samples, n_features = X_working.shape
            n_targets = y.shape[1]

            # Initialize module if weight is None or if not warm_start
            if self.weight is None or not self.warm_start:
                self._init_module(X_working, y)
                with torch.no_grad():
                    self.weight.zero_()
                    if self.bias is not None:
                        self.bias.zero_()

            # Handle Parallelism (n_jobs)
            prev_threads = torch.get_num_threads()
            if self.n_jobs is not None:
                num_threads = self.n_jobs if self.n_jobs > 0 else torch.get_num_threads()
                torch.set_num_threads(num_threads)

            try:
                # Handle Alpha
                alpha = self.alpha
                if isinstance(alpha, (float, int)):
                    alpha = torch.full((n_targets,), float(alpha), device=target_device, dtype=target_dtype)
                elif isinstance(alpha, (list, tuple)):
                    alpha = torch.tensor(alpha, device=target_device, dtype=target_dtype)
                elif isinstance(alpha, torch.Tensor):
                    alpha = alpha.to(device=target_device, dtype=target_dtype).view(-1)

                # Handle l1_norm
                l1_norm = self.l1_norm
                if isinstance(l1_norm, (float, int)):
                    l1_norm = torch.full((n_targets,), float(l1_norm), device=target_device, dtype=target_dtype)
                elif isinstance(l1_norm, (list, tuple)):
                    l1_norm = torch.tensor(l1_norm, device=target_device, dtype=target_dtype)
                elif isinstance(l1_norm, torch.Tensor):
                    l1_norm = l1_norm.to(device=target_device, dtype=target_dtype).view(-1)

                # Penalties
                l1_penalty = alpha * l1_norm
                l2_penalty = alpha * (1 - l1_norm)

                # Centering to avoid regularizing intercept
                if self.fit_intercept:
                    if sample_weight is not None:
                        sw_sum = sample_weight.sum(dim=0, keepdim=True)
                        X_offset = (sample_weight * X_working).sum(dim=0, keepdim=True) / sw_sum
                        y_offset = (sample_weight * y).sum(dim=0, keepdim=True) / sw_sum
                    else:
                        X_offset = X_working.mean(dim=0, keepdim=True)
                        y_offset = y.mean(dim=0, keepdim=True)
                    X_working = X_working - X_offset
                    y_centered = y - y_offset
                else:
                    X_offset = None
                    y_offset = None
                    y_centered = y

                # Solver Selection
                solver = self.solver
                if solver == "auto":
                    solver = "cd"  # Default for ElasticNet

                if solver == "cd":
                    # --- Coordinate Descent Solver ---
                    # Weighting: X_weighted = sqrt(sw) * X, y_weighted = sqrt(sw) * y
                    if sample_weight is not None:
                        sw_sqrt = torch.sqrt(sample_weight)
                        X_weighted = sw_sqrt * X_working
                        y_weighted = sw_sqrt * y_centered
                    else:
                        X_weighted = X_working
                        y_weighted = y_centered

                    # Precompute Gram matrix if requested
                    if self.precompute is True or isinstance(self.precompute, (torch.Tensor, list, tuple)):
                        if isinstance(self.precompute, torch.Tensor):
                            Gram = self.precompute.to(device=target_device, dtype=target_dtype)
                        elif isinstance(self.precompute, (list, tuple)):
                            Gram = torch.tensor(self.precompute, device=target_device, dtype=target_dtype)
                        else:
                            Gram = X_weighted.T @ X_weighted
                    else:
                        Gram = None

                    XtY = X_weighted.T @ y_weighted
                    X_norms_sq = torch.sum(X_weighted ** 2, dim=0)

                    # Weight shape in MLModule is (out_features, in_features) -> (n_targets, n_features)
                    W = self.weight.data.T.clone()

                    self.num_iter = 0
                    max_iter = self.max_iter or 1000

                    features_idx = torch.arange(n_features)
                    if self.random_state is not None:
                        torch.manual_seed(self.random_state)

                    for iteration in range(max_iter):
                        self.num_iter += 1
                        W_old = W.clone()

                        if self.selection == "random":
                            features_idx = features_idx[torch.randperm(n_features)]

                        for j in features_idx:
                            # rho_j = X_j^T (y - Xw + X_j w_j)
                            if Gram is not None:
                                rho_j = XtY[j] - Gram[j] @ W + Gram[j, j] * W[j]
                            else:
                                if iteration == 0 and j == 0:
                                    Gram = X_weighted.T @ X_weighted
                                rho_j = XtY[j] - Gram[j] @ W + Gram[j, j] * W[j]

                            # Soft-thresholding with L1 and L2
                            threshold = n_samples * l1_penalty
                            denom = X_norms_sq[j] + n_samples * l2_penalty

                            W[j] = torch.sign(rho_j) * torch.clamp(torch.abs(rho_j) - threshold, min=0)

                            # Handle division by zero
                            mask = denom > 1e-12
                            W[j, mask] = W[j, mask] / denom[mask]
                            W[j, ~mask] = 0.0

                            if self.positive:
                                W[j] = torch.clamp(W[j], min=0)

                        # Check convergence
                        if torch.max(torch.abs(W - W_old)) < self.tol:
                            break

                    with torch.no_grad():
                        self.weight.copy_(W.T)

                else:
                    # --- Proximal Gradient Descent ---
                    optimizer = torch.optim.Adam([self.weight] + ([self.bias] if self.fit_intercept else []), lr=0.01)
                    self.num_iter = 0
                    max_iter = self.max_iter or 1000

                    l1_expanded = l1_penalty.unsqueeze(1)
                    l2_expanded = l2_penalty.unsqueeze(1)

                    for _ in range(max_iter):
                        self.num_iter += 1
                        optimizer.zero_grad()
                        pred = F.linear(X_working, self.weight, self.bias)
                        custom_loss_fn = getattr(self, '_fit_loss_fn', None)
                        if custom_loss_fn is not None:
                            loss = custom_loss_fn(y_centered, pred)
                        elif sample_weight is not None:
                            loss = torch.mean(sample_weight * (pred - y_centered) ** 2)
                        else:
                            loss = F.mse_loss(pred, y_centered)
                        # Add L2 penalty (MSE based)
                        l2_loss = torch.sum(l2_expanded * torch.sum(self.weight ** 2, dim=1)) / (n_targets * 2)
                        (loss + l2_loss).backward()
                        optimizer.step()

                        # Proximal step for L1
                        with torch.no_grad():
                            lr = 0.01
                            threshold = lr * l1_expanded
                            self.weight.data = torch.sign(self.weight.data) * torch.clamp(
                                torch.abs(self.weight.data) - threshold, min=0)

                            if self.positive:
                                self.weight.data.clamp_(min=0)

                        if loss.item() < self.tol:
                            break

                # Finalize intercept
                with torch.no_grad():
                    if self.fit_intercept:
                        w_final = self.weight.T
                        bias_val = (y_offset - X_offset @ w_final).squeeze(0)
                        if is_1d_y:
                            self.bias.copy_(bias_val.squeeze(-1))
                        else:
                            self.bias.copy_(bias_val)
                    else:
                        if self.bias is not None:
                            self.bias.zero_()

                # Update internal tracking
                self.device = str(target_device)
                self.dtype = target_dtype

                # Calculate dual gap (placeholder consistent with Lasso)
                self.dual_gap_ = torch.tensor([self.tol], device=target_device, dtype=target_dtype)

            finally:
                if self.n_jobs is not None:
                    torch.set_num_threads(prev_threads)

            self.fit_status = True


            # Update internal tracking


            self.device = str(target_device)


            self.dtype = target_dtype
            # Update internal tracking
            self.device = str(target_device)
            self.dtype = target_dtype

            # Update internal tracking

            self.device = str(target_device)

            self.dtype = target_dtype


            return self
        finally:
            self._fit_loss_fn = None


class ElasticNetCV(MLRegressor):
    def __init__(self,
                 l1_ratio: Union[float, List[float], Tuple[float], torch.Tensor] = 0.5,
                 eps: float = 1e-3,
                 n_alphas: int = 100,
                 alphas: Union[List[float], Tuple[float], torch.Tensor] = None,
                 fit_intercept: bool = True,
                 precompute: Union[bool, str, List[list], Tuple[tuple], torch.Tensor] = 'auto',
                 max_iter: int = 1000,
                 tol: float = 1e-4,
                 cv: Union[str, int, Iterable, Callable, MLModule] = None,
                 cv_config: dict = None,
                 copy_X: bool = True,
                 verbose: Union[bool, int] = 0,
                 n_jobs: int = None,
                 positive: bool = False,
                 random_state: int = None,
                 selection: str = 'cyclic',
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 store_cv_values: bool = False,
                 alpha_per_target: bool = False,
                 scoring: Union[str, Callable] = None,
                 warm_start: bool = False,
                 *args, **kwargs):
        super().__init__()
        self.l1_ratio = l1_ratio
        self.eps = eps
        self.n_alphas = n_alphas
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.precompute = precompute
        self.max_iter = max_iter
        self.tol = tol
        self.cv = cv
        self.cv_config = cv_config
        self.copy_X = copy_X
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.positive = positive
        self.random_state = random_state
        self.selection = selection
        self.device = device
        self.dtype = dtype
        self.store_cv_values = store_cv_values
        self.alpha_per_target = alpha_per_target
        self.scoring = scoring
        self.warm_start = warm_start

        # Process l1_ratio into a list for GridSearchCV
        if isinstance(l1_ratio, (float, int)):
            self.l1_ratio_list = [float(l1_ratio)]
        elif isinstance(l1_ratio, torch.Tensor):
            self.l1_ratio_list = l1_ratio.tolist()
        else:
            self.l1_ratio_list = list(l1_ratio)

        # Initialize base ElasticNet for cloning
        # ElasticNet uses l1_norm
        self.elastic_net = ElasticNet(
            alpha=1.0,  # Placeholder
            l1_norm=self.l1_ratio_list[0],
            fit_intercept=fit_intercept,
            precompute=precompute if precompute != 'auto' else False,
            copy_X=copy_X,
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            n_jobs=n_jobs,
            positive=positive,
            random_state=random_state,
            selection=selection,
            device=device,
            dtype=dtype,
            *args, **kwargs
        )

        # Search Setup (alphas placeholders, populated in fit)
        # Note: ElasticNet expects 'l1_norm'
        self.params_dict = {
            "alpha": [1.0],  # Placeholder
            "l1_norm": self.l1_ratio_list,
            "fit_intercept": [fit_intercept],
            "max_iter": [max_iter],
            "tol": [tol],
            "selection": [selection]
        }

        cv_config = cv_config if cv_config is not None else {}
        if cv is None:
            from ...cross_validation.splitters import KFoldCV
            cv = KFoldCV(n_splits=5, **cv_config)

        self.search_params = {
            "estimator": self.elastic_net,
            "param_grid": self.params_dict,
            "scoring": scoring,
            "n_jobs": n_jobs,
            "cv": cv,
            "cv_config": cv_config,
            "return_train_score": True,
            "device": device,
            "dtype": dtype,
            "store_cv_values": store_cv_values,
            "verbose": verbose
        }

        from ...cross_validation.search_cv import GridSearchCV
        self.search = GridSearchCV(**self.search_params)

        self.estimator = None
        self.fit_status = False
        self._mse_path_ = None
        self._alphas_ = None
        self._alpha_ = None
        self._l1_ratio_ = None

    def fit(self, data_or_X, y=None, **kwargs):
        # Determine target device and dtype from input data
        target_device = data_or_X.device if isinstance(data_or_X, torch.Tensor) else self.device
        target_dtype = data_or_X.dtype if isinstance(data_or_X, torch.Tensor) else self.dtype

        try:
            # Determine target device and dtype from input data
            
            
            
            X = torch.as_tensor(data_or_X, device=target_device, dtype=target_dtype)
            y = torch.as_tensor(y, device=target_device, dtype=target_dtype)

            if y.ndim == 1:
                y = y.unsqueeze(1)

            n_samples, n_features = X.shape

            # Alpha path generation logic
            if self.alphas is None:
                X_working = X.clone()
                y_working = y.clone()
                if self.fit_intercept:
                    X_working -= X_working.mean(dim=0)
                    y_working -= y_working.mean(dim=0)

                # Heuristic: alpha_max = max(abs(X.T @ y)) / (n_samples * min_l1_ratio)
                # To ensure even with small l1_ratio, we have a zeroing alpha.
                min_l1 = max(1e-4, min(self.l1_ratio_list))
                scores = torch.abs(X_working.T @ y_working)
                alpha_max = torch.max(scores) / (n_samples * min_l1)
                alpha_min = self.eps * alpha_max

                import math
                self._alphas_ = torch.logspace(math.log10(alpha_max.item()), math.log10(alpha_min.item()),
                                               self.n_alphas, device=target_device, dtype=target_dtype)
                self._alphas_ = torch.flip(self._alphas_, dims=[0])
            else:
                if isinstance(self.alphas, torch.Tensor):
                    self._alphas_ = self.alphas.to(target_device, target_dtype)
                else:
                    self._alphas_ = torch.tensor(self.alphas, device=target_device, dtype=target_dtype)

            # Update params_dict with actual alphas
            self.params_dict["alpha"] = self._alphas_.tolist()
            self.search.param_grid = self.params_dict

            if y.ndim > 1 and y.shape[1] > 1 and self.alpha_per_target:
                from ...cross_validation.search_cv import GridSearchCV
                n_targets = y.shape[1]
                best_alphas = []
                best_l1_ratios = []
                mse_paths = []

                for i in range(n_targets):
                    y_target = y[:, i:i + 1]
                    search = GridSearchCV(**self.search_params)
                    search.param_grid = self.params_dict
                    search.fit(X, y_target, **kwargs)

                    best_alphas.append(search.best_params_['alpha'])
                    best_l1_ratios.append(search.best_params_['l1_norm'])

                    # Extract MSE Path
                    n_splits = search.n_splits_
                    mses = []
                    for k in range(n_splits):
                        key = f"split{k}_test_score"
                        if key in search.cv_results_:
                            # Shape (n_l1_ratio * n_alphas)
                            mses.append(torch.tensor(search.cv_results_[key], device=target_device))
                    if mses:
                        # Stack to (n_candidates, n_splits)
                        # Then reshape to (n_l1_ratio, n_alphas, n_splits)
                        path = torch.stack(mses, dim=1)
                        path = path.view(len(self.l1_ratio_list), len(self._alphas_), n_splits)
                        mse_paths.append(path)

                self._alpha_ = best_alphas
                self._l1_ratio_ = best_l1_ratios
                if mse_paths:
                    # (n_targets, n_l1_ratio, n_alphas, n_splits)
                    self._mse_path_ = torch.stack(mse_paths, dim=0)

                # Final Refit
                self.estimator = ElasticNet(
                    alpha=best_alphas,
                    l1_norm=best_l1_ratios,
                    fit_intercept=self.fit_intercept,
                    precompute=self.precompute if self.precompute != 'auto' else False,
                    copy_X=self.copy_X,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    warm_start=self.warm_start,
                    n_jobs=self.n_jobs,
                    positive=self.positive,
                    random_state=self.random_state,
                    selection=self.selection,
                    device=target_device,
                    dtype=target_dtype
                )
                self.estimator.fit(X, y, **kwargs)

            else:
                self.search.fit(X, y, **kwargs)
                self.estimator = self.search.best_estimator_
                self._alpha_ = self.search.best_params_['alpha']
                self._l1_ratio_ = self.search.best_params_['l1_norm']

                # Extract MSE Path
                n_splits = self.search.n_splits_
                mses = []
                for i in range(n_splits):
                    key = f"split{i}_test_score"
                    if key in self.search.cv_results_:
                        mses.append(torch.tensor(self.search.cv_results_[key], device=target_device))
                if mses:
                    path = torch.stack(mses, dim=1)
                    self._mse_path_ = path.view(len(self.l1_ratio_list), len(self._alphas_), n_splits)

            self.fit_status = True


            # Update internal tracking


            self.device = str(target_device)


            self.dtype = target_dtype
            # Update internal tracking
            self.device = str(target_device)
            self.dtype = target_dtype
            # Update internal tracking
            self.device = str(target_device)
            self.dtype = target_dtype
            return self

        except Exception as e:
            print(f"DEBUG EXCEPTION IN ElasticNetCV.fit: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def forward(self, X: torch.Tensor, y: torch.Tensor = None):
        if not self.fit_status:
            if y is not None:
                self.fit(X, y)
            else:
                raise RuntimeError("Model not fitted")
        return self.estimator(X)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if self.estimator is None:
            raise ValueError("ElasticNetCV must be fitted before predict().")
        return self.estimator.predict(X)

    @property
    def alpha_(self):
        return self._alpha_

    @property
    def l1_ratio_(self):
        return self._l1_ratio_

    @property
    def coef_(self):
        return self.estimator.coef_

    @property
    def intercept_(self):
        return self.estimator.intercept_

    @property
    def mse_path_(self):
        return self._mse_path_

    @property
    def alphas_(self):
        return self._alphas_

    @property
    def dual_gap_(self):
        return self.estimator.dual_gap_

    @property
    def n_iter_(self):
        return self.estimator.n_iter_

    @property
    def n_features_in_(self):
        return self.estimator.n_features_in_






































































































class HuberRegressor(LinearRegression):

    def __init__(self,
                 epsilon: float = 1.35,
                 alpha: float = 0.0001,
                 fit_intercept: bool = True,
                 copy_X: bool = True,
                 max_iter: int = None,
                 tol: float = 1e-6,
                 n_jobs: int = None,
                 positive: bool = False,
                 warm_start: bool = False,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__(fit_intercept=fit_intercept,
                         copy_X=copy_X,
                         tol=tol,
                         n_jobs=n_jobs,
                         positive=positive,
                         device=device,
                         dtype=dtype,
                         *args, **kwargs)
        self.epsilon = epsilon
        self.alpha = alpha
        self.max_iter = max_iter
        self.warm_start = warm_start
        self.num_iter = None
        self.sigma_ = None
        self.outliers_mask_ = None

    @property
    def n_iter_(self):
        return self.num_iter

    def scale_(self):
        return self.sigma_

    @property
    def outliers_(self):
        return self.outliers_mask_

    def fit(self, data_or_X, y=None, **kwargs):
        # Determine target device and dtype from input data
        target_device = data_or_X.device if isinstance(data_or_X, torch.Tensor) else self.device
        target_dtype = data_or_X.dtype if isinstance(data_or_X, torch.Tensor) else self.dtype

        # Determine target device and dtype from input data
        
        

        self._fit_loss_fn = kwargs.get('loss_fn', None)
        try:
            # Determine target device and dtype from input data
            
            
            
            # 1. Preprocessing and joint parameter setup
            X_working = torch.as_tensor(data_or_X, device=target_device, dtype=target_dtype)
            if self.copy_X:
                X_working = X_working.clone()

            if y is None:
                raise ValueError("y cannot be None for fitting.")

            y = torch.as_tensor(y, device=target_device, dtype=target_dtype)
            sample_weight = kwargs.get("sample_weight", None)
            if sample_weight is not None:
                sample_weight = torch.as_tensor(sample_weight, device=target_device, dtype=target_dtype)
                if sample_weight.ndim == 1:
                    sample_weight = sample_weight.unsqueeze(1)

            if y.ndim == 1:
                y = y.unsqueeze(1)
                is_1d_y = True
            else:
                is_1d_y = False

            n_samples, n_features = X_working.shape
            n_targets = y.shape[1]

            # Initialize module parameters
            if self.weight is None or not self.warm_start:
                self._init_module(X_working, y)
                with torch.no_grad():
                    self.weight.zero_()
                    if self.bias is not None:
                        self.bias.zero_()

            # Scale parameter optimization (one per target)
            if self.sigma_ is None or not self.warm_start:
                self.sigma_ = torch.ones(n_targets, device=target_device, dtype=target_dtype, requires_grad=True)
            else:
                self.sigma_ = self.sigma_.detach().clone()
                self.sigma_.requires_grad_(True)

            params = [self.weight, self.sigma_]
            if self.bias is not None:
                params.append(self.bias)

            prev_threads = torch.get_num_threads()
            if self.n_jobs is not None:
                num_threads = self.n_jobs if self.n_jobs > 0 else torch.get_num_threads()
                torch.set_num_threads(num_threads)

            try:
                optimizer = torch.optim.LBFGS(params, lr=1, max_iter=self.max_iter or 100,
                                          tolerance_grad=self.tol, tolerance_change=self.tol,
                                          line_search_fn="strong_wolfe")

                def closure():
                    optimizer.zero_grad()
                    pred = F.linear(X_working, self.weight, self.bias)
                    custom_loss_fn = getattr(self, '_fit_loss_fn', None)
                    if custom_loss_fn is not None:
                        term3 = self.alpha * torch.sum(self.weight ** 2, dim=1)
                        objective = custom_loss_fn(y, pred) + torch.sum(term3)
                    else:
                        # Use absolute value to allow gradients to flow even if sigma < 0
                        si = torch.abs(self.sigma_) + 1e-6
                        residual = (y - pred) / si.view(1, -1)
                        abs_res = torch.abs(residual)
                        mask_linear = abs_res > self.epsilon
                        mask_quadratic = ~mask_linear
                        loss_vals = torch.zeros_like(abs_res)
                        loss_vals[mask_quadratic] = residual[mask_quadratic] ** 2
                        loss_vals[mask_linear] = 2 * self.epsilon * abs_res[mask_linear] - self.epsilon ** 2
                        if sample_weight is not None:
                            sw_sum = sample_weight.sum(dim=0)
                            term1 = sw_sum * si
                            term2 = si * torch.sum(sample_weight * loss_vals, dim=0)
                        else:
                            term1 = n_samples * si
                            term2 = si * torch.sum(loss_vals, dim=0)
                        term3 = self.alpha * torch.sum(self.weight ** 2, dim=1)
                        objective = torch.sum(term1 + term2 + term3)
                    objective.backward()
                    return objective

                optimizer.step(closure)

                with torch.no_grad():
                    self.sigma_.data.abs_().add_(1e-6)

                # Identify outliers
                with torch.no_grad():
                    pred_final = self.forward(X_working)
                    if is_1d_y:
                        pred_final = pred_final.unsqueeze(1)

                    abs_residuals = torch.abs(y - pred_final)
                    self.outliers_mask_ = abs_residuals / (self.sigma_.view(1, -1) + 1e-12) > self.epsilon
                    if is_1d_y:
                        self.outliers_mask_ = self.outliers_mask_.squeeze(1)

            finally:
                if self.n_jobs is not None:
                    torch.set_num_threads(prev_threads)

            self.fit_status = True


            # Update internal tracking


            self.device = str(target_device)


            self.dtype = target_dtype
            # Update internal tracking
            self.device = str(target_device)
            self.dtype = target_dtype

            # Update internal tracking

            self.device = str(target_device)

            self.dtype = target_dtype


            return self
        finally:
            self._fit_loss_fn = None




class OrthogonalMatchingPursuit(LinearRegression):
    def __init__(self,
                 n_nonzero_coefs: int = None,
                 tol: float = None,
                 fit_intercept: bool = True,
                 precompute: Union[str, bool] = "auto",
                 copy_X: bool = False,
                 positive: bool = False,
                 max_iter: int = None,
                 n_jobs: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__(fit_intercept=fit_intercept,
                         copy_X=copy_X,
                         tol=tol,
                         n_jobs=n_jobs,
                         positive=positive,
                         device=device,
                         dtype=dtype,
                         *args, **kwargs)
        self.n_nonzero_coefs = n_nonzero_coefs
        self.precompute = precompute
        self.max_iter = max_iter
        self.weight = None
        self.bias = None
        self.num_iter = None
        self.out_features = None
        self.in_features = None

    @property
    def n_iter_(self):
        return self.num_iter

    @property
    def n_nonzero_coefs_(self):
        if self.n_nonzero_coefs is None:
            if self.in_features is not None:
                return max(int(0.1 * self.in_features), 1)
            return 1
        return self.n_nonzero_coefs

    def _init_module(self, X: torch.Tensor, y: torch.Tensor):
        in_features = X.size(-1)
        out_features = y.size(-1)
        self.weight = nn.Parameter(
            torch.randn(
                (out_features, in_features), device=self.device, dtype=self.dtype
            )
        )
        if self.fit_intercept:
            self.bias = nn.Parameter(
                torch.zeros((out_features,), device=self.device, dtype=self.dtype)
            )
        self.in_features = in_features
        self.out_features = out_features
        return self

    def fit(self, data_or_X, y=None, **kwargs):
        # Determine target device and dtype from input data
        target_device = data_or_X.device if isinstance(data_or_X, torch.Tensor) else self.device
        target_dtype = data_or_X.dtype if isinstance(data_or_X, torch.Tensor) else self.dtype

        # Determine target device and dtype from input data
        
        
        
        # 1. Preprocessing
        X_working = torch.as_tensor(data_or_X, device=target_device, dtype=target_dtype)
        if self.copy_X:
            X_working = X_working.clone()

        if y is None:
            raise ValueError("y cannot be None for fitting.")

        y = torch.as_tensor(y, device=target_device, dtype=target_dtype)
        sample_weight = kwargs.get("sample_weight", None)
        if sample_weight is not None:
            sample_weight = torch.as_tensor(sample_weight, device=target_device, dtype=target_dtype)
            if sample_weight.ndim == 1:
                sample_weight = sample_weight.unsqueeze(1)

        if y.ndim == 1:
            y = y.unsqueeze(1)
            is_1d_y = True
        else:
            is_1d_y = False

        n_samples, n_features = X_working.shape
        n_targets = y.shape[1]

        # Initialize module
        self._init_module(X_working, y)
        with torch.no_grad():
            self.weight.zero_()
            if self.bias is not None:
                self.bias.zero_()

        # Handle Centering
        if self.fit_intercept:
            if sample_weight is not None:
                sw_sum = sample_weight.sum(dim=0, keepdim=True)
                X_mean = (sample_weight * X_working).sum(dim=0, keepdim=True) / sw_sum
                y_mean = (sample_weight * y).sum(dim=0, keepdim=True) / sw_sum
            else:
                X_mean = X_working.mean(dim=0, keepdim=True)
                y_mean = y.mean(dim=0, keepdim=True)
            X_working = X_working - X_mean
            y_active = y - y_mean
        else:
            y_active = y

        # Apply sample weight transformation
        if sample_weight is not None:
            sw_sqrt = torch.sqrt(sample_weight)
            X_working = sw_sqrt * X_working
            y_active = sw_sqrt * y_active

        max_features = self.n_nonzero_coefs_
        tol = self.tol if self.tol is not None else -1.0

        # Precompute
        if self.precompute == "auto":
            Gram = X_working.T @ X_working if n_samples > n_features else None
            Xty = X_working.T @ y_active
        elif self.precompute is True:
            Gram = X_working.T @ X_working
            Xty = X_working.T @ y_active
        elif isinstance(self.precompute, torch.Tensor):
            Gram = self.precompute.to(device=target_device, dtype=target_dtype)
            Xty = X_working.T @ y_active
        else:
            Gram = None
            Xty = None

        # Handle Parallelism (n_jobs)
        prev_threads = torch.get_num_threads()
        if self.n_jobs is not None:
            num_threads = self.n_jobs if self.n_jobs > 0 else torch.get_num_threads()
            torch.set_num_threads(num_threads)

        try:
            W_final = torch.zeros((n_targets, n_features), device=target_device, dtype=target_dtype)
            total_iters = 0

            # Precompute column norms for correlation normalization
            col_norms = torch.norm(X_working, dim=0)
            col_norms[col_norms == 1e-4] = 1.0  # Small epsilon instead of exactly 1.0?
            col_norms[col_norms == 0] = 1.0

            # Process n_nonzero_coefs into per-target limit
            if isinstance(max_features, (int, float)):
                limits = [int(max_features)] * n_targets
            elif isinstance(max_features, (list, tuple)):
                limits = [int(v) for v in max_features]
            elif isinstance(max_features, torch.Tensor):
                limits = max_features.long().tolist()
                if len(limits) == 1:
                    limits = limits * n_targets
            else:
                limits = [int(max_features)] * n_targets

            for target_idx in range(n_targets):
                yi = y_active[:, target_idx]
                residual = yi.clone()
                active_set = []

                # Iterations restricted by both n_nonzero_coefs and max_iter if provided
                limit = limits[target_idx]
                if self.max_iter is not None:
                    limit = min(limit, self.max_iter)

                w_S = None
                for it in range(limit):
                    # 1. Selection
                    if Gram is not None and Xty is not None:
                        # Correlation = X^T r = X^T (y - Xw) = X^T y - (X^T X) w
                        if w_S is None:
                            correlations = Xty[:, target_idx] / col_norms
                        else:
                            # Gram[:, active_set] is (n_features, len(active_set))
                            # w_S is (len(active_set),)
                            correlations = (Xty[:, target_idx] - Gram[:, active_set] @ w_S) / col_norms
                    else:
                        correlations = (X_working.T @ residual) / col_norms

                    if self.positive:
                        # Only consider features with positive correlation
                        mask = correlations > 0
                        if not torch.any(mask):
                            break
                        best_idx = torch.argmax(correlations * mask.float()).item()
                    else:
                        best_idx = torch.argmax(torch.abs(correlations)).item()

                    if best_idx in active_set:
                        break

                    # Check tolerance
                    if tol >= 0 and torch.norm(residual) ** 2 <= tol:
                        break

                    active_set.append(best_idx)

                    # 2. Orthogonal Projection
                    X_S = X_working[:, active_set]
                    try:
                        if self.positive:
                            # Use vectorized NNLS helper (requires y to be (n_samples, 1))
                            w_S = self._nnls(X_S, yi.unsqueeze(1)).squeeze(1)
                        else:
                            w_S = torch.linalg.lstsq(X_S, yi).solution
                    except RuntimeError:
                        active_set.pop()
                        break

                    # 3. Update Residual
                    residual = yi - X_S @ w_S

                if active_set:
                    W_final[target_idx, active_set] = w_S
                total_iters += len(active_set)

            with torch.no_grad():
                self.weight.copy_(W_final)
                if self.fit_intercept:
                    # (y_mean: 1xT, X_mean: 1xF, weight: TxF)
                    # X_mean @ weight.T is 1xT
                    bias_val = (y_mean - X_mean @ self.weight.T).squeeze(0)
                    if is_1d_y:
                        self.bias.copy_(bias_val.squeeze(-1))
                    else:
                        self.bias.copy_(bias_val)
                else:
                    if self.bias is not None:
                        self.bias.zero_()

            self.num_iter = total_iters // n_targets if n_targets > 0 else 0

        finally:
            torch.set_num_threads(prev_threads)

        self.fit_status = True


        # Update internal tracking


        self.device = str(target_device)


        self.dtype = target_dtype
        # Update internal tracking
        self.device = str(target_device)
        self.dtype = target_dtype

        # Update internal tracking

        self.device = str(target_device)

        self.dtype = target_dtype


        return self


class OrthogonalMatchingPursuitCV(MLRegressor):
    def __init__(self,
                 n_nonzero_coefs: Union[int, List[int], Tuple[int], torch.Tensor] = None,
                 tol: float = None,
                 fit_intercept: bool = True,
                 precompute: Union[str, bool] = "auto",
                 copy_X: bool = True,
                 cv: Union[int, str, Callable, Iterable, MLModule] = None,
                 cv_config: dict = None,
                 store_cv_results: bool = False,
                 verbose: Union[int, bool] = False,
                 scoring: Union[str, Callable, nn.Module] = None,
                 positive: bool = False,
                 max_iter: int = None,
                 n_jobs: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 alpha_per_target: bool = False,
                 *args, **kwargs):
        super().__init__()
        self.n_nonzero_coefs = n_nonzero_coefs
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.precompute = precompute
        self.copy_X = copy_X
        self.cv = cv
        self.cv_config = cv_config
        self.store_cv_results = store_cv_results
        self.verbose = verbose
        self.scoring = scoring
        self.positive = positive
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype
        self.alpha_per_target = alpha_per_target

        # Process n_nonzero_coefs into a search grid
        if isinstance(n_nonzero_coefs, int):
            self.n_nonzero_coefs_list = [n_nonzero_coefs]
        elif isinstance(n_nonzero_coefs, (list, tuple)):
            self.n_nonzero_coefs_list = list(n_nonzero_coefs)
        elif isinstance(n_nonzero_coefs, torch.Tensor):
            self.n_nonzero_coefs_list = n_nonzero_coefs.tolist()
        else:
            self.n_nonzero_coefs_list = None  # To be generated in fit

        # Initialize base OMP for cloning
        self.ortho = OrthogonalMatchingPursuit(
            n_nonzero_coefs=self.n_nonzero_coefs_list[0] if self.n_nonzero_coefs_list else None,
            tol=tol,
            fit_intercept=fit_intercept,
            precompute=precompute,
            copy_X=copy_X,
            positive=positive,
            max_iter=max_iter,
            n_jobs=n_jobs,
            device=device,
            dtype=dtype,
            *args, **kwargs
        )

        # Search Setup
        self.params_dict = {
            "n_nonzero_coefs": self.n_nonzero_coefs_list if self.n_nonzero_coefs_list else [1],
            "fit_intercept": [fit_intercept],
            "positive": [positive]
        }

        cv_config = cv_config if cv_config is not None else {}
        if cv is None:
            from ...cross_validation.splitters import KFoldCV
            cv = KFoldCV(n_splits=5, **cv_config)

        self.search_params = {
            "estimator": self.ortho,
            "param_grid": self.params_dict,
            "scoring": scoring,
            "n_jobs": n_jobs,
            "cv": cv,
            "cv_config": cv_config,
            "verbose": verbose,
            "return_train_score": True,
            "store_cv_values": store_cv_results,
            "device": device,
            "dtype": dtype
        }

        from ...cross_validation.search_cv import GridSearchCV
        self.search = GridSearchCV(**self.search_params)

        self.estimator = None
        self.fit_status = False
        self._n_nonzero_coefs_ = None
        self._n_features_in_ = None

    def fit(self, data_or_X, y=None, **kwargs):
        # Determine target device and dtype from input data
        target_device = data_or_X.device if isinstance(data_or_X, torch.Tensor) else self.device
        target_dtype = data_or_X.dtype if isinstance(data_or_X, torch.Tensor) else self.dtype

        # Determine target device and dtype from input data
        
        

        try:
            X = torch.as_tensor(data_or_X, device=target_device, dtype=target_dtype)
            y = torch.as_tensor(y, device=target_device, dtype=target_dtype)

            if y.ndim == 1:
                y = y.unsqueeze(1)

            n_samples, n_features = X.shape
            self._n_features_in_ = n_features

            # n_nonzero_coefs generation logic
            if self.n_nonzero_coefs_list is None:
                limit = min(n_features, int(0.1 * n_samples))
                limit = max(limit, 5) if n_features >= 5 else n_features
                self.n_nonzero_coefs_list = list(range(1, limit + 1))

            self.params_dict["n_nonzero_coefs"] = self.n_nonzero_coefs_list
            self.search.param_grid = self.params_dict

            if y.shape[1] > 1 and self.alpha_per_target:
                from ...cross_validation.search_cv import GridSearchCV
                n_targets = y.shape[1]
                best_n_nonzero = []

                for i in range(n_targets):
                    y_target = y[:, i:i + 1]
                    search = GridSearchCV(**self.search_params)
                    search.param_grid = self.params_dict
                    search.fit(X, y_target, **kwargs)
                    best_n_nonzero.append(search.best_params_['n_nonzero_coefs'])

                self._n_nonzero_coefs_ = best_n_nonzero

                # Final Refit
                self.estimator = OrthogonalMatchingPursuit(
                    n_nonzero_coefs=best_n_nonzero,
                    tol=self.tol,
                    fit_intercept=self.fit_intercept,
                    precompute=self.precompute,
                    copy_X=self.copy_X,
                    positive=self.positive,
                    max_iter=self.max_iter,
                    n_jobs=self.n_jobs,
                    device=target_device,
                    dtype=target_dtype
                )
                self.estimator.fit(X, y, **kwargs)
            else:
                self.search.fit(X, y, **kwargs)
                self.estimator = self.search.best_estimator_
                self._n_nonzero_coefs_ = self.search.best_params_['n_nonzero_coefs']

            self.fit_status = True


            # Update internal tracking


            self.device = str(target_device)


            self.dtype = target_dtype
            # Update internal tracking
            self.device = str(target_device)
            self.dtype = target_dtype
            # Update internal tracking
            self.device = str(target_device)
            self.dtype = target_dtype
            return self

        except Exception as e:
            print(f"DEBUG EXCEPTION IN OrthogonalMatchingPursuitCV.fit: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def forward(self, X: torch.Tensor, y: torch.Tensor = None):
        if not self.fit_status:
            if y is not None:
                self.fit(X, y)
            else:
                raise RuntimeError("Model not fitted")
        return self.estimator(X)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if self.estimator is None:
            raise ValueError("OrthogonalMatchingPursuitCV must be fitted before predict().")
        return self.estimator.predict(X)

    @property
    def intercept_(self):
        return self.estimator.intercept_

    @property
    def coef_(self):
        return self.estimator.coef_

    @property
    def n_nonzero_coefs_(self):
        return self._n_nonzero_coefs_

    @property
    def n_iter_(self):
        return self.estimator.n_iter_

    @property
    def n_features_in_(self):
        return self._n_features_in_


class TheilSenRegressor(LinearRegression):
    def __init__(self,
                 fit_intercept: bool = True,
                 max_subpopulation: int = 1e4,
                 n_subsamples: int = None,
                 tol: float = None,
                 copy_X: bool = False,
                 positive: bool = False,
                 max_iter: int = None,
                 random_state: int = None,
                 n_jobs: int = None,
                 verbose: bool = False,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__(fit_intercept=fit_intercept,
                         copy_X=copy_X,
                         tol=tol,
                         n_jobs=n_jobs,
                         positive=positive,
                         device=device,
                         dtype=dtype,
                         *args, **kwargs)
        self.max_subpopulation = max_subpopulation
        self.n_subsamples = n_subsamples
        self.max_iter = max_iter if max_iter is not None else 1000
        self.random_state = random_state
        self.verbose = verbose
        self.num_iter = None
        self.breakdown_point = None
        self.n_subpopulation_value = None

    @property
    def n_iter_(self):
        return self.num_iter

    @property
    def breakdown_(self):
        return self.breakdown_point

    @property
    def n_subpopulation_(self):
        return self.n_subpopulation_value

    def fit(self, data_or_X, y=None, **kwargs):
        # Determine target device and dtype from input data
        target_device = data_or_X.device if isinstance(data_or_X, torch.Tensor) else self.device
        target_dtype = data_or_X.dtype if isinstance(data_or_X, torch.Tensor) else self.dtype

        # Determine target device and dtype from input data
        
        
        
        # 1. Preprocessing
        X_working = torch.as_tensor(data_or_X, device=target_device, dtype=target_dtype)
        if self.copy_X:
            X_working = X_working.clone()

        if y is None:
            raise ValueError("y cannot be None for fitting.")

        y = torch.as_tensor(y, device=target_device, dtype=target_dtype)
        if y.ndim == 1:
            y = y.unsqueeze(1)
            is_1d_y = True
        else:
            is_1d_y = False

        n_samples, n_features = X_working.shape
        n_targets = y.shape[1]

        # Determine n_subsamples
        if self.n_subsamples is None:
            n_subsamples = n_features + 1 if self.fit_intercept else n_features
        else:
            n_subsamples = self.n_subsamples

        if n_subsamples < n_features:
            raise ValueError(f"n_subsamples {n_subsamples} cannot be less than n_features {n_features}.")
        if n_subsamples > n_samples:
            n_subsamples = n_samples

        # Calculate breakdown point: 1 - (0.5)^(1/n_subsamples)
        self.breakdown_point = 1 - (0.5) ** (1 / n_subsamples)

        # 2. Subsampling
        try:
            n_combinations = math.comb(n_samples, n_subsamples)
        except OverflowError:
            n_combinations = float('inf')

        n_subpopulation = min(int(self.max_subpopulation), n_combinations)
        self.n_subpopulation_value = n_subpopulation

        if self.verbose:
            print(f"TheilSenRegressor: Sampling {n_subpopulation} subsets from {n_combinations} combinations.")

        # Handle Parallelism (n_jobs)
        prev_threads = torch.get_num_threads()
        if self.n_jobs is not None:
            num_threads = self.n_jobs if self.n_jobs > 0 else torch.get_num_threads()
            torch.set_num_threads(num_threads)

        try:
            # Generate indices for subsampling using torch
            g = torch.Generator(device='cpu')  # Generator is always CPU-based for seeds
            if self.random_state is not None:
                g.manual_seed(self.random_state)

            # Collect solutions
            all_params = []

            for i in range(n_subpopulation):
                # Sample indices without replacement using torch.randperm
                indices = torch.randperm(n_samples, generator=g)[:n_subsamples].to(target_device)
                X_subset = X_working[indices]
                y_subset = y[indices]

                if self.fit_intercept:
                    X_subset_const = torch.cat(
                        [X_subset, torch.ones((n_subsamples, 1), device=target_device, dtype=target_dtype)], dim=1)
                else:
                    X_subset_const = X_subset

                try:
                    if self.positive:
                        # Use vectorized NNLS if positive constraint is requested
                        params = self._nnls(X_subset_const, y_subset)
                    else:
                        params = torch.linalg.lstsq(X_subset_const, y_subset).solution
                    all_params.append(params)
                except RuntimeError:
                    continue

            if not all_params:
                raise RuntimeError("No valid OLS solutions found during subsampling.")

            all_params = torch.stack(all_params)  # Shape: (samples, n_params, targets)

            # 3. Spatial Median (Weiszfeld's algorithm)
            W_final = torch.zeros((n_targets, n_features), device=target_device, dtype=target_dtype)
            if self.fit_intercept:
                b_final = torch.zeros((n_targets,), device=target_device, dtype=target_dtype)

            total_iters = 0
            for t in range(n_targets):
                params_t = all_params[:, :, t]
                current_median = torch.median(params_t, dim=0).values

                for i in range(self.max_iter):
                    diffs = params_t - current_median
                    norms = torch.norm(diffs, dim=1, keepdim=True)
                    norms[norms < 1e-9] = 1e-9
                    weights = 1.0 / norms
                    new_median = torch.sum(params_t * weights, dim=0) / torch.sum(weights)

                    if torch.norm(new_median - current_median) < (self.tol if self.tol else 1e-6):
                        current_median = new_median
                        break
                    current_median = new_median

                total_iters = max(total_iters, i + 1)

                if self.fit_intercept:
                    W_final[t] = current_median[:-1]
                    b_final[t] = current_median[-1]
                else:
                    W_final[t] = current_median

            self.num_iter = total_iters

            # 4. Final parameters
            self._init_module(X_working, y)
            with torch.no_grad():
                self.weight.copy_(W_final)
                if self.fit_intercept:
                    self.bias.copy_(b_final)
        finally:
            torch.set_num_threads(prev_threads)

        self.fit_status = True


        # Update internal tracking


        self.device = str(target_device)


        self.dtype = target_dtype
        # Update internal tracking
        self.device = str(target_device)
        self.dtype = target_dtype

        # Update internal tracking

        self.device = str(target_device)

        self.dtype = target_dtype


        return self


class RANSACRegressor(LinearRegression):
    def __init__(self,
                 fit_intercept: bool = True,
                 estimator: Union[Callable, object] = None,
                 min_samples: int = None,
                 residual_threshold: float = None,
                 is_data_valid: Callable = None,
                 is_model_valid: Callable = None,
                 tol: float = None,
                 copy_X: bool = False,
                 positive: bool = False,
                 max_trials: int = None,
                 max_skips: int = None,
                 stop_n_inliers: float = float("inf"),
                 stop_score: float = float("inf"),
                 stop_probability: float = 0.99,
                 loss: Union[str, Callable, nn.Module, MLModule] = "absolute_error",
                 random_state: int = None,
                 n_jobs: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__(fit_intercept=fit_intercept,
                         copy_X=copy_X,
                         tol=tol,
                         n_jobs=n_jobs,
                         positive=positive,
                         device=device,
                         dtype=dtype,
                         *args, **kwargs)
        self.estimator = estimator
        self.min_samples = min_samples
        self.residual_threshold = residual_threshold
        self.is_data_valid = is_data_valid
        self.is_model_valid = is_model_valid
        self.max_trials = max_trials
        self.max_skips = max_skips
        self.stop_n_inliers = stop_n_inliers
        self.stop_score = stop_score
        self.stop_probability = stop_probability
        self.loss = loss
        self.random_state = random_state
        self.num_trials = None
        self._inlier_mask = None
        self._n_skips_no_inliers = None
        self._n_skips_invalid_data = None
        self._n_skips_invalid_model = None

    @property
    def estimator_(self):
        return self.estimator

    @property
    def n_trials_(self):
        return self.num_trials

    @property
    def inlier_mask_(self):
        return self._inlier_mask

    @property
    def n_skips_no_inliers_(self):
        return self._n_skips_no_inliers

    @property
    def n_skips_invalid_data_(self):
        return self._n_skips_invalid_data

    @property
    def n_skips_invalid_model(self):
        return self._n_skips_invalid_model

    def fit(self, data_or_X, y=None, **kwargs):
        # Determine target device and dtype from input data
        target_device = data_or_X.device if isinstance(data_or_X, torch.Tensor) else self.device
        target_dtype = data_or_X.dtype if isinstance(data_or_X, torch.Tensor) else self.dtype

        # Determine target device and dtype from input data
        
        
        
        # 1. Preprocessing
        X_working = torch.as_tensor(data_or_X, device=target_device, dtype=target_dtype)
        if self.copy_X:
            X_working = X_working.clone()

        if y is None:
            raise ValueError("y cannot be None for fitting.")

        y = torch.as_tensor(y, device=target_device, dtype=target_dtype)
        sample_weight = kwargs.get("sample_weight", None)
        if sample_weight is not None:
            sample_weight = torch.as_tensor(sample_weight, device=target_device, dtype=target_dtype)
            if sample_weight.ndim == 1:
                sample_weight = sample_weight.unsqueeze(1)

        is_1d_y = False
        if y.ndim == 1:
            y = y.unsqueeze(1)
            is_1d_y = True

        n_samples, n_features = X_working.shape
        n_targets = y.shape[1]

        # 2. Initialization of parameters
        if self.estimator is None:
            # Use Local LinearRegression as default
            base_estimator = LinearRegression(
                fit_intercept=self.fit_intercept,
                copy_X=self.copy_X,
                tol=self.tol if self.tol is not None else 1e-6,
                n_jobs=self.n_jobs,
                positive=self.positive,
                device=target_device,
                dtype=target_dtype
            )
        else:
            base_estimator = self.estimator

        if self.min_samples is None:
            min_samples = n_features + (1 if self.fit_intercept else 0)
        elif self.min_samples < 1:
            min_samples = int(math.ceil(self.min_samples * n_samples))
        else:
            min_samples = int(self.min_samples)

        if min_samples > n_samples:
            raise ValueError(f"min_samples {min_samples} is greater than the number of samples {n_samples}.")

        if self.residual_threshold is None:
            # Default to MAD of y
            y_median = torch.median(y, dim=0).values
            residual_threshold = torch.median(torch.abs(y - y_median)).item()
            if residual_threshold == 0:
                residual_threshold = 1e-6
        else:
            residual_threshold = self.residual_threshold

        max_trials = self.max_trials if self.max_trials is not None else 100
        max_skips = self.max_skips if self.max_skips is not None else float('inf')

        # Loss function setup
        if isinstance(self.loss, str):
            if self.loss == "absolute_error":
                def loss_fn(y_true, y_pred):
                    return torch.abs(y_true - y_pred).sum(dim=1)
            elif self.loss == "squared_error":
                def loss_fn(y_true, y_pred):
                    return ((y_true - y_pred) ** 2).sum(dim=1)
            else:
                raise ValueError(f"Unknown loss function string: {self.loss}")
        elif callable(self.loss):
            loss_fn = self.loss
        else:
            raise ValueError("Loss must be 'absolute_error', 'squared_error', or a callable.")

        # 3. Iterative RANSAC loop using torch for randomness
        g = torch.Generator(device='cpu')
        if self.random_state is not None:
            g.manual_seed(self.random_state)
        best_model_state = None
        best_inlier_mask = None
        best_inlier_count = -1
        best_score = -float('inf')

        num_trials = 0
        num_skips_invalid_data = 0
        num_skips_invalid_model = 0
        num_skips_no_inliers = 0

        # Handle Parallelism (n_jobs)
        prev_threads = torch.get_num_threads()
        if self.n_jobs is not None:
            num_threads = self.n_jobs if self.n_jobs > 0 else torch.get_num_threads()
            torch.set_num_threads(num_threads)

        try:
            for trial in range(max_trials):
                num_trials += 1

                # Sample min_samples using torch.randperm
                indices = torch.randperm(n_samples, generator=g)[:min_samples].to(target_device)
                X_subset = X_working[indices]
                y_subset = y[indices]
                sw_subset = sample_weight[indices] if sample_weight is not None else None

                # Validate data
                if self.is_data_valid is not None:
                    if not self.is_data_valid(X_subset, y_subset):
                        num_skips_invalid_data += 1
                        if (num_skips_invalid_data + num_skips_invalid_model + num_skips_no_inliers) >= max_skips:
                            break
                        continue

                # Fit model on subset
                try:
                    if sw_subset is not None:
                        current_model = base_estimator.fit(X_subset, y_subset, sample_weight=sw_subset)
                    else:
                        current_model = base_estimator.fit(X_subset, y_subset)
                except Exception:
                    num_skips_invalid_model += 1
                    if (num_skips_invalid_data + num_skips_invalid_model + num_skips_no_inliers) >= max_skips:
                        break
                    continue

                # Validate model
                if self.is_model_valid is not None:
                    if not self.is_model_valid(current_model, X_subset, y_subset):
                        num_skips_invalid_model += 1
                        if (num_skips_invalid_data + num_skips_invalid_model + num_skips_no_inliers) >= max_skips:
                            break
                        continue

                # Compute consensus set (inliers)
                y_pred = current_model.predict(X_working)
                if y_pred.ndim == 1:
                    y_pred = y_pred.unsqueeze(1)

                residuals = loss_fn(y, y_pred)
                inlier_mask = residuals <= residual_threshold
                inlier_count = torch.sum(inlier_mask).item()

                if inlier_count == 0:
                    num_skips_no_inliers += 1
                    if (num_skips_invalid_data + num_skips_invalid_model + num_skips_no_inliers) >= max_skips:
                        break
                    continue

                # Score consensus set
                try:
                    # Some estimators might not have score()
                    score = current_model.score(X_working[inlier_mask], y[inlier_mask])
                except Exception:
                    score = inlier_count

                # Keep the best model
                if (inlier_count > best_inlier_count) or (inlier_count == best_inlier_count and score > best_score):
                    best_inlier_count = inlier_count
                    best_score = score
                    best_inlier_mask = inlier_mask
                    # Save best state
                    best_model_state = {
                        'weight': current_model.weight.clone() if hasattr(current_model, 'weight') else None,
                        'bias': current_model.bias.clone() if hasattr(current_model,
                                                                      'bias') and current_model.bias is not None else None
                    }

                # Check stopping criteria
                if best_inlier_count >= self.stop_n_inliers or best_score >= self.stop_score:
                    break

                # Adaptive number of trials
                if self.stop_probability < 1.0:
                    epsilon = 1.0 - (best_inlier_count / n_samples)
                    if epsilon < 1.0:
                        try:
                            prob_all_inliers = (1.0 - epsilon) ** min_samples
                            if prob_all_inliers > 0:
                                div = math.log(1.0 - prob_all_inliers)
                                num_trials_needed = math.log(1.0 - self.stop_probability) / div
                                if trial >= num_trials_needed:
                                    break
                        except (ValueError, ZeroDivisionError):
                            pass

            if best_inlier_mask is None:
                raise RuntimeError("RANSAC could not find a valid consensus set.")

            # 4. Final Refinement on all inliers
            X_inliers = X_working[best_inlier_mask]
            y_inliers = y[best_inlier_mask]
            sw_inliers = sample_weight[best_inlier_mask] if sample_weight is not None else None

            # Fit final model
            if sw_inliers is not None:
                self.estimator = base_estimator.fit(X_inliers, y_inliers, sample_weight=sw_inliers)
            else:
                self.estimator = base_estimator.fit(X_inliers, y_inliers)

            # Populate attributes
            self._init_module(X_working, y)
            with torch.no_grad():
                self.weight.copy_(self.estimator.weight)
                if self.fit_intercept and self.estimator.bias is not None:
                    self.bias.copy_(self.estimator.bias)
                elif self.bias is not None:
                    self.bias.zero_()

            self.num_trials = num_trials
            self._inlier_mask = best_inlier_mask
            self._n_skips_no_inliers = num_skips_no_inliers
            self._n_skips_invalid_data = num_skips_invalid_data
            self._n_skips_invalid_model = num_skips_invalid_model
        finally:
            torch.set_num_threads(prev_threads)

        self.fit_status = True


        # Update internal tracking


        self.device = str(target_device)


        self.dtype = target_dtype
        # Update internal tracking
        self.device = str(target_device)
        self.dtype = target_dtype

        # Update internal tracking

        self.device = str(target_device)

        self.dtype = target_dtype


        return self


class QuantileRegressor(LinearRegression):
    """
    Linear regression model that predicts conditional quantiles.

    The linear QuantileRegressor optimizes the pinball loss for a desired quantile and is robust to outliers.

    This model uses an L1 regularization like Lasso.
    """

    def __init__(self,
                 fit_intercept: bool = True,
                 quantile: float = 0.5,
                 # The quantile that the model tries to predict. It must be strictly between 0 and 1. If 0.5 (default), the model predicts the 50% quantile, i.e. the median.
                 alpha: float = 1.0,  # Regularization constant that multiplies the L1 penalty term.
                 tol: float = None,
                 copy_X: bool = False,
                 positive: bool = False,
                 solver: str = "admm",
                 # {‘admm’, ‘highs-ds’, ‘highs-ipm’, ‘highs’, ‘interior-point’, ‘revised simplex’}
                 solver_options: dict = None,
                 # If None and if solver='interior-point', then {"lstsq": True} is passed to for the sake of stability
                 max_iter: int = 1000,
                 n_jobs: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__(fit_intercept=fit_intercept,
                         copy_X=copy_X,
                         tol=tol,
                         n_jobs=n_jobs,
                         positive=positive,
                         device=device,
                         dtype=dtype,
                         *args, **kwargs)
        self.quantile = max(0.0, min(1.0, abs(quantile)))
        self.alpha = alpha
        self.solver = solver
        self.solver_options = solver_options
        self.max_iter = max_iter
        self.num_iter = None

    @property
    def n_iter_(self):
        return self.num_iter

    def _solve_admm(self, X, y, quantile, alpha, max_iter, tol, sample_weight=None, rho=1.0, lmbda=1.0):
        """
        Solves min sum w_i * rho_tau(y - Xw - b) + alpha * ||w||_1 using ADMM.
        Vectorized for multi-target y.
        """
        # Use double precision internally for stability
        original_dtype = X.dtype
        X_d = X.to(dtype=torch.float64)
        y_d = y.to(dtype=torch.float64)
        sw_d = sample_weight.to(dtype=torch.float64) if sample_weight is not None else None

        n_samples, n_features = X_d.shape
        n_targets = y_d.shape[1]

        # Augmented system for (w, b)
        ones = torch.ones((n_samples, 1), device=X_d.device, dtype=torch.float64)
        A = torch.cat([X_d, ones], dim=1)

        AtA = A.T @ A
        GtG = torch.zeros((n_features + 1, n_features + 1), device=X_d.device, dtype=torch.float64)
        GtG[:n_features, :n_features] = torch.eye(n_features, device=X_d.device, dtype=torch.float64)

        M = rho * AtA + lmbda * GtG

        # Variables
        try:
            theta = torch.linalg.solve(AtA + 1e-6 * torch.eye(n_features + 1, device=X_d.device, dtype=torch.float64),
                                       A.T @ y_d)
        except Exception:
            theta = torch.zeros((n_features + 1, n_targets), device=X_d.device, dtype=torch.float64)

        z = torch.zeros((n_samples, n_targets), device=X_d.device, dtype=torch.float64)
        v = theta[:n_features, :].clone()
        if self.positive:
            v = torch.clamp(v, min=0)

        u = torch.zeros((n_samples, n_targets), device=X_d.device, dtype=torch.float64)
        a = torch.zeros((n_features, n_targets), device=X_d.device, dtype=torch.float64)

        for i in range(max_iter):
            theta_old = theta.clone()

            # 1. Update (w, b)
            rhs = rho * A.T @ (y_d - z - u)
            rhs[:n_features, :] += lmbda * (v - a)

            try:
                theta = torch.linalg.solve(M, rhs)
            except Exception:
                theta = torch.linalg.inv(M) @ rhs

            w = theta[:n_features, :]
            b = theta[n_features:, :]

            # 2. Update z (residual)
            target_z = y_d - X_d @ w - b - u
            if sw_d is not None:
                z = torch.where(target_z > (sw_d * quantile) / rho, target_z - (sw_d * quantile) / rho,
                                torch.where(target_z < (sw_d * (quantile - 1.0)) / rho, target_z - (sw_d * (quantile - 1.0)) / rho,
                                            torch.zeros_like(target_z)))
            else:
                z = torch.where(target_z > quantile / rho, target_z - quantile / rho,
                                torch.where(target_z < (quantile - 1.0) / rho, target_z - (quantile - 1.0) / rho,
                                            torch.zeros_like(target_z)))

            # 3. Update v (penalty)
            target_v = w + a
            threshold = alpha / lmbda
            v = torch.sign(target_v) * torch.clamp(torch.abs(target_v) - threshold, min=0)

            if self.positive:
                v = torch.clamp(v, min=0)

            # 4. Dual updates
            u = u + (X_d @ w + b + z - y_d)
            a = a + (w - v)

            # Convergence check
            if torch.max(torch.abs(theta - theta_old)) < tol:
                break

        return w.to(dtype=original_dtype), b.to(dtype=original_dtype), i + 1

    def _solve_linprog(self, X, y, quantile, alpha, max_iter, tol, sample_weight=None):
        """
        Solves the Quantile Regression problem using scipy.optimize.linprog.
        Formulated as a Linear Program (LP).
        Loops over targets because linprog is not vectorized for multi-output.
        """
        if opt is None:
            raise ImportError(
                "SciPy is required for 'highs', 'interior-point', or 'simplex' solvers. Please install scipy.")

        X_np = X.cpu().numpy().astype('float64')
        y_np = y.cpu().numpy().astype('float64')
        n_samples, n_features = X_np.shape
        n_targets = y_np.shape[1]

        W_final = torch.zeros((n_targets, n_features), device=X.device, dtype=X.dtype)
        b_final = torch.zeros(n_targets, device=X.device, dtype=X.dtype)

        # Reformulate as LP: min c^T z s.t. A_eq z = b_eq, z >= 0
        # Variables z:
        # If not positive: [w+, w-, b+, b-, u+, u-] -> 2*F + 2 + 2*N
        # If positive:     [w, b+, b-, u+, u-]     -> F + 2 + 2*N

        if self.positive:
            n_vars = n_features + 2 + 2 * n_samples
            c = torch.zeros(n_vars, device=X.device, dtype=torch.float64)
            c[:n_features] = alpha
            if sample_weight is not None:
                sw_flat = sample_weight.to(torch.float64).reshape(-1)
                c[n_features + 2: n_features + 2 + n_samples] = quantile * sw_flat
                c[n_features + 2 + n_samples:] = (1 - quantile) * sw_flat
            else:
                c[n_features + 2: n_features + 2 + n_samples] = quantile
                c[n_features + 2 + n_samples:] = 1 - quantile
        else:
            n_vars = 2 * n_features + 2 + 2 * n_samples
            c = torch.zeros(n_vars, device=X.device, dtype=torch.float64)
            c[:2 * n_features] = alpha
            if sample_weight is not None:
                sw_flat = sample_weight.to(torch.float64).reshape(-1)
                c[2 * n_features + 2: 2 * n_features + 2 + n_samples] = quantile * sw_flat
                c[2 * n_features + 2 + n_samples:] = (1 - quantile) * sw_flat
            else:
                c[2 * n_features + 2: 2 * n_features + 2 + n_samples] = quantile
                c[2 * n_features + 2 + n_samples:] = 1 - quantile

        # Convert c to numpy once for all targets
        c_np = c.cpu().numpy()

        for t in range(n_targets):
            y_t = y_np[:, t]

            A_eq = torch.zeros((n_samples, n_vars), device=X.device, dtype=torch.float64)
            if self.positive:
                A_eq[:, :n_features] = X.to(torch.float64)
                offset = n_features
            else:
                A_eq[:, :n_features] = X.to(torch.float64)
                A_eq[:, n_features:2 * n_features] = -X.to(torch.float64)
                offset = 2 * n_features

            if self.fit_intercept:
                A_eq[:, offset] = 1.0
                A_eq[:, offset + 1] = -1.0

            A_eq[:, offset + 2: offset + 2 + n_samples] = torch.eye(n_samples, device=X.device, dtype=torch.float64)
            A_eq[:, offset + 2 + n_samples:] = -torch.eye(n_samples, device=X.device, dtype=torch.float64)

            # Map solver name to linprog method
            method = self.solver

            res = opt.linprog(c_np, A_eq=A_eq.cpu().numpy(), b_eq=y_t, method=method, options=self.solver_options)

            if not res.success:
                warnings.warn(f"QuantileRegressor solver '{self.solver}' failed for target {t}: {res.message}")

            z = res.x
            if self.positive:
                w = z[:n_features]
                b = z[n_features] - z[n_features + 1]
            else:
                w = z[:n_features] - z[n_features:2 * n_features]
                b = z[2 * n_features] - z[2 * n_features + 1]

            W_final[t] = torch.from_numpy(w).to(device=X.device, dtype=X.dtype)
            b_final[t] = torch.tensor(b, device=X.device, dtype=X.dtype) if self.fit_intercept else torch.tensor(0.0,
                                                                                                                 device=X.device,
                                                                                                                 dtype=X.dtype)

        # Return in format expected by fit: W_final (n_features, n_targets), b_final (n_targets, 1)
        return W_final.T, b_final.unsqueeze(1), 0

    def fit(self, data_or_X, y=None, **kwargs):
        # Determine target device and dtype from input data
        target_device = data_or_X.device if isinstance(data_or_X, torch.Tensor) else self.device
        target_dtype = data_or_X.dtype if isinstance(data_or_X, torch.Tensor) else self.dtype

        # Determine target device and dtype from input data
        
        
        
        # 1. Preprocessing
        X_working = torch.as_tensor(data_or_X, device=target_device, dtype=target_dtype)
        if self.copy_X:
            X_working = X_working.clone()

        if y is None:
            raise ValueError("y cannot be None for fitting.")

        y = torch.as_tensor(y, device=target_device, dtype=target_dtype)
        sample_weight = kwargs.get("sample_weight", None)
        if sample_weight is not None:
            sample_weight = torch.as_tensor(sample_weight, device=target_device, dtype=target_dtype)
            if sample_weight.ndim == 1:
                sample_weight = sample_weight.unsqueeze(1)

        if y.ndim == 1:
            y = y.unsqueeze(1)
            is_1d_y = True
        else:
            is_1d_y = False

        n_samples, n_features = X_working.shape
        n_targets = y.shape[1]

        # 2. Solver Parameters
        max_iter = self.max_iter if self.max_iter is not None else 1000
        tol = self.tol if self.tol is not None else 1e-4

        # 3. Handle Parallelism (n_jobs)
        prev_threads = torch.get_num_threads()
        if self.n_jobs is not None:
            num_threads = self.n_jobs if self.n_jobs > 0 else torch.get_num_threads()
            torch.set_num_threads(num_threads)

        try:
            # Dispatch to appropriate solver
            if self.solver == "admm":
                W_final, b_final, iters = self._solve_admm(
                    X_working, y,
                    quantile=self.quantile,
                    alpha=self.alpha,
                    max_iter=max_iter,
                    tol=tol,
                    sample_weight=sample_weight
                )
            else:
                # Use SciPy linprog solvers
                W_final, b_final, iters = self._solve_linprog(
                    X_working, y,
                    quantile=self.quantile,
                    alpha=self.alpha,
                    max_iter=max_iter,
                    tol=tol,
                    sample_weight=sample_weight
                )

            self.num_iter = iters

            # Initialize module and copy parameters
            self._init_module(X_working, y)
            with torch.no_grad():
                self.weight.copy_(W_final.T)
                if self.fit_intercept:
                    self.bias.copy_(b_final.reshape(-1))
                elif self.bias is not None:
                    self.bias.zero_()

        finally:
            torch.set_num_threads(prev_threads)

        self.fit_status = True


        # Update internal tracking


        self.device = str(target_device)


        self.dtype = target_dtype
        # Update internal tracking
        self.device = str(target_device)
        self.dtype = target_dtype

        # Update internal tracking

        self.device = str(target_device)

        self.dtype = target_dtype


        return self


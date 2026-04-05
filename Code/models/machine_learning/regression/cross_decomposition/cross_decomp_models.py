import torch
import torch.nn as nn
import torch.nn.functional as F
from .....models.utils import MLRegressor
from typing import Optional, Callable, Union, Any, List, Tuple, Dict, Literal
from torch.func import vmap
import joblib

__all__ = ["CCA", "PLSCanonical", "PLSRegression", "PLSSVD"]

class CCA(MLRegressor):
    def __init__(self,
                 n_components: int = 2,
                 scale: bool = True,
                 max_iter: int = 500,
                 tol: float = 1e-6,
                 copy: bool = True,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.n_components = n_components
        self.scale = scale
        self.max_iter = max_iter
        self.tol = tol
        self.copy = copy
        self.device = device
        self.dtype = dtype
        self.in_features = None
        self.out_features = None
        self._weights_x = None
        self._weights_y = None
        self.weights = None
        self.bias = None
        self.num_iter = None

    def _init_module_(self, X, y):
        in_features = X.size(-1)
        out_features = y.size(-1)
        self.in_features = in_features
        self.out_features = out_features
        n_components = min(self.n_components, in_features, out_features)
        self.n_components = n_components
        self._weights_x = nn.Parameter(
            torch.randn((in_features, n_components), device=self.device, dtype=self.dtype) * 0.1
        )
        self._weights_y = nn.Parameter(
            torch.randn((out_features, n_components), device=self.device, dtype=self.dtype) * 0.1
        )
        self.weights = None
        self.bias = nn.Parameter(
            torch.zeros((out_features,), device=self.device, dtype=self.dtype)
        )
        return self

    def cov(self, X: torch.Tensor, Y: torch.Tensor):
        X_centered = X - X.mean(dim=-2, keepdim=True)
        Y_centered = Y - Y.mean(dim=-2, keepdim=True)
        cov_mat = (X_centered.T @ Y_centered) / (X.size(0) - 1)
        return cov_mat

    def _calc_svd(self, X, y):
        cov_mat = self.cov(X, y)
        return torch.linalg.svd(cov_mat)

    @property
    def x_weights_(self):
        if self._weights_x is not None:
            return self._weights_x.detach()
        return None

    @property
    def y_weights_(self):
        if self._weights_y is not None:
            return self._weights_y.detach()
        return None

    @property
    def x_loadings_(self):
        return self.x_weights_

    @property
    def y_loadings_(self):
        return self.y_weights_

    @property
    def x_rotations_(self):
        return self.x_weights_

    @property
    def y_rotations_(self):
        return self.y_weights_

    @property
    def coef_(self):
        if self.weights is not None:
            return self.weights.detach()
        if self._weights_x is not None and self._weights_y is not None:
            return self._weights_y @ self._weights_x.T
        return None

    @property
    def intercept_(self):
        return self.bias.detach()

    @property
    def n_iter_(self):
        return self.num_iter

    @property
    def n_features_in_(self):
        return self.in_features

    def __get_rho_constraints(self, X, Y):
        XY = self.cov(X, Y)
        XX = self.cov(X, X)
        YY = self.cov(Y, Y)

        num = self._weights_x.T @ XY @ self._weights_y

        VarX = self._weights_x.T @ XX @ self._weights_x
        VarY = self._weights_y.T @ YY @ self._weights_y
        denom = torch.sqrt(torch.abs(VarX * VarY)) + 1e-8

        rho = num / denom

        const_X = VarX - torch.eye(self.n_components, device=self.device, dtype=self.dtype)
        const_Y = VarY - torch.eye(self.n_components, device=self.device, dtype=self.dtype)
        return rho, const_X, const_Y

    def forward(self, X: torch.Tensor, y: torch.Tensor = None):
        X = X - X.mean(dim=-2, keepdim=True)
        if y is not None:
            y = y - y.mean(dim=-2, keepdim=True)
        if self._weights_x is None:
            if y is not None:
                self.fit(X, y)
            else:
                raise ValueError("Model must be fitted before calling forward")

        # Linear approximation
        if self.coef_ is not None:
            return F.linear(X, self.coef_, self.intercept_)
        return X

    def predict(self, X: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Predict Y from X using the fitted regression coefficients."""
        if self.coef_ is None:
            raise RuntimeError("CCA must be fitted before predict().")
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        return F.linear(X, self.coef_, self.intercept_)

    def fit(self, data_or_X, y=None, **kwargs):
        X = data_or_X
        if y is None:
            raise ValueError("y must be provided for CCA fit.")

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=self.dtype, device=self.device)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=self.dtype, device=self.device)
        if y.ndim == 1:
            y = y.unsqueeze(1)

        self._init_module_(X, y)

        # Optimization Loop
        optimizer = torch.optim.Adam([self._weights_x, self._weights_y], lr=1e-2)

        # Store for iterations
        for i in range(self.max_iter):
            optimizer.zero_grad()

            rho, const_X, const_Y = self.__get_rho_constraints(X, y)

            loss_corr = -torch.trace(rho)
            loss_const = torch.norm(const_X) + torch.norm(const_Y)
            loss = loss_corr + loss_const

            loss.backward()
            optimizer.step()

            if loss.abs() < self.tol:
                self.num_iter = i
                break
        else:
            self.num_iter = self.max_iter

        # 1. Compute X scores
        # We need X_c (centered) for consistent calculation
        if self.copy:
            # Recalculate if it was modified or not stored (it was local variable)
            X_c = X - X.mean(dim=0, keepdim=True)
            Y_c = y - y.mean(dim=0, keepdim=True)
        else:
            # Already centered in place if copy was False? No, standard in fit is to treat input as constant if possible or copy
            # We just recalculate here to be safe and clear.
            X_c = X - X.mean(dim=0, keepdim=True)
            Y_c = y - y.mean(dim=0, keepdim=True)

        if self.scale:
            X_std = X_c.std(dim=0, keepdim=True) + 1e-8
            Y_std = Y_c.std(dim=0, keepdim=True) + 1e-8
            X_c = X_c / X_std
            Y_c = Y_c / Y_std
        else:
            X_std = 1.0
            Y_std = 1.0

        # Calculate regression coefficients: Y ~ X @ B + b
        # Chain: B = W_x @ D @ Q_y
        X_scores = X_c @ self._weights_x
        Y_scores = Y_c @ self._weights_y

        Q_y = torch.linalg.pinv(Y_scores) @ Y_c
        D = torch.linalg.pinv(X_scores) @ Y_scores

        B = self._weights_x @ D @ Q_y

        if self.scale:
            B = B / X_std.T * Y_std

        if self.weights is None:
            self.weights = nn.Parameter(B.T)
        else:
            self.weights.data = B.T

        X_mean = X.mean(dim=0)
        Y_mean = y.mean(dim=0)
        self.bias.data = Y_mean - X_mean @ B
        return self


class PLSCanonical(CCA):
    def __init__(self,
                 n_components: int = 2,
                 scale: bool = True,
                 algorithm: str = "nipals",
                 max_iter: int = 500,
                 tol: float = 1e-6,
                 copy: bool = True,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(
            n_components=n_components,
            scale=scale,
            max_iter=max_iter,
            tol=tol,
            copy=copy,
            device=device,
            dtype=dtype,
            *args, **kwargs
        )
        self.algorithm = algorithm.lower()

    @property
    def x_weights_(self):
        return self._weights_x.detach() if self._weights_x is not None else None

    @property
    def y_weights_(self):
        return self._weights_y.detach() if self._weights_y is not None else None

    @property
    def x_loadings_(self):
        return self._loadings_x.detach() if self._loadings_x is not None else None

    @property
    def y_loadings_(self):
        return self._loadings_y.detach() if self._loadings_y is not None else None

    @property
    def x_rotations_(self):
        # Calculate rotation matrix R = W (P^T W)^-1
        if self._weights_x is not None and self._loadings_x is not None:
            # W: (p, k), P: (p, k)
            # P.T @ W -> (k, k)
            # R = W @ (P.T @ W).inverse()
            ptw = self._loadings_x.T @ self._weights_x
            try:
                rot = self._weights_x @ torch.linalg.inv(ptw)
                return rot.detach()
            except RuntimeError:  # Singular matrix
                return self._weights_x.detach()
        return self._weights_x.detach() if self._weights_x is not None else None

    @property
    def y_rotations_(self):
        if self._weights_y is not None and self._loadings_y is not None:
            ctq = self._loadings_y.T @ self._weights_y
            try:
                rot = self._weights_y @ torch.linalg.inv(ctq)
                return rot.detach()
            except RuntimeError:
                return self._weights_y.detach()
        return self._weights_y.detach() if self._weights_y is not None else None

    def _nipals_algorithm(self, X, Y):
        """
        Partial Least Squares Canonical (PLS-C) via the NIPALS algorithm.
        """
        n_samples, n_features = X.shape
        n_targets = Y.shape[1]
        n_components = self.n_components

        # Initialize storage
        W = torch.zeros((n_features, n_components), device=self.device, dtype=self.dtype)
        C = torch.zeros((n_targets, n_components), device=self.device, dtype=self.dtype)
        T = torch.zeros((n_samples, n_components), device=self.device, dtype=self.dtype)
        U = torch.zeros((n_samples, n_components), device=self.device, dtype=self.dtype)
        P = torch.zeros((n_features, n_components), device=self.device, dtype=self.dtype)
        Q = torch.zeros((n_targets, n_components), device=self.device, dtype=self.dtype)

        # Working copies for deflation
        X_k = X.clone()
        Y_k = Y.clone()

        for k in range(n_components):
            # 1. Initialize u (random or column with max variance)
            # Use column of Y with highest variance as start
            # u_k = Y_k[:, torch.argmax(Y_k.var(dim=0))].clone().unsqueeze(1)
            # Simplified: just use first column or random if zero
            if Y_k.abs().sum() < 1e-8:
                u_k = torch.randn(n_samples, 1, device=self.device, dtype=self.dtype)
            else:
                u_k = Y_k[:, 0].unsqueeze(1)  # Start with first column

            w_k = torch.zeros((n_features, 1), device=self.device, dtype=self.dtype)
            c_k = torch.zeros((n_targets, 1), device=self.device, dtype=self.dtype)
            t_k = torch.zeros((n_samples, 1), device=self.device, dtype=self.dtype)

            # Inner loop for convergence
            for i in range(self.max_iter):
                # 2. X weights
                # w = X^T u / (u^T u)
                if u_k.abs().sum() < 1e-8:
                    w_k = torch.randn(n_features, 1, device=self.device, dtype=self.dtype)
                else:
                    w_k = X_k.T @ u_k / (u_k.T @ u_k + 1e-8)

                # Normalize w
                w_k = w_k / (torch.norm(w_k) + 1e-8)

                # 3. Calculate t
                # t = X w
                t_k = X_k @ w_k

                # 4. Y weights
                # c = Y^T t / (t^T t)
                c_k = Y_k.T @ t_k / (t_k.T @ t_k + 1e-8)

                # Normalize c (optional? usually c is normalized in PLS-C mode B symmetric?)
                # In standard NIPALS for PLS2 (regression), c is not usually normalized to 1 in loop except for convergence check
                # But for canonical logic (symmetric), let's normalize. 
                # sklearn PLSModeB normalizes both.
                c_k = c_k / (torch.norm(c_k) + 1e-8)

                # 5. Calculate u
                # u = Y c
                u_new = Y_k @ c_k

                # Check convergence on t (or u)
                if i > 0:
                    diff = torch.norm(u_new - u_old) / (torch.norm(u_new) + 1e-8)  # Checking u convergence effectively
                    if diff < self.tol:
                        u_k = u_new
                        break
                u_k = u_new
                u_old = u_k.clone()

            # Post-loop
            # Ensure sign flip consistency (e.g. based on t)
            # sklearn flips signs to make max abs element of w positive (?) or correlation positive.

            # Calculate loadings (regressing X on t, Y on u or t)
            # For PLS Canonical (Mode B), use deflation:
            # p = X^T t / (t^T t)
            # q = Y^T u / (u^T u)
            # Deflate X = X - t p^T
            # Deflate Y = Y - u q^T

            # Re-calculate t, u with final w, c to be sure
            t_k = X_k @ w_k
            u_k = Y_k @ c_k

            # Loadings
            p_k = X_k.T @ t_k / (t_k.T @ t_k + 1e-8)
            q_k = Y_k.T @ u_k / (u_k.T @ u_k + 1e-8)  # Using u for Y deflation in Canonical

            # Deflation
            X_k = X_k - t_k @ p_k.T
            Y_k = Y_k - u_k @ q_k.T

            # Store
            W[:, k] = w_k.squeeze()
            C[:, k] = c_k.squeeze()
            T[:, k] = t_k.squeeze()
            U[:, k] = u_k.squeeze()
            P[:, k] = p_k.squeeze()
            Q[:, k] = q_k.squeeze()

        return W, C, T, U, P, Q

    def fit(self, data_or_X, y=None, **kwargs):
        """
        Fit model to data.

        Parameters
        ----------
        data_or_X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of predictors.

        y : array-like of shape (n_samples, n_targets)
            Target vectors, where `n_samples` is the number of samples and
            `n_targets` is the number of response variables.
        """
        X = data_or_X
        if y is None:
            raise ValueError("y must be provided for PLS fit.")

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=self.dtype, device=self.device)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=self.dtype, device=self.device)

        n_samples, n_features = X.shape
        n_targets = y.shape[1]

        # Set n_components
        self.n_components = min(self.n_components, n_features, n_targets, n_samples)

        # Center and Scale
        X_mean = X.mean(dim=0, keepdim=True)
        Y_mean = y.mean(dim=0, keepdim=True)
        X_c = X - X_mean
        Y_c = y - Y_mean

        if self.scale:
            X_std = X_c.std(dim=0, keepdim=True) + 1e-8
            Y_std = Y_c.std(dim=0, keepdim=True) + 1e-8
            X_c = X_c / X_std
            Y_c = Y_c / Y_std
        else:
            X_std = torch.ones_like(X_mean)
            Y_std = torch.ones_like(Y_mean)

        self._weights_x = None
        self._weights_y = None
        self._loadings_x = None
        self._loadings_y = None

        if self.algorithm == 'svd':
            # SVD on Cross-Covariance
            # C = X.T @ Y
            C = X_c.T @ Y_c
            U, S, Vh = torch.linalg.svd(C, full_matrices=False)
            V = Vh.T

            # Slice to n_components
            self._weights_x = U[:, :self.n_components]  # (p, k)
            self._weights_y = V[:, :self.n_components]  # (q, k)

            # Scores
            X_scores = X_c @ self._weights_x  # (n, k)
            Y_scores = Y_c @ self._weights_y  # (n, k)

            # Loadings = Weights for SVD case (orthonormal)
            # But strictly, P = X.T @ T / T.T @ T
            # T = X @ W -> P = X.T @ X @ W / (W.T @ X.T @ X @ W)
            # If X is whitened, X.T @ X = I? No.

            # Calculate exact loadings from scores
            # P: (p, k)
            self._loadings_x = X_c.T @ X_scores @ torch.linalg.inv(
                X_scores.T @ X_scores + 1e-8 * torch.eye(self.n_components, device=self.device))
            self._loadings_y = Y_c.T @ Y_scores @ torch.linalg.inv(
                Y_scores.T @ Y_scores + 1e-8 * torch.eye(self.n_components, device=self.device))

        else:  # nipals
            W, C, T, U, P, Q = self._nipals_algorithm(X_c, Y_c)
            self._weights_x = W
            self._weights_y = C
            self._loadings_x = P
            self._loadings_y = Q

        # Calculate coefficients
        # Y ~ X @ B + b
        # B = W (P^T W)^-1 Q^T (standard PLS regression formula, also works for Canonical if we define B this way)
        # Check specific formula for Canonical: sometimes it's just about maximizing correlation, but if we want regression coefs:
        # Y ~ T Q^T and T = X R, so Y ~ X R Q^T => B = R Q^T

        if self.n_components > 0:
            # R = W (P^T W)^-1
            # Note: _loadings_x is P
            # _weights_x is W

            # For SVD, if P=W, R=W.

            # Using property to get rotation
            R = self.x_rotations_  # (p, k) if correct
            Q = self.y_loadings_  # (q, k)

            if R is not None and Q is not None:
                # B = R @ Q.T -> (p, k) @ (k, q) -> (p, q)
                B = R @ Q.T

                # Rescale B
                if self.scale:
                    B = B / X_std.T * Y_std

                # weights shape should be (q, p) based on base class CCA implementation (n_targets, n_features)
                self.weights = nn.Parameter(B.T)

                bias_val = Y_mean.squeeze() - (X_mean @ B).squeeze()
                if bias_val.ndim == 0:
                    bias_val = bias_val.unsqueeze(0)
                self.bias = nn.Parameter(bias_val)

        return self


class PLSRegression(CCA):
    def __init__(self,
                 n_components: int = 2,
                 scale: bool = True,
                 max_iter: int = 500,
                 tol: float = 1e-6,
                 copy: bool = True,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(
            n_components=n_components,
            scale=scale,
            max_iter=max_iter,
            tol=tol,
            copy=copy,
            device=device,
            dtype=dtype,
            *args, **kwargs
        )

    def _nipals_algorithm(self, X: torch.Tensor, Y: torch.Tensor):
        n_samples, n_features = X.shape
        n_targets = Y.shape[1]
        n_components = self.n_components

        # Initialize storage
        W = torch.zeros((n_features, n_components), device=self.device, dtype=self.dtype)
        T = torch.zeros((n_samples, n_components), device=self.device, dtype=self.dtype)
        P = torch.zeros((n_features, n_components), device=self.device, dtype=self.dtype)
        Q = torch.zeros((n_targets, n_components), device=self.device, dtype=self.dtype)

        # Working copies for deflation
        X_k = X.clone()
        Y_k = Y.clone()

        # Storing number of iterations
        n_iters = []

        for k in range(n_components):
            # 1. Initialize u (random or column with max variance)
            if Y_k.abs().sum() < 1e-8:
                u_k = torch.randn(n_samples, 1, device=self.device, dtype=self.dtype)
            else:
                u_k = Y_k[:, 0].unsqueeze(1)  # Start with first column

            w_k = torch.zeros((n_features, 1), device=self.device, dtype=self.dtype)
            t_k = torch.zeros((n_samples, 1), device=self.device, dtype=self.dtype)
            q_k = torch.zeros((n_targets, 1), device=self.device, dtype=self.dtype)

            # Inner loop for convergence
            iter_count = 0
            for i in range(self.max_iter):
                # 2. X weights
                # w = X^T u / (u^T u)
                if u_k.abs().sum() < 1e-8:
                    w_k = torch.randn(n_features, 1, device=self.device, dtype=self.dtype)
                else:
                    w_k = X_k.T @ u_k / (u_k.T @ u_k + 1e-8)

                # Normalize w
                w_k = w_k / (torch.norm(w_k) + 1e-8)

                # 3. Calculate t
                # t = X w
                t_k = X_k @ w_k

                # 4. Y weights/loadings
                # q = Y^T t / (t^T t)
                # Note: docstring says q_h = Y.T t / || Y.T t || in loop step.
                # But typically PLS2 (regression) uses q as loading for Y. Here q is direction in Y space.
                q_k = Y_k.T @ t_k
                q_k = q_k / (torch.norm(q_k) + 1e-8)

                # 5. Calculate u
                # u = Y q
                u_new = Y_k @ q_k

                # Check convergence on t (or u)
                if i > 0:
                    # Check change in t or u? Docstring says "u_i - u_{i-1}"
                    diff = torch.norm(u_new - u_old) / (torch.norm(u_new) + 1e-8)  # Checking u convergence
                    if diff < self.tol:
                        u_k = u_new
                        iter_count = i
                        break
                u_k = u_new
                u_old = u_k.clone()
            else:
                iter_count = self.max_iter

            n_iters.append(iter_count)

            # Post-loop
            # Calculate loadings
            # p = X^T t / (t^T t)
            t_sq_norm = t_k.T @ t_k + 1e-8
            p_k = X_k.T @ t_k / t_sq_norm

            # Calculate y-loading (regression coef of Y on t) for deflation
            # b = u^T t / t^T t
            # Y deflation: Y = Y - b * t * q^T?
            # Docstring says: b_h = u_h^T t_h / (t_h^T t_h)
            # Y_h = Y_{h-1} - b_h t_h q_h^T
            # Wait. If q_h is normalized, then b_h accounts for scale.
            b_k = (u_k.T @ t_k) / t_sq_norm

            # Deflation
            X_k = X_k - t_k @ p_k.T
            Y_k = Y_k - b_k * (t_k @ q_k.T)

            # Store
            W[:, k] = w_k.squeeze()
            T[:, k] = t_k.squeeze()
            P[:, k] = p_k.squeeze()
            # Store Y-loadings as b * q because coef calculation asks for Q.T where Y ~ T Q.T
            # If Y ~ b t q^T, then Q_eff = b * q
            Q[:, k] = (b_k * q_k).squeeze()

        return W, T, P, Q, torch.tensor(n_iters, device=self.device)

    def fit(self, data_or_X, y=None, **kwargs):
        """
        Fit model to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of predictors.

        y : array-like of shape (n_samples, n_targets)
            Target vectors, where `n_samples` is the number of samples and
            `n_targets` is the number of response variables.
        """
        X = data_or_X
        if y is None:
            raise ValueError("y must be provided for PLS fit.")

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=self.dtype, device=self.device)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=self.dtype, device=self.device)

        n_samples, n_features = X.shape
        n_targets = y.shape[1]

        # Set n_components
        self.n_components = min(self.n_components, n_features, n_targets, n_samples)

        # Center and Scale
        X_mean = X.mean(dim=0, keepdim=True)
        Y_mean = y.mean(dim=0, keepdim=True)
        X_c = X - X_mean
        Y_c = y - Y_mean

        if self.scale:
            X_std = X_c.std(dim=0, keepdim=True) + 1e-8
            Y_std = Y_c.std(dim=0, keepdim=True) + 1e-8
            X_c = X_c / X_std
            Y_c = Y_c / Y_std
        else:
            X_std = torch.ones_like(X_mean)
            Y_std = torch.ones_like(Y_mean)

        self._weights_x = None
        self._weights_y = None  # Not used/returned by NIPALS really but maybe needed for base class compat?
        self._loadings_x = None
        self._loadings_y = None
        self._x_scores = None
        self._y_scores = None  # Not tracked explicitly in loop? Usually Y scores = U.
        self._n_iter = None

        # NIPALS
        W, T, P, Q, n_iters = self._nipals_algorithm(X_c, Y_c)

        self._weights_x = W
        self._loadings_x = P
        self._loadings_y = Q
        self._x_scores = T
        # Y scores? Typically U. But for regression we care about T.
        # Can store U if needed. But _nipals_algorithm didn't return U.
        # Let's just set y_scores to None or derived?
        # Standard sklearn PLSRegression has y_scores_ (U).
        # We can reconstruct or return U from nipals.
        # Re-run nipals to return U? Or just ignore for now as it's not critical for prediction.

        self._n_iter = n_iters

        # Calculate coefficients
        # Y ~ X @ B + b
        # B = W (P^T W)^-1 Q^T

        if self.n_components > 0:
            # R = W (P^T W)^-1
            # Note: _loadings_x is P
            # _weights_x is W

            # Using property to get rotation
            R = self.x_rotations_  # (p, k)
            Q = self.y_loadings_  # (q, k)

            if R is not None and Q is not None:
                # B = R @ Q.T -> (p, k) @ (k, q) -> (p, q)
                B = R @ Q.T

                # Rescale B
                if self.scale:
                    B = B / X_std.T * Y_std

                # weights shape should be (q, p) based on base class CCA implementation (n_targets, n_features)
                self.weights = nn.Parameter(B.T)

                bias_val = Y_mean.squeeze() - (X_mean @ B).squeeze()
                if bias_val.ndim == 0:
                    bias_val = bias_val.unsqueeze(0)
                self.bias = nn.Parameter(bias_val)

        return self

    @property
    def x_weights_(self):
        return self._weights_x.detach() if self._weights_x is not None else None

    @property
    def y_weights_(self):
        # We don't have y_weights (C) stored from loop in current implementation?
        # NIPALS loop computes q (loadings).
        # The true "weights" for Y in PLS2 (C) aren't stored.
        # But for PLS Regression, y_weights_ usually refers to the weights used to compute U.
        # Which is q in step 2.
        # But we stored Q as loadings (b*q).
        # It's okay to return None or approximations if not critical.
        return None

    @property
    def x_loadings_(self):
        return self._loadings_x.detach() if self._loadings_x is not None else None

    @property
    def y_loadings_(self):
        return self._loadings_y.detach() if self._loadings_y is not None else None

    @property
    def x_scores_(self):
        return self._x_scores.detach() if self._x_scores is not None else None

    @property
    def y_scores_(self):
        return self._y_scores.detach() if self._y_scores is not None else None

    @property
    def x_rotations_(self):
        # Calculate rotation matrix R = W (P^T W)^-1
        if self._weights_x is not None and self._loadings_x is not None:
            ptw = self._loadings_x.T @ self._weights_x
            try:
                rot = self._weights_x @ torch.linalg.inv(ptw)
                return rot.detach()
            except RuntimeError:  # Singular matrix
                return self._weights_x.detach()
        return self._weights_x.detach() if self._weights_x is not None else None

    @property
    def y_rotations_(self):
        return None  # Not strictly defined/needed for PLS regression Y side usually

    @property
    def n_iter_(self):
        return self._n_iter.detach().cpu().tolist() if self._n_iter is not None else None


class PLSSVD(CCA):
    def __init__(self,
                 n_components: int = 2,
                 scale: bool = True,
                 copy: bool = True,
                 algorithm: Union[str, Callable] = "full",
                 max_iter: int = 500,
                 tol: float = 1e-6,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args: Any,
                 **kwargs: Any,
                 ) -> None:
        super().__init__(
            n_components=n_components,
            scale=scale,
            max_iter=max_iter,
            tol=tol,
            copy=copy,
            device=device,
            dtype=dtype,
            *args,
            **kwargs,
        )
        self.algorithm = str(algorithm).lower() if isinstance(algorithm, str) else algorithm
        self.kwargs = dict(kwargs)
        self.kwargs["_args"] = args

        self._x_weights = None
        self._y_weights = None
        self._x_mean = None
        self._y_mean = None
        self._x_std = None
        self._y_std = None
        self.fit_status = False

    def _arpack_svd(
        self,
        C: torch.Tensor,
        n_components: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_features, n_targets = C.shape
        k = min(n_components, min(n_features, n_targets))
        k = max(1, k)
        eps = 1e-12
        tol = 1e-8
        if n_features >= n_targets:
            G = C @ C.T
            n_g = n_features
            transposed = False
        else:
            G = C.T @ C
            n_g = n_targets
            transposed = True
        S_sq = []
        V_list = []
        G_work = G.clone()
        for j in range(k):
            v = torch.randn(n_g, device=C.device, dtype=C.dtype)
            v = v / (v.norm().clamp(min=eps))
            for _ in range(max(50, n_g)):
                v_new = G_work @ v
                v_new = v_new / (v_new.norm().clamp(min=eps))
                if (v_new - v).norm() < tol:
                    break
                v = v_new
            lam = (v @ G_work @ v).item()
            lam = max(lam, eps)
            s = (lam ** 0.5)
            S_sq.append(s)
            V_list.append(v.clone())
            G_work = G_work - lam * torch.outer(v, v)
        S = torch.tensor(S_sq, device=C.device, dtype=C.dtype)
        V_stack = torch.stack(V_list, dim=1)
        if transposed:
            # G = C.T @ C, V_stack = right singular vectors of C
            U = (C @ V_stack) / S.clamp(min=eps)
            Vh = V_stack.T
            return U[:, :k], S, Vh[:k]
        else:
            # G = C @ C.T, V_stack = left singular vectors U of C
            U = V_stack
            Vh = (V_stack.T @ C) / S.clamp(min=eps).unsqueeze(1)
            return U[:, :k], S, Vh[:k]

    def _randomized_svd(
        self,
        C: torch.Tensor,
        n_components: int,
        n_oversamples: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_f, n_t = C.shape
        k = min(n_components, min(n_f, n_t))
        k = max(1, k)
        n_oversamples = min(n_oversamples, max(n_f, n_t) - k)
        n_oversamples = max(0, n_oversamples)
        l = k + n_oversamples
        l = min(l, min(n_f, n_t))

        if n_f >= n_t:
            Omega = torch.randn(n_t, l, device=C.device, dtype=C.dtype)
            Y = C @ Omega
            Q, _ = torch.linalg.qr(Y)
            B = Q.T @ C
            U_b, S, Vh_b = torch.linalg.svd(B, full_matrices=False)
            U = Q @ U_b[:, :k]
            Vh = Vh_b[:k]
            S = S[:k]
        else:
            Omega = torch.randn(n_f, l, device=C.device, dtype=C.dtype)
            Y = C.T @ Omega
            Q, _ = torch.linalg.qr(Y)
            B = C @ Q
            U_b, S, Vh_b = torch.linalg.svd(B, full_matrices=False)
            Vh = (Vh_b[:, :k].T @ Q.T).T
            U = U_b[:, :k]
            S = S[:k]

        return U, S, Vh

    def _svd_cross_covariance(
        self,
        C: torch.Tensor,
        n_components: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute truncated SVD of cross-covariance C. Uses self.algorithm or kwargs."""
        algorithm = self.kwargs.get("algorithm", self.algorithm)
        if isinstance(algorithm, str):
            algorithm = str(algorithm).lower()
        if algorithm == "arpack":
            return self._arpack_svd(C, n_components)
        if algorithm == "randomized":
            n_oversamples = self.kwargs.get("n_oversamples", 10)
            return self._randomized_svd(C, n_components, n_oversamples)
        U, S, Vh = torch.linalg.svd(C, full_matrices=False)
        k = min(n_components, S.shape[0])
        return U[:, :k], S[:k], Vh[:k]

    @property
    def x_weights_(self) -> Optional[torch.Tensor]:
        return self._x_weights.detach() if self._x_weights is not None else None

    @property
    def y_weights_(self) -> Optional[torch.Tensor]:
        return self._y_weights.detach() if self._y_weights is not None else None

    def fit(self, data_or_X, y=None, **kwargs) -> "PLSSVD":
        if y is None:
            raise ValueError("y must be provided for PLSSVD fit.")
        X = torch.as_tensor(data_or_X, device=self.device, dtype=self.dtype)
        y = torch.as_tensor(y, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(1)
        n_samples, n_features = X.shape
        n_targets = y.shape[1]
        self.in_features = n_features

        if self.copy:
            X = X.clone()
            y = y.clone()

        self._x_mean = X.mean(dim=0, keepdim=True)
        self._y_mean = y.mean(dim=0, keepdim=True)
        X_c = X - self._x_mean
        y_c = y - self._y_mean

        if self.scale:
            self._x_std = X_c.std(dim=0, keepdim=True).clamp(min=1e-8)
            self._y_std = y_c.std(dim=0, keepdim=True).clamp(min=1e-8)
            X_c = X_c / self._x_std
            y_c = y_c / self._y_std
        else:
            self._x_std = torch.ones_like(self._x_mean)
            self._y_std = torch.ones_like(self._y_mean)

        n_comp = min(self.n_components, n_features, n_targets, n_samples)
        n_comp = max(1, n_comp)

        C = X_c.T @ y_c
        U, S, Vh = self._svd_cross_covariance(C, n_comp)
        self._x_weights = U
        self._y_weights = Vh.T
        self.in_features = n_features
        self.out_features = n_targets
        self.fit_status = True
        return self

    def transform(
        self,
        X: Union[torch.Tensor, Any],
        y: Optional[Union[torch.Tensor, Any]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self._x_weights is None:
            raise RuntimeError("PLSSVD instance is not fitted yet.")
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        X_c = X - self._x_mean
        X_c = X_c / self._x_std
        x_scores = X_c @ self._x_weights
        if y is not None:
            y = torch.as_tensor(y, device=self.device, dtype=self.dtype)
            if y.dim() == 1:
                y = y.unsqueeze(1)
            y_c = y - self._y_mean
            y_c = y_c / self._y_std
            y_scores = y_c @ self._y_weights
            return x_scores, y_scores
        return x_scores

    def fit_transform(
        self,
        data_or_X: Union[torch.Tensor, Any],
        y: Optional[Union[torch.Tensor, Any]] = None,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        self.fit(data_or_X, y=y, **kwargs)
        return self.transform(data_or_X, y)

    def predict(self, X: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Predict y from X using the learned cross-covariance structure. Returns (n_samples, n_targets)."""
        if self._x_weights is None:
            raise RuntimeError("PLSSVD instance is not fitted yet.")
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        X_c = (X - self._x_mean) / self._x_std
        # y_pred = y_mean + (X_c @ x_weights) @ y_weights.T  -> (n, n_targets)
        x_scores = X_c @ self._x_weights
        y_pred = self._y_mean + x_scores @ self._y_weights.T
        return y_pred

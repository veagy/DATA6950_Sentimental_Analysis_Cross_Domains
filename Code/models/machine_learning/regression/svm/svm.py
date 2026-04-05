import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Union

__all__ = ["SVR", "LinearSVR", "NuSVR"]
from .....models.utils import MLModule, MLRegressor
from .kernels import *
from torch.func import vmap
import joblib


class SVR(MLRegressor):
    def __init__(self,
                 kernel: Union[str, Callable, MLModule, nn.Module] = "rbf",
                 gamma: Union[str, float] = 'scale',
                 degree: int = 3,
                 coef0: float = 0.0,
                 tol: float = 1e-3,
                 C: float = 1.0,
                 epsilon: float = 0.1,
                 shrinking: float = True,
                 cache_size: float = 200,
                 verbose: bool = False,
                 max_iter: int = 1000,
                 trainable_kernel: bool = False,
                 n_support_vectors: int = 100,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.gamma = gamma
        self.in_features = None
        self.out_features = None
        self.device = device
        self.dtype = dtype
        self.degree = degree
        self.args = args
        self.kwargs = kwargs
        self.trainable_kernel = trainable_kernel
        self.n_support_vectors = n_support_vectors

        self.kernel_kwargs = {
            "device": device,
            "dtype": dtype,
            "gamma": gamma,
            "degree": degree,
            "args": args,
            "trainable": trainable_kernel,
            "num_support_vectors": n_support_vectors,
            **kwargs
        }
        self.kernel = kernel
        # Dynamically retrieve kernel class if string is provided
        # No hardcoded dictionary anymore

        self.coef0 = coef0
        self.tol = tol
        self.C = C
        self.epsilon = epsilon
        self.shrinking = shrinking
        self.verbose = verbose
        self.max_iter = max_iter
        self._num_iter = None
        self.cache_size = cache_size
        self._linear_kernel_weights = None
        self._dual_coef = None
        self._fit_status = -1
        self._intercept = None
        self._support_vectors = None
        self.support_indices_ = None

    def _init_module(self, X: torch.Tensor, y: torch.Tensor):
        in_features = X.size(-1)
        out_features = y.size(-1)

        if isinstance(self.gamma, str):
            if self.gamma.lower() == "scale":
                self.gamma = 1 / (in_features * X.var(dim=[0, 1])) if X.numel() > 1 else 1.0
            elif self.gamma.lower() == "auto":
                self.gamma = 1 / in_features
        elif isinstance(self.gamma, float):
            self.gamma = abs(self.gamma)

        self.in_features = in_features
        self.out_features = out_features

        # Update kernel_kwargs with computed gamma and dimensions
        self.kernel_kwargs.update({
            "gamma": self.gamma,
            "num_features": in_features,
            "num_classes": out_features
        })

        if isinstance(self.kernel, str):
            kernel_class = get_kernel_class(self.kernel)
            if kernel_class:
                self.kernel = kernel_class(**self.kernel_kwargs)
            else:
                available_kernels = KernelRegistry.list_kernels()
                raise ValueError(f"Unknown kernel type: '{self.kernel}'.\nAvailable kernels: {available_kernels}")
        return self

    @property
    def coef_(self):  # Weights assigned to the features when kernel="linear".
        if isinstance(self.kernel, LinearKernel):
            if self.trainable_kernel:
                # _dual_coef: (M, C), SVs: (C, M, D)
                # Need to sum over support vectors M
                # beta_c (M,) * SV_c (M, D) -> (D,)
                # Output (C, D)
                weights = []
                for c in range(self.out_features):
                    w = self._dual_coef[:, c] @ self._support_vectors[c]
                    weights.append(w)
                return torch.stack(weights)
            return self._dual_coef.T @ self._support_vectors
        else:
            return None

    @property
    def dual_coef_(self):
        return self._dual_coef

    @property
    def fit_status_(self):
        return self._fit_status

    @property
    def intercept_(self):
        return self._intercept

    @property
    def n_features_in_(
            self):
        return self.in_features

    @property
    def n_iter_(self):
        return self._num_iter

    @property
    def n_support_(self):
        if self._support_vectors is not None:
            if self.trainable_kernel:
                return self.n_support_vectors
            return self._support_vectors.size(0)
        return 0

    @property
    def shape_fit_(self):
        if self._support_vectors is not None:
            return self.n_support_, self.in_features
        return None

    @property
    def support_vectors_(self):  # Support vectors.
        return self._support_vectors

    @property
    def support_(self):  # Indices of support vectors.
        return self.support_indices_

    def forward(self, X: torch.Tensor, y: torch.Tensor = None):
        fitted = getattr(self, 'fit_status', None) or getattr(self, '_fit_status', -1) == 0
        if not fitted and y is not None:
            self.fit(X, y)
        return self.predict(X)

    def fit(self, data_or_X, y=None, **kwargs):
        X = torch.as_tensor(data_or_X, dtype=self.dtype, device=self.device)
        y = torch.as_tensor(y, dtype=self.dtype, device=self.device)

        if y.ndim == 1:
            y = y.view(-1, 1)

        self._init_module(X, y)

        N = X.size(0)
        n_targets = y.size(1)

        # Primal variables logic
        if self.trainable_kernel:
            # Parametric Support Vectors Case
            # beta: (num_support_vectors, n_targets)
            num_sv = self.n_support_vectors

            beta = nn.Parameter(torch.zeros(num_sv, n_targets, device=self.device, dtype=self.dtype))
            bias = nn.Parameter(torch.zeros(1, n_targets, device=self.device, dtype=self.dtype))

            # Add kernel parameters (support vectors) to optimizer
            params = [beta, bias] + list(self.kernel.parameters())

        else:
            # Standard SVR Case
            beta = nn.Parameter(torch.zeros(N, n_targets, device=self.device, dtype=self.dtype))
            bias = nn.Parameter(torch.zeros(1, n_targets, device=self.device, dtype=self.dtype))
            params = [beta, bias]

        optimizer = torch.optim.LBFGS(params,
                                      lr=1.0,
                                      max_iter=self.max_iter,
                                      tolerance_grad=self.tol,
                                      tolerance_change=self.tol,
                                      history_size=10,
                                      line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()

            if self.trainable_kernel:
                # K_xz: Kernel between X and parametric support vectors (handled inside kernel)
                # kernel(X) -> (C, N, M)
                K_xz = self.kernel(X)

                # K_zz: Kernel between support vectors and themselves (regularization)
                # kernel.fn_support_vectors is (C, M, D)
                # kernel(None, None) -> (C, M, M) if implementation allows, or kernel(SVs, SVs)
                # We need to explicitly call kernel on SVs.
                SVs = self.kernel.fn_support_vectors
                # We need to temporarily override 'trainable' or pass SVs explicitly as both args?
                # kernel(SVs, SVs) -> dist(SVs_i, SVs_j).
                # But BaseKernel._get_vectors logic handles broadcasting?
                # Simpler: kernel(SVs, SVs)
                # Problem: kernel expects forward(xi, xj).
                # If we pass (SVs, SVs), it sees 3D tensors.
                K_zz = self.kernel(SVs, SVs)  # (C, M, M)

                # Prediction:
                # K_xz is (C, N, M). beta is (M, C).
                # Need to match dimensions.
                # preds: for each class c, K_xz[c] @ beta[:, c] + bias[c]
                # Vectorized:
                # K_xz shape (C, N, M)
                # beta_exp: (C, M, 1) <- beta.T.unsqueeze(-1)
                # preds_c = K_xz @ beta_exp -> (C, N, 1)
                # preds = preds_c.squeeze(-1).T -> (N, C)

                beta_exp = beta.transpose(0, 1).unsqueeze(-1)  # (C, M, 1)
                preds_c = torch.matmul(K_xz, beta_exp)  # (C, N, 1)
                preds = preds_c.squeeze(-1).T + bias  # (N, C) + (1, C) broadcast

                # Regularization: 0.5 * beta.T * K * beta
                # Per class: beta_c.T @ K_zz_c @ beta_c
                # beta_exp: (C, M, 1). beta_exp_T: (C, 1, M)
                # K_zz: (C, M, M)
                reg_loss_c = 0.5 * torch.matmul(torch.matmul(beta_exp.transpose(1, 2), K_zz), beta_exp)  # (C, 1, 1)
                reg_loss = reg_loss_c.sum()

            else:
                # Standard Case
                # K: (N, N)
                K = self.kernel(X, X)

                # preds: K @ beta + bias
                preds = K @ beta + bias

                # Regularization: 0.5 * diag(beta.T @ K @ beta)
                # Vectorized: sum(beta * (K @ beta)) * 0.5
                reg_term = torch.sum(beta * (K @ beta))
                reg_loss = 0.5 * reg_term

            # Epsilon-Insensitive Loss
            residuals = torch.abs(y - preds)
            loss_data = torch.clamp(residuals - self.epsilon, min=0)
            data_loss = self.C * loss_data.sum()

            total_loss = reg_loss + data_loss

            if total_loss.requires_grad:
                total_loss.backward()
            return total_loss

        optimizer.step(closure)

        self._dual_coef = beta.detach()
        self._intercept = bias.detach()
        self._num_iter = optimizer.state_dict()['state'].get(0, {}).get('func_evals', 0)

        if self.trainable_kernel:
            self._support_vectors = self.kernel.fn_support_vectors.detach()  # (C, M, D)
            # No sparsification for parametric SVR usually
            self.support_indices_ = None
        else:
            # Standard sparsification
            max_coef = torch.max(torch.abs(self._dual_coef))
            threshold = 1e-5 * max(max_coef.item(), 1.0)
            support_mask = torch.any(torch.abs(self._dual_coef) > threshold, dim=1)  # (N,)
            self.support_indices_ = torch.where(support_mask)[0]
            self._support_vectors = X[self.support_indices_]
            self._dual_coef = self._dual_coef[self.support_indices_]

        self._fit_status = 0
        return self

    def predict(self, X: torch.Tensor):
        if self._fit_status != 0:
            raise RuntimeError("Model is not fitted yet.")

        X = torch.as_tensor(X, dtype=self.dtype, device=self.device)

        if self.trainable_kernel:
            # self.kernel(X) uses internal SVs -> (C, N, M)
            K_test = self.kernel(X)
            beta_exp = self._dual_coef.transpose(0, 1).unsqueeze(-1)  # (C, M, 1)
            preds_c = torch.matmul(K_test, beta_exp)  # (C, N, 1)
            preds = preds_c.squeeze(-1).T + self._intercept
        else:
            K_test = self.kernel(X, self._support_vectors)  # (M, N_support)
            preds = K_test @ self._dual_coef + self._intercept

        if self.out_features == 1 and preds.shape[1] == 1:
            return preds.view(-1)
        return preds


class LinearSVR(SVR):
    def __init__(self,
                 tol: float = 1e-3,
                 C: float = 1.0,
                 epsilon: float = 0.1,
                 loss: str = "epsilon_insensitive",
                 fit_intercept: bool = True,
                 intercept_scaling: float = 1.0,
                 dual: Union[str, bool] = "auto",
                 shrinking: float = True,
                 cache_size: float = 200,
                 verbose: bool = False,
                 max_iter: int = 1000,
                 random_state: int = None,
                 trainable_kernel: bool = False,
                 n_support_vectors: int = 100,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__(
            kernel="linear",
            gamma=1.0,
            degree=1,
            coef0=0.0,
            epsilon=epsilon,
            tol=tol,
            C=C,
            shrinking=shrinking,
            cache_size=cache_size,
            verbose=verbose,
            max_iter=max_iter,
            trainable_kernel=trainable_kernel,
            n_support_vectors=n_support_vectors,
            device=device,
            dtype=dtype,
            *args, **kwargs)
        self.loss = loss
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.dual = dual
        self.random_state = random_state

    def fit(self, data_or_X, y=None, **kwargs):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        X = torch.as_tensor(data_or_X, dtype=self.dtype, device=self.device)
        y = torch.as_tensor(y, dtype=self.dtype, device=self.device)

        if y.ndim == 1:
            y = y.view(-1, 1)

        N, D = X.shape
        n_targets = y.size(1)

        # Determine optimization mode (Primal vs Dual)
        # linear kernel w/ N > D -> Primal (w) is faster
        # linear kernel w/ N < D -> Dual (alpha) is faster
        use_dual = self.dual
        if isinstance(use_dual, str) and use_dual == "auto":
            use_dual = N < D

        if use_dual:
            # --- Dual Optimization (Alpha-Space) ---
            # Linear Kernel Matrix: K = X @ X.T
            # Solves: min 0.5 * alpha.T * K * alpha + epsilon * |alpha| - y.T * alpha
            # Subject to constraints based on C and loss type.

            # Use base SVR implementation logic but adapted for LinearSVR specific losses
            # Since SVR.fit is hardcoded for epsilon_insensitive (L1), we need custom closure here

            # Using LinearKernel explicitly
            K = torch.matmul(X, X.T)  # (N, N)

            # Dual variables
            beta = nn.Parameter(torch.zeros(N, n_targets, device=self.device, dtype=self.dtype))
            bias = nn.Parameter(torch.zeros(1, n_targets, device=self.device, dtype=self.dtype))

            optimizer = torch.optim.LBFGS([beta, bias],
                                          lr=1.0,
                                          max_iter=self.max_iter,
                                          tolerance_grad=self.tol,
                                          tolerance_change=self.tol,
                                          history_size=10,
                                          line_search_fn="strong_wolfe")

            def dual_closure():
                optimizer.zero_grad()

                # Model output: preds = K @ beta + bias
                # Note: beta here represents dual coefficients (alpha)
                preds = torch.matmul(K, beta) + bias

                # Regularization term: 0.5 * beta.T * K * beta
                # Vectorized: 0.5 * sum(beta * (K @ beta))
                reg_loss = 0.5 * torch.sum(beta * torch.matmul(K, beta))

                # Data Loss
                residuals = torch.abs(y - preds)

                if self.loss == "epsilon_insensitive":
                    # L1 Loss: sum(max(0, |y - preds| - epsilon))
                    loss_data = torch.clamp(residuals - self.epsilon, min=0)
                    data_loss = self.C * loss_data.sum()
                elif self.loss == "squared_epsilon_insensitive":
                    # L2 Loss: sum(max(0, |y - preds| - epsilon)^2)
                    loss_data = torch.clamp(residuals - self.epsilon, min=0)
                    data_loss = self.C * (loss_data ** 2).sum()
                else:
                    raise ValueError(f"Unsupported loss '{self.loss}'")

                total_loss = reg_loss + data_loss
                if total_loss.requires_grad:
                    total_loss.backward()
                return total_loss

            optimizer.step(dual_closure)

            self._dual_coef = beta.detach()
            self._intercept = bias.detach()

            # Store support vectors (all training data for linear kernel dual)
            # In linear case w = X.T @ alpha. We can compute and store w instead of SVs for prediction.
            # But SVR class expects _dual_coef and _support_vectors or _linear_kernel_weights?
            # Let's compute weights w for fast prediction
            self._linear_kernel_weights = torch.matmul(X.T, self._dual_coef)  # (D, n_targets)
            self._fit_status = 0

            # For compatibility with SVR properties
            self._support_vectors = X
            self.support_indices_ = torch.arange(N, device=self.device)

        else:
            # --- Primal Optimization (W-Space) ---
            # Solves: min 0.5 * ||w||^2 + C * Loss(y, Xw + b)

            # Handle intercept by augmentation
            if self.fit_intercept:
                # Add constant feature column
                intercept_col = torch.full((N, 1), self.intercept_scaling, device=self.device, dtype=self.dtype)
                X_aug = torch.cat([X, intercept_col], dim=1)
                D_aug = D + 1
            else:
                X_aug = X
                D_aug = D

            # Primal variables: w (D_aug, n_targets)
            # Initialize w with zeros
            w = nn.Parameter(torch.zeros(D_aug, n_targets, device=self.device, dtype=self.dtype))

            optimizer = torch.optim.LBFGS([w],
                                          lr=1.0,
                                          max_iter=self.max_iter,
                                          tolerance_grad=self.tol,
                                          tolerance_change=self.tol,
                                          history_size=10,
                                          line_search_fn="strong_wolfe")

            def primal_closure():
                optimizer.zero_grad()

                # Predictions: y_pred = X_aug @ w
                preds = torch.matmul(X_aug, w)

                # Regularization: 0.5 * ||w||^2
                # If fitting intercept, usually we don't regularize the bias term (last element of w)
                if self.fit_intercept:
                    w_reg = w[:-1, :]  # Exclude intercept
                else:
                    w_reg = w

                reg_loss = 0.5 * torch.sum(w_reg ** 2)

                # Data Loss
                residuals = torch.abs(y - preds)

                if self.loss == "epsilon_insensitive":
                    loss_data = torch.clamp(residuals - self.epsilon, min=0)
                    data_loss = self.C * loss_data.sum()
                elif self.loss == "squared_epsilon_insensitive":
                    loss_data = torch.clamp(residuals - self.epsilon, min=0)
                    data_loss = self.C * (loss_data ** 2).sum()
                else:
                    raise ValueError(f"Unsupported loss '{self.loss}'")

                total_loss = reg_loss + data_loss
                if total_loss.requires_grad:
                    total_loss.backward()
                return total_loss

            optimizer.step(primal_closure)

            # Extract w and b
            if self.fit_intercept:
                self._linear_kernel_weights = w[:-1, :].detach()  # (D, n_targets)
                # Intercept is w_last * intercept_scaling
                # bias_term = w[-1]
                # proper intercept = bias_term * scaling ? No, pred = x*w + bias*scaling
                # scikit-learn: intercept_ = w[-1] if scaling=1.
                # If scaling C, feature is C. w[-1]*C is the bias added.
                self._intercept = w[-1, :].unsqueeze(0).detach() * self.intercept_scaling
            else:
                self._linear_kernel_weights = w.detach()
                self._intercept = torch.zeros(1, n_targets, device=self.device, dtype=self.dtype)

            self._fit_status = 0
            self._dual_coef = None  # Not available in primal
            self._support_vectors = None  # Not available in primal

        return self

    def predict(self, X: torch.Tensor):
        if self._fit_status != 0:
            raise RuntimeError("Model is not fitted yet.")

        X = torch.as_tensor(X, dtype=self.dtype, device=self.device)

        # Linear prediction is simply X @ w + b
        if self._linear_kernel_weights is not None:
            preds = torch.matmul(X, self._linear_kernel_weights) + self._intercept
        else:
            # Fallback to base class predict if for some reason we populated dual coefs (e.g. from dual path without converting to w)
            # But our dual path computes w.
            return super().predict(X)

        if self.out_features == 1 and preds.shape[1] == 1:
            return preds.view(-1)
        return preds


class NuSVR(SVR):
    def __init__(self,
                 kernel: Union[str, Callable, MLModule, nn.Module] = "rbf",
                 gamma: Union[str, float] = 'scale',
                 degree: int = 3,
                 coef0: float = 0.0,
                 tol: float = 1e-3,
                 nu: float = 0.5,
                 C: float = 1.0,
                 epsilon: float = 0.1,
                 shrinking: float = True,
                 cache_size: float = 200,
                 verbose: bool = False,
                 max_iter: int = 1000,
                 trainable_kernel: bool = False,
                 n_support_vectors: int = 100,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__(
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            epsilon=epsilon,
            tol=tol,
            C=C,
            shrinking=shrinking,
            cache_size=cache_size,
            verbose=verbose,
            max_iter=max_iter,
            trainable_kernel=trainable_kernel,
            n_support_vectors=n_support_vectors,
            device=device,
            dtype=dtype,
            *args, **kwargs)
        self.nu = nu if 0.0 <= nu <= 1.0 else 0.5

    def fit(self, data_or_X, y=None, **kwargs):
        X = torch.as_tensor(data_or_X, dtype=self.dtype, device=self.device)
        y = torch.as_tensor(y, dtype=self.dtype, device=self.device)
        
        if y.ndim == 1:
            y = y.view(-1, 1)

        self._init_module(X, y)
        
        N = X.size(0)
        n_targets = y.size(1)

        # Initialize epsilon (nu-SVR optimizes epsilon)
        # We start with self.epsilon but treat it as a variable
        # Ideally start small? or heuristic?
        # LibSVM initializes rho (related to epsilon/bias) based on initial alpha.
        epsilon_param = nn.Parameter(torch.tensor([self.epsilon], device=self.device, dtype=self.dtype))

        # Primal variables logic
        if self.trainable_kernel:
            # Parametric Support Vectors Case
            num_sv = self.n_support_vectors
            
            beta = nn.Parameter(torch.zeros(num_sv, n_targets, device=self.device, dtype=self.dtype))
            bias = nn.Parameter(torch.zeros(1, n_targets, device=self.device, dtype=self.dtype))
            
            params = [beta, bias, epsilon_param] + list(self.kernel.parameters())
            
        else:
            # Standard SVR Case
            beta = nn.Parameter(torch.zeros(N, n_targets, device=self.device, dtype=self.dtype))
            bias = nn.Parameter(torch.zeros(1, n_targets, device=self.device, dtype=self.dtype))
            params = [beta, bias, epsilon_param]

        optimizer = torch.optim.LBFGS(params, 
                                      lr=1.0, 
                                      max_iter=self.max_iter, 
                                      tolerance_grad=self.tol, 
                                      tolerance_change=self.tol, 
                                      history_size=10, 
                                      line_search_fn="strong_wolfe")
        
        def closure():
            optimizer.zero_grad()
            
            # Epsilon must be non-negative
            # We can enforce this via softplus or absolute value in the loss usage
            # Softplus is smoother.
            curr_epsilon = F.softplus(epsilon_param) # scalar
            
            if self.trainable_kernel:
                K_xz = self.kernel(X) 
                
                # Check BaseKernel._get_vectors logic again
                # self.kernel.fn_support_vectors is (C, M, D)
                SVs = self.kernel.fn_support_vectors
                K_zz = self.kernel(SVs, SVs) # (C, M, M)

                # Prediction (Vectorized)
                beta_exp = beta.transpose(0, 1).unsqueeze(-1) # (C, M, 1)
                preds_c = torch.matmul(K_xz, beta_exp) # (C, N, 1)
                preds = preds_c.squeeze(-1).T + bias # (N, C)
                
                # Regularization: 0.5 * beta.T * K * beta
                reg_loss_c = 0.5 * torch.matmul(torch.matmul(beta_exp.transpose(1, 2), K_zz), beta_exp) # (C, 1, 1)
                reg_loss = reg_loss_c.sum()
                
            else:
                # Standard Case
                K = self.kernel(X, X)
                preds = K @ beta + bias
                reg_term = torch.sum(beta * (K @ beta))
                reg_loss = 0.5 * reg_term
            
            # Nu-SVR Loss:
            # 0.5*||w||^2 + C * (nu * epsilon + 1/N * sum(max(0, |y - f(x)| - epsilon)))
            
            # Term 1: C * nu * epsilon
            epsilon_loss = self.C * self.nu * curr_epsilon * n_targets # Multiply by targets? Usually summed over targets?
            # nu is fraction of SVs.
            
            # Term 2: C/N * sum(xi)
            residuals = torch.abs(y - preds)
            loss_data = torch.clamp(residuals - curr_epsilon, min=0)
            data_loss = (self.C / N) * loss_data.sum()
            
            total_loss = reg_loss + epsilon_loss + data_loss
            
            if total_loss.requires_grad:
                total_loss.backward()
            return total_loss
        
        optimizer.step(closure)
        
        # Store results
        self.epsilon = F.softplus(epsilon_param).item() # Update epsilon
        
        self._dual_coef = beta.detach()
        self._intercept = bias.detach()
        self._num_iter = optimizer.state_dict()['state'].get(0, {}).get('func_evals', 0)
        
        if self.trainable_kernel:
             self._support_vectors = self.kernel.fn_support_vectors.detach()
             self.support_indices_ = None 
        else:
            # Support vectors are where alpha != 0 (beta != 0)
            # In NuSVR, bounded SVs have alpha = C/N? Unbounded < C/N.
            # Ideally filter small values.
            max_coef = torch.max(torch.abs(self._dual_coef))
            threshold = 1e-5 * max(max_coef.item(), 1.0)
            support_mask = torch.any(torch.abs(self._dual_coef) > threshold, dim=1)
            self.support_indices_ = torch.where(support_mask)[0]
            self._support_vectors = X[self.support_indices_]
            self._dual_coef = self._dual_coef[self.support_indices_]
        
        self._fit_status = 0
        return self


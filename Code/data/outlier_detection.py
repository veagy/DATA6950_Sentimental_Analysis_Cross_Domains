import torch
from typing import Union
from ..data.scaling import MLModule

class OneClassSVM(MLModule):
    """
    Native PyTorch implementation of OneClassSVM for Outlier Detection.
    Optimizes a dual objective using LBFGS to identify a decision boundary enclosing
    normal data, distinguishing outliers.
    """
    def __init__(self,
                 kernel: str = 'rbf',
                 degree: int = 3,
                 gamma: Union[str, float] = 'scale',
                 coef0: float = 0.0,
                 tol: float = 1e-3,
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
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        # Restrict max iterations if unspecified
        self.max_iter = max_iter if max_iter > 0 else 2000 
        self.device = device
        self.dtype = dtype

        self.support_ = None
        self.support_vectors_ = None
        self.dual_coef_ = None
        self.intercept_ = None
        self.fit_status = False

    def _compute_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        if self.kernel == 'linear':
            return torch.matmul(X, Y.T)
        elif self.kernel == 'poly':
            return (self._gamma * torch.matmul(X, Y.T) + self.coef0) ** self.degree
        elif self.kernel == 'rbf':
            x_sq = torch.sum(X ** 2, dim=1).view(-1, 1)
            y_sq = torch.sum(Y ** 2, dim=1).view(1, -1)
            dist_sq = x_sq + y_sq - 2.0 * torch.matmul(X, Y.T)
            return torch.exp(-self._gamma * dist_sq)
        elif self.kernel == 'sigmoid':
            return torch.tanh(self._gamma * torch.matmul(X, Y.T) + self.coef0)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel}")

    def fit(self, X: torch.Tensor, y: torch.Tensor = None, **kwargs):
        X = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        n_samples, n_features = X.shape

        if isinstance(self.gamma, str):
            if self.gamma == 'scale':
                var = X.var()
                self._gamma = 1.0 / (n_features * var) if var > 0 else 1.0
            elif self.gamma == 'auto':
                self._gamma = 1.0 / n_features
            else:
                raise ValueError("gamma must be a float or 'scale' or 'auto'")
        else:
            self._gamma = float(self.gamma)

        K = self._compute_kernel(X, X)

        # Alpha parameters (dual coefficients) to optimize
        alpha = torch.ones(n_samples, requires_grad=True, device=self.device, dtype=self.dtype) / n_samples
        optimizer = torch.optim.LBFGS([alpha], lr=1.0, max_iter=self.max_iter, tolerance_grad=self.tol)

        C = 1.0 / (self.nu * n_samples)

        def closure():
            optimizer.zero_grad()
            # Enforce bounds [0, C] explicitly via clamping inside the graph
            alpha_c = torch.clamp(alpha, 0.0, C)
            
            # Objective: 0.5 * alpha^T K alpha
            obj = 0.5 * torch.dot(alpha_c, torch.matmul(K, alpha_c))
            
            # Equality constraint loss penalty: sum(alpha) = 1
            constraint_loss = 1000.0 * ((torch.sum(alpha_c) - 1.0) ** 2)
            
            loss = obj + constraint_loss
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            alpha_final = torch.clamp(alpha, 0.0, C)
            support_idx = torch.where(alpha_final > 1e-4)[0]
            
            self.support_ = support_idx
            self.support_vectors_ = X[support_idx]
            self.dual_coef_ = alpha_final[support_idx]
            
            # Intercept calculation using support vectors on the margin (0 < alpha < C)
            margin_idx = torch.where((alpha_final > 1e-4) & (alpha_final < C - 1e-4))[0]
            if len(margin_idx) > 0:
                self.intercept_ = torch.matmul(K[margin_idx], alpha_final).mean()
            elif len(support_idx) > 0: # Fallback
                self.intercept_ = torch.matmul(K[support_idx], alpha_final).mean()
            else:
                self.intercept_ = torch.tensor(0.0, device=self.device)

        self.fit_status = True
        return self

    def decision_function(self, X: torch.Tensor) -> torch.Tensor:
        if not self.fit_status:
            raise RuntimeError("Model essentially needs to be fitted first.")
            
        X = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        K = self._compute_kernel(X, self.support_vectors_)
        scores = torch.matmul(K, self.dual_coef_) - self.intercept_
        return scores

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        scores = self.decision_function(X)
        # return +1 (inlier) if scores >= 0, else -1 (outlier)
        return torch.where(scores >= 0, torch.tensor(1.0, device=X.device), torch.tensor(-1.0, device=X.device))

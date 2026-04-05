import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Callable
from .....models.utils import MLClassifier, MLModule
from ...regression.svm.kernels import *
from torch.func import vmap
import joblib


__all__ = ["SVC", "LinearSVC", "NuSVC"]


class SVC(MLClassifier):
    def __init__(self,
                 kernel: Union[str, Callable, MLModule, nn.Module] = "rbf",
                 gamma: Union[str, float] = 'scale',
                 degree: int = 3,
                 coef0: float = 0.0,
                 tol: float = 1e-3,
                 C: float = 1.0,
                 epsilon: float = 0.1,
                 shrinking: float = True,
                 probability: bool = False,
                 class_weight: Union[str, dict] = None,
                 decision_function_shape: str = 'ovr',  # {'ovo'(one-vs-one), 'ovr'(one-vs-rest)}
                 break_ties: bool = False,
                 random_state: int = None,
                 cache_size: float = 200,
                 verbose: bool = False,
                 max_iter: int = 1000,
                 trainable_kernel: bool = False,
                 n_support_vectors: int = 100,
                 warm_start: bool = False,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.warm_start = warm_start
        self.class_weight_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.coef_ = None
        self.intercept_ = None
        self.dual_coef_ = None
        self.n_features_in_ = None
        self.n_iter_ = None
        self.n_support_ = None
        self.probA_ = None
        self.probB_ = None
        self.shape_fit_ = None
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
            "bias": coef0,
            "args": args,
            "trainable": trainable_kernel,
            "num_support_vectors": n_support_vectors,
            **kwargs
        }
        self.kernel = kernel
        self.coef0 = coef0
        self.tol = tol
        self.C = C
        self.epsilon = epsilon
        self.shrinking = shrinking
        self.verbose = verbose
        self.max_iter = max_iter
        self.cache_size = cache_size
        if random_state is not None:
            torch.manual_seed(random_state)
            self.random_state = torch.Generator().manual_seed(random_state)
        else:
            self.random_state = None
        self.class_weight = class_weight
        self.probability = probability
        self.decision_function_shape = decision_function_shape
        self.break_ties = break_ties

    def _init_module(self, X: torch.Tensor, y: torch.Tensor):
        in_features = X.size(-1)
        
        # Binary classification if y has 2 unique values, else multiclass
        self.classes_ = torch.unique(y)
        self.n_classes_ = len(self.classes_)
        
        if isinstance(self.gamma, str):
            if self.gamma.lower() == "scale":
                self.gamma = 1 / (in_features * X.var()) if X.numel() > 1 else 1.0
            elif self.gamma.lower() == "auto":
                self.gamma = 1 / in_features
        elif isinstance(self.gamma, float):
            self.gamma = abs(self.gamma)

        self.n_features_in_ = in_features
        self.kernel_kwargs.update({
            "gamma": self.gamma,
            "num_features": in_features,
            "num_classes": self.n_classes_
        })

        if isinstance(self.kernel, str):
            kernel_class = get_kernel_class(self.kernel)
            if kernel_class:
                self.kernel = kernel_class(**self.kernel_kwargs)
            else:
                available_kernels = KernelRegistry.list_kernels()
                raise ValueError(f"Unknown kernel type: '{self.kernel}'.\nAvailable kernels: {available_kernels}")
        return self

    def _get_class_weight_vector(self, y: torch.Tensor):
        if self.class_weight is None:
            return torch.ones(self.n_classes_, device=self.device, dtype=self.dtype)
        
        if self.class_weight == 'balanced':
            counts = torch.bincount(y.long())
            weights = len(y) / (self.n_classes_ * counts.float())
            return weights
        
        if isinstance(self.class_weight, dict):
            weights = torch.ones(self.n_classes_, device=self.device, dtype=self.dtype)
            for cls_idx, weight in self.class_weight.items():
                if cls_idx in self.classes_:
                    idx = (self.classes_ == cls_idx).nonzero(as_tuple=True)[0]
                    weights[idx] = weight
            return weights
        
        return torch.ones(self.n_classes_, device=self.device, dtype=self.dtype)

    def fit(self, data_or_X, y=None, **kwargs):
        self._fit_loss_fn = kwargs.get('loss_fn', None)
        try:
            kwargs.get('warm_start', getattr(self, 'warm_start', False))  # Accept for API consistency
            X = torch.as_tensor(data_or_X, dtype=self.dtype, device=self.device)
            y = torch.as_tensor(y, dtype=self.dtype, device=self.device)
            
            self._init_module(X, y)
            class_weights = self._get_class_weight_vector(y)
            self.class_weight_ = class_weights

            if self.n_classes_ < 2:
                raise ValueError("The number of classes has to be greater than or equal to 2.")

            # One-Vs-One Strategy
            self._binary_classifiers = []
            n_classifiers = self.n_classes_ * (self.n_classes_ - 1) // 2
            self.intercept_ = torch.zeros(n_classifiers, device=self.device, dtype=self.dtype)
            self.n_iter_ = torch.zeros(n_classifiers, dtype=torch.long)
            
            all_support_vectors = []
            all_dual_coefs = []
            self.n_support_ = torch.zeros(self.n_classes_, dtype=torch.int32)

            clf_idx = 0
            for i in range(self.n_classes_):
                for j in range(i + 1, self.n_classes_):
                    cls_i, cls_j = self.classes_[i], self.classes_[j]
                    mask = (y == cls_i) | (y == cls_j)
                    X_bin, y_bin = X[mask], y[mask]
                    
                    # Relabel to -1, 1
                    y_bin_labeled = torch.where(y_bin == cls_i, -1.0, 1.0).to(self.dtype)
                    
                    C_i = self.C * class_weights[i]
                    C_j = self.C * class_weights[j]
                    sample_weight = torch.where(y_bin == cls_i, C_i, C_j)

                    # Binary SVM fit
                    N_bin = X_bin.size(0)
                    beta = nn.Parameter(torch.zeros(N_bin, 1, device=self.device, dtype=self.dtype))
                    bias = nn.Parameter(torch.zeros(1, 1, device=self.device, dtype=self.dtype))
                    
                    optimizer = torch.optim.LBFGS([beta, bias], lr=1.0, max_iter=self.max_iter,
                                                 tolerance_grad=self.tol, tolerance_change=self.tol,
                                                 history_size=10, line_search_fn="strong_wolfe")

                    def closure():
                        optimizer.zero_grad()
                        x_in = torch.as_tensor(X_bin, device=self.device, dtype=self.dtype)
                        K = self.kernel(x_in, x_in)
                        preds = K @ beta + bias
                        custom_loss_fn = getattr(self, '_fit_loss_fn', None)
                        if custom_loss_fn is not None:
                            data_loss = custom_loss_fn(y_bin_labeled.view(-1, 1), preds)
                        else:
                            loss_data = torch.clamp(1 - y_bin_labeled.view(-1, 1) * preds, min=0)
                            data_loss = (sample_weight.view(-1, 1) * loss_data).sum()
                        reg_loss = 0.5 * torch.sum(beta * (K @ beta))
                        total_loss = reg_loss + data_loss
                        if total_loss.requires_grad: total_loss.backward()
                        return total_loss

                    optimizer.step(closure)
                    
                    # Sparsify
                    max_coef = torch.max(torch.abs(beta))
                    threshold = 1e-5 * max(max_coef.item(), 1.0)
                    support_mask = (torch.abs(beta) > threshold).view(-1)
                    
                    self._binary_classifiers.append({
                        'dual_coef': beta[support_mask].detach(),
                        'support_vectors': X_bin[support_mask].detach(),
                        'intercept': bias.detach(),
                        'classes': (cls_i, cls_j)
                    })
                    
                    self.intercept_[clf_idx] = bias.item()
                    self.n_iter_[clf_idx] = optimizer.state_dict()['state'].get(0, {}).get('func_evals', 0)
                    clf_idx += 1
        
            self.fit_status = True
            return self
        finally:
            self._fit_loss_fn = None

    def decision_function(self, X: torch.Tensor):
        X = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        n_samples = X.shape[0]
        n_classifiers = len(self._binary_classifiers) if getattr(self, '_binary_classifiers', None) else 0
        if n_classifiers == 0 or self.n_classes_ is None or self.n_classes_ < 1:
            return torch.zeros(n_samples, max(1, self.n_classes_ or 1), device=self.device, dtype=self.dtype)
        ovo_scores = torch.zeros(n_samples, n_classifiers, device=self.device, dtype=self.dtype)
        
        for idx, clf in enumerate(self._binary_classifiers):
            K = self.kernel(torch.as_tensor(X, device=self.device, dtype=self.dtype), 
                           torch.as_tensor(clf['support_vectors'], device=self.device, dtype=self.dtype))
            ovo_scores[:, idx] = (K @ clf['dual_coef'] + clf['intercept']).view(-1)
            
        if self.n_classes_ == 2:
            return ovo_scores.view(-1)
            
        if self.decision_function_shape == 'ovo':
            return ovo_scores
        
        ovr_scores = torch.zeros(n_samples, self.n_classes_, device=self.device, dtype=self.dtype)
        clf_idx = 0
        for i in range(self.n_classes_):
            for j in range(i + 1, self.n_classes_):
                score = ovo_scores[:, clf_idx]
                ovr_scores[:, i] -= score
                ovr_scores[:, j] += score
                clf_idx += 1
        return ovr_scores / (self.n_classes_ - 1)

    def predict(self, X: torch.Tensor):
        if getattr(self, '_binary_classifiers', None) is None or len(self._binary_classifiers) == 0:
            n_samples = torch.as_tensor(X, device=self.device).shape[0]
            return self.classes_[0].expand(n_samples) if self.classes_ is not None and len(self.classes_) > 0 else torch.zeros(n_samples, device=self.device, dtype=torch.long)
        scores = self.decision_function(X)
        if self.n_classes_ == 2:
            # Binary case
            return torch.where(scores.view(-1) > 0, self.classes_[1], self.classes_[0])
        
        n_samples = X.shape[0]
        votes = torch.zeros(n_samples, self.n_classes_, device=self.device, dtype=torch.long)
        
        # Let's re-use decision_function logic for voting
        clf_idx = 0
        for i in range(self.n_classes_):
            for j in range(i + 1, self.n_classes_):
                K = self.kernel(torch.as_tensor(X, device=self.device, dtype=self.dtype), 
                               torch.as_tensor(self._binary_classifiers[clf_idx]['support_vectors'], device=self.device, dtype=self.dtype))
                score = (K @ self._binary_classifiers[clf_idx]['dual_coef'] + self._binary_classifiers[clf_idx]['intercept']).view(-1)
                
                votes[score > 0, j] += 1
                votes[score <= 0, i] += 1
                clf_idx += 1
        
        winning_indices = torch.argmax(votes, dim=1)
        return self.classes_[winning_indices]

    def predict_log_proba(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute log probabilities of possible outcomes for samples in X.
        """
        scores = self.decision_function(X)
        if self.n_classes_ == 2:
            proba_pos = torch.sigmoid(scores)
            proba_neg = 1 - proba_pos
            probabilities = torch.stack([proba_neg, proba_pos], dim=1)
        else:
            if self.decision_function_shape == 'ovo':
                ovr_scores = torch.zeros(X.shape[0], self.n_classes_, device=self.device, dtype=self.dtype)
                clf_idx = 0
                for i in range(self.n_classes_):
                    for j in range(i + 1, self.n_classes_):
                        score = scores[:, clf_idx]
                        ovr_scores[:, i] -= score
                        ovr_scores[:, j] += score
                        clf_idx += 1
                scores = ovr_scores / (self.n_classes_ - 1)
            
            probabilities = F.softmax(scores, dim=1)
        
        # Clamp probabilities to avoid log(0)
        return torch.log(torch.clamp(probabilities, min=1e-15))


class LinearSVC(SVC):
    def __init__(self,
                 penalty: str = 'l2',
                 loss: str = 'squared_hinge',
                 dual: Union[str, bool] = 'auto',
                 tol: float = 1e-4,
                 C: float = 1.0,
                 multi_class: str = 'ovr',
                 fit_intercept: bool = True,
                 intercept_scaling: float = 1.0,
                 class_weight: Union[str, dict] = None,
                 verbose: bool = False,
                 random_state: int = None,
                 max_iter: int = 1000,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs
                 ):
        super().__init__(
            kernel="linear",
            tol=tol,
            C=C,
            class_weight=class_weight,
            verbose=verbose,
            random_state=random_state,
            max_iter=max_iter,
            device=device,
            dtype=dtype,
            *args, **kwargs
        )
        self.penalty = penalty
        self.loss = loss
        self.dual = dual
        self.multi_class = multi_class
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.coef_ = None

    def fit(self, data_or_X, y=None, **kwargs):
        self._fit_loss_fn = kwargs.get('loss_fn', None)
        try:
            X = torch.as_tensor(data_or_X, dtype=self.dtype, device=self.device)
            y = torch.as_tensor(y, dtype=self.dtype, device=self.device)
            
            self.classes_ = torch.unique(y)
            self.n_classes_ = len(self.classes_)
            self.n_features_in_ = X.size(-1)
            
            class_weights = self._get_class_weight_vector(y)

            N, D = X.shape
            
            use_dual = self.dual
            if isinstance(use_dual, str) and use_dual == 'auto':
                use_dual = N < D

            if self.n_classes_ == 2:
                y_bin = torch.where(y == self.classes_[0], -1.0, 1.0).to(self.dtype)
                sample_weight = torch.where(y == self.classes_[0], class_weights[0], class_weights[1])
                self._fit_binary(X, y_bin, sample_weight, use_dual)
            else:
                self.coef_ = torch.zeros(self.n_classes_, D, device=self.device, dtype=self.dtype)
                self.intercept_ = torch.zeros(self.n_classes_, device=self.device, dtype=self.dtype)
                for i in range(self.n_classes_):
                    y_bin = torch.where(y == self.classes_[i], 1.0, -1.0).to(self.dtype)
                    binary_weights = torch.where(y == self.classes_[i], class_weights[i], 1.0) 
                    
                    w, b = self._fit_binary(X, y_bin, binary_weights, use_dual, return_params=True)
                    self.coef_[i] = w.view(-1)
                    self.intercept_[i] = b.view(-1)
                
            self.fit_status = True
            return self
        finally:
            self._fit_loss_fn = None

    def _fit_binary(self, X, y, weights, use_dual, return_params=False):
        N, D = X.shape
        class_weights = weights.view(-1, 1)

        if use_dual:
            K = torch.matmul(X, X.T)
            alpha = nn.Parameter(torch.zeros(N, 1, device=self.device, dtype=self.dtype))
            bias = nn.Parameter(torch.zeros(1, 1, device=self.device, dtype=self.dtype))
            
            optimizer = torch.optim.LBFGS([alpha, bias], lr=1.0, max_iter=self.max_iter,
                                         tolerance_grad=self.tol, tolerance_change=self.tol,
                                         history_size=10, line_search_fn="strong_wolfe")

            def closure():
                optimizer.zero_grad()
                preds = K @ alpha + bias
                custom_loss_fn = getattr(self, '_fit_loss_fn', None)
                if custom_loss_fn is not None:
                    data_loss = custom_loss_fn(y.view(-1, 1), preds)
                else:
                    if self.loss == 'hinge':
                        loss = torch.clamp(1 - y.view(-1, 1) * preds, min=0)
                    else: # squared_hinge
                        loss = torch.clamp(1 - y.view(-1, 1) * preds, min=0) ** 2
                    data_loss = (class_weights * loss).sum()
                reg_loss = 0.5 * torch.sum(alpha * (K @ alpha))
                total_loss = reg_loss + data_loss
                if total_loss.requires_grad: total_loss.backward()
                return total_loss

            optimizer.step(closure)
            w = torch.matmul(X.T, alpha.detach())
            b = bias.detach()
        else:
            w = nn.Parameter(torch.zeros(D, 1, device=self.device, dtype=self.dtype))
            b = nn.Parameter(torch.zeros(1, 1, device=self.device, dtype=self.dtype))
            
            optimizer = torch.optim.LBFGS([w, b], lr=1.0, max_iter=self.max_iter,
                                         tolerance_grad=self.tol, tolerance_change=self.tol,
                                         history_size=10, line_search_fn="strong_wolfe")

            def closure():
                optimizer.zero_grad()
                preds = X @ w + b
                custom_loss_fn = getattr(self, '_fit_loss_fn', None)
                if custom_loss_fn is not None:
                    data_loss = custom_loss_fn(y.view(-1, 1), preds)
                else:
                    if self.loss == 'hinge':
                        loss = torch.clamp(1 - y.view(-1, 1) * preds, min=0)
                    else: # squared_hinge
                        loss = torch.clamp(1 - y.view(-1, 1) * preds, min=0) ** 2
                    data_loss = (class_weights * loss).sum()
                reg_loss = 0.5 * torch.sum(w ** 2)
                total_loss = reg_loss + data_loss
                if total_loss.requires_grad: total_loss.backward()
                return total_loss

            optimizer.step(closure)
            w = w.detach()
            b = b.detach()

        if return_params:
            return w, b
        
        self.coef_ = w.view(1, -1)
        self.intercept_ = b.view(-1)

    def decision_function(self, X: torch.Tensor):
        X = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        return torch.matmul(X, self.coef_.T) + self.intercept_

    def predict(self, X: torch.Tensor):
        scores = self.decision_function(X)
        if self.n_classes_ == 2:
            return torch.where(scores.view(-1) > 0, self.classes_[1], self.classes_[0])
        return self.classes_[torch.argmax(scores, dim=1)]


class NuSVC(SVC):
    def __init__(self,
                 nu: float = 0.5,
                 kernel: Union[str, Callable, MLModule, nn.Module] = "rbf",
                 degree: int = 3,
                 gamma: Union[str, float] = 'scale',
                 coef0: float = 0.0,
                 shrinking: bool = True,
                 probability: bool = False,
                 tol: float = 1e-3,
                 cache_size: float = 200,
                 class_weight: Union[str, dict] = None,
                 verbose: bool = False,
                 max_iter: int = -1,
                 decision_function_shape: str = 'ovr',
                 break_ties: bool = False,
                 random_state: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            probability=probability,
            tol=tol,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state,
            device=device,
            dtype=dtype,
            *args, **kwargs
        )
        self.nu = nu

    def fit(self, data_or_X, y=None, **kwargs):
        self._fit_loss_fn = kwargs.get('loss_fn', None)
        try:
            X = data_or_X

            X = torch.as_tensor(X, dtype=self.dtype, device=self.device)
            y = torch.as_tensor(y, dtype=self.dtype, device=self.device)
            self._init_module(X, y)
            class_weights = self._get_class_weight_vector(y)

            if self.n_classes_ < 2:
                raise ValueError("The number of classes has to be greater than or equal to 2.")

            self._binary_classifiers = []
            n_classifiers = self.n_classes_ * (self.n_classes_ - 1) // 2
            self.intercept_ = torch.zeros(n_classifiers, device=self.device, dtype=self.dtype)
            
            for i in range(self.n_classes_):
                for j in range(i + 1, self.n_classes_):
                    cls_i, cls_j = self.classes_[i], self.classes_[j]
                    mask = (y == cls_i) | (y == cls_j)
                    X_bin, y_bin = X[mask], y[mask]
                    y_bin_labeled = torch.where(y_bin == cls_i, -1.0, 1.0).to(self.dtype)
                    
                    N_bin = X_bin.size(0)
                    beta = nn.Parameter(torch.zeros(N_bin, 1, device=self.device, dtype=self.dtype))
                    bias = nn.Parameter(torch.zeros(1, 1, device=self.device, dtype=self.dtype))
                    rho = nn.Parameter(torch.tensor([1.0], device=self.device, dtype=self.dtype))
                    
                    optimizer = torch.optim.LBFGS([beta, bias, rho], lr=1.0, max_iter=self.max_iter if self.max_iter > 0 else 1000,
                                                 history_size=10, line_search_fn="strong_wolfe")

                    C_i = self.C * class_weights[i]
                    C_j = self.C * class_weights[j]
                    sample_weight = torch.where(y_bin == cls_i, C_i, C_j)

                def closure():
                    optimizer.zero_grad()
                    X_bin_tensor = torch.as_tensor(X_bin, device=self.device, dtype=self.dtype)
                    K = self.kernel(X_bin_tensor, X_bin_tensor)
                    preds = K @ beta + bias
                    custom_loss_fn = getattr(self, '_fit_loss_fn', None)
                    if custom_loss_fn is not None:
                        data_loss = custom_loss_fn(y_bin_labeled.view(-1, 1), preds)
                    else:
                        curr_rho = F.softplus(rho)
                        loss_data = torch.clamp(curr_rho - y_bin_labeled.view(-1, 1) * preds, min=0)
                        data_loss = (1.0 / N_bin) * (sample_weight.view(-1, 1) * loss_data).sum() + self.C * self.nu * curr_rho
                    reg_loss = 0.5 * torch.sum(beta * (K @ beta))
                    total_loss = reg_loss + data_loss
                    if total_loss.requires_grad: total_loss.backward()
                    return total_loss

                    optimizer.step(closure)
                    
                    max_coef = torch.max(torch.abs(beta))
                    threshold = 1e-5 * max(max_coef.item(), 1.0)
                    support_mask = (torch.abs(beta) > threshold).view(-1)
                    self._binary_classifiers.append({
                        'dual_coef': beta[support_mask].detach(),
                        'support_vectors': X_bin[support_mask].detach(),
                        'intercept': bias.detach(),
                        'classes': (cls_i, cls_j)
                    })

            self.fit_status = True
            return self
        finally:
            self._fit_loss_fn = None

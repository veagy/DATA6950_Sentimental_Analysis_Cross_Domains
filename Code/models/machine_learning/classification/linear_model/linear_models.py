import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple, List, Callable, Iterable
import warnings
from .....models.utils import MLClassifier
from ...regression.linear_model import Ridge, Lasso, Lars, ElasticNet, RidgeLars, LassoLars, ElasticNetLars
from torch.func import vmap
import joblib

__all__ = [
    "LogisticRegression",
    "LogisticRegressionCV",
    "RidgeClassifier",
    "LassoClassifier",
    "ElasticNetClassifier",
    "LarsClassifier",
    "RidgeLarsClassifier",
    "LassoLarsClassifier",
    "ElasticNetLarsClassifier",
    "RidgeClassifierCV",
    "LassoClassifierCV",
    "ElasticNetClassifierCV",
    "LarsClassifierCV",
    "RidgeLarsClassifierCV",
    "LassoLarsClassifierCV",
    "ElasticNetLarsClassifierCV",
]

class LogisticRegression(MLClassifier):
    def __init__(self,
                 C: float = 1.0,
                 l1_ratio: float = 0.0,
                 dual: bool = False,
                 tol: float = 1e-4,
                 fit_intercept: bool = True,
                 intercept_scaling: float = 1,
                 class_weight: Union[dict, str] = None,
                 random_state: int = None,
                 solver: str = "lbfgs",
                 max_iter: int = 100,
                 verbose: int = 0,
                 warm_start: bool = False,
                 n_jobs: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.C = C
        self.l1_ratio = l1_ratio
        self.dual = dual
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        if random_state is not None:
            torch.manual_seed(random_state)
            self.random_state = torch.Generator().manual_seed(random_state)
        else:
            self.random_state = None
        self.solver = solver
        self.max_iter = max_iter
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype
        self.register_parameter('_weights', None)
        self.register_parameter('_bias', None)
        self._num_iter = 0
        self.in_features = None
        self.num_classes = None
        self.fit_status = False

    def _init_module_(self, X: torch.Tensor, y: torch.Tensor):
        if self.warm_start and self.fit_status and self._weights is not None:
            return self
        y_flat = y.reshape(-1)
        self._classes, counts = torch.unique(y_flat, return_counts=True)
        self.num_classes = self._classes.size(0)
        self.in_features = X.size(-1)
        n_obs = y_flat.size(0)

        # Intercept Scaling for liblinear
        if self.solver.lower() == "liblinear" and self.fit_intercept:
            effective_in_features = self.in_features + 1
            self._use_synthetic_intercept = True
        else:
            effective_in_features = self.in_features
            self._use_synthetic_intercept = False

        device = X.device
        dtype = X.dtype
        self._weights = nn.Parameter(
            torch.randn((self.num_classes, effective_in_features), generator=self.random_state, device=device,
                        dtype=dtype) * 0.01
        )

        if self.fit_intercept and not self._use_synthetic_intercept:
            self._bias = nn.Parameter(
                torch.zeros((self.num_classes,), device=device, dtype=dtype)
            )
        else:
            self._bias = None

        if self.class_weight is None:
            self.class_weight_tensor = torch.ones((self.num_classes,), device=device,
                                                  dtype=dtype) / self.num_classes
        elif isinstance(self.class_weight, str) and self.class_weight.lower() == "balanced":
            self.class_weight_tensor = n_obs / (self.num_classes * counts.to(self.dtype))
        elif isinstance(self.class_weight, dict):
            sorted_labels = sorted(self.class_weight.keys())
            weights_list = [self.class_weight.get(label, 1.0) for label in sorted_labels]
            self.class_weight_tensor = torch.tensor(weights_list, device=device, dtype=dtype)

            if self.class_weight_tensor.size(0) < self.num_classes:
                padding = torch.ones(self.num_classes - self.class_weight_tensor.size(0), device=self.device,
                                     dtype=self.dtype)
                self.class_weight_tensor = torch.cat([self.class_weight_tensor, padding], dim=0)
            elif self.class_weight_tensor.size(0) > self.num_classes:
                self.class_weight_tensor = self.class_weight_tensor[:self.num_classes]

            self.class_weight_tensor = self.class_weight_tensor * (
                    self.num_classes / (self.class_weight_tensor.sum() + 1e-10))

        if self.num_classes >= 3:
            if self.solver.lower() == "liblinear":
                warnings.warn(f"The 'liblinear' solver does not support multinomial multiclass. "
                              f"Changing to default solver 'lbfgs'.", UserWarning)
                self.solver = "lbfgs"

        if self.dual and (self.solver.lower() != "liblinear" or self.l1_ratio > 0):
            warnings.warn(f"Dual formulation is only implemented for L2 penalty with liblinear solver. "
                          f"Setting dual=False.", UserWarning)
            self.dual = False

        return self

    def decision_function(self, X: torch.Tensor) -> torch.Tensor:
        """Predict confidence scores for samples."""
        # Dynamic sourcing of device and dtype from parameters to ensure graph safety
        target_device = self._weights.device
        target_dtype = self._weights.dtype
        X = torch.as_tensor(X, device=target_device, dtype=target_dtype)
        
        if hasattr(self, "_use_synthetic_intercept") and self._use_synthetic_intercept:
            if X.size(-1) == self._weights.size(-1):
                scores = F.linear(X, self._weights, None)
            else:
                batch_dims = X.shape[:-1]
                bias_col = torch.full((*batch_dims, 1), self.intercept_scaling, device=target_device, dtype=target_dtype)
                X_eff = torch.cat([X, bias_col], dim=-1)
                scores = F.linear(X_eff, self._weights, None)
        else:
            scores = F.linear(X, self._weights, self._bias)
        return scores

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """Probability estimates."""
        X = X.to(device=self.device, dtype=self.dtype)
        scores = self.decision_function(X)
        return F.softmax(scores, dim=-1)

    def predict_log_proba(self, X: torch.Tensor) -> torch.Tensor:
        """Predict logarithm of probability estimates."""
        X = X.to(device=self.device, dtype=self.dtype)
        return torch.log(self.predict_proba(X).clamp(min=1e-9))

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict class labels for samples in X."""
        X = X.to(device=self.device, dtype=self.dtype)
        probs = self.predict_proba(X)
        indices = torch.argmax(probs, dim=-1)
        return self._classes[indices]

    def _get_loss_and_grad(self, X, y, weights, bias):
        """Compute loss and gradient for current weights and bias."""
        weights.requires_grad_(True)
        if bias is not None:
            bias.requires_grad_(True)

        scores = F.linear(X, weights, bias)
        criterion = nn.CrossEntropyLoss()
        y_indices = y.view(-1)
        loss = criterion(scores, y_indices)

        l2_reg = 0.5 / self.C * torch.sum(weights ** 2)
        total_loss = loss + l2_reg
        total_loss.backward()

        grad_w = weights.grad.detach()
        grad_b = bias.grad.detach() if bias is not None else None
        weights.grad.zero_()
        if bias is not None:
            bias.grad.zero_()

        return total_loss.detach(), grad_w, grad_b

    def _newton_cg(self, X, y):
        """Newton Conjugate Gradient solver."""
        from torch.autograd import grad

        def get_params():
            if self._bias is not None:
                return torch.cat([self._weights.view(-1), self._bias.view(-1)])
            return self._weights.view(-1)

        def set_params(p):
            offset = self._weights.numel()
            self._weights.data.copy_(p[:offset].view_as(self._weights))
            if self._bias is not None:
                self._bias.data.copy_(p[offset:].view_as(self._bias))

        for i in range(self.max_iter):
            self.zero_grad()
            scores = self.decision_function(X)
            loss = F.cross_entropy(scores, y.view(-1), weight=self.class_weight_tensor)
            l2_reg = 0.5 / self.C * torch.sum(self._weights ** 2)
            total_loss = loss + l2_reg

            grads = grad(total_loss, self.parameters(), create_graph=True)
            g = torch.cat([g.view(-1) for g in grads])

            if g.norm() < self.tol:
                if self.verbose > 0: print(f"Newton-CG converged at iteration {i}")
                break

            p = torch.zeros_like(g)
            r = -g.clone()
            d = r.clone()

            for _ in range(20):
                h_v = grad(grads, self.parameters(), grad_outputs=[
                    d[:self._weights.numel()].view_as(self._weights),
                    d[self._weights.numel():].view_as(self._bias) if self._bias is not None else None
                ], retain_graph=True)
                Hd = torch.cat([h.contiguous().view(-1) for h in h_v])

                alpha = torch.dot(r, r) / (torch.dot(d, Hd) + 1e-10)
                p = p + alpha * d
                r_new = r - alpha * Hd
                if r_new.norm() < 1e-10:
                    break
                beta = torch.dot(r_new, r_new) / torch.dot(r, r)
                d = r_new + beta * d
                r = r_new

            new_p = get_params() + p
            set_params(new_p)
            self._num_iter += 1
            if self.verbose > 0 and i % (self.verbose if self.verbose > 0 else 1) == 0:
                print(f"Newton-CG iter {i}, loss {total_loss.item():.4f}")

    def _newton_cholesky(self, X, y):
        """Newton-Cholesky solver (Explicit Hessian)."""
        for i in range(self.max_iter):
            self.zero_grad()
            scores = self.decision_function(X)
            probs = F.softmax(scores, dim=-1)

            max_delta = 0
            for c in range(self.num_classes):
                pc = probs[:, c:c + 1]
                W = pc * (1 - pc)
                target_c = (y.view(-1) == c).to(self.dtype).view(-1, 1)

                if self.class_weight_tensor is not None:
                    target_weights = self.class_weight_tensor[y.view(-1)].view(-1, 1)
                    W = W * target_weights
                    target_diff = (pc - target_c) * target_weights
                else:
                    target_diff = pc - target_c

                XtW = X.t() * W.view(-1)
                H = XtW @ X + (1.0 / self.C) * torch.eye(X.size(1), device=self.device, dtype=self.dtype)
                g = X.t() @ target_diff + self._weights.data[c:c + 1].t() / self.C

                delta = torch.linalg.solve(H, -g)
                self._weights.data[c] += delta.view(-1)
                max_delta = max(max_delta, delta.norm().item())

                if self._bias is not None:
                    gb = torch.sum(target_diff)
                    Hb = torch.sum(W)
                    self._bias.data[c] -= gb / (Hb + 1e-10)

            self._num_iter += 1
            if max_delta < self.tol:
                if self.verbose > 0: print(f"Newton-Cholesky converged at iteration {i}")
                break
            if self.verbose > 0 and i % (self.verbose if self.verbose > 0 else 1) == 0:
                print(f"Newton-Cholesky iter {i}, max_delta {max_delta:.6f}")

    def _sag_saga(self, X, y, version="saga"):
        """Stochastic Average Gradient (and SAGA)."""
        n_samples = X.size(0)
        grad_buffer = torch.zeros((n_samples, *self._weights.shape), device=self.device, dtype=self.dtype)
        if self._bias is not None:
            bias_grad_buffer = torch.zeros((n_samples, *self._bias.shape), device=self.device, dtype=self.dtype)

        avg_grad_w = torch.zeros_like(self._weights)
        avg_grad_b = torch.zeros_like(self._bias) if self._bias is not None else None
        lr = 0.01

        for i in range(self.max_iter * n_samples):
            idx = torch.randint(0, n_samples, (1,), generator=self.random_state).item()
            xi, yi = X[idx:idx + 1], y[idx:idx + 1]

            self.zero_grad()
            scores = self.decision_function(xi)
            loss = F.cross_entropy(scores, yi.view(-1), weight=self.class_weight_tensor)
            loss.backward()

            gi_w = self._weights.grad.detach()
            gi_b = self._bias.grad.detach() if self._bias is not None else None

            if i % n_samples == 0:
                if avg_grad_w.norm() < self.tol:
                    if self.verbose > 0: print(f"{version} converged at step {i}")
                    break
                if self.verbose > 0 and (i // n_samples) % (self.verbose if self.verbose > 0 else 1) == 0:
                    print(f"{version} iter {i // n_samples}, grad_norm {avg_grad_w.norm().item():.6f}")

            if version == "saga":
                diff_w = gi_w - grad_buffer[idx]
                self._weights.data -= lr * (diff_w + avg_grad_w + self._weights.data / self.C)
                avg_grad_w += diff_w / n_samples
                grad_buffer[idx] = gi_w
                if self._bias is not None:
                    diff_b = gi_b - bias_grad_buffer[idx]
                    self._bias.data -= lr * (diff_b + avg_grad_b)
                    avg_grad_b += diff_b / n_samples
                    bias_grad_buffer[idx] = gi_b
                if self.l1_ratio > 0:
                    thresh = lr * self.l1_ratio / self.C
                    self._weights.data = torch.sign(self._weights.data) * torch.clamp(
                        torch.abs(self._weights.data) - thresh, min=0)
            else:
                diff_w = gi_w - grad_buffer[idx]
                avg_grad_w += diff_w / n_samples
                self._weights.data -= lr * (avg_grad_w + self._weights.data / self.C)
                grad_buffer[idx] = gi_w
                if self._bias is not None:
                    diff_b = gi_b - bias_grad_buffer[idx]
                    avg_grad_b += diff_b / n_samples
                    self._bias.data -= lr * avg_grad_b
                    bias_grad_buffer[idx] = gi_b

            self._num_iter += 1

    def fit(self, data_or_X, y=None, **kwargs):
        # Determine target device and dtype from input data
        target_device = data_or_X.device if isinstance(data_or_X, torch.Tensor) else self.device
        target_dtype = data_or_X.dtype if isinstance(data_or_X, torch.Tensor) else self.dtype

        # Determine target device and dtype from input data
        
        
        
        X = torch.as_tensor(data_or_X, device=target_device, dtype=target_dtype)
        y = torch.as_tensor(y, device=target_device, dtype=target_dtype)
        X_flat, y_flat = X.reshape(-1, X.size(-1)), y.reshape(-1)
        self.train()

        self._init_module_(X_flat, y_flat)

        n_samples = X_flat.size(0)
        label_to_idx = {val.item(): i for i, val in enumerate(self._classes)}
        y_indices = torch.tensor([label_to_idx[val.item()] for val in y_flat], device=target_device)

        if hasattr(self, "_use_synthetic_intercept") and self._use_synthetic_intercept:
            bias_col = torch.full((n_samples, 1), self.intercept_scaling, device=target_device, dtype=target_dtype)
            X_train = torch.cat([X_flat, bias_col], dim=1)
        else:
            X_train = X_flat

        solver = self.solver.lower()
        self._num_iter = 0

        if solver == "lbfgs":
            optimizer = torch.optim.LBFGS(self.parameters(), lr=1.0, max_iter=self.max_iter, tolerance_grad=self.tol)
            criterion = nn.CrossEntropyLoss(weight=self.class_weight_tensor)

            def closure():
                optimizer.zero_grad()
                outputs = self.decision_function(X_flat)
                loss = criterion(outputs, y_indices)
                total_loss = loss + 0.5 / self.C * torch.sum(self._weights ** 2)
                total_loss.backward()
                self._num_iter += 1
                if self.verbose > 0 and self._num_iter % (self.verbose if self.verbose > 0 else 1) == 0:
                    print(f"Iteration {self._num_iter}, Loss: {total_loss.item():.4f}")
                return total_loss

            optimizer.step(closure)
        elif solver == "newton-cg":
            self._newton_cg(X_train, y_indices)
        elif solver == "newton-cholesky":
            self._newton_cholesky(X_train, y_indices)
        elif solver in ["sag", "saga", "liblinear"]:
            version = "saga" if solver in ["saga", "liblinear"] else "sag"
            self._sag_saga(X_train, y_indices, version=version)
            self._num_iter = self._num_iter // n_samples
        else:
            warnings.warn(f"Solver {solver} fallback to AdamW.")
            optimizer = torch.optim.AdamW(self.parameters(), lr=0.01, eps=self.tol)
            for i in range(self.max_iter):
                optimizer.zero_grad()
                loss = F.cross_entropy(self.decision_function(X_flat), y_indices, weight=self.class_weight_tensor)
                total_loss = loss + 0.5 / self.C * torch.sum(self._weights ** 2) + self.l1_ratio / self.C * torch.sum(
                    torch.abs(self._weights))
                total_loss.backward()
                optimizer.step()
                self._num_iter += 1
                if self.verbose > 0 and i % (self.verbose if self.verbose > 0 else 1) == 0:
                    print(f"Iteration {i}, Loss: {total_loss.item():.4f}")
                if total_loss < self.tol: break
        self.fit_status = True

        self.fit_status = True

        # Update internal tracking with effective device/dtype
        self.device = str(target_device)
        self.dtype = target_dtype
        return self

    @property
    def classes_(self):
        return self._classes

    @property
    def coef_(self):
        if hasattr(self, "_use_synthetic_intercept") and self._use_synthetic_intercept:
            return self._weights[:, :-1].detach()
        return self._weights.detach()

    @property
    def intercept_(self):
        if self._bias is not None:
            return self._bias.detach()
        elif hasattr(self, "_use_synthetic_intercept") and self._use_synthetic_intercept:
            return (self._weights[:, -1] * self.intercept_scaling).detach()
        return torch.zeros(self.num_classes, device=self.device, dtype=self.dtype)

    @property
    def n_iter_(self):
        return torch.tensor([self._num_iter], dtype=torch.int32)

    @property
    def n_features_in_(self):
        return self.in_features

    def forward(self, X: torch.Tensor, y: torch.Tensor = None, **kwargs):
        if y is not None:
            return self.fit(X, y, **kwargs)
        if not self.fit_status:
            raise RuntimeError("Model not fitted. Call fit(X, y) first.")
        return self.predict(X)

class LogisticRegressionCV(MLClassifier):
    def __init__(self,
                 Cs: Union[int, float, List[float], Tuple[float]] = 10,
                 l1_ratios: Union[float, List[float], Tuple[float]] = None,
                 dual: bool = False,
                 tol: float = 1e-4,
                 fit_intercept: bool = True,
                 intercept_scalings: Union[float, List[float]] = 1,
                 cv: Union[str, int, Callable, Iterable] = None,
                 cv_config: Optional[dict] = None,
                 scoring: Union[str, Callable] = None,
                 class_weight: Union[dict, str] = None,
                 random_state: int = None,
                 refit: bool = True,
                 solver: Union[str, List[str], Tuple[str]] = "lbfgs",
                 max_iter: int = 100,
                 verbose: int = 0,
                 warm_start: bool = False,
                 n_jobs: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()

        # Determine Cs grid
        if isinstance(Cs, int):
            self.Cs_values = torch.logspace(-4, 4, Cs).tolist()
        elif isinstance(Cs, (list, tuple)):
            self.Cs_values = list(Cs)
        else:
            self.Cs_values = [Cs]

        # Determine l1_ratios grid
        if l1_ratios is None:
            self.l1_ratios_values = [0.0]
        elif isinstance(l1_ratios, (list, tuple)):
            self.l1_ratios_values = list(l1_ratios)
        else:
            self.l1_ratios_values = [l1_ratios]

        # Determine intercept_scalings grid
        if isinstance(intercept_scalings, (list, tuple)):
            self.intercept_scalings_values = list(intercept_scalings)
        else:
            self.intercept_scalings_values = [intercept_scalings]

        # Determine solver grid
        if isinstance(solver, (list, tuple)):
            self.solver_values = list(solver)
        else:
            self.solver_values = [solver]

        # Template estimator
        self._classifier = LogisticRegression(
            C=self.Cs_values[0],
            l1_ratio=self.l1_ratios_values[0],
            dual=dual,
            tol=tol,
            fit_intercept=fit_intercept,
            intercept_scaling=self.intercept_scalings_values[0],
            class_weight=class_weight,
            random_state=random_state,
            solver=self.solver_values[0],
            max_iter=max_iter,
            verbose=verbose,
            warm_start=warm_start,
            n_jobs=n_jobs,
            device=device,
            dtype=dtype,
            *args, **kwargs
        )

        self._param_grid = {
            'C': self.Cs_values,
            'l1_ratio': self.l1_ratios_values,
            'intercept_scaling': self.intercept_scalings_values,
            'solver': self.solver_values
        }

        self.search_config = {
            "estimator": self._classifier,
            "param_grid": self._param_grid,
            "scoring": scoring,
            "n_jobs": n_jobs,
            "refit": refit,
            "cv": cv,
            "cv_config": cv_config,
            "verbose": verbose,
            "return_train_score": True,
            "store_cv_values": True,
            "return_estimators": True,
            "device": device,
            "dtype": dtype
        }
        from ...cross_validation import GridSearchCV
        self.search = GridSearchCV(**self.search_config)
        self.device = device
        self.dtype = dtype

    def fit(self, data_or_X, y=None, **kwargs):
        # Determine target device and dtype from input data
        target_device = data_or_X.device if isinstance(data_or_X, torch.Tensor) else self.device
        target_dtype = data_or_X.dtype if isinstance(data_or_X, torch.Tensor) else self.dtype

        # Determine target device and dtype from input data
        
        

        X = torch.as_tensor(data_or_X, device=target_device, dtype=target_dtype)
        y = torch.as_tensor(y, device=target_device, dtype=target_dtype)

        # Initialize internal classifier to get classes_
        self._classifier._init_module_(X, y)

        self.search.fit(X, y, **kwargs)

        if self.search.refit:
            self._classifier = self.search.best_estimator_

        self.fit_status = True

        # Update internal tracking
        self.device = str(target_device)
        self.dtype = target_dtype
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self._classifier.predict(X)

    def decision_function(self, X: torch.Tensor) -> torch.Tensor:
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        return self._classifier.decision_function(X)

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        return self._classifier.predict_proba(X)

    @property
    def classes_(self):
        return self._classifier.classes_

    @property
    def coef_(self):
        return self._classifier.coef_

    @property
    def intercept_(self):
        return self._classifier.intercept_

    @property
    def Cs_(self):
        return torch.tensor(self.Cs_values, device=self.device, dtype=self.dtype)

    @property
    def l1_ratios_(self):
        return torch.tensor(self.l1_ratios_values, device=self.device, dtype=self.dtype)

    @property
    def coefs_paths_(self):
        if self.search.cv_results_ is None:
            return None

        n_folds = self.search.n_splits_
        n_cs = len(self.Cs_values)
        n_l1_ratios = len(self.l1_ratios_values)

        # Determine n_dof
        # Peek at one estimator
        first_est = self.search.cv_results_["split0_estimator"][0]
        n_features = first_est.in_features
        n_classes = self.classes_.size(0)
        has_intercept = first_est.fit_intercept
        n_dof = n_features + (1 if has_intercept else 0)

        res = {}
        for cls_idx, cls_val in enumerate(self.classes_):
            # For binary, scikit-learn stores one path (the index 1 class)
            if n_classes <= 2:
                if cls_idx == 0: continue
                target_cls_idx = 0
            else:
                target_cls_idx = cls_idx

            path = torch.zeros((n_folds, n_cs, n_l1_ratios, n_dof),
                               device=self.device, dtype=self.dtype)

            for f in range(n_folds):
                fold_ests = self.search.cv_results_[f"split{f}_estimator"]
                for cand_idx, est in enumerate(fold_ests):
                    params = self.search.cv_results_["params"][cand_idx]
                    c_val = params["C"]
                    l1_val = params["l1_ratio"]

                    c_idx = self.Cs_values.index(c_val)
                    l1_idx = self.l1_ratios_values.index(l1_val)

                    cur_coef = est.coef_[target_cls_idx]  # (n_features,)
                    if has_intercept:
                        cur_intercept = est.intercept_[target_cls_idx:target_cls_idx + 1]
                        full_coef = torch.cat([cur_coef, cur_intercept])
                    else:
                        full_coef = cur_coef

                    path[f, c_idx, l1_idx] = full_coef

            if n_l1_ratios == 1:
                path = path.squeeze(2)

            res[cls_val.item()] = path

        return res

    @property
    def scores_(self):
        if self.search.cv_results_ is None:
            return None

        n_folds = self.search.n_splits_
        n_cs = len(self.Cs_values)
        n_l1_ratios = len(self.l1_ratios_values)

        res = {}
        for cls in self.classes_:
            scores_tensor = torch.zeros((n_folds, n_cs, n_l1_ratios),
                                        device=self.device, dtype=self.dtype)

            for cand_idx in range(len(self.search.cv_results_["params"])):
                params = self.search.cv_results_["params"][cand_idx]
                c_idx = self.Cs_values.index(params["C"])
                l1_idx = self.l1_ratios_values.index(params["l1_ratio"])

                for f in range(n_folds):
                    scores_tensor[f, c_idx, l1_idx] = self.search.cv_results_[f"split{f}_test_score"][cand_idx]

            if n_l1_ratios == 1:
                scores_tensor = scores_tensor.squeeze(2)

            res[cls.item()] = scores_tensor
        return res

    @property
    def C_(self):
        if self.search.best_params_ is not None:
            best_C = self.search.best_params_["C"]
            n_classes = self.classes_.size(0)
            if n_classes <= 2:
                return torch.tensor([best_C], device=self.device, dtype=self.dtype)
            else:
                return torch.full((n_classes,), best_C, device=self.device, dtype=self.dtype)
        return None

    @property
    def l1_ratio_(self):
        if self.search.best_params_ is not None:
            best_l1 = self.search.best_params_["l1_ratio"]
            n_classes = self.classes_.size(0)
            if n_classes <= 2:
                return torch.tensor([best_l1], device=self.device, dtype=self.dtype)
            else:
                return torch.full((n_classes,), best_l1, device=self.device, dtype=self.dtype)
        return None

    @property
    def n_iter_(self):
        return self._classifier.n_iter_

    @property
    def n_features_in_(self):
        return self._classifier.n_features_in_

class RidgeClassifier(MLClassifier):
    def __init__(self,
                 alpha: Union[float, List[float], Tuple[float], torch.Tensor] = 1.0,
                 fit_intercept: bool = True,
                 copy_X: bool = True,
                 max_iter: int = None,
                 tol: float = 1e-6,
                 solver: str = "auto",
                 class_weight: dict = None,
                 n_jobs: int = None,
                 positive: bool = False,
                 random_state: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.ridge = Ridge(
            alpha=alpha,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            max_iter=max_iter,
            tol=tol,
            solver=solver,
            n_jobs=n_jobs,
            positive=positive,
            random_state=random_state,
            device=device,
            dtype=dtype,
            *args, **kwargs
        )
        self.class_weight = class_weight
        self.device = device
        self.dtype = dtype
        self._classes = None
        self.num_classes = None
        self.fit_status = False

    def _init_module_(self, X: torch.Tensor, y: torch.Tensor, classes: torch.Tensor = None):
        n_obs = X.size(0)
        if classes is not None:
            self._classes = classes
        else:
            self._classes, counts = torch.unique(y, return_counts=True)

        self.num_classes = self._classes.size(0)
        self.in_features = X.size(-1)

        device = X.device
        dtype = X.dtype
        # Class Weights
        if self.class_weight is None:
            self.class_weight_tensor = torch.ones((self.num_classes,), device=device, dtype=dtype)
        elif isinstance(self.class_weight, str) and self.class_weight.lower() == "balanced":
            counts = torch.bincount(y.long(), minlength=self.num_classes)
            self.class_weight_tensor = n_obs / (self.num_classes * counts.to(dtype))
        elif isinstance(self.class_weight, dict):
            weights_list = [float(self.class_weight.get(label.item(), 1.0)) for label in self._classes]
            self.class_weight_tensor = torch.tensor(weights_list, device=device, dtype=dtype)

        if self.class_weight_tensor is not None:
            self.class_weight_tensor = self.class_weight_tensor / self.class_weight_tensor.sum() * self.num_classes

        # Initialize ridge regressor with correct target shape
        y_dummy = torch.zeros((n_obs, self.num_classes), device=device, dtype=dtype)
        self.ridge._init_module(X, y_dummy)
        return self

    def fit(self, data_or_X, y=None, **kwargs):
        # Determine target device and dtype from input data
        target_device = data_or_X.device if isinstance(data_or_X, torch.Tensor) else self.device
        target_dtype = data_or_X.dtype if isinstance(data_or_X, torch.Tensor) else self.dtype

        X = torch.as_tensor(data_or_X, device=target_device, dtype=target_dtype)
        y = torch.as_tensor(y, device=target_device, dtype=target_dtype)
        X = X.reshape(-1, X.size(-1))
        y = y.reshape(-1)
        self._init_module_(X, y, kwargs.get("classes", None))

        # Calculate sample_weight from class_weight_tensor
        sample_weight = kwargs.get("sample_weight", None)
        if sample_weight is None and hasattr(self, "class_weight_tensor") and self.class_weight_tensor is not None:
            label_to_idx = {val.item(): i for i, val in enumerate(self._classes)}
            y_indices = torch.tensor([label_to_idx[val.item()] for val in y], device=target_device)
            sample_weight = self.class_weight_tensor[y_indices]
            kwargs["sample_weight"] = sample_weight

        y_hot = F.one_hot(y.long(), num_classes=self.num_classes).to(target_dtype)
        self.ridge.fit(X, y_hot, **kwargs)
        self.fit_status = True

        # Update internal tracking with effective device/dtype
        self.device = str(target_device)
        self.dtype = target_dtype
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(device=self.device, dtype=self.dtype)
        scores = self.ridge.predict(X)
        scores = F.softmax(scores, dim=-1)
        indices = torch.argmax(scores, dim=-1)
        return self._classes[indices]

    def decision_function(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(device=self.device, dtype=self.dtype)
        return self.ridge.predict(X)

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(device=self.device, dtype=self.dtype)
        return F.softmax(self.ridge.predict(X), dim=-1)

    def forward(self, X: torch.Tensor, y: torch.Tensor = None, **kwargs):
        if not self.fit_status:
            self.fit(X, y, **kwargs)
        return self.predict(X)

    @property
    def coef_(self):
        return self.ridge.coef_

    @property
    def intercept_(self):
        return self.ridge.intercept_

    @property
    def n_iter_(self):
        return self.ridge.n_iter_

    @property
    def classes_(self):
        return self._classes

    @property
    def n_features_in_(self):
        return self.ridge.n_features_in_

    @property
    def solver_(self):
        return self.ridge.solver

class LassoClassifier(MLClassifier):
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
                 class_weight: dict = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.lasso = Lasso(
            alpha=alpha,
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
        self.class_weight = class_weight
        self.device = device
        self.dtype = dtype
        self._classes = None
        self.num_classes = None
        self.fit_status = False

    def _init_module_(self, X: torch.Tensor, y: torch.Tensor, classes: torch.Tensor = None):
        n_obs = X.size(0)
        if classes is not None:
            self._classes = classes
        else:
            self._classes, counts = torch.unique(y, return_counts=True)

        self.num_classes = self._classes.size(0)
        self.in_features = X.size(-1)

        device = X.device
        dtype = X.dtype
        # Class Weights
        if self.class_weight is None:
            self.class_weight_tensor = torch.ones((self.num_classes,), device=device, dtype=dtype)
        elif isinstance(self.class_weight, str) and self.class_weight.lower() == "balanced":
            counts = torch.bincount(y.long(), minlength=self.num_classes)
            self.class_weight_tensor = n_obs / (self.num_classes * counts.to(dtype))
        elif isinstance(self.class_weight, dict):
            weights_list = [float(self.class_weight.get(label.item(), 1.0)) for label in self._classes]
            self.class_weight_tensor = torch.tensor(weights_list, device=device, dtype=dtype)

        if self.class_weight_tensor is not None:
            self.class_weight_tensor = self.class_weight_tensor / self.class_weight_tensor.sum() * self.num_classes

        # Initialize lasso regressor with correct target shape
        y_dummy = torch.zeros((n_obs, self.num_classes), device=device, dtype=dtype)
        self.lasso._init_module(X, y_dummy)
        return self

    def fit(self, data_or_X, y=None, **kwargs):
        # Determine target device and dtype from input data
        target_device = data_or_X.device if isinstance(data_or_X, torch.Tensor) else self.device
        target_dtype = data_or_X.dtype if isinstance(data_or_X, torch.Tensor) else self.dtype

        # Determine target device and dtype from input data
        
        
        
        X = torch.as_tensor(data_or_X, device=target_device, dtype=target_dtype)
        y = torch.as_tensor(y, device=target_device, dtype=target_dtype)
        X = X.reshape(-1, X.size(-1))
        y = y.reshape(-1)
        self._init_module_(X, y, kwargs.get("classes", None))

        # Calculate sample_weight from class_weight_tensor
        sample_weight = kwargs.get("sample_weight", None)
        if sample_weight is None and hasattr(self, "class_weight_tensor") and self.class_weight_tensor is not None:
            label_to_idx = {val.item(): i for i, val in enumerate(self._classes)}
            y_indices = torch.tensor([label_to_idx[val.item()] for val in y], device=target_device)
            sample_weight = self.class_weight_tensor[y_indices]
            kwargs["sample_weight"] = sample_weight

        y_hot = F.one_hot(y.long(), num_classes=self.num_classes).to(target_dtype)
        self.lasso.fit(X, y_hot, **kwargs)
        self.fit_status = True

        # Update internal tracking
        self.device = str(target_device)
        self.dtype = target_dtype
        # Update internal device/dtype tracking
        self.device = str(target_device)
        self.dtype = target_dtype
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(device=self.device, dtype=self.dtype)
        scores = self.lasso.predict(X)
        scores = F.softmax(scores, dim=-1)
        indices = torch.argmax(scores, dim=-1)
        return self._classes[indices]

    def decision_function(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(device=self.device, dtype=self.dtype)
        return self.lasso.predict(X)

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(device=self.device, dtype=self.dtype)
        return F.softmax(self.lasso.predict(X), dim=-1)

    def forward(self, X: torch.Tensor, y: torch.Tensor = None, **kwargs):
        if not self.fit_status:
            self.fit(X, y, **kwargs)
        return self.predict(X)

    @property
    def coef_(self):
        return self.lasso.coef_

    @property
    def intercept_(self):
        return self.lasso.intercept_

    @property
    def n_iter_(self):
        return self.lasso.n_iter_

    @property
    def classes_(self):
        return self._classes

    @property
    def n_features_in_(self):
        return self.lasso.n_features_in_

    @property
    def solver_(self):
        return self.lasso.solver

class ElasticNetClassifier(MLClassifier):
    def __init__(self,
                 alpha: Union[float, List[float], Tuple[float], torch.Tensor] = 1.0,
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
                 class_weight: dict = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.elastic_net = ElasticNet(
            alpha=alpha,
            l1_norm=l1_norm,
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
        self.class_weight = class_weight
        self.device = device
        self.dtype = dtype
        self._classes = None
        self.num_classes = None
        self.fit_status = False

    def _init_module_(self, X: torch.Tensor, y: torch.Tensor, classes: torch.Tensor = None):
        n_obs = X.size(0)
        if classes is not None:
            self._classes = classes
        else:
            self._classes, counts = torch.unique(y, return_counts=True)

        self.num_classes = self._classes.size(0)
        self.in_features = X.size(-1)

        device = X.device
        dtype = X.dtype
        # Class Weights
        if self.class_weight is None:
            self.class_weight_tensor = torch.ones((self.num_classes,), device=device, dtype=dtype)
        elif isinstance(self.class_weight, str) and self.class_weight.lower() == "balanced":
            counts = torch.bincount(y.long(), minlength=self.num_classes)
            self.class_weight_tensor = n_obs / (self.num_classes * counts.to(dtype))
        elif isinstance(self.class_weight, dict):
            weights_list = [float(self.class_weight.get(label.item(), 1.0)) for label in self._classes]
            self.class_weight_tensor = torch.tensor(weights_list, device=device, dtype=dtype)

        if self.class_weight_tensor is not None:
            self.class_weight_tensor = self.class_weight_tensor / self.class_weight_tensor.sum() * self.num_classes

        # Initialize elastic net regressor with correct target shape
        y_dummy = torch.zeros((n_obs, self.num_classes), device=device, dtype=dtype)
        self.elastic_net._init_module(X, y_dummy)
        return self

    def fit(self, data_or_X, y=None, **kwargs):
        # Determine target device and dtype from input data
        target_device = data_or_X.device if isinstance(data_or_X, torch.Tensor) else self.device
        target_dtype = data_or_X.dtype if isinstance(data_or_X, torch.Tensor) else self.dtype

        # Determine target device and dtype from input data
        
        
        
        X = torch.as_tensor(data_or_X, device=target_device, dtype=target_dtype)
        y = torch.as_tensor(y, device=target_device, dtype=target_dtype)
        X = X.reshape(-1, X.size(-1))
        y = y.reshape(-1)
        self._init_module_(X, y, kwargs.get("classes", None))

        # Calculate sample_weight from class_weight_tensor
        sample_weight = kwargs.get("sample_weight", None)
        if sample_weight is None and hasattr(self, "class_weight_tensor") and self.class_weight_tensor is not None:
            label_to_idx = {val.item(): i for i, val in enumerate(self._classes)}
            y_indices = torch.tensor([label_to_idx[val.item()] for val in y], device=target_device)
            sample_weight = self.class_weight_tensor[y_indices]
            kwargs["sample_weight"] = sample_weight

        y_hot = F.one_hot(y.long(), num_classes=self.num_classes).to(target_dtype)
        self.elastic_net.fit(X, y_hot, **kwargs)
        self.fit_status = True

        # Update internal tracking
        self.device = str(target_device)
        self.dtype = target_dtype
        # Update internal device/dtype tracking
        self.device = str(target_device)
        self.dtype = target_dtype
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(device=self.device, dtype=self.dtype)
        scores = self.elastic_net.predict(X)
        scores = F.softmax(scores, dim=-1)
        indices = torch.argmax(scores, dim=-1)
        return self._classes[indices]

    def decision_function(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(device=self.device, dtype=self.dtype)
        return self.elastic_net.predict(X)

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(device=self.device, dtype=self.dtype)
        return F.softmax(self.elastic_net.predict(X), dim=-1)

    def forward(self, X: torch.Tensor, y: torch.Tensor = None, **kwargs):
        if not self.fit_status:
            self.fit(X, y, **kwargs)
        return self.predict(X)

    @property
    def coef_(self):
        return self.elastic_net.coef_

    @property
    def intercept_(self):
        return self.elastic_net.intercept_

    @property
    def n_iter_(self):
        return self.elastic_net.n_iter_

    @property
    def classes_(self):
        return self._classes

    @property
    def n_features_in_(self):
        return self.elastic_net.n_features_in_

    @property
    def solver_(self):
        return self.elastic_net.solver

class LarsClassifier(MLClassifier):
    def __init__(self,
                 fit_intercept: bool = True,
                 verbose: Union[bool, int] = False,
                 precompute: Union[bool, str, List, Tuple, torch.Tensor] = "auto",
                 n_nonzero_coefs: int = 500,
                 eps: float = torch.finfo(torch.float32).eps,
                 fit_path: bool = True,
                 max_iter: int = None,
                 jitter: float = None,
                 tol: float = None,
                 copy_X: bool = True,
                 positive: bool = False,
                 random_state: int = None,
                 n_jobs: int = None,
                 class_weight: dict = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.lars = Lars(
            fit_intercept=fit_intercept,
            verbose=verbose,
            precompute=precompute,
            n_nonzero_coefs=n_nonzero_coefs,
            eps=eps,
            fit_path=fit_path,
            max_iter=max_iter,
            jitter=jitter,
            tol=tol,
            copy_X=copy_X,
            positive=positive,
            random_state=random_state,
            n_jobs=n_jobs,
            device=device,
            dtype=dtype,
            *args, **kwargs
        )
        self.class_weight = class_weight
        self.device = device
        self.dtype = dtype
        self._classes = None
        self.num_classes = None
        self.fit_status = False

    def _init_module_(self, X: torch.Tensor, y: torch.Tensor, classes: torch.Tensor = None):
        n_obs = X.size(0)
        if classes is not None:
            self._classes = classes
        else:
            self._classes, counts = torch.unique(y, return_counts=True)

        self.num_classes = self._classes.size(0)
        self.in_features = X.size(-1)

        device = X.device
        dtype = X.dtype
        # Class Weights
        if self.class_weight is None:
            self.class_weight_tensor = torch.ones((self.num_classes,), device=device, dtype=dtype)
        elif isinstance(self.class_weight, str) and self.class_weight.lower() == "balanced":
            counts = torch.bincount(y.long(), minlength=self.num_classes)
            self.class_weight_tensor = n_obs / (self.num_classes * counts.to(dtype))
        elif isinstance(self.class_weight, dict):
            weights_list = [float(self.class_weight.get(label.item(), 1.0)) for label in self._classes]
            self.class_weight_tensor = torch.tensor(weights_list, device=device, dtype=dtype)

        if self.class_weight_tensor is not None:
            self.class_weight_tensor = self.class_weight_tensor / self.class_weight_tensor.sum() * self.num_classes

        # Initialize lars regressor with correct target shape
        y_dummy = torch.zeros((n_obs, self.num_classes), device=device, dtype=dtype)
        self.lars._init_module(X, y_dummy)
        return self

    def fit(self, data_or_X, y=None, **kwargs):
        # Determine target device and dtype from input data
        target_device = data_or_X.device if isinstance(data_or_X, torch.Tensor) else self.device
        target_dtype = data_or_X.dtype if isinstance(data_or_X, torch.Tensor) else self.dtype

        # Determine target device and dtype from input data
        
        
        
        X = torch.as_tensor(data_or_X, device=target_device, dtype=target_dtype)
        y = torch.as_tensor(y, device=target_device, dtype=target_dtype)
        X = X.reshape(-1, X.size(-1))
        y = y.reshape(-1)
        self._init_module_(X, y, kwargs.get("classes", None))

        # Calculate sample_weight from class_weight_tensor
        sample_weight = kwargs.get("sample_weight", None)
        if sample_weight is None and hasattr(self, "class_weight_tensor") and self.class_weight_tensor is not None:
            label_to_idx = {val.item(): i for i, val in enumerate(self._classes)}
            y_indices = torch.tensor([label_to_idx[val.item()] for val in y], device=target_device)
            sample_weight = self.class_weight_tensor[y_indices]
            kwargs["sample_weight"] = sample_weight

        y_hot = F.one_hot(y.long(), num_classes=self.num_classes).to(target_dtype)
        self.lars.fit(X, y_hot, **kwargs)
        self.fit_status = True

        # Update internal tracking
        self.device = str(target_device)
        self.dtype = target_dtype
        # Update internal device/dtype tracking
        self.device = str(target_device)
        self.dtype = target_dtype
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(device=self.device, dtype=self.dtype)
        scores = self.lars.predict(X)
        scores = F.softmax(scores, dim=-1)
        indices = torch.argmax(scores, dim=-1)
        return self._classes[indices]

    def decision_function(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(device=self.device, dtype=self.dtype)
        return self.lars.predict(X)

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(device=self.device, dtype=self.dtype)
        return F.softmax(self.lars.predict(X), dim=-1)

    def forward(self, X: torch.Tensor, y: torch.Tensor = None, **kwargs):
        if not self.fit_status:
            self.fit(X, y, **kwargs)
        return self.predict(X)

    @property
    def coef_(self):
        return self.lars.coef_

    @property
    def intercept_(self):
        return self.lars.intercept_

    @property
    def n_iter_(self):
        return self.lars.n_iter_

    @property
    def classes_(self):
        return self._classes

    @property
    def n_features_in_(self):
        return self.lars.n_features_in_

class RidgeLarsClassifier(MLClassifier):
    def __init__(self,
                 fit_intercept: bool = True,
                 alpha: float = 1.0,
                 verbose: Union[bool, int] = False,
                 precompute: Union[bool, str, List, Tuple, torch.Tensor] = "auto",
                 n_nonzero_coefs: int = 500,
                 eps: float = torch.finfo(torch.float32).eps,
                 fit_path: bool = True,
                 max_iter: int = None,
                 jitter: float = None,
                 tol: float = None,
                 copy_X: bool = True,
                 positive: bool = False,
                 random_state: int = None,
                 n_jobs: int = None,
                 class_weight: dict = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.ridge_lars = RidgeLars(
            fit_intercept=fit_intercept,
            alpha=alpha,
            verbose=verbose,
            precompute=precompute,
            n_nonzero_coefs=n_nonzero_coefs,
            eps=eps,
            fit_path=fit_path,
            max_iter=max_iter,
            jitter=jitter,
            tol=tol,
            copy_X=copy_X,
            positive=positive,
            random_state=random_state,
            n_jobs=n_jobs,
            device=device,
            dtype=dtype,
            *args, **kwargs
        )
        self.class_weight = class_weight
        self.device = device
        self.dtype = dtype
        self._classes = None
        self.num_classes = None
        self.fit_status = False

    def _init_module_(self, X: torch.Tensor, y: torch.Tensor, classes: torch.Tensor = None):
        n_obs = X.size(0)
        if classes is not None:
            self._classes = classes
        else:
            self._classes, counts = torch.unique(y, return_counts=True)

        self.num_classes = self._classes.size(0)
        self.in_features = X.size(-1)

        device = X.device
        dtype = X.dtype
        # Class Weights
        if self.class_weight is None:
            self.class_weight_tensor = torch.ones((self.num_classes,), device=device, dtype=dtype)
        elif isinstance(self.class_weight, str) and self.class_weight.lower() == "balanced":
            counts = torch.bincount(y.long(), minlength=self.num_classes)
            self.class_weight_tensor = n_obs / (self.num_classes * counts.to(dtype))
        elif isinstance(self.class_weight, dict):
            weights_list = [float(self.class_weight.get(label.item(), 1.0)) for label in self._classes]
            self.class_weight_tensor = torch.tensor(weights_list, device=device, dtype=dtype)

        if self.class_weight_tensor is not None:
            self.class_weight_tensor = self.class_weight_tensor / self.class_weight_tensor.sum() * self.num_classes

        # Initialize ridge lars regressor with correct target shape
        y_dummy = torch.zeros((n_obs, self.num_classes), device=device, dtype=dtype)
        self.ridge_lars._init_module(X, y_dummy)
        return self

    def fit(self, data_or_X, y=None, **kwargs):
        # Determine target device and dtype from input data
        target_device = data_or_X.device if isinstance(data_or_X, torch.Tensor) else self.device
        target_dtype = data_or_X.dtype if isinstance(data_or_X, torch.Tensor) else self.dtype

        # Determine target device and dtype from input data
        
        
        
        X = torch.as_tensor(data_or_X, device=target_device, dtype=target_dtype)
        y = torch.as_tensor(y, device=target_device, dtype=target_dtype)
        X = X.reshape(-1, X.size(-1))
        y = y.reshape(-1)
        self._init_module_(X, y, kwargs.get("classes", None))

        # Calculate sample_weight from class_weight_tensor
        sample_weight = kwargs.get("sample_weight", None)
        if sample_weight is None and hasattr(self, "class_weight_tensor") and self.class_weight_tensor is not None:
            label_to_idx = {val.item(): i for i, val in enumerate(self._classes)}
            y_indices = torch.tensor([label_to_idx[val.item()] for val in y], device=target_device)
            sample_weight = self.class_weight_tensor[y_indices]
            kwargs["sample_weight"] = sample_weight

        y_hot = F.one_hot(y.long(), num_classes=self.num_classes).to(target_dtype)
        self.ridge_lars.fit(X, y_hot, **kwargs)
        self.fit_status = True

        # Update internal tracking
        self.device = str(target_device)
        self.dtype = target_dtype
        # Update internal device/dtype tracking
        self.device = str(target_device)
        self.dtype = target_dtype
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(device=self.device, dtype=self.dtype)
        scores = self.ridge_lars.predict(X)
        scores = F.softmax(scores, dim=-1)
        indices = torch.argmax(scores, dim=-1)
        return self._classes[indices]

    def decision_function(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(device=self.device, dtype=self.dtype)
        return self.ridge_lars.predict(X)

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(device=self.device, dtype=self.dtype)
        return F.softmax(self.ridge_lars.predict(X), dim=-1)

    def forward(self, X: torch.Tensor, y: torch.Tensor = None, **kwargs):
        if not self.fit_status:
            self.fit(X, y, **kwargs)
        return self.predict(X)

    @property
    def coef_(self):
        return self.ridge_lars.coef_

    @property
    def intercept_(self):
        return self.ridge_lars.intercept_

    @property
    def n_iter_(self):
        return self.ridge_lars.n_iter_

    @property
    def classes_(self):
        return self._classes

    @property
    def n_features_in_(self):
        return self.ridge_lars.n_features_in_

class LassoLarsClassifier(MLClassifier):
    def __init__(self,
                 fit_intercept: bool = True,
                 alpha: float = 1.0,
                 verbose: Union[bool, int] = False,
                 precompute: Union[bool, str, List, Tuple, torch.Tensor] = "auto",
                 n_nonzero_coefs: int = 500,
                 eps: float = torch.finfo(torch.float32).eps,
                 fit_path: bool = True,
                 max_iter: int = None,
                 jitter: float = None,
                 tol: float = None,
                 copy_X: bool = False,
                 positive: bool = False,
                 random_state: int = None,
                 n_jobs: int = None,
                 class_weight: dict = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.lasso_lars = LassoLars(
            fit_intercept=fit_intercept,
            alpha=alpha,
            verbose=verbose,
            precompute=precompute,
            n_nonzero_coefs=n_nonzero_coefs,
            eps=eps,
            fit_path=fit_path,
            max_iter=max_iter,
            jitter=jitter,
            tol=tol,
            copy_X=copy_X,
            positive=positive,
            random_state=random_state,
            n_jobs=n_jobs,
            device=device,
            dtype=dtype,
            *args, **kwargs
        )
        self.class_weight = class_weight
        self.device = device
        self.dtype = dtype
        self._classes = None
        self.num_classes = None
        self.fit_status = False

    def _init_module_(self, X: torch.Tensor, y: torch.Tensor, classes: torch.Tensor = None):
        n_obs = X.size(0)
        if classes is not None:
            self._classes = classes
        else:
            self._classes, counts = torch.unique(y, return_counts=True)

        self.num_classes = self._classes.size(0)
        self.in_features = X.size(-1)

        device = X.device
        dtype = X.dtype
        # Class Weights
        if self.class_weight is None:
            self.class_weight_tensor = torch.ones((self.num_classes,), device=device, dtype=dtype)
        elif isinstance(self.class_weight, str) and self.class_weight.lower() == "balanced":
            counts = torch.bincount(y.long(), minlength=self.num_classes)
            self.class_weight_tensor = n_obs / (self.num_classes * counts.to(dtype))
        elif isinstance(self.class_weight, dict):
            weights_list = [float(self.class_weight.get(label.item(), 1.0)) for label in self._classes]
            self.class_weight_tensor = torch.tensor(weights_list, device=device, dtype=dtype)

        if self.class_weight_tensor is not None:
            self.class_weight_tensor = self.class_weight_tensor / self.class_weight_tensor.sum() * self.num_classes

        # Initialize lasso lars regressor with correct target shape
        y_dummy = torch.zeros((n_obs, self.num_classes), device=device, dtype=dtype)
        self.lasso_lars._init_module(X, y_dummy)
        return self

    def fit(self, data_or_X, y=None, **kwargs):
        # Determine target device and dtype from input data
        target_device = data_or_X.device if isinstance(data_or_X, torch.Tensor) else self.device
        target_dtype = data_or_X.dtype if isinstance(data_or_X, torch.Tensor) else self.dtype

        # Determine target device and dtype from input data
        
        
        
        X = torch.as_tensor(data_or_X, device=target_device, dtype=target_dtype)
        y = torch.as_tensor(y, device=target_device, dtype=target_dtype)
        X = X.reshape(-1, X.size(-1))
        y = y.reshape(-1)
        self._init_module_(X, y, kwargs.get("classes", None))

        # Calculate sample_weight from class_weight_tensor
        sample_weight = kwargs.get("sample_weight", None)
        if sample_weight is None and hasattr(self, "class_weight_tensor") and self.class_weight_tensor is not None:
            label_to_idx = {val.item(): i for i, val in enumerate(self._classes)}
            y_indices = torch.tensor([label_to_idx[val.item()] for val in y], device=target_device)
            sample_weight = self.class_weight_tensor[y_indices]
            kwargs["sample_weight"] = sample_weight

        y_hot = F.one_hot(y.long(), num_classes=self.num_classes).to(target_dtype)
        self.lasso_lars.fit(X, y_hot, **kwargs)
        self.fit_status = True

        # Update internal tracking
        self.device = str(target_device)
        self.dtype = target_dtype
        # Update internal device/dtype tracking
        self.device = str(target_device)
        self.dtype = target_dtype
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(device=self.device, dtype=self.dtype)
        scores = self.lasso_lars.predict(X)
        scores = F.softmax(scores, dim=-1)
        indices = torch.argmax(scores, dim=-1)
        return self._classes[indices]

    def decision_function(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(device=self.device, dtype=self.dtype)
        return self.lasso_lars.predict(X)

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(device=self.device, dtype=self.dtype)
        return F.softmax(self.lasso_lars.predict(X), dim=-1)

    def forward(self, X: torch.Tensor, y: torch.Tensor = None, **kwargs):
        if not self.fit_status:
            self.fit(X, y, **kwargs)
        return self.predict(X)

    @property
    def coef_(self):
        return self.lasso_lars.coef_

    @property
    def intercept_(self):
        return self.lasso_lars.intercept_

    @property
    def n_iter_(self):
        return self.lasso_lars.n_iter_

    @property
    def classes_(self):
        return self._classes

    @property
    def n_features_in_(self):
        return self.lasso_lars.n_features_in_

class ElasticNetLarsClassifier(MLClassifier):
    def __init__(self,
                 fit_intercept: bool = True,
                 alpha: float = 1.0,
                 l1_ratio: float = 0.5,
                 verbose: Union[bool, int] = False,
                 precompute: Union[bool, str, List, Tuple, torch.Tensor] = "auto",
                 n_nonzero_coefs: int = 500,
                 eps: float = torch.finfo(torch.float32).eps,
                 fit_path: bool = True,
                 max_iter: int = None,
                 jitter: float = None,
                 tol: float = None,
                 copy_X: bool = False,
                 positive: bool = False,
                 random_state: int = None,
                 n_jobs: int = None,
                 class_weight: dict = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.elastic_net_lars = ElasticNetLars(
            fit_intercept=fit_intercept,
            alpha=alpha,
            l1_ratio=l1_ratio,
            verbose=verbose,
            precompute=precompute,
            n_nonzero_coefs=n_nonzero_coefs,
            eps=eps,
            fit_path=fit_path,
            max_iter=max_iter,
            jitter=jitter,
            tol=tol,
            copy_X=copy_X,
            positive=positive,
            random_state=random_state,
            n_jobs=n_jobs,
            device=device,
            dtype=dtype,
            *args, **kwargs
        )
        self.class_weight = class_weight
        self.device = device
        self.dtype = dtype
        self._classes = None
        self.num_classes = None
        self.fit_status = False

    def _init_module_(self, X: torch.Tensor, y: torch.Tensor, classes: torch.Tensor = None):
        n_obs = X.size(0)
        if classes is not None:
            self._classes = classes
        else:
            self._classes, counts = torch.unique(y, return_counts=True)

        self.num_classes = self._classes.size(0)
        self.in_features = X.size(-1)

        device = X.device
        dtype = X.dtype
        # Class Weights
        if self.class_weight is None:
            self.class_weight_tensor = torch.ones((self.num_classes,), device=device, dtype=dtype)
        elif isinstance(self.class_weight, str) and self.class_weight.lower() == "balanced":
            counts = torch.bincount(y.long(), minlength=self.num_classes)
            self.class_weight_tensor = n_obs / (self.num_classes * counts.to(dtype))
        elif isinstance(self.class_weight, dict):
            weights_list = [float(self.class_weight.get(label.item(), 1.0)) for label in self._classes]
            self.class_weight_tensor = torch.tensor(weights_list, device=device, dtype=dtype)

        if self.class_weight_tensor is not None:
            self.class_weight_tensor = self.class_weight_tensor / self.class_weight_tensor.sum() * self.num_classes

        # Initialize elastic net lars regressor with correct target shape
        y_dummy = torch.zeros((n_obs, self.num_classes), device=device, dtype=dtype)
        self.elastic_net_lars._init_module(X, y_dummy)
        return self

    def fit(self, data_or_X, y=None, **kwargs):
        # Determine target device and dtype from input data
        target_device = data_or_X.device if isinstance(data_or_X, torch.Tensor) else self.device
        target_dtype = data_or_X.dtype if isinstance(data_or_X, torch.Tensor) else self.dtype

        # Determine target device and dtype from input data
        
        
        
        X = torch.as_tensor(data_or_X, device=target_device, dtype=target_dtype)
        y = torch.as_tensor(y, device=target_device, dtype=target_dtype)
        X = X.reshape(-1, X.size(-1))
        y = y.reshape(-1)
        self._init_module_(X, y, kwargs.get("classes", None))

        # Calculate sample_weight from class_weight_tensor
        sample_weight = kwargs.get("sample_weight", None)
        if sample_weight is None and hasattr(self, "class_weight_tensor") and self.class_weight_tensor is not None:
            label_to_idx = {val.item(): i for i, val in enumerate(self._classes)}
            y_indices = torch.tensor([label_to_idx[val.item()] for val in y], device=target_device)
            sample_weight = self.class_weight_tensor[y_indices]
            kwargs["sample_weight"] = sample_weight

        y_hot = F.one_hot(y.long(), num_classes=self.num_classes).to(target_dtype)
        self.elastic_net_lars.fit(X, y_hot, **kwargs)
        self.fit_status = True

        # Update internal tracking
        self.device = str(target_device)
        self.dtype = target_dtype
        # Update internal device/dtype tracking
        self.device = str(target_device)
        self.dtype = target_dtype
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(device=self.device, dtype=self.dtype)
        scores = self.elastic_net_lars.predict(X)
        scores = F.softmax(scores, dim=-1)
        indices = torch.argmax(scores, dim=-1)
        return self._classes[indices]

    def decision_function(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(device=self.device, dtype=self.dtype)
        return self.elastic_net_lars.predict(X)

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(device=self.device, dtype=self.dtype)
        return F.softmax(self.elastic_net_lars.predict(X), dim=-1)

    def forward(self, X: torch.Tensor, y: torch.Tensor = None, **kwargs):
        if not self.fit_status:
            self.fit(X, y, **kwargs)
        return self.predict(X)

    @property
    def coef_(self):
        return self.elastic_net_lars.coef_

    @property
    def intercept_(self):
        return self.elastic_net_lars.intercept_

    @property
    def n_iter_(self):
        return self.elastic_net_lars.n_iter_

    @property
    def classes_(self):
        return self._classes

    @property
    def n_features_in_(self):
        return self.elastic_net_lars.n_features_in_

class RidgeClassifierCV(MLClassifier):
    def __init__(self,
                 alphas: Union[int, float, List[float], Tuple[float], torch.Tensor] = 1.0,
                 fit_intercept: bool = True,
                 copy_X: bool = True,
                 max_iter: int = None,
                 tol: float = 1e-6,
                 refit: bool = True,
                 cv: Union[str, int, Callable, Iterable, nn.Module] = None,
                 cv_config: dict = None,
                 scoring: Union[str, Callable] = None,
                 store_cv_values: bool = False,
                 solver: Union[str, List[str], Tuple[str]] = "auto",
                 class_weights: Union[dict, List[dict], Tuple[dict]] = None,
                 n_jobs: int = None,
                 positive: bool = False,
                 random_state: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        if isinstance(alphas, int):
            alphas = [0.5] * alphas
        elif isinstance(alphas, float):
            alphas = [alphas]
        elif isinstance(alphas, torch.Tensor):
            alphas = alphas.tolist()
        if isinstance(solver, str):
            solver = [solver]
        if class_weights is None:
            class_weights = [None]
        elif isinstance(class_weights, dict):
            class_weights = [class_weights]

        self._classifier = RidgeClassifier(
            alpha=alphas[0] if alphas else 1.0,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            max_iter=max_iter,
            tol=tol,
            solver=solver[0],
            class_weight=class_weights[0] if class_weights else None,
            n_jobs=n_jobs,
            positive=positive,
            random_state=random_state,
            device=device,
            dtype=dtype,
            *args, **kwargs
        )
        self.params_dict = {
            "alpha": alphas,
            "solver": solver,
            "class_weight": class_weights
        }
        self.search_params = {
            "estimator": self._classifier,
            "param_grid": self.params_dict,
            "scoring": scoring,
            "n_jobs": n_jobs,
            "refit": refit,
            "cv": cv,
            "cv_config": cv_config,
            "return_train_score": True,
            "store_cv_values": store_cv_values,
            "return_estimators": True,
            "device": device,
            "dtype": dtype
        }
        self.device = device
        self.dtype = dtype
        from ...cross_validation import GridSearchCV
        self._search = GridSearchCV(**self.search_params)
        self.fit_status = False
        self.cv_results_ = None
        self.coef_ = None
        self.intercept_ = None
        self.best_score_ = None
        self.classes_ = None
        self.alpha_ = None
        self.n_features_in_ = None

    def fit(self, data_or_X, y=None, **kwargs):
        # Determine target device and dtype from input data
        target_device = data_or_X.device if isinstance(data_or_X, torch.Tensor) else self.device
        target_dtype = data_or_X.dtype if isinstance(data_or_X, torch.Tensor) else self.dtype

        # Determine target device and dtype from input data
        
        

        X = torch.as_tensor(data_or_X, device=target_device, dtype=target_dtype)
        y = torch.as_tensor(y, device=target_device, dtype=target_dtype)
        self._classifier._init_module_(X, y, kwargs.get("classes", None))
        self._search.fit(X, y, kwargs.get("groups", None))
        self._classifier = self._search.best_estimator_
        self.fit_status = True

        # Update internal tracking
        self.device = str(target_device)
        self.dtype = target_dtype
        self.cv_results_ = self._search.cv_results_
        self.coef_ = self._classifier.coef_
        self.intercept_ = self._classifier.intercept_
        self.best_score_ = self._search.best_score_
        self.classes_ = self._classifier.classes_
        self.alpha_ = self._search.best_params_.get("alpha")
        self.n_features_in_ = self._classifier.n_features_in_
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self._classifier.predict(X)

    def decision_function(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(device=self.device, dtype=self.dtype)
        return self._classifier.decision_function(X)

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(device=self.device, dtype=self.dtype)
        return self._classifier.predict_proba(X)

    def forward(self, X: torch.Tensor, y: torch.Tensor = None, **kwargs):
        if not self.fit_status:
            self.fit(X, y, **kwargs)
        return self.predict(X)

class LassoClassifierCV(MLClassifier):
    def __init__(self,
                 alphas: Union[float, List[float], Tuple[float], torch.Tensor] = None,
                 fit_intercept: bool = True,
                 precompute: Union[bool, List[list], Tuple[tuple], torch.Tensor] = False,
                 copy_X: bool = True,
                 max_iter: int = None,
                 tol: float = 1e-6,
                 warm_start: bool = False,
                 positive: bool = False,
                 selection: Union[str, List[str]] = "cyclic",
                 class_weights: Union[dict, List[dict], Tuple[dict]] = None,
                 cv: Union[str, int, Callable, Iterable, nn.Module] = None,
                 cv_config: dict = None,
                 scoring: Union[str, Callable] = None,
                 n_jobs: int = None,
                 random_state: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        if alphas is None:
            alphas = [0.1, 1.0, 10.0]
        if isinstance(alphas, float):
            alphas = [alphas]
        if isinstance(selection, str):
            selection = [selection]
        if class_weights is None:
            class_weights = [None]
        elif isinstance(class_weights, dict):
            class_weights = [class_weights]

        self._classifier = LassoClassifier(
            alpha=alphas[0],
            fit_intercept=fit_intercept,
            precompute=precompute,
            copy_X=copy_X,
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            positive=positive,
            selection=selection[0],
            class_weight=class_weights[0] if class_weights else None,
            random_state=random_state,
            device=device,
            dtype=dtype,
            *args, **kwargs
        )
        self.params_dict = {
            "alpha": alphas,
            "selection": selection,
            "class_weight": class_weights
        }
        from ...cross_validation import GridSearchCV
        self._search = GridSearchCV(
            estimator=self._classifier,
            param_grid=self.params_dict,
            scoring=scoring,
            cv=cv,
            cv_config=cv_config,
            n_jobs=n_jobs,
            device=device,
            dtype=dtype
        )
        self.device = device
        self.dtype = dtype
        self.fit_status = False

    def fit(self, data_or_X, y=None, **kwargs):
        # Determine target device and dtype from input data
        target_device = data_or_X.device if isinstance(data_or_X, torch.Tensor) else self.device
        target_dtype = data_or_X.dtype if isinstance(data_or_X, torch.Tensor) else self.dtype

        # Determine target device and dtype from input data
        
        

        X = torch.as_tensor(data_or_X, device=target_device, dtype=target_dtype)
        y = torch.as_tensor(y, device=target_device, dtype=target_dtype)
        self._classifier._init_module_(X, y, kwargs.get("classes", None))
        self._search.fit(X, y, kwargs.get("groups", None))
        self._classifier = self._search.best_estimator_
        self.fit_status = True

        # Update internal tracking
        self.device = str(target_device)
        self.dtype = target_dtype
        self.cv_results_ = self._search.cv_results_
        self.best_score_ = self._search.best_score_
        self.best_params_ = self._search.best_params_
        self.alpha_ = self.best_params_.get("alpha")
        self.coef_ = self._classifier.coef_
        self.intercept_ = self._classifier.intercept_
        self.classes_ = self._classifier.classes_
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self._classifier.predict(X)

    def decision_function(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(device=self.device, dtype=self.dtype)
        return self._classifier.decision_function(X)

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(device=self.device, dtype=self.dtype)
        return self._classifier.predict_proba(X)

    def forward(self, X: torch.Tensor, y: torch.Tensor = None, **kwargs):
        if not self.fit_status:
            self.fit(X, y, **kwargs)
        return self.predict(X)

class ElasticNetClassifierCV(MLClassifier):
    def __init__(self,
                 alphas: Union[float, List[float], Tuple[float], torch.Tensor] = None,
                 l1_norms: Union[float, List[float], Tuple[float]] = None,
                 fit_intercept: bool = True,
                 precompute: Union[bool, List[list], Tuple[tuple], torch.Tensor] = False,
                 copy_X: bool = True,
                 max_iter: int = None,
                 tol: float = 1e-6,
                 warm_start: bool = False,
                 positive: bool = False,
                 selection: Union[str, List[str]] = "cyclic",
                 class_weights: Union[dict, List[dict], Tuple[dict]] = None,
                 cv: Union[str, int, Callable, Iterable, nn.Module] = None,
                 cv_config: dict = None,
                 scoring: Union[str, Callable] = None,
                 n_jobs: int = None,
                 random_state: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        if alphas is None:
            alphas = [0.1, 1.0, 10.0]
        if l1_norms is None:
            l1_norms = [0.1, 0.5, 0.9]
        if isinstance(alphas, float):
            alphas = [alphas]
        if isinstance(l1_norms, float):
            l1_norms = [l1_norms]
        if isinstance(selection, str):
            selection = [selection]
        if class_weights is None:
            class_weights = [None]
        elif isinstance(class_weights, dict):
            class_weights = [class_weights]

        self._classifier = ElasticNetClassifier(
            alpha=alphas[0],
            l1_norm=l1_norms[0],
            fit_intercept=fit_intercept,
            precompute=precompute,
            copy_X=copy_X,
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            positive=positive,
            selection=selection[0],
            class_weight=class_weights[0] if class_weights else None,
            random_state=random_state,
            device=device,
            dtype=dtype,
            *args, **kwargs
        )
        self.params_dict = {
            "alpha": alphas,
            "l1_norm": l1_norms,
            "selection": selection,
            "class_weight": class_weights
        }
        from ...cross_validation import GridSearchCV
        self._search = GridSearchCV(
            estimator=self._classifier,
            param_grid=self.params_dict,
            scoring=scoring,
            cv=cv,
            cv_config=cv_config,
            n_jobs=n_jobs,
            device=device,
            dtype=dtype
        )
        self.device = device
        self.dtype = dtype
        self.fit_status = False

    def fit(self, data_or_X, y=None, **kwargs):
        # Determine target device and dtype from input data
        target_device = data_or_X.device if isinstance(data_or_X, torch.Tensor) else self.device
        target_dtype = data_or_X.dtype if isinstance(data_or_X, torch.Tensor) else self.dtype

        # Determine target device and dtype from input data
        
        

        X = torch.as_tensor(data_or_X, device=target_device, dtype=target_dtype)
        y = torch.as_tensor(y, device=target_device, dtype=target_dtype)
        self._classifier._init_module_(X, y, kwargs.get("classes", None))
        self._search.fit(X, y, kwargs.get("groups", None))
        self._classifier = self._search.best_estimator_
        self.fit_status = True

        # Update internal tracking
        self.device = str(target_device)
        self.dtype = target_dtype
        self.cv_results_ = self._search.cv_results_
        self.best_params_ = self._search.best_params_
        self.best_score_ = self._search.best_score_
        self.alpha_ = self.best_params_.get("alpha")
        self.l1_norm_ = self.best_params_.get("l1_norm")
        self.coef_ = self._classifier.coef_
        self.intercept_ = self._classifier.intercept_
        self.classes_ = self._classifier.classes_
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self._classifier.predict(X)

    def decision_function(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(device=self.device, dtype=self.dtype)
        return self._classifier.decision_function(X)

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(device=self.device, dtype=self.dtype)
        return self._classifier.predict_proba(X)

    def forward(self, X: torch.Tensor, y: torch.Tensor = None, **kwargs):
        if not self.fit_status:
            self.fit(X, y, **kwargs)
        return self.predict(X)

class LarsClassifierCV(MLClassifier):
    def __init__(self,
                 fit_intercept: bool = True,
                 verbose: Union[bool, int] = False,
                 precompute: Union[bool, str, List, Tuple, torch.Tensor] = "auto",
                 n_nonzero_coefs_list: List[int] = None,
                 eps: float = torch.finfo(torch.float32).eps,
                 fit_path: bool = True,
                 max_iter: int = None,
                 jitter: float = None,
                 tol: float = None,
                 copy_X: bool = True,
                 positive: bool = False,
                 class_weights: Union[dict, List[dict], Tuple[dict]] = None,
                 cv: Union[str, int, Callable, Iterable, nn.Module] = None,
                 cv_config: dict = None,
                 scoring: Union[str, Callable] = None,
                 n_jobs: int = None,
                 random_state: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        if n_nonzero_coefs_list is None:
            n_nonzero_coefs_list = [10, 50, 100, 500]
        if class_weights is None:
            class_weights = [None]
        elif isinstance(class_weights, dict):
            class_weights = [class_weights]

        self._classifier = LarsClassifier(
            fit_intercept=fit_intercept,
            verbose=verbose,
            precompute=precompute,
            n_nonzero_coefs=n_nonzero_coefs_list[0],
            eps=eps,
            fit_path=fit_path,
            max_iter=max_iter,
            jitter=jitter,
            tol=tol,
            copy_X=copy_X,
            positive=positive,
            class_weight=class_weights[0] if class_weights else None,
            random_state=random_state,
            device=device,
            dtype=dtype,
            *args, **kwargs
        )
        self.params_dict = {
            "n_nonzero_coefs": n_nonzero_coefs_list,
            "class_weight": class_weights
        }
        from ...cross_validation import GridSearchCV
        self._search = GridSearchCV(
            estimator=self._classifier,
            param_grid=self.params_dict,
            scoring=scoring,
            cv=cv,
            cv_config=cv_config,
            n_jobs=n_jobs,
            device=device,
            dtype=dtype
        )
        self.device = device
        self.dtype = dtype
        self.fit_status = False

    def fit(self, data_or_X, y=None, **kwargs):
        # Determine target device and dtype from input data
        target_device = data_or_X.device if isinstance(data_or_X, torch.Tensor) else self.device
        target_dtype = data_or_X.dtype if isinstance(data_or_X, torch.Tensor) else self.dtype

        # Determine target device and dtype from input data
        
        

        X = torch.as_tensor(data_or_X, device=target_device, dtype=target_dtype)
        y = torch.as_tensor(y, device=target_device, dtype=target_dtype)
        self._classifier._init_module_(X, y, kwargs.get("classes", None))
        self._search.fit(X, y, kwargs.get("groups", None))
        self._classifier = self._search.best_estimator_
        self.fit_status = True

        # Update internal tracking
        self.device = str(target_device)
        self.dtype = target_dtype
        self.cv_results_ = self._search.cv_results_
        self.best_params_ = self._search.best_params_
        self.best_score_ = self._search.best_score_
        self.n_nonzero_coefs_ = self.best_params_.get("n_nonzero_coefs")
        self.coef_ = self._classifier.coef_
        self.intercept_ = self._classifier.intercept_
        self.classes_ = self._classifier.classes_
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self._classifier.predict(X)

    def decision_function(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(device=self.device, dtype=self.dtype)
        return self._classifier.decision_function(X)

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(device=self.device, dtype=self.dtype)
        return self._classifier.predict_proba(X)

    def forward(self, X: torch.Tensor, y: torch.Tensor = None, **kwargs):
        if not self.fit_status:
            self.fit(X, y, **kwargs)
        return self.predict(X)

class RidgeLarsClassifierCV(MLClassifier):
    def __init__(self,
                 fit_intercept: bool = True,
                 alphas: Union[float, List[float]] = None,
                 verbose: Union[bool, int] = False,
                 precompute: Union[bool, str, List, Tuple, torch.Tensor] = "auto",
                 n_nonzero_coefs_list: List[int] = None,
                 eps: float = torch.finfo(torch.float32).eps,
                 fit_path: bool = True,
                 max_iter: int = None,
                 jitter: float = None,
                 tol: float = None,
                 copy_X: bool = True,
                 positive: bool = False,
                 class_weights: Union[dict, List[dict], Tuple[dict]] = None,
                 cv: Union[str, int, Callable, Iterable, nn.Module] = None,
                 cv_config: dict = None,
                 scoring: Union[str, Callable] = None,
                 n_jobs: int = None,
                 random_state: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        if alphas is None:
            alphas = [0.1, 1.0, 10.0]
        if n_nonzero_coefs_list is None:
            n_nonzero_coefs_list = [500]
        if isinstance(alphas, float):
            alphas = [alphas]
        if class_weights is None:
            class_weights = [None]
        elif isinstance(class_weights, dict):
            class_weights = [class_weights]

        self._classifier = RidgeLarsClassifier(
            fit_intercept=fit_intercept,
            alpha=alphas[0],
            verbose=verbose,
            precompute=precompute,
            n_nonzero_coefs=n_nonzero_coefs_list[0],
            eps=eps,
            fit_path=fit_path,
            max_iter=max_iter,
            jitter=jitter,
            tol=tol,
            copy_X=copy_X,
            positive=positive,
            class_weight=class_weights[0] if class_weights else None,
            random_state=random_state,
            device=device,
            dtype=dtype,
            *args, **kwargs
        )
        self.params_dict = {
            "alpha": alphas,
            "n_nonzero_coefs": n_nonzero_coefs_list,
            "class_weight": class_weights
        }
        from ...cross_validation import GridSearchCV
        self._search = GridSearchCV(
            estimator=self._classifier,
            param_grid=self.params_dict,
            scoring=scoring,
            cv=cv,
            cv_config=cv_config,
            n_jobs=n_jobs,
            device=device,
            dtype=dtype
        )
        self.device = device
        self.dtype = dtype
        self.fit_status = False

    def fit(self, data_or_X, y=None, **kwargs):
        # Determine target device and dtype from input data
        target_device = data_or_X.device if isinstance(data_or_X, torch.Tensor) else self.device
        target_dtype = data_or_X.dtype if isinstance(data_or_X, torch.Tensor) else self.dtype

        # Determine target device and dtype from input data
        
        

        X = torch.as_tensor(data_or_X, device=target_device, dtype=target_dtype)
        y = torch.as_tensor(y, device=target_device, dtype=target_dtype)
        self._classifier._init_module_(X, y, kwargs.get("classes", None))
        self._search.fit(X, y, kwargs.get("groups", None))
        self._classifier = self._search.best_estimator_
        self.fit_status = True

        # Update internal tracking
        self.device = str(target_device)
        self.dtype = target_dtype
        self.cv_results_ = self._search.cv_results_
        self.best_params_ = self._search.best_params_
        self.best_score_ = self._search.best_score_
        self.alpha_ = self.best_params_.get("alpha")
        self.coef_ = self._classifier.coef_
        self.intercept_ = self._classifier.intercept_
        self.classes_ = self._classifier.classes_
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self._classifier.predict(X)

    def decision_function(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(device=self.device, dtype=self.dtype)
        return self._classifier.decision_function(X)

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(device=self.device, dtype=self.dtype)
        return self._classifier.predict_proba(X)

    def forward(self, X: torch.Tensor, y: torch.Tensor = None, **kwargs):
        if not self.fit_status:
            self.fit(X, y, **kwargs)
        return self.predict(X)

class LassoLarsClassifierCV(MLClassifier):
    def __init__(self,
                 fit_intercept: bool = True,
                 alphas: Union[float, List[float]] = None,
                 verbose: Union[bool, int] = False,
                 precompute: Union[bool, str, List, Tuple, torch.Tensor] = "auto",
                 n_nonzero_coefs_list: List[int] = None,
                 eps: float = torch.finfo(torch.float32).eps,
                 fit_path: bool = True,
                 max_iter: int = None,
                 jitter: float = None,
                 tol: float = None,
                 copy_X: bool = False,
                 positive: bool = False,
                 class_weights: Union[dict, List[dict], Tuple[dict]] = None,
                 cv: Union[str, int, Callable, Iterable, nn.Module] = None,
                 cv_config: dict = None,
                 scoring: Union[str, Callable] = None,
                 n_jobs: int = None,
                 random_state: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        if alphas is None:
            alphas = [0.1, 1.0, 10.0]
        if n_nonzero_coefs_list is None:
            n_nonzero_coefs_list = [500]
        if isinstance(alphas, float):
            alphas = [alphas]
        if class_weights is None:
            class_weights = [None]
        elif isinstance(class_weights, dict):
            class_weights = [class_weights]

        self._classifier = LassoLarsClassifier(
            fit_intercept=fit_intercept,
            alpha=alphas[0],
            verbose=verbose,
            precompute=precompute,
            n_nonzero_coefs=n_nonzero_coefs_list[0],
            eps=eps,
            fit_path=fit_path,
            max_iter=max_iter,
            jitter=jitter,
            tol=tol,
            copy_X=copy_X,
            positive=positive,
            class_weight=class_weights[0] if class_weights else None,
            random_state=random_state,
            device=device,
            dtype=dtype,
            *args, **kwargs
        )
        self.params_dict = {
            "alpha": alphas,
            "n_nonzero_coefs": n_nonzero_coefs_list,
            "class_weight": class_weights
        }
        from ...cross_validation import GridSearchCV
        self._search = GridSearchCV(
            estimator=self._classifier,
            param_grid=self.params_dict,
            scoring=scoring,
            cv=cv,
            cv_config=cv_config,
            n_jobs=n_jobs,
            device=device,
            dtype=dtype
        )
        self.device = device
        self.dtype = dtype
        self.fit_status = False

    def fit(self, data_or_X, y=None, **kwargs):
        # Determine target device and dtype from input data
        target_device = data_or_X.device if isinstance(data_or_X, torch.Tensor) else self.device
        target_dtype = data_or_X.dtype if isinstance(data_or_X, torch.Tensor) else self.dtype

        # Determine target device and dtype from input data
        
        

        X = torch.as_tensor(data_or_X, device=target_device, dtype=target_dtype)
        y = torch.as_tensor(y, device=target_device, dtype=target_dtype)
        self._classifier._init_module_(X, y, kwargs.get("classes", None))
        self._search.fit(X, y, kwargs.get("groups", None))
        self._classifier = self._search.best_estimator_
        self.fit_status = True

        # Update internal tracking
        self.device = str(target_device)
        self.dtype = target_dtype
        self.cv_results_ = self._search.cv_results_
        self.best_params_ = self._search.best_params_
        self.best_score_ = self._search.best_score_
        self.alpha_ = self.best_params_.get("alpha")
        self.coef_ = self._classifier.coef_
        self.intercept_ = self._classifier.intercept_
        self.classes_ = self._classifier.classes_
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self._classifier.predict(X)

    def decision_function(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(device=self.device, dtype=self.dtype)
        return self._classifier.decision_function(X)

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(device=self.device, dtype=self.dtype)
        return self._classifier.predict_proba(X)

    def forward(self, X: torch.Tensor, y: torch.Tensor = None, **kwargs):
        if not self.fit_status:
            self.fit(X, y, **kwargs)
        return self.predict(X)

class ElasticNetLarsClassifierCV(MLClassifier):
    def __init__(self,
                 fit_intercept: bool = True,
                 alphas: Union[float, List[float]] = None,
                 l1_ratios: Union[float, List[float]] = None,
                 verbose: Union[bool, int] = False,
                 precompute: Union[bool, str, List, Tuple, torch.Tensor] = "auto",
                 n_nonzero_coefs_list: List[int] = None,
                 eps: float = torch.finfo(torch.float32).eps,
                 fit_path: bool = True,
                 max_iter: int = None,
                 jitter: float = None,
                 tol: float = None,
                 copy_X: bool = False,
                 positive: bool = False,
                 class_weights: Union[dict, List[dict], Tuple[dict]] = None,
                 cv: Union[str, int, Callable, Iterable, nn.Module] = None,
                 cv_config: dict = None,
                 scoring: Union[str, Callable] = None,
                 n_jobs: int = None,
                 random_state: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        if alphas is None:
            alphas = [0.1, 1.0, 10.0]
        if l1_ratios is None:
            l1_ratios = [0.1, 0.5, 0.9]
        if n_nonzero_coefs_list is None:
            n_nonzero_coefs_list = [500]
        if isinstance(alphas, float):
            alphas = [alphas]
        if isinstance(l1_ratios, float):
            l1_ratios = [l1_ratios]
        if class_weights is None:
            class_weights = [None]
        elif isinstance(class_weights, dict):
            class_weights = [class_weights]

        self._classifier = ElasticNetLarsClassifier(
            fit_intercept=fit_intercept,
            alpha=alphas[0],
            l1_ratio=l1_ratios[0],
            verbose=verbose,
            precompute=precompute,
            n_nonzero_coefs=n_nonzero_coefs_list[0],
            eps=eps,
            fit_path=fit_path,
            max_iter=max_iter,
            jitter=jitter,
            tol=tol,
            copy_X=copy_X,
            positive=positive,
            class_weight=class_weights[0] if class_weights else None,
            random_state=random_state,
            device=device,
            dtype=dtype,
            *args, **kwargs
        )
        self.params_dict = {
            "alpha": alphas,
            "l1_ratio": l1_ratios,
            "n_nonzero_coefs": n_nonzero_coefs_list,
            "class_weight": class_weights
        }
        from ...cross_validation import GridSearchCV
        self._search = GridSearchCV(
            estimator=self._classifier,
            param_grid=self.params_dict,
            scoring=scoring,
            cv=cv,
            cv_config=cv_config,
            n_jobs=n_jobs,
            device=device,
            dtype=dtype
        )
        self.device = device
        self.dtype = dtype
        self.fit_status = False

    def fit(self, data_or_X, y=None, **kwargs):
        # Determine target device and dtype from input data
        target_device = data_or_X.device if isinstance(data_or_X, torch.Tensor) else self.device
        target_dtype = data_or_X.dtype if isinstance(data_or_X, torch.Tensor) else self.dtype

        # Determine target device and dtype from input data
        
        

        X = torch.as_tensor(data_or_X, device=target_device, dtype=target_dtype)
        y = torch.as_tensor(y, device=target_device, dtype=target_dtype)
        self._classifier._init_module_(X, y, kwargs.get("classes", None))
        self._search.fit(X, y, kwargs.get("groups", None))
        self._classifier = self._search.best_estimator_
        self.fit_status = True

        # Update internal tracking
        self.device = str(target_device)
        self.dtype = target_dtype
        self.cv_results_ = self._search.cv_results_
        self.best_params_ = self._search.best_params_
        self.best_score_ = self._search.best_score_
        self.alpha_ = self.best_params_.get("alpha")
        self.l1_ratio_ = self.best_params_.get("l1_ratio")
        self.coef_ = self._classifier.coef_
        self.intercept_ = self._classifier.intercept_
        self.classes_ = self._classifier.classes_
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self._classifier.predict(X)

    def decision_function(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(device=self.device, dtype=self.dtype)
        return self._classifier.decision_function(X)

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(device=self.device, dtype=self.dtype)
        return self._classifier.predict_proba(X)

    def forward(self, X: torch.Tensor, y: torch.Tensor = None, **kwargs):
        if not self.fit_status:
            self.fit(X, y, **kwargs)
        return self.predict(X)


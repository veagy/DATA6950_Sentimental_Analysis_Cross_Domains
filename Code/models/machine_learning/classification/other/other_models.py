import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, List, Callable, Iterable
from .....models.utils import MLClassifier, MLModule
from copy import deepcopy
from ...regression.svm.kernels import get_kernel_class
from torch.func import vmap
import joblib

__all__ = [
    "DummyClassifier",
    "CalibratedClassifierCV",
    "FixedThresholdClassifier",
    "LabelPropagation",
    "LabelSpreading",
    "SelfTrainingClassifier",
    "TunedThresholdClassifierCV",
]


class DummyClassifier(MLClassifier):
    def __init__(self,
                 strategy: str = "prior",
                 random_state: Union[int, None] = None,
                 constant: Union[int, str, List[Union[int, str]],
                 Tuple[Union[int, str]], torch.Tensor, None] = None,
                 warm_start: bool = False,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.warm_start = warm_start
        self.strategy = strategy.lower() if (strategy.lower() in
                                             ["most_frequent", "prior",
                                              "stratified", "uniform", "constant"]) \
            else "prior"
        if random_state is not None:
            torch.manual_seed(random_state)
            self.random_state = torch.Generator().manual_seed(random_state)
        else:
            self.random_state = None
        self.constant = constant
        self.device = device
        self.dtype = dtype
        self.classes_ = []
        self.n_classes_ = []
        self.class_prior_ = []
        self.n_features_in_ = None
        self.n_outputs_ = None
        self.sparse_output_ = False

    def _init_module(self, X: torch.Tensor, y: torch.Tensor):
        self.n_features_in_ = X.size(-1)
        self.n_outputs_ = y.size(-1)
        n_obs = X.size(-2)
        for i in range(self.n_outputs_):
            y_idx = y[..., i:i+1]
            classes_, counts = torch.unique(y_idx, return_counts=True)
            self.n_classes_.append(classes_.size(0))
            self.class_prior_.append(counts.float() / n_obs)
            self.classes_.append(classes_)
        return self

    def fit(self, data_or_X, y=None, **kwargs):
        kwargs.get('warm_start', getattr(self, 'warm_start', False))  # API consistency
        X = torch.as_tensor(data_or_X, dtype=self.dtype, device=self.device)
        y = torch.as_tensor(y, device=self.device, dtype=self.dtype)
        self.classes_ = []
        self.n_classes_ = []
        self.class_prior_ = []
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        self._init_module(X, y)
        self.fit_status = True
        return self

    def decision_function(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, (list, tuple)):
            X = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        n_samples = X.size(0)
        out_list = []
        for i in range(self.n_outputs_):
            n_cls = self.n_classes_[i]
            priors = self.class_prior_[i]

            if self.strategy == "most_frequent":
                # One-hot for the most frequent class
                most_frequent_idx = torch.argmax(priors)
                out = torch.zeros((n_samples, n_cls), device=self.device)
                out[:, most_frequent_idx] = 1.0

            elif self.strategy == "prior":
                # Always returns the empirical class distribution
                out = priors.unsqueeze(0).expand(n_samples, -1)

            elif self.strategy == "stratified":
                # Randomly samples based on priors
                # Sample class indices for each sample, then one-hot encode
                idx = torch.multinomial(
                    priors.unsqueeze(0).expand(n_samples, -1),
                    num_samples=1,
                    replacement=True,
                    generator=self.random_state
                ).squeeze(-1)  # (n_samples,)
                out = F.one_hot(idx, num_classes=n_cls).to(self.dtype)

            elif self.strategy == "uniform":
                # Equal probability for all classes
                out = torch.full((n_samples, n_cls), 1.0 / n_cls,
                                 device=self.device, dtype=self.dtype)

            elif self.strategy == "constant":
                if self.constant is None:
                    raise ValueError(
                        "Strategy is 'constant' but constant is None. "
                        "Please provide a valid constant value."
                    )
                # Handle per-output or shared constant
                if isinstance(self.constant, (list, tuple)):
                    const_val = self.constant[i] if i < len(self.constant) else self.constant[-1]
                else:
                    const_val = self.constant

                const_tensor = torch.tensor(const_val, device=self.device)
                matches = (self.classes_[i] == const_tensor).nonzero(as_tuple=True)[0]
                if matches.numel() == 0:
                    raise ValueError(
                        f"The constant value {const_val} is not a known class label "
                        f"for output {i}. Known classes: {self.classes_[i].tolist()}"
                    )
                const_idx = matches[0]
                out = torch.zeros((n_samples, n_cls), device=self.device, dtype=self.dtype)
                out[:, const_idx] = 1.0

            else:
                raise ValueError(
                    f"Unknown strategy '{self.strategy}'. "
                    "Valid strategies: 'most_frequent', 'prior', 'stratified', 'uniform', 'constant'."
                )

            out_list.append(out)

        # Return stacked result for multi-output or single tensor for single output
        if self.n_outputs_ > 1:
            return torch.stack(out_list, dim=1)  # (n_samples, n_outputs, n_classes)
        return out_list[0]  # (n_samples, n_classes)

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """Return probability estimates.

        For strategies 'prior' and 'most_frequent' these are the empirical
        class priors / one-hot, for 'stratified'/'uniform' they are sampled
        probabilities, and for 'constant' they are a one-hot vector.
        They are normalised via softmax so they sum to 1.
        """
        return F.softmax(self.decision_function(X), dim=-1)

    def predict_log_proba(self, X: torch.Tensor) -> torch.Tensor:
        """Return log-probability estimates."""
        return F.log_softmax(self.decision_function(X), dim=-1)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Return predicted class labels.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Input data. Feature values are ignored; only sample count matters.

        Returns
        -------
        y_pred : torch.Tensor of shape (n_samples,) or (n_samples, n_outputs)
            Predicted class labels.
        """
        scores = self.decision_function(X)
        # scores shape: (n_samples, n_classes) or (n_samples, n_outputs, n_classes)
        if self.n_outputs_ == 1:
            idx = torch.argmax(scores, dim=-1)  # (n_samples,)
            return self.classes_[0][idx]
        else:
            # Multi-output: (n_samples, n_outputs, n_classes)
            idx = torch.argmax(scores, dim=-1)  # (n_samples, n_outputs)
            result = []
            for i in range(self.n_outputs_):
                result.append(self.classes_[i][idx[:, i]])
            return torch.stack(result, dim=-1)  # (n_samples, n_outputs)

    def forward(self, X: torch.Tensor, y: torch.Tensor = None, **kwargs) -> torch.Tensor:
        if not self.fit_status:
            self.fit(X, y, **kwargs)
        return self.predict(X)


class CalibratedClassifierCV(MLClassifier):
    def __init__(self,
                 estimator: MLClassifier = None,
                 method: str = "sigmoid",
                 cv:  Union[str, Callable, Iterable, MLClassifier, int] = None,
                 cv_config: dict = None,
                 n_jobs: int = None,
                 ensemble: Union[str, bool] = "auto",
                 warm_start: bool = False,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.warm_start = warm_start
        if estimator is not None:
            self._base_estimator = estimator
        else:
            from ..svm import SVC
            self._base_estimator = SVC(*args,**kwargs)
        self.method = method.lower()
        from ...cross_validation.splitters import CVSplitManager
        self.cv = CVSplitManager(
            splitter=cv,
            cv_config=cv_config,
            device=device,
            dtype=dtype,
            *args, **kwargs
        )
        self.n_jobs = n_jobs
        self.ensemble = ensemble
        self.device = device
        self.dtype = dtype
        self.classes_ = None
        self.n_classes_ = None
        self.calibrated_classifiers_ = nn.ModuleList()
        self.n_features_in_ = None

    def _init_module(self, X: torch.Tensor, y: torch.Tensor):
        self.n_features_in_ = X.size(-1)
        self.classes_ = torch.unique(y)
        self.n_classes_ = self.classes_.size(0)
        return self

    # ------------------------------------------------------------------
    # Internal calibrators
    # ------------------------------------------------------------------

    def _get_scores(self, estimator: MLClassifier, X: torch.Tensor) -> torch.Tensor:
        """Get raw scores from estimator: prefer decision_function, else predict_proba."""
        if hasattr(estimator, "decision_function"):
            return estimator.decision_function(X)
        return estimator.predict_proba(X)

    def _build_calibrator(self, scores: torch.Tensor, y: torch.Tensor) -> nn.Module:
        """
        Build and fit a calibration layer for the given (scores, y) pair.

        * 'sigmoid'     – Platt scaling: fit a 1-D logistic regression A*s + B.
        * 'isotonic'    – Isotonic / pool-adjacent-violators (PAV) mapping.
        * 'temperature' – Fit a single temperature parameter T on the logits.
        """
        if self.method == "sigmoid":
            return self._fit_sigmoid(scores, y)
        elif self.method == "isotonic":
            return self._fit_isotonic(scores, y)
        elif self.method == "temperature":
            return self._fit_temperature(scores, y)
        else:
            raise ValueError(f"Unknown calibration method: '{self.method}'")

    class _PlattCalibrator(nn.Module):
        """Platt scaling: sigmoid(A * score + B)."""
        def __init__(self, n_classes: int, device: str, dtype: torch.dtype):
            super().__init__()
            self.A = nn.Parameter(torch.ones(n_classes, device=device, dtype=dtype))
            self.B = nn.Parameter(torch.zeros(n_classes, device=device, dtype=dtype))

        def forward(self, scores: torch.Tensor) -> torch.Tensor:
            # scores: (n_samples, n_classes)
            calibrated = torch.sigmoid(self.A * scores + self.B)
            # Renormalise to sum to 1
            return calibrated / calibrated.sum(dim=-1, keepdim=True).clamp(min=1e-12)

    class _IsotonicCalibrator(nn.Module):
        """Pool-Adjacent-Violators (PAV) isotonic regression calibrator."""
        def __init__(self):
            super().__init__()
            self.mapping_ = None  # list of (threshold, value) per class

        def _pav(self, scores_1d: torch.Tensor, labels_1d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Fit isotonic regression via PAV algorithm."""
            # Sort by score
            order = torch.argsort(scores_1d)
            s = scores_1d[order]
            t = labels_1d[order].float()

            # PAV: pool blocks that violate monotonicity
            n = s.size(0)
            blocks = [[float(s[i].item()), float(t[i].item()), 1] for i in range(n)]

            def _merge_blocks(blocks_):
                changed = True
                while changed:
                    changed = False
                    i = 0
                    new_blocks = []
                    while i < len(blocks_):
                        if i + 1 < len(blocks_) and blocks_[i][1] > blocks_[i + 1][1]:
                            # Merge blocks i and i+1
                            total = blocks_[i][2] + blocks_[i + 1][2]
                            merged_val = (blocks_[i][1] * blocks_[i][2] +
                                          blocks_[i + 1][1] * blocks_[i + 1][2]) / total
                            new_blocks.append([blocks_[i][0], merged_val, total])
                            i += 2
                            changed = True
                        else:
                            new_blocks.append(blocks_[i])
                            i += 1
                    blocks_ = new_blocks
                return blocks_

            blocks = _merge_blocks(blocks)
            # Expand blocks back to per-sample values
            iso_vals = []
            for bl in blocks:
                iso_vals.extend([bl[1]] * bl[2])
            iso_tensor = torch.tensor(iso_vals, dtype=scores_1d.dtype, device=scores_1d.device)
            # Align with original order
            result = torch.empty_like(iso_tensor)
            result[order] = iso_tensor
            return s, result  # sorted scores, isotonic outputs (unsorted)

        def fit(self, scores: torch.Tensor, labels: torch.Tensor):
            """scores: (n, C), labels: (n,) integer class indices."""
            n_classes = scores.size(-1)
            self.mapping_ = []
            for c in range(n_classes):
                # Binary label: 1 if this class, 0 otherwise
                binary = (labels == c).float()
                s_sorted, iso_out = self._pav(scores[:, c], binary)
                self.mapping_.append((s_sorted, iso_out))
            return self

        def forward(self, scores: torch.Tensor) -> torch.Tensor:
            if self.mapping_ is None:
                raise RuntimeError("IsotonicCalibrator must be fitted before calling forward.")
            n_samples, n_classes = scores.shape
            out = torch.zeros_like(scores)
            for c in range(n_classes):
                s_sorted, iso_out = self.mapping_[c]
                # Interpolate new score values using the fitted isotonic map
                new_s = scores[:, c]
                # Find insertion positions
                idx = torch.bucketize(new_s, s_sorted, right=True).clamp(0, len(iso_out) - 1)
                out[:, c] = iso_out[idx]
            # Renormalise
            row_sum = out.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            return out / row_sum

    class _TemperatureScaler(nn.Module):
        """Temperature scaling with a single learnable scalar T."""
        def __init__(self, device: str, dtype: torch.dtype):
            super().__init__()
            self.temperature = nn.Parameter(torch.ones(1, device=device, dtype=dtype))

        def forward(self, logits: torch.Tensor) -> torch.Tensor:
            T = self.temperature.clamp(min=1e-6)
            return F.softmax(logits / T, dim=-1)

    def _fit_sigmoid(self, scores: torch.Tensor, y: torch.Tensor) -> nn.Module:
        """Fit Platt scaling calibrator."""
        n_classes = scores.size(-1) if scores.ndim > 1 else 1
        calibrator = self._PlattCalibrator(n_classes, self.device, self.dtype)
        optimizer = torch.optim.LBFGS(calibrator.parameters(), lr=0.01, max_iter=50)
        y_long = y.long()
        if y_long.ndim > 1:
            y_long = y_long.squeeze(-1)

        def _closure():
            optimizer.zero_grad()
            probs = calibrator(scores)
            custom_loss_fn = getattr(self, '_fit_loss_fn', None)
            if custom_loss_fn is not None:
                loss = custom_loss_fn(y_long, probs)
            else:
                loss = F.nll_loss(torch.log(probs.clamp(min=1e-12)), y_long)
            loss.backward()
            return loss

        optimizer.step(_closure)
        return calibrator

    def _fit_isotonic(self, scores: torch.Tensor, y: torch.Tensor) -> nn.Module:
        """Fit isotonic regression calibrator (PAV)."""
        calibrator = self._IsotonicCalibrator()
        y_long = y.long()
        if y_long.ndim > 1:
            y_long = y_long.squeeze(-1)
        # Ensure scores are 2D: (n, n_classes)
        if scores.ndim == 1:
            scores = scores.unsqueeze(-1)
        calibrator.fit(scores.detach(), y_long.detach())
        return calibrator

    def _fit_temperature(self, scores: torch.Tensor, y: torch.Tensor) -> nn.Module:
        """Fit temperature scaler."""
        calibrator = self._TemperatureScaler(self.device, self.dtype)
        optimizer = torch.optim.LBFGS(calibrator.parameters(), lr=0.01, max_iter=50)
        y_long = y.long()
        if y_long.ndim > 1:
            y_long = y_long.squeeze(-1)

        def _closure():
            optimizer.zero_grad()
            probs = calibrator(scores)
            custom_loss_fn = getattr(self, '_fit_loss_fn', None)
            if custom_loss_fn is not None:
                loss = custom_loss_fn(y_long, probs)
            else:
                loss = F.nll_loss(torch.log(probs.clamp(min=1e-12)), y_long)
            loss.backward()
            return loss

        optimizer.step(_closure)
        return calibrator

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, data_or_X, y=None, **kwargs):
        """Fit the calibrated classifier.

        Parameters
        ----------
        data_or_X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self
        """
        self._fit_loss_fn = kwargs.get('loss_fn', None)
        try:
            warm_start = kwargs.get('warm_start', getattr(self, 'warm_start', False))
            X = torch.as_tensor(data_or_X, dtype=self.dtype, device=self.device)
            y = torch.as_tensor(y, device=self.device, dtype=self.dtype)
            self._init_module(X, y)

            if y.ndim == 1:
                y = y.unsqueeze(-1)

            # Reset any previously fitted calibrated classifiers
            self.calibrated_classifiers_ = nn.ModuleList()

            # Resolve ensemble flag
            _ensemble = self.ensemble
            if _ensemble == "auto":
                _ensemble = True  # treat non-frozen estimators as ensemble=True

            if _ensemble:
                # ---- Ensemble mode: fit + calibrate per fold ----
                for train_idx, val_idx in self.cv.split(X, y.squeeze(-1)):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]

                    # Fit a fresh copy of the base estimator on the training fold
                    model = deepcopy(self._base_estimator)
                    model.fit(X_train, y_train.squeeze(-1), warm_start=warm_start)

                    # Obtain raw scores on the validation fold for calibration
                    with torch.no_grad():
                        scores = self._get_scores(model, X_val)
                        if scores.ndim == 1:
                            scores = scores.unsqueeze(-1)
                    
                    # Fit calibrator on validation scores
                    calibrator = self._build_calibrator(scores, y_val)

                    # Store as a (model, calibrator) tuple wrapped to be nn.Module-compatible
                    pair = nn.ModuleList([model, calibrator])
                    self.calibrated_classifiers_.append(pair)

            else:
                # ---- Non-ensemble mode: cross-val predict then calibrate all-data model ----
                n_samples = X.size(0)
                n_classes = int(self.n_classes_)
                oof_scores = torch.zeros((n_samples, n_classes), dtype=self.dtype, device=self.device)
                oof_mask = torch.zeros(n_samples, dtype=torch.bool, device=self.device)

                for train_idx, val_idx in self.cv.split(X, y.squeeze(-1)):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train = y[train_idx]

                    model = deepcopy(self._base_estimator)
                    model.fit(X_train, y_train.squeeze(-1), warm_start=warm_start)

                    with torch.no_grad():
                        scores = self._get_scores(model, X_val)
                        if scores.ndim == 1:
                            scores = scores.unsqueeze(-1)
                        if scores.size(-1) != n_classes:
                            scores = scores[..., :n_classes]

                    oof_scores[val_idx] = scores
                    oof_mask[val_idx] = True

                # Fit calibrator on the out-of-fold predictions
                calibrator = self._build_calibrator(oof_scores[oof_mask], y[oof_mask])

                # Fit the base estimator on all data
                full_model = deepcopy(self._base_estimator)
                full_model.fit(X, y.squeeze(-1), warm_start=warm_start)

                pair = nn.ModuleList([full_model, calibrator])
                self.calibrated_classifiers_.append(pair)

            self.fit_status = True
            return self
        finally:
            self._fit_loss_fn = None

    def decision_function(self, X: torch.Tensor) -> torch.Tensor:
        """Compute the raw decision scores (log of calibrated probabilities).

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)

        Returns
        -------
        scores : torch.Tensor of shape (n_samples, n_classes)
        """
        if not self.fit_status:
            raise RuntimeError("CalibratedClassifierCV must be fitted before calling decision_function.")
        X = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        # Return log of averaged calibrated probabilities
        return torch.log(self._predict_proba_raw(X).clamp(min=1e-12))

    def _predict_proba_raw(self, X: torch.Tensor) -> torch.Tensor:
        """Return averaged calibrated probabilities across all calibrated classifiers."""
        proba_list = []
        for pair in self.calibrated_classifiers_:
            model, calibrator = pair[0], pair[1]
            with torch.no_grad():
                scores = self._get_scores(model, X)
                if scores.ndim == 1:
                    scores = scores.unsqueeze(-1)
            # Apply calibration
            cal_proba = calibrator(scores)
            proba_list.append(cal_proba)
        # Average calibrated probabilities across all pairs
        stacked = torch.stack(proba_list, dim=0)  # (n_pairs, n_samples, n_classes)
        return stacked.mean(dim=0)                 # (n_samples, n_classes)

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """Return calibrated probability estimates.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)

        Returns
        -------
        proba : torch.Tensor of shape (n_samples, n_classes)
            Calibrated class probabilities that sum to 1 across classes.
        """
        if not self.fit_status:
            raise RuntimeError("CalibratedClassifierCV must be fitted before calling predict_proba.")
        X = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        return self._predict_proba_raw(X)

    def predict_log_proba(self, X: torch.Tensor) -> torch.Tensor:
        """Return log of calibrated probability estimates.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)

        Returns
        -------
        log_proba : torch.Tensor of shape (n_samples, n_classes)
        """
        return torch.log(self.predict_proba(X).clamp(min=1e-12))

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)

        Returns
        -------
        y_pred : torch.Tensor of shape (n_samples,)
            Predicted class label per sample (from self.classes_).
        """
        if not self.fit_status:
            raise RuntimeError("CalibratedClassifierCV must be fitted before calling predict.")
        X = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        proba = self.predict_proba(X)              # (n_samples, n_classes)
        idx = torch.argmax(proba, dim=-1)          # (n_samples,)
        return self.classes_[idx]

    def forward(self, X: torch.Tensor, y: torch.Tensor = None, **kwargs) -> torch.Tensor:
        if not self.fit_status:
            self.fit(X, y, **kwargs)
        return self.predict(X)


class FixedThresholdClassifier(MLClassifier):
    def __init__(self,
                 estimator: MLClassifier = None,
                 threshold: Union[str, float] = "auto",
                 pos_label: Union[int, float, bool, str] = None,
                 response_method: str = "auto",
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        from ...classification import DecisionTreeClassifier
        self.estimator_ = estimator if estimator is not None else DecisionTreeClassifier(*args, **kwargs)
        self.response_method = response_method
        if isinstance(threshold, str):
            if threshold.lower() == "auto":
                if self.response_method == "predict_proba":
                    self.threshold = 0.5
                else:
                    self.threshold = 0.0
        elif isinstance(threshold, float):
            self.threshold = threshold
        self.pos_label = pos_label
        self.device = device
        self.dtype = dtype
        self.classes_ = None
        self.n_classes_ = None
        self.n_features_in_ = None

    def _init_module(self, X: torch.Tensor, y: torch.Tensor):
        self.n_features_in_ = X.size(-1)
        self.classes_ = torch.unique(y)
        self.n_classes_ = self.classes_.size(0)
        return self

    def _resolve_pos_label(self) -> torch.Tensor:
        """Resolve the positive class label tensor.

        If `pos_label` was supplied use it directly; otherwise default to 1
        when classes are in {-1, 1} or {0, 1}; for multi-class, default to
        classes_[1] (or classes_[-1] if only one class).
        """
        if self.pos_label is not None:
            return torch.tensor(self.pos_label, dtype=self.dtype, device=self.device)

        if self.classes_ is None:
            raise RuntimeError("Classifier has not been fitted yet.")

        classes_set = set(self.classes_.tolist())
        if classes_set in ({0, 1}, {-1, 1}):
            return torch.tensor(1, dtype=self.dtype, device=self.device)

        # Multi-class: default to classes_[1] (or classes_[-1] if n_classes < 2)
        if len(self.classes_) > 2:
            idx = min(1, len(self.classes_) - 1)
            return self.classes_[idx].reshape(()).to(dtype=self.dtype, device=self.device)

        raise ValueError(
            "pos_label=None is only supported when classes are {0, 1} or {-1, 1}. "
            f"Got classes: {self.classes_.tolist()}. Pass an explicit pos_label."
        )

    def _get_response(self, X: torch.Tensor) -> torch.Tensor:
        """Return the 1-D score for the positive class using the chosen response_method.

        * ``'predict_proba'``    – probability of the positive class column.
        * ``'decision_function'``– raw decision score.  For multi-column output
          the column corresponding to `pos_label` is selected.
        * ``'auto'``             – tries `predict_proba` first, then
          `decision_function`.

        Returns
        -------
        scores : torch.Tensor of shape (n_samples,)
        """
        if hasattr(self.estimator_, "to") and self.estimator_ is not None:
            self.estimator_.to(self.device)
        pos_label_tensor = self._resolve_pos_label()

        def _from_predict_proba() -> torch.Tensor:
            proba = self.estimator_.predict_proba(X)          # (n, n_classes)
            if proba.ndim == 1:
                return proba
            # Find column for pos_label
            match = (self.classes_ == pos_label_tensor).nonzero(as_tuple=True)[0]
            if match.numel() == 0:
                raise ValueError(
                    f"pos_label {pos_label_tensor.item()} not found in classes_."
                )
            return proba[:, match[0]]                          # (n_samples,)

        def _from_decision_function() -> torch.Tensor:
            scores = self.estimator_.decision_function(X)     # (n,) or (n, n_classes)
            if scores.ndim == 1:
                return scores
            match = (self.classes_ == pos_label_tensor).nonzero(as_tuple=True)[0]
            if match.numel() == 0:
                raise ValueError(
                    f"pos_label {pos_label_tensor.item()} not found in classes_."
                )
            return scores[:, match[0]]

        rm = self.response_method.lower() if isinstance(self.response_method, str) else "auto"

        if rm == "predict_proba":
            if not hasattr(self.estimator_, "predict_proba"):
                raise AttributeError(
                    f"{type(self.estimator_).__name__} does not implement predict_proba."
                )
            return _from_predict_proba()

        elif rm == "decision_function":
            if not hasattr(self.estimator_, "decision_function"):
                raise AttributeError(
                    f"{type(self.estimator_).__name__} does not implement decision_function."
                )
            return _from_decision_function()

        else:  # "auto" – try predict_proba first, then decision_function
            if hasattr(self.estimator_, "predict_proba"):
                return _from_predict_proba()
            elif hasattr(self.estimator_, "decision_function"):
                return _from_decision_function()
            else:
                raise AttributeError(
                    f"{type(self.estimator_).__name__} implements neither "
                    "predict_proba nor decision_function."
                )

    def fit(self, data_or_X, y=None, **kwargs):
        """Fit the underlying estimator.

        Parameters
        ----------
        data_or_X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target (binary) labels.

        Returns
        -------
        self
        """
        X = torch.as_tensor(data_or_X, dtype=self.dtype, device=self.device)
        y = torch.as_tensor(y, device=self.device, dtype=self.dtype)
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        self._init_module(X, y)

        # Resolve threshold lazily now that response_method is known
        if not hasattr(self, "threshold") or self.threshold is None:
            if self.response_method == "predict_proba":
                self.threshold = 0.5
            else:
                self.threshold = 0.0

        # Fit the underlying estimator
        self.estimator_.fit(X, y.squeeze(-1), **kwargs)
        self.fit_status = True
        return self

    def decision_function(self, X: torch.Tensor) -> torch.Tensor:
        """Compute continuous decision scores for positive class membership.

        The score returned is the raw output of ``_get_response``: either the
        predicted probability for the positive class or the raw decision score,
        depending on ``response_method``.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)

        Returns
        -------
        scores : torch.Tensor of shape (n_samples,)
            Continuous score for the positive class.
        """
        if not self.fit_status:
            raise RuntimeError(
                "FixedThresholdClassifier must be fitted before calling decision_function."
            )
        X = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        return self._get_response(X)

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """Return probability estimates for both classes.

        Uses ``_get_response`` to obtain the positive-class score and constructs
        a 2-column probability matrix ``[1 - p_pos, p_pos]``.  When
        ``response_method`` is ``'decision_function'`` the raw scores are
        squashed through sigmoid before returning.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)

        Returns
        -------
        proba : torch.Tensor of shape (n_samples, 2)
            Probability estimates for the negative and positive classes.
        """
        if not self.fit_status:
            raise RuntimeError(
                "FixedThresholdClassifier must be fitted before calling predict_proba."
            )
        X = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        scores = self._get_response(X)  # (n_samples,)

        rm = self.response_method.lower() if isinstance(self.response_method, str) else "auto"

        # If we obtained raw decision scores squash them to [0, 1] via sigmoid
        if rm == "decision_function":
            p_pos = torch.sigmoid(scores)
        elif rm == "predict_proba":
            p_pos = scores
        else:
            # "auto": if estimator has predict_proba the scores are already probabilities
            if hasattr(self.estimator_, "predict_proba"):
                p_pos = scores
            else:
                p_pos = torch.sigmoid(scores)

        p_neg = 1.0 - p_pos
        return torch.stack([p_neg, p_pos], dim=-1)  # (n_samples, 2)

    def predict_log_proba(self, X: torch.Tensor) -> torch.Tensor:
        """Return log-probability estimates.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)

        Returns
        -------
        log_proba : torch.Tensor of shape (n_samples, 2)
        """
        return torch.log(self.predict_proba(X).clamp(min=1e-12))

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict binary class labels using the fixed decision threshold.

        A sample is assigned to the positive class when its score (from
        ``_get_response``) is **greater than or equal to** ``self.threshold``,
        and to the negative class otherwise.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)

        Returns
        -------
        y_pred : torch.Tensor of shape (n_samples,)
            Predicted class labels drawn from ``self.classes_``.
        """
        if not self.fit_status:
            raise RuntimeError(
                "FixedThresholdClassifier must be fitted before calling predict."
            )
        X = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        scores = self._get_response(X)  # (n_samples,)

        pos_label_tensor = self._resolve_pos_label()
        # Identify the positive and negative class indices in self.classes_
        pos_mask = (self.classes_ == pos_label_tensor)
        neg_mask = ~pos_mask

        pos_idx = pos_mask.nonzero(as_tuple=True)[0]
        neg_idx = neg_mask.nonzero(as_tuple=True)[0]

        if pos_idx.numel() == 0:
            raise ValueError(
                f"pos_label {pos_label_tensor.item()} not found in classes_ "
                f"{self.classes_.tolist()}."
            )

        pos_class = self.classes_[pos_idx[0]]
        neg_class = self.classes_[neg_idx[0]] if neg_idx.numel() > 0 else self.classes_[0]

        # Apply threshold
        positive = scores >= self.threshold          # (n_samples,) bool
        y_pred = torch.where(positive, pos_class, neg_class)
        return y_pred

    def forward(self, X: torch.Tensor, y: torch.Tensor = None, **kwargs) -> torch.Tensor:
        if not self.fit_status:
            self.fit(X, y, **kwargs)
        return self.predict(X)


class LabelPropagation(MLClassifier):
    def __init__(self,
                 kernel: Union[str, MLModule, Callable] = "rbf",
                 gamma: float = 20.0,
                 n_neighbors: int = 7,
                 max_iter: int = 1000,
                 tol: float = 1e-3,
                 n_jobs: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.kernel = kernel
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.max_iter = max_iter
        self.tol = tol
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype
        # Attributes set during fit
        self.X_ = None
        self.classes_ = None
        self.label_distributions_ = None
        self.transduction_ = None
        self.n_features_in_ = None
        self.n_iter_ = None

    # ------------------------------------------------------------------
    # Kernel builders
    # ------------------------------------------------------------------

    def _get_kernel(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        """Internal helper to get kernel matrix using the common kernel library or internal KNN."""
        if callable(self.kernel) and not isinstance(self.kernel, str):
            W = self.kernel(X1, X2)
            if not isinstance(W, torch.Tensor):
                W = torch.as_tensor(W, dtype=self.dtype, device=self.device)
            return W

        if isinstance(self.kernel, str):
            k_name = self.kernel.lower()
            if k_name == "knn":
                return self._build_knn_kernel(X1, X2)
            
            # Use centralized kernel library
            kernel_class = get_kernel_class(k_name)
            if kernel_class:
                # Prepare kernel config
                k_config = {
                    "gamma": self.gamma,
                    "device": self.device,
                    "dtype": self.dtype
                }
                # RBF and Sigmoid expect 'bias' (coef0) or 'gamma'
                # Poly expects 'degree', 'gamma', 'bias'
                # Extract extra args from self.kwargs if available
                if hasattr(self, 'kwargs'):
                    k_config.update({
                        "degree": self.kwargs.get("degree", 3),
                        "bias": self.kwargs.get("bias", self.kwargs.get("coef0", 1.0)),
                    })
                else:
                    # Fallback defaults
                    k_config.update({
                        "degree": 3,
                        "bias": 1.0,
                    })

                k_inst = kernel_class(**k_config)
                return k_inst(X1, X2)

        raise ValueError(
            f"Unknown kernel '{self.kernel}'. "
            "Expected 'knn', 'rbf', 'poly', 'sigmoid', 'linear', or a library kernel name / callable."
        )

    def _build_rbf_kernel(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        """Deprecated: Use _get_kernel instead. Kept for backwards compatibility if needed."""
        # This now delegates to the library via _get_kernel if self.kernel is "rbf"
        # but if called directly we implement the fallback logic.
        sq_norms_1 = (X1 ** 2).sum(dim=-1, keepdim=True)
        sq_norms_2 = (X2 ** 2).sum(dim=-1, keepdim=True)
        sq_dists = sq_norms_1 + sq_norms_2.T - 2.0 * (X1 @ X2.T)
        sq_dists = sq_dists.clamp(min=0.0)
        return torch.exp(-self.gamma * sq_dists)

    def _build_knn_kernel(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        """Compute the sparse KNN affinity matrix.

        For each row (sample in X1) the k nearest neighbours in X2 receive
        weight 1; all other_decomposition entries are 0.  The result is symmetrised so that
        if i→j or j→i exists the edge is retained.

        Parameters
        ----------
        X1 : torch.Tensor of shape (n1, n_features)
        X2 : torch.Tensor of shape (n2, n_features)

        Returns
        -------
        W : torch.Tensor of shape (n1, n2)  (dense binary/float matrix)
        """
        k = min(self.n_neighbors, X2.size(0))
        sq_norms_1 = (X1 ** 2).sum(dim=-1, keepdim=True)
        sq_norms_2 = (X2 ** 2).sum(dim=-1, keepdim=True)
        sq_dists = sq_norms_1 + sq_norms_2.T - 2.0 * (X1 @ X2.T)
        sq_dists = sq_dists.clamp(min=0.0)

        # For each row find the k nearest column indices
        _, nn_indices = torch.topk(sq_dists, k=k, dim=-1, largest=False)  # (n1, k)

        n1, n2 = X1.size(0), X2.size(0)
        W = torch.zeros((n1, n2), dtype=self.dtype, device=self.device)
        row_idx = torch.arange(n1, device=self.device).unsqueeze(-1).expand_as(nn_indices)
        W[row_idx, nn_indices] = 1.0

        # Symmetrise when X1 is X2 (inductive case uses non-square; skip)
        if X1.data_ptr() == X2.data_ptr() or (n1 == n2 and torch.equal(X1, X2)):
            W = ((W + W.T) > 0).to(self.dtype)

        return W

    def _build_graph(self, X: torch.Tensor) -> torch.Tensor:
        """Build the normalised weight (affinity) matrix for all training points.

        Returns row-stochastic transition matrix T of shape (n_samples, n_samples).
        """
        W = self._get_kernel(X, X)

        # Zero out self-loops
        W.fill_diagonal_(0.0)

        # Row-normalise (T[i,j] = W[i,j] / sum_j W[i,j])
        row_sums = W.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        T = W / row_sums
        return T

    # ------------------------------------------------------------------
    # Internal module initialisation
    # ------------------------------------------------------------------

    def _init_module(self, X: torch.Tensor, y: torch.Tensor):
        self.n_features_in_ = X.size(-1)
        self.classes_ = torch.unique(y)
        return self

    # ------------------------------------------------------------------
    # Label matrix helpers
    # ------------------------------------------------------------------

    def _label_matrix(self, y: torch.Tensor) -> torch.Tensor:
        """Build initial label distribution matrix F of shape (n_samples, n_classes).

        Labelled samples receive a one-hot row; unlabelled samples (labelled -1)
        receive a uniform row.
        """
        n_samples = y.size(0)
        n_classes = self.classes_.size(0)
        f = torch.zeros((n_samples, n_classes), dtype=self.dtype, device=self.device)

        for i in range(n_samples):
            label = y[i].item()
            if label == -1:
                # Unlabelled: uniform prior
                f[i] = 1.0 / n_classes
            else:
                match = (self.classes_ == y[i]).nonzero(as_tuple=True)[0]
                if match.numel() > 0:
                    f[i, match[0]] = 1.0
                else:
                    f[i] = 1.0 / n_classes  # fallback
        return f

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, data_or_X, y=None, **kwargs):
        """Fit the Label Propagation model.

        Parameters
        ----------
        data_or_X : array-like of shape (n_samples, n_features)
            Training data.  Unlabelled samples should be assigned label -1.
        y : array-like of shape (n_samples,)
            Target vector.  Use -1 to denote unlabelled samples.

        Returns
        -------
        self
        """
        X = torch.as_tensor(data_or_X, dtype=self.dtype, device=self.device)
        y = torch.as_tensor(y, device=self.device, dtype=self.dtype)
        if y.ndim > 1:
            y = y.squeeze(-1)

        self._init_module(X, y)
        self.X_ = X

        n_samples = X.size(0)
        n_classes = self.classes_.size(0)

        # Build the row-stochastic transition matrix T (n_samples, n_samples)
        T = self._build_graph(X)

        # Initial label distribution matrix (hard clamping for labelled nodes)
        Y0 = self._label_matrix(y)  # (n_samples, n_classes)

        # Boolean mask of labelled samples
        labelled_mask = (y != -1)  # (n_samples,)

        f = Y0.clone()

        # --- Label Propagation iteration ---
        # F = T @ F  then clamp labelled rows back to Y0
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            F_prev = f.clone()

            # Propagate: F = T @ F
            f = T @ f

            # Hard clamping: labelled nodes always keep their original labels
            f[labelled_mask] = Y0[labelled_mask]

            # Normalise rows to sum to 1
            row_sums = f.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            f = f / row_sums

            # Convergence check
            delta = (f - F_prev).abs().max().item()
            if delta < self.tol:
                break

        self.n_iter_ = n_iter
        self.label_distributions_ = f
        # Transduction: assign each sample the argmax class
        argmax_idx = torch.argmax(f, dim=-1)            # (n_samples,)
        self.transduction_ = self.classes_[argmax_idx]  # (n_samples,)
        self.fit_status = True
        return self

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _infer_kernel_matrix(self, X_new: torch.Tensor) -> torch.Tensor:
        """Return the affinity matrix between X_new and training X_.

        Shape: (n_new, n_train).
        """
        W = self._get_kernel(X_new, self.X_)
        # Row-normalise
        row_sums = W.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        return W / row_sums  # (n_new, n_train)

    # ------------------------------------------------------------------
    # Public prediction API
    # ------------------------------------------------------------------

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """Return probability estimates for new samples.

        For each new sample, the label distribution is computed as the
        row-normalised weighted sum of the training label distributions,
        where the weights come from the kernel between the new sample and all
        training points.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)

        Returns
        -------
        proba : torch.Tensor of shape (n_samples, n_classes)
            Probability estimates summing to 1 along axis 1.
        """
        if not self.fit_status:
            raise RuntimeError(
                "LabelPropagation must be fitted before calling predict_proba."
            )
        X = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        # W_new: (n_new, n_train),  label_distributions_: (n_train, n_classes)
        W_new = self._infer_kernel_matrix(X)           # row-stochastic
        proba = W_new @ self.label_distributions_      # (n_new, n_classes)
        row_sums = proba.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        return proba / row_sums

    def predict_log_proba(self, X: torch.Tensor) -> torch.Tensor:
        """Return log-probability estimates.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)

        Returns
        -------
        log_proba : torch.Tensor of shape (n_samples, n_classes)
        """
        return torch.log(self.predict_proba(X).clamp(min=1e-12))

    def decision_function(self, X: torch.Tensor) -> torch.Tensor:
        """Compute the prediction confidence scores.

        Returns the raw (un-normalised) label distributions for new samples,
        obtained as the weighted combination of training label distributions
        via the kernel.  These are the logits analogues for multi-class output.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)

        Returns
        -------
        scores : torch.Tensor of shape (n_samples, n_classes)
        """
        if not self.fit_status:
            raise RuntimeError(
                "LabelPropagation must be fitted before calling decision_function."
            )
        X = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        W_new = self._infer_kernel_matrix(X)
        return W_new @ self.label_distributions_       # (n_new, n_classes)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict class labels for new samples.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)

        Returns
        -------
        y_pred : torch.Tensor of shape (n_samples,)
            Predicted class label per sample.
        """
        if not self.fit_status:
            raise RuntimeError(
                "LabelPropagation must be fitted before calling predict."
            )
        X = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        proba = self.predict_proba(X)                  # (n_samples, n_classes)
        idx = torch.argmax(proba, dim=-1)              # (n_samples,)
        return self.classes_[idx]

    def forward(self, X: torch.Tensor, y: torch.Tensor = None, **kwargs) -> torch.Tensor:
        if not self.fit_status:
            self.fit(X, y, **kwargs)
        return self.predict(X)


class LabelSpreading(LabelPropagation):
    def __init__(self,
                 kernel: Union[str, MLModule, Callable] = "rbf",
                 gamma: float = 20.0,
                 n_neighbors: int = 7,
                 alpha: float = 0.2,
                 max_iter: int = 1000,
                 tol: float = 1e-3,
                 n_jobs: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(
            kernel=kernel,
            gamma=gamma,
            n_neighbors=n_neighbors,
            max_iter=max_iter,
            tol=tol,
            n_jobs=n_jobs,
            device=device,
            dtype=dtype,
            *args, **kwargs
        )
        # alpha in (0, 1): how much to adopt neighbour information vs. initial labels.
        # alpha=0 → keep original labels; alpha=1 → full neighbour information.
        if not (0.0 <= alpha < 1.0):
            raise ValueError(
                f"alpha must be in [0, 1). Got {alpha}."
            )
        self.alpha = alpha

    # ------------------------------------------------------------------
    # Override: normalised graph Laplacian transition matrix
    # ------------------------------------------------------------------

    def _build_graph(self, X: torch.Tensor) -> torch.Tensor:
        """Build the normalised graph Laplacian affinity matrix S.

        LabelSpreading uses the symmetric normalised Laplacian:

            S = D^{-1/2} W D^{-1/2}

        where W is the raw affinity matrix (RBF or KNN) and D is the
        diagonal degree matrix.  Unlike LabelPropagation, self-loops
        are *kept* so that D includes the self-affinity weight.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)

        Returns
        -------
        S : torch.Tensor of shape (n_samples, n_samples)
            Symmetric normalised affinity matrix.
        """
        W = self._get_kernel(X, X)

        # Degree vector: sum of affinities (including self-loops for rbf;
        # self-loops are already 0 for knn since the point is its own nearest neighbour
        # in 0-distance but topk with largest=False returns it — we zero it out here
        # to be consistent with sklearn's implementation).
        W.fill_diagonal_(0.0)

        # D^{-1/2}
        d = W.sum(dim=-1)                               # (n_samples,)
        d_inv_sqrt = d.pow(-0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0       # guard against isolated nodes

        # S = D^{-1/2} W D^{-1/2}  (element-wise outer-product scaling)
        S = d_inv_sqrt.unsqueeze(-1) * W * d_inv_sqrt.unsqueeze(0)
        return S

    # ------------------------------------------------------------------
    # Override: soft-clamping fit
    # ------------------------------------------------------------------

    def fit(self, data_or_X, y=None, **kwargs):
        """Fit the Label Spreading model.

        Parameters
        ----------
        data_or_X : array-like of shape (n_samples, n_features)
            Training data.  Unlabelled samples should be assigned label -1.
        y : array-like of shape (n_samples,)
            Target vector.  Use -1 to denote unlabelled samples.

        Returns
        -------
        self
        """
        X = torch.as_tensor(data_or_X, dtype=self.dtype, device=self.device)
        y_t = torch.as_tensor(y, device=self.device, dtype=self.dtype)
        if y_t.ndim > 1:
            y_t = y_t.squeeze(-1)

        self._init_module(X, y_t)
        self.X_ = X

        # Build the normalised-Laplacian affinity matrix S
        S = self._build_graph(X)                        # (n, n)

        # Initial label distribution Y0 (the clamping target)
        Y0 = self._label_matrix(y_t)                    # (n, n_classes)

        # Start from Y0
        f = Y0.clone()

        # --- Label Spreading iteration ---
        # F_new = alpha * S @ F + (1 - alpha) * Y0
        # No hard-clamping; labelled rows are pulled back softly via Y0.
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            F_prev = f.clone()

            f = self.alpha * (S @ f) + (1.0 - self.alpha) * Y0

            # Row-normalise so probabilities sum to 1
            row_sums = f.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            f = f / row_sums

            # Convergence check
            delta = (f - F_prev).abs().max().item()
            if delta < self.tol:
                break

        self.n_iter_ = n_iter
        self.label_distributions_ = f

        # Transduction: assign the argmax class to every sample
        argmax_idx = torch.argmax(f, dim=-1)            # (n_samples,)
        self.transduction_ = self.classes_[argmax_idx]  # (n_samples,)
        self.fit_status = True
        return self

   
class SelfTrainingClassifier(MLClassifier):
    def __init__(self,
                 estimator: MLClassifier = None,
                 threshold: float = 0.75,
                 criterion: str = "threshold",
                 k_best: int = 10,
                 max_iter: int = 10,
                 verbose: bool = False,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        from ...classification import DecisionTreeClassifier
        self.estimator_ = estimator if estimator is not None else DecisionTreeClassifier(*args, **kwargs)
        self.threshold = threshold
        self.criterion = criterion.lower() if isinstance(criterion, str) else "threshold"
        self.k_best = k_best
        self.max_iter = max_iter
        self.verbose = verbose
        self.device = device
        self.dtype = dtype
        # Attributes set during fit
        self.classes_ = None
        self.transduction_ = None
        self.labeled_iter_ = None
        self.n_features_in_ = None
        self.n_iter_ = None
        self.termination_condition_ = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        sd['_self_training_metadata'] = {
            'classes_': getattr(self, 'classes_', None),
            'n_classes_': getattr(self, 'n_classes_', None),
            'n_features_in_': getattr(self, 'n_features_in_', None),
            'n_iter_': getattr(self, 'n_iter_', None),
            'termination_condition_': getattr(self, 'termination_condition_', None),
            # Save base estimator state dict
            'estimator_state': self.estimator_.state_dict() if hasattr(self, 'estimator_') and hasattr(self.estimator_, 'state_dict') else None
        }
        return sd

    def load_state_dict(self, state_dict, strict=True, *args, **kwargs):
        metadata = state_dict.pop('_self_training_metadata', None)
        try:
            super().load_state_dict(state_dict, strict=strict, *args, **kwargs)
        except RuntimeError:
            super().load_state_dict(state_dict, strict=False, *args, **kwargs)
            
        if metadata:
            self.classes_ = metadata.get('classes_')
            self.n_classes_ = metadata.get('n_classes_')
            self.n_features_in_ = metadata.get('n_features_in_')
            self.n_iter_ = metadata.get('n_iter_')
            self.termination_condition_ = metadata.get('termination_condition_')
            
            # Restore base estimator state dict
            est_state = metadata.get('estimator_state')
            if est_state is not None and hasattr(self, 'estimator_') and hasattr(self.estimator_, 'load_state_dict'):
                try:
                    self.estimator_.load_state_dict(est_state, strict=True)
                except RuntimeError:
                    self.estimator_.load_state_dict(est_state, strict=False)
        return self

    def _init_module(self, X: torch.Tensor, y: torch.Tensor):
        """Initialise feature/class metadata from the labelled portion of y."""
        self.n_features_in_ = X.size(-1)
        # classes_ is taken from the trained estimator after the first fit;
        # pre-populate from the known labelled labels (exclude -1).
        labelled_mask = y != -1
        if labelled_mask.any():
            self.classes_ = torch.unique(y[labelled_mask])
        else:
            raise ValueError("SelfTrainingClassifier requires at least one labeled sample (y != -1).")
        return self

    def _select_pseudo_labels(
        self,
        proba: torch.Tensor,
        unlabelled_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select which unlabelled samples get pseudo-labels this iteration.

        Parameters
        ----------
        proba : torch.Tensor of shape (n_unlabelled, n_classes)
            Predicted probabilities for each unlabelled sample.
        unlabelled_indices : torch.Tensor of shape (n_unlabelled,)
            Global indices of the currently unlabelled samples.

        Returns
        -------
        selected_global_idx : torch.Tensor
            Global indices of samples chosen for pseudo-labelling.
        pseudo_labels : torch.Tensor
            Predicted class labels for the selected samples.
        """
        max_proba, pred_class_local = proba.max(dim=-1)   # (n_unlabelled,)

        if self.criterion == "threshold":
            # Select samples whose max probability exceeds the threshold
            mask = max_proba >= self.threshold
            selected_local = mask.nonzero(as_tuple=True)[0]

        elif self.criterion == "k_best":
            # Select the top-k samples by max probability
            k = min(self.k_best, proba.size(0))
            _, selected_local = torch.topk(max_proba, k=k, largest=True, sorted=False)

        else:
            raise ValueError(
                f"Unknown criterion '{self.criterion}'. "
                "Valid options: 'threshold', 'k_best'."
            )

        selected_global_idx = unlabelled_indices[selected_local]
        # Map local argmax indices to actual class labels
        self._ensure_classes_()
        if self.classes_ is None:
            raise RuntimeError("SelfTrainingClassifier.classes_ is None; cannot map predictions to labels.")
        pseudo_labels = self.classes_[pred_class_local[selected_local]]
        return selected_global_idx, pseudo_labels

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, data_or_X, y=None, **kwargs):
        """Fit the self-training classifier.

        Iteratively fits the base estimator on labelled data, predicts
        pseudo-labels for unlabelled samples (those with label -1), selects
        the most confident predictions according to ``criterion``, adds them
        to the labelled pool, and repeats until a termination condition is met.

        Parameters
        ----------
        data_or_X : array-like of shape (n_samples, n_features)
            Training data.  Unlabelled samples must have ``y == -1``.
        y : array-like of shape (n_samples,)
            Labels.  Use -1 to denote an unlabelled sample.

        Returns
        -------
        self
        """
        if hasattr(data_or_X, "columns"):
            self.feature_names_in_ = list(data_or_X.columns)

        X = torch.as_tensor(data_or_X, dtype=self.dtype, device=self.device)
                    
        y_orig = torch.as_tensor(y, device=self.device, dtype=self.dtype)
        if y_orig.ndim > 1:
            y_orig = y_orig.squeeze(-1)

        self._init_module(X, y_orig)
        n_samples = X.size(0)

        # Working copy of labels (will be updated with pseudo-labels)
        y_work = y_orig.clone()

        # labeled_iter_: 0 = originally labelled, -1 = never labelled, >0 = iteration
        labeled_iter = torch.full((n_samples,), -1, dtype=torch.long, device=self.device)
        labeled_iter[y_orig != -1] = 0

        termination_condition = "max_iter"
        n_iter = 0

        # Initial fit on labelled data is mandatory to ensure estimator is fit even if no unlabelled samples exist
        labelled_mask_init = y_work != -1
        if labelled_mask_init.any():
            self.estimator_.fit(X[labelled_mask_init], y_work[labelled_mask_init], **kwargs)
            if hasattr(self.estimator_, "classes_") and self.estimator_.classes_ is not None:
                self.classes_ = self.estimator_.classes_
            if hasattr(self.estimator_, "n_classes_") and self.estimator_.n_classes_ is not None:
                self.n_classes_ = self.estimator_.n_classes_

        for n_iter in range(1, (self.max_iter + 1) if self.max_iter is not None else 10 ** 9):
            # Identify currently labelled / unlabelled samples
            labelled_mask = y_work != -1
            unlabelled_mask = ~labelled_mask

            # If all samples are labelled, stop
            if not unlabelled_mask.any():
                termination_condition = "all_labeled"
                n_iter -= 1   # last fit already done outside loop
                break

            labelled_idx = labelled_mask.nonzero(as_tuple=True)[0]
            unlabelled_idx = unlabelled_mask.nonzero(as_tuple=True)[0]

            X_labelled = X[labelled_idx]
            y_labelled = y_work[labelled_idx]
            X_unlabelled = X[unlabelled_idx]

            # Fit estimator on currently labelled data
            fitted_est = deepcopy(self.estimator_)
            fitted_est.fit(X_labelled, y_labelled, **kwargs)

            # Update classes_ from the fitted estimator if available
            if hasattr(fitted_est, "classes_") and fitted_est.classes_ is not None:
                self.classes_ = fitted_est.classes_
            if hasattr(fitted_est, "n_classes_") and fitted_est.n_classes_ is not None:
                self.n_classes_ = fitted_est.n_classes_

            # Predict probabilities for unlabelled samples
            with torch.no_grad():
                proba = fitted_est.predict_proba(X_unlabelled)  # (n_unlabelled, n_classes)

            # Select pseudo-labels
            selected_global, pseudo_labels = self._select_pseudo_labels(proba, unlabelled_idx)

            if self.verbose:
                print(
                    f"[SelfTrainingClassifier] iter {n_iter}: "
                    f"{selected_global.numel()} pseudo-labels added "
                    f"(total unlabelled: {unlabelled_idx.numel()})."
                )

            # No new pseudo-labels → stop
            if selected_global.numel() == 0:
                # Do final fit on all currently labelled data before stopping
                fitted_est_final = deepcopy(self.estimator_)
                labelled_mask_final = y_work != -1
                fitted_est_final.fit(
                    X[labelled_mask_final],
                    y_work[labelled_mask_final],
                    **kwargs
                )
                self.estimator_ = fitted_est_final
                termination_condition = "no_change"
                break

            # Add selected pseudo-labels to working label array
            y_work[selected_global] = pseudo_labels
            labeled_iter[selected_global] = n_iter

            # Check if all are now labelled after this addition
            if (y_work != -1).all():
                # Final fit with all labels known
                fitted_est_final = deepcopy(self.estimator_)
                fitted_est_final.fit(X, y_work, **kwargs)
                self.estimator_ = fitted_est_final
                termination_condition = "all_labeled"
                break

            # Store the estimator fitted this iteration (will be overwritten next iter)
            self.estimator_ = fitted_est

        else:
            # Loop exhausted max_iter without breaking
            termination_condition = "max_iter"
            # Do a final fit with whatever labels we have
            labelled_mask_final = y_work != -1
            if labelled_mask_final.any():
                fitted_est_final = deepcopy(self.estimator_)
                fitted_est_final.fit(
                    X[labelled_mask_final],
                    y_work[labelled_mask_final],
                    **kwargs
                )
                self.estimator_ = fitted_est_final

        # Store results
        self.transduction_ = y_work           # includes pseudo-labels
        self.labeled_iter_ = labeled_iter
        self.n_iter_ = n_iter
        self.termination_condition_ = termination_condition

        # Sync classes_ with the final estimator
        if hasattr(self.estimator_, "classes_") and self.estimator_.classes_ is not None:
            self.classes_ = self.estimator_.classes_

        self.fit_status = True
        return self

    def _ensure_classes_(self) -> None:
        """Propagate classes_ from estimator_ if None (e.g. after load)."""
        if self.classes_ is None and hasattr(self, "estimator_") and self.estimator_ is not None:
            if hasattr(self.estimator_, "classes_") and self.estimator_.classes_ is not None:
                self.classes_ = self.estimator_.classes_

    # ------------------------------------------------------------------
    # Prediction API — delegate entirely to the fitted estimator_
    # ------------------------------------------------------------------

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """Return probability estimates from the fitted estimator.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)

        Returns
        -------
        proba : torch.Tensor of shape (n_samples, n_classes)
        """
        if not self.fit_status:
            raise RuntimeError(
                "SelfTrainingClassifier must be fitted before calling predict_proba."
            )
        self._ensure_classes_()
        X = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        return self.estimator_.predict_proba(X)

    def predict_log_proba(self, X: torch.Tensor) -> torch.Tensor:
        """Return log-probability estimates from the fitted estimator.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)

        Returns
        -------
        log_proba : torch.Tensor of shape (n_samples, n_classes)
        """
        return torch.log(self.predict_proba(X).clamp(min=1e-12))

    def decision_function(self, X: torch.Tensor) -> torch.Tensor:
        """Compute decision scores from the fitted estimator.

        Delegates to ``estimator_.decision_function`` if it exists,
        otherwise returns the log-probabilities as a proxy.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)

        Returns
        -------
        scores : torch.Tensor of shape (n_samples,) or (n_samples, n_classes)
        """
        if not self.fit_status:
            raise RuntimeError(
                "SelfTrainingClassifier must be fitted before calling decision_function."
            )
        X = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        if hasattr(self.estimator_, "decision_function"):
            return self.estimator_.decision_function(X)
        # Fallback: log-probabilities serve as decision scores
        return self.predict_log_proba(X)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)

        Returns
        -------
        y_pred : torch.Tensor of shape (n_samples,)
            Predicted class labels.
        """
        if not self.fit_status:
            raise RuntimeError(
                "SelfTrainingClassifier must be fitted before calling predict."
            )
        self._ensure_classes_()
        if self.classes_ is None:
            raise RuntimeError("SelfTrainingClassifier.classes_ is None; cannot predict. Ensure the model was fitted successfully.")
        X = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        proba = self.predict_proba(X)              # (n_samples, n_classes)
        idx = torch.argmax(proba, dim=-1)          # (n_samples,)
        return self.classes_[idx]

    def forward(self, X: torch.Tensor, y: torch.Tensor = None, **kwargs) -> torch.Tensor:
        if not self.fit_status:
            self.fit(X, y, **kwargs)
        return self.predict(X)


class TunedThresholdClassifierCV(MLClassifier):
    def __init__(self,
                 estimator: MLClassifier = None,
                 scoring: Union[str, Callable, nn.Module] = "balanced_accuracy",
                 response_method: str = "auto",
                 thresholds: Union[int, float, List[float], Tuple[float], torch.Tensor] = 100,
                 cv: Union[str, Callable, Iterable, MLModule] = None,
                 cv_config: dict = None,
                 refit: bool = True,
                 n_jobs: int = None,
                 random_state: int = None,
                 store_cv_results: bool = False,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        if isinstance(thresholds, int):
            self.thresholds = torch.linspace(0, 1, thresholds, device=device, dtype=dtype).tolist()
        elif isinstance(thresholds, float):
            self.thresholds = [thresholds]
        elif isinstance(thresholds, (list, tuple)):
            self.thresholds = [*thresholds]
        else:
            self.thresholds = thresholds.tolist()
        self.estimator_ = FixedThresholdClassifier(
            estimator=estimator,
            threshold=self.thresholds[0],
            response_method=response_method,
            device=device,
            dtype=dtype,
            *args, **kwargs
        )

        self.device = device
        self.dtype = dtype
        if random_state is not None:
            torch.manual_seed(random_state)
        # Guard: cv_config may be None; build a mutable copy before inserting random_state
        cv_config = dict(cv_config) if cv_config is not None else {}
        if random_state is not None:
            cv_config["random_state"] = random_state
        self.param_dict = {
            "threshold": self.thresholds
        }
        self.search_params = {
            "estimator": self.estimator_,
            "param_grid": self.param_dict,
            "scoring": scoring,
            "n_jobs": n_jobs,
            "refit": refit,
            "cv": cv,
            "cv_config": cv_config,
            "return_train_score": True,
            "store_cv_values": store_cv_results,
            "device": device,
            "dtype": dtype,
        }
        from ...cross_validation.search_cv import GridSearchCV
        self.search = GridSearchCV(**self.search_params)

        self.best_threshold_ = None
        self.best_score_ = None
        self.cv_results_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.n_features_in_ = None

    def _init_module(self, X: torch.Tensor, y: torch.Tensor):
        self.n_features_in_ = X.size(-1)
        self.classes_ = torch.unique(y)
        self.n_classes_ = self.classes_.size(0)

    def fit(self, data_or_X, y=None, **kwargs):
        X = torch.as_tensor(data_or_X, dtype=self.dtype, device=self.device)
        y = torch.as_tensor(y, dtype=self.dtype, device=self.device)
        self._init_module(X, y)
        self.search.fit(X, y)
        self.estimator_ = self.search.best_estimator_
        self.best_score_ = self.search.best_score_
        self.best_threshold_ = self.search.best_params_["threshold"]
        self.cv_results_ = self.search.cv_results_
        self.fit_status = True
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if hasattr(self.estimator_, "to") and self.estimator_ is not None:
            self.estimator_.to(self.device)
        return self.estimator_.predict(X)

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        if hasattr(self.estimator_, "to") and self.estimator_ is not None:
            self.estimator_.to(self.device)
        return self.estimator_.predict_proba(X)

    def predict_log_proba(self, X: torch.Tensor) -> torch.Tensor:
        if hasattr(self.estimator_, "to") and self.estimator_ is not None:
            self.estimator_.to(self.device)
        return self.estimator_.predict_log_proba(X)

    def decision_function(self, X: torch.Tensor) -> torch.Tensor:
        if hasattr(self.estimator_, "to") and self.estimator_ is not None:
            self.estimator_.to(self.device)
        return self.estimator_.decision_function(X)

    def forward(self, X: torch.Tensor, y: torch.Tensor = None, **kwargs) -> torch.Tensor:
        if not self.fit_status:
            self.fit(X, y, **kwargs)
        return self.predict(X)


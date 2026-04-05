"""Sklearn-backed ensemble classifiers for thesis JSON configs (Track A tabular)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from Code.models.utils.utils import MLClassifier, _to_numpy

__all__ = ["RandomForestClassifier"]


class RandomForestClassifier(MLClassifier):
    """Wraps ``sklearn.ensemble.RandomForestClassifier``; full state via joblib in ``train_single``."""

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Any = "sqrt",
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None,
        verbose: int = 0,
        warm_start: bool = False,
        class_weight: Optional[Any] = None,
        ccp_alpha: float = 0.0,
        max_samples: Optional[Any] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples
        self.device = device
        self.dtype = dtype
        self._clf = None
        self.classes_: Optional[torch.Tensor] = None
        self.n_classes_: Optional[int] = None
        self.n_features_in_: Optional[int] = None

    def _sklearn_kwargs(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "min_weight_fraction_leaf": self.min_weight_fraction_leaf,
            "max_features": self.max_features,
            "max_leaf_nodes": self.max_leaf_nodes,
            "min_impurity_decrease": self.min_impurity_decrease,
            "bootstrap": self.bootstrap,
            "oob_score": self.oob_score,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "warm_start": self.warm_start,
            "class_weight": self.class_weight,
            "ccp_alpha": self.ccp_alpha,
            "max_samples": self.max_samples,
        }

    def fit(self, data_or_X, y=None, **kwargs):
        from sklearn.ensemble import RandomForestClassifier as SklearnRF

        X = np.asarray(_to_numpy(data_or_X), dtype=np.float64)
        if y is None:
            raise ValueError("y required")
        yn = np.asarray(y).reshape(-1).astype(np.int64)
        self._clf = SklearnRF(**self._sklearn_kwargs())
        self._clf.fit(X, yn)
        self.classes_ = torch.tensor(np.array(self._clf.classes_, dtype=np.int64), dtype=torch.long)
        self.n_classes_ = int(len(self._clf.classes_))
        self.n_features_in_ = int(X.shape[1])
        self.fit_status = True
        return self

    def predict(self, X):
        if self._clf is None:
            raise RuntimeError("RandomForestClassifier is not fitted")
        Xn = np.asarray(_to_numpy(X), dtype=np.float64)
        pred = self._clf.predict(Xn)
        out = torch.tensor(pred.astype(np.int64), dtype=torch.long, device=self.device)
        return out

    def predict_proba(self, X):
        if self._clf is None:
            raise RuntimeError("RandomForestClassifier is not fitted")
        Xn = np.asarray(_to_numpy(X), dtype=np.float64)
        p = self._clf.predict_proba(Xn)
        return torch.tensor(p, dtype=self.dtype, device=self.device)

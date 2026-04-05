import warnings
import torch
import torch.nn as nn
from typing import Optional, Callable, Union, Any, List, Tuple, Dict, Literal, Iterable
from ....models.utils import MLModule
import pandas as pd  # pyright: ignore[reportMissingImports]
import numpy as np
from torch.func import vmap
import joblib


__all__ = ["DictVectorizer"]


class DictVectorizer(MLModule):
    def __init__(self,
                 separator: str = "=",
                 sparse: bool = True,
                 sort: bool = True,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.separator = separator
        self.sparse = sparse
        self.sort = sort
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.args = args
        self.kwargs = kwargs
        # Fitted attributes
        self.vocabulary_: Dict[str, int] = {}
        self.feature_names_: List[str] = []
        self.fit_status = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_features(self, d: dict) -> List[Tuple[str, float]]:
        """Return (feature_name, value) pairs from a single sample dict."""
        if isinstance(d, torch.Tensor):
            # Fallback: convert 1D tensor to dict (indices as keys)
            if d.dim() == 1:
                d = {str(i): float(v.item()) for i, v in enumerate(d.tolist())}
            else:
                raise TypeError("DictVectorizer expects dict-like samples with .items(), not Tensor.")
        features: List[Tuple[str, float]] = []
        for key, val in d.items():
            key_str = str(key)
            if isinstance(val, str):
                features.append((f"{key_str}{self.separator}{val}", 1.0))
            elif isinstance(val, (list, tuple, set, frozenset)):
                for v in val:
                    if isinstance(v, str):
                        features.append((f"{key_str}{self.separator}{v}", 1.0))
                    else:
                        try:
                            features.append((key_str, float(v)))
                        except (TypeError, ValueError):
                            pass
            else:
                try:
                    features.append((key_str, float(val)))
                except (TypeError, ValueError):
                    pass
        return features

    def _load_input(self, X: Any) -> List[dict]:
        """Accept list/tuple/array of dicts, or a single dict, or tensor (auto-converted)."""
        if isinstance(X, dict):
            return [X]
        if isinstance(X, torch.Tensor):
            # Auto-convert tensor to list of dicts (each row -> dict of col_idx: value)
            if X.dim() == 1:
                return [{str(i): float(v.item()) for i, v in enumerate(X.tolist())}]
            return [{str(j): float(X[i, j].item()) for j in range(X.shape[1])} for i in range(X.shape[0])]
        return list(X)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X: Any,
        y: Optional[Any] = None,
        **kwargs,
    ) -> "DictVectorizer":
        samples = self._load_input(X)
        vocab: Dict[str, int] = {}
        for d in samples:
            for fname, _ in self._extract_features(d):
                if fname not in vocab:
                    vocab[fname] = len(vocab)

        if self.sort:
            self.feature_names_ = sorted(vocab.keys())
            self.vocabulary_ = {name: i for i, name in enumerate(self.feature_names_)}
        else:
            self.vocabulary_ = vocab
            self.feature_names_ = list(vocab.keys())

        self.fit_status = True
        return self

    def transform(
        self,
        X: Any,
        **kwargs,
    ) -> torch.Tensor:
        if not self.fit_status:
            raise RuntimeError("DictVectorizer is not fitted. Call fit() first.")

        samples = self._load_input(X)
        n_samples = len(samples)
        n_features = len(self.vocabulary_)
        result = torch.zeros(n_samples, n_features, device=self.device, dtype=self.dtype)

        for i, d in enumerate(samples):
            for fname, val in self._extract_features(d):
                if fname in self.vocabulary_:
                    j = self.vocabulary_[fname]
                    result[i, j] += val

        if self.sparse:
            return result.to_sparse()
        return result

    def fit_transform(
        self,
        X: Any,
        y: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Fit to X, then transform X."""
        return self.fit(X, y, **kwargs).transform(X)

    def inverse_transform(
        self,
        X: Any,
        dict_type: type = dict,
    ) -> List[dict]:
        if not self.fit_status:
            raise RuntimeError("DictVectorizer is not fitted. Call fit() first.")

        if isinstance(X, torch.Tensor):
            if X.is_sparse:
                X = X.to_dense()
        else:
            X = torch.as_tensor(X, device=self.device, dtype=self.dtype)

        result: List[dict] = []
        for row in X:
            d = dict_type()
            for j, val in enumerate(row):
                v = val.item()
                if v != 0.0:
                    d[self.feature_names_[j]] = v
            result.append(d)
        return result

    def get_feature_names_out(self) -> List[str]:
        """Return feature names in the order they appear in the output."""
        if not self.fit_status:
            raise RuntimeError("DictVectorizer is not fitted. Call fit() first.")
        return list(self.feature_names_)

    def restrict(self, support: Union[torch.Tensor, List[bool], List[int]], indices: bool = False) -> "DictVectorizer":
        if not self.fit_status:
            raise RuntimeError("DictVectorizer is not fitted. Call fit() first.")

        if indices:
            keep = list(support)
        else:
            keep = [i for i, s in enumerate(support) if s]

        new_names = [self.feature_names_[i] for i in keep]
        self.feature_names_ = new_names
        self.vocabulary_ = {name: i for i, name in enumerate(new_names)}
        return self

    def forward(
        self,
        X: Any,
        y: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        if not self.fit_status:
            return self.fit_transform(X, y, **kwargs)
        return self.transform(X, **kwargs)

import warnings
import torch
import torch.nn as nn
from typing import Optional, Callable, Union, Any, List, Tuple, Dict, Literal
from .....models.utils import MLModule
import pandas as pd  # pyright: ignore[reportMissingImports]
import numpy as np
from torch.func import vmap
import joblib

__all__ = [
    "ContinuousFeatureDiscretizer",
    "LabelEncoder",
    "LabelBinarizer",
    "MultiLabelBinarizer",
    "OneHotEncoder",
    "OrdinalEncoder",
    "TargetEncoder",
    "KBinsDiscretizer",
]


def _mean_agg(x: torch.Tensor) -> torch.Tensor:
    return torch.mean(x)


def _median_agg(x: torch.Tensor) -> torch.Tensor:
    out = torch.median(x)
    return out.values if isinstance(out, tuple) else out


def _rms_agg(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(x ** 2))


def _mean_abs_agg(x: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(x))


def _is_missing(val: Any) -> bool:
    """Return True if *val* represents a missing / NaN entry."""
    if val is None:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    return False


def _compute_cats_for_feature(
        col_values: List[Any],
        freq_counts: Dict[Any, int],
        n_samples: int,
        min_frequency: Optional[Union[int, float]],
        max_categories: Optional[int],
) -> Tuple[List[Any], Optional[List[Any]]]:
    if min_frequency is not None:
        if isinstance(min_frequency, float):
            min_freq = min_frequency * n_samples
        else:
            min_freq = float(min_frequency)
        frequent: List[Any] = [v for v in col_values if freq_counts.get(v, 0) >= min_freq]
        infrequent: List[Any] = [v for v in col_values if freq_counts.get(v, 0) < min_freq]
    else:
        frequent = list(col_values)
        infrequent = []

    if max_categories is not None:
        n_total = len(frequent) + (1 if infrequent else 0)
        if n_total > max_categories:
            n_keep = max(max_categories - 1, 0)
            sorted_by_freq = sorted(
                frequent, key=lambda v: freq_counts.get(v, 0), reverse=True
            )
            infrequent = sorted_by_freq[n_keep:] + infrequent
            frequent = sorted_by_freq[:n_keep]

    try:
        frequent = sorted(frequent)
    except TypeError:
        frequent = sorted(frequent, key=str)

    return frequent, (infrequent if infrequent else None)


class ContinuousFeatureDiscretizer(MLModule):
    """
    Discretize continuous features by iterative epsilon-based merging.

    Algorithm (per feature):
    1. Sort values in ascending order (keep original indices).
    2. Compute differences between consecutive sorted values.
    3. Merge consecutive values when diff in [diff - eps, diff + eps] (i.e., diff <= eps).
    4. Replace each merged group with its aggregate (mean, rms, mean_abs, etc.).
    5. Repeat steps 2–4 until no more merges occur.
    6. Assign labels 1, 2, ..., p_classes_i to the final groups.
    7. Map labels back to original sample order.
    """

    _MERGE_FUNCS = {
        "mean": _mean_agg,
        "median": _median_agg,
        "rms": _rms_agg,
        "abs": _mean_abs_agg,
    }

    def __init__(
            self,
            eps: float = 1e-5,
            merge_func: Union[Literal["mean", "median", "rms", "abs"], Callable] = "mean",
            device: Union[str, torch.device] = "cpu",
            dtype: torch.dtype = torch.float,
            *args,
            **kwargs,
    ):
        super().__init__()
        self.eps = eps
        self.merge_func = (
            self._MERGE_FUNCS.get(merge_func, merge_func)
            if isinstance(merge_func, str)
            else merge_func
        )
        self.device = device
        self.dtype = dtype
        self.n_features_in_: Optional[int] = None
        self.codebook_: Dict[int, Dict[int, torch.Tensor]] = {}

    def _aggregate(self, group: torch.Tensor) -> torch.Tensor:
        """Apply merge function to a group tensor."""
        out = self.merge_func(group)
        if isinstance(out, tuple):
            out = out[0]
        elif hasattr(out, "values") and not isinstance(out, torch.Tensor):
            out = out.values
        # Handle callable return (e.g. lambda x: x.mean returns a method)
        if callable(out) and not isinstance(out, torch.Tensor):
            out = out()
        if not isinstance(out, torch.Tensor):
            out = torch.tensor(float(out), dtype=self.dtype, device=group.device)
        return out.to(dtype=self.dtype, device=group.device)

    def _merge_groups_iterative(
            self, values: torch.Tensor, eps: float
    ) -> List[torch.Tensor]:
        if values.numel() == 0:
            return []
        if values.numel() == 1:
            return [values.clone()]

        # Start: each value is its own group (values is 1D)
        groups: List[torch.Tensor] = [values[j: j + 1].clone() for j in range(values.numel())]
        changed = True
        while changed:
            changed = False
            n_groups = len(groups)
            if n_groups <= 1:
                break

            reps = torch.stack([self._aggregate(g) for g in groups], dim=0)
            new_groups: List[torch.Tensor] = []
            i = 0
            while i < n_groups:
                start = i
                while i + 1 < n_groups and (reps[i + 1] - reps[i]).item() <= eps:
                    i += 1
                    changed = True
                end = i + 1
                merged = torch.cat([groups[j] for j in range(start, end)])
                new_groups.append(merged)
                i = end
            groups = new_groups

        return groups

    def fit(self, data_or_X, y=None, **kwargs):
        X = torch.as_tensor(data_or_X, dtype=self.dtype, device=self.device)
        if X.dim() == 1:
            X = X.unsqueeze(1)
        n, d = X.shape
        self.n_features_in_ = d
        self.codebook_ = {}

        for i in range(d):
            col = X[:, i]
            sorted_vals, orig_idx = torch.sort(col)

            groups = self._merge_groups_iterative(sorted_vals, self.eps)

            label_to_repr: Dict[int, torch.Tensor] = {}
            sorted_labels = torch.zeros(n, dtype=torch.long, device=X.device)
            pos = 0
            for label, group in enumerate(groups, start=1):
                repr_val = self._aggregate(group)
                label_to_repr[label] = repr_val
                group_size = group.numel()
                sorted_labels[pos: pos + group_size] = label
                pos += group_size

            self.codebook_[i] = label_to_repr

        self.fit_status = True
        return self

    def transform(self, data_or_X, **kwargs) -> torch.Tensor:
        if not self.fit_status or not self.codebook_:
            raise RuntimeError("Discretizer must be fitted before transform.")
        X = torch.as_tensor(data_or_X, dtype=self.dtype, device=self.device)
        if X.dim() == 1:
            X = X.unsqueeze(1)
        n, d = X.shape
        X_discrete = torch.zeros(n, d, dtype=torch.long, device=X.device)

        for i in range(d):
            col = X[:, i]
            mapping = self.codebook_[i]
            labels = sorted(mapping.keys())
            repr_vals = torch.stack(
                [mapping[l].to(X.device) for l in labels], dim=0
            )
            dists = torch.abs(col.unsqueeze(1) - repr_vals.unsqueeze(0))
            closest = torch.argmin(dists, dim=1)
            labels_tensor = torch.tensor(labels, device=X.device, dtype=torch.long)
            X_discrete[:, i] = labels_tensor[closest]
        return X_discrete

    def inverse_transform(self, data_or_X) -> torch.Tensor:
        if not self.fit_status or not self.codebook_:
            raise RuntimeError("Discretizer must be fitted before inverse_transform.")
        X = torch.as_tensor(data_or_X, dtype=torch.long, device=self.device)
        if X.dim() == 1:
            X = X.unsqueeze(1)
        n, d = X.shape
        X_recon = torch.zeros(n, d, dtype=self.dtype, device=X.device)

        for i in range(d):
            col = X[:, i]
            mapping = self.codebook_[i]
            for label, repr_val in mapping.items():
                mask = col == label
                if mask.any():
                    X_recon[mask, i] = repr_val.to(X.device)
        return X_recon

    def fit_transform(self, data_or_X, y=None, **kwargs) -> torch.Tensor:
        return self.fit(data_or_X, y, **kwargs).transform(data_or_X, **kwargs)

    def forward(
            self, X: torch.Tensor, y: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        return self.fit_transform(X, y, **kwargs)


def _labelencoder_to_columns(data: Any) -> Tuple[List[List[Any]], bool]:
    if isinstance(data, pd.DataFrame):
        return [data.iloc[:, i].tolist() for i in range(data.shape[1])], False
    if isinstance(data, pd.Series):
        return [data.tolist()], True
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            return [data.tolist()], True
        return [data[:, i].tolist() for i in range(data.shape[1])], False
    if isinstance(data, torch.Tensor):
        arr = data.detach().cpu().numpy()
        if arr.ndim == 1:
            return [arr.tolist()], True
        return [arr[:, i].tolist() for i in range(arr.shape[1])], False
    if isinstance(data, (list, tuple)):
        if not data:
            return [[]], True
        first = data[0]
        if isinstance(first, (list, tuple)) and len(first) > 0:
            return [[row[i] for row in data] for i in range(len(first))], False
        return [list(data)], True
    return [[data]], True


class LabelEncoder(MLModule):
    def __init__(
            self,
            device: Union[str, torch.device] = "cpu",
            dtype: torch.dtype = torch.long,
            handle_unknown: Literal["error", "ignore"] = "error",
            *args,
            **kwargs,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.classes_: Optional[List[Any]] = None
        self._class_to_idx: Dict[Any, int] = {}
        self._idx_to_class: Dict[int, Any] = {}

    def fit(self, data_or_X, y=None, **kwargs) -> "LabelEncoder":
        cols, _ = _labelencoder_to_columns(data_or_X)
        col = cols[0]
        try:
            unique = sorted(set(col))
        except TypeError:
            unique = sorted(set(col), key=str)
        self.classes_ = unique
        self._class_to_idx = {c: i for i, c in enumerate(unique)}
        self._idx_to_class = {i: c for i, c in enumerate(unique)}
        self.fit_status = True
        return self

    def transform(self, data_or_X, **kwargs) -> torch.Tensor:
        if not self.fit_status:
            raise RuntimeError("LabelEncoder must be fitted before transform.")
        cols, _ = _labelencoder_to_columns(data_or_X)
        col = cols[0]
        result: List[int] = []
        for val in col:
            if val in self._class_to_idx:
                result.append(self._class_to_idx[val])
            elif self.handle_unknown == "error":
                raise ValueError(
                    f"Unknown label encountered: {val!r}. "
                    f"Set handle_unknown='ignore' to suppress."
                )
            else:
                result.append(-1)
        return torch.tensor(result, dtype=self.dtype, device=self.device)

    def inverse_transform(self, data_or_X) -> List[Optional[Any]]:
        cols, _ = _labelencoder_to_columns(data_or_X)
        col = cols[0]
        return [self._idx_to_class.get(int(v), None) for v in col]

    def fit_transform(self, data_or_X, y=None, **kwargs) -> torch.Tensor:
        return self.fit(data_or_X, y, **kwargs).transform(data_or_X, **kwargs)

    def forward(
            self, X: torch.Tensor, y: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        return self.fit_transform(X, y, **kwargs)


class LabelBinarizer(MLModule):
    def __init__(
            self,
            neg_label: int = 0,
            pos_label: int = 1,
            sparse_output: bool = False,
            device: Union[str, torch.device] = "cpu",
            dtype: torch.dtype = torch.long,
            handle_unknown: Literal["error", "ignore"] = "error",
            *args,
            **kwargs,
    ):
        super().__init__()
        self.neg_label = neg_label
        self.pos_label = pos_label
        self.sparse_output = sparse_output
        self.device = device
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.classes_: Optional[List[Any]] = None
        self.y_type_: str = ""
        self.sparse_input_: bool = False
        self._class_to_idx: Dict[Any, int] = {}

    def fit(self, data_or_X, y=None, **kwargs) -> "LabelBinarizer":
        cols, _ = _labelencoder_to_columns(data_or_X)
        col = cols[0]
        try:
            unique = sorted(set(col))
        except TypeError:
            unique = sorted(set(col), key=str)
        self.classes_ = unique
        self._class_to_idx = {c: i for i, c in enumerate(unique)}
        self.y_type_ = "binary" if len(unique) <= 2 else "multiclass"
        self.fit_status = True
        return self

    def transform(self, data_or_X, **kwargs) -> torch.Tensor:
        if not self.fit_status:
            raise RuntimeError("LabelBinarizer must be fitted before transform.")
        cols, _ = _labelencoder_to_columns(data_or_X)
        col = cols[0]
        n = len(col)
        n_classes = len(self.classes_)

        if self.y_type_ == "binary":
            result = torch.full(
                (n, 1), self.neg_label, dtype=self.dtype, device=self.device
            )
            for idx, val in enumerate(col):
                if val in self._class_to_idx:
                    if self._class_to_idx[val] == 1:
                        result[idx, 0] = self.pos_label
                elif self.handle_unknown == "error":
                    raise ValueError(f"Unknown label encountered: {val!r}")
        else:
            result = torch.full(
                (n, n_classes), self.neg_label, dtype=self.dtype, device=self.device
            )
            for idx, val in enumerate(col):
                if val in self._class_to_idx:
                    result[idx, self._class_to_idx[val]] = self.pos_label
                elif self.handle_unknown == "error":
                    raise ValueError(f"Unknown label encountered: {val!r}")

        if self.sparse_output:
            return result.to_sparse()
        return result

    def inverse_transform(
            self, Y, threshold: Optional[float] = None
    ) -> List[Any]:
        if not self.fit_status:
            raise RuntimeError("LabelBinarizer must be fitted before inverse_transform.")
        if not isinstance(Y, torch.Tensor):
            Y = torch.as_tensor(Y, dtype=torch.float32)
        if threshold is None:
            threshold = (self.neg_label + self.pos_label) / 2.0
        if Y.dim() == 1 or (Y.dim() == 2 and Y.shape[1] == 1):
            Y = Y.reshape(-1)
            return [
                self.classes_[1] if float(v) > threshold else self.classes_[0]
                for v in Y
            ]
        return [self.classes_[int(row.argmax())] for row in Y]

    def fit_transform(self, data_or_X, y=None, **kwargs) -> torch.Tensor:
        return self.fit(data_or_X, y, **kwargs).transform(data_or_X, **kwargs)

    def forward(
            self, X: torch.Tensor, y: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        return self.fit_transform(X, y, **kwargs)


class MultiLabelBinarizer(MLModule):
    def __init__(
            self,
            neg_label: int = 0,
            pos_label: int = 1,
            handle_unknown: Literal["error", "ignore"] = "error",
            device: Union[str, torch.device] = "cpu",
            dtype: torch.dtype = torch.long,
            *args,
            **kwargs,
    ):
        super().__init__()
        self.neg_label = neg_label
        self.pos_label = pos_label
        self.handle_unknown = handle_unknown
        self.device = device
        self.dtype = dtype
        self.classes_: Optional[List[Any]] = None
        self._class_to_idx: Dict[Any, int] = {}

    def fit(self, data_or_X, y=None, **kwargs) -> "MultiLabelBinarizer":
        all_labels: set = set()
        for sample in data_or_X:
            for label in sample:
                all_labels.add(label)
        try:
            unique = sorted(all_labels)
        except TypeError:
            unique = sorted(all_labels, key=str)
        self.classes_ = unique
        self._class_to_idx = {c: i for i, c in enumerate(unique)}
        self.fit_status = True
        return self

    def transform(self, data_or_X, **kwargs) -> torch.Tensor:
        if not self.fit_status:
            raise RuntimeError(
                "MultiLabelBinarizer must be fitted before transform."
            )
        samples = list(data_or_X)
        n = len(samples)
        n_classes = len(self.classes_)
        result = torch.full(
            (n, n_classes), self.neg_label, dtype=self.dtype, device=self.device
        )
        for i, sample in enumerate(samples):
            for label in sample:
                if label in self._class_to_idx:
                    result[i, self._class_to_idx[label]] = self.pos_label
                elif self.handle_unknown == "error":
                    raise ValueError(f"Unknown label encountered: {label!r}")
        return result

    def inverse_transform(self, Y) -> List[List[Any]]:
        if not self.fit_status:
            raise RuntimeError(
                "MultiLabelBinarizer must be fitted before inverse_transform."
            )
        if not isinstance(Y, torch.Tensor):
            Y = torch.as_tensor(Y, dtype=torch.float32)
        threshold = (self.neg_label + self.pos_label) / 2.0
        return [
            [
                self.classes_[j]
                for j in range(len(self.classes_))
                if float(Y[i, j]) > threshold
            ]
            for i in range(Y.shape[0])
        ]

    def fit_transform(self, data_or_X, y=None, **kwargs) -> torch.Tensor:
        return self.fit(data_or_X, y, **kwargs).transform(data_or_X, **kwargs)

    def forward(
            self, X: torch.Tensor, y: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        return self.fit_transform(X, y, **kwargs)


class OneHotEncoder(MLModule):
    def __init__(
            self,
            categories: Union[Literal["auto"], list, tuple,
            np.ndarray, pd.Series, pd.DataFrame,
            torch.Tensor] = 'auto',
            drop: Union[Literal["first", "if_binary"], list, tuple,
            np.ndarray, pd.Series, pd.DataFrame,
            torch.Tensor] = None,
            sparse_output: bool = True,
            handle_unknown: Union[Literal["error", "ignore",
            "infrequent_if_exist", "warn"], Callable] = 'error',
            min_frequency: Union[int, float] = None,
            max_categories: int = None,
            feature_name_combiner: Union[Literal["concat"], Callable] = 'concat',
            device: Union[str, torch.device] = "cpu",
            dtype: torch.dtype = torch.long,
            *args,
            **kwargs,
    ):
        super().__init__()
        self.categories = categories
        self.drop = drop
        self.sparse_output = sparse_output
        self.handle_unknown = handle_unknown
        self.min_frequency = min_frequency
        self.max_categories = max_categories
        self.feature_name_combiner = feature_name_combiner
        self.device = device
        self.dtype = dtype
        self.categories_: List[List[Any]] = []
        self.drop_idx_: Optional[List[Optional[int]]] = None
        self.infrequent_categories_: List[Optional[List[Any]]] = []
        self.n_features_in_: Optional[int] = None
        self.feature_names_in_: Optional[List[str]] = None

    def fit(self, data_or_X, y=None, **kwargs) -> "OneHotEncoder":
        cols, _ = _labelencoder_to_columns(data_or_X)
        n_features = len(cols)
        n_samples = len(cols[0]) if cols else 0
        self.n_features_in_ = n_features

        if isinstance(data_or_X, pd.DataFrame):
            self.feature_names_in_ = list(data_or_X.columns)
        else:
            self.feature_names_in_ = None

        self.categories_ = []
        self.infrequent_categories_ = []

        cats_provided = (
            self.categories
            if self.categories != "auto" and not isinstance(self.categories, str)
            else None
        )

        for i, col in enumerate(cols):
            freq_counts: Dict[Any, int] = {}
            for v in col:
                freq_counts[v] = freq_counts.get(v, 0) + 1

            if cats_provided is not None:
                raw = cats_provided[i]
                if isinstance(raw, (np.ndarray, pd.Series)):
                    raw = raw.tolist()
                elif isinstance(raw, torch.Tensor):
                    raw = raw.tolist()
                else:
                    raw = list(raw)
                col_values = raw
            else:
                col_values = list(freq_counts.keys())

            frequent, infrequent = _compute_cats_for_feature(
                col_values, freq_counts, n_samples,
                self.min_frequency, self.max_categories
            )
            self.categories_.append(frequent)
            self.infrequent_categories_.append(infrequent)

        # Compute drop_idx_
        if self.drop is None:
            self.drop_idx_ = None
        elif self.drop == "first":
            self.drop_idx_ = [0 if len(cats) > 0 else None for cats in self.categories_]
        elif self.drop == "if_binary":
            self.drop_idx_ = []
            for i, cats in enumerate(self.categories_):
                infreq = self.infrequent_categories_[i]
                n_total = len(cats) + (1 if infreq is not None else 0)
                self.drop_idx_.append(0 if n_total == 2 else None)
        else:
            drop_arr = self.drop
            if isinstance(drop_arr, (np.ndarray, pd.Series)):
                drop_arr = drop_arr.tolist()
            elif isinstance(drop_arr, torch.Tensor):
                drop_arr = drop_arr.tolist()
            self.drop_idx_ = []
            for i, (cats, drop_cat) in enumerate(zip(self.categories_, drop_arr)):
                if drop_cat in cats:
                    self.drop_idx_.append(cats.index(drop_cat))
                else:
                    self.drop_idx_.append(None)

        self.fit_status = True
        return self

    def transform(self, data_or_X, **kwargs) -> torch.Tensor:
        if not self.fit_status:
            raise RuntimeError("OneHotEncoder must be fitted before transform.")

        cols, _ = _labelencoder_to_columns(data_or_X)
        n = len(cols[0]) if cols else 0
        blocks: List[torch.Tensor] = []

        for i, col in enumerate(cols):
            cats = self.categories_[i]
            infreq = self.infrequent_categories_[i]
            cat_to_idx = {c: j for j, c in enumerate(cats)}
            n_freq = len(cats)
            n_total = n_freq + (1 if infreq is not None else 0)
            infreq_set: set = set(infreq) if infreq is not None else set()
            infreq_idx: Optional[int] = n_freq if infreq is not None else None

            block = torch.zeros(n, n_total, dtype=self.dtype, device=self.device)

            for row_idx, val in enumerate(col):
                if val in cat_to_idx:
                    j = int(cat_to_idx[val])
                    block[row_idx, j] = 1
                elif val in infreq_set:
                    if infreq_idx is not None:
                        block[row_idx, int(infreq_idx)] = 1
                else:
                    handle = self.handle_unknown
                    if callable(handle) and not isinstance(handle, str):
                        result_idx = handle(val, i)
                        if isinstance(result_idx, (int, np.integer)) and 0 <= int(result_idx) < n_total:
                            block[row_idx, int(result_idx)] = 1
                    elif handle == "error":
                        raise ValueError(
                            f"Found unknown category {val!r} in feature {i} "
                            f"during transform."
                        )
                    elif handle == "ignore":
                        pass  # all-zeros row for this feature
                    elif handle in ("infrequent_if_exist", "warn"):
                        if handle == "warn":
                            warnings.warn(
                                f"Found unknown category {val!r} in feature {i}. "
                                f"Encoding as infrequent.",
                                UserWarning,
                                stacklevel=2,
                            )
                        if infreq_idx is not None:
                            block[row_idx, int(infreq_idx)] = 1

            # Apply column drop
            if self.drop_idx_ is not None and i < len(self.drop_idx_):
                drop_col = self.drop_idx_[i]
                if drop_col is not None and int(drop_col) < n_total:
                    keep = [int(j) for j in range(n_total) if j != drop_col]
                    block = block[:, keep]

            blocks.append(block)

        result = (
            torch.cat(blocks, dim=1)
            if blocks
            else torch.zeros(n, 0, dtype=self.dtype, device=self.device)
        )
        if self.sparse_output:
            return result.to_sparse()
        return result

    def inverse_transform(self, X) -> List[List[Optional[Any]]]:
        if not self.fit_status:
            raise RuntimeError("OneHotEncoder must be fitted before inverse_transform.")
        if isinstance(X, torch.Tensor):
            data = X.to_dense().float() if X.is_sparse else X.float()
        else:
            data = torch.as_tensor(X, dtype=torch.float32)

        n = data.shape[0]
        result: List[List[Optional[Any]]] = [[] for _ in range(n)]
        col_offset = 0

        for i, cats in enumerate(self.categories_):
            infreq = self.infrequent_categories_[i]
            n_freq = len(cats)
            n_total_full = n_freq + (1 if infreq is not None else 0)

            drop_col: Optional[int] = None
            if self.drop_idx_ is not None and i < len(self.drop_idx_):
                drop_col = self.drop_idx_[i]

            n_out = n_total_full - (1 if drop_col is not None else 0)
            block = data[:, col_offset: col_offset + n_out]
            col_offset += n_out

            full_indices = (
                [j for j in range(n_total_full) if j != drop_col]
                if drop_col is not None
                else list(range(n_total_full))
            )

            for row_idx in range(n):
                row = block[row_idx]
                if float(row.max()) == 0:
                    result[row_idx].append(None)
                else:
                    max_col = int(row.argmax().item())
                    original_idx = full_indices[max_col]
                    result[row_idx].append(
                        cats[original_idx] if original_idx < n_freq else None
                    )

        return result

    def get_feature_names_out(
            self, input_features: Optional[List[str]] = None
    ) -> List[str]:
        if not self.fit_status:
            raise RuntimeError(
                "OneHotEncoder must be fitted before get_feature_names_out."
            )
        if input_features is not None:
            feat_names = list(input_features)
        elif self.feature_names_in_ is not None:
            feat_names = list(self.feature_names_in_)
        else:
            feat_names = [f"x{i}" for i in range(self.n_features_in_)]

        if self.feature_name_combiner == "concat" or self.feature_name_combiner is None:
            def combiner(feat: str, cat: Any) -> str:
                return f"{feat}_{cat}"
        else:
            combiner = self.feature_name_combiner  # type: ignore[assignment]

        names: List[str] = []
        for i, (feat, cats) in enumerate(zip(feat_names, self.categories_)):
            infreq = self.infrequent_categories_[i]
            drop_col: Optional[int] = None
            if self.drop_idx_ is not None and i < len(self.drop_idx_):
                drop_col = self.drop_idx_[i]

            all_cats: List[Any] = list(cats)
            if infreq is not None:
                all_cats.append("infrequent_sklearn_subgroup")

            for j, cat in enumerate(all_cats):
                if drop_col is not None and j == drop_col:
                    continue
                names.append(combiner(feat, cat))

        return names

    def fit_transform(self, data_or_X, y=None, **kwargs) -> torch.Tensor:
        return self.fit(data_or_X, y, **kwargs).transform(data_or_X, **kwargs)

    def forward(
            self, X: torch.Tensor, y: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        return self.fit_transform(X, y, **kwargs)


class OrdinalEncoder(MLModule):
    def __init__(
            self,
            categories: Union[Literal["auto"], list, tuple,
            np.ndarray, pd.Series, pd.DataFrame,
            torch.Tensor] = 'auto',
            handle_unknown: Union[Literal["error", "ignore",
            "infrequent_if_exist", "warn"], Callable] = 'error',
            unknown_value: Union[int, torch.inf] = None,
            encoded_missing_value: Union[int, torch.inf] = None,
            min_frequency: Union[int, float] = None,
            max_categories: int = None,
            device: Union[str, torch.device] = "cpu",
            dtype: torch.dtype = torch.long,
            *args,
            **kwargs,
    ):
        super().__init__()
        self.categories = categories
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.encoded_missing_value = encoded_missing_value
        self.min_frequency = min_frequency
        self.max_categories = max_categories
        self.device = device
        self.dtype = dtype
        self.categories_: List[List[Any]] = []
        self.n_features_in_: Optional[int] = None
        self.feature_names_in_: Optional[List[str]] = None
        self.infrequent_categories_: List[Optional[List[Any]]] = []

    def fit(self, data_or_X, y=None, **kwargs) -> "OrdinalEncoder":
        cols, _ = _labelencoder_to_columns(data_or_X)
        n_features = len(cols)
        n_samples = len(cols[0]) if cols else 0
        self.n_features_in_ = n_features

        if isinstance(data_or_X, pd.DataFrame):
            self.feature_names_in_ = list(data_or_X.columns)
        else:
            self.feature_names_in_ = None

        self.categories_ = []
        self.infrequent_categories_ = []

        cats_provided = (
            self.categories
            if self.categories != "auto" and not isinstance(self.categories, str)
            else None
        )

        for i, col in enumerate(cols):
            freq_counts: Dict[Any, int] = {}
            for v in col:
                if not _is_missing(v):
                    freq_counts[v] = freq_counts.get(v, 0) + 1

            if cats_provided is not None:
                raw = cats_provided[i]
                if isinstance(raw, (np.ndarray, pd.Series)):
                    raw = raw.tolist()
                elif isinstance(raw, torch.Tensor):
                    raw = raw.tolist()
                else:
                    raw = list(raw)
                col_values = raw
            else:
                col_values = list(freq_counts.keys())

            frequent, infrequent = _compute_cats_for_feature(
                col_values, freq_counts, n_samples,
                self.min_frequency, self.max_categories
            )
            self.categories_.append(frequent)
            self.infrequent_categories_.append(infrequent)

        self.fit_status = True
        return self

    def transform(self, data_or_X, **kwargs) -> torch.Tensor:
        if not self.fit_status:
            raise RuntimeError("OrdinalEncoder must be fitted before transform.")

        cols, _ = _labelencoder_to_columns(data_or_X)
        n = len(cols[0]) if cols else 0
        n_features = len(cols)

        # Decide output dtype: we need float when NaN must be stored.
        needs_float = False
        missing_sentinel: float = float("nan")
        if self.encoded_missing_value is not None:
            try:
                import math
                if math.isnan(float(self.encoded_missing_value)):
                    needs_float = True
                missing_sentinel = float(self.encoded_missing_value)
            except (TypeError, ValueError):
                missing_sentinel = float(self.encoded_missing_value)
        else:
            needs_float = True  # NaN is the default for missing

        unknown_sentinel: float = -1.0
        if self.unknown_value is not None:
            try:
                import math
                if math.isnan(float(self.unknown_value)):
                    needs_float = True
                unknown_sentinel = float(self.unknown_value)
            except (TypeError, ValueError):
                unknown_sentinel = float(self.unknown_value)

        out_dtype = (
            torch.float64
            if needs_float and not self.dtype.is_floating_point
            else self.dtype
        )

        result = torch.full(
            (n, n_features), float("nan") if needs_float else 0,
            dtype=out_dtype, device=self.device
        )

        for i, col in enumerate(cols):
            cats = self.categories_[i]
            infreq = self.infrequent_categories_[i]
            cat_to_idx: Dict[Any, int] = {c: j for j, c in enumerate(cats)}
            infreq_code = float(len(cats))
            infreq_set: set = set(infreq) if infreq is not None else set()

            for row_idx, val in enumerate(col):
                if _is_missing(val):
                    result[row_idx, i] = missing_sentinel
                    continue

                if val in cat_to_idx:
                    result[row_idx, i] = float(cat_to_idx[val])
                elif val in infreq_set:
                    result[row_idx, i] = infreq_code
                else:
                    handle = self.handle_unknown
                    if callable(handle) and not isinstance(handle, str):
                        result[row_idx, i] = float(handle(val, i))
                    elif handle == "error":
                        raise ValueError(
                            f"Found unknown category {val!r} in feature {i} "
                            f"during transform."
                        )
                    elif handle == "ignore":
                        result[row_idx, i] = unknown_sentinel
                    elif handle in ("infrequent_if_exist", "warn"):
                        if handle == "warn":
                            warnings.warn(
                                f"Found unknown category {val!r} in feature {i}.",
                                UserWarning,
                                stacklevel=2,
                            )
                        if infreq is not None:
                            result[row_idx, i] = infreq_code
                        else:
                            result[row_idx, i] = unknown_sentinel
                    else:
                        result[row_idx, i] = unknown_sentinel

        # Cast back to requested dtype if safe (no NaN values present)
        if out_dtype != self.dtype:
            has_nan = torch.isnan(result).any()
            if not has_nan or self.dtype.is_floating_point:
                try:
                    result = result.to(dtype=self.dtype)
                except RuntimeError:
                    pass  # keep float64 when cast is unsafe

        return result

    def inverse_transform(self, X) -> List[List[Optional[Any]]]:
        if not self.fit_status:
            raise RuntimeError(
                "OrdinalEncoder must be fitted before inverse_transform."
            )
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X, dtype=torch.float64)
        data = X.to(torch.float64)
        n, n_feats = data.shape
        result: List[List[Optional[Any]]] = [[None] * n_feats for _ in range(n)]

        for i, cats in enumerate(self.categories_):
            infreq = self.infrequent_categories_[i]
            n_freq = len(cats)
            for row_idx in range(n):
                code = data[row_idx, i]
                if torch.isnan(code):
                    result[row_idx][i] = None
                    continue
                code_int = int(round(code.item()))
                if 0 <= code_int < n_freq:
                    result[row_idx][i] = cats[code_int]
                elif code_int == n_freq and infreq is not None:
                    result[row_idx][i] = None  # infrequent bucket
                else:
                    result[row_idx][i] = None  # unknown

        return result

    def fit_transform(self, data_or_X, y=None, **kwargs) -> torch.Tensor:
        return self.fit(data_or_X, y, **kwargs).transform(data_or_X, **kwargs)

    def forward(
            self, X: torch.Tensor, y: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        return self.fit_transform(X, y, **kwargs)


class TargetEncoder(MLModule):
    def __init__(
            self,
            categories: Union[Literal["auto"], list, tuple,
            np.ndarray, pd.Series, pd.DataFrame,
            torch.Tensor] = 'auto',
            target_type: Literal["auto", "continuous", "binary", "multiclass"] = 'auto',
            smooth: Union[float, Literal["auto"]] = 'auto',
            cv: Union[int, str, Callable, MLModule] = 5,
            shuffle: bool = True,
            random_state: Union[int, torch.Generator] = None,
            device: Union[str, torch.device] = "cpu",
            dtype: torch.dtype = torch.long,
            *args,
            **kwargs,
    ):
        super().__init__()
        self.categories = categories
        self.target_type = target_type
        self.smooth = smooth
        self.cv = cv
        self.shuffle = shuffle
        self.random_state = random_state
        self.device = device
        self.dtype = dtype
        self.encodings_: List[torch.Tensor] = []
        self.categories_: List[List[Any]] = []
        self.target_type_: str = ""
        self.target_mean_: float = 0.0
        self.n_features_in_: Optional[int] = None
        self.feature_names_in_: Optional[List[str]] = None
        self.classes_: Optional[List[Any]] = None
        self._class_global_means_: Optional[List[float]] = None

    def _infer_target_type(self, y: torch.Tensor) -> str:
        """Infer whether y represents a continuous, binary, or multiclass target."""
        if y.is_floating_point():
            return "continuous"
        n_unique = len(torch.unique(y))
        return "binary" if n_unique <= 2 else "multiclass"

    def _encode_one_column(
            self,
            col: List[Any],
            y_col: torch.Tensor,
            cats_i: List[Any],
    ) -> torch.Tensor:
        global_mean = y_col.mean()
        global_var = y_col.var() if y_col.numel() > 1 else torch.tensor(
            1e-6, dtype=torch.float32, device=self.device
        )

        # Group row indices by category value
        cat_to_indices: Dict[Any, List[int]] = {c: [] for c in cats_i}
        for row_idx, val in enumerate(col):
            if val in cat_to_indices:
                cat_to_indices[val].append(row_idx)

        enc = torch.zeros(len(cats_i), dtype=torch.float32, device=self.device)

        for k, cat in enumerate(cats_i):
            indices = cat_to_indices.get(cat, [])
            if not indices:
                enc[k] = global_mean
                continue

            n_c = len(indices)
            idx_t = torch.tensor(indices, dtype=torch.long, device=self.device)
            cat_vals = y_col[idx_t]
            mean_c = cat_vals.mean()

            if self.smooth == "auto":
                within_var = cat_vals.var() if n_c > 1 else global_var
                if within_var.item() < 1e-10:
                    lam = torch.tensor(float(n_c), device=self.device)
                else:
                    lam = n_c * global_var / within_var
            else:
                lam = torch.tensor(float(self.smooth), device=self.device)

            enc[k] = (n_c * mean_c + lam * global_mean) / (n_c + lam)

        return enc

    def fit(self, data_or_X, y=None, **kwargs) -> "TargetEncoder":
        if y is None:
            raise ValueError("TargetEncoder requires y for fitting.")

        cols, _ = _labelencoder_to_columns(data_or_X)
        n_features = len(cols)
        n_samples = len(cols[0]) if cols else 0
        self.n_features_in_ = n_features

        if isinstance(data_or_X, pd.DataFrame):
            self.feature_names_in_ = list(data_or_X.columns)
        else:
            self.feature_names_in_ = None

        # Coerce y to float32 tensor
        if isinstance(y, torch.Tensor):
            y_t = y.to(dtype=torch.float32, device=self.device)
        elif isinstance(y, (list, np.ndarray)):
            y_t = torch.tensor(y, dtype=torch.float32, device=self.device)
        else:
            y_t = torch.tensor(list(y), dtype=torch.float32, device=self.device)

        # Determine target type
        self.target_type_ = (
            self._infer_target_type(y_t)
            if self.target_type == "auto"
            else self.target_type
        )

        self.target_mean_ = float(y_t.mean())

        # Build per-class target columns
        if self.target_type_ in ("binary", "multiclass"):
            lb = LabelBinarizer(device=self.device, dtype=torch.long)
            y_list = y_t.long().tolist()
            lb.fit(y_list)
            y_bin = lb.transform(y_list).float()
            self.classes_ = lb.classes_
            if self.target_type_ == "binary":
                y_cols = [y_bin.squeeze(-1)]
            else:
                y_cols = [y_bin[:, j] for j in range(len(self.classes_))]
            self._class_global_means_ = [float(yc.mean()) for yc in y_cols]
        else:
            self.classes_ = None
            y_cols = [y_t]
            self._class_global_means_ = None

        # Determine per-feature categories
        cats_provided = (
            self.categories
            if self.categories != "auto" and not isinstance(self.categories, str)
            else None
        )

        self.categories_ = []
        self.encodings_ = []

        for i, col in enumerate(cols):
            if cats_provided is not None:
                raw = cats_provided[i]
                if isinstance(raw, (np.ndarray, pd.Series)):
                    raw = raw.tolist()
                elif isinstance(raw, torch.Tensor):
                    raw = raw.tolist()
                else:
                    raw = list(raw)
                cats_i = raw
            else:
                all_vals = set(col)
                try:
                    cats_i = sorted(all_vals)
                except TypeError:
                    cats_i = sorted(all_vals, key=str)

            self.categories_.append(cats_i)

            # Encode against each target column
            for y_col in y_cols:
                enc = self._encode_one_column(col, y_col, cats_i)
                self.encodings_.append(enc)

        self.fit_status = True
        return self

    def transform(self, data_or_X, **kwargs) -> torch.Tensor:
        if not self.fit_status:
            raise RuntimeError("TargetEncoder must be fitted before transform.")

        cols, _ = _labelencoder_to_columns(data_or_X)
        n = len(cols[0]) if cols else 0
        n_features = len(cols)

        is_multiclass = self.target_type_ == "multiclass"
        n_classes = len(self.classes_) if is_multiclass and self.classes_ else 1

        out_cols = n_features * n_classes
        result = torch.zeros(n, out_cols, dtype=torch.float32, device=self.device)

        for i, col in enumerate(cols):
            cats_i = self.categories_[i]
            cat_to_idx: Dict[Any, int] = {c: k for k, c in enumerate(cats_i)}

            for j in range(n_classes):
                enc_list_idx = j + i * n_classes
                enc = self.encodings_[enc_list_idx]
                out_col = enc_list_idx if is_multiclass else i

                fallback = (
                    self._class_global_means_[j]
                    if is_multiclass and self._class_global_means_
                    else self.target_mean_
                )
                fallback_t = torch.tensor(fallback, dtype=torch.float32, device=self.device)

                for row_idx, val in enumerate(col):
                    if val in cat_to_idx:
                        result[row_idx, out_col] = enc[cat_to_idx[val]]
                    else:
                        result[row_idx, out_col] = fallback_t

        return result

    def fit_transform(self, data_or_X, y=None, **kwargs) -> torch.Tensor:
        if y is None:
            raise ValueError("TargetEncoder requires y for fit_transform.")

        cols, _ = _labelencoder_to_columns(data_or_X)
        n_features = len(cols)
        n = len(cols[0]) if cols else 0

        # Coerce y
        if isinstance(y, torch.Tensor):
            y_t = y.to(dtype=torch.float32, device=self.device)
        elif isinstance(y, (list, np.ndarray)):
            y_t = torch.tensor(y, dtype=torch.float32, device=self.device)
        else:
            y_t = torch.tensor(list(y), dtype=torch.float32, device=self.device)

        # Number of folds
        n_folds = int(self.cv) if isinstance(self.cv, (int, float)) and self.cv >= 2 else 5
        n_folds = max(2, min(n_folds, n))

        # Build permutation
        if self.shuffle:
            if isinstance(self.random_state, int):
                g = torch.Generator()
                g.manual_seed(self.random_state)
                perm = torch.randperm(n, generator=g)
            elif isinstance(self.random_state, torch.Generator):
                perm = torch.randperm(n, generator=self.random_state)
            else:
                perm = torch.randperm(n)
        else:
            perm = torch.arange(n)

        # First fit on full data to set fit_status, target_type_, and output shape
        self.fit(data_or_X, y, **kwargs)

        is_multiclass = self.target_type_ == "multiclass"
        n_classes = len(self.classes_) if is_multiclass and self.classes_ else 1
        out = torch.zeros(n, n_features * n_classes, dtype=torch.float32, device=self.device)

        fold_size = n // n_folds

        for fold_i in range(n_folds):
            start = fold_i * fold_size
            end = start + fold_size if fold_i < n_folds - 1 else n
            val_idx = perm[start:end]
            train_idx = torch.cat([perm[:start], perm[end:]])

            if train_idx.numel() == 0:
                continue

            # Gather column data for this fold
            train_cols = [[col[int(k)] for k in train_idx] for col in cols]
            val_cols = [[col[int(k)] for k in val_idx] for col in cols]
            y_train = y_t[train_idx]

            if n_features > 1:
                train_data: Any = [
                    tuple(train_cols[c][r] for c in range(n_features))
                    for r in range(len(train_idx))
                ]
                val_data: Any = [
                    tuple(val_cols[c][r] for c in range(n_features))
                    for r in range(len(val_idx))
                ]
            else:
                train_data = train_cols[0]
                val_data = val_cols[0]

            fold_enc = TargetEncoder(
                categories=self.categories,
                target_type=self.target_type,
                smooth=self.smooth,
                shuffle=False,
                device=self.device,
                dtype=self.dtype,
            )
            fold_enc.fit(train_data, y_train)
            fold_out = fold_enc.transform(val_data)
            out[val_idx] = fold_out

        return out

    def forward(
            self, X: torch.Tensor, y: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        if not self.fit_status:
            return self.fit_transform(X, y, **kwargs)
        return self.transform(X, **kwargs)


class KBinsDiscretizer(MLModule):
    # Maps sklearn quantile_method names to torch.quantile interpolation modes
    _QUANTILE_METHOD_MAP: Dict[str, str] = {
        "linear":                     "linear",
        "inverted_cdf":               "lower",
        "averaged_inverted_cdf":      "midpoint",
        "closest_observation":        "nearest",
        "interpolated_inverted_cdf":  "linear",
        "hazen":                      "midpoint",
        "weibull":                    "linear",
        "median_unbiased":            "midpoint",
        "normal_unbiased":            "linear",
    }

    def __init__(self,
                 n_bins: Union[int, list, tuple, torch.Tensor] = 5,
                 encode: Literal["onehot", "onehot-dense", "ordinal"] = 'onehot',
                 strategy: Union[Literal["uniform", "quantile", "kmeans"], Callable, nn.Module] = 'quantile',
                 quantile_method: Union[Literal["inverted_cdf", "averaged_inverted_cdf",
                    "closest_observation", "interpolated_inverted_cdf", "hazen",
                    "weibull", "linear", "median_unbiased", "normal_unbiased"],
                    Callable, nn.Module] = 'linear',
                 subsample: int = 2 * 1e5,
                 random_state: Union[int, torch.Generator] = None,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.long,
                 *args,
                 **kwargs,
                 ):
        super().__init__()
        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy
        self.quantile_method = quantile_method
        self.subsample = subsample
        self.random_state = random_state
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.args = args
        self.kwargs = kwargs
        # Fitted attributes
        self.bin_edges_: List[torch.Tensor] = []
        self.n_bins_: Optional[torch.Tensor] = None
        self.n_features_in_: Optional[int] = None
        self.feature_names_in_: Optional[List[str]] = None
        self.fit_status = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_generator(self) -> Optional[torch.Generator]:
        if self.random_state is None:
            return None
        if isinstance(self.random_state, torch.Generator):
            return self.random_state
        g = torch.Generator(device=self.device)
        g.manual_seed(int(self.random_state))
        return g

    def _resolve_n_bins(self, n_features: int) -> List[int]:
        """Expand n_bins to a per-feature list and validate."""
        nb = self.n_bins
        if isinstance(nb, torch.Tensor):
            nb = nb.flatten().tolist()
        if isinstance(nb, (list, tuple)):
            if len(nb) != n_features:
                raise ValueError(
                    f"n_bins has {len(nb)} elements but X has {n_features} features."
                )
            result = [int(b) for b in nb]
        else:
            result = [int(nb)] * n_features

        for i, b in enumerate(result):
            if b < 2:
                raise ValueError(
                    f"n_bins[{i}] = {b} is too small. n_bins must be >= 2."
                )
        return result

    def _subsample_col(self, col: torch.Tensor, generator: Optional[torch.Generator]) -> torch.Tensor:
        """Randomly subsample a column when n_samples > subsample."""
        if self.subsample is None:
            return col
        max_n = int(self.subsample)
        if col.shape[0] <= max_n:
            return col
        if generator is not None:
            idx = torch.randperm(col.shape[0], device=self.device, generator=generator)[:max_n]
        else:
            idx = torch.randperm(col.shape[0], device=self.device)[:max_n]
        return col[idx]

    # ---- Bin-edge computation strategies ----

    def _edges_uniform(self, col: torch.Tensor, n_bins_i: int) -> torch.Tensor:
        """Uniform bins: equal width from min to max."""
        col_min = col.min().item()
        col_max = col.max().item()
        if col_min == col_max:
            # Degenerate: all values identical → single bin
            return torch.tensor(
                [col_min - 0.5, col_max + 0.5], device=self.device, dtype=torch.float64
            )
        return torch.linspace(col_min, col_max, n_bins_i + 1, device=self.device, dtype=torch.float64)

    def _edges_quantile(self, col: torch.Tensor, n_bins_i: int) -> torch.Tensor:
        """Quantile bins: equal-population width using torch.quantile."""
        col_f = col.to(dtype=torch.float64)
        percentiles = torch.linspace(0.0, 1.0, n_bins_i + 1, device=self.device, dtype=torch.float64)

        # Resolve interpolation mode
        qm = self.quantile_method
        if callable(qm) and not isinstance(qm, str):
            # Custom quantile callable: qm(col, percentiles) -> Tensor
            edges = qm(col_f, percentiles)
            if not isinstance(edges, torch.Tensor):
                edges = torch.tensor(edges, device=self.device, dtype=torch.float64)
            return edges

        interp = self._QUANTILE_METHOD_MAP.get(str(qm).lower(), "linear")
        edges = torch.quantile(col_f, percentiles, interpolation=interp)
        return edges

    def _edges_kmeans(self, col: torch.Tensor, n_bins_i: int, generator: Optional[torch.Generator]) -> torch.Tensor:
        col_f = col.to(dtype=torch.float64)
        n = col_f.shape[0]
        k = n_bins_i

        if n < k:
            # Fall back to uniform if too few samples
            return self._edges_uniform(col_f, k)

        # ---- k-means++ initialisation ----
        if generator is not None:
            first_idx = torch.randint(n, (1,), device=self.device, generator=generator).item()
        else:
            first_idx = torch.randint(n, (1,), device=self.device).item()
        centers = col_f[first_idx].unsqueeze(0)   # (1,)

        for _ in range(k - 1):
            dists = torch.cdist(col_f.unsqueeze(1), centers.unsqueeze(1), p=2).min(dim=1).values
            probs = dists ** 2
            total = probs.sum()
            if total < 1e-12:
                # All equidistant → pick randomly
                if generator is not None:
                    idx = torch.randint(n, (1,), device=self.device, generator=generator).item()
                else:
                    idx = torch.randint(n, (1,), device=self.device).item()
            else:
                probs = probs / total
                if generator is not None:
                    idx = int(torch.multinomial(probs, 1, generator=generator).item())
                else:
                    idx = int(torch.multinomial(probs, 1).item())
            centers = torch.cat([centers, col_f[idx].unsqueeze(0)])

        # ---- Lloyd iterations ----
        for _ in range(300):
            # Assign each point to nearest center  (1-D: just abs diff)
            diffs = torch.abs(col_f.unsqueeze(1) - centers.unsqueeze(0))  # (n, k)
            labels = diffs.argmin(dim=1)                                    # (n,)

            new_centers = torch.zeros(k, device=self.device, dtype=torch.float64)
            changed = False
            for j in range(k):
                mask = labels == j
                if mask.any():
                    nc = col_f[mask].mean()
                else:
                    nc = centers[j]
                if abs((nc - centers[j]).item()) > 1e-12:
                    changed = True
                new_centers[j] = nc
            centers = new_centers
            if not changed:
                break

        centers, _ = torch.sort(centers)

        # Interior edges: midpoints between adjacent sorted centers
        midpoints = 0.5 * (centers[:-1] + centers[1:])           # (k-1,)

        # Extend with data min/max as exterior edges
        col_min = col_f.min()
        col_max = col_f.max()
        edges = torch.cat([col_min.unsqueeze(0), midpoints, col_max.unsqueeze(0)])  # (k+1,)
        return edges

    def _custom_strategy(self, col: torch.Tensor, n_bins_i: int, generator: Optional[torch.Generator]) -> torch.Tensor:
        """Invoke a user-supplied strategy callable or nn.Module."""
        strat = self.strategy
        if callable(strat):
            result = strat(col, n_bins_i)
        else:
            result = strat(col, n_bins_i)
        if not isinstance(result, torch.Tensor):
            result = torch.tensor(result, device=self.device, dtype=torch.float64)
        return result.to(dtype=torch.float64, device=self.device)

    def _remove_narrow_bins(self, edges: torch.Tensor, n_bins_i: int, feature_idx: int) -> torch.Tensor:
        """Drop interior edges that produce bins with width <= 1e-8 (warn)."""
        widths = edges[1:] - edges[:-1]
        mask = widths > 1e-8                                    # (n_bins,)
        n_removed = int((~mask).sum().item())
        if n_removed > 0:
            warnings.warn(
                f"Feature {feature_idx}: {n_removed} bin(s) with width <= 1e-8 "
                f"were removed. Continuing with {int(mask.sum().item())} bin(s).",
                UserWarning,
                stacklevel=4,
            )
            # Keep only edges that bound valid-width bins
            keep_edges = [edges[0]]
            for j, ok in enumerate(mask):
                if ok:
                    keep_edges.append(edges[j + 1])
            if len(keep_edges) < 2:
                keep_edges = [edges[0], edges[-1]]
            edges = torch.stack(keep_edges)
        return edges

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X: Any,
        y: Optional[Any] = None,
        **kwargs,
    ) -> "KBinsDiscretizer":
        # --- Convert input ---
        if not isinstance(X, (torch.Tensor, np.ndarray)) and hasattr(X, "values"):   # pandas DataFrame / Series
            if hasattr(X, "columns"):
                self.feature_names_in_ = list(X.columns)
            X = X.values
        else:
            self.feature_names_in_ = None

        if isinstance(X, np.ndarray):
            X_t = torch.from_numpy(X.astype(np.float64)).to(device=self.device)
        elif isinstance(X, torch.Tensor):
            X_t = X.to(device=self.device, dtype=torch.float64)
        else:
            X_t = torch.tensor(X, dtype=torch.float64, device=self.device)

        if X_t.dim() == 1:
            X_t = X_t.unsqueeze(1)

        n_samples, n_features = X_t.shape
        self.n_features_in_ = n_features

        n_bins_list = self._resolve_n_bins(n_features)
        generator = self._make_generator()

        self.bin_edges_ = []
        actual_n_bins: List[int] = []

        for i in range(n_features):
            col = X_t[:, i]
            col_sub = self._subsample_col(col, generator)
            n_b = n_bins_list[i]

            strat = self.strategy
            if isinstance(strat, str):
                s = strat.lower()
                if s == "uniform":
                    edges = self._edges_uniform(col_sub, n_b)
                elif s == "quantile":
                    edges = self._edges_quantile(col_sub, n_b)
                elif s == "kmeans":
                    edges = self._edges_kmeans(col_sub, n_b, generator)
                else:
                    raise ValueError(
                        f"Unknown strategy '{strat}'. Choose from "
                        f"'uniform', 'quantile', 'kmeans', or pass a callable."
                    )
            else:
                # Custom callable / nn.Module strategy
                edges = self._custom_strategy(col_sub, n_b, generator)

            edges = self._remove_narrow_bins(edges, n_b, i)

            # Extend exterior edges slightly so that the exact min/max are
            # included in the first/last bin during transform.
            edges[0]  = edges[0]  - 1e-8
            edges[-1] = edges[-1] + 1e-8

            self.bin_edges_.append(edges.to(dtype=torch.float64, device=self.device))
            actual_n_bins.append(len(edges) - 1)

        self.n_bins_ = torch.tensor(actual_n_bins, dtype=torch.long, device=self.device)
        self.fit_status = True
        return self

    def transform(
        self,
        X: Any,
        **kwargs,
    ) -> torch.Tensor:
        if not self.fit_status:
            raise RuntimeError("KBinsDiscretizer is not fitted. Call fit() first.")

        if not isinstance(X, (torch.Tensor, np.ndarray)) and hasattr(X, "values"):
            X = X.values
        if isinstance(X, np.ndarray):
            X_t = torch.from_numpy(X.astype(np.float64)).to(device=self.device)
        elif isinstance(X, torch.Tensor):
            X_t = X.to(device=self.device, dtype=torch.float64)
        else:
            X_t = torch.tensor(X, dtype=torch.float64, device=self.device)

        if X_t.dim() == 1:
            X_t = X_t.unsqueeze(1)

        n_samples, n_features = X_t.shape
        if n_features != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {n_features}."
            )

        # Assign each value to its bin index using binary search (bucketize)
        # bin_idx in [0, n_bins_i - 1]
        bin_indices = torch.zeros(n_samples, n_features, dtype=torch.long, device=self.device)
        for i in range(n_features):
            edges = self.bin_edges_[i]           # (n_bins_i + 1,)
            n_b_i = int(self.n_bins_[i].item())
            col = X_t[:, i]
            # torch.bucketize: returns index of the first edge > val
            # So values in [edges[j], edges[j+1]) → bucket j
            idx = torch.bucketize(col, edges, right=False) - 1
            # Clip to valid range [0, n_bins_i - 1]
            idx = idx.clamp(min=0, max=n_b_i - 1)
            bin_indices[:, i] = idx

        # --- Encode ---
        encode = self.encode
        out_dtype = self.dtype

        if encode == "ordinal":
            return bin_indices.to(dtype=out_dtype)

        # onehot or onehot-dense: build one-hot blocks per feature then concat
        n_total_cols = int(self.n_bins_.sum().item())
        result = torch.zeros(n_samples, n_total_cols, device=self.device, dtype=out_dtype)
        col_offset = 0
        for i in range(n_features):
            n_b_i = int(self.n_bins_[i].item())
            rows = torch.arange(int(n_samples), device=self.device, dtype=torch.long)
            cols_idx = (bin_indices[:, i] + col_offset).long().clamp(0, n_total_cols - 1)
            result[rows, cols_idx] = 1
            col_offset += n_b_i

        if encode == "onehot":
            return result.to_sparse()
        return result   # onehot-dense

    def inverse_transform(
        self,
        X: Any,
        **kwargs,
    ) -> torch.Tensor:
        if not self.fit_status:
            raise RuntimeError(
                "KBinsDiscretizer is not fitted. Call fit() first."
            )

        if isinstance(X, torch.Tensor):
            if X.is_sparse:
                X_d = X.to_dense().float()
            else:
                X_d = X.float()
        else:
            X_d = torch.tensor(X, dtype=torch.float32, device=self.device)

        n_samples = X_d.shape[0]
        n_features = self.n_features_in_
        result = torch.zeros(n_samples, n_features, dtype=torch.float64, device=self.device)

        encode = self.encode

        if encode == "ordinal":
            # X_d has shape (n_samples, n_features) with bin indices
            bin_idx_mat = X_d.long()
        else:
            # onehot or onehot-dense: recover bin indices from one-hot blocks
            bin_idx_mat = torch.zeros(n_samples, n_features, dtype=torch.long, device=self.device)
            col_offset = 0
            for i in range(n_features):
                n_b_i = int(self.n_bins_[i].item())
                block = X_d[:, col_offset: col_offset + n_b_i]
                bin_idx_mat[:, i] = block.argmax(dim=1)
                col_offset += n_b_i

        # Map bin index → bin midpoint
        for i in range(n_features):
            edges = self.bin_edges_[i]          # (n_bins_i + 1,) float64
            midpoints = 0.5 * (edges[:-1] + edges[1:])  # (n_bins_i,)
            # Clamp indices to valid range
            idx = bin_idx_mat[:, i].clamp(min=0, max=midpoints.shape[0] - 1)
            result[:, i] = midpoints[idx]

        return result

    def get_feature_names_out(
        self,
        input_features: Optional[List[str]] = None,
    ) -> List[str]:
        if not self.fit_status:
            raise RuntimeError(
                "KBinsDiscretizer is not fitted. Call fit() first."
            )

        if input_features is not None:
            feat_names = list(input_features)
        elif self.feature_names_in_ is not None:
            feat_names = list(self.feature_names_in_)
        else:
            feat_names = [f"x{i}" for i in range(self.n_features_in_)]

        names: List[str] = []
        for i, fname in enumerate(feat_names):
            n_b_i = int(self.n_bins_[i].item())
            if self.encode == "ordinal":
                names.append(f"{fname}_binned")
            else:
                for j in range(n_b_i):
                    names.append(f"{fname}_bin_{j}")
        return names

    def fit_transform(
        self,
        X: Any,
        y: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Fit to X and then discretize X."""
        return self.fit(X, y, **kwargs).transform(X)

    def forward(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if not self.fit_status:
            return self.fit_transform(X, y, **kwargs)
        return self.transform(X, **kwargs)


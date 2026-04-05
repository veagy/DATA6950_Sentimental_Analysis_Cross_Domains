import torch
from typing import Union, List, Callable, Iterable
from ....models.utils import MLModule, MLRegressor
from abc import abstractmethod
import math
import itertools

__all__ = [
    "KFoldCV",
    "GroupKFold",
    "StratifiedKFold",
    "TimeSeriesSplit",
    "LeaveOneOut",
    "LeavePOut",
    "LeaveOneGroupOut",
    "LeavePGroupsOut",
    "PredefinedSplit",
    "ShuffleSplit",
    "GroupShuffleSplit",
    "StratifiedShuffleSplit",
    "RepeatedKFold",
    "RepeatedStratifiedKFold",
    "CVSplitManager"
]


class BaseSplitterCV(MLModule):
    def __init__(self,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.device = device
        self.dtype = dtype

    def split(self, X: torch.Tensor, y: torch.Tensor = None,
              groups: Union[int, torch.Tensor] = None):
        """
        Default fallback executing a standard 5-Fold split if subclass doesn't override.
        """
        n_samples = X.shape[0]
        indices = torch.arange(n_samples, device=self.device)
        fold_sizes = torch.full((5,), n_samples // 5, dtype=torch.long, device=self.device)
        fold_sizes[:n_samples % 5] += 1
        
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            train_indices = torch.cat((indices[:start], indices[stop:]))
            yield train_indices, test_indices
            current = stop

    def get_n_splits(self, X: torch.Tensor, y: torch.Tensor = None,
                     groups: Union[int, torch.Tensor] = None):
        return 5


class KFoldCV(BaseSplitterCV):
    def __init__(self,
                 n_splits: int = 5,
                 shuffle: bool = False,
                 random_state: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(device=device,
                         dtype=dtype,
                         *args, **kwargs)
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: torch.Tensor, y: torch.Tensor = None,
              groups: Union[int, torch.Tensor] = None):
        n_samples = X.shape[0]
        indices = torch.arange(n_samples, device=self.device)

        if self.shuffle:
            if self.random_state is not None:
                g = torch.Generator(device=self.device)
                g.manual_seed(self.random_state)
                indices = indices[torch.randperm(n_samples, generator=g, device=self.device)]
            else:
                indices = indices[torch.randperm(n_samples, device=self.device)]

        fold_sizes = torch.full((self.n_splits,), n_samples // self.n_splits, dtype=torch.long, device=self.device)
        fold_sizes[:n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            train_indices = torch.cat((indices[:start], indices[stop:]))
            yield train_indices, test_indices
            current = stop

    def get_n_splits(self, X: torch.Tensor, y: torch.Tensor = None,
                     groups: Union[int, torch.Tensor] = None):
        return self.n_splits


class GroupKFold(BaseSplitterCV):
    def __init__(self, n_splits: int = 5,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(device=device, dtype=dtype, *args, **kwargs)
        self.n_splits = n_splits

    def split(self, X: torch.Tensor, y: torch.Tensor = None,
              groups: torch.Tensor = None):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")

        unique_groups, groups_indices = torch.unique(groups, return_inverse=True)
        n_groups = len(unique_groups)

        if self.n_splits > n_groups:
            raise ValueError(
                f"Cannot have number of splits n_splits={self.n_splits} greater than the number of groups: {n_groups}.")

        indices = torch.arange(n_groups, device=self.device)
        fold_sizes = torch.full((self.n_splits,), n_groups // self.n_splits, dtype=torch.long, device=self.device)
        fold_sizes[:n_groups % self.n_splits] += 1

        current = 0
        group_test_indices_list = []
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            group_test_indices = unique_groups[start:stop]
            group_test_indices_list.append(group_test_indices)
            current = stop

        for group_test_idxs in group_test_indices_list:
            mask = torch.isin(groups, group_test_idxs)
            test_indices = torch.nonzero(mask, as_tuple=True)[0]
            train_indices = torch.nonzero(~mask, as_tuple=True)[0]
            yield train_indices, test_indices

    def get_n_splits(self, X: torch.Tensor = None, y: torch.Tensor = None,
                     groups: torch.Tensor = None):
        return self.n_splits


class StratifiedKFold(BaseSplitterCV):
    def __init__(self, n_splits: int = 5,
                 shuffle: bool = False,
                 random_state: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(device=device, dtype=dtype, *args, **kwargs)
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: torch.Tensor, y: torch.Tensor = None,
              groups: Union[int, torch.Tensor] = None):
        if y is None:
            raise ValueError("The 'y' parameter should not be None for StratifiedKFold.")

        n_samples = X.shape[0]
        y_flat = torch.as_tensor(y, device=self.device).flatten()
        unique_classes, y_inverse = torch.unique(y_flat, return_inverse=True)
        y_inverse = y_inverse.to(torch.long)
        class_counts = torch.bincount(y_inverse)

        fold_assignment = torch.full((n_samples,), -1, dtype=torch.long, device=self.device)

        for cls_idx, count in enumerate(class_counts):
            # indices of this class
            cls_mask = (y_inverse == cls_idx).bool()
            cls_indices = torch.nonzero(cls_mask, as_tuple=True)[0]

            if self.shuffle:
                if self.random_state is not None:
                    g = torch.Generator(device=self.device)
                    g.manual_seed(self.random_state)
                    perm = torch.randperm(count, generator=g, device=self.device)
                else:
                    perm = torch.randperm(count, device=self.device)
                cls_indices = cls_indices[perm]

            # Distribute into folds
            fold_sizes = torch.full((self.n_splits,), count // self.n_splits, dtype=torch.long, device=self.device)
            fold_sizes[:count % self.n_splits] += 1

            current = 0
            for fold_id, fold_size in enumerate(fold_sizes):
                start, stop = current, current + fold_size
                fold_assignment[cls_indices[start:stop]] = fold_id
                current = stop

        # Now generate the splits
        for i in range(self.n_splits):
            test_indices = torch.nonzero(fold_assignment == i, as_tuple=True)[0]
            train_indices = torch.nonzero(fold_assignment != i, as_tuple=True)[0]
            yield train_indices, test_indices

    def get_n_splits(self, X: torch.Tensor = None, y: torch.Tensor = None,
                     groups: torch.Tensor = None):
        return self.n_splits


class TimeSeriesSplit(BaseSplitterCV):
    def __init__(self, n_splits: int = 5,
                 max_train_size: int = None,
                 test_size: int = None,
                 gap: int = 0,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(device=device, dtype=dtype, *args, **kwargs)
        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap

    def split(self, X: torch.Tensor, y: torch.Tensor = None,
              groups: torch.Tensor = None):
        n_samples = X.shape[0]
        n_splits = self.n_splits
        n_folds = n_splits + 1

        if n_samples < n_folds:
            raise ValueError(
                f"Cannot have number of folds plus 1 greater than the number of samples: {n_samples} < {n_folds}.")

        if self.test_size is not None:
            test_size = self.test_size
        else:
            test_size = n_samples // n_folds

        # Verify split possible
        test_start = n_samples - n_splits * test_size
        if test_start - self.gap < 0:
            # Just raise error or adjust? sklearn raises
            raise ValueError("Too many splits for number of samples and test_size.")

        indices = torch.arange(n_samples, device=self.device)

        for i in range(n_splits):
            test_start_idx = n_samples - (n_splits - i) * test_size
            test_end_idx = test_start_idx + test_size

            test_indices = indices[test_start_idx:test_end_idx]

            train_end_idx = test_start_idx - self.gap
            if self.max_train_size is not None:
                train_start_idx = max(0, train_end_idx - self.max_train_size)
            else:
                train_start_idx = 0

            train_indices = indices[train_start_idx:train_end_idx]

            yield train_indices, test_indices

    def get_n_splits(self, X: torch.Tensor = None, y: torch.Tensor = None,
                     groups: torch.Tensor = None):
        return self.n_splits


class LeaveOneOut(BaseSplitterCV):
    def __init__(self, gcv_mode: str = 'auto', device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        """
        Leave-One-Out cross-validator.

        Parameters
        ----------
        gcv_mode : {'auto', 'svd', 'eigen'}, default='auto'
            Flag indicating which strategy to use when performing Leave-One-Out Cross-Validation.
            Options are:
            'auto' : use 'svd' if n_samples > n_features, otherwise use 'eigen'
            'svd' : force use of singular value decomposition of X when X is dense, eigenvalue decomposition of X^T.X when X is sparse.
            'eigen' : force computation via eigendecomposition of X.X^T
        """
        super().__init__(device=device, dtype=dtype, *args, **kwargs)
        self.gcv_mode = gcv_mode

    def split(self, X: torch.Tensor, y: torch.Tensor = None,
              groups: torch.Tensor = None):
        n_samples = X.shape[0]
        indices = torch.arange(n_samples, device=self.device)

        for i in range(n_samples):
            test_indices = indices[i:i + 1]
            train_indices = torch.cat((indices[:i], indices[i + 1:]))
            yield train_indices, test_indices

    def get_n_splits(self, X: torch.Tensor, y: torch.Tensor = None,
                     groups: torch.Tensor = None):
        return X.shape[0]


class LeavePOut(BaseSplitterCV):
    def __init__(self, p: int,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(device=device, dtype=dtype, *args, **kwargs)
        self.p = p

    def split(self, X: torch.Tensor, y: torch.Tensor = None,
              groups: torch.Tensor = None):
        n_samples = X.shape[0]
        indices = torch.arange(n_samples, device=self.device)

        for test_idx_tuple in itertools.combinations(range(n_samples), self.p):
            test_indices = torch.tensor(test_idx_tuple, dtype=torch.long, device=self.device)
            mask = torch.ones(n_samples, dtype=torch.bool, device=self.device)
            mask[test_indices] = False
            train_indices = indices[mask]
            yield train_indices, test_indices

    def get_n_splits(self, X: torch.Tensor, y: torch.Tensor = None,
                     groups: torch.Tensor = None):
        n_samples = X.shape[0]
        return math.comb(n_samples, self.p)


class LeaveOneGroupOut(BaseSplitterCV):
    def __init__(self, device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(device=device, dtype=dtype, *args, **kwargs)

    def split(self, X: torch.Tensor, y: torch.Tensor = None,
              groups: torch.Tensor = None):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")

        unique_groups = torch.unique(groups)

        for group in unique_groups:
            # Test where groups == group
            test_mask = (groups == group)
            test_indices = torch.nonzero(test_mask, as_tuple=True)[0]
            train_indices = torch.nonzero(~test_mask, as_tuple=True)[0]
            yield train_indices, test_indices

    def get_n_splits(self, X: torch.Tensor = None, y: torch.Tensor = None,
                     groups: torch.Tensor = None):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        return len(torch.unique(groups))


class LeavePGroupsOut(BaseSplitterCV):
    def __init__(self, n_groups: int,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(device=device, dtype=dtype, *args, **kwargs)
        self.n_groups = n_groups

    def split(self, X: torch.Tensor, y: torch.Tensor = None,
              groups: torch.Tensor = None):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")

        unique_groups = torch.unique(groups)
        n_unique_groups = len(unique_groups)

        if self.n_groups > n_unique_groups:
            raise ValueError(f"Cannot leave {self.n_groups} groups out from {n_unique_groups} groups.")

        for test_groups_tuple in itertools.combinations(unique_groups.tolist(), self.n_groups):
            test_groups_tensor = torch.tensor(test_groups_tuple, device=self.device, dtype=groups.dtype)

            test_mask = torch.isin(groups, test_groups_tensor)
            test_indices = torch.nonzero(test_mask, as_tuple=True)[0]
            train_indices = torch.nonzero(~test_mask, as_tuple=True)[0]
            yield train_indices, test_indices

    def get_n_splits(self, X: torch.Tensor = None, y: torch.Tensor = None,
                     groups: torch.Tensor = None):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        return math.comb(len(torch.unique(groups)), self.n_groups)


class PredefinedSplit(BaseSplitterCV):
    def __init__(self, test_fold: Union[List, torch.Tensor],
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(device=device, dtype=dtype, *args, **kwargs)
        if not isinstance(test_fold, torch.Tensor):
            self.test_fold = torch.tensor(test_fold, device=device)
        else:
            self.test_fold = test_fold.to(device)

    def split(self, X: torch.Tensor = None, y: torch.Tensor = None,
              groups: torch.Tensor = None):
        unique_folds = torch.unique(self.test_fold)
        unique_folds = unique_folds[unique_folds != -1]
        unique_folds, _ = torch.sort(unique_folds)

        for fold_idx in unique_folds:
            test_mask = (self.test_fold == fold_idx)
            test_indices = torch.nonzero(test_mask, as_tuple=True)[0]
            train_mask = (self.test_fold != fold_idx)
            train_indices = torch.nonzero(train_mask, as_tuple=True)[0]

            yield train_indices, test_indices

    def get_n_splits(self, X: torch.Tensor = None, y: torch.Tensor = None,
                     groups: torch.Tensor = None):
        unique_folds = torch.unique(self.test_fold)
        return len(unique_folds[unique_folds != -1])


class ShuffleSplit(BaseSplitterCV):
    def __init__(self, n_splits: int = 10,
                 test_size: Union[float, int] = None,
                 train_size: Union[float, int] = None,
                 random_state: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(device=device, dtype=dtype, *args, **kwargs)
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state

    def split(self, X: torch.Tensor, y: torch.Tensor = None,
              groups: torch.Tensor = None):
        n_samples = X.shape[0]
        n_train, n_test = self._validate_shuffle_split(n_samples, self.test_size, self.train_size)

        for i in range(self.n_splits):
            if self.random_state is not None:
                g = torch.Generator(device=self.device)
                # Vary seed per split to get different splits
                g.manual_seed(self.random_state + i)
            else:
                g = None

            # randperm is better for sampling without replacement
            indices = torch.randperm(n_samples, generator=g, device=self.device) if g else torch.randperm(n_samples,
                                                                                                          device=self.device)

            train_indices = indices[:n_train]
            test_indices = indices[n_train:n_train + n_test]

            yield train_indices, test_indices

    def get_n_splits(self, X: torch.Tensor = None, y: torch.Tensor = None,
                     groups: torch.Tensor = None):
        return self.n_splits

    def _validate_shuffle_split(self, n_samples, test_size, train_size):
        if test_size is None and train_size is None:
            # Default to 0.1 test
            n_test = int(0.1 * n_samples)
            n_train = n_samples - n_test
        elif test_size is not None and train_size is None:
            if isinstance(test_size, float):
                n_test = int(test_size * n_samples)
            else:
                n_test = test_size
            n_train = n_samples - n_test
        elif test_size is None and train_size is not None:
            if isinstance(train_size, float):
                n_train = int(train_size * n_samples)
            else:
                n_train = train_size
            n_test = n_samples - n_train
        else:
            if isinstance(test_size, float):
                n_test = int(test_size * n_samples)
            else:
                n_test = test_size
            if isinstance(train_size, float):
                n_train = int(train_size * n_samples)
            else:
                n_train = train_size

        if n_train + n_test > n_samples:
            raise ValueError("The sum of train_size and test_size cannot be greater than n_samples.")

        # Sklearn allows n_train + n_test < n_samples (rest are unused)
        return n_train, n_test


class GroupShuffleSplit(BaseSplitterCV):
    def __init__(self, n_splits: int = 5,
                 test_size: Union[float, int] = None,
                 train_size: Union[float, int] = None,
                 random_state: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(device=device, dtype=dtype, *args, **kwargs)
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state

    def split(self, X: torch.Tensor, y: torch.Tensor = None,
              groups: torch.Tensor = None):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")

        unique_groups = torch.unique(groups)
        n_groups = len(unique_groups)

        if self.test_size is None and self.train_size is None:
            n_test = max(1, int(0.2 * n_groups))  # Sklearn default is 0.2? Doc says default=None.
            n_train = n_groups - n_test
        else:
            if self.test_size is not None:
                if isinstance(self.test_size, float):
                    n_test = int(self.test_size * n_groups)
                else:
                    n_test = self.test_size

            if self.train_size is not None:
                if isinstance(self.train_size, float):
                    n_train = int(self.train_size * n_groups)
                else:
                    n_train = self.train_size

            if self.test_size is None:
                n_test = n_groups - n_train
            if self.train_size is None:
                n_train = n_groups - n_test

        if n_train + n_test > n_groups:
            raise ValueError("The sum of train_size and test_size (groups) cannot be greater than n_groups.")

        for i in range(self.n_splits):
            if self.random_state is not None:
                g = torch.Generator(device=self.device)
                g.manual_seed(self.random_state + i)
            else:
                g = None

            shuffled_indices = torch.randperm(n_groups, generator=g, device=self.device) if g else torch.randperm(
                n_groups, device=self.device)

            train_group_indices = shuffled_indices[:n_train]
            test_group_indices = shuffled_indices[n_train:n_train + n_test]

            train_groups = unique_groups[train_group_indices]
            test_groups = unique_groups[test_group_indices]

            test_mask = torch.isin(groups, test_groups)
            train_mask = torch.isin(groups, train_groups)

            test_indices = torch.nonzero(test_mask, as_tuple=True)[0]
            train_indices = torch.nonzero(train_mask, as_tuple=True)[0]

            yield train_indices, test_indices

    def get_n_splits(self, X: torch.Tensor = None, y: torch.Tensor = None,
                     groups: torch.Tensor = None):
        return self.n_splits


class StratifiedShuffleSplit(BaseSplitterCV):
    def __init__(self, n_splits: int = 10,
                 test_size: Union[float, int] = None,
                 train_size: Union[float, int] = None,
                 random_state: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(device=device, dtype=dtype, *args, **kwargs)
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state

    def split(self, X: torch.Tensor, y: torch.Tensor = None,
              groups: Union[int, torch.Tensor] = None):
        if y is None:
            raise ValueError("The 'y' parameter should not be None.")

        n_samples = X.shape[0]
        y_flat = torch.as_tensor(y, device=self.device).flatten()
        unique_classes, y_inverse = torch.unique(y_flat, return_inverse=True)
        y_inverse = y_inverse.to(torch.long)
        class_counts = torch.bincount(y_inverse)

        # Validate and calculate global train/test sizes
        # Logic similar to ShuffleSplit to get total n_test
        if self.test_size is None and self.train_size is None:
            n_test = int(0.1 * n_samples)
        elif self.test_size is not None:
            if isinstance(self.test_size, float):
                n_test = int(self.test_size * n_samples)
            else:
                n_test = self.test_size
        else:
            if isinstance(self.train_size, float):
                n_train = int(self.train_size * n_samples)
            else:
                n_train = self.train_size
            n_test = n_samples - n_train

        for i in range(self.n_splits):
            train_indices_list = []
            test_indices_list = []

            if self.random_state is not None:
                seed = self.random_state + i
            else:
                seed = None

            for cls_idx, count in enumerate(class_counts):
                cls_mask = (y_inverse == cls_idx)
                cls_indices = torch.nonzero(cls_mask, as_tuple=True)[0]

                # Proportional distribution
                # n_test_cls = total_test_samples * (class_count / total_samples)
                n_test_cls = int(n_test * (count / n_samples))
                n_train_cls = count - n_test_cls

                # Shuffle class indices
                if seed is not None:
                    g = torch.Generator(device=self.device)
                    g.manual_seed(seed)
                    perm = torch.randperm(count, generator=g, device=self.device)
                else:
                    perm = torch.randperm(count, device=self.device)

                cls_indices = cls_indices[perm]

                test_indices_list.append(cls_indices[:n_test_cls])
                train_indices_list.append(cls_indices[n_test_cls:])

            train_indices = torch.cat(train_indices_list)
            test_indices = torch.cat(test_indices_list)

            yield train_indices, test_indices

    def get_n_splits(self, X: torch.Tensor = None, y: torch.Tensor = None,
                     groups: torch.Tensor = None):
        return self.n_splits


class RepeatedKFold(BaseSplitterCV):
    def __init__(self, n_splits: int = 5,
                 n_repeats: int = 10,
                 random_state: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(device=device, dtype=dtype, *args, **kwargs)
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(self, X: torch.Tensor, y: torch.Tensor = None,
              groups: torch.Tensor = None):
        for i in range(self.n_repeats):
            # Seed changes per repeat
            seed = self.random_state + i if self.random_state is not None else None

            kf = KFoldCV(n_splits=self.n_splits, shuffle=True, random_state=seed,
                         device=self.device, dtype=self.dtype)

            yield from kf.split(X, y, groups)

    def get_n_splits(self, X: torch.Tensor = None, y: torch.Tensor = None,
                     groups: torch.Tensor = None):
        return self.n_splits * self.n_repeats


class RepeatedStratifiedKFold(BaseSplitterCV):
    def __init__(self, n_splits: int = 5,
                 n_repeats: int = 10,
                 random_state: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(device=device, dtype=dtype, *args, **kwargs)
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(self, X: torch.Tensor, y: torch.Tensor = None,
              groups: Union[int, torch.Tensor] = None):
        for i in range(self.n_repeats):
            # Seed changes per repeat
            seed = self.random_state + i if self.random_state is not None else None

            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=seed,
                                  device=self.device, dtype=self.dtype)

            yield from skf.split(X, y, groups)

    def get_n_splits(self, X: torch.Tensor = None, y: torch.Tensor = None,
                     groups: torch.Tensor = None):
        return self.n_splits * self.n_repeats


class CVSplitManager(BaseSplitterCV):
    def __init__(self,
                 splitter: Union[str, Callable, Iterable, MLModule, int],
                 cv_config: dict = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(
            device=device,
            dtype=dtype,
            *args, **kwargs
        )
        _cv_config = cv_config if cv_config is not None else {}

        if isinstance(splitter, str):
            mapping_dict = {
                "k_fold": KFoldCV,
                "group_k_fold": GroupKFold,
                "stratified_k_fold": StratifiedKFold,
                "time_series_split": TimeSeriesSplit,
                "leave_one_out": LeaveOneOut,
                "leave_p_out": LeavePOut,
                "leave_one_groups_out": LeaveOneGroupOut,
                "leave_p_groups_out": LeavePGroupsOut,
                "predefined_split": PredefinedSplit,
                "shuffle_split": ShuffleSplit,
                "group_shuffle_split": GroupShuffleSplit,
                "stratified_shuffle_split": StratifiedShuffleSplit,
                "repeated_k_fold": RepeatedKFold,
                "repeated_stratified_k_fold": RepeatedStratifiedKFold
            }
            vals_list = mapping_dict.values()
            vals_list = [cls.__name__ for cls in vals_list]
            self.splitter_flag = None
            if splitter.lower() in mapping_dict.keys():
                self.splitter_cls = mapping_dict[splitter.lower()]
            elif splitter in vals_list:
                for cls in mapping_dict.values():
                    if cls.__name__ == splitter:
                        self.splitter_cls = cls
                        break
            else:
                self.splitter_cls = KFoldCV
            self.splitter = self.splitter_cls(**_cv_config)
            self.splitter_flag = 0
        elif isinstance(splitter, int):
            _cv_config.setdefault('n_splits', splitter)
            self.splitter = KFoldCV(**_cv_config)
            self.splitter_flag = 0
        elif isinstance(splitter, (Callable, Iterable)):
            self.splitter = lambda X, y, groups: splitter(X, y, groups, **_cv_config)
            self.n_splits = lambda X, y, groups: len(self.splitter(X, y, groups))
            self.splitter_flag = 1
        elif isinstance(splitter, MLRegressor):
            self.splitter = splitter(**_cv_config)
            self.splitter_flag = 2
        else:
            self.splitter = KFoldCV(**_cv_config)
            self.splitter_flag = 0

    def split(self, X: torch.Tensor, y: torch.Tensor = None,
              groups: Union[int, torch.Tensor] = None):
        match self.splitter_flag:
            case 0:
                return self.splitter.split(X, y, groups)
            case 1:
                return self.splitter(X, y, groups)
            case 2:
                return self.splitter.split(X, y, groups)

    def get_n_splits(self, X: torch.Tensor, y: torch.Tensor = None,
                     groups: Union[int, torch.Tensor] = None):
        match self.splitter_flag:
            case 0:
                return self.splitter.get_n_splits(X, y, groups)
            case 1:
                return self.n_splits(X, y, groups)
            case 2:
                return self.splitter.get_n_splits(X, y, groups)

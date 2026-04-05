"""Config templates for splitters."""
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from ....config import ConfigTemplate

try:
    import torch
except ImportError:
    torch = None

"""Generated config for BaseSplitterCV."""
class BaseSplitterCVConfig(ConfigTemplate):
    model_name = "BaseSplitterCV"
    model_path = "Code.models.machine_learning.cross_validation.splitters"

    def __init__(self,
        immutable: bool = True,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.device = device
        self.dtype = dtype


"""Generated config for CVSplitManager."""
class CVSplitManagerConfig(ConfigTemplate):
    model_name = "CVSplitManager"
    model_path = "Code.models.machine_learning.cross_validation.splitters"

    def __init__(self,
        immutable: bool = True,
        splitter: Union[str, Callable, Iterable, MLModule, int] = None,
        cv_config: dict = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.splitter = splitter
        self.cv_config = cv_config
        self.device = device
        self.dtype = dtype


"""Generated config for GroupKFold."""
class GroupKFoldConfig(ConfigTemplate):
    model_name = "GroupKFold"
    model_path = "Code.models.machine_learning.cross_validation.splitters"

    def __init__(self,
        immutable: bool = True,
        n_splits: int = 5,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_splits = n_splits
        self.device = device
        self.dtype = dtype


"""Generated config for GroupShuffleSplit."""
class GroupShuffleSplitConfig(ConfigTemplate):
    model_name = "GroupShuffleSplit"
    model_path = "Code.models.machine_learning.cross_validation.splitters"

    def __init__(self,
        immutable: bool = True,
        n_splits: int = 5,
        test_size: Union[float, int] = None,
        train_size: Union[float, int] = None,
        random_state: int = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self.device = device
        self.dtype = dtype


"""Generated config for KFoldCV."""
class KFoldCVConfig(ConfigTemplate):
    model_name = "KFoldCV"
    model_path = "Code.models.machine_learning.cross_validation.splitters"

    def __init__(self,
        immutable: bool = True,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: int = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.device = device
        self.dtype = dtype


"""Generated config for LeaveOneGroupOut."""
class LeaveOneGroupOutConfig(ConfigTemplate):
    model_name = "LeaveOneGroupOut"
    model_path = "Code.models.machine_learning.cross_validation.splitters"

    def __init__(self,
        immutable: bool = True,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.device = device
        self.dtype = dtype


"""Generated config for LeaveOneOut."""
class LeaveOneOutConfig(ConfigTemplate):
    model_name = "LeaveOneOut"
    model_path = "Code.models.machine_learning.cross_validation.splitters"

    def __init__(self,
        immutable: bool = True,
        gcv_mode: str = 'auto',
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.gcv_mode = gcv_mode
        self.device = device
        self.dtype = dtype


"""Generated config for LeavePGroupsOut."""
class LeavePGroupsOutConfig(ConfigTemplate):
    model_name = "LeavePGroupsOut"
    model_path = "Code.models.machine_learning.cross_validation.splitters"

    def __init__(self,
        immutable: bool = True,
        n_groups: int = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_groups = n_groups
        self.device = device
        self.dtype = dtype


"""Generated config for LeavePOut."""
class LeavePOutConfig(ConfigTemplate):
    model_name = "LeavePOut"
    model_path = "Code.models.machine_learning.cross_validation.splitters"

    def __init__(self,
        immutable: bool = True,
        p: int = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.p = p
        self.device = device
        self.dtype = dtype


"""Generated config for PredefinedSplit."""
class PredefinedSplitConfig(ConfigTemplate):
    model_name = "PredefinedSplit"
    model_path = "Code.models.machine_learning.cross_validation.splitters"

    def __init__(self,
        immutable: bool = True,
        test_fold: Union[List, torch.Tensor] = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.test_fold = test_fold
        self.device = device
        self.dtype = dtype


"""Generated config for RepeatedKFold."""
class RepeatedKFoldConfig(ConfigTemplate):
    model_name = "RepeatedKFold"
    model_path = "Code.models.machine_learning.cross_validation.splitters"

    def __init__(self,
        immutable: bool = True,
        n_splits: int = 5,
        n_repeats: int = 10,
        random_state: int = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.device = device
        self.dtype = dtype


"""Generated config for RepeatedStratifiedKFold."""
class RepeatedStratifiedKFoldConfig(ConfigTemplate):
    model_name = "RepeatedStratifiedKFold"
    model_path = "Code.models.machine_learning.cross_validation.splitters"

    def __init__(self,
        immutable: bool = True,
        n_splits: int = 5,
        n_repeats: int = 10,
        random_state: int = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.device = device
        self.dtype = dtype


"""Generated config for ShuffleSplit."""
class ShuffleSplitConfig(ConfigTemplate):
    model_name = "ShuffleSplit"
    model_path = "Code.models.machine_learning.cross_validation.splitters"

    def __init__(self,
        immutable: bool = True,
        n_splits: int = 10,
        test_size: Union[float, int] = None,
        train_size: Union[float, int] = None,
        random_state: int = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self.device = device
        self.dtype = dtype


"""Generated config for StratifiedKFold."""
class StratifiedKFoldConfig(ConfigTemplate):
    model_name = "StratifiedKFold"
    model_path = "Code.models.machine_learning.cross_validation.splitters"

    def __init__(self,
        immutable: bool = True,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: int = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.device = device
        self.dtype = dtype


"""Generated config for StratifiedShuffleSplit."""
class StratifiedShuffleSplitConfig(ConfigTemplate):
    model_name = "StratifiedShuffleSplit"
    model_path = "Code.models.machine_learning.cross_validation.splitters"

    def __init__(self,
        immutable: bool = True,
        n_splits: int = 10,
        test_size: Union[float, int] = None,
        train_size: Union[float, int] = None,
        random_state: int = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self.device = device
        self.dtype = dtype


"""Generated config for TimeSeriesSplit."""
class TimeSeriesSplitConfig(ConfigTemplate):
    model_name = "TimeSeriesSplit"
    model_path = "Code.models.machine_learning.cross_validation.splitters"

    def __init__(self,
        immutable: bool = True,
        n_splits: int = 5,
        max_train_size: int = None,
        test_size: int = None,
        gap: int = 0,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap
        self.device = device
        self.dtype = dtype

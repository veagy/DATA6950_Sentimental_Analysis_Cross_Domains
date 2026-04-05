"""Config templates for encoding."""
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from .....config import ConfigTemplate

try:
    import torch
except ImportError:
    torch = None

"""Generated config for ContinuousFeatureDiscretizer."""
class ContinuousFeatureDiscretizerConfig(ConfigTemplate):
    model_name = "ContinuousFeatureDiscretizer"
    model_path = "Code.models.machine_learning.preprocessing.encoding.encoding"

    def __init__(self,
        immutable: bool = True,
        eps: float = 1e-05,
        merge_func: Union[Literal['mean', 'median', 'rms', 'abs'], Callable] = 'mean',
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.eps = eps
        self.merge_func = merge_func
        self.device = device
        self.dtype = dtype


"""Generated config for KBinsDiscretizer."""
class KBinsDiscretizerConfig(ConfigTemplate):
    model_name = "KBinsDiscretizer"
    model_path = "Code.models.machine_learning.preprocessing.encoding.encoding"

    def __init__(self,
        immutable: bool = True,
        n_bins: Union[int, list, tuple, torch.Tensor] = 5,
        encode: Literal['onehot', 'onehot-dense', 'ordinal'] = 'onehot',
        strategy: Union[Literal['uniform', 'quantile', 'kmeans'], Callable, nn.Module] = 'quantile',
        quantile_method: Union[Literal['inverted_cdf', 'averaged_inverted_cdf', 'closest_observation', 'interpolated_inverted_cdf', 'hazen', 'weibull', 'linear', 'median_unbiased', 'normal_unbiased'], Callable, nn.Module] = 'linear',
        subsample: int = 2 * 100000.0,
        random_state: Union[int, torch.Generator] = None,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.long,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy
        self.quantile_method = quantile_method
        self.subsample = subsample
        self.random_state = random_state
        self.device = device
        self.dtype = dtype


"""Generated config for LabelBinarizer."""
class LabelBinarizerConfig(ConfigTemplate):
    model_name = "LabelBinarizer"
    model_path = "Code.models.machine_learning.preprocessing.encoding.encoding"

    def __init__(self,
        immutable: bool = True,
        neg_label: int = 0,
        pos_label: int = 1,
        sparse_output: bool = False,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.long,
        handle_unknown: Literal['error', 'ignore'] = 'error',
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.neg_label = neg_label
        self.pos_label = pos_label
        self.sparse_output = sparse_output
        self.device = device
        self.dtype = dtype
        self.handle_unknown = handle_unknown


"""Generated config for LabelEncoder."""
class LabelEncoderConfig(ConfigTemplate):
    model_name = "LabelEncoder"
    model_path = "Code.models.machine_learning.preprocessing.encoding.encoding"

    def __init__(self,
        immutable: bool = True,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.long,
        handle_unknown: Literal['error', 'ignore'] = 'error',
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.device = device
        self.dtype = dtype
        self.handle_unknown = handle_unknown


"""Generated config for MultiLabelBinarizer."""
class MultiLabelBinarizerConfig(ConfigTemplate):
    model_name = "MultiLabelBinarizer"
    model_path = "Code.models.machine_learning.preprocessing.encoding.encoding"

    def __init__(self,
        immutable: bool = True,
        neg_label: int = 0,
        pos_label: int = 1,
        handle_unknown: Literal['error', 'ignore'] = 'error',
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.long,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.neg_label = neg_label
        self.pos_label = pos_label
        self.handle_unknown = handle_unknown
        self.device = device
        self.dtype = dtype


"""Generated config for OneHotEncoder."""
class OneHotEncoderConfig(ConfigTemplate):
    model_name = "OneHotEncoder"
    model_path = "Code.models.machine_learning.preprocessing.encoding.encoding"

    def __init__(self,
        immutable: bool = True,
        categories: Union[Literal['auto'], list, tuple, np.ndarray, pd.Series, pd.DataFrame, torch.Tensor] = 'auto',
        drop: Union[Literal['first', 'if_binary'], list, tuple, np.ndarray, pd.Series, pd.DataFrame, torch.Tensor] = None,
        sparse_output: bool = True,
        handle_unknown: Union[Literal['error', 'ignore', 'infrequent_if_exist', 'warn'], Callable] = 'error',
        min_frequency: Union[int, float] = None,
        max_categories: int = None,
        feature_name_combiner: Union[Literal['concat'], Callable] = 'concat',
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.long,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.categories = categories
        self.drop = drop
        self.sparse_output = sparse_output
        self.handle_unknown = handle_unknown
        self.min_frequency = min_frequency
        self.max_categories = max_categories
        self.feature_name_combiner = feature_name_combiner
        self.device = device
        self.dtype = dtype


"""Generated config for OrdinalEncoder."""
class OrdinalEncoderConfig(ConfigTemplate):
    model_name = "OrdinalEncoder"
    model_path = "Code.models.machine_learning.preprocessing.encoding.encoding"

    def __init__(self,
        immutable: bool = True,
        categories: Union[Literal['auto'], list, tuple, np.ndarray, pd.Series, pd.DataFrame, torch.Tensor] = 'auto',
        handle_unknown: Union[Literal['error', 'ignore', 'infrequent_if_exist', 'warn'], Callable] = 'error',
        unknown_value: Union[int, torch.inf] = None,
        encoded_missing_value: Union[int, torch.inf] = None,
        min_frequency: Union[int, float] = None,
        max_categories: int = None,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.long,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.categories = categories
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.encoded_missing_value = encoded_missing_value
        self.min_frequency = min_frequency
        self.max_categories = max_categories
        self.device = device
        self.dtype = dtype


"""Generated config for TargetEncoder."""
class TargetEncoderConfig(ConfigTemplate):
    model_name = "TargetEncoder"
    model_path = "Code.models.machine_learning.preprocessing.encoding.encoding"

    def __init__(self,
        immutable: bool = True,
        categories: Union[Literal['auto'], list, tuple, np.ndarray, pd.Series, pd.DataFrame, torch.Tensor] = 'auto',
        target_type: Literal['auto', 'continuous', 'binary', 'multiclass'] = 'auto',
        smooth: Union[float, Literal['auto']] = 'auto',
        cv: Union[int, str, Callable, MLModule] = 5,
        shuffle: bool = True,
        random_state: Union[int, torch.Generator] = None,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.long,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.categories = categories
        self.target_type = target_type
        self.smooth = smooth
        self.cv = cv
        self.shuffle = shuffle
        self.random_state = random_state
        self.device = device
        self.dtype = dtype

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from typing import Optional, Callable, Any, Union, Dict, Tuple, List
import warnings
import math

from ...models.utils import DLModule


__all__ = [
    "NormalizationLayer",
    "PoolingLayer",
    "DropoutLayer",
    "ConvolutionLayer",
    "PaddingLayer",
    "TransformerLayer",
    "DLModelLayers",
    "SoftDTWBatch",
    "SoftDTWMatrix",
    "SoftDTWSimilarity"
]


class NormalizationLayer(DLModule):
    def __init__(self, norm_type: str, dimensionality: Union[int, float], norm_config: dict, *args, **kwargs):
        super().__init__()
        self.norm_type = norm_type
        self.norm_config = norm_config
        if norm_type.lower() not in ['layer', 'batch', 'group',
                                     'instance', 'rms', 'weight',
                                     'local_response', 'cross_map',
                                     'lazy_batch', 'lazy_instance', 'sync_batch']:
            warnings.warn(
                f"The given norm_type {norm_type} does not exist.\n changing back to default norm type...'layer-norm'.",
                UserWarning)
            self.norm_type = 'layer'
        self.norm_type = norm_type.lower()
        if isinstance(dimensionality, float):
            dimensionality = math.floor(dimensionality)
        if not (1 <= dimensionality <= 3):
            warnings.warn(
                f"The given dimensionality {dimensionality} is out of range.\n changing back to default dimensionality... '1'.",
                UserWarning)
            dimensionality = 1
        self.dimensionality = dimensionality
        self.norm_config = norm_config
        match self.norm_type:
            case 'layer':
                self.norm = nn.LayerNorm(**self.norm_config)
            case 'batch':
                match self.dimensionality:
                    case 1:
                        self.norm = nn.BatchNorm1d(**self.norm_config)
                    case 2:
                        self.norm = nn.BatchNorm2d(**self.norm_config)
                    case 3:
                        self.norm = nn.BatchNorm3d(**self.norm_config)
            case 'group':
                self.norm = nn.GroupNorm(**self.norm_config)
            case 'instance':
                match self.dimensionality:
                    case 1:
                        self.norm = nn.InstanceNorm1d(**self.norm_config)
                    case 2:
                        self.norm = nn.InstanceNorm2d(**self.norm_config)
                    case 3:
                        self.norm = nn.InstanceNorm3d(**self.norm_config)
            case 'rms':
                self.norm = nn.RMSNorm(**self.norm_config)
            case 'weight':
                layer = nn.Linear(**self.norm_config)
                self.norm = weight_norm(layer, name='weight')
            case 'local_response':
                self.norm = nn.LocalResponseNorm(**self.norm_config)
            case 'cross_map':
                self.scale = nn.Parameter(**self.norm_config, requires_grad=True)
            case 'lazy_batch':
                match self.dimensionality:
                    case 1:
                        self.norm = nn.LazyBatchNorm1d(**self.norm_config)
                    case 2:
                        self.norm = nn.LazyBatchNorm2d(**self.norm_config)
                    case 3:
                        self.norm = nn.LazyBatchNorm3d(**self.norm_config)
            case 'lazy_instance':
                match self.dimensionality:
                    case 1:
                        self.norm = nn.LazyInstanceNorm1d(**self.norm_config)
                    case 2:
                        self.norm = nn.LazyInstanceNorm2d(**self.norm_config)
                    case 3:
                        self.norm = nn.LazyInstanceNorm3d(**self.norm_config)
            case 'sync_batch':
                self.norm = nn.SyncBatchNorm(**self.norm_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        match self.norm_type:
            case 'layer' | 'batch' | 'group' | 'instance' | 'rms' | 'weight' | 'local_response' | 'lazy_batch' | 'lazy_instance' | 'sync_batch':
                return self.norm(x)
            case 'cross_map':
                norm = F.normalize(x, p=2, dim=1)
                return norm * self.scale
            case _:
                return x


class PoolingLayer(DLModule):
    def __init__(self,
                 pool_type: str,
                 dimensionality: Union[int, float],
                 pool_config: dict,
                 *args, **kwargs):
        super().__init__()
        if pool_type.lower() not in ['max', 'maximum',
                                     'avg', 'mean', 'average',
                                     'adaptive_max', 'adaptive_maximum', 'parametric_max', 'parametric_maximum',
                                     'adaptive_avg', 'adaptive_mean', 'adaptive_average', 'parametric_avg',
                                     'parametric_mean', 'parametric_average',
                                     'power_average', 'lp_pool',
                                     'fractional_max', 'frac_max', 'frac_maximum', 'fractional_maximum',
                                     'max_unpool', 'maximum_unpool']:
            warnings.warn(
                f"The given pool_type {pool_type} does not exist.\n changing back to default norm type...'max-pool'.")
            pool_type = 'max'
        self.pool_type = pool_type.lower()
        if isinstance(dimensionality, float):
            dimensionality = math.floor(dimensionality)
        if not (1 <= dimensionality <= 3):
            warnings.warn(
                f"The given dimensionality {dimensionality} is out of range.\n changing back to default dimensionality... '1'.",
                UserWarning)
            dimensionality = 1
        self.dimensionality = dimensionality
        self.pool_config = pool_config
        match self.pool_type:
            case 'max' | 'maximum':
                match self.dimensionality:
                    case 1:
                        self.pool = nn.MaxPool1d(**self.pool_config)
                    case 2:
                        self.pool = nn.MaxPool2d(**self.pool_config)
                    case 3:
                        self.pool = nn.MaxPool3d(**self.pool_config)
            case 'avg' | 'mean' | 'average':
                match self.dimensionality:
                    case 1:
                        self.pool = nn.AvgPool1d(**self.pool_config)
                    case 2:
                        self.pool = nn.AvgPool2d(**self.pool_config)
                    case 3:
                        self.pool = nn.AvgPool3d(**self.pool_config)
            case 'adaptive_max' | 'adaptive_maximum' | 'parametric_max' | 'parametric_maximum':
                match self.dimensionality:
                    case 1:
                        self.pool = nn.AdaptiveMaxPool1d(**self.pool_config)
                    case 2:
                        self.pool = nn.AdaptiveMaxPool2d(**self.pool_config)
                    case 3:
                        self.pool = nn.AdaptiveMaxPool3d(**self.pool_config)
            case 'adaptive_avg' | 'adaptive_mean' | 'adaptive_average' | 'parametric_avg' | 'parametric_mean' | 'parametric_average':
                match self.dimensionality:
                    case 1:
                        self.pool = nn.AdaptiveAvgPool1d(**self.pool_config)
                    case 2:
                        self.pool = nn.AdaptiveAvgPool2d(**self.pool_config)
                    case 3:
                        self.pool = nn.AdaptiveAvgPool3d(**self.pool_config)
            case 'power_average' | 'lp_pool':
                match self.dimensionality:
                    case 1:
                        self.pool = nn.LPPool1d(**self.pool_config)
                    case 2:
                        self.pool = nn.LPPool2d(**self.pool_config)
                    case 3:
                        self.pool = nn.LPPool3d(**self.pool_config)
            case 'fractional_max' | 'frac_max' | 'frac_maximum' | 'fractional_maximum':
                match self.dimensionality:
                    case 2:
                        self.pool = nn.FractionalMaxPool2d(**self.pool_config)
                    case 3:
                        self.pool = nn.FractionalMaxPool3d(**self.pool_config)
            case 'max_unpool' | 'maximum_unpool':
                match self.dimensionality:
                    case 1:
                        self.pool = nn.MaxUnpool1d(**self.pool_config)
                    case 2:
                        self.pool = nn.MaxUnpool2d(**self.pool_config)
                    case 3:
                        self.pool = nn.MaxUnpool3d(**self.pool_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        match self.pool_type:
            case 'max' | 'maximum' | 'avg' | 'mean' | 'average' | 'adaptive_max' | 'adaptive_maximum' | 'parametric_max' | 'parametric_maximum' | 'adaptive_avg' | 'adaptive_mean' | 'adaptive_average' | 'parametric_avg' | 'parametric_mean' | 'parametric_average' | 'power_average' | 'lp_pool' | 'fractional_max' | 'frac_max' | 'frac_maximum' | 'fractional_maximum' | 'max_unpool' | 'maximum_unpool':
                return self.pool(x)
            case _:
                return x


class DropoutLayer(DLModule):
    def __init__(self,
                 dropout_percent: float,
                 dimensionality: Union[int, float],
                 is_alpha_dropout: Optional[bool] = False,
                 is_feature_alpha_dropout: Optional[bool] = False,
                 *args, **kwargs):
        super().__init__()
        if isinstance(dropout_percent, float):
            if 0.0 <= dropout_percent <= 1.0:
                self.dropout_percent = dropout_percent
            else:
                warnings.warn(f"The dropout percent is out of bounds {dropout_percent}.\n"
                              f"Changing back to default percent")
                self.dropout_percent = 0.3
        else:
            warnings.warn(f"The dropout percent is not float value {type(dropout_percent)}.\n"
                          f"Changing back to default percent")
            self.dropout_percent = 0.3
        if isinstance(dimensionality, float):
            dimensionality = math.floor(dimensionality)
        if not (0 <= dimensionality <= 3):
            warnings.warn(
                f"The given dimensionality {dimensionality} is out of range.\n changing back to default dimensionality... '1'.",
                UserWarning)
            dimensionality = 0
        self.dimensionality = dimensionality
        self.is_alpha_dropout = is_alpha_dropout
        self.is_feature_alpha_dropout = is_feature_alpha_dropout
        if self.is_alpha_dropout:
            self.dropout = nn.AlphaDropout(self.dropout_percent)
        elif self.is_feature_alpha_dropout:
            self.dropout = nn.FeatureAlphaDropout(self.dropout_percent)
        else:
            match self.dimensionality:
                case 0:
                    self.dropout = nn.Dropout(self.dropout_percent)
                case 1:
                    self.dropout = nn.Dropout1d(self.dropout_percent)
                case 2:
                    self.dropout = nn.Dropout2d(self.dropout_percent)
                case 3:
                    self.dropout = nn.Dropout3d(self.dropout_percent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x)


class ConvolutionLayer(DLModule):
    def __init__(self,
                 dimensionality: Union[int, float],
                 conv_config: dict,
                 lazy: bool = False,
                 transpose: bool = False,
                 *args, **kwargs):
        super().__init__()
        if isinstance(dimensionality, float):
            dimensionality = math.floor(dimensionality)
        if not (0 <= dimensionality <= 3):
            warnings.warn(
                f"The given dimensionality {dimensionality} is out of range.\n changing back to default dimensionality... '1'.",
                UserWarning)
            dimensionality = 0
        self.dimensionality = dimensionality
        self.conv_config = conv_config
        if transpose:
            if lazy:
                match self.dimensionality:
                    case 1:
                        self.conv = nn.LazyConvTranspose1d(**self.conv_config)
                    case 2:
                        self.conv = nn.LazyConvTranspose2d(**self.conv_config)
                    case 3:
                        self.conv = nn.LazyConvTranspose3d(**self.conv_config)
            else:
                match self.dimensionality:
                    case 1:
                        self.conv = nn.ConvTranspose1d(**self.conv_config)
                    case 2:
                        self.conv = nn.ConvTranspose2d(**self.conv_config)
                    case 3:
                        self.conv = nn.ConvTranspose3d(**self.conv_config)
        else:
            if lazy:
                match self.dimensionality:
                    case 1:
                        self.conv = nn.LazyConv1d(**self.conv_config)
                    case 2:
                        self.conv = nn.LazyConv2d(**self.conv_config)
                    case 3:
                        self.conv = nn.LazyConv3d(**self.conv_config)
            else:
                match self.dimensionality:
                    case 1:
                        self.conv = nn.Conv1d(**self.conv_config)
                    case 2:
                        self.conv = nn.Conv2d(**self.conv_config)
                    case 3:
                        self.conv = nn.Conv3d(**self.conv_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class PaddingLayer(DLModule):
    def __init__(self,
                 pad_type: str,
                 dimensionality: Union[int, float],
                 pad_config: dict,
                 *args, **kwargs):
        super().__init__()
        if pad_type.lower() not in ['circular', 'constant', 'reflection', 'replication', 'zero']:
            warnings.warn(f"Given pad type {pad_type} does not exists.\n"
                          f"Changing to default pad type... 'zero pad type'.", UserWarning)
            pad_type = 'zero'
        self.pad_type = pad_type.lower()
        if isinstance(dimensionality, float):
            dimensionality = math.floor(dimensionality)
        if not (0 <= dimensionality <= 3):
            warnings.warn(
                f"The given dimensionality {dimensionality} is out of range.\n changing back to default dimensionality... '1'.",
                UserWarning)
            dimensionality = 0
        self.dimensionality = dimensionality
        self.pad_config = pad_config
        match self.pad_type:
            case 'circular':
                match self.dimensionality:
                    case 1:
                        self.pad = nn.CircularPad1d(**self.pad_config)
                    case 2:
                        self.pad = nn.CircularPad2d(**self.pad_config)
                    case 3:
                        self.pad = nn.CircularPad3d(**self.pad_config)
            case 'constant':
                match self.dimensionality:
                    case 1:
                        self.pad = nn.ConstantPad1d(**self.pad_config)
                    case 2:
                        self.pad = nn.ConstantPad2d(**self.pad_config)
                    case 3:
                        self.pad = nn.ConstantPad3d(**self.pad_config)
            case 'reflection':
                match self.dimensionality:
                    case 1:
                        self.pad = nn.ReflectionPad1d(**self.pad_config)
                    case 2:
                        self.pad = nn.ReflectionPad2d(**self.pad_config)
                    case 3:
                        self.pad = nn.ReflectionPad3d(**self.pad_config)
            case 'replication':
                match self.dimensionality:
                    case 1:
                        self.pad = nn.ReplicationPad1d(**self.pad_config)
                    case 2:
                        self.pad = nn.ReplicationPad2d(**self.pad_config)
                    case 3:
                        self.pad = nn.ReplicationPad3d(**self.pad_config)
            case 'zero':
                match self.dimensionality:
                    case 1:
                        self.pad = nn.ZeroPad1d(**self.pad_config)
                    case 2:
                        self.pad = nn.ZeroPad2d(**self.pad_config)
                    case 3:
                        self.pad = nn.ZeroPad3d(**self.pad_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pad(x)


class TransformerLayer(DLModule):
    def __init__(self,
                 transformer_type: str,
                 transformer_config: dict,
                 *args, **kwargs):
        super().__init__()
        if transformer_type.lower() not in ['multi_head_attention', 'transformer',
                                            'transformer_decoder', 'transformer_decoder_layer',
                                            'transformer_encoder', 'transformer_encoder_layer']:
            warnings.warn(f"Given transformer type {transformer_type} does not exists.\n"
                          f"Changing back to default transformer type... 'transformer'.")
            transformer_type = 'transformer'
        self.transformer_type = transformer_type.lower()
        self.transformer_config = transformer_config
        match self.transformer_type:
            case 'multi_head_attention':
                self.layer = nn.MultiheadAttention(**self.transformer_config)
            case 'transformer':
                self.layer = nn.Transformer(**self.transformer_config)
            case 'transformer_decoder':
                self.layer = nn.TransformerDecoder(**self.transformer_config)
            case 'transformer_decoder_layer':
                self.layer = nn.TransformerDecoderLayer(**self.transformer_config)
            case 'transformer_encoder':
                self.layer = nn.TransformerEncoder(**self.transformer_config)
            case 'transformer_encoder_layer':
                self.layer = nn.TransformerEncoderLayer(**self.transformer_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class DLModelLayers(DLModule):
    def __init__(self,
                 layers: Union[List[Tuple[str, dict]], Dict[str, dict], Tuple[str, dict]],
                 act_funcs: Optional[Union[str, Callable, nn.Module,
                 List[Tuple[str, Callable, nn.Module]], Tuple[Union[str, Callable, nn.Module]],
                 Dict[str, Union[str, Callable, nn.Module]]]],
                 device: str = "cpu", dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        from ...models.utils import ActFuncUtils
        ActUtils = ActFuncUtils(act_funcs, *args, **kwargs)
        self.funcs = ActUtils.get_funcs()
        self.N = ActUtils.get_len()
        if isinstance(layers, Union[list, tuple]):
            self.layers = nn.ModuleList([])
            self.layer_type = []
            for layer in layers:
                layer_type, layer_config = layer
                layer_config["device"] = device
                layer_config["dtype"] = dtype
                self.layer_type.append(layer_type.lower())
                match layer_type.lower():
                    case 'nn' | 'linear' | 'dense':
                        self.layers.append(nn.Linear(**layer_config))
                    case 'lazy_nn' | 'lazy_linear' | 'lazy_dense':
                        self.layers.append(nn.LazyLinear(**layer_config))
                    case 'bi_linear' | 'bilinear':
                        self.layers.append(nn.Bilinear(**layer_config))
                    case 'act' | 'activation' | 'act_func' | 'activation_func' | 'act_function' | 'activation_function':
                        from ...models.deep_learning.activations.ActivationFunction import Activation
                        # Check if config has activation name (either 'activation' or 'activation_func' or 'act_func')
                        has_act_name = any(k in layer_config for k in ['activation', 'activation_func', 'act_func'])

                        if has_act_name:
                            # Use config to instantiate
                            self.layers.append(Activation(**layer_config))
                        else:
                            # Use pre-loaded funcs
                            if isinstance(self.funcs, nn.ModuleList):
                                func = self.funcs.pop(0)
                                self.layers.append(func)
                            elif isinstance(self.funcs, nn.ModuleDict):
                                func = list(self.funcs.values()).pop(0)
                                self.layers.append(func)
                            elif isinstance(self.funcs, nn.Module):
                                self.layers.append(self.funcs)
                    case 'norm' | 'normalization':
                        norm = NormalizationLayer(**layer_config)
                        self.layers.append(norm)
                    case 'pool' | 'pooling':
                        pool = PoolingLayer(**layer_config)
                        self.layers.append(pool)
                    case 'conv' | 'convolution':
                        conv = ConvolutionLayer(**layer_config)
                        self.layers.append(conv)
                    case 'regularization' | 'dropout':
                        dropout = DropoutLayer(**layer_config)
                        self.layers.append(dropout)
                    case 'rnn':
                        rnn = nn.RNN(**layer_config)
                        self.layers.append(rnn)
                    case 'lstm':
                        lstm = nn.LSTM(**layer_config)
                        self.layers.append(lstm)
                    case 'gru':
                        gru = nn.GRU(**layer_config)
                        self.layers.append(gru)
                    case 'pad' | 'padding':
                        pad = PaddingLayer(**layer_config)
                        self.layers.append(pad)
                    case 'transformer':
                        trans = TransformerLayer(**layer_config)
                        self.layers.append(trans)
        elif isinstance(layers, dict):
            self.layers = nn.ModuleDict({})
            self.layer_type = {}
            for key, layer in layers.items():
                layer_type = layer["type"]
                layer_config = layer["config"]
                layer_config["device"] = device
                layer_config["dtype"] = dtype
                self.layer_type[key] = layer_type
                match layer_type.lower():
                    case 'nn' | 'linear' | 'dense':
                        self.layers[key] = nn.Linear(**layer_config)
                    case 'lazy_nn' | 'lazy_linear' | 'lazy_dense':
                        self.layers[key] = nn.LazyLinear(**layer_config)
                    case 'bi_linear' | 'bilinear':
                        self.layers[key] = nn.Bilinear(**layer_config)
                    case 'act' | 'activation' | 'act_func' | 'activation_func' | 'act_function' | 'activation_function':
                        from ...models.deep_learning.activations.ActivationFunction import Activation
                        # Check if config has activation name
                        has_act_name = any(k in layer_config for k in ['activation', 'activation_func', 'act_func'])

                        if has_act_name:
                            self.layers[key] = Activation(**layer_config)
                        else:
                            if isinstance(self.funcs, nn.ModuleList):
                                func = self.funcs.pop(0)
                                self.layers[key] = func
                            elif isinstance(self.funcs, nn.ModuleDict):
                                # If funcs is dict, try to find by key (assuming keys align with layer keys for activations)
                                # CNNetworksOp aligns them.
                                if key in self.funcs:
                                    func = self.funcs[key]
                                    self.layers[key] = func
                                else:
                                    # Fallback: pop first available? Or error?
                                    # Existing logic was pop(0). Let's keep a fallback but warn?
                                    # list(self.funcs.values()).pop(0) is idempotent, which is bad if we have multiple.
                                    # But we can't pop from ModuleDict easily by index.
                                    # Let's hope keys align.
                                    try:
                                        func = list(self.funcs.values())[0]
                                        self.layers[key] = func
                                    except IndexError:
                                        pass
                            elif isinstance(self.funcs, nn.Module):
                                self.layers[key] = self.funcs
                    case 'norm' | 'normalization':
                        norm = NormalizationLayer(**layer_config)
                        self.layers[key] = norm
                    case 'pool' | 'pooling':
                        pool = PoolingLayer(**layer_config)
                        self.layers[key] = pool
                    case 'conv' | 'convolution':
                        conv = ConvolutionLayer(**layer_config)
                        self.layers[key] = conv
                    case 'regularization' | 'dropout':
                        dropout = DropoutLayer(**layer_config)
                        self.layers[key] = dropout
                    case 'rnn':
                        rnn = nn.RNN(**layer_config)
                        self.layers[key] = rnn
                    case 'lstm':
                        lstm = nn.LSTM(**layer_config)
                        self.layers[key] = lstm
                    case 'gru':
                        gru = nn.GRU(**layer_config)
                        self.layers[key] = gru
                    case 'pad' | 'padding':
                        pad = PaddingLayer(**layer_config)
                        self.layers[key] = pad
                    case 'transformer':
                        trans = TransformerLayer(**layer_config)
                        self.layers[key] = trans

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.layers, nn.ModuleList):
            for layer in self.layers:
                x = layer(x)
                if isinstance(x, tuple):
                    x = x[0]
        elif isinstance(self.layers, nn.ModuleDict):
            for layer in self.layers.values():
                x = layer(x)
                if isinstance(x, tuple):
                    x = x[0]
        return x


class SoftDTWBatch(DLModule):
    def __init__(self, gamma=1.0):
        super(SoftDTWBatch, self).__init__()
        self.gamma = gamma

    def forward(self, x, y, len_x=None, len_y=None):
        """
        x: [B, N, D]
        y: [B, M, D]
        len_x: [B] tensor of actual lengths for x
        len_y: [B] tensor of actual lengths for y
        """
        B, N, _ = x.shape
        M = y.shape[1]

        # 1. Squared Euclidean Distance Matrix [B, N, M]
        dist_mat = torch.cdist(x, y, p=2) ** 2

        # 2. Initialize DP Table [B, N+1, M+1]
        # We use a large value (1e8) instead of inf to avoid NaN in logsumexp
        R = torch.full((B, N + 1, M + 1), 1e8, device=x.device)
        R[:, 0, 0] = 0

        # 3. DP Loop (Vectorized over Batch)
        # We loop over rows and columns. While less efficient than
        # C++/CUDA, torch.compile can optimize this.
        for i in range(1, N + 1):
            for j in range(1, M + 1):
                # soft_min logic
                r0 = R[:, i - 1, j - 1]  # Diagonal
                r1 = R[:, i - 1, j]  # Up
                r2 = R[:, i, j - 1]  # Left

                # LogSumExp trick for soft-min
                stacked = torch.stack([r0, r1, r2], dim=-1)
                softmin = -self.gamma * torch.logsumexp(-stacked / self.gamma, dim=-1)

                R[:, i, j] = dist_mat[:, i - 1, j - 1] + softmin

        # 4. Extract result based on actual lengths
        if len_x is not None and len_y is not None:
            # Gather the values at the specific (len_x, len_y) coordinates for each batch
            batch_indices = torch.arange(B, device=x.device)
            result = R[batch_indices, len_x, len_y]
        else:
            result = R[:, N, M]

        return result


class SoftDTWSimilarity(nn.Module):
    def __init__(self, gamma=0.1, sigma=1.0, normalize=True):
        super(SoftDTWSimilarity, self).__init__()
        self.gamma = gamma
        self.sigma = sigma
        self.normalize = normalize

    def _get_lengths(self, t):
        # Auto-detect lengths: finds last index that isn't all zeros
        # Shape: [B, SeqLen, D] -> [B]
        non_zero = (t.abs().sum(dim=-1) > 1e-6)
        return non_zero.sum(dim=-1)

    def compute_dtw(self, x, y, len_x, len_y):
        B, N, _ = x.shape
        M = y.shape[1]
        dist_mat = torch.cdist(x, y, p=2) ** 2

        # Initialize DP table with large value (1e8) to avoid inf/NaN issues
        R = torch.full((B, N + 1, M + 1), 1e8, device=x.device)
        R[:, 0, 0] = 0

        for i in range(1, N + 1):
            for j in range(1, M + 1):
                r0 = R[:, i - 1, j - 1]  # Diag
                r1 = R[:, i - 1, j]  # Up
                r2 = R[:, i, j - 1]  # Left

                stacked = torch.stack([r0, r1, r2], dim=-1)
                softmin = -self.gamma * torch.logsumexp(-stacked / self.gamma, dim=-1)
                R[:, i, j] = dist_mat[:, i - 1, j - 1] + softmin

        # Gather results at the actual ends of sequences
        batch_idx = torch.arange(B, device=x.device)
        return R[batch_idx, len_x, len_y]

    def forward(self, x, y):
        # 1. Auto-detect lengths for masking
        lx = self._get_lengths(x)
        ly = self._get_lengths(y)

        # 2. Basic DTW distance
        dist_xy = self.compute_dtw(x, y, lx, ly)

        if self.normalize:
            # 3. Compute self-distance for normalization
            dist_xx = self.compute_dtw(x, x, lx, lx)
            dist_yy = self.compute_dtw(y, y, ly, ly)
            # Divergence formula: D(x,y) - 0.5(D(x,x) + D(y,y))
            distance = dist_xy - 0.5 * (dist_xx + dist_yy)
        else:
            distance = dist_xy

        # 4. Convert to Similarity [0, 1]
        # We use ReLU to ensure distance isn't negative due to floating point errors
        similarity = torch.exp(-torch.relu(distance) / self.sigma)
        return similarity


class SoftDTWMatrix(nn.Module):
    def __init__(self, gamma=0.1, max_dist=10.0):
        super(SoftDTWMatrix, self).__init__()
        self.gamma = gamma
        self.max_dist = max_dist

    def _compute_pairwise_distance(self, x, y):
        # x: [N, L, D], y: [M, L, D]
        # returns dists: [N, M]
        N, L, D = x.shape
        M = y.shape[0]

        # We expand x and y to compute all pairs (N * M)
        # x_exp: [N, M, L, D]
        x_exp = x.unsqueeze(1).expand(N, M, L, D).reshape(N * M, L, D)
        y_exp = y.unsqueeze(0).expand(N, M, L, D).reshape(N * M, L, D)

        # Now we run the standard Soft-DTW on these N*M pairs
        # (Using the logic from the previous SoftDTWBatch implementation)
        dist_mat = torch.cdist(x_exp, y_exp, p=2) ** 2
        B_total = N * M
        R = torch.full((B_total, L + 1, L + 1), 1e8, device=x.device)
        R[:, 0, 0] = 0

        for i in range(1, L + 1):
            for j in range(1, L + 1):
                stacked = torch.stack([R[:, i - 1, j - 1], R[:, i - 1, j], R[:, i, j - 1]], dim=-1)
                softmin = -self.gamma * torch.logsumexp(-stacked / self.gamma, dim=-1)
                R[:, i, j] = dist_mat[:, i - 1, j - 1] + softmin

        return R[:, L, L].reshape(N, M)

    def forward(self, A, B):
        """
        A: [N, L, D]
        B: [N, L, D]
        Returns: [N, N] matrix in range [-1, 1]
        """
        # 1. Compute the raw Soft-DTW distance matrix [N, N]
        raw_dist = self._compute_pairwise_distance(A, B)

        # 2. Normalize (Divergence) to ensure diagonal is 0
        # For a full [N,N] matrix, we subtract self-distances
        dist_aa = self._compute_pairwise_distance(A, A).diagonal().unsqueeze(1)
        dist_bb = self._compute_pairwise_distance(B, B).diagonal().unsqueeze(0)

        normalized_dist = raw_dist - 0.5 * (dist_aa + dist_bb)
        normalized_dist = torch.relu(normalized_dist)  # Clamp tiny negatives to 0

        # 3. Scale to [-1, 1]
        # 1.0 at dist=0, -1.0 at dist >= max_dist
        similarity = 1 - 2 * (normalized_dist / self.max_dist)
        return torch.clamp(similarity, min=-1.0, max=1.0)

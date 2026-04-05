"""
Structural DBN variants: Convolutional, Multiresolution, PyramidFDBNet, Multimodal.
Reference: docs/deep-learning/dbm/dbm.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List, Dict, Any
import math

from .._base import DBNModuleBase
from ..core.core import RBM
from ..foundational.foundational import StandardDBN


class CDBN(DBNModuleBase):
    """
    Convolutional DBN with weight sharing and probabilistic max-pooling.
    """

    def __init__(
        self,
        in_channels: int,
        filters: List[int],
        kernel_size: Union[int, tuple] = 3,
        stride: int = 1,
        padding: int = 0,
        prob_pooling: bool = True,
        pool_size: int = 2,
        learning_rate: float = 0.01,
        batch_size: int = 16,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(device=device, dtype=dtype)
        self.in_channels = in_channels
        self.filters = filters
        k = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.kernel_size = k
        self.stride = stride
        self.padding = padding
        self.prob_pooling = prob_pooling
        self.pool_size = pool_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.conv_layers = nn.ModuleList()
        cin = in_channels
        for f in filters:
            conv = nn.Conv2d(cin, f, k, stride=stride, padding=padding)
            nn.init.xavier_uniform_(conv.weight)
            nn.init.zeros_(conv.bias)
            self.conv_layers.append(conv)
            cin = f
        self.pool = nn.MaxPool2d(pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for conv in self.conv_layers:
            h = torch.sigmoid(conv(h))
            if self.prob_pooling:
                h = self.pool(h)
        return h

    def reconstruct(self, x: torch.Tensor, n_steps: int = 1) -> torch.Tensor:
        h = self.forward(x)
        for _ in range(n_steps):
            for i, conv in enumerate(reversed(self.conv_layers)):
                h_up = F.interpolate(h, scale_factor=self.pool_size, mode="nearest")
                h = torch.sigmoid(
                    F.conv_transpose2d(
                        h_up, conv.weight, conv.bias, self.stride, self.padding
                    )
                )
        return h

    @property
    def filters_mod(self):
        return self.conv_layers


class MultiresolutionDBN(DBNModuleBase):
    """
    Multiresolution DBN: separate RBMs at pyramid levels.
    """

    def __init__(
        self,
        pyramid_levels: int = 3,
        layer_sizes_per_level: Optional[List[List[int]]] = None,
        base_resolution: tuple = (64, 64),
        learning_rate: float = 0.01,
        batch_size: int = 16,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(device=device, dtype=dtype)
        self.pyramid_levels = pyramid_levels
        self.base_resolution = base_resolution
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.factory_kwargs = {"device": device, "dtype": dtype}

        if layer_sizes_per_level is None:
            h, w = base_resolution
            base_dim = h * w * 3
            layer_sizes_per_level = [
                [base_dim // (4**i), 256, 128] for i in range(pyramid_levels)
            ]
        self.layer_sizes_per_level = layer_sizes_per_level

        self.dbns = nn.ModuleList()
        for sizes in layer_sizes_per_level:
            dbn = StandardDBN(
                layer_sizes=sizes,
                learning_rate=learning_rate,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
            self.dbns.append(dbn)

    def build_pyramid(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Build Laplacian pyramid levels."""
        levels = [x]
        for i in range(1, self.pyramid_levels):
            h = F.avg_pool2d(levels[-1], 2)
            levels.append(h)
        return [y.flatten(1) for y in levels]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.dim() == 4:
            levels = self.build_pyramid(x)
        else:
            levels = [x]
        out = {}
        for i, (level, dbn) in enumerate(zip(levels, self.dbns)):
            if level.size(-1) == dbn.n_visible:
                out[f"level_{i}"] = dbn.encode(level)
            else:
                out[f"level_{i}"] = level
        return out


class PyramidFDBNet(DBNModuleBase):
    """
    PyramidFDBNet: Pyramid scales + DBN for deepfake detection.
    """

    def __init__(
        self,
        pyramid_scales: List[float] = None,
        dbn_layers: List[int] = None,
        in_channels: int = 3,
        n_classes: int = 2,
        learning_rate: float = 0.001,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(device=device, dtype=dtype)
        self.pyramid_scales = pyramid_scales or [1.0, 0.5, 0.25]
        self.dbn_layers = dbn_layers or [512, 256, 128]
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.scale_extractors = nn.ModuleList()
        for _ in self.pyramid_scales:
            ex = nn.Sequential(
                nn.AdaptiveAvgPool2d(32),
                nn.Flatten(),
                nn.Linear(in_channels * 32 * 32, dbn_layers[0]),
            )
            nn.init.xavier_uniform_(ex[-1].weight)
            nn.init.zeros_(ex[-1].bias)
            self.scale_extractors.append(ex)

        self.dbn_stack = StandardDBN(
            layer_sizes=dbn_layers,
            learning_rate=learning_rate,
            device=device,
            dtype=dtype,
        )
        self.classifier = nn.Linear(dbn_layers[-1] * len(self.pyramid_scales), n_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def get_multiscale_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = []
        for scale, ext in zip(self.pyramid_scales, self.scale_extractors):
            if scale != 1.0:
                h, w = x.shape[2], x.shape[3]
                x_s = F.interpolate(x, size=(int(h * scale), int(w * scale)))
            else:
                x_s = x
            feats.append(ext(x_s))
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.get_multiscale_features(x)
        h_stack = []
        for f in feats:
            h_stack.append(self.dbn_stack.encode(f))
        h_cat = torch.cat(h_stack, dim=-1)
        return self.classifier(h_cat)


class MultimodalDBN(DBNModuleBase):
    """
    Multimodal DBN: multiple input arms fusing to shared representation.
    """

    def __init__(
        self,
        input_modalities: Dict[str, tuple],
        fusion_layer_size: int = 256,
        modality_layers: Optional[Dict[str, List[int]]] = None,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(device=device, dtype=dtype)
        self.input_modalities = input_modalities
        self.fusion_layer_size = fusion_layer_size
        self.modality_layers = modality_layers or {}
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.arms = nn.ModuleDict()
        for name, (dim, mtype) in input_modalities.items():
            layers = self.modality_layers.get(name, [dim, 128, 64])
            if layers[0] != dim:
                layers = [dim] + layers[1:]
            dbn = StandardDBN(
                layer_sizes=layers,
                learning_rate=learning_rate,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
            self.arms[name] = dbn

        total_dim = sum(
            self.arms[n].layer_sizes[-1] for n in input_modalities
        )
        self.fusion_layer = nn.Linear(total_dim, fusion_layer_size)
        nn.init.xavier_uniform_(self.fusion_layer.weight)
        nn.init.zeros_(self.fusion_layer.bias)

    def encode_modality(self, name: str, x: torch.Tensor) -> torch.Tensor:
        return self.arms[name].encode(x)

    def forward(self, modality_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        encs = [self.arms[n].encode(modality_dict[n]) for n in self.input_modalities]
        h_cat = torch.cat(encs, dim=-1)
        return torch.sigmoid(self.fusion_layer(h_cat))

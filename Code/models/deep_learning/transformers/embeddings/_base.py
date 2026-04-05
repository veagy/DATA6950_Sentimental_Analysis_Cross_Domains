"""
Base class for embedding modules. Provides device/dtype helpers.
Reference: docs/deep-learning/embeddings/embeddings.md
"""

import torch
import torch.nn as nn
from typing import Optional, Union

from .....models.utils import DLModule


class BaseEmbedding(DLModule):
    """
    Base class for all embedding modules.
    Ensures consistent device and dtype handling across embeddings.
    """

    def __init__(
        self,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._device = torch.device(device) if isinstance(device, str) else device
        self._dtype = dtype

    def _to_device_dtype(self, x: torch.Tensor) -> torch.Tensor:
        """Move tensor to model device and dtype."""
        return x.to(device=self.device_param, dtype=self.dtype_param)

    def _ensure_device_dtype(self, *tensors: torch.Tensor) -> tuple:
        """Ensure all tensors are on correct device and dtype."""
        return tuple(self._to_device_dtype(t) for t in tensors)

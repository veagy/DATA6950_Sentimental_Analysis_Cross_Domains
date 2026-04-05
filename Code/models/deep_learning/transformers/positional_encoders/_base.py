"""
Base class for positional encoding modules. Provides device/dtype helpers.
Reference: docs/deep-learning/positional encoders/positional_encoder.md
"""

import torch
import torch.nn as nn
from typing import Optional, Union

from .....models.utils import DLModule


class BasePositionalEncoder(DLModule):
    """
    Base class for all positional encoding modules.
    Ensures consistent device and dtype handling across encoders.
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
        dev = self.device_param if hasattr(self, 'parameters') and any(self.parameters()) else self._device
        dt = self.dtype_param if hasattr(self, 'parameters') and any(self.parameters()) else self._dtype
        return x.to(device=dev, dtype=dt)

    def _ensure_device_dtype(self, *tensors: torch.Tensor) -> tuple:
        """Ensure all tensors are on correct device and dtype."""
        return tuple(self._to_device_dtype(t) for t in tensors)

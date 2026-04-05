"""
Base class for normalization modules. Provides device/dtype helpers.
Reference: docs/deep-learning/norm/norm.md
"""

import torch
import torch.nn as nn
from typing import Optional, Union

from .....models.utils import DLModule


class BaseNorm(DLModule):
    """
    Base class for all normalization modules.
    Ensures consistent device and dtype handling across normalizers.
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

    def _to_device_dtype(self, x: torch.Tensor, device: Optional[Union[str, torch.device]] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Move tensor to model device and dtype, or override if provided."""
        dev = device or (next(self.parameters()).device if any(self.parameters()) else self._device)
        dt = dtype or (next(self.parameters()).dtype if any(self.parameters()) else self._dtype)
        return x.to(device=dev, dtype=dt)

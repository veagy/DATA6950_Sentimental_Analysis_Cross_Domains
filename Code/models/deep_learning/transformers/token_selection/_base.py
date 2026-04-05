"""
Base class for token selection modules. Provides device/dtype helpers.
Reference: docs/deep-learning/token-selection/token_selection.md
"""

import torch
from typing import Optional, Union

from .....models.utils import DLModule


class BaseTokenSelector(DLModule):
    """
    Base class for all token selection modules.
    Ensures consistent device and dtype handling.
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

    def _to_device_dtype(
        self,
        t: torch.Tensor,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """Move tensor to model device and dtype, or override if provided."""
        try:
            dev = device or next(self.parameters()).device
            dt = dtype or next(self.parameters()).dtype
        except StopIteration:
            dev = device or self._device
            dt = dtype or self._dtype
        return t.to(device=dev, dtype=dt)

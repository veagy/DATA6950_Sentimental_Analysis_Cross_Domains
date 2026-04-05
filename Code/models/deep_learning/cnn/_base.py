"""
Base class for CNN modules with device/dtype support.
Reference: docs/deep-learning/cnn/cnn.md
"""

import torch
from typing import Optional, Union, Any, Dict

from ....models.utils import DLModule


__all__ = ["CNNModuleBase"]


class CNNModuleBase(DLModule):
    """
    Base class for CNN modules with device/dtype support.
    Subclasses should set factory_kwargs and pass to nn.Parameter, torch.* constructors.
    """

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = "cpu",
        dtype: Optional[torch.dtype] = torch.float32,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._device = device
        self._dtype = dtype
        self.factory_kwargs: Dict[str, Any] = {
            "device": device if device is not None else "cpu",
            "dtype": dtype if dtype is not None else torch.float32,
        }

    def _get_device_dtype(self) -> tuple:
        """Return (device, dtype) from first parameter or factory_kwargs."""
        try:
            p = next(self.parameters())
            return p.device, p.dtype
        except StopIteration:
            dev = self.factory_kwargs.get("device", "cpu")
            dt = self.factory_kwargs.get("dtype", torch.float32)
            if isinstance(dev, str):
                dev = torch.device(dev)
            return dev, dt

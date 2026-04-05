"""
Base class and utilities for linear layers.
Reference: docs/deep-learning/linear-layers/linear.md
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Any, Dict

from .....models.utils import DLModule


__all__ = ["LinearLayerBase"]


class LinearLayerBase(DLModule):
    """
    Base class for linear layers with device/dtype support.
    Subclasses should set factory_kwargs and pass to nn.Parameter, torch.* constructors.
    """

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._device = device
        self._dtype = dtype
        self.factory_kwargs: Dict[str, Any] = {
            "device": device,
            "dtype": dtype,
        }

    def _get_factory_kwargs(self, x: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Infer device/dtype from first parameter or input tensor if not set."""
        kwargs = dict(self.factory_kwargs)
        if kwargs.get("device") is None or kwargs.get("dtype") is None:
            if x is not None:
                if kwargs.get("device") is None:
                    kwargs["device"] = x.device
                if kwargs.get("dtype") is None:
                    kwargs["dtype"] = x.dtype
            else:
                try:
                    p = next(self.parameters())
                    if kwargs.get("device") is None:
                        kwargs["device"] = p.device
                    if kwargs.get("dtype") is None:
                        kwargs["dtype"] = p.dtype
                except StopIteration:
                    if kwargs.get("device") is None:
                        kwargs["device"] = torch.device("cpu")
                    if kwargs.get("dtype") is None:
                        kwargs["dtype"] = torch.float32
        return {k: v for k, v in kwargs.items() if v is not None}

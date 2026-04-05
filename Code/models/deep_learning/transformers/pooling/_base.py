"""
Base class for pooling modules. Provides device/dtype and input unpacking helpers.
Reference: docs/deep-learning/pooling/pooling.md
"""

import torch
from typing import Optional, Tuple, Union

from .....models.utils import DLModule


class BasePooling(DLModule):
    """
    Base class for all pooling modules.
    Ensures consistent device/dtype handling and unified input API (dict or tensor).
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

    def _get_inputs(
        self,
        x: Union[torch.Tensor, dict],
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unpack inputs: x can be dict with token_embeddings/attention_mask or tensor.
        Returns (token_embeddings, attention_mask).
        """
        if isinstance(x, dict):
            te = x["token_embeddings"]
            m = x.get(
                "attention_mask",
                torch.ones(te.shape[0], te.shape[1], device=te.device, dtype=torch.int64)
            )
            if m.dtype != te.dtype and te.dtype in (torch.float16, torch.float32, torch.bfloat16):
                m = m.to(te.dtype)
            return te, m
        if mask is None:
            mask = torch.ones(x.shape[0], x.shape[1], device=x.device, dtype=x.dtype)
        return x, mask

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

    def get_sentence_embedding_dimension(self) -> int:
        """Return output embedding dimension. Override in subclasses."""
        raise NotImplementedError

"""
Base class for tokenizer modules. Provides device/dtype helpers.
Reference: docs/deep-learning/tokenizer/tokenizer.md
"""

import torch
from typing import Optional, Union, List, Dict, Any

from .....models.utils import DLModule


class BaseTokenizer(DLModule):
    """
    Base class for all tokenizer modules.
    Ensures consistent device and dtype handling across tokenizers.
    Pure tokenizers return list[int] from encode(); neural tokenizers may return tensors.
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

    def encode(
        self,
        text: str,
        *,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        return_tensors: bool = False,
        **kwargs
    ) -> Union[List[int], Dict[str, Any]]:
        """
        Encode text to token IDs. Override in subclasses.
        When return_tensors=True, returns dict with tensors on device/dtype.
        """
        raise NotImplementedError

    def decode(
        self,
        token_ids: List[int],
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
        **kwargs
    ) -> str:
        """Decode token IDs to text. Override in subclasses."""
        raise NotImplementedError

    def tokenize(self, text: str, **kwargs) -> List[str]:
        """Tokenize text to token strings. Override in subclasses."""
        raise NotImplementedError

    def get_vocab(self) -> Dict[str, int]:
        """Return vocabulary (token -> id). Override in subclasses."""
        raise NotImplementedError

    def get_vocab_size(self) -> int:
        """Return vocabulary size. Override in subclasses."""
        raise NotImplementedError

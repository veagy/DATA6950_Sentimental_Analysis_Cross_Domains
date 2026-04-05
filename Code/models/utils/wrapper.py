"""
DLModuleWrapper — promote any nn.Module into a full DLModule.

Wrapping any PyTorch module with this class gives it the complete
DLModule feature set:
  - fit()           — full training loop (AMP, gradient accumulation,
                      accelerate, checkpointing)
  - fine_tune()     — PEFT / LoRA / QLoRA fine-tuning
  - from_pretrained / save_pretrained / load_model / save_model
  - quantize() / dequantize()
  - register_forward_hook / register_backward_hook
  - CLI / system integration

This is particularly useful for:
  - Third-party nn.Module objects (e.g. HuggingFace models loaded
    manually, torchvision models, custom research implementations)
  - Any nn.Module that needs to plug into the project's training
    infrastructure without rewriting it to inherit DLModule directly

Usage
-----
>>> import torch.nn as nn
>>> from Code.models.utils import DLModuleWrapper
>>>
>>> # Wrap any nn.Module
>>> plain_model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
>>> wrapped = DLModuleWrapper(plain_model)
>>> wrapped.fit(train_data, epochs=10, ...)
>>>
>>> # One-step factory shorthand
>>> wrapped = DLModuleWrapper.wrap(plain_model)
>>>
>>> # Wrap a HuggingFace model (alternative to HuggingFaceTransformer)
>>> from transformers import AutoModel
>>> hf = AutoModel.from_pretrained("bert-base-uncased")
>>> wrapped_hf = DLModuleWrapper.wrap(hf)
>>> wrapped_hf.save_pretrained("./my_bert")
"""

import torch
import torch.nn as nn
from typing import Any, Optional

from .utils import DLModule


class DLModuleWrapper(DLModule):
    """
    Promote any ``nn.Module`` to a full ``DLModule``.

    All ``DLModule`` capabilities become available on the wrapped object
    without requiring the original class to inherit from ``DLModule``.
    ``forward()`` delegates transparently to the inner module, so the
    wrapped object behaves identically to the original in all forward-pass
    contexts.

    Parameters
    ----------
    module : nn.Module
        The module to wrap.  Must already be an ``nn.Module`` instance.
    **kwargs
        Additional keyword arguments forwarded to ``DLModule.__init__``
        (e.g. ``device``, ``dtype``).

    Attributes
    ----------
    module : nn.Module
        The underlying wrapped module. Registered as a proper PyTorch
        submodule so ``parameters()``, ``state_dict()``, ``to()``, etc.
        all work correctly through the wrapper.

    Examples
    --------
    >>> import torch.nn as nn
    >>> from Code.models.utils import DLModuleWrapper
    >>>
    >>> net = nn.Linear(10, 2)
    >>> w = DLModuleWrapper.wrap(net)
    >>> x = torch.randn(4, 10)
    >>> w(x).shape
    torch.Size([4, 2])
    >>>
    >>> # Training
    >>> w.fit(train_dataset, epochs=5, batch_size=32, ...)
    >>>
    >>> # Fine-tuning with LoRA
    >>> w.fine_tune(train_dataset, fine_tune_type="lora", ...)
    >>>
    >>> # Save / load
    >>> w.save_pretrained("./checkpoints/my_model")
    """

    def __init__(self, module: nn.Module, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(module, nn.Module):
            raise TypeError(
                f"DLModuleWrapper requires an nn.Module instance, "
                f"got {type(module).__name__!r}."
            )
        # Assigning to self.module registers it as a proper PyTorch
        # submodule, ensuring parameters(), to(), state_dict(), etc.
        # all traverse into the wrapped model correctly.
        self.module = module

    # ------------------------------------------------------------------
    # Core delegation
    # ------------------------------------------------------------------

    def forward(self, *args, **kwargs) -> Any:
        """Delegate forward pass to the wrapped module."""
        return self.module(*args, **kwargs)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @classmethod
    def wrap(cls, module: nn.Module, **kwargs) -> "DLModuleWrapper":
        """
        One-step factory shorthand.

        Parameters
        ----------
        module : nn.Module
            The module to wrap.
        **kwargs
            Forwarded to ``DLModuleWrapper.__init__``.

        Returns
        -------
        DLModuleWrapper
        """
        return cls(module, **kwargs)

    def unwrap(self) -> nn.Module:
        """
        Return the underlying ``nn.Module`` without the wrapper.

        Returns
        -------
        nn.Module
        """
        return self.module

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  module={self.module!r}\n"
            f")"
        )

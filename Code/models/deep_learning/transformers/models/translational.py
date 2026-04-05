"""Translational LM placeholder."""

from __future__ import annotations

import torch

from ._base import BaseTransformerModel


class TranslationalLM(BaseTransformerModel):
    def __init__(self, device: str = "cpu", dtype: torch.dtype = torch.float32, **kwargs):
        super().__init__(device=device, dtype=dtype, **kwargs)

    def forward(self, x):
        raise NotImplementedError("TranslationalLM is not bundled in this checkout.")

"""Decoder LM placeholder (import-only; not used by thesis HRM)."""

from __future__ import annotations

import torch.nn as nn

from ._base import BaseTransformerModel


class DecoderLM(BaseTransformerModel):
    def __init__(self, device: str = "cpu", dtype=None, **kwargs):
        super().__init__(device=device, dtype=dtype or __import__("torch").float32, **kwargs)

    def forward(self, x):
        raise NotImplementedError("DecoderLM is not bundled in this checkout.")

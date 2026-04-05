"""Stacked transformer encoder used by HRM (EncoderLM)."""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any, Dict

from ._base import BaseTransformerModel
from ..attention.attention_units import GeneralAttentionBlock
from ..transformers.transformers import GeneralTransformer


class EncoderLM(BaseTransformerModel):
    """
    ``n_layers`` stacked pre-norm transformer blocks (attention + FFN).
    Built with explicit submodules so optional registry JSON files are not required.
    """

    def __init__(
        self,
        n_layers: int,
        model_config: Dict[str, Any],
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__(device=device, dtype=dtype, **kwargs)
        tcfg = model_config.get("transformer", model_config)
        att = tcfg["attention"]
        hidden = int(att["hidden_size"])
        heads = int(att["heads"])
        ctx = int(att["context_length"])
        self.layers = nn.ModuleList()
        for _ in range(int(n_layers)):
            attn_mod = GeneralAttentionBlock(
                input_size=hidden,
                hidden_size=hidden,
                context_length=ctx,
                heads=heads,
                causal=bool(att.get("causal", False)),
                multiheaded=bool(att.get("multiheaded", True)),
                device=device,
                dtype=dtype,
            )
            ff = nn.Sequential(
                nn.Linear(hidden, 4 * hidden, device=device, dtype=dtype),
                nn.GELU(),
                nn.Linear(4 * hidden, hidden, device=device, dtype=dtype),
            )
            self.layers.append(
                GeneralTransformer(
                    norm1_module=nn.LayerNorm(hidden, device=device, dtype=dtype),
                    attention_module=attn_mod,
                    dropout1_module=nn.Identity(),
                    norm2_module=nn.LayerNorm(hidden, device=device, dtype=dtype),
                    neural_network_module=ff,
                    dropout2_module=nn.Identity(),
                    module_config=None,
                    causal=False,
                    cross=False,
                    device=device,
                    dtype=dtype,
                )
            )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

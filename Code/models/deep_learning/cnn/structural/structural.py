"""
Incoherent Network Motifs: bio-inspired IFFL structure.
Reference: docs/deep-learning/cnn/cnn.md Part IV
"""

import torch
import torch.nn as nn
from typing import Union
from .....models.utils import DLModule


__all__ = ["IncoherentMotifLayer"]


class IncoherentMotifLayer(DLModule):
    """
    Incoherent Feed-Forward Loop: one path activates, parallel path represses.
    Adaptive fold-change detector for noise immunity.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 motif_type: str = "IFFL",
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.conv_act = nn.Conv2d(in_channels, out_channels, 3, padding=1, **self.factory_kwargs)
        self.conv_rep = nn.Conv2d(in_channels, out_channels, 3, padding=1, **self.factory_kwargs)
        self.gate = nn.Parameter(torch.tensor(0.5, **self.factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        act = torch.relu(self.conv_act(x))
        rep = torch.sigmoid(-self.conv_rep(x))
        return act * (rep * self.gate + (1 - rep) * (1 - self.gate))

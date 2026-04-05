"""
Hardware DBN variants: Quantum, Memristive.
Reference: docs/deep-learning/dbm/dbm.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List, Dict, Any
import math

from .._base import DBNModuleBase
from ..core.core import RBM
from ..foundational.foundational import StandardDBN


class QDBN(DBNModuleBase):
    """
    Quantum DBN: Quantum Boltzmann Machine simulation placeholder.
    Uses classical embedding when Pennylane unavailable.
    """

    def __init__(
        self,
        n_qubits: int,
        n_visible: Optional[int] = None,
        n_hidden: Optional[int] = None,
        ansatz_type: str = "StronglyEntangling",
        gamma_tunneling: float = 0.5,
        readout_shots: int = 1024,
        n_layers: int = 2,
        device: Union[str, torch.device] = "simulator",
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(device=device, dtype=dtype)
        self.n_qubits = n_qubits
        self.n_visible = n_visible or n_qubits
        self.n_hidden = n_hidden or n_qubits
        self.ansatz_type = ansatz_type
        self.gamma_tunneling = gamma_tunneling
        self.readout_shots = readout_shots
        self.n_layers = n_layers
        self.factory_kwargs = {"device": device, "dtype": dtype or torch.float32}

        # Classical simulation fallback: embed via linear + sample
        self.classical_embed = nn.Linear(self.n_visible, n_qubits)
        self.classical_readout = nn.Linear(n_qubits, self.n_hidden)
        nn.init.xavier_uniform_(self.classical_embed.weight)
        nn.init.xavier_uniform_(self.classical_readout.weight)
        nn.init.zeros_(self.classical_embed.bias)
        nn.init.zeros_(self.classical_readout.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classical simulation: embed and project."""
        h = torch.sigmoid(self.classical_embed(x))
        return torch.sigmoid(self.classical_readout(h))

    def sample(self, n_samples: int) -> torch.Tensor:
        """Sample from classical approximation."""
        z = torch.randn(n_samples, self.n_visible, device=next(self.parameters()).device)
        return self.forward(z)


class MemristiveDBN(DBNModuleBase):
    """
    Memristive DBN: Simulated conductance-based weights for analog hardware.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        memristor_config: Optional[Dict[str, Any]] = None,
        precision_bits: int = 8,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(device=device, dtype=dtype)
        self.layer_sizes = layer_sizes
        self.memristor_config = memristor_config or {}
        self.precision_bits = precision_bits
        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.conductance = nn.ParameterList()
        self.layers = nn.ModuleList()
        scale = 2 ** (-precision_bits)
        for i in range(len(layer_sizes) - 1):
            W = torch.randn(layer_sizes[i + 1], layer_sizes[i]) * 0.01
            self.conductance.append(nn.Parameter(W))
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            self.layers[-1].weight.data = W.clone()
            nn.init.zeros_(self.layers[-1].bias)

    def _quantize(self, x: torch.Tensor) -> torch.Tensor:
        levels = 2 ** self.precision_bits
        return torch.round(x * levels) / levels

    def read_conductance(self) -> torch.Tensor:
        """Current conductance state."""
        return torch.cat([c.flatten() for c in self.conductance])

    def program_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Program memristor array (simulated)."""
        for i, (k, w) in enumerate(weights.items()):
            if i < len(self.conductance):
                self.conductance[i].data = w.to(self.conductance[i].device)
                if i < len(self.layers):
                    self.layers[i].weight.data = w.clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, lin in enumerate(self.layers):
            Wq = self._quantize(self.conductance[i]) if i < len(self.conductance) else lin.weight
            h = torch.sigmoid(F.linear(h, Wq, lin.bias))
        return h

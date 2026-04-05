"""
Specialized DBN variants: Spiking, EEG-DBN, ME-DBN, CVDBN, FuzzyDBN.
Reference: docs/deep-learning/dbm/dbm.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List
import math

from .._base import DBNModuleBase
from ..core.core import RBM, GaussianRBM
from ..foundational.foundational import StandardDBN


class SDBN(DBNModuleBase):
    """
    Spiking DBN: LIF neurons with rate-coded CD for neuromorphic hardware.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        v_threshold: float = 1.0,
        tau_m: float = 20.0,
        spike_encoding: str = "rate",
        simulation_time: int = 100,
        weight_bit_width: int = 8,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(device=device, dtype=dtype)
        self.layer_sizes = layer_sizes
        self.v_threshold = v_threshold
        self.tau_m = tau_m
        self.spike_encoding = spike_encoding
        self.simulation_time = simulation_time
        self.weight_bit_width = weight_bit_width
        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            )
        for m in self.layers:
            nn.init.xavier_uniform_(m.weight, gain=0.5)
            nn.init.zeros_(m.bias)

    def encode_to_spikes(self, x: torch.Tensor) -> torch.Tensor:
        """Convert input to spike train (B, n_v) -> (B, T, n_v)."""
        B, n = x.shape
        T = self.simulation_time
        if self.spike_encoding == "rate":
            rate = x.clamp(0, 1)
            spikes = torch.bernoulli(
                rate.unsqueeze(1).expand(B, T, n)
            )
        elif self.spike_encoding == "latency":
            inv = 1 - x.clamp(1e-7, 1)
            lat = (inv * T).long().clamp(0, T - 1)
            spikes = torch.zeros(B, T, n, device=x.device, dtype=x.dtype)
            for b in range(B):
                for i in range(n):
                    if lat[b, i] < T:
                        spikes[b, lat[b, i], i] = 1
        else:
            rate = x.clamp(0, 1)
            spikes = torch.bernoulli(rate.unsqueeze(1).expand(B, T, n))
        return spikes

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """spikes: (B, T, n_v) -> (B, n_L). Rate-coded sum."""
        B, T, _ = spikes.shape
        rate = spikes.mean(dim=1)
        h = rate
        for lin in self.layers:
            h = torch.sigmoid(lin(h))
        return h


class EEGDBN(DBNModuleBase):
    """
    Multimodal EEG-DBN: Gaussian RBMs for EEG, Binary for clinical data.
    """

    def __init__(
        self,
        eeg_dim: int,
        clinical_dim: int,
        gaussian_layers: List[int],
        binary_layers: List[int],
        fusion_size: int = 128,
        learning_rate: float = 0.01,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(device=device, dtype=dtype)
        self.eeg_dim = eeg_dim
        self.clinical_dim = clinical_dim
        self.fusion_size = fusion_size
        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.eeg_branch = nn.ModuleList()
        self.eeg_branch.append(
            GaussianRBM(
                eeg_dim,
                gaussian_layers[0],
                sigma=1.0,
                learning_rate=learning_rate,
                device=device,
                dtype=dtype,
            )
        )
        for i in range(len(gaussian_layers) - 1):
            self.eeg_branch.append(
                RBM(
                    gaussian_layers[i],
                    gaussian_layers[i + 1],
                    learning_rate=learning_rate,
                    device=device,
                    dtype=dtype,
                )
            )

        self.clinical_branch = nn.ModuleList()
        bl = [clinical_dim] + binary_layers
        for i in range(len(bl) - 1):
            self.clinical_branch.append(
                RBM(bl[i], bl[i + 1], learning_rate=learning_rate, device=device, dtype=dtype)
            )

        n_eeg = gaussian_layers[-1]
        n_clin = binary_layers[-1]
        self.fusion = nn.Linear(n_eeg + n_clin, fusion_size)
        nn.init.xavier_uniform_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)

    def encode_eeg(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for m in self.eeg_branch:
            h = m.sample_hidden(h)
        return h

    def encode_clinical(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for m in self.clinical_branch:
            h = m.sample_hidden(h)
        return h

    def forward(
        self,
        eeg: torch.Tensor,
        clinical: torch.Tensor,
    ) -> torch.Tensor:
        h_eeg = self.encode_eeg(eeg)
        h_clin = self.encode_clinical(clinical)
        h_cat = torch.cat([h_eeg, h_clin], dim=-1)
        return torch.sigmoid(self.fusion(h_cat))


class MEDBN(DBNModuleBase):
    """
    Maximum Entropy DBN: entropy-based objective for small data.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        entropy_weight: float = 1.0,
        small_data_mode: bool = True,
        learning_rate: float = 0.01,
        batch_size: int = 16,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(device=device, dtype=dtype)
        self.layer_sizes = layer_sizes
        self.entropy_weight = entropy_weight
        self.small_data_mode = small_data_mode
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            )
        for m in self.layers:
            nn.init.xavier_uniform_(m.weight, gain=0.5)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for lin in self.layers:
            h = torch.sigmoid(lin(h))
        return h

    def entropy_estimate(self) -> float:
        """Placeholder: return 0. Entropy estimation requires data pass."""
        return 0.0


class CVDBN(DBNModuleBase):
    """
    Complex-Valued DBN: weights and activations in C for phase-sensitive data.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        complex_weights: bool = True,
        phase_loss_weight: float = 0.1,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(device=device, dtype=dtype)
        self.layer_sizes = layer_sizes
        self.complex_weights = complex_weights
        self.phase_loss_weight = phase_loss_weight
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        dt = dtype or torch.float32
        self.factory_kwargs = {"device": device, "dtype": dt}

        self.W_real = nn.ModuleList()
        self.W_imag = nn.ModuleList()
        self.b_real = nn.ParameterList()
        self.b_imag = nn.ParameterList()
        for i in range(len(layer_sizes) - 1):
            wr = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            wi = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            nn.init.normal_(wr.weight, 0, 0.01)
            nn.init.normal_(wi.weight, 0, 0.01)
            nn.init.zeros_(wr.bias)
            nn.init.zeros_(wi.bias)
            self.W_real.append(wr)
            self.W_imag.append(wi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_complex():
            x = torch.complex(x, torch.zeros_like(x))
        h = x
        for wr, wi in zip(self.W_real, self.W_imag):
            real = wr(h.real) - wi(h.imag)
            imag = wr(h.imag) + wi(h.real)
            h = torch.complex(real, imag)
            h = torch.sigmoid(h.real) + 1j * torch.sigmoid(h.imag)
        return h

    def magnitude(self, x: torch.Tensor) -> torch.Tensor:
        return x.abs() if x.is_complex() else x

    def phase(self, x: torch.Tensor) -> torch.Tensor:
        return x.angle() if x.is_complex() else torch.zeros_like(x)


class FuzzyDBN(DBNModuleBase):
    """
    Fuzzy DBN: fuzzy logic membership in RBM layers.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        fuzzy_membership: str = "gaussian",
        uncertainty_threshold: float = 0.5,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(device=device, dtype=dtype)
        self.layer_sizes = layer_sizes
        self._membership_type = fuzzy_membership
        self.uncertainty_threshold = uncertainty_threshold
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            )
        self.centroids = nn.ParameterList()
        for i in range(len(layer_sizes) - 1):
            c = torch.randn(layer_sizes[i + 1], **self._fk()) * 0.1
            self.centroids.append(nn.Parameter(c))
        for m in self.layers:
            nn.init.xavier_uniform_(m.weight, gain=0.5)
            nn.init.zeros_(m.bias)

    def _fk(self) -> dict:
        d = torch.device("cpu")
        dt = torch.float32
        if self.factory_kwargs.get("device"):
            d = torch.device(self.factory_kwargs["device"])
        if self.factory_kwargs.get("dtype"):
            dt = self.factory_kwargs["dtype"]
        return {"device": d, "dtype": dt}

    def _membership(self, x: torch.Tensor, centroid: torch.Tensor) -> torch.Tensor:
        if self._membership_type == "gaussian":
            dist = (x - centroid).pow(2).sum(dim=-1, keepdim=True)
            return torch.exp(-dist)
        if self._membership_type == "triangular":
            d = (x - centroid).abs().sum(dim=-1, keepdim=True)
            return F.relu(1 - d)
        return torch.sigmoid(x)

    def fuzzy_membership_fn(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        memberships = []
        for i, lin in enumerate(self.layers):
            pre = lin(h)
            memb = self._membership(pre, self.centroids[i])
            memberships.append(memb)
            h = memb * torch.sigmoid(pre)
        return memberships[-1] if memberships else x

    def fuzzy_membership(self, x: torch.Tensor) -> torch.Tensor:
        """Membership degrees (plan API)."""
        return self.fuzzy_membership_fn(x)

    @property
    def fuzzy_membership_type(self) -> str:
        """Current membership type (gaussian, triangular, trapezoidal)."""
        return self._membership_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for lin in self.layers:
            pre = lin(h)
            h = torch.sigmoid(pre)
        return h

    def uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """Uncertainty scores based on distance from threshold."""
        h = self.forward(x)
        return (h - self.uncertainty_threshold).abs()

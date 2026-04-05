"""
Hybrid DBN variants (2025-2026): HTST, MSE, ODBN, PSO, OAFDBN, etc.
Reference: docs/deep-learning/dbm/dbm.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Union, List, Dict, Any, Tuple
import math
import copy

from .._base import DBNModuleBase
from ..core.core import RBM
from ..foundational.foundational import StandardDBN, GaussianDBN
from ..structural.structural import CDBN


class HTSTDBN(DBNModuleBase):
    """Hybrid Time Series Transformer-DBN for sequence + hierarchical abstraction."""

    def __init__(
        self,
        in_features: int,
        seq_len: int,
        attention_heads: int = 4,
        d_model: int = 64,
        dbn_hierarchy: List[int] = None,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(device=device, dtype=dtype)
        self.attention_heads = attention_heads
        self.d_model = d_model
        self.seq_len = seq_len
        self.dbn_hierarchy = dbn_hierarchy or [256, 128, 64]
        self.dropout_val = dropout
        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.in_proj = nn.Linear(in_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=attention_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        dbn_in = d_model
        self.dbn_stack = StandardDBN(
            layer_sizes=[dbn_in] + dbn_hierarchy,
            learning_rate=learning_rate,
            device=device,
            dtype=dtype,
        )
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_in) -> (B, n_L)."""
        x = self.in_proj(x)
        z = self.transformer(x)
        z = z.transpose(1, 2)
        z = self.pool(z).squeeze(-1)
        return self.dbn_stack.encode(z)


class MSEDBN(DBNModuleBase):
    """Multi-Stage Enhanced DBN with Mixstyle for domain generalization."""

    def __init__(
        self,
        layer_sizes: List[int],
        mixstyle_lambda: float = 0.5,
        attention_type: str = "additive",
        zero_day_detection: bool = True,
        learning_rate: float = 0.01,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(device=device, dtype=dtype)
        self.layer_sizes = layer_sizes
        self.mixstyle_lambda = mixstyle_lambda
        self.attention_type = attention_type
        self.zero_day_detection = zero_day_detection
        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.dbn_stack = StandardDBN(
            layer_sizes=layer_sizes,
            learning_rate=learning_rate,
            device=device,
            dtype=dtype,
        )
        self.attn_proj = nn.Linear(layer_sizes[-1], layer_sizes[-1])
        nn.init.xavier_uniform_(self.attn_proj.weight)
        nn.init.zeros_(self.attn_proj.bias)

    def _mixstyle(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.mixstyle_lambda <= 0:
            return x
        B = x.size(0)
        perm = torch.randperm(B, device=x.device)
        mu = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True) + 1e-6
        mu2 = x[perm].mean(dim=0, keepdim=True)
        std2 = x[perm].std(dim=0, keepdim=True) + 1e-6
        lam = self.mixstyle_lambda
        new_mu = lam * mu + (1 - lam) * mu2
        new_std = lam * std + (1 - lam) * std2
        return (x - mu) / std * new_std + new_mu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._mixstyle(x)
        h = self.dbn_stack.encode(x)
        if self.attention_type == "additive":
            h = h + torch.tanh(self.attn_proj(h))
        return h


class ODBNFDFTS(DBNModuleBase):
    """Optimization-Based DBN with ARO/BOA for fraud detection."""

    def __init__(
        self,
        layer_sizes: List[int],
        aro_population: int = 30,
        boa_iterations: int = 100,
        fraud_detection_mode: bool = True,
        learning_rate: float = 0.01,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(device=device, dtype=dtype)
        self.layer_sizes = layer_sizes
        self.aro_population = aro_population
        self.boa_iterations = boa_iterations
        self.fraud_detection_mode = fraud_detection_mode
        self.selected_features = list(range(layer_sizes[0]))
        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.dbn_stack = StandardDBN(
            layer_sizes=layer_sizes,
            learning_rate=learning_rate,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.selected_features:
            x = x[..., self.selected_features]
        return self.dbn_stack.encode(x)

    def optimize_structure(self, data: torch.Tensor) -> Dict[str, Any]:
        """Placeholder: ARO/BOA would select features."""
        return {"selected_features": self.selected_features}


class PSODBN(DBNModuleBase):
    """Particle Swarm Optimized DBN for structure search."""

    def __init__(
        self,
        n_visible: int,
        swarm_size: int = 20,
        n_iterations: int = 50,
        structure_bounds: tuple = (1, 5, 32, 512),
        learning_rate: float = 0.01,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(device=device, dtype=dtype)
        self.n_visible = n_visible
        self.swarm_size = swarm_size
        self.n_iterations = n_iterations
        self.structure_bounds = structure_bounds
        self.best_structure = {"layer_sizes": [n_visible, 256, 128]}
        self.factory_kwargs = {"device": device, "dtype": dtype}

        layer_sizes = self.best_structure["layer_sizes"]
        self.dbn_stack = StandardDBN(
            layer_sizes=layer_sizes,
            learning_rate=learning_rate,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dbn_stack.encode(x)

    def optimize_structure(self, data: torch.Tensor) -> StandardDBN:
        """Placeholder: PSO would search layer_sizes."""
        return self.dbn_stack


class OAFDBN(DBNModuleBase):
    """Online Adaptive Fine-Tuning DBN for concept drift."""

    def __init__(
        self,
        layer_sizes: List[int],
        adaptation_rate: float = 0.01,
        drift_detection_threshold: float = 0.1,
        learning_rate: float = 0.01,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(device=device, dtype=dtype)
        self.layer_sizes = layer_sizes
        self.adaptation_rate = adaptation_rate
        self.drift_detection_threshold = drift_detection_threshold
        self.drift_detected = False
        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.dbn_stack = StandardDBN(
            layer_sizes=layer_sizes,
            learning_rate=learning_rate,
            device=device,
            dtype=dtype,
        )
        self._recent_loss = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dbn_stack.encode(x)

    def adapt_online(self, x: torch.Tensor) -> None:
        """Online weight update."""
        self.train()
        h = self.dbn_stack.encode(x)
        recon = self.dbn_stack.reconstruct(x, n_steps=1)
        loss = F.mse_loss(x, recon)
        if loss.item() - self._recent_loss > self.drift_detection_threshold:
            self.drift_detected = True
        self._recent_loss = loss.item()
        loss.backward()
        for p in self.parameters():
            if p.grad is not None:
                p.data.add_(p.grad, alpha=-self.adaptation_rate)
        self.zero_grad()


class DiffusionDBN(DBNModuleBase):
    """DBN as structured latent prior for Diffusion models."""

    def __init__(
        self,
        layer_sizes: List[int],
        prior_strength: float = 1.0,
        diffusion_steps: int = 1000,
        latent_dim: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(device=device, dtype=dtype)
        ld = latent_dim or layer_sizes[-1]
        self.layer_sizes = layer_sizes
        self.prior_strength = prior_strength
        self.diffusion_steps = diffusion_steps
        self.latent_dim = ld
        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.dbn_stack = StandardDBN(
            layer_sizes=layer_sizes,
            device=device,
            dtype=dtype,
        )
        self.proj = nn.Linear(ld, layer_sizes[0])
        self.inv_proj = nn.Linear(layer_sizes[-1], ld)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.inv_proj.weight)

    def get_prior(self, z: torch.Tensor) -> torch.Tensor:
        x = self.proj(z)
        h = self.dbn_stack.encode(x)
        return self.prior_strength * self.inv_proj(h)

    def sample_prior(self, n_samples: int) -> torch.Tensor:
        v = self.dbn_stack.sample(n_samples, n_steps=10)
        h = self.dbn_stack.encode(v)
        return self.inv_proj(h)


class IDBN(DBNModuleBase):
    """Improved DBN for cyber-physical intrusion detection."""

    def __init__(
        self,
        layer_sizes: List[int],
        learning_rate_boost: float = 2.0,
        intrusion_detection_mode: bool = True,
        batch_size: int = 32,
        n_epochs: int = 50,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(device=device, dtype=dtype)
        self.layer_sizes = layer_sizes
        self.learning_rate_boost = learning_rate_boost
        self.intrusion_detection_mode = intrusion_detection_mode
        lr = 0.01 * learning_rate_boost
        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.dbn_stack = StandardDBN(
            layer_sizes=layer_sizes,
            learning_rate=lr,
            batch_size=batch_size,
            n_epochs=n_epochs,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dbn_stack.encode(x)

    def detect_anomaly(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        recon = self.dbn_stack.reconstruct(x, n_steps=1)
        err = (x - recon).pow(2).mean(dim=-1)
        return err > threshold


class SSDBN(DBNModuleBase):
    """Semi-Supervised DBN with labeled guidance during pre-training."""

    def __init__(
        self,
        layer_sizes: List[int],
        n_classes: int,
        labeled_ratio: float = 0.1,
        supervised_weight: float = 0.5,
        learning_rate: float = 0.01,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(device=device, dtype=dtype)
        self.layer_sizes = layer_sizes
        self.labeled_ratio = labeled_ratio
        self.supervised_weight = supervised_weight
        self.n_classes = n_classes
        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.dbn_stack = StandardDBN(
            layer_sizes=layer_sizes,
            learning_rate=learning_rate,
            device=device,
            dtype=dtype,
        )
        self.classifier = nn.Linear(layer_sizes[-1], n_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dbn_stack.encode(x)

    def pretrain_semi(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        """Semi-supervised pre-train: mask indicates labeled samples."""
        self.dbn_stack.pretrain(x)
        labeled = mask.bool()
        if labeled.sum() > 0:
            h = self.dbn_stack.encode(x[labeled])
            logits = self.classifier(h)
            loss = F.cross_entropy(logits, labels[labeled].long())
            loss.backward()


class SpikingCDBN(DBNModuleBase):
    """Convolutional + Spiking DBN for neuromorphic vision."""

    def __init__(
        self,
        in_channels: int,
        filters: List[int],
        kernel_size: Union[int, tuple] = 3,
        v_threshold: float = 1.0,
        tau_m: float = 20.0,
        spike_encoding: str = "rate",
        simulation_time: int = 100,
        weight_bit_width: int = 8,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(device=device, dtype=dtype)
        self.v_threshold = v_threshold
        self.tau_m = tau_m
        self.spike_encoding = spike_encoding
        self.simulation_time = simulation_time
        self.weight_bit_width = weight_bit_width
        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.conv = CDBN(
            in_channels=in_channels,
            filters=filters,
            kernel_size=kernel_size,
            device=device,
            dtype=dtype,
        )
        flat_dim = self._flat_dim(in_channels, 32, 32, filters)
        self.spike_layers = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.Sigmoid(),
            nn.Linear(256, 128),
        )
        nn.init.xavier_uniform_(self.spike_layers[0].weight)
        nn.init.xavier_uniform_(self.spike_layers[2].weight)

    def _flat_dim(self, c, h, w, filters):
        for _ in filters:
            h = (h - 2) // 2 + 1
            w = (w - 2) // 2 + 1
        return filters[-1] * h * w

    def encode_to_spikes(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.flatten(1)
        h = self.spike_layers(h)
        rate = torch.sigmoid(h)
        B, n = rate.shape
        return torch.bernoulli(rate.unsqueeze(1).expand(B, self.simulation_time, n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.flatten(1)
        return self.spike_layers(h)


class GANDBN(DBNModuleBase):
    """GAN with DBN as Discriminator."""

    def __init__(
        self,
        layer_sizes: List[int],
        generator_config: Optional[Dict] = None,
        discriminator_dbn_layers: Optional[List[int]] = None,
        latent_dim: int = 64,
        learning_rate: float = 0.0002,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(device=device, dtype=dtype)
        d_layers = discriminator_dbn_layers or layer_sizes
        self.layer_sizes = d_layers
        self.latent_dim = latent_dim
        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.discriminator_dbn = StandardDBN(
            layer_sizes=d_layers,
            learning_rate=learning_rate,
            device=device,
            dtype=dtype,
        )
        self.disc_head = nn.Linear(d_layers[-1], 1)
        nn.init.xavier_uniform_(self.disc_head.weight)
        nn.init.zeros_(self.disc_head.bias)

        n_v = d_layers[0]
        self.generator_net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, n_v),
            nn.Sigmoid(),
        )
        for m in self.generator_net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def discriminator(self, x: torch.Tensor) -> torch.Tensor:
        h = self.discriminator_dbn.encode(x)
        return self.disc_head(h)

    def generator(self, z: torch.Tensor) -> torch.Tensor:
        """Generate samples from latent z."""
        return self.generator_net(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)


class EEGBCIDBN(DBNModuleBase):
    """DBN for EEG/BCI with non-stationary regularization."""

    def __init__(
        self,
        eeg_channels: int,
        trial_length: int,
        layer_sizes: List[int],
        regularization: str = "domain_adapt",
        learning_rate: float = 0.01,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(device=device, dtype=dtype)
        self.eeg_channels = eeg_channels
        self.trial_length = trial_length
        self.regularization = regularization
        in_dim = eeg_channels * trial_length
        if layer_sizes[0] != in_dim:
            layer_sizes = [in_dim] + layer_sizes[1:]
        self.layer_sizes = layer_sizes
        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.dbn_stack = StandardDBN(
            layer_sizes=layer_sizes,
            learning_rate=learning_rate,
            device=device,
            dtype=dtype,
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """eeg: (B, channels, length) -> (B, n_L)."""
        B, C, L = eeg.shape
        x = eeg.view(B, -1)
        return self.dbn_stack.encode(x)


class EnsembleDBN(DBNModuleBase):
    """Ensemble of DBNs with vote/average/stacked aggregation."""

    def __init__(
        self,
        n_models: int = 10,
        layer_sizes: List[int] = None,
        bootstrap_ratio: float = 0.8,
        aggregation: str = "vote",
        learning_rate: float = 0.01,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(device=device, dtype=dtype)
        if layer_sizes is None:
            layer_sizes = [784, 500, 200]
        self.n_models = n_models
        self.layer_sizes = layer_sizes
        self.bootstrap_ratio = bootstrap_ratio
        self.aggregation = aggregation
        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.ensemble = nn.ModuleList()
        for _ in range(n_models):
            dbn = StandardDBN(
                layer_sizes=layer_sizes,
                learning_rate=learning_rate,
                device=device,
                dtype=dtype,
            )
            self.ensemble.append(dbn)

        if aggregation == "stacked":
            self.stacker = nn.Linear(layer_sizes[-1] * n_models, layer_sizes[-1])
            nn.init.xavier_uniform_(self.stacker.weight)
            nn.init.zeros_(self.stacker.bias)
        else:
            self.stacker = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [dbn.encode(x) for dbn in self.ensemble]
        if self.aggregation == "average":
            return torch.stack(outs, dim=0).mean(dim=0)
        if self.aggregation == "vote":
            return torch.stack(outs, dim=0).mean(dim=0)
        if self.aggregation == "stacked" and self.stacker is not None:
            h_cat = torch.cat(outs, dim=-1)
            return torch.sigmoid(self.stacker(h_cat))
        return outs[0]

"""
100-D feature encoders + MSE reconstruction decoder for merged transformed all-data pretrain.

Total trainable params target: 50k–80k (encoder + decoder). Checkpoints export **encoder** only.
Finetune: ``FeatureEncoderClassifier`` = loaded encoder + Linear(100, n_classes).
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

ArchName = Literal["ffnn", "cnn", "lstm", "gru", "rnn"]

LATENT_DIM = 100


def _mlp_decoder(in_dim: int = LATENT_DIM, hidden: int = 96, out_dim: int = 100) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, out_dim),
    )


class _FFNNEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        h1, h2 = 160, 140
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(inplace=True),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _CNNEncoder(nn.Module):
    """Input x: [B, feat_dim] -> treat as [B, 3, L] with channel repeat."""

    def __init__(self, input_dim: int, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.body = nn.Sequential(
            nn.Conv1d(3, 36, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.Conv1d(36, 56, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(5),
        )
        with torch.no_grad():
            t = torch.zeros(1, 3, input_dim)
            t = self.body(t)
            flat = t.numel()
        mid = max(flat // 3, 96)
        self.proj = nn.Sequential(
            nn.Linear(flat, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError("expected [B, feat_dim]")
        x = x.unsqueeze(-1).expand(-1, -1, 3).transpose(1, 2).contiguous()
        z = self.body(x).flatten(1)
        return self.proj(z)  # type: ignore[operator]


class _SeqEncoder(nn.Module):
    """Shared pattern: [B, seq, 1] through RNN family -> last hidden -> Linear -> latent."""

    def __init__(
        self,
        arch: ArchName,
        seq_len: int,
        latent_dim: int = LATENT_DIM,
        hidden_size: int = 88,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        if arch == "lstm":
            self.rnn = nn.LSTM(
                1, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False
            )
        elif arch == "gru":
            self.rnn = nn.GRU(
                1, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False
            )
        elif arch == "rnn":
            self.rnn = nn.RNN(
                1,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=False,
                nonlinearity="tanh",
            )
        else:
            raise ValueError(arch)
        self.proj = nn.Linear(hidden_size, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        out, h_n = self.rnn(x)
        if isinstance(h_n, tuple):
            last = h_n[0][-1]
        else:
            last = h_n[-1]
        return self.proj(last)


class FeaturePretrainAutoencoder(nn.Module):
    """
    Encoder -> 100D, decoder -> reconstruct input_dim (MSE). Export encoder weights for finetune.
    """

    def __init__(
        self,
        architecture: str,
        input_dim: int = 100,
        latent_dim: int = LATENT_DIM,
    ) -> None:
        super().__init__()
        arch = architecture.lower().strip()
        if arch not in ("ffnn", "cnn", "lstm", "gru", "rnn"):
            raise ValueError(f"unknown architecture: {architecture}")
        self.architecture = arch
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)

        if arch == "ffnn":
            self.encoder = _FFNNEncoder(self.input_dim, self.latent_dim)
        elif arch == "cnn":
            self.encoder = _CNNEncoder(self.input_dim, self.latent_dim)
        else:
            if arch == "lstm":
                hs = 88
            elif arch == "gru":
                hs = 96
            else:
                hs = 168
            self.encoder = _SeqEncoder(
                arch, seq_len=self.input_dim, latent_dim=self.latent_dim, hidden_size=hs
            )

        self.decoder = _mlp_decoder(self.latent_dim, hidden=96, out_dim=self.input_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon

    def encoder_state_dict(self) -> dict[str, torch.Tensor]:
        return {k.replace("encoder.", "", 1): v for k, v in self.state_dict().items() if k.startswith("encoder.")}

    def encoder_checkpoint_tensors(self) -> dict[str, torch.Tensor]:
        """Keys ``encoder.*`` for safetensors (matches ``FeatureEncoderClassifier.load_encoder_from_safetensors``)."""
        return {f"encoder.{k}": v.detach().cpu().contiguous() for k, v in self.encoder.state_dict().items()}


class FeatureEncoderClassifier(nn.Module):
    """Load encoder from pretrain checkpoint + classification head (finetune)."""

    def __init__(
        self,
        architecture: str,
        n_classes: int,
        input_dim: int = 100,
        latent_dim: int = LATENT_DIM,
    ) -> None:
        super().__init__()
        self.architecture = architecture.lower().strip()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.n_classes = int(n_classes)

        if self.architecture == "ffnn":
            self.encoder = _FFNNEncoder(self.input_dim, self.latent_dim)
        elif self.architecture == "cnn":
            self.encoder = _CNNEncoder(self.input_dim, self.latent_dim)
        else:
            if self.architecture == "lstm":
                hs = 88
            elif self.architecture == "gru":
                hs = 96
            else:
                hs = 168
            self.encoder = _SeqEncoder(
                self.architecture,
                seq_len=self.input_dim,
                latent_dim=self.latent_dim,
                hidden_size=hs,
            )
        self.head = nn.Linear(self.latent_dim, self.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.head(z)

    def load_encoder_from_safetensors(self, path: str, map_location: str | torch.device = "cpu") -> None:
        from Code.thesis.common.checkpoint_io import load_safetensors_state

        blob = load_safetensors_state(path, map_location=map_location)
        enc_sd = {}
        for k, v in blob.items():
            if k.startswith("encoder."):
                enc_sd[k[len("encoder.") :]] = v
        self.encoder.load_state_dict(enc_sd, strict=True)

"""
B11-style stack: frozen pretrained **CNN** encoder (100→100) then frozen **LSTM** encoder (100→100),
then trainable ``100 -> hidden -> GELU -> K``.

Pretrain weights: ``checkpoints/pretrain/pretrain_cnn[_3labels].safetensors`` and
``pretrain_lstm[_3labels].safetensors`` (``encoder.*`` keys only).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from Code.thesis.common.checkpoint_io import load_safetensors_state
from Code.thesis.common.feature_pretrain_models import FeaturePretrainAutoencoder


def _encoder_from_pretrain_safetensors(
    architecture: str,
    ckpt: Path,
    *,
    input_dim: int = 100,
    latent_dim: int = 100,
) -> nn.Module:
    ae = FeaturePretrainAutoencoder(architecture, input_dim=input_dim, latent_dim=latent_dim)
    blob = load_safetensors_state(ckpt, map_location="cpu")
    enc_sd = {k[len("encoder.") :]: v for k, v in blob.items() if k.startswith("encoder.")}
    ae.encoder.load_state_dict(enc_sd, strict=True)
    enc = ae.encoder
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc


class FrozenCNNLSTMStackGeLUHeadClassifier(nn.Module):
    """x [B,100] → CNN_enc → [B,100] → LSTM_enc → [B,100] → GeLU MLP → logits [B,K]."""

    latent_dim: int = 100

    def __init__(
        self,
        n_classes: int,
        pretrain_cnn: Path,
        pretrain_lstm: Path,
        *,
        hidden_dim: int = 400,
        input_dim: int = 100,
    ) -> None:
        super().__init__()
        self.n_classes = int(n_classes)
        self.input_dim = int(input_dim)

        self.encoder_cnn = _encoder_from_pretrain_safetensors(
            "cnn", pretrain_cnn, input_dim=self.input_dim, latent_dim=self.latent_dim
        )
        self.encoder_lstm = _encoder_from_pretrain_safetensors(
            "lstm", pretrain_lstm, input_dim=self.input_dim, latent_dim=self.latent_dim
        )

        self.head = nn.Sequential(
            nn.Linear(self.latent_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder_cnn(x)
        z = self.encoder_lstm(z)
        return self.head(z)

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v for k, v in self.state_dict().items() if k.startswith("head.")}

    def stacked_export_state_dict(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for k, v in self.encoder_cnn.state_dict().items():
            out[f"encoder_cnn.{k}"] = v.detach().cpu().contiguous()
        for k, v in self.encoder_lstm.state_dict().items():
            out[f"encoder_lstm.{k}"] = v.detach().cpu().contiguous()
        for k, v in self.head.state_dict().items():
            out[f"head.{k}"] = v.detach().cpu().contiguous()
        return out

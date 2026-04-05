"""
MoE facades: (1) DistilBERT text gate + experts, or (2) feature-only gate (no transformer in forward).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
from Code.models.deep_learning.llm.llm_models import LLMModule
from Code.models.moe.gating import GatingNetwork


class FeatureGatedMoE(nn.Module):
    """
    Routes with a trainable gate on ``feats`` [B, D] only. No LLM, no tokenizer, no HRM.
    Experts may be dense (``feats``) or text; typically all ``dense`` when using this class.
    """

    def __init__(
        self,
        n_classes: int,
        expert_modules: nn.ModuleList,
        feat_dim: int,
        sparse_top_k: Optional[int] = 2,
        gate_hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.experts = expert_modules
        self.feat_dim = int(feat_dim)
        self.gate = GatingNetwork(
            in_features=self.feat_dim,
            num_experts=len(expert_modules),
            hidden_dim=gate_hidden_dim,
            sparse_top_k=sparse_top_k,
        )

    def forward(
        self,
        texts: Sequence[str],
        feats: torch.Tensor,
        expert_modalities: Optional[Sequence[str]] = None,
    ) -> torch.Tensor:
        device = feats.device
        w = self.gate(feats.to(dtype=next(self.gate.parameters()).dtype))
        modes = expert_modalities or ["dense"] * len(self.experts)
        outs = []
        for ex, mode in zip(self.experts, modes):
            if mode == "dense":
                o = ex(feats)
            else:
                o = ex(list(texts), return_type="logits")
            if o.dim() == 1:
                o = o.unsqueeze(-1)
            outs.append(o)
        stacked = torch.stack(outs, dim=1)
        if stacked.size(-1) != self.n_classes:
            raise RuntimeError(
                f"Expert logits dim {stacked.size(-1)} != n_classes={self.n_classes}"
            )
        mixed = (stacked * w.unsqueeze(-1)).sum(dim=1)
        return mixed


class HeterogeneousMoE(nn.Module):
    """
    Expects parallel batches: ``texts`` (for text experts + DistilBERT gate) and ``feats`` [B,D] for dense experts.
    Each expert must return logits [B, n_classes]. Use :class:`FeatureGatedMoE` to avoid any transformer in the graph.
    """

    def __init__(
        self,
        n_classes: int,
        expert_modules: nn.ModuleList,
        sparse_top_k: Optional[int] = 2,
        gate_hidden_dim: Optional[int] = 256,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.experts = expert_modules
        self.gate_encoder = LLMModule(
            model_name="distilbert-base-uncased",
            tokenizer_name="distilbert-base-uncased",
            n_classes=n_classes,
            single_linear_head=True,
            embed_dim=768,
            checkpoint_dir=str(Path(__file__).resolve().parents[3] / "checkpoints" / "deep_learning" / "llm"),
            device="cpu",
        )
        for p in self.gate_encoder.parameters():
            p.requires_grad = False
        self.gate_encoder.eval()
        self.gate = GatingNetwork(
            in_features=768,
            num_experts=len(expert_modules),
            hidden_dim=gate_hidden_dim,
            sparse_top_k=sparse_top_k,
        )

    def forward(
        self,
        texts: Sequence[str],
        feats: torch.Tensor,
        expert_modalities: Optional[Sequence[str]] = None,
    ) -> torch.Tensor:
        """
        expert_modalities: same length as experts, each 'text' or 'dense'.
        """
        device = feats.device
        self.gate_encoder.to(device)
        with torch.no_grad():
            z = self.gate_encoder.get_embeddings(list(texts), pooling_strategy="mean", layer_strategy="last")
        z = z.to(device=device, dtype=feats.dtype)
        w = self.gate(z)
        modes = expert_modalities or ["text"] * len(self.experts)
        outs = []
        for ex, mode in zip(self.experts, modes):
            if mode == "dense":
                o = ex(feats)
            else:
                o = ex(list(texts), return_type="logits")
            if o.dim() == 1:
                o = o.unsqueeze(-1)
            outs.append(o)
        stacked = torch.stack(outs, dim=1)
        if stacked.size(-1) != self.n_classes:
            raise RuntimeError(
                f"Expert logits dim {stacked.size(-1)} != n_classes={self.n_classes}"
            )
        mixed = (stacked * w.unsqueeze(-1)).sum(dim=1)
        return mixed

"""
Unit tests for Transformer architectures.

Tests exercise DecoderLM, EncoderLM, and ViT with real embedding modules
injected so the pipeline stages receive float tensors.
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from .....models.deep_learning.transformers.models import (
    DecoderLM,
    EncoderLM,
    ViT,
)

# ─── Shared constants ──────────────────────────────────────────────────────────
VOCAB_SIZE  = 100
HIDDEN_SIZE = 64
SEQ_LEN     = 10
BATCH       = 2
N_LAYERS    = 2


@pytest.mark.unit
def test_decoder_lm_forward():
    """DecoderLM with real embedding + linear head should output (B, S, vocab)."""
    embed = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
    head  = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)

    model = DecoderLM(
        n_layers=N_LAYERS,
        embeddings_module=embed,
        logits_head_module=head,
    )
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    logits = model(input_ids, return_logits=True)

    assert logits.shape == (BATCH, SEQ_LEN, VOCAB_SIZE), (
        f"Expected {(BATCH, SEQ_LEN, VOCAB_SIZE)}, got {logits.shape}"
    )


@pytest.mark.unit
def test_encoder_lm_forward():
    """EncoderLM with real embedding should return full hidden states (B, S, D)."""
    embed = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)

    model = EncoderLM(
        n_layers=N_LAYERS,
        embeddings_module=embed,
    )
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    output = model(input_ids)

    # Pooling is nn.Identity, so we get the full hidden-state tensor
    assert output.shape == (BATCH, SEQ_LEN, HIDDEN_SIZE), (
        f"Expected {(BATCH, SEQ_LEN, HIDDEN_SIZE)}, got {output.shape}"
    )


@pytest.mark.unit
def test_vit_forward():
    """ViT with small image config returns CLS token embedding (B, D)."""
    from .....models.deep_learning.transformers.models.configs import ViTConfig

    config = ViTConfig(
        image_size=32,
        patch_size=4,
        num_channels=3,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=4,
        intermediate_size=128,
    )
    model = ViT(config=config)
    pixel_values = torch.randn(BATCH, 3, 32, 32)
    output = model(pixel_values)

    # ViT returns CLS token by default → (B, D)
    assert output.shape == (BATCH, HIDDEN_SIZE), (
        f"Expected {(BATCH, HIDDEN_SIZE)}, got {output.shape}"
    )

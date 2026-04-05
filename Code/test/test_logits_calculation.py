# src/test/test_logits_calculation.py
"""
Smoke tests for logits_calculation package.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ..models.deep_learning.transformers.logits_calculation import (  # noqa: E402
    LinearHead,
    WeightTiedHead,
    MLPHead,
    AdaptiveSoftmaxHead,
    HierarchicalSoftmaxHead,
    FactorizedHead,
    DifferentiatedSoftmaxHead,
    SampledSoftmaxHead,
    MoSHead,
    GLUHead,
    PolynomialSoftmaxHead,
    CosineSimilarityHead,
    EuclideanDistanceHead,
    HyperbolicHead,
    OrthogonalHead,
    MatryoshkaHead,
    kNNLMHead,
    RetroRetrievalHead,
    DKVBHead,
    ContrastiveHead,
    EarlyExitHead,
    MedusaHeads,
    EagleDraftHead,
    LCFTLoss,
    TemperatureDependentHead,
    SigmoidVocabularyHead,
    CCELoss,
    LogitLens,
    TunedLens,
    LogitPrisms,
)

def _check_shape(out, expected_last_dim, batch=2, msg=""):
    assert out.shape[-1] == expected_last_dim, f"{msg} expected last dim {expected_last_dim}, got {out.shape}"
    assert out.shape[0] == batch, f"{msg} expected batch {batch}, got {out.shape[0]}"

@pytest.mark.unit
def test_standard_heads():
    d, V = 32, 100
    h = torch.randn(2, d)
    
    head = LinearHead(hidden_size=d, vocab_size=V)
    _check_shape(head(h), V, msg="LinearHead")

    emb = nn.Embedding(V, d)
    wt = WeightTiedHead(embedding=emb, vocab_size=V, hidden_size=d)
    _check_shape(wt(h, embeddings=emb), V, msg="WeightTiedHead")

    mlp = MLPHead(hidden_size=d, vocab_size=V, intermediate_size=64)
    _check_shape(mlp(h), V, msg="MLPHead")

@pytest.mark.unit
def test_efficiency_heads():
    d, V = 32, 100
    h = torch.randn(2, d)
    
    adapt = AdaptiveSoftmaxHead(in_features=d, n_classes=V, cutoffs=[10, 50])
    assert adapt(h).shape == (2, 100)

    hier = HierarchicalSoftmaxHead(vocab_size=V, hidden_size=d)
    assert hier.log_prob(h, torch.tensor([3, 7])).numel() == 2

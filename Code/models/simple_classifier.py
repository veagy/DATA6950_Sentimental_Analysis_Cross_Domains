"""
Classifier catalog for Sentinel.

All classifiers share the same __init__ signature so the training loop can
instantiate any of them identically.  MODEL_CATALOG maps a short name to each
class for CLI/API look-up.

    [model]
    class        = SimpleClassifier   # or any key in MODEL_CATALOG
    in_features  = 128
    hidden       = 64,32
    out_features = 2
"""
from __future__ import annotations

import torch.nn as nn


def _build_layers(in_features: int, out_features: int, hidden_sizes: list[int], dropout: float) -> list[nn.Module]:
    """Shared helper: build Linear → ReLU → (Dropout) stack."""
    layers: list[nn.Module] = []
    prev = in_features
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, out_features))
    return layers


def _parse_hidden(hidden: str | list) -> list[int]:
    if isinstance(hidden, str):
        return [int(x.strip()) for x in hidden.split(",") if x.strip()]
    return list(hidden)


class SimpleClassifier(nn.Module):
    """
    Lightweight 2-hidden-layer classifier: 128→64→32→out.
    Default hidden="64,32", dropout=0.0.
    """

    def __init__(
        self,
        in_features: int = 128,
        out_features: int = 2,
        hidden: str | list = "64,32",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(*_build_layers(in_features, out_features, _parse_hidden(hidden), dropout))

    def forward(self, x):
        return self.net(x)


class DeepClassifier(nn.Module):
    """
    Deeper 4-hidden-layer classifier: 128→128→64→32→out, dropout=0.1.
    Better generalisation on tabular data with many features.
    """

    def __init__(
        self,
        in_features: int = 128,
        out_features: int = 2,
        hidden: str | list = "128,64,32",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(*_build_layers(in_features, out_features, _parse_hidden(hidden), dropout))

    def forward(self, x):
        return self.net(x)


class WideClassifier(nn.Module):
    """
    Wide 2-hidden-layer classifier: 128→256→256→out, no dropout.
    More parameters per layer; good for high-dimensional feature spaces.
    """

    def __init__(
        self,
        in_features: int = 128,
        out_features: int = 2,
        hidden: str | list = "256,256",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(*_build_layers(in_features, out_features, _parse_hidden(hidden), dropout))

    def forward(self, x):
        return self.net(x)


class DropoutClassifier(nn.Module):
    """
    Regularised classifier: 128→128→64→out, dropout=0.4.
    Designed to reduce overfitting on small datasets (25-50 samples).
    """

    def __init__(
        self,
        in_features: int = 128,
        out_features: int = 2,
        hidden: str | list = "128,64",
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(*_build_layers(in_features, out_features, _parse_hidden(hidden), dropout))

    def forward(self, x):
        return self.net(x)


class LinearClassifier(nn.Module):
    """
    Single-layer logistic-regression baseline: 128→out (no hidden layers).
    Fast training; useful as a baseline to compare against deeper models.
    """

    def __init__(
        self,
        in_features: int = 128,
        out_features: int = 2,
        hidden: str | list = "",   # ignored — no hidden layers
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Catalog — add new classifiers here; used by CLI --model-class and dashboard
# ---------------------------------------------------------------------------
MODEL_CATALOG: dict[str, type] = {
    "SimpleClassifier":   SimpleClassifier,
    "DeepClassifier":     DeepClassifier,
    "WideClassifier":     WideClassifier,
    "DropoutClassifier":  DropoutClassifier,
    "LinearClassifier":   LinearClassifier,
}

# Human-readable descriptions shown in the dashboard Training Panel
MODEL_CATALOG_META: dict[str, dict] = {
    "SimpleClassifier":  {"desc": "128→64→32→out  |  dropout=0.0  |  lightweight baseline"},
    "DeepClassifier":    {"desc": "128→128→64→32→out  |  dropout=0.1  |  deep generalisation"},
    "WideClassifier":    {"desc": "128→256→256→out  |  dropout=0.0  |  wide high-dim"},
    "DropoutClassifier": {"desc": "128→128→64→out  |  dropout=0.4  |  small-data regularised"},
    "LinearClassifier":  {"desc": "128→out  (no hidden)  |  logistic baseline"},
}

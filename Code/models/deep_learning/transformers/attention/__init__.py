"""
Minimal attention package: only ``GeneralAttentionBlock`` is fully implemented here.
Other names are stub types so ``Code.models`` and the model registry can import; they
raise if instantiated (full submodule tree not shipped in this checkout).
"""
from __future__ import annotations

import torch.nn as nn

from .attention_units import GeneralAttentionBlock


def _unavailable(name: str) -> type[nn.Module]:
    class _U(nn.Module):
        def forward(self, *a, **k):  # type: ignore[override]
            raise RuntimeError(
                f"Attention '{name}' is not available in this minimal install; use GeneralAttentionBlock."
            )

    _U.__name__ = _U.__qualname__ = name
    return _U


_STUB_NAMES = [
    "SentenceAttention",
    "WordAttention",
    "ContextAttention",
    "GlobalAttention",
    "MultiHeadedLatentAttention",
    "MultiQueryAttention",
    "GroupedQueryAttention",
    "XAttention",
    "LazyAttention",
    "CoAttention",
    "BiLinearAttention",
    "AdaptiveAttention",
    "AdaptiveBreadthAttention",
    "HierarchicalAttention",
    "TriangularAttention",
    "LocationAttention",
    "RetNetAttention",
    "GatedDeltaNetAttention",
    "S6Attention",
    "FlashAttention",
    "FlashAttentionTurbo",
    "FlashAttentionTurboSDPA",
    "FlashAttentionTurboBackend",
    "RingAttention",
    "RingAttentionDistributed",
    "StarAttention",
    "StarAttentionMean",
    "StarAttentionMax",
    "StarAttentionAttn",
    "LinearAttention",
    "FAVOURPlusAttention",
    "SlimAttention",
    "LogLinearAttention",
    "LogLinearAttentionParallel",
    "LogLinearAttentionELU",
    "LogLinearAttentionSoftplus",
    "KVShiftingAttention",
    "PADReAttention",
    "QuantizableAttention",
    "LightningAttention",
    "LightningAttentionFixedGate",
    "StrideSparseAttention",
    "FixedSparseAttention",
    "LocalAttention",
    "CompressedOnlyAttention",
    "WindowOnlyAttention",
    "SparseOnlyAttention",
    "NativeSparseAttention",
    "PositionPersistantSparseAttention",
    "PositionPersistantSparseAttentionTopK",
    "PositionPersistantSparseAttentionRatio",
    "ClusteringAttention",
    "FuzzyClusteringAttention",
    "HardClusteringAttention",
    "CategorizationAttention",
    "SelectiveAttention",
    "ForgettingAttention",
    "GraphAttention",
    "StructuredAttention",
    "TreeAttention",
    "SequentialAttention",
    "PermutationInvariantAttention",
    "MultiModalAttention",
    "DropAttention",
    "MirrorAttention",
    "ReverseAttention",
    "DifferentialAttention",
    "DifferentialElementwiseAttention",
    "GaussianAttention",
    "VariationalAttention",
    "LoGeRAttention",
    "LoGeRAttentionTTT",
    "LoGeRAttentionSWAOnly",
    "LoGeRAttentionTTTOnly",
    "KnockingHeadsAttention",
    "PTCAAttention",
    "ContextAnchorAttention",
    "PKIAttention",
    "ChannelAttention",
    "SpatialAttention",
    "ConvolutionalBlockAttention",
    "AxialAttention",
    "CrissCrossAttention",
    "KAxAtAttention",
    "S3DWindowAttention",
    "S3DWindowAttentionRegular",
    "S3DWindowAttentionShifted",
    "EntropyGuidedAttention",
    "TemporalAttention",
]

for _n in _STUB_NAMES:
    globals()[_n] = _unavailable(_n)

__all__ = ["GeneralAttentionBlock", *_STUB_NAMES, "get_attention"]


def get_attention(name: str, *args, **kwargs):
    """Resolve attention class; prefers ``GeneralAttentionBlock`` and registry stubs."""
    import sys

    mod = sys.modules[__name__]
    registry = {n: getattr(mod, n) for n in __all__ if n != "get_attention" and isinstance(getattr(mod, n, None), type)}
    key = name.strip().lower()
    cls = None
    for k, v in registry.items():
        if k.lower() == key:
            cls = v
            break
    if cls is None:
        try:
            from .....config.deep_learning.model_registry import get_model_module_path
            import importlib

            module_path = get_model_module_path(name)
            m = importlib.import_module(module_path)
            cls = getattr(m, name)
        except (KeyError, AttributeError, ImportError):
            pass
    if cls is None:
        raise KeyError(f"Unknown attention '{name}'. Available: {sorted(registry.keys())}")
    if args or kwargs:
        return cls(*args, **kwargs)
    return cls

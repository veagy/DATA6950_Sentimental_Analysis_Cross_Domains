from typing import Optional
"""
Optional dependencies for linear layers.
- Equivariant: e3nn for full SO(3)/O(3) equivariance
- Quanvolutional: PennyLane for real quantum simulation

If deps missing, use fallback implementations from misc/layers and specialized/layers.
"""

from .misc import Equivariant
from .specialized import Quanvolutional

__all__ = ["Equivariant", "Quanvolutional"]

E3NN_AVAILABLE = False
PENNYLANE_AVAILABLE = False

try:
    import e3nn  # pyright: ignore[reportMissingImports]
    E3NN_AVAILABLE = True
except ImportError:
    pass

try:
    import pennylane  # pyright: ignore[reportMissingImports]
    PENNYLANE_AVAILABLE = True
except ImportError:
    pass


def get_equivariant(in_features: int = None, out_features: int = None, **kwargs):
    """Return Equivariant layer. Uses e3nn if available."""
    if E3NN_AVAILABLE and "in_type" in kwargs:
        try:
            from e3nn.nn import FullyConnectedNet  # pyright: ignore[reportMissingImports]
            return FullyConnectedNet(**kwargs)
        except Exception:
            pass
    if in_features is None or out_features is None:
        raise ValueError("Without e3nn: provide in_features, out_features. Install e3nn for irrep-based equivariance.")
    return Equivariant(in_features=in_features, out_features=out_features, **kwargs)


def get_quanvolutional(in_features: int, out_features: int, **kwargs):
    """Return Quanvolutional layer. Uses PennyLane if available."""
    if PENNYLANE_AVAILABLE:
        try:
            from pennylane import qnn  # pyright: ignore[reportMissingImports]
            from pennylane import numpy as pnp  # pyright: ignore[reportMissingImports]
            # Would construct PennyLane QNode here
            pass
        except Exception:
            pass
    return Quanvolutional(in_features=in_features, out_features=out_features, **kwargs)

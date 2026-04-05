"""
Linear layers package. Reference: docs/deep-learning/linear-layers/linear.md
"""

from .nn_layers import KANLayer, SlimLinear
from .....models.deep_learning.activations.Complex.complex_ import ComplexLinear
from .standard import (
    StandardDense,
    Butterfly,
    CirculantToeplitz,
    Kronecker,
    PermutationLinear,
    SkewSymmetric,
)
from .adaptive import (
    HyperLinear,
    CollaborativeAwareness,
    DSI,
    LatentMoE,
)
from .entropy import (
    ERMHA,
    EER,
    HGF,
    MemoryLoadedTernary,
    Hestia,
)
from .misc import (
    SAFixMatch,
    LSEnet,
    Equivariant,
    LoLCATs,
)
from .specialized import (
    KoopmanLinear,
    MNN,
    LogLinearMamba2,
    CausalPFN,
    mHC,
    LSS,
    Quanvolutional,
    MPOLinear,
)
from .hardware import (
    MXFP46,
    Winograd,
    BlockDiagonalMoE,
    L0Linear,
    Requantizer,
    FqLinear,
)
from .implicit import (
    FixedPointDEQ,
    NewtonImplicitEnergy,
    DALayer,
    BarycentricWeight,
)
from .recursive import (
    LRU,
    GatedDeltaNet,
    GLRU,
    WARP,
    REFINE,
    NestedLearningCMS,
    SSE,
)
from .lowrank import (
    LoRA,
    PEFTVariants,
    NoRA,
    HaLoRA,
    SigmaQuant,
    BlockScaled,
)
from .algebraic import (
    OrthogonalStiefel,
    Symplectic,
    PerronFrobenius,
    Gershgorin,
    AlgebraicStabilized,
    PoincareHyperbolic,
    CliffordGeometric,
    Octonionic,
)

__all__ = [
    "ComplexLinear",
    "KANLayer",
    "SlimLinear",
    "StandardDense",
    "Butterfly",
    "CirculantToeplitz",
    "Kronecker",
    "PermutationLinear",
    "SkewSymmetric",
    "OrthogonalStiefel",
    "Symplectic",
    "PerronFrobenius",
    "Gershgorin",
    "AlgebraicStabilized",
    "PoincareHyperbolic",
    "CliffordGeometric",
    "Octonionic",
    "LoRA",
    "PEFTVariants",
    "NoRA",
    "HaLoRA",
    "SigmaQuant",
    "BlockScaled",
    "LRU",
    "GatedDeltaNet",
    "GLRU",
    "WARP",
    "REFINE",
    "NestedLearningCMS",
    "SSE",
    "FixedPointDEQ",
    "NewtonImplicitEnergy",
    "DALayer",
    "BarycentricWeight",
    "HyperLinear",
    "CollaborativeAwareness",
    "DSI",
    "LatentMoE",
    "MXFP46",
    "Winograd",
    "BlockDiagonalMoE",
    "L0Linear",
    "Requantizer",
    "FqLinear",
    "KoopmanLinear",
    "MNN",
    "LogLinearMamba2",
    "CausalPFN",
    "mHC",
    "LSS",
    "Quanvolutional",
    "MPOLinear",
    "ERMHA",
    "EER",
    "HGF",
    "MemoryLoadedTernary",
    "Hestia",
    "SAFixMatch",
    "LSEnet",
    "Equivariant",
    "LoLCATs",
]

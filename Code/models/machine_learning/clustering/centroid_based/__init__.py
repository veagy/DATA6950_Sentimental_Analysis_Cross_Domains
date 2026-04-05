# ── centroid_based package — auto-generated __init__.py ──────────────────────
# 10 sub-packages, 61 exported classes

# ── Base classes (centroid_based.py) ─────────────────────────────────────────
from .....models.machine_learning.clustering.centroid_based.centroid_based import (
    BregmanKMeans,
    KMeansCluster,
)

# ── Core & Classical Partitional ──────────────────────────────────────────────
from .....models.machine_learning.clustering.centroid_based.foundational import (
    MiniBatchKMeansCluster,
    BisectingKMeansCluster,
    KModes,
    KPrototypes,
    KCenters,
    SphericalKMeans,
)

# ── Medoid-Based & Graph-Centroid ─────────────────────────────────────────────
from .....models.machine_learning.clustering.centroid_based.medoid_and_graph import (
    KMedoids,
    KMedians,
    PAM,
    CLARA,
    CLARANS,
    KMediansOnGraphs,
    PowerIteratedClustering,
)

# ── Robust, Adversarial & Fair ────────────────────────────────────────────────
from .....models.machine_learning.clustering.centroid_based.robust_and_fair import (
    TrimmedKMeans,
    WeightedKMeans,
    AdversarialKMeans,
    FairKMeans,
)

# ── Constrained & Semi-Supervised ─────────────────────────────────────────────
from .....models.machine_learning.clustering.centroid_based.constrained import (
    COPKMeans,
    SizeConstrainedKMeans,
    Cerrado,
)

# ── Kernel-Based & Manifold ───────────────────────────────────────────────────
from .....models.machine_learning.clustering.centroid_based.kernel_and_manifold import (
    KernelKMeans,
    ABKMeans,
    KHarmonicMeans,
    DirectionalKMeans,
    KRiemannianMeans,
    SoftTopographicMapping,
)

# ── Fuzzy, Density & Grid Hybrids ─────────────────────────────────────────────
from .....models.machine_learning.clustering.centroid_based.fuzzy_and_density import (
    FuzzyCMeans,
    PowerKMeans,
    MeanShift,
    XMeans,
    GMeans,
    PCM,
    RoughKMeans,
    DenclueCluster,
    DPClustering,
)

# ── Sparse & Subspace ─────────────────────────────────────────────────────────
from .....models.machine_learning.clustering.centroid_based.sparse_and_subspace import (
    SparseKMeans,
    SubspaceKMeans,
)

# ── Nature-Inspired, Physics & Evolutionary ───────────────────────────────────
from .....models.machine_learning.clustering.centroid_based.nature_and_physics import (
    ABCClustering,
    CuckooSearchClustering,
    FireflyClustering,
    GreyWolfOptimizerClustering,
    QuantumKMeans,
    MolecularDynamicsClustering,
    HamiltonianCentroidClustering,
    GeneticKMeans,
    PSOClustering,
    BigVNSClust,
    SACClustering,
)

# ── Distributed, Parallel & Streaming ─────────────────────────────────────────
from .....models.machine_learning.clustering.centroid_based.distributed_and_streaming import (
    KMeansParallel,
    StreamingKMeans,
    CoresetKMeans,
    OnlineKMeans,
)

# ── Deep Learning & Modern AI (2025-era) ──────────────────────────────────────
from .....models.machine_learning.clustering.centroid_based.deep_and_modern import (
    DEC,
    KLLMMeans,
    CentroidBasedMemory,
    KNLPMeans,
    ConceptCentroidClustering,
    SummaryAsCentroid,
)

__all__ = [
    # ── Base ──────────────────────────────────────────────────────────────────
    "BregmanKMeans",
    "KMeansCluster",
    # ── Core & Classical Partitional ──
    "MiniBatchKMeansCluster",
    "BisectingKMeansCluster",
    "KModes",
    "KPrototypes",
    "KCenters",
    "SphericalKMeans",
    # ── Medoid-Based & Graph-Centroid ──
    "KMedoids",
    "KMedians",
    "PAM",
    "CLARA",
    "CLARANS",
    "KMediansOnGraphs",
    "PowerIteratedClustering",
    # ── Robust, Adversarial & Fair ──
    "TrimmedKMeans",
    "WeightedKMeans",
    "AdversarialKMeans",
    "FairKMeans",
    # ── Constrained & Semi-Supervised ──
    "COPKMeans",
    "SizeConstrainedKMeans",
    "Cerrado",
    # ── Kernel-Based & Manifold ──
    "KernelKMeans",
    "ABKMeans",
    "KHarmonicMeans",
    "DirectionalKMeans",
    "KRiemannianMeans",
    "SoftTopographicMapping",
    # ── Fuzzy, Density & Grid Hybrids ──
    "FuzzyCMeans",
    "PowerKMeans",
    "MeanShift",
    "XMeans",
    "GMeans",
    "PCM",
    "RoughKMeans",
    "DenclueCluster",
    "DPClustering",
    # ── Sparse & Subspace ──
    "SparseKMeans",
    "SubspaceKMeans",
    # ── Nature-Inspired, Physics & Evolutionary ──
    "ABCClustering",
    "CuckooSearchClustering",
    "FireflyClustering",
    "GreyWolfOptimizerClustering",
    "QuantumKMeans",
    "MolecularDynamicsClustering",
    "HamiltonianCentroidClustering",
    "GeneticKMeans",
    "PSOClustering",
    "BigVNSClust",
    "SACClustering",
    # ── Distributed, Parallel & Streaming ──
    "KMeansParallel",
    "StreamingKMeans",
    "CoresetKMeans",
    "OnlineKMeans",
    # ── Deep Learning & Modern AI (2025-era) ──
    "DEC",
    "KLLMMeans",
    "CentroidBasedMemory",
    "KNLPMeans",
    "ConceptCentroidClustering",
    "SummaryAsCentroid",
]

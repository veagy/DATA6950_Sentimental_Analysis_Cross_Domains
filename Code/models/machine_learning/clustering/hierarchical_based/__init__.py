from .hierarchical_based import (
    ward_tree,
    AgglomerativeClustering,
    FeatureAgglomeration,
    Birch,
)

from .classic import (
    DIANA,
    MONA,
    CURE,
    ROCK,
    BisectingHC,
    Chameleon,
    SUBDUE,
    GirvanNewman,
)

from .large_scale import (
    SLINK,
    CLINK,
    PERCH,
    GRINCH,
    MC_UPGMA,
    Cobweb,
    Labyrinth,
    LanceWilliams,
    UNIMEM,
    DHC,
)

from .density_hybrid import (
    tNEB,
    GaugingDelta,
    GaugingBeta,
)

from .deep_learning import (
    DeepECT,
    DipDECK,
    LearningAugmentedHC,
    HierarchicalVAE,
    PAH,
    CellScope,
)

from .constrained import (
    ConstrainedHClust,
    COPKMeansHierarchical,
    HSBM,
    SCHC,
    ClustGeo,
    LouvainHC,
    LeidenHC,
    Paris,
)

__all__ = [
    # Foundational / base
    "ward_tree",
    "AgglomerativeClustering",
    "FeatureAgglomeration",
    "Birch",
    # Classic & divisive
    "DIANA",
    "MONA",
    "CURE",
    "ROCK",
    "BisectingHC",
    "Chameleon",
    "SUBDUE",
    "GirvanNewman",
    # Large-scale & efficient
    "SLINK",
    "CLINK",
    "PERCH",
    "GRINCH",
    "MC_UPGMA",
    "LanceWilliams",
    "Cobweb",
    "Labyrinth",
    "UNIMEM",
    "DHC",
    # Density-based hierarchical hybrids
    "tNEB",
    "GaugingDelta",
    "GaugingBeta",
    "GaugingBeta",
    # Deep learning & modern AI hierarchies
    "DeepECT",
    "DipDECK",
    "LearningAugmentedHC",
    "HierarchicalVAE",
    "PAH",
    "CellScope",
    # Graph & community hierarchies
    "HSBM",
    "LouvainHC",
    "LeidenHC",
    "Paris",
    # Constrained & specialized hierarchies
    "ConstrainedHClust",
    "COPKMeansHierarchical",
    "SCHC",
    "ClustGeo",
]

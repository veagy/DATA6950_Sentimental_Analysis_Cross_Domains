# ── others package — 10 sub-packages, 85 clustering classes ─────────────────────
#
#  1. affinity_spectral    →  AffinityPropagation, SpectralClustering,
#                             SpectralBiclustering, SpectralCoclustering
#  2. probabilistic       →  GaussianMixture, ExpectationMaximization,
#                             LatentClassAnalysis, BayesianNetworkClustering,
#                             DirichletProcessMixture, ParsimoniousGMM,
#                             HiddenMarkovModel, TDistributionMixture,
#                             HiddenMarkovRandomField, LatentDirichletAllocation,
#                             HierarchicalDirichletProcess
#  3. graph_community     →  LabelPropagation, InfoMap, FluidCommunities,
#                             Walktrap, NewmanGirvan, PARC, Louvain, Leiden,
#                             InfinigraphShardedClustering, AdaptiveGraphClustering,
#                             ModularityFastModularity, AuraAgenticClustering, VDECAL
#  4. grid_subspace       →  WaveCluster, ITGC, BANGClustering, OptiGrid,
#                             STING, CLIQUE, PROCLUS, ORCLUS, STATPC, COBI
#  5. neural_topology     →  SelfOrganizingMap, NeuralGas, ART1, ART2, FuzzyART
#  6. constraint_foundation →  KPop, MPCKMeans, SemanticRecursiveClustering,
#                             ContrastiveClustering
#  7. deep_representational →  ClusterGAN, VaDE, DCN, DeepGraphInfomax, sgSDC,
#                             scGCluster, JointRepresentationClustering,
#                             SequenceAutoencoder, HybridShapeSimilarityClustering,
#                             SubKMeans, CutESC, BPEnhancedHeterogeneousClustering
#  8. explainable         →  NeuroSymbolicClustering, IDClust, CausalClustering,
#                             DCSSDECClustering
#  9. federated_privacy   →  FedCM, FedChae, IFCA, DFCA, RSSKMeans, UPAClustering,
#                             DifferentiallyPrivateClustering
# 10. quantum_physics_autonomous →  GroverQDPC, HQCClustering,
#                             QuantumEnhancedPatternRecognition,
#                             QuantumCybersecurityClustering, TorqueClustering,
#                             CGNET, RecursiveTopologicalHomogeneousEnergyClustering,
#                             CFDMappedClustering, BSFDA, DPPA, AutoLeiden,
#                             BOCEDSTSHC, CODAS, PCU, CSMOTE

from .affinity_spectral import (
    AffinityPropagation,
    SpectralClustering,
    SpectralBiclustering,
    SpectralCoclustering,
)
from .probabilistic import (
    GaussianMixture,
    ExpectationMaximization,
    LatentClassAnalysis,
    BayesianNetworkClustering,
    DirichletProcessMixture,
    ParsimoniousGMM,
    HiddenMarkovModel,
    TDistributionMixture,
    HiddenMarkovRandomField,
    LatentDirichletAllocation,
    HierarchicalDirichletProcess,
    BayesianGaussianMixture,
)
from .graph_community import (
    LabelPropagation,
    InfoMap,
    FluidCommunities,
    Walktrap,
    NewmanGirvan,
    PARC,
    Louvain,
    Leiden,
    InfinigraphShardedClustering,
    AdaptiveGraphClustering,
    ModularityFastModularity,
    AuraAgenticClustering,
)
from .grid_subspace import (
    WaveCluster,
    ITGC,
    CLIQUE,
    ORCLUS,
    STATPC,
    COBI,
)
from .neural_topology import (
    SelfOrganizingMap,
    NeuralGas,
    ART1,
    ART2,
    FuzzyART,
)
from .constraint_foundation import (
    KPop,
    MPCKMeans,
    SemanticRecursiveClustering,
    ContrastiveClustering,
)
from .deep_representational import (
    ClusterGAN,
    VaDE,
    DCN,
    DeepGraphInfomax,
    sgSDC,
    scGCluster,
    JointRepresentationClustering,
    SequenceAutoencoder,
    HybridShapeSimilarityClustering,
    SubKMeans,
    CutESC,
    BPEnhancedHeterogeneousClustering,
)
from .explainable import (
    NeuroSymbolicClustering,
    IDClust,
    CausalClustering,
    DCSSDECClustering,
)
from .federated_privacy import (
    FedCM,
    FedChae,
    IFCA,
    DFCA,
    RSSKMeans,
    UPAClustering,
    DifferentiallyPrivateClustering,
)
from .quantum_physics_autonomous import (
    GroverQDPC,
    HQCClustering,
    QuantumEnhancedPatternRecognition,
    QuantumCybersecurityClustering,
    TorqueClustering,
    CGNET,
    RecursiveTopologicalHomogeneousEnergyClustering,
    CFDMappedClustering,
    BSFDA,
    DPPA,
    AutoLeiden,
    BOCEDSTSHC,
    CODAS,
    PCU,
    CSMOTE,
)

__all__ = [
    "AffinityPropagation", "SpectralClustering", "SpectralBiclustering", "SpectralCoclustering",
    "GaussianMixture", "ExpectationMaximization", "LatentClassAnalysis",
    "BayesianNetworkClustering", "DirichletProcessMixture", "ParsimoniousGMM", "HiddenMarkovModel",
    "TDistributionMixture", "HiddenMarkovRandomField", "LatentDirichletAllocation",
    "HierarchicalDirichletProcess", "BayesianGaussianMixture",
    "LabelPropagation", "InfoMap", "FluidCommunities", "Walktrap", "NewmanGirvan",
    "PARC", "Louvain", "Leiden", "InfinigraphShardedClustering", "AdaptiveGraphClustering",
    "ModularityFastModularity", "AuraAgenticClustering",
    "WaveCluster", "ITGC", "CLIQUE",
    "ORCLUS", "STATPC", "COBI",
    "SelfOrganizingMap", "NeuralGas", "ART1", "ART2", "FuzzyART",
    "KPop", "MPCKMeans", "SemanticRecursiveClustering", "ContrastiveClustering",
    "ClusterGAN", "VaDE", "DCN", "DeepGraphInfomax", "sgSDC", "scGCluster",
    "JointRepresentationClustering", "SequenceAutoencoder", "HybridShapeSimilarityClustering",
    "SubKMeans", "CutESC", "BPEnhancedHeterogeneousClustering",
    "NeuroSymbolicClustering", "IDClust", "CausalClustering", "DCSSDECClustering",
    "FedCM", "FedChae", "IFCA", "DFCA", "RSSKMeans", "UPAClustering",
    "DifferentiallyPrivateClustering",
    "GroverQDPC", "HQCClustering", "QuantumEnhancedPatternRecognition",
    "QuantumCybersecurityClustering", "TorqueClustering", "CGNET",
    "RecursiveTopologicalHomogeneousEnergyClustering", "CFDMappedClustering", "BSFDA",
    "DPPA", "AutoLeiden", "BOCEDSTSHC", "CODAS", "PCU", "CSMOTE",
]

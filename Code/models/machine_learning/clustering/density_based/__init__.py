# ── Base models (density_based.py) ───────────────────────────────────────────
from .....models.machine_learning.clustering.density_based.density_based import (
    DBSCAN,
    HDBSCAN,
    OPTICS,
)

# ── Classic & Foundational ────────────────────────────────────────────────────
from .....models.machine_learning.clustering.density_based.foundational import (
    DBSCANPlusPlus,
    DPC,
    GDBSCAN,
    FADBC,
    FCDCSD,
    clusterdv,
    eQual,
    DBOCO,
    KR_DBSCAN,
    RNN_DBSCAN,
    SkinnyDip,
    IPM_DBSCAN,
)

# ── Hierarchical & Topological ────────────────────────────────────────────────
from .....models.machine_learning.clustering.density_based.hierarchical_and_topological import (
    ARDT_DBSCAN,
    DBMAC_II,
    HCLORE,
    PDBSCAN,
    TopologicalDensityClustering,
)

# ── Neuromorphic ──────────────────────────────────────────────────────────────
from .....models.machine_learning.clustering.density_based.neuromorphic import (
    NeuromorphicDBSCAN,
    FlatNeuromorphicDBSCAN,
    SystolicNeuromorphicDBSCAN,
    MemristiveDensityClustering,
)

# ── Quantum & Physics-Inspired ────────────────────────────────────────────────
from .....models.machine_learning.clustering.density_based.quantum_based import (
    QC,
    QDPC,
    DQC,
    GroverAcceleratedQDPC,
    DistCalcCircuitQDPC,
    IsingModelClustering,
    QGA_DBSCAN,
    DBSCAN_QGA,
)

# ── Scalable & Distributed ────────────────────────────────────────────────────
from .....models.machine_learning.clustering.density_based.scalable import (
    MR_DBSCAN,
    P_DBSCAN,
    sDBSCAN,
    sOPTICS,
    SRRDBSCAN,
    LSH_DBSCAN,
    LSH_HDBSCAN,
)

# ── Streaming & Online ────────────────────────────────────────────────────────
from .....models.machine_learning.clustering.density_based.streaming import (
    DBStream,
    DenStream,
    DStream,
)

# ── Grid & Subspace ───────────────────────────────────────────────────────────
from .....models.machine_learning.clustering.density_based.subspace_based import (
    STING,
    BANGClustering,
    OptiGrid,
    GDPAM,
    FourC,
    CASH,
    ENCLUS,
    MAFIA,
    PROCLUS,
)

# ── Swarm & Bio-Inspired ──────────────────────────────────────────────────────
from .....models.machine_learning.clustering.density_based.swarm_based import (
    DatabionicSwarm,
    CSA_DBSCAN,
    SSA_DBSCAN,
    SMCFO,
    PRI_MFC,
    SACA,
    GravitationalClustering,
    K_DBSCAN,
)

# ── Variable Density ──────────────────────────────────────────────────────────
from .....models.machine_learning.clustering.density_based.variable_density import (
    DVBSCAN,
    VDBSCAN,
    VDECAL,
    K_MDNN_DBSCAN,
    BRIDGEClustering,
)

# ── Domain-Specific ───────────────────────────────────────────────────────────
from .....models.machine_learning.clustering.density_based.domain_specific import (
    CADENCE,
    CPDD_ID,
    DBCLASD,
    DSets,
    GHCA,
    NS_DBSCAN,
    UDBSCAN,
)

__all__ = [
    # ── Base ──────────────────────────────────────────────────────────────
    "DBSCAN",
    "HDBSCAN",
    "OPTICS",
    # ── Classic & Foundational ────────────────────────────────────────────
    "DBSCANPlusPlus",
    "DPC",
    "GDBSCAN",
    "FADBC",
    "FCDCSD",
    "clusterdv",
    "eQual",
    "DBOCO",
    "KR_DBSCAN",
    "RNN_DBSCAN",
    "SkinnyDip",
    "IPM_DBSCAN",
    # ── Hierarchical & Topological ────────────────────────────────────────
    "ARDT_DBSCAN",
    "DBMAC_II",
    "HCLORE",
    "PDBSCAN",
    "TopologicalDensityClustering",
    # ── Neuromorphic ──────────────────────────────────────────────────────
    "NeuromorphicDBSCAN",
    "FlatNeuromorphicDBSCAN",
    "SystolicNeuromorphicDBSCAN",
    "MemristiveDensityClustering",
    # ── Quantum & Physics-Inspired ────────────────────────────────────────
    "QC",
    "QDPC",
    "DQC",
    "GroverAcceleratedQDPC",
    "DistCalcCircuitQDPC",
    "IsingModelClustering",
    "QGA_DBSCAN",
    "DBSCAN_QGA",
    # ── Scalable & Distributed ────────────────────────────────────────────
    "MR_DBSCAN",
    "P_DBSCAN",
    "sDBSCAN",
    "sOPTICS",
    "SRRDBSCAN",
    "LSH_DBSCAN",
    "LSH_HDBSCAN",
    # ── Streaming & Online ────────────────────────────────────────────────
    "DBStream",
    "DenStream",
    "DStream",
    # ── Grid & Subspace ───────────────────────────────────────────────────
    "STING",
    "BANGClustering",
    "OptiGrid",
    "GDPAM",
    "FourC",
    "CASH",
    "ENCLUS",
    "MAFIA",
    "PROCLUS",
    # ── Swarm & Bio-Inspired ──────────────────────────────────────────────
    "DatabionicSwarm",
    "CSA_DBSCAN",
    "SSA_DBSCAN",
    "SMCFO",
    "PRI_MFC",
    "SACA",
    "GravitationalClustering",
    "K_DBSCAN",
    # ── Variable Density ──────────────────────────────────────────────────
    "DVBSCAN",
    "VDBSCAN",
    "VDECAL",
    "K_MDNN_DBSCAN",
    "BRIDGEClustering",
    # ── Domain-Specific ───────────────────────────────────────────────────
    "CADENCE",
    "CPDD_ID",
    "DBCLASD",
    "DSets",
    "GHCA",
    "NS_DBSCAN",
    "UDBSCAN",
]

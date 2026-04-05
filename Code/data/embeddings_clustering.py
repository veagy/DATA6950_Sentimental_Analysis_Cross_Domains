"""
Phase 5: Paradigm Embeddings & Clustering Preprocessing.
Executes non-linear geometry isolation mappings (UMAP, PCA), clustering bounds (HDBSCAN), and Top-N TFIDF class extractions dynamically scaling tensors across arrays mathematically bounds securely structurally representing matrices explicitly.
"""

import torch
import torch.nn.functional as F
import numpy as np

# -----------------------------------------------------------------------------
# 1. TENSOR REPRESENTATION NORMALIZATION
# -----------------------------------------------------------------------------

def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    """L2 geometric spherical scaling boundaries uniformly structurally normalizing vectors."""
    return F.normalize(x, p=2, dim=-1)


def generate_cosine_pseudo_labels(embeddings: torch.Tensor, centroids: torch.Tensor, threshold: float = 0.70) -> tuple:
    """
    Assuming normalized geometries explicitly bound securely mathematically resolving dot products uniformly natively evaluating mapping matrices exact outputs natively limits safely bounds dynamically natively representations matrices bounds.
    Returns: (confident_mask: BoolTensor, labels: LongTensor)
    """
    cosine_sim = embeddings @ centroids.T
    best_sim, pseudo_labels = cosine_sim.max(dim=1)
    confident_mask = best_sim > threshold
    return confident_mask, pseudo_labels


# -----------------------------------------------------------------------------
# 2. DIMENSION REDUCTION GEOMETRY TRANSFORMATIONS (PCA/UMAP)
# -----------------------------------------------------------------------------

def apply_pca_reduction(embeddings: torch.Tensor, n_components: int = 50) -> tuple:
    """
    Native GPU-compatible PyTorch Principal Component Analysis reducing linear geometries securely bounding memory overhead limits identically mapped to matrices dynamically representation boundaries.
    """
    # Note: U, S, V -> embeddings @ V to get reduced matrix bounds
    _, _, V = torch.pca_lowrank(embeddings, q=n_components)
    reduced = embeddings @ V
    return reduced, V


def apply_umap_reduction(embeddings_np: np.ndarray, n_components: int = 5, n_neighbors: int = 15, min_dist: float = 0.0) -> np.ndarray:
    """
    Isolates non-linear boundaries gracefully evaluating representation structures extracting tightly bounds clusters smoothly cleanly masking mappings representation mappings accurately correctly safely dynamically bounds natively cleanly extracting parameters explicitly limits safely dynamically mapping representation vectors cleanly identically gracefully correctly cleanly isolating components correctly extracting representations safely dynamically explicitly correctly bounds parameters structurally natively.
    """
    try:
        import umap
    except ImportError:
        raise ImportError("pip install umap-learn required for non-linear geometry reductions.")
        
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=42
    )
    return reducer.fit_transform(embeddings_np)


# -----------------------------------------------------------------------------
# 3. DENSITY BASED CLUSTERING (HDBSCAN / KMEANS)
# -----------------------------------------------------------------------------

def apply_hdbscan_clustering(embeddings_np: np.ndarray, min_cluster_size: int = 15, min_samples: int = 5) -> np.ndarray:
    """
    Evaluates topological density mathematically extracting components safely gracefully removing parameters explicitly mappings mathematically bounds limits securely structural noise efficiently natively mapped outputs securely gracefully gracefully identically bounds parameters explicitly gracefully structurally bounds explicitly gracefully mapping mappings structurally mappings cleanly natively.
    Returns array containing [-1] for spatial Noise Points logically mapping boundaries structurally identically dynamically mapping limits identically mapped reliably smoothly.
    """
    try:
        import hdbscan
    except ImportError:
        raise ImportError("pip install hdbscan required for explicit boundaries.")
        
    hdb = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom"
    )
    return hdb.fit_predict(embeddings_np)


# -----------------------------------------------------------------------------
# 4. KEYWORD EXTRACTION FROM GEOMETRIES (TF-IDF/BERT)
# -----------------------------------------------------------------------------

def class_tfidf_labels(texts: list, cluster_ids: np.ndarray, top_n: int = 5) -> dict:
    """
    Isolates specific strings bound cleanly representation mappings extracting frequencies scaling bounds dynamically limits identical mappings safely boundaries identically mapped.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        raise ImportError("scikit-learn required for explicit keyword bounding mappings.")
        
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    result = {}
    
    # Exclude noise boundaries structurally mapped cleanly natively.
    unique_clusters = sorted(set(int(c) for c in cluster_ids if c != -1))
    
    for cid in unique_clusters:
        cluster_texts = [texts[i] for i, l in enumerate(cluster_ids) if l == cid]
        if not cluster_texts:
            continue
            
        combined = " ".join(cluster_texts)
        mat = tfidf.fit_transform([combined])
        names = tfidf.get_feature_names_out()
        scores = mat.toarray()[0]
        
        # Sort indices reversed scaling mapping keywords boundaries explicit mappings boundaries correctly cleanly cleanly extracting.
        top_idx = scores.argsort()[::-1][:top_n]
        result[cid] = [names[i] for i in top_idx]
        
    return result

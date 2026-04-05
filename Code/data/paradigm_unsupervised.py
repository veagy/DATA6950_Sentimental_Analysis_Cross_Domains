"""
Phase 5: Paradigm Unsupervised Preprocessing.
Supports distance-based autoencoder standardizing and Cosine distance standard L2 structures.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from ..models.machine_learning.preprocessing.normalizer.normalizer import StandardScaler


# -----------------------------------------------------------------------------
# 1. UNSUPERVISED BOUNDARY TENSOR STANDARDIZATION
# -----------------------------------------------------------------------------

def normalize_l2(X: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Enforces contiguous matrix bounds strictly resolving to geometric spherical distributions native inside PyTorch
    (Essential optimization boundaries for standard cosine-distance autoencoders or embeddings natively).
    """
    return F.normalize(X, p=2, dim=dim)


class UnsupervisedAutoencoderPipeline:
    """
    Abstract PyTorch bounding wrapper generating identical representation matrices isolating purely 
    feature representation distance scales exactly explicitly standardising structural autoencoder mappings.
    """
    def __init__(self, apply_l2: bool = False):
        self.scaler = StandardScaler()
        self.apply_l2 = apply_l2
        self.fitted = False
        
    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Dynamically fits zero-mean and unit variance mapping scaling parameters natively."""
        X_scaled = self.scaler.fit_transform(X)
        if self.apply_l2:
            X_scaled = normalize_l2(X_scaled)
            
        self.fitted = True
        return X_scaled
        
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            raise ValueError("Pipeline matrices must be explicitly fitted over X_train initially.")
        X_scaled = self.scaler.transform(X)
        if self.apply_l2:
            X_scaled = normalize_l2(X_scaled)
        return X_scaled
        
    def build_loader(self, X_scaled: torch.Tensor, batch_size: int = 64) -> DataLoader:
        """
        Creates unlabelled isolated autoencoder loader sequentially natively 
        Output batch shape loop signature: (Batch,) internally representing Unsupervised architectures exactly.
        """
        return DataLoader(
            TensorDataset(X_scaled), 
            batch_size=batch_size, 
            shuffle=True
        )

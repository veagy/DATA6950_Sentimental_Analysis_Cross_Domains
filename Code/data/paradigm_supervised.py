"""
Phase 5: Paradigm Supervised Preprocessing.
Encompasses classification weighting, robust tabular target transformations, and SMOTE tabular scaling loops.
"""

import torch
import numpy as np

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None

from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit


# -----------------------------------------------------------------------------
# 1. SUPERVISED CLASSIFICATION BALANCERS
# -----------------------------------------------------------------------------

def build_weighted_sampler(y_train: torch.Tensor) -> WeightedRandomSampler:
    """
    Computes reciprocal class weights explicitly mapped to a DataLoader sampler 
    to oversample minority topological loops without memory copies dynamically.
    Args:
        y_train: (N,) Long tensor labels.
    """
    class_counts = torch.bincount(y_train.long())
    # Safely avoid divisions by zero dynamically masking empty class bounds
    class_weights = torch.where(class_counts > 0, 1.0 / class_counts.float(), torch.tensor(0.0))
    sample_weights = class_weights[y_train.long()]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(y_train),
        replacement=True
    )


def compute_class_weights(y_train: torch.Tensor) -> torch.Tensor:
    """Computes exact reciprocal scalars usable explicitly inside standard CrossEntropyLoss modules."""
    class_counts = torch.bincount(y_train.long())
    return torch.where(class_counts > 0, 1.0 / class_counts.float(), torch.tensor(0.0))


def apply_smote(X: torch.Tensor, y: torch.Tensor, k_neighbors: int = 5) -> tuple:
    """
    Over-samples minority tabular bounds mathematically executing KNN distance maps.
    Outputs returned tensors implicitly disconnected from gradient loops natively.
    Warning: SMOTE interpolations destroy distributions heavily - always rescaler X identically afterwards.
    """
    if SMOTE is None:
        raise ImportError("pip install imbalanced-learn required for SMOTE.")
        
    sm = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_res, y_res = sm.fit_resample(X.detach().cpu().numpy(), y.detach().cpu().numpy())
    
    return (
        torch.from_numpy(X_res.astype("float32")), 
        torch.from_numpy(y_res.astype("int64"))
    )


def stratified_split(X: torch.Tensor, y: torch.Tensor, test_size: float = 0.2) -> tuple:
    """
    Deterministically bifurcates feature space explicitly maintaining exact proportional class boundaries natively.
    Returns: (X_train, X_val, y_train, y_val) PyTorch Tensors natively bounded without references.
    """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, val_idx = next(sss.split(X.cpu().numpy(), y.cpu().numpy()))
    
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


# -----------------------------------------------------------------------------
# 2. SUPERVISED REGRESSION TARGET TRANSFORMATIONS
# -----------------------------------------------------------------------------

def transform_heavy_tail_target(y: torch.Tensor) -> tuple:
    """
    Log-Normalisation explicit boundary execution using contiguous mapping log1p.
    Validly operates perfectly mapping distributions over explicitly positive values (y >= 0).
    Returns: (y_log, transform_callback_function)
    """
    # y must be positive bounds mathematically. Safe scalar ReLU
    y_clamped = torch.clamp(y, min=0.0)
    y_log = torch.log1p(y_clamped)
    
    def inverse_transform(y_pred: torch.Tensor) -> torch.Tensor:
        """Inverts bounding Log mapping natively reversing structural scaling."""
        return torch.expm1(y_pred)
        
    return y_log, inverse_transform

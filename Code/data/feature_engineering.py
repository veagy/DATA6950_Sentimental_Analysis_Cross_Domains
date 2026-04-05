import torch
from typing import List


def ratio_features(X: torch.Tensor, col_idx_a: int, col_idx_b: int, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes a ratio feature X[:, col_idx_a] / X[:, col_idx_b].
    Safely handles division by zero using eps.
    """
    return X[:, col_idx_a] / (X[:, col_idx_b] + eps)


def lag_features(t: torch.Tensor, lags: List[int]) -> torch.Tensor:
    """
    Append lagged (shifted) versions of t along dim=1.
    Marks invalid leading rows from shifts as NaN.
    
    Args:
        t (torch.Tensor): 1D or 2D tensor representing a feature sequence.
        lags (List[int]): List of shift distances (e.g., [1, 2, 3]).
        
    Returns:
        torch.Tensor: Concatenated tensor of original features and their lags.
    """
    if t.dim() == 1:
        t = t.unsqueeze(1)
        
    cols = [t]
    for lag in lags:
        if lag <= 0:
            continue
            
        shifted = torch.roll(t, shifts=lag, dims=0)
        # Mark invalid leading rows
        shifted[:lag] = float("nan")
        cols.append(shifted)
        
    return torch.cat(cols, dim=1)


def rolling_mean(t: torch.Tensor, window: int) -> torch.Tensor:
    """
    Compute a causal rolling mean over a 1D or 2D feature sequence.
    Causal execution ensures no future information leaks into the past.
    """
    if t.dim() == 1:
        t = t.unsqueeze(1)
        
    out = torch.zeros_like(t)
    for i in range(t.shape[0]):
        start = max(0, i - window + 1)
        out[i] = t[start:i+1].mean(dim=0)
    return out


def pairwise_interactions(X: torch.Tensor) -> torch.Tensor:
    """
    Computes all pairwise interaction cross-terms (i * j) 
    for all feature combinations without squared self-terms (i * i).
    
    Returns:
        torch.Tensor: (n_samples, C(n_features, 2)) interaction matrix.
    """
    n_feat = X.shape[1]
    if n_feat < 2:
        return torch.empty((X.shape[0], 0), dtype=X.dtype, device=X.device)
        
    cols = []
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            cols.append(X[:, i] * X[:, j])
            
    return torch.stack(cols, dim=1)

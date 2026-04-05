import torch
import torch.nn.functional as F


def winsorise(X: torch.Tensor, lower: float = 0.05, upper: float = 0.95) -> torch.Tensor:
    """
    Winsorise a feature tensor by clipping values to the given percentiles.
    This limits the effect of extreme outliers without removing data points.
    """
    if X.numel() == 0:
        return X
        
    # Calculate quantiles along dimension 0 (batch/samples)
    lo = torch.quantile(X, lower, dim=0)
    hi = torch.quantile(X, upper, dim=0)
    return torch.clamp(X, min=lo, max=hi)


def log_transform_positive(X: torch.Tensor) -> torch.Tensor:
    """
    Log-transform heavy-tailed positive features.
    Safely handles negative values by capping at 0 before applying log1p:
    log(1 + max(0, x)).
    """
    return torch.log1p(F.relu(X))


def mark_outliers_as_nan(X: torch.Tensor, labels: torch.Tensor, outlier_label: int = -1) -> torch.Tensor:
    """
    Mark detected outliers as NaN so they can be imputed later.
    
    Args:
        X (torch.Tensor): The feature tensor.
        labels (torch.Tensor): An array of anomaly labels returned by IsolationForest or LOF.
        outlier_label (int): The label value signifying an outlier. Defaults to -1.
        
    Returns:
        torch.Tensor: A clone of X with outlier rows set to NaN.
    """
    mask_outlier = (labels == outlier_label)
    X_marked = X.clone()
    X_marked[mask_outlier] = float("nan")
    return X_marked

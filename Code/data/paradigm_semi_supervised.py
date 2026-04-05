"""
Phase 5: Paradigm Semi-Supervised Preprocessing.
Executes mathematical contrastive masking frameworks (SimCLR / NT-Xent) alongside consistency regularizations.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.nn as nn


# -----------------------------------------------------------------------------
# 1. CONTRASTIVE DATA AUGMENTATIONS AND BATCHING (SimCLR)
# -----------------------------------------------------------------------------

class ContrastiveDataset(Dataset):
    """
    Structural PyTorch wrapper returning disjoint views explicitly evaluating representations identically
    bounding the sequence inputs across stochastic transforms securely without modifying original images.
    """
    def __init__(self, images: list, transform):
        """
        Args:
            images: Source arrays representing structural matrices (Paths, PIL, or Tensors)
            transform: Callable generator executing stochastic mutations randomly explicitly per-call.
        """
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        x = self.images[idx]
        # Return two independent views structurally bounding mappings dynamically.
        return self.transform(x), self.transform(x)


def contrastive_loss_ntxent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    NT-Xent (InfoNCE) symmetrical loss generator bounding vectors explicitly identically isolating 
    representation distances mapped to normalized structural geometries across continuous batch lengths (2B).
    
    Args:
        z1: (B, D) PyTorch representations natively.
        z2: (B, D) Correlated augmented bounds natively.
    """
    B = z1.shape[0]
    
    # Concatenation and L2 geometric projection
    z = torch.cat([F.normalize(z1, dim=1), F.normalize(z2, dim=1)], dim=0) # (2B, D)
    
    # Cosine topological representations scaled via Temperature scalar implicitly
    sim = (z @ z.T) / temperature # (2B, 2B)
    
    # Mask Identity symmetric bounds safely out evaluating boundaries cleanly avoiding trivial limits.
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -1e9) 
    
    # Shift correlation matrix arrays pointing exact label bounds explicitly mapping z1 -> z2 and z2 -> z1 symmetrically
    labels = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    
    return F.cross_entropy(sim, labels)


# -----------------------------------------------------------------------------
# 2. CONSISTENCY REGULARIZATION EXTENSIONS (FixMatch style)
# -----------------------------------------------------------------------------

def consistency_loss(X_unlab: torch.Tensor, augment_fn, model: nn.Module, threshold: float = 0.90) -> torch.Tensor:
    """
    Mathematical Weak-to-Strong confidence threshold bounds. Model representations map 
    across topological structural gaps securely extracting gradient gradients strictly mapping boundaries.
    """
    with torch.no_grad():
        weak_logits = model(augment_fn(X_unlab, weak=True))
        pseudo = torch.softmax(weak_logits, dim=1)
        conf, label = pseudo.max(dim=1)
        mask = conf >= threshold
    
    # Skip calculations gracefully returning zeroes identically avoiding exception structural loops natively if masked empty
    if not mask.any():
        return torch.tensor(0.0, device=X_unlab.device, requires_grad=True)
        
    strong_logits = model(augment_fn(X_unlab, weak=False))
    loss_vector = F.cross_entropy(strong_logits, label, reduction="none")
    
    # Only limit to gradients safely mapped by confidence thresholds boundary
    return loss_vector[mask].mean()

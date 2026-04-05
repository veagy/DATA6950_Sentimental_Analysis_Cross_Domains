"""
GatingNetwork: Routes inputs among N experts dynamically.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ...models.utils import DLModule
except ImportError:
    from ...models.utils.utils import DLModule

class GatingNetwork(DLModule):
    """
    Learns dynamic weighting for N expert models.
    Takes latent representation features in, projects to num_experts.
    """
    
    def __init__(
        self,
        in_features: int,
        num_experts: int,
        hidden_dim: Optional[int] = None,
        sparse_top_k: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_features = in_features
        self.num_experts = num_experts
        self.sparse_top_k = sparse_top_k
        
        if hidden_dim:
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_experts)
            )
        else:
            self.net = nn.Linear(in_features, num_experts)
            
    def _get_system_pipeline(self):
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features, typically representation from HRM.
            
        Returns:
            torch.Tensor: Softmax probabilities over experts. Shape: (batch_size, num_experts)
        """
        logits = self.net(x)
        if self.sparse_top_k is not None and self.sparse_top_k < logits.size(-1):
            k = self.sparse_top_k
            top_v, top_i = torch.topk(logits, k, dim=-1)
            masked = torch.full_like(logits, float("-inf"))
            masked.scatter_(-1, top_i, top_v)
            logits = masked
        weights = F.softmax(logits, dim=-1)
        return weights
        
    def get_class_type(self) -> str:
        return "GatingNetwork"

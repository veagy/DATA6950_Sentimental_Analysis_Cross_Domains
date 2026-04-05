"""
ParallelMoE: Mixture-of-Experts processing experts concurrently via GatingNetwork.
"""

from typing import Dict, Optional, Union
import torch
import torch.nn as nn

try:
    from ...models.utils import DLModule
except ImportError:
    from ...models.utils.utils import DLModule

from .gating import GatingNetwork

class ParallelMoE(DLModule):
    """
    Executes multiple expert modules in parallel and dynamically weights their
    outputs using a GatingNetwork based on standard input representations.
    """
    def __init__(
        self, 
        experts: nn.ModuleDict, 
        in_features: int, 
        hrm: Optional[nn.Module] = None,
        gating_hidden_dim: Optional[int] = None,
        **kwargs
    ):
        """
        Args:
            experts (nn.ModuleDict): The dictionary of expert modules.
            in_features (int): Input feature size for the gating network.
            hrm (nn.Module, optional): Hierarchical Reasoning layer or feature extractor.
            gating_hidden_dim (int, optional): Hidden units for gating network.
        """
        super().__init__(**kwargs)
        
        if not isinstance(experts, nn.ModuleDict):
            experts = nn.ModuleDict(experts)
            
        self.experts = experts
        self.num_experts = len(experts)
        self.expert_keys = list(experts.keys())
        
        self.hrm = hrm
        self.gating_network = GatingNetwork(
            in_features=in_features,
            num_experts=self.num_experts,
            hidden_dim=gating_hidden_dim
        )

    def _get_system_pipeline(self):
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass executing Parallel Mixture of Experts.
        
        Args:
            x (torch.Tensor): Input features.
            
        Returns:
            torch.Tensor: Dynamically weighted expert predictions.
        """
        # Base representation layer (if provided)
        if self.hrm is not None:
            features = self.hrm(x)
        else:
            features = x
            
        # 1. Gating mechanism distribution
        # Shape: (batch_size, num_experts)
        gating_weights = self.gating_network(features)
        
        # 2. Parallel distinct experts evaluation
        expert_outputs = []
        for name, expert in self.experts.items():
            out = expert(x)
            expert_outputs.append(out)
            
        # 3. Stack and Combine
        # expert_outputs stack to -> (batch_size, num_experts, output_dim)
        stacked_outputs = torch.stack(expert_outputs, dim=1)
        
        # gating_weights unsqueeze to -> (batch_size, num_experts, 1)
        expanded_weights = gating_weights.unsqueeze(-1)
        
        # Multiply weights by outputs and sum over experts dimension
        # (batch_size, num_experts, output_dim) * (batch_size, num_experts, 1) -> sum(dim=1) -> (batch_size, output_dim)
        final_output = torch.sum(stacked_outputs * expanded_weights, dim=1)
        
        return final_output
        
    def get_class_type(self) -> str:
        return "ParallelMoE"

"""
SequentialMoE: Cascade of experts passing inputs sequentially until threshold is met.
"""

from typing import List, Callable, Optional, Union, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ...models.utils import DLModule
except ImportError:
    from ...models.utils.utils import DLModule

class SequentialMoE(DLModule):
    """
    Evaluates inputs sequentially through experts. Allows early exit mechanism 
    when a specific expert confidence threshold is met. Great for efficiency.
    """
    
    def __init__(
        self,
        experts: nn.ModuleList,
        thresholds: Union[float, List[float]],
        confidence_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        **kwargs
    ):
        """
        Args:
            experts (nn.ModuleList): Ordered list of experts to run sequentially.
            thresholds (float or List[float]): Confidence thresholds for early exit.
            confidence_fn (callable, optional): Method to calculate confidence from 
                                                expert output. Defaults to max(softmax).
        """
        super().__init__(**kwargs)
        
        if not isinstance(experts, nn.ModuleList):
            experts = nn.ModuleList([e for e in experts])
            
        self.experts = experts
        self.num_experts = len(experts)
        
        if isinstance(thresholds, float):
            self.thresholds = [thresholds] * (self.num_experts - 1)
        else:
            self.thresholds = thresholds
            
        # Default confidence: max value of softmax probability
        if confidence_fn is None:
            self.confidence_fn = lambda x: torch.max(F.softmax(x, dim=-1), dim=-1)[0]
        else:
            self.confidence_fn = confidence_fn

    def _get_system_pipeline(self):
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sequential execution logic. Optimized for batch-wise early exiting.
        Note: True batch-wise early exiting splits the tensor, maintaining index mapping
        is complex, so here we use an accumulating logic or evaluate all non-exited.
        """
        batch_size = x.size(0)
        final_output = torch.zeros(
            (batch_size, self.experts[0](x[:1]).size(-1)), 
            dtype=x.dtype, device=x.device
        )
        
        # Track which items in the batch are still pending
        active_indices = torch.arange(batch_size, device=x.device)
        
        for i, expert in enumerate(self.experts):
            if len(active_indices) == 0:
                break
                
            x_active = x[active_indices]
            
            # Forward pass through this expert
            out = expert(x_active)
            
            if i == self.num_experts - 1:
                # Last expert handles remaining unconditionally
                final_output[active_indices] = out
                break
                
            # Confidence check
            conf = self.confidence_fn(out)
            
            # Mask of items that meet the threshold to exit early
            thresh = self.thresholds[i] if i < len(self.thresholds) else 0.0
            mask_exit = conf >= thresh
            
            # Indices relative to current active_indices that are exiting
            idx_exit_rel = torch.where(mask_exit)[0]
            # Indices relative to global batch that are exiting
            idx_exit_global = active_indices[idx_exit_rel]
            
            if len(idx_exit_global) > 0:
                final_output[idx_exit_global] = out[idx_exit_rel]
            
            # Keep items that failed to meet the threshold
            active_indices = active_indices[~mask_exit]
            
        return final_output
        
    def get_class_type(self) -> str:
        return "SequentialMoE"

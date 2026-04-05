"""
ExpertWrapper: Adapts scikit-learn or custom ML models to PyTorch nn.Module.
"""

from typing import Any
import torch
import torch.nn as nn

try:
    from ...models.utils import DLModule
except ImportError:
    from ...models.utils.utils import DLModule

class ExpertWrapper(DLModule):
    """
    Wraps traditional Machine Learning models (e.g., scikit-learn SVM, Logistic Regression)
    so they can be passed as PyTorch nn.Module experts in an MoE pipeline.
    """
    
    def __init__(self, ml_model: Any, out_features: int, **kwargs):
        """
        Args:
            ml_model: The underlying ML model instance (must implement .predict or .predict_proba).
            out_features: The expected output dimensionality.
        """
        super().__init__(**kwargs)
        self.ml_model = ml_model
        self.out_features = out_features
        # PyTorch requires parameters for gradient tracking; dummy parameter for state dicts
        self.dummy_param = nn.Parameter(torch.empty(0))

    def _get_system_pipeline(self):
        """Fallback for system-linked deep models."""
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes the ML model without computing gradients.
        
        Args:
            x (torch.Tensor): Input features.
            
        Returns:
            torch.Tensor: ML model predictions cast to tensor.
        """
        # Execute ML model outside autograd
        with torch.no_grad():
            x_np = x.detach().cpu().numpy()
            
            # Use predict_proba if available, otherwise predict
            if hasattr(self.ml_model, 'predict_proba'):
                try:
                    out = self.ml_model.predict_proba(x_np)
                except Exception:
                    out = self.ml_model.predict(x_np)
            elif hasattr(self.ml_model, 'predict'):
                out = self.ml_model.predict(x_np)
            else:
                out = self.ml_model(x_np) # Callable
                
            out_tensor = torch.tensor(out, dtype=x.dtype, device=x.device)
            
            # Ensure correct output features matching if necessary (flatten or reshape)
            if out_tensor.dim() == 1:
                out_tensor = out_tensor.unsqueeze(1)
                
            return out_tensor
            
    def get_class_type(self) -> str:
        return "ExpertWrapper"

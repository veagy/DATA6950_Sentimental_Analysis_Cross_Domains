import torch
from typing import List, Tuple, Any, Dict, Union
from ..data.scaling import MLModule

class Pipeline(MLModule):
    """
    A Sequential Pipeline of transforms with a final estimator, identical to 
    sklearn.pipeline.Pipeline but operating sequentially on torch Tensors.
    """
    def __init__(self, steps: List[Tuple[str, Any]], memory=None, verbose=False, device="cpu", dtype=torch.float, *args, **kwargs):
        super().__init__()
        self.steps = steps
        self.memory = memory
        self.verbose = verbose
        self.device = device
        self.dtype = dtype
        
        # Unpack steps
        self.named_steps = {name: transform for name, transform in self.steps}
        
    def _validate_steps(self):
        names, estimators = zip(*self.steps)
        if len(set(names)) != len(names):
            raise ValueError("Pipeline step names must be unique.")
            
    def fit(self, X: torch.Tensor, y: torch.Tensor = None, **fit_params):
        """Fit all transforms sequentially and final estimator."""
        self._validate_steps()
        X_t = X
        
        for name, transformer in self.steps[:-1]:
            if transformer is None or transformer == 'passthrough':
                continue
                
            # Filter kwargs meant for this step
            step_params = {k.split('__')[1]: v for k, v in fit_params.items() if k.startswith(f"{name}__")}
            
            if hasattr(transformer, "fit_transform"):
                X_t = transformer.fit_transform(X_t, y, **step_params)
            else:
                X_t = transformer.fit(X_t, y, **step_params).transform(X_t)
                
        # Fit final step
        final_name, final_est = self.steps[-1]
        if final_est is not None and final_est != 'passthrough':
            step_params = {k.split('__')[1]: v for k, v in fit_params.items() if k.startswith(f"{final_name}__")}
            final_est.fit(X_t, y, **step_params)
            
        self.fit_status = True
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Apply transforms sequentially."""
        X_t = X
        for name, transformer in self.steps:
            if transformer is None or transformer == 'passthrough':
                continue
            if hasattr(transformer, "transform"):
                X_t = transformer.transform(X_t)
        return X_t

    def fit_transform(self, X: torch.Tensor, y: torch.Tensor = None, **fit_params) -> torch.Tensor:
        """Fit all transforms and return identically transformed output."""
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def predict(self, X: torch.Tensor, **predict_params) -> torch.Tensor:
        """Transform data and predict with final estimator."""
        X_t = X
        for name, transformer in self.steps[:-1]:
            if transformer is not None and transformer != 'passthrough':
                X_t = transformer.transform(X_t)
                
        final_name, final_est = self.steps[-1]
        if final_est is None or final_est == 'passthrough':
            return X_t
            
        return final_est.predict(X_t, **predict_params)

    def inverse_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Apply inverse_transform sequentially in reverse order."""
        X_t = X
        for name, transformer in reversed(self.steps):
            if transformer is None or transformer == 'passthrough':
                continue
            if not hasattr(transformer, "inverse_transform"):
                raise ValueError(f"Transformer {name} does not implement inverse_transform.")
            X_t = transformer.inverse_transform(X_t)
            
        return X_t
    
    def forward(self, X: torch.Tensor, **kwargs) -> torch.Tensor:
        if hasattr(self.steps[-1][1], "predict"):
            return self.predict(X, **kwargs)
        return self.transform(X)

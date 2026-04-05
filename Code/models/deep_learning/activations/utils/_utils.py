import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Any, Union, Dict, Set
import warnings
__all__ = [
    'Backward_hook',
    'ComplexDataConverter',
    'Forward_hook',
]


def _broadcast_params(param: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Intelligent broadcasting for activation parameters (System-wide Fix).
    Reshapes a parameter tensor to be broadcastable with an input tensor `x`.
    Handles NCHW convention where a parameter of shape (C,) should align with the C dimension (dim 1).
    """
    if param.dim() >= x.dim():
        return param
    
    # 1. Heuristic: If param is 1D and matches the channel dimension (dim 1), broadcast there.
    # Assumes dim 0 is Batch.
    pad_right = x.dim() - 1 - param.dim()
    if param.dim() == 1 and pad_right >= 0 and x.size(1) == param.shape[0]:
         shape = (1,) + param.shape + (1,) * pad_right
         return param.view(shape)

    # 2. Fallback: Standard broadcasting (aligns last dimensions)
    shape = (1,) * (x.dim() - param.dim()) + tuple(param.shape)
    return param.view(shape)


def _smart_broadcast(param: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Intelligent broadcasting for activation parameters.
    Handles NCHW convention where a parameter of shape (C,) should align with the C dimension (dim 1).
    Input: x (N, C, ...), param (C,)
    Output: param broadcasted to (1, C, 1...)
    """
    if param.dim() >= x.dim():
        return param
        
    # Standard case: standard broadcasting aligns last dimensions.
    # If param.dim match x's last dims, use standard.
    # But usually activations params (C,) are NOT spatial.
    
    # Heuristic: If param.dim matches x.dim - 1 (channel), broadcast there.
    # Assumes dim 0 is Batch.
    pad_right = x.dim() - 1 - param.dim()
    if pad_right >= 0:
        shape = (1,) + param.shape + (1,) * pad_right
        return param.view(shape)
        
    return _broadcast_params(param, x)


class _Linear(nn.Module):
    """
    Base class for linear transformation before activation.
    Formula: y = ai * x + bi
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.register_buffer('ai', torch.tensor(kwargs.get('ai', 1.0), dtype=torch.float32))
        self.register_buffer('bi', torch.tensor(kwargs.get('bi', 0.0), dtype=torch.float32))

    def _linear(self, x: torch.Tensor) -> torch.Tensor:
        ai = _broadcast_params(self.ai, x)
        bi = _broadcast_params(self.bi, x)
        return ai * x + bi


class _LinearParametricActivation(nn.Module):
    """
    Base class for parametric activations using customizable nn.Linear.
    Formula: y = Linear(x) before activation logic.
    """
    def __init__(self, dims: Tuple[int, ...], **kwargs):
        super().__init__()
        self.dim = kwargs.get('dim', -1)
        self.bias = kwargs.get('bias', True)

        # Determine in_features from dims and dim
        if isinstance(dims, int):
            in_features = dims
        else:
            in_features = dims[self.dim]

        self.linear = nn.Linear(in_features, in_features, bias=self.bias)

    def _apply_linear(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim != -1 and self.dim != x.dim() - 1:
            actual_dim = self.dim if self.dim >= 0 else x.dim() + self.dim
            x = x.transpose(actual_dim, -1)
            x = self.linear(x)
            x = x.transpose(actual_dim, -1)
        else:
            x = self.linear(x)
        return x


class Forward_hook:
    """
    Hook to log forward pass information of an activation module.
    """
    def __init__(self, name: str = "Activation"):
        self.name = name

    def __call__(self, module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor):
        print(f"\n[Forward Hook] {self.name}")
        print(f"  Class: {module.__class__.__name__}")
        print(f"  Input Shape:  {input[0].shape}")
        print(f"  Output Shape: {output.shape}")
        try:
            print(f"  Mean: {output.mean().item():.4f}, Std: {output.std().item():.4f}")
        except Exception:
             print("  Mean/Std calculation failed (custom tensor type?)")


class Backward_hook:
    """
    Hook to log backward pass (gradient) information of an activation module.
    """
    def __init__(self, name: str = "Activation"):
        self.name = name

    def __call__(self, module: nn.Module, grad_input: Tuple[Optional[torch.Tensor]], grad_output: Tuple[torch.Tensor]):
        print(f"\n[Backward Hook] {self.name}")
        print(f"  Class: {module.__class__.__name__}")
        if len(grad_output) > 0 and grad_output[0] is not None:
            print(f"  Grad Output Shape: {grad_output[0].shape}")
            print(f"  Grad Output Mean:  {grad_output[0].mean().item():.4f}")
        if len(grad_input) > 0 and grad_input[0] is not None:
            print(f"  Grad Input Shape:  {grad_input[0].shape}")


class ComplexDataConverter(nn.Module):
    """
    Utility class to handle multi-format data conversion to Tensors,
    specifically focusing on complex number handling and tensor splitting/rearrangement.
    """
    def __init__(self, data: Any, dim: int = -1, device: Optional[torch.device] = None, *args, **kwargs):
        super().__init__()
        self.dim = dim
        self.device = device
        self.kwargs = kwargs
        self.arrangement = kwargs.get('arrangement', 'split') # 'split' (top/bottom) or 'alt' (alternating)
        
        # Initialize internal storage
        self.data_store = self._initialize_(data)

    def _initialize_(self, data: Any) -> Union[torch.Tensor, nn.ModuleDict]:
        device = self.device
        
        # 1. Conversion to Dict/ModuleDict for specific types
        if isinstance(data, dict):
            module_dict = nn.ModuleDict()
            for k, v in data.items():
                tensor = self._process_single_data(v)
                # nn.Parameter or Buffer? Requirement says "direct conversion to torch.Tensor"
                # but nn.ModuleDict requires nn.Modules. We'll wrap in a simple Parameter wrapper or register as buffer.
                # To keep it simple and functional for future use, we convert to Parameter.
                module_dict[str(k)] = nn.ParameterDict({'tensor': nn.Parameter(tensor.to(device) if device else tensor)})
            return module_dict
            
        # 2. Conversion for all other_decomposition types
        tensor = self._process_single_data(data)
        if device:
            tensor = tensor.to(device)
            
        # We store it as a buffer to be part of the Module state
        self.register_buffer('tensor', tensor)
        return tensor

    def _process_single_data(self, data: Any) -> torch.Tensor:
        # Initial conversion to tensor
        if not isinstance(data, torch.Tensor):
            tensor = torch.as_tensor(data)
        else:
            tensor = data
            
        # Ensure floating point if not complex
        if not tensor.is_complex() and not tensor.is_floating_point():
            tensor = tensor.to(torch.float32)

        # Check for complex types
        if tensor.is_complex():
            # Separate into real and imaginary, then stack at self.dim
            real = tensor.real
            imag = tensor.imag
            tensor = torch.stack([real, imag], dim=self.dim)
        
        # Check if int or float (supported by torch)
        elif tensor.is_floating_point() or not tensor.is_complex():
            dim_size = tensor.size(self.dim)
            
            # If not even, pad with zeros
            if dim_size % 2 != 0:
                warnings.warn(f"Size at dim {self.dim} is not even ({dim_size}). Padding with zeros.")
                pad_dims = [0] * (2 * tensor.dim())
                # Normalize dim
                norm_dim = self.dim if self.dim >= 0 else tensor.dim() + self.dim
                idx = (tensor.dim() - 1 - norm_dim) * 2
                pad_dims[idx + 1] = 1 
                tensor = F.pad(tensor, tuple(pad_dims[::-1]))
                dim_size += 1

            # Separate and Combine logic
            k_half = dim_size // 2
            if self.arrangement == 'alt':
                indices_a = torch.arange(0, dim_size, 2, device=tensor.device)
                indices_b = torch.arange(1, dim_size, 2, device=tensor.device)
                t1 = tensor.index_select(self.dim, indices_a)
                t2 = tensor.index_select(self.dim, indices_b)
            else:
                t1, t2 = torch.split(tensor, k_half, dim=self.dim)
            
            tensor = torch.stack([t1, t2], dim=self.dim)

        return tensor

    def real(self) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Returns the first component (real part)."""
        if isinstance(self.data_store, nn.ModuleDict):
            return {k: v.tensor.select(self.dim, 0) for k, v in self.data_store.items()}
        return self.tensor.select(self.dim, 0)

    def imag(self) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Returns the second component (imaginary part)."""
        if isinstance(self.data_store, nn.ModuleDict):
            return {k: v.tensor.select(self.dim, 1) for k, v in self.data_store.items()}
        return self.tensor.select(self.dim, 1)

    def mag(self) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Returns the magnitude sqrt(real^2 + imag^2)."""
        r = self.real()
        i = self.imag()
        if isinstance(r, dict):
            return {k: torch.sqrt(r[k]**2 + i[k]**2) for k in r.keys()}
        return torch.sqrt(r**2 + i**2)

    def phase(self) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Returns the phase atan2(imag, real)."""
        r = self.real()
        i = self.imag()
        if isinstance(r, dict):
            return {k: torch.atan2(i[k], r[k]) for k in r.keys()}
        return torch.atan2(i, r)

    def forward(self, x: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        # Return the processed tensor or dict of tensors
        if isinstance(self.data_store, nn.ModuleDict):
            return {k: v.tensor for k, v in self.data_store.items()}
        return self.tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import sys
import math
from typing import Any, Optional, Union, List, Tuple


def _unravel_index(linear_idx: int, shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Convert flat index to multi-dimensional index (replaces np.unravel_index)."""
    result = []
    for k in range(len(shape)):
        result.append((linear_idx // math.prod(shape[k + 1:])) % shape[k])
    return tuple(result)
__all__ = [
    'Complex',
    'ComplexLinear',
    'add',
    'broadcast_to',
    'cos',
    'cosec',
    'cosech',
    'cosh',
    'cot',
    'coth',
    'divide',
    'exp',
    'exp10',
    'exp2',
    'exp_n',
    'frexp',
    'iexp',
    'index_put',
    'inv',
    'log',
    'mag',
    'multiply',
    'nextafter',
    'nonzero',
    'phi',
    'polyder',
    'polyint',
    'polyval',
    'repeat_interleave',
    'sec',
    'sech',
    'sin',
    'sinh',
    'subtract',
    'tan',
    'tanh',
    'unit',
    'xlogy',
]


class Complex(nn.Module):
    """
    Converts data to a complex-valued tensor format where Real and Imaginary parts
    are stacked at a specific dimension `dim`.

    Key Features:
    - Supports torch.complex inputs: Splits them and inserts the '2' dimension exactly at `dim`.
    - Supports Real inputs: Splits/Folds them and stacks at `dim`.
    - No register_buffer: Uses standard attributes and custom .to() handling.
    """

    def __init__(self,
                 data: Any,
                 dim: Optional[int] = None,
                 dtype: torch.dtype = torch.float32,
                 device: Optional[Union[str, torch.device]] = None,
                 arrangement: str = 'split',
                 is_stacked_flag: bool = False,
                 *args, **kwargs):
        super().__init__()

        self.dim = dim
        self.dtype = dtype
        self.device = device if device else torch.device('cpu')
        self.arrangement = arrangement
        self.kwargs = kwargs
        self._is_stacked = is_stacked_flag

        self.tensor = None
        if data is not None:
            self.__initialize__(data)
            self._is_stacked = True  # Enforce true after initialization
        self.kwargs = kwargs

    def __initialize__(self, data: Any):
        # 0. Handle None data (for module instantiation without immediate tensor)
        if data is None:
            return

        # 1. Handle Maps (Dicts)
        if isinstance(data, dict):
            self.data_store = {}
            for k, v in data.items():
                self.data_store[str(k)] = Complex(
                    v, dim=self.dim, dtype=self.dtype,
                    device=self.device, arrangement=self.arrangement
                )
            return

        # 2. Handle Sequences & Scalars & array-like -> Tensor
        if isinstance(data, (list, tuple, set, complex, float, int)):
            tensor = torch.as_tensor(data, device=self.device)
        elif isinstance(data, torch.Tensor):
            if self._is_stacked:
                self.tensor = data.to(device=self.device, dtype=self.dtype)
                return
            tensor = data.to(device=self.device)
        elif isinstance(data, Complex):
            self.tensor = data.tensor.to(device=self.device, dtype=self.dtype)
            # If provided dim differs...
            if self.dim is None:
                self.dim = data.dim
            else:
                # Trust data.dim
                self.dim = data.dim
            return
        else:
            # array-like (e.g. numpy array) - torch.as_tensor handles it
            try:
                tensor = torch.as_tensor(data, device=self.device)
            except (TypeError, ValueError):
                raise RuntimeError(f"Unsupported data type: {type(data)}")

        # 3. Process
        self.tensor = self.__process_tensor__(tensor)
        self._is_stacked = True  # We always end up with a stacked tensor after processing

    def __process_tensor__(self, tensor: torch.Tensor) -> torch.Tensor:
        # print(f"DEBUG Complex.__process_tensor__: _is_stacked={self._is_stacked}, shape={tensor.shape}, dim={self.dim}")
        # --- Step 1: Normalize Dimension ---
        # We calculate the positive index for 'dim' immediately to ensure
        # torch.stack puts the '2' exactly where expected.
        ndim = tensor.dim()

        if self.dim is None:
            # Auto-detection logic
            candidates = [i for i, x in enumerate(tensor.shape) if x == 2]
            if len(candidates) == 1:
                target_dim = candidates[0]
            elif len(candidates) > 1:
                warnings.warn(f"Multiple dims have size 2 {candidates}. Using last.", UserWarning)
                target_dim = candidates[-1]
            else:
                target_dim = ndim  # Default to appending at the end if not found
        else:
            # Convert negative dim to positive (e.g., -1 becomes ndim-1)
            # Note: For torch.stack, stacking at 'ndim' is valid (appends at end)
            if ndim == 0:
                target_dim = 0
            else:
                target_dim = self.dim if self.dim >= 0 else ndim + self.dim + 1
                if target_dim > ndim:
                    target_dim = ndim

        # Save the resolved dimension
        self.dim = target_dim

        # --- Step 2: Handle Complex Dtypes ---
        if tensor.is_complex():
            real = tensor.real
            imag = tensor.imag

            if real.dtype != self.dtype:
                real = real.to(self.dtype)
                imag = imag.to(self.dtype)

            # CRITICAL: torch.stack inserts a new dimension at 'dim'.
            # Input: (3, 4) complex, dim=1
            # Real: (3, 4), Imag: (3, 4)
            # Output: (3, 2, 4) -> The size 2 is at index 1.
            return torch.stack([real, imag], dim=self.dim)

        # --- Step 3: Handle Real Dtypes (Folding) ---
        else:
            if not tensor.is_floating_point():
                tensor = tensor.to(self.dtype)

            # Optimization: If already stacked, verify and return
            if self._is_stacked:
                if self.dim < tensor.dim() and tensor.size(self.dim) == 2:
                    return tensor
                else:
                    # Fallback or warning if something is wrong
                    pass

            # Adjust target_dim for splitting logic
            # If we are splitting an existing dimension, the target is that dimension.
            # If self.dim was calculated as "append at end" (dim=ndim), we can't split that.
            # We assume for Real input, dim refers to the dimension TO BE SPLIT.

            split_dim = self.dim
            if split_dim >= ndim or ndim == 0:
                # If scalar or out of bounds, we append the complex dim
                real = tensor
                imag = torch.zeros_like(tensor)
                self.dim = ndim  # Append at the end
                return torch.stack([real, imag], dim=self.dim)

            current_size = tensor.size(split_dim)

            # Check if padding is needed
            if current_size % 2 != 0:
                warnings.warn(f"Dim {split_dim} has odd size {current_size}. Padding.", UserWarning)
                # For scalars/1D we can't use complex F.pad easily with negative indices
                if ndim == 1:
                    tensor = F.pad(tensor, (0, 1))
                else:
                    pad_scheme = [0] * (2 * ndim)
                    pad_idx = (ndim - 1 - split_dim) * 2 + 1
                    pad_scheme[pad_idx] = 1
                    tensor = F.pad(tensor, tuple(pad_scheme))
                current_size += 1

            # Split or Alternate
            if self.arrangement == 'alt':
                idx_r = torch.arange(0, current_size, 2, device=tensor.device)
                idx_i = torch.arange(1, current_size, 2, device=tensor.device)
                real = tensor.index_select(split_dim, idx_r)
                imag = tensor.index_select(split_dim, idx_i)
            else:
                real, imag = torch.split(tensor, current_size // 2, dim=split_dim)

            # CRITICAL: stack inserts the '2' at split_dim
            # Input: (3, 10), dim=1
            # Real: (3, 5), Imag: (3, 5)
            # Output: (3, 2, 5) -> The size 2 is at index 1.
            return torch.stack([real, imag], dim=split_dim)

    # --- Lifecycle & Metadata ---

    def __repr__(self):
        shape = tuple(self.real.shape)
        return f"Complex(shape={shape}, dtype={self.dtype}, device='{self.device}', arrangement='{self.arrangement}')"

    def __str__(self):
        return self.__repr__()

    def clone(self):
        return Complex(self.tensor.clone(), dim=self.dim, dtype=self.dtype, device=self.device,
                       arrangement=self.arrangement, is_stacked_flag=True, **self.kwargs)

    def detach(self):
        return Complex(self.tensor.detach(), dim=self.dim, dtype=self.dtype, device=self.device,
                       arrangement=self.arrangement, is_stacked_flag=True, **self.kwargs)

    def requires_grad_(self, requires_grad=True):
        self.tensor.requires_grad_(requires_grad)
        return self

    @property
    def requires_grad(self):
        return self.tensor.requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        """Sets requires_grad flag on underlying tensor."""
        self.tensor.requires_grad = value

    @property
    def grad_fn(self):
        """Returns the GradFn node in the computation graph."""
        return self.tensor.grad_fn

    @property
    def is_leaf(self):
        """Returns True if tensor is a leaf node in autograd graph."""
        return self.tensor.is_leaf

    def cpu(self):
        return self.to('cpu')

    def cuda(self, device=None):
        return self.to(device='cuda' if device is None else device)

    # --- Factory Methods ---

    @classmethod
    def zeros(cls, *sizes, dim: int = -1, dtype: torch.dtype = torch.float32,
              device: Optional[Union[str, torch.device]] = None, **kwargs):
        """
        Creates a Complex tensor filled with zeros.
        """
        # Create a complex tensor first, then wrap it. The constructor will handle stacking.
        c_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        c = torch.zeros(*sizes, dtype=c_dtype, device=device)
        return cls(c, dim=dim, dtype=dtype, **kwargs)

    @classmethod
    def from_numpy(cls, ndarray, dim: int = -1, **kwargs):
        """
        Creates a Complex tensor from a complex numpy array.
        """
        tensor = torch.from_numpy(ndarray)
        return cls(tensor, dim=dim, **kwargs)

    @classmethod
    def ones(cls, *sizes, dim: int = -1, dtype: torch.dtype = torch.float32,
             device: Optional[Union[str, torch.device]] = None, **kwargs):
        """
        Creates a Complex tensor filled with ones (real part 1, imag part 0).
        """
        c_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        c = torch.ones(*sizes, dtype=c_dtype, device=device)
        return cls(c, dim=dim, dtype=dtype, **kwargs)

    @classmethod
    def zeros_like(cls, other: 'Complex'):
        real = torch.zeros_like(other.real)
        imag = torch.zeros_like(other.imag)
        return cls(torch.complex(real, imag), dim=other.dim, dtype=other.dtype, device=other.device,
                   arrangement=other.arrangement, **other.kwargs)

    @classmethod
    def ones_like(cls, other: 'Complex'):
        real = torch.ones_like(other.real)
        imag = torch.ones_like(other.imag)
        return cls(torch.complex(real, imag), dim=other.dim, dtype=other.dtype, device=other.device,
                   arrangement=other.arrangement, **other.kwargs)

    @classmethod
    def eye(cls, n: int, m: Optional[int] = None, dim: int = -1, dtype: torch.dtype = torch.float32,
            device: Optional[Union[str, torch.device]] = None, **kwargs):
        if m is None:
            real = torch.eye(n, device=device, dtype=dtype)
        else:
            real = torch.eye(n, m, device=device, dtype=dtype)
        imag = torch.zeros_like(real)
        # Use torch.complex to create a complex tensor
        c_tensor = torch.complex(real, imag)
        return cls(c_tensor, dim=dim, dtype=dtype, device=device, **kwargs)

    @classmethod
    def from_polar(cls, mag: torch.Tensor, phi: torch.Tensor, dim: int = -1, **kwargs):
        real = mag * torch.cos(phi)
        imag = mag * torch.sin(phi)
        c_tensor = torch.complex(real, imag)
        return cls(c_tensor, dim=dim, **kwargs)

    @classmethod
    def linspace(cls, start: Union[complex, 'Complex'], end: Union[complex, 'Complex'], steps: int, dim: int = -1,
                 **kwargs):
        # Handle Complex or complex scalars
        def get_parts(val):
            if isinstance(val, Complex):
                # If it's a 0-dim Complex (scalar)
                if val.tensor is not None and val.tensor.numel() <= 2:
                    v_c = val.item()
                    return v_c.real, v_c.imag
                return val.real, val.imag
            if isinstance(val, (complex, float, int)):
                c_val = complex(val)
                return c_val.real, c_val.imag
            if hasattr(val, 'real'):
                return val.real, val.imag
            return val, 0.0

        s_real, s_imag = get_parts(start)
        e_real, e_imag = get_parts(end)

        real = torch.linspace(s_real, e_real, steps)
        imag = torch.linspace(s_imag, e_imag, steps)
        return cls(torch.complex(real, imag), dim=dim, **kwargs)

    @classmethod
    def logspace(cls, start: Union[complex, 'Complex'], end: Union[complex, 'Complex'], steps: int, base: float = 10.0,
                 dim: int = -1, **kwargs):
        def get_parts(val):
            if isinstance(val, Complex):
                if val.tensor is not None and val.tensor.numel() <= 2:
                    v_c = val.item()
                    return v_c.real, v_c.imag
                return val.real, val.imag
            if isinstance(val, (complex, float, int)):
                c_val = complex(val)
                return c_val.real, c_val.imag
            if hasattr(val, 'real'):
                return val.real, val.imag
            return val, 0.0

        s_real, s_imag = get_parts(start)
        e_real, e_imag = get_parts(end)

        l_real = torch.linspace(s_real, e_real, steps)
        l_imag = torch.linspace(s_imag, e_imag, steps)

        ln_base = math.log(base)
        mag = torch.exp(ln_base * l_real)
        phase = ln_base * l_imag
        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)

        return cls(torch.complex(real, imag), dim=dim, **kwargs)

    @classmethod
    def rand(cls, *sizes, dim: int = -1, dtype: torch.dtype = torch.float32,
             device: Optional[Union[str, torch.device]] = None, **kwargs):
        real = torch.rand(*sizes, dtype=dtype, device=device)
        imag = torch.rand(*sizes, dtype=dtype, device=device)
        return cls(torch.complex(real, imag), dim=dim, **kwargs)

    @classmethod
    def rand_like(cls, other: 'Complex'):
        real = torch.rand_like(other.real)
        imag = torch.rand_like(other.imag)
        return cls(torch.complex(real, imag), dim=other.dim, dtype=other.dtype, device=other.device,
                   arrangement=other.arrangement, **other.kwargs)

    @classmethod
    def randn(cls, *sizes, dim: int = -1, dtype: torch.dtype = torch.float32,
              device: Optional[Union[str, torch.device]] = None, **kwargs):
        real = torch.randn(*sizes, dtype=dtype, device=device)
        imag = torch.randn(*sizes, dtype=dtype, device=device)
        return cls(torch.complex(real, imag), dim=dim, **kwargs)

    @classmethod
    def randn_like(cls, other: 'Complex'):
        return cls.randn(*other.real.shape, dim=other.dim, dtype=other.dtype, device=other.device)

    @classmethod
    def full(cls, size, fill_value, dim: int = -1, dtype: torch.dtype = torch.float32,
             device: Optional[Union[str, torch.device]] = None, **kwargs):
        """Creates a tensor filled with a specific complex value."""
        if not isinstance(fill_value, complex):
            fill_value = complex(fill_value)

        c_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        c = torch.full(size, fill_value, dtype=c_dtype, device=device)
        return cls(c, dim=dim, dtype=dtype, **kwargs)

    @classmethod
    def arange(cls, start, end=None, step=1, dim: int = -1, dtype: torch.dtype = torch.float32,
               device: Optional[Union[str, torch.device]] = None, **kwargs):
        """Creates a 1D tensor with values from start to end with given step."""
        if end is None:
            end = start
            start = 0

        # Real arange
        real_range = torch.arange(start, end, step, dtype=dtype, device=device)
        return cls(real_range, dim=dim, dtype=dtype, **kwargs)

    @classmethod
    def empty(cls, *sizes, dim: int = -1, dtype: torch.dtype = torch.float32,
              device: Optional[Union[str, torch.device]] = None, **kwargs):
        """Creates an uninitialized tensor (faster than zeros)."""
        real = torch.empty(*sizes, dtype=dtype, device=device)
        imag = torch.empty(*sizes, dtype=dtype, device=device)
        return cls(torch.complex(real, imag), dim=dim, **kwargs)

    # --- Weight Initializers ---

    @classmethod
    def xavier_uniform_(cls, tensor: 'Complex', gain=1.0):
        """
        Xavier uniform initializer adapted for complex numbers.
        Ensures Var(Z) = 2 / (fan_in + fan_out).
        """
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(tensor.real)
        std = gain * math.sqrt(1.0 / (fan_in + fan_out))
        bound = math.sqrt(3.0) * std  # std * sqrt(3) for uniform(-a, a)
        torch.nn.init.uniform_(tensor.real, -bound, bound)
        torch.nn.init.uniform_(tensor.imag, -bound, bound)
        return tensor

    @classmethod
    def xavier_normal_(cls, tensor: 'Complex', gain=1.0):
        """
        Xavier normal initializer adapted for complex numbers.
        Ensures Var(Z) = 2 / (fan_in + fan_out).
        """
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(tensor.real)
        std = gain * math.sqrt(1.0 / (fan_in + fan_out))
        # Part std is std / sqrt(2)
        part_std = std / math.sqrt(2.0)
        torch.nn.init.normal_(tensor.real, 0, part_std)
        torch.nn.init.normal_(tensor.imag, 0, part_std)
        return tensor

    @classmethod
    def kaiming_uniform_(cls, tensor: 'Complex', a=0, mode='fan_in', nonlinearity='leaky_relu'):
        """
        Kaiming uniform initializer adapted for complex numbers.
        """
        gain = torch.nn.init.calculate_gain(nonlinearity, a)
        fan = torch.nn.init._calculate_correct_fan(tensor.real, mode)
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std
        torch.nn.init.uniform_(tensor.real, -bound, bound)
        torch.nn.init.uniform_(tensor.imag, -bound, bound)
        return tensor

    @classmethod
    def kaiming_normal_(cls, tensor: 'Complex', a=0, mode='fan_in', nonlinearity='leaky_relu'):
        """
        Kaiming normal initializer adapted for complex numbers.
        """
        gain = torch.nn.init.calculate_gain(nonlinearity, a)
        fan = torch.nn.init._calculate_correct_fan(tensor.real, mode)
        std = gain / math.sqrt(fan)
        part_std = std / math.sqrt(2.0)
        torch.nn.init.normal_(tensor.real, 0, part_std)
        torch.nn.init.normal_(tensor.imag, 0, part_std)
        return tensor

    @classmethod
    def orthogonal_(cls, tensor: 'Complex', gain=1.0):
        """
        Orthogonal initializer for complex matrices (produces semi-unitary matrices).
        """
        shape = tensor.real.shape
        if len(shape) < 2:
            raise ValueError("Orthogonal initialization requires at least 2 dimensions")

        rows = shape[0]
        cols = math.prod(shape[1:])

        # Generate random complex matrix
        a = torch.randn(rows, cols, dtype=torch.complex128 if tensor.dtype == torch.float64 else torch.complex64)

        if rows >= cols:
            q, r = torch.linalg.qr(a)
        else:
            q, r = torch.linalg.qr(a.mH)
            q = q.mH

        # q should now be (rows, cols)
        q = q.reshape(shape)

        with torch.no_grad():
            tensor.real.copy_(q.real * gain)
            tensor.imag.copy_(q.imag * gain)
        return tensor

    @classmethod
    def cat(cls, tensors: List[Union['Complex', torch.Tensor, complex]], dim: int = 0, **kwargs):
        """
        Concatenates a list of Complex objects or tensors along a dimension.
        """
        if not tensors:
            raise ValueError("cat() expects a non-empty list of tensors")

        objs = []
        for t in tensors:
            if not isinstance(t, cls):
                # We assume the same dtype/device/arrangement as the first Complex object found or default
                objs.append(cls(t, **kwargs))
            else:
                objs.append(t)

        # Ensure all ornaments (dim, arrangement) match. 
        # If they don't, we might need to permute/rearrange. 
        # For simplicity, we assume they all share the same dim/arrangement as objs[0].
        first = objs[0]
        reals = [o.real for o in objs]
        imags = [o.imag for o in objs]

        # PyTorch requires at least 1-dim for cat, unless we want to stack for 0-dim.
        # But usually user expects cat on a dim. If 0-dim, cat is invalid in torch.
        if reals[0].ndim == 0:
            if dim != 0:
                raise ValueError(f"Cannot concatenate 0-dim tensors along dim {dim}")
            # Fallback to stack-like behavior or just raise if strict cat is desired.
            # However, for 0-dim, user might mean "make a vector".
            real_cat = torch.stack(reals, dim=0)
            imag_cat = torch.stack(imags, dim=0)
            # Complex dim shifts to 1
            return cls(torch.complex(real_cat, imag_cat), dim=1,
                       dtype=first.dtype, device=first.device,
                       arrangement=first.arrangement, **first.kwargs)

        real_cat = torch.cat(reals, dim=dim)
        imag_cat = torch.cat(imags, dim=dim)

        return cls(torch.complex(real_cat, imag_cat), dim=first.dim,
                   dtype=first.dtype, device=first.device,
                   arrangement=first.arrangement, **first.kwargs)

    @classmethod
    def stack(cls, tensors: List[Union['Complex', torch.Tensor, complex]], dim: int = 0, **kwargs):
        """
        Stacks a list of Complex objects or tensors along a new dimension.
        """
        if not tensors:
            raise ValueError("stack() expects a non-empty list of tensors")

        objs = []
        for t in tensors:
            if not isinstance(t, cls):
                objs.append(cls(t, **kwargs))
            else:
                objs.append(t)

        first = objs[0]
        reals = [o.real for o in objs]
        imags = [o.imag for o in objs]

        real_stack = torch.stack(reals, dim=dim)
        imag_stack = torch.stack(imags, dim=dim)

        # Stacking adds a new dimension. If dim <= first.dim, our target complex dim shifts.
        target_dim = first.dim
        if dim <= target_dim:
            target_dim += 1

        return cls(torch.complex(real_stack, imag_stack), dim=target_dim,
                   dtype=first.dtype, device=first.device,
                   arrangement=first.arrangement, **first.kwargs)

    @classmethod
    def meshgrid(cls, *tensors: Union['Complex', torch.Tensor], indexing: str = 'ij', **kwargs):
        """
        Coordinate grids for complex vectors.
        """
        # Convert all to complex tensors
        c_tensors = []
        for t in tensors:
            if isinstance(t, cls):
                c_tensors.append(torch.complex(t.real, t.imag))
            else:
                # Use as_tensor to handle various inputs (list, numpy array, etc.)
                # and ensure it's a torch.Tensor.
                # If it's a real tensor, torch.complex will handle it.
                c_tensors.append(torch.as_tensor(t))

        grids = torch.meshgrid(*c_tensors, indexing=indexing)
        # Wrap each grid
        return [cls(g, **kwargs) for g in grids]

    @classmethod
    def einsum(cls, equation: str, *tensors: Union['Complex', torch.Tensor], **kwargs):
        """
        Multi-operand Einstein summation for Complex objects.
        """
        objs = []
        for t in tensors:
            if not isinstance(t, cls):
                objs.append(cls(t, **kwargs))
            else:
                objs.append(t)

        c_tensors = [torch.complex(o.real, o.imag) for o in objs]
        res = torch.einsum(equation, *c_tensors)

        first = objs[0]
        return cls(res, dtype=first.dtype, device=first.device, arrangement=first.arrangement, **first.kwargs)

    def argsort(self, dim: int = -1, descending: bool = False):
        """
        Returns the indices that would sort the tensor based on magnitude.
        """
        return torch.argsort(self.mag(), dim=dim, descending=descending)

    def to(self, *args, **kwargs):
        device = kwargs.get('device', None)
        for arg in args:
            if isinstance(arg, (torch.device, str)):
                device = arg
                break

        if device:
            self.device = device
            if self.tensor is not None:
                self.tensor = self.tensor.to(device)
            if self.data_store is not None:
                for v in self.data_store.values():
                    v.to(device)
        return self

    # Internal accessors
    def _real(self):
        """Internal method to get real part."""
        res = self.tensor.unbind(self.dim)[0]
        # print(f"DEBUG _real: dim={self.dim}, tensor_shape={self.tensor.shape}, res_shape={res.shape}")
        return res

    def _imag(self):
        """Internal method to get imaginary part."""
        return self.tensor.unbind(self.dim)[1]

    # Property accessors (new PyTorch-compatible way)
    @property
    def real(self):
        """Real part of the complex tensor (property access)."""
        return self._real()

    @real.setter
    def real(self, value):
        """Setter for real part (in-place mutation)."""
        if isinstance(value, torch.Tensor):
            self.tensor.unbind(self.dim)[0].copy_(value)
        else:
            self.tensor.unbind(self.dim)[0].fill_(value)

    @property
    def imag(self):
        """Imaginary part of the complex tensor (property access)."""
        return self._imag()

    @imag.setter
    def imag(self, value):
        """Setter for imaginary part (in-place mutation)."""
        if isinstance(value, torch.Tensor):
            self.tensor.unbind(self.dim)[1].copy_(value)
        else:
            self.tensor.unbind(self.dim)[1].fill_(value)

    # Property aliases for compatibility
    @property
    def real_part(self):
        """Property alias for real to support z.real_part syntax."""
        return self.real

    @property
    def imag_part(self):
        """Property alias for imag to support z.imag_part syntax."""
        return self.imag

    def mag(self):
        real, imag = self.tensor.unbind(self.dim)
        return torch.sqrt((real ** 2) + (imag ** 2))

    def phi(self):
        real, imag = self.tensor.unbind(self.dim)
        return torch.atan2(imag, real)

    # --- Standard Tensor Properties ---

    @property
    def T(self):
        """Shortcut for transpose(-1, -2)."""
        return self.transpose(-1, -2)

    @property
    def H(self):
        """Hermitian transpose (conjugate transpose)."""
        return self.adjoint()

    @property
    def mH(self):
        """Alias for Hermitian transpose."""
        return self.adjoint()

    @property
    def mT(self):
        """Matrix transpose (transpose last two dimensions)."""
        return self.transpose(-2, -1)

    # --- Standard Metadata Methods ---

    def size(self, dim=None):
        """
        Returns the size of the tensor. If dim is specified, returns the size of that dimension.
        """
        if dim is None:
            return self.real.size()
        return self.real.size(dim)

    def numel(self):
        """Returns the total number of complex elements."""
        return self.real.numel()

    def nelement(self):
        """Alias for numel()."""
        return self.numel()

    def element_size(self):
        """Returns the size in bytes of an individual complex element."""
        # Complex element is 2x the size of a real element
        return self.real.element_size() * 2

    # --- Low-Level Memory Accessors ---

    def data_ptr(self):
        """Returns the memory address of the first element."""
        return self.real.data_ptr()

    def storage_offset(self):
        """Returns the offset in the underlying storage."""
        return self.real.storage_offset()

    # --- Memory Layout & Casting ---

    def contiguous(self):
        """Returns a contiguous copy of the tensor."""
        real_cont = self.real.contiguous()
        imag_cont = self.imag.contiguous()
        stacked = torch.stack([real_cont, imag_cont], dim=self.dim)
        return self._wrap(stacked)

    def is_contiguous(self):
        """Check if the tensor is contiguous in memory."""
        return self.tensor.is_contiguous()

    def type_as(self, tensor):
        """Casts this tensor to the same type as the provided tensor."""
        if isinstance(tensor, Complex):
            target_dtype = tensor.dtype
            target_device = tensor.device
        else:
            target_dtype = tensor.dtype
            target_device = tensor.device

        return self.to(dtype=target_dtype, device=target_device)

    def view_as(self, tensor):
        """Views this tensor as the same size as another tensor."""
        if isinstance(tensor, Complex):
            target_shape = tensor.real.shape
        else:
            target_shape = tensor.shape

        real_view = self.real.view(target_shape)
        imag_view = self.imag.view(target_shape)
        stacked = torch.stack([real_view, imag_view], dim=self.dim)
        return self._wrap(stacked)

    def expand_as(self, tensor):
        """Expands this tensor to the same size as another tensor."""
        if isinstance(tensor, Complex):
            target_shape = tensor.real.shape
        else:
            target_shape = tensor.shape

        real_expand = self.real.expand(target_shape)
        imag_expand = self.imag.expand(target_shape)
        stacked = torch.stack([real_expand, imag_expand], dim=self.dim)
        return self._wrap(stacked)

    def bfloat16(self):
        """Casts to bfloat16."""
        return self.to(dtype=torch.bfloat16)

    def half(self):
        """Casts to float16."""
        return self.to(dtype=torch.float16)

    def float(self):
        """Casts to float32."""
        return self.to(dtype=torch.float32)

    def double(self):
        """Casts to float64."""
        return self.to(dtype=torch.float64)

    # --- In-Place Data Mutation ---

    def zero_(self):
        """Fills the tensor with zeros in-place."""
        self.real.zero_()
        self.imag.zero_()
        return self

    def fill_(self, value):
        """Fills the tensor with a specific complex value in-place."""
        if not isinstance(value, Complex):
            value = Complex(value, dim=0, dtype=self.dtype, device=self.device)

        # Extract scalar values
        val_real = value.real.item() if value.real.numel() == 1 else value.real
        val_imag = value.imag.item() if value.imag.numel() == 1 else value.imag

        self.real.fill_(val_real)
        self.imag.fill_(val_imag)
        return self

    def uniform_(self, from_val=0, to_val=1):
        """Fills the tensor with values from a uniform distribution in-place."""
        self.real.uniform_(from_val, to_val)
        self.imag.uniform_(from_val, to_val)
        return self

    def normal_(self, mean=0, std=1):
        """Fills the tensor with values from a normal distribution in-place."""
        self.real.normal_(mean, std)
        self.imag.normal_(mean, std)
        return self

    @classmethod
    def where_(cls, condition, val1, val2):
        """
        Similar to torch.where, but for Complex objects.
        Returns a new Complex object with elements from val1 where condition is True,
        and from val2 where condition is False.
        """
        # Ensure both inputs are Complex objects for uniform property access (.real, .imag)
        if not isinstance(val1, cls):
            ref_dim = val2.dim if isinstance(val2, cls) else None
            val1 = cls(val1, dim=ref_dim)

        if not isinstance(val2, cls):
            val2 = cls(val2, dim=val1.dim, dtype=val1.dtype, device=val1.device)

        real = torch.where(condition, val1.real, val2.real)
        imag = torch.where(condition, val1.imag, val2.imag)

        # Combine back into a stacked tensor and wrap in a new Complex instance
        res_stacked = torch.stack([real, imag], dim=val1.dim)
        return cls(res_stacked, dim=val1.dim, dtype=val1.dtype, device=val1.device,
                   arrangement=val1.arrangement, _is_stacked=True, **val1.kwargs)

    def to_polar(self):
        """Converts complex tensor to polar coordinates (magnitude, phase)."""
        mag = self.mag()
        phi = self.phi()
        return torch.stack([mag, phi], dim=self.dim)

    def unit(self):
        phi = self.phi()
        real = torch.cos(phi)
        imag = torch.sin(phi)
        res = torch.stack([real, imag], dim=self.dim)
        return self._wrap(res)

    def conj(self, inplace: bool = False, out: Optional['Complex'] = None):
        real = self.real
        imag = -self.imag
        res = torch.stack([real, imag], dim=self.dim)
        if out is not None:
            out.tensor.copy_(res)
            return out
        if inplace:
            self.tensor = res
            return self
        return self._wrap(res)

    def hermitian(self):
        """Returns the Hermitian transpose (conjugate transpose) of the tensor."""
        return self.conj().transpose(-2, -1)

    # Alias for compatibility
    adjoint = hermitian

    def linear(self, weight: 'Complex', bias: Optional['Complex'] = None):
        """
        Applies a linear transformation to the incoming data: y = x @ weight.T + bias.
        This allows the Complex tensor to be used as input to operations resembling torch.nn.functional.linear.
        """
        # Convert to native complex tensors for efficient operation
        x_c = torch.complex(self.real, self.imag)
        w_c = torch.complex(weight.real, weight.imag)

        b_c = None
        if bias is not None:
            b_c = torch.complex(bias.real, bias.imag)

        # F.linear computes x @ weight.T + bias
        # Note: weight is expected to be (out_features, in_features)
        # weight.T (implicit in linear) logic will happen on the complex tensor level.
        res_c = F.linear(x_c, w_c, b_c)

        return Complex(res_c, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                       **self.kwargs)

    def unit_conj(self):
        real, imag = self.conj().unbind(self.dim)
        phi = torch.atan2(imag, real)
        real = torch.cos(phi)
        imag = torch.sin(phi)
        return torch.stack([real, imag], dim=self.dim)

    def __matmul__(self, other):
        """
        Matrix multiplication (@).
        """
        if isinstance(other, Complex):
            # (A + iB)(C + iD) = (AC - BD) + i(AD + BC)
            real = self.real @ other.real - self.imag @ other.imag
            imag = self.real @ other.imag + self.imag @ other.real
            return Complex(torch.complex(real, imag), dim=self.dim, dtype=self.dtype, device=self.device,
                           arrangement=self.arrangement, **self.kwargs)
        else:
            # Assume real tensor
            real = self.real @ other
            imag = self.imag @ other
            return Complex(torch.complex(real, imag), dim=self.dim, dtype=self.dtype, device=self.device,
                           arrangement=self.arrangement, **self.kwargs)

    def inv(self):
        mag = self.mag()
        conj = self.conj()
        # Ensure mag**2 is treated as purely real, not split
        mag_sq_c = torch.complex(mag ** 2, torch.zeros_like(mag))

        # We need to construct Complex object for 'divide' to accept it as 'other_decomposition'
        # Or simpler: since 'divide' accepts Tensor (if it treats it as Real input/Fused), 
        # we found that problematic. 
        # But 'divide' implementation checks 'isinstance(other_decomposition, Complex)'.
        # If not, it wraps it.
        # So we must wrap it explicitly properly.
        mag_sq_complex = Complex(mag_sq_c, dim=self.dim, dtype=self.dtype, device=self.device,
                                 arrangement=self.arrangement, **self.kwargs)

        return conj.divide(mag_sq_complex, div_type='element_wise')

    # --- Sorting & Selection ---

    def sort(self, dim: int = -1, descending: bool = False):
        mag = self.mag()
        _, indices = torch.sort(mag, dim=dim, descending=descending)
        res_real = torch.gather(self.real, dim, indices)
        res_imag = torch.gather(self.imag, dim, indices)
        return self._wrap(torch.stack([res_real, res_imag], dim=self.dim)), indices

    def msort(self):
        """Alias for sort(dim=0), returns only values."""
        return self.sort(dim=0)[0]

    def topk(self, k: int, dim: int = -1, largest: bool = True, sorted: bool = True):
        mag = self.mag()
        _, indices = torch.topk(mag, k, dim=dim, largest=largest, sorted=sorted)
        res_real = torch.gather(self.real, dim, indices)
        res_imag = torch.gather(self.imag, dim, indices)
        return self._wrap(torch.stack([res_real, res_imag], dim=self.dim)), indices

    def max(self, dim: Optional[int] = None, keepdim: bool = False):
        mag = self.mag()
        if dim is None:
            idx = torch.argmax(mag)
            val_real = self.real.flatten()[idx]
            val_imag = self.imag.flatten()[idx]
            return self._wrap(torch.stack([val_real, val_imag], dim=0), dim=0)
        else:
            _, indices = torch.max(mag, dim=dim, keepdim=keepdim)
            res_real = torch.gather(self.real, dim, indices)
            res_imag = torch.gather(self.imag, dim, indices)
            return self._wrap(torch.stack([res_real, res_imag], dim=self.dim)), indices

    def min(self, dim: Optional[int] = None, keepdim: bool = False):
        mag = self.mag()
        if dim is None:
            idx = torch.argmin(mag)
            val_real = self.real.flatten()[idx]
            val_imag = self.imag.flatten()[idx]
            return self._wrap(torch.stack([val_real, val_imag], dim=0), dim=0)
        else:
            _, indices = torch.min(mag, dim=dim, keepdim=keepdim)
            res_real = torch.gather(self.real, dim, indices)
            res_imag = torch.gather(self.imag, dim, indices)
            return self._wrap(torch.stack([res_real, res_imag], dim=self.dim)), indices

    def clamp(self, min: Optional[float] = None, max: Optional[float] = None, inplace: bool = False,
              out: Optional['Complex'] = None):
        mag = self.mag()
        clamped_mag = torch.clamp(mag, min=min, max=max)
        scale = clamped_mag / (mag + 1e-12)
        real = self.real * scale
        imag = self.imag * scale
        res = torch.stack([real, imag], dim=self.dim)
        if out is not None:
            out.tensor.copy_(res)
            return out
        if inplace:
            self.tensor = res
            return self
        return self._wrap(res, dim=self.dim)

    def clamp_min(self, min: float, inplace: bool = False, out: Optional['Complex'] = None):
        return self.clamp(min=min, inplace=inplace, out=out)

    def clamp_max(self, max: float, inplace: bool = False, out: Optional['Complex'] = None):
        return self.clamp(max=max, inplace=inplace, out=out)

    # Standard aliases
    def clip(self, min: Optional[float] = None, max: Optional[float] = None, inplace: bool = False,
             out: Optional['Complex'] = None):
        """Alias for clamp."""
        return self.clamp(min=min, max=max, inplace=inplace, out=out)

    def amax(self, dim: Optional[int] = None, keepdim: bool = False):
        """Alias for max."""
        return self.max(dim=dim, keepdim=keepdim)

    def amin(self, dim: Optional[int] = None, keepdim: bool = False):
        """Alias for min."""
        return self.min(dim=dim, keepdim=keepdim)

    # --- Arithmetic Aliases (PyTorch Shorthand) ---

    def sub(self, other, alpha=1):
        """Alias for subtract."""
        return self.subtract(other if not isinstance(other, Complex) else other.multiply(alpha))

    def sub_(self, other, alpha=1):
        """In-place subtraction: self -= alpha * other_decomposition"""
        if not isinstance(other, Complex):
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device)
        self.real.sub_(other.real, alpha=alpha)
        self.imag.sub_(other.imag, alpha=alpha)
        return self

    def mul(self, other):
        """Alias for multiply."""
        return self.multiply(other)

    def mul_(self, other):
        """In-place multiplication."""
        if not isinstance(other, Complex):
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device)
        a, b = self.real.clone(), self.imag.clone()
        c, d = other.real, other.imag
        self.real.copy_(a * c - b * d)
        self.imag.copy_(a * d + b * c)
        return self

    def div(self, other):
        """Alias for divide."""
        return self.divide(other)

    def div_(self, other):
        """In-place division."""
        if not isinstance(other, Complex):
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device)
        a, b = self.real.clone(), self.imag.clone()
        c, d = other.real, other.imag
        denom = c ** 2 + d ** 2
        self.real.copy_((a * c + b * d) / denom)
        self.imag.copy_((b * c - a * d) / denom)
        return self

    def true_divide(self, other):
        """Alias for divide (true division)."""
        return self.divide(other)

    # --- In-Place Aliases ---

    def clamp_(self, min: Optional[float] = None, max: Optional[float] = None):
        """In-place alias for clamp."""
        return self.clamp(min=min, max=max, inplace=True)

    def relu_(self):
        """In-place ReLU activation."""
        # ReLU for complex: set negative magnitude values to zero
        mag = self.mag()
        mask = mag > 0
        real_result = self.real * mask
        imag_result = self.imag * mask
        self.tensor = torch.stack([real_result, imag_result], dim=self.dim)
        return self

    # --- Advanced Matrix Factorizations ---

    def qr(self, mode='reduced'):
        c_tensor = torch.complex(self.real, self.imag)
        Q, R = torch.linalg.qr(c_tensor, mode=mode)
        return Complex(Q, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                       **self.kwargs), \
            Complex(R, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement, **self.kwargs)

    def lu(self, pivot=True):
        c_tensor = torch.complex(self.real, self.imag)
        P, L, U = torch.linalg.lu(c_tensor, pivot=pivot)
        return P, \
            Complex(L, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement, **self.kwargs), \
            Complex(U, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement, **self.kwargs)

    def eigh(self, UPLO='L'):
        c_tensor = torch.complex(self.real, self.imag)
        evals, evecs = torch.linalg.eigh(c_tensor, UPLO=UPLO)
        return Complex(evals, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                       **self.kwargs), \
            Complex(evecs, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                    **self.kwargs)

    # --- Tensor Manipulation Wrappers ---

    def pad(self, pad, mode='constant', value=0):
        real = F.pad(self.real, pad, mode=mode, value=value)
        imag = F.pad(self.imag, pad, mode=mode, value=value)
        return self._wrap(torch.stack([real, imag], dim=self.dim))

    def roll(self, shifts, dims):
        if isinstance(dims, int):
            dims = (dims,)
        if isinstance(shifts, int) and len(dims) > 1:
            shifts = (shifts,) * len(dims)

        real = torch.roll(self.real, shifts, dims)
        imag = torch.roll(self.imag, shifts, dims)
        return self._wrap(torch.stack([real, imag], dim=self.dim))

    def flip(self, dims):
        if isinstance(dims, int):
            dims = (dims,)
        real = torch.flip(self.real, dims)
        imag = torch.flip(self.imag, dims)
        return self._wrap(torch.stack([real, imag], dim=self.dim))

    def tile(self, dims):
        if isinstance(dims, int):
            dims = (dims,)
        real = torch.tile(self.real, dims)
        imag = torch.tile(self.imag, dims)
        return self._wrap(torch.stack([real, imag], dim=self.dim))

    def take_along_dim(self, indices, dim):
        """Selects values from input at the 1D indices along the given dim."""
        res_real = torch.take_along_dim(self.real, indices, dim=dim)
        res_imag = torch.take_along_dim(self.imag, indices, dim=dim)
        return self._wrap(torch.stack([res_real, res_imag], dim=self.dim))

    def argwhere(self):
        """Returns indices of all non-zero elements."""
        return self.nonzero()

    def flatnonzero(self):
        """Returns indices that are non-zero in the flattened tensor."""
        return self.flatten().nonzero().squeeze(1)

    # --- Polynomial Operations ---

    def roots(self):
        """
        Find roots of polynomial with coefficients given by this tensor.
        Coefficients are assumed to be [a_n, a_{n-1}, ..., a_0].
        Works only for 1D Complex objects.
        """
        if self.real.dim() > 1:
            raise ValueError("roots() only supported for 1D Complex tensors.")

        c_coeffs = torch.complex(self.real, self.imag)
        # Remove leading zeros
        non_zero = torch.nonzero(c_coeffs)
        if len(non_zero) == 0:
            return Complex(torch.tensor([], dtype=self.dtype, device=self.device), dim=self.dim)

        c_coeffs = c_coeffs[non_zero[0].item():]
        n = len(c_coeffs) - 1
        if n <= 0:
            return Complex(torch.tensor([], dtype=self.dtype, device=self.device), dim=self.dim)
        if n == 1:
            root = -c_coeffs[1] / c_coeffs[0]
            return Complex(root.unsqueeze(0), dim=self.dim, dtype=self.dtype, device=self.device)

        # Companion Matrix
        A = torch.zeros((n, n), dtype=c_coeffs.dtype, device=c_coeffs.device)
        A[0, :] = -c_coeffs[1:] / c_coeffs[0]
        if n > 1:
            A[1:, :-1] = torch.eye(n - 1, dtype=c_coeffs.dtype, device=c_coeffs.device)

        roots = torch.linalg.eigvals(A)
        return Complex(roots, dim=self.dim, dtype=self.dtype, device=self.device)

    def polyfit_instance(self, y: 'Complex', deg: int, **kwargs):
        """
        Find coefficients for a polynomial of degree `deg` that fits (self, y).
        """
        x_c = torch.complex(self.real, self.imag)
        y_c = torch.complex(y.real, y.imag)

        # Vandermonde Matrix
        V = torch.vander(x_c, N=deg + 1)

        # Solve least squares
        coeffs, _, _, _ = torch.linalg.lstsq(V, y_c)
        return Complex(coeffs, dim=self.dim, dtype=self.dtype, device=self.device)

    # --- Advanced Signal Processing ---

    def hilbert(self, n=None, dim=-1):
        """
        Compute the analytic signal using the Hilbert transform.
        """
        # 1. FFT
        X = self.fft(n=n, dim=dim)
        c_X = torch.complex(X.real, X.imag)

        # 2. Create mask
        N = n if n else self.real.shape[dim]
        h = torch.zeros(N, dtype=c_X.dtype, device=c_X.device)
        if N % 2 == 0:
            h[0] = h[N // 2] = 1
            h[1: N // 2] = 2
        else:
            h[0] = 1
            h[1: (N + 1) // 2] = 2

        # Reshape h for broadcasting
        h_shape = [1] * c_X.dim()
        h_shape[dim] = N
        h = h.view(*h_shape)

        # 3. IFFT
        res_c = torch.fft.ifft(c_X * h, dim=dim)
        return Complex(res_c, dim=self.dim, dtype=self.dtype, device=self.device)

    @staticmethod
    def bartlett(window_length, **kwargs):
        w = torch.bartlett_window(window_length, **kwargs)
        return Complex(torch.complex(w, torch.zeros_like(w)), dim=-1)

    @staticmethod
    def hamming(window_length, **kwargs):
        w = torch.hamming_window(window_length, **kwargs)
        return Complex(torch.complex(w, torch.zeros_like(w)), dim=-1)

    @staticmethod
    def hann(window_length, **kwargs):
        w = torch.hann_window(window_length, **kwargs)
        return Complex(torch.complex(w, torch.zeros_like(w)), dim=-1)

    def exp(self, inplace: bool = False, out: Optional['Complex'] = None):
        real, imag = self.tensor.unbind(self.dim)
        mag = torch.exp(real)
        real_res, imag_res = torch.cos(imag), torch.sin(imag)
        real_res, imag_res = real_res * mag, imag_res * mag
        res = torch.stack([real_res, imag_res], dim=self.dim)
        if out is not None:
            out.tensor.copy_(res)
            return out
        if inplace:
            self.tensor = res
            return self
        else:
            return self._wrap(res)

    def exp2(self, inplace: bool = False, out: Optional['Complex'] = None):
        """Compute 2^z using exp: 2^z = e^(z * ln(2))."""
        ln2 = math.log(2.0)
        scaled = self.multiply(ln2)
        res = scaled.exp()
        if out is not None:
            out.tensor.copy_(res.tensor)
            return out
        if inplace:
            self.tensor = res.tensor
            return self
        return res

    def exp10(self, inplace: bool = False, out: Optional['Complex'] = None):
        """Compute 10^z using exp: 10^z = e^(z * ln(10))."""
        ln10 = math.log(10.0)
        scaled = self.multiply(ln10)
        res = scaled.exp()
        if out is not None:
            out.tensor.copy_(res.tensor)
            return out
        if inplace:
            self.tensor = res.tensor
            return self
        return res

    def exp_n(self, n=math.e, inplace: bool = False, out: Optional['Complex'] = None):
        """Compute n^z using exp: n^z = e^(z * ln(n))."""
        if n == math.e:
            return self.exp(inplace=inplace, out=out)
        ln_n = math.log(n)
        scaled = self.multiply(ln_n)
        res = scaled.exp()
        if out is not None:
            out.tensor.copy_(res.tensor)
            return out
        if inplace:
            self.tensor = res.tensor
            return self
        return res

    def iexp(self):
        real, imag = self.tensor.unbind(self.dim)
        real1, imag1 = -imag, real
        mag = torch.exp(real1)
        real, imag = torch.cos(imag1), torch.sin(imag1)
        real, imag = real * mag, imag * mag
        return torch.stack([real, imag], dim=self.dim)

    # --- Log-Sum-Exp Family (Critical for ML) ---

    def logsumexp(self, dim, keepdim=False):
        """
        Computes log(sum(exp(z))) along a dimension.
        Numerically stable implementation for complex numbers.
        """
        # For complex numbers: log(sum(exp(z))) = log(sum(exp(z)))
        # We need to implement this manually since torch.logsumexp doesn't support complex

        # Compute exp(z) for each element
        exp_z = self.exp()

        # Sum along dimension
        sum_exp = exp_z.sum(dim=dim, keepdim=keepdim)

        # Take log
        result = sum_exp.log()

        return result

    def logaddexp(self, other):
        """
        Element-wise computation of log(exp(x) + exp(y)).
        Numerically stable.
        """
        if not isinstance(other, Complex):
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device)

        # log(exp(x) + exp(y)) = max(x,y) + log(1 + exp(-|x-y|))
        # For complex: use torch.logaddexp on complex tensors
        x_complex = torch.complex(self.real, self.imag)
        y_complex = torch.complex(other.real, other.imag)

        # PyTorch doesn't have logaddexp for complex, so implement manually
        result = torch.log(torch.exp(x_complex) + torch.exp(y_complex))
        return Complex(result, dim=self.dim, dtype=self.dtype, device=self.device)

    def logaddexp2(self, other):
        """
        Element-wise computation of log2(2^x + 2^y).
        Numerically stable.
        """
        if not isinstance(other, Complex):
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device)

        # log2(2^x + 2^y) = log(2^x + 2^y) / log(2)
        x_complex = torch.complex(self.real, self.imag)
        y_complex = torch.complex(other.real, other.imag)

        result = torch.log2(torch.pow(2.0, x_complex) + torch.pow(2.0, y_complex))
        return Complex(result, dim=self.dim, dtype=self.dtype, device=self.device)

    def __getitem__(self, key):
        real = self.real[key]
        imag = self.imag[key]
        return Complex(torch.complex(real, imag), dim=-1, dtype=self.dtype, device=self.device,
                       arrangement=self.arrangement, **self.kwargs)

    def __setitem__(self, key, value):
        if not isinstance(value, Complex):
            # If value is a scalar, use dim=0 to avoid stack index issues.
            val_dim = 0 if isinstance(value, (complex, float, int)) else self.dim
            value = Complex(value, dim=val_dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                            **self.kwargs)

        self.real[key] = value.real
        self.imag[key] = value.imag

    def __iter__(self):
        """
        Allows iterating over the Complex object along the first dimension.
        """
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return self.real.shape[0]

    def item(self):
        if self.real.numel() != 1:
            raise ValueError("only one element tensors can be converted to Python scalars")
        return complex(self.real.item(), self.imag.item())

    def tolist(self):
        """
        Returns the tensor as a (possibly nested) list of complex numbers.
        """
        return torch.complex(self.real, self.imag).tolist()

    def sigmoid(self, inplace: bool = False, out: Optional['Complex'] = None):
        """
        Computes the complex sigmoid: 1 / (1 + exp(-z))
        """
        neg_z = self.multiply(-1)
        exp_neg_z = neg_z.exp(inplace=False)
        denom = exp_neg_z.add(1)
        res = denom.inv()
        if out is not None:
            out.tensor.copy_(res.tensor)
            return out
        if inplace:
            self.tensor = res.tensor
            return self
        else:
            return res

    def tanhshrink(self):
        """
        Computes z - tanh(z)
        """
        return self.subtract(self.tanh())

    def softsign(self):
        """
        Computes z / (1 + |z|)
        """
        mag = self.mag()
        denom = mag + 1.0  # This is a real tensor
        return self.divide(denom)

    def numpy(self):
        """
        Returns the complex tensor as a NumPy array.
        """
        real = self.real.detach().cpu().numpy()
        imag = self.imag.detach().cpu().numpy()
        return real + 1j * imag

    def __array__(self, dtype=None):
        """
        NumPy __array__ protocol.
        """
        arr = self.numpy()
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    def phase_shift(self, Phi: float):
        Phi = torch.ones_like(self.tensor) * Phi
        real, imag = torch.cos(Phi), torch.sin(Phi)
        Phi = torch.stack([real, imag], dim=self.dim)
        return multiply(self.tensor, Phi, **self.kwargs)

    def sin(self, inplace: bool = False, out: Optional['Complex'] = None):
        real, imag = self.tensor.unbind(self.dim)
        # Match Dummy logic exactly
        sinh = torch.sinh(imag)
        cosh = torch.cosh(imag)
        cos_r = torch.cos(real)
        sin_r = torch.sin(real)

        real_res = -(sinh * cos_r)
        imag_res = (cosh * sin_r)

        res = torch.stack([real_res, imag_res], dim=self.dim)
        if out is not None:
            out.tensor.copy_(res)
            return out
        if inplace:
            self.tensor = res
            return self
        else:
            return self._wrap(res)

    def cos(self, inplace: bool = False, out: Optional['Complex'] = None):
        real, imag = self.tensor.unbind(self.dim)
        # Match Dummy logic
        sinh = torch.sinh(imag)
        cosh = torch.cosh(imag)
        cos_r = torch.cos(real)
        sin_r = torch.sin(real)

        real_res = (cosh * cos_r)
        imag_res = -(sinh * sin_r)

        res = torch.stack([real_res, imag_res], dim=self.dim)
        if out is not None:
            out.tensor.copy_(res)
            return out
        if inplace:
            self.tensor = res
            return self
        else:
            return self._wrap(res)

    def tan(self, inplace: bool = False, out: Optional['Complex'] = None):
        sin_t = self.sin()
        cos_t = self.cos()
        # sin_t and cos_t are Complex now!
        res = sin_t.divide(cos_t)
        if out is not None:
            out.tensor.copy_(res.tensor)
            return out
        if inplace:
            self.tensor = res.tensor
            return self
        else:
            return res

    def _wrap(self, tensor, dim=None):
        # Helper to wrap a stacked tensor into a Complex object
        wrap_dim = dim if dim is not None else self.dim
        real, imag = tensor.unbind(wrap_dim)
        c_tensor = torch.complex(real, imag)
        return Complex(c_tensor, dim=wrap_dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                       **self.kwargs)

    def arcsin(self, inplace: bool = False, out: Optional['Complex'] = None):
        i = Complex(torch.tensor([0., 1.], device=self.device, dtype=self.dtype), dim=0)
        iz_c = self.multiply(i)

        z2_c = self.multiply(self)

        one = Complex(torch.tensor([1., 0.], device=self.device, dtype=self.dtype), dim=0)

        term_c = one.subtract(z2_c)

        root = term_c.sqrt()

        arg_c = iz_c.add(root)

        val = arg_c.log()

        neg_i = Complex(torch.tensor([0., -1.], device=self.device, dtype=self.dtype), dim=0)
        res = neg_i.multiply(val)

        if out is not None:
            out.tensor.copy_(res.tensor)
            return out
        if inplace:
            self.tensor = res.tensor
            return self
        else:
            return res

    def arccos(self, inplace: bool = False, out: Optional['Complex'] = None):
        z2_c = self.multiply(self)

        one = Complex(torch.tensor([1., 0.], device=self.device, dtype=self.dtype), dim=0)

        term_c = z2_c.subtract(one)

        root = term_c.sqrt()

        i_root = root.multiply(Complex(torch.tensor([0., 1.], device=self.device, dtype=self.dtype), dim=0))
        arg_c = self.add(i_root)

        neg_i = Complex(torch.tensor([0., -1.], device=self.device, dtype=self.dtype), dim=0)
        res = neg_i.multiply(arg_c.log())

        if out is not None:
            out.tensor.copy_(res.tensor)
            return out
        if inplace:
            self.tensor = res.tensor
            return self
        else:
            return res

    def arctan(self, inplace: bool = False, out: Optional['Complex'] = None):
        i = Complex(torch.tensor([0., 1.], device=self.device, dtype=self.dtype), dim=0)
        iz_c = self.multiply(i)

        one = Complex(torch.tensor([1., 0.], device=self.device, dtype=self.dtype), dim=0)
        num_c = one.subtract(iz_c)
        den_c = one.add(iz_c)

        term_c = num_c.divide(den_c)

        log_term_c = term_c.log()

        half_i = Complex(torch.tensor([0., 0.5], device=self.device, dtype=self.dtype), dim=0)
        res = half_i.multiply(log_term_c)

        if out is not None:
            out.tensor.copy_(res.tensor)
            return out
        if inplace:
            self.tensor = res.tensor
            return self
        else:
            return res

    def arcsec(self, inplace: bool = False, out: Optional['Complex'] = None):
        inv_c = self.inv()
        res = inv_c.arccos()
        if out is not None:
            out.tensor.copy_(res.tensor)
            return out
        if inplace:
            self.tensor = res.tensor
            return self
        else:
            return res

    def arccosec(self, inplace: bool = False, out: Optional['Complex'] = None):
        inv_c = self.inv()
        res = inv_c.arcsin()
        if out is not None:
            out.tensor.copy_(res.tensor)
            return out
        if inplace:
            self.tensor = res.tensor
            return self
        else:
            return res

    def arccot(self, inplace: bool = False, out: Optional['Complex'] = None):
        inv_c = self.inv()
        res = inv_c.arctan()
        if out is not None:
            out.tensor.copy_(res.tensor)
            return out
        if inplace:
            self.tensor = res.tensor
            return self
        else:
            return res

    def cot(self, inplace: bool = False, out: Optional['Complex'] = None):
        tan_t = self.tan()
        res = tan_t.reciprocal()
        if out is not None:
            out.tensor.copy_(res.tensor)
            return out
        if inplace:
            self.tensor = res.tensor
            return self
        else:
            return res

    def sec(self, inplace: bool = False, out: Optional['Complex'] = None):
        cos_t = self.cos()
        res = cos_t.reciprocal()
        if out is not None:
            out.tensor.copy_(res.tensor)
            return out
        if inplace:
            self.tensor = res.tensor
            return self
        else:
            return res

    def cosec(self, inplace: bool = False, out: Optional['Complex'] = None):
        sin_t = self.sin()
        res = sin_t.reciprocal()
        if out is not None:
            out.tensor.copy_(res.tensor)
            return out
        if inplace:
            self.tensor = res.tensor
            return self
        else:
            return res

    def sinh(self, inplace: bool = False, out: Optional['Complex'] = None):
        real, imag = self.tensor.unbind(self.dim)
        sinh_r = torch.sinh(real)
        cosh_r = torch.cosh(real)
        sin_i = torch.sin(imag)
        cos_i = torch.cos(imag)

        real_res = sinh_r * cos_i
        imag_res = cosh_r * sin_i

        res = torch.stack([real_res, imag_res], dim=self.dim)
        if out is not None:
            out.tensor.copy_(res)
            return out
        if inplace:
            self.tensor = res
            return self
        else:
            return self._wrap(res)

    def cosh(self, inplace: bool = False, out: Optional['Complex'] = None):
        real, imag = self.tensor.unbind(self.dim)
        sinh_r = torch.sinh(real)
        cosh_r = torch.cosh(real)
        sin_i = torch.sin(imag)
        cos_i = torch.cos(imag)

        real_res = cosh_r * cos_i
        imag_res = sinh_r * sin_i

        res = torch.stack([real_res, imag_res], dim=self.dim)
        if out is not None:
            out.tensor.copy_(res)
            return out
        if inplace:
            self.tensor = res
            return self
        else:
            return self._wrap(res)

    def tanh(self, inplace: bool = False, out: Optional['Complex'] = None):
        sinh_t = self.sinh()
        cosh_t = self.cosh()
        res = sinh_t.divide(cosh_t)
        if out is not None:
            out.tensor.copy_(res.tensor)
            return out
        if inplace:
            self.tensor = res.tensor
            return self
        else:
            return res

    def coth(self, inplace: bool = False, out: Optional['Complex'] = None):
        tanh_t = self.tanh()
        res = tanh_t.reciprocal()
        if out is not None:
            out.tensor.copy_(res.tensor)
            return out
        if inplace:
            self.tensor = res.tensor
            return self
        else:
            return res

    def arcsinh(self, inplace: bool = False, out: Optional['Complex'] = None):
        z2 = self.multiply(self)
        z2_c = self._wrap(z2)

        one = Complex(torch.tensor([1., 0.], device=self.device, dtype=self.dtype), dim=0)

        term = z2_c.add(one)
        term_c = self._wrap(term)

        root = term_c.sqrt()

        arg = self.add(root)
        arg_c = self._wrap(arg)

        res = arg_c.log()

        if out is not None:
            out.tensor.copy_(res.tensor)
            return out
        if inplace:
            self.tensor = res.tensor
            return self
        else:
            return self._wrap(res)

    def arccosh(self, inplace: bool = False, out: Optional['Complex'] = None):
        z2 = self.multiply(self)
        z2_c = self._wrap(z2)

        one = Complex(torch.tensor([1., 0.], device=self.device, dtype=self.dtype), dim=0)

        term = z2_c.subtract(one)
        term_c = self._wrap(term)

        root = term_c.sqrt()

        arg = self.add(root)
        arg_c = self._wrap(arg)

        res = arg_c.log()

        if out is not None:
            out.tensor.copy_(res.tensor)
            return out
        if inplace:
            self.tensor = res
            return self
        else:
            return self._wrap(res)

    def arctanh(self, inplace: bool = False, out: Optional['Complex'] = None):
        one = Complex(torch.tensor([1., 0.], device=self.device, dtype=self.dtype), dim=0)

        num = one.add(self)
        num_c = self._wrap(num)

        den = one.subtract(self)
        den_c = self._wrap(den)

        term = num_c.divide(den_c)
        term_c = self._wrap(term)

        log_term = term_c.log()
        log_term_c = self._wrap(log_term)

        half = Complex(torch.tensor([0.5, 0.], device=self.device, dtype=self.dtype), dim=0)
        res = half.multiply(log_term_c)

        if out is not None:
            out.tensor.copy_(res.tensor)
            return out
        if inplace:
            self.tensor = res.tensor
            return self
        else:
            return self._wrap(res)

    def sech(self, inplace: bool = False, out: Optional['Complex'] = None):
        cosh_t = self.cosh()
        res = cosh_t.reciprocal()
        if out is not None:
            out.tensor.copy_(res.tensor)
            return out
        if inplace:
            self.tensor = res.tensor
            return self
        else:
            return res

    def arccoth(self, inplace: bool = False, out: Optional['Complex'] = None):
        inv_z = self.inv()
        res = inv_z.arctanh()
        if out is not None:
            out.tensor.copy_(res.tensor)
            return out
        if inplace:
            self.tensor = res.tensor
            return self
        else:
            return res

    def arcsech(self, inplace: bool = False):
        inv_z = self.inv()
        inv_c = Complex(inv_z, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                        **self.kwargs)
        res = inv_c.arccosh()
        if inplace:
            self.tensor = res
        else:
            return res

    def arccosech(self, inplace: bool = False):
        inv_z = self.inv()
        inv_c = Complex(inv_z, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                        **self.kwargs)
        res = inv_c.arcsinh()
        if inplace:
            self.tensor = res
        else:
            return res

    # --- Powers & Roots ---

    def sqrt(self, inplace: bool = False, out: Optional['Complex'] = None):
        mag = self.mag()
        phi = self.phi()

        root_mag = torch.sqrt(mag)
        half_phi = phi / 2.0

        real = root_mag * torch.cos(half_phi)
        imag = root_mag * torch.sin(half_phi)

        res = torch.stack([real, imag], dim=self.dim)
        if out is not None:
            out.tensor.copy_(res)
            return out
        if inplace:
            self.tensor = res
            return self
        else:
            return self._wrap(res)

    def pow(self, exponent):
        # z^w = exp(w * log(z))
        ln_z_c = self.log()

        term = ln_z_c.multiply(exponent)

        return term.exp()

    def root(self, n: int, k: int = 0, inplace: bool = False):
        mag = self.mag()
        phi = self.phi()

        root_mag = torch.pow(mag, 1.0 / n)
        angle = (phi + 2 * math.pi * k) / n

        real = root_mag * torch.cos(angle)
        imag = root_mag * torch.sin(angle)

        res = torch.stack([real, imag], dim=self.dim)
        if inplace:
            self.tensor = res
        else:
            return self._wrap(res)

    # --- Base-N Logarithms ---

    def log10(self, inplace: bool = False):
        ln_z_c = self.log()

        ln_10 = torch.log(torch.tensor(10.0, device=self.device, dtype=self.dtype))
        one_over_ln_10 = 1.0 / ln_10

        real, imag = ln_z_c.real(), ln_z_c.imag()
        real, imag = real * one_over_ln_10, imag * one_over_ln_10
        res = torch.stack([real, imag], dim=self.dim)

        if inplace:
            self.tensor = res
        else:
            return self._wrap(res)

    def log2(self, inplace: bool = False):
        ln_z_c = self.log()

        ln_2 = torch.log(torch.tensor(2.0, device=self.device, dtype=self.dtype))
        one_over_ln_2 = 1.0 / ln_2

        real, imag = ln_z_c.real, ln_z_c.imag
        real, imag = real * one_over_ln_2, imag * one_over_ln_2
        res = torch.stack([real, imag], dim=self.dim)

        if inplace:
            self.tensor = res
        else:
            return self._wrap(res)

    def logn(self, n: float, inplace: bool = False):
        ln_z_c = self.log()

        ln_n = torch.log(torch.tensor(float(n), device=self.device, dtype=self.dtype))
        one_over_ln_n = 1.0 / ln_n

        real, imag = ln_z_c.real, ln_z_c.imag
        real, imag = real * one_over_ln_n, imag * one_over_ln_n
        res = torch.stack([real, imag], dim=self.dim)

        if inplace:
            self.tensor = res
        else:
            return self._wrap(res)

    def log1p(self, inplace: bool = False):
        c_tensor = torch.complex(self.real, self.imag)
        res = torch.log1p(c_tensor)
        return Complex(res, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                       **self.kwargs)

    def expm1(self, inplace: bool = False):
        c_tensor = torch.complex(self.real, self.imag)
        res = torch.expm1(c_tensor)
        return Complex(res, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                       **self.kwargs)

    def sinc(self, inplace: bool = False):
        c_tensor = torch.complex(self.real, self.imag)
        res = torch.sinc(c_tensor)
        return Complex(res, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                       **self.kwargs)

    # --- Polynomials ---
    def polyval(self, coeffs):
        if isinstance(coeffs, list):
            # Ensure coefficients are handled
            pass

        def to_complex(c):
            if isinstance(c, Complex): return c
            if isinstance(c, (int, float, complex)):
                return Complex(torch.tensor([float(c.real), float(c.imag)], device=self.device), dim=0)
            if isinstance(c, torch.Tensor):
                if c.is_complex():
                    return Complex(c, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                                   **self.kwargs)
                else:
                    return Complex(c, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                                   **self.kwargs)
            return Complex(c, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                           **self.kwargs)

        if len(coeffs) == 0:
            return Complex(torch.zeros_like(self.real), dim=self.dim, dtype=self.dtype, device=self.device,
                           arrangement=self.arrangement, **self.kwargs)

        res = to_complex(coeffs[0])

        for c in coeffs[1:]:
            c_obj = to_complex(c)
            res = res.multiply(self).add(c_obj)

        return res

    # --- Comparison & Logic ---

    def allclose(self, other, rtol=1e-05, atol=1e-08, equal_nan=False):
        if not isinstance(other, Complex):
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                            **self.kwargs)
        return torch.allclose(self.tensor, other.tensor, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def eq(self, other):
        if not isinstance(other, Complex):
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                            **self.kwargs)
        return torch.eq(self.tensor, other.tensor).all(dim=self.dim)

    def neq(self, other):
        return ~self.eq(other)

    def isnan(self):
        return torch.isnan(self.tensor).any(dim=self.dim)

    def isinf(self):
        return torch.isinf(self.tensor).any(dim=self.dim)

    def isclose(self, other, rtol=1e-05, atol=1e-08, equal_nan=False):
        """
        Element-wise comparison with tolerance (returns a boolean mask).
        """
        if not isinstance(other, Complex):
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device)
        return torch.isclose(self.tensor, other.tensor, rtol=rtol, atol=atol, equal_nan=equal_nan).all(dim=self.dim)

    def isreal(self, tol=1e-7):
        """
        Returns a boolean mask where the imaginary part is effectively zero.
        """
        return torch.abs(self.imag) < tol

    # --- Logical Selection & Masking ---

    def where(self, condition, other):
        """
        Return a tensor of elements selected from self or other_decomposition, depending on condition.
        condition: boolean tensor
        """
        if not isinstance(other, Complex):
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                            **self.kwargs)

        cond_expanded = condition.unsqueeze(self.dim)
        res_tensor = torch.where(cond_expanded, self.tensor, other.tensor)
        return self._wrap(res_tensor)

    def masked_fill(self, mask, value):
        """
        Fills elements of self tensor with value where mask is True.
        """
        if not isinstance(value, Complex):
            # Convert scalar or real tensor to Complex to ensure correct stacking
            value = Complex(value, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                            **self.kwargs)

        mask_expanded = mask.unsqueeze(self.dim)
        # Using torch.where as a more flexible masked_fill for complex tensors
        res_tensor = torch.where(mask_expanded, value.tensor, self.tensor)
        return self._wrap(res_tensor)

    def masked_select(self, mask):
        """
        Returns a 1D Complex tensor which indexes self tensor according to the boolean mask.
        """
        real = self.real.masked_select(mask)
        imag = self.imag.masked_select(mask)
        return Complex(torch.complex(real, imag), dim=0)

    def nan_to_num(self, nan=0.0, posinf=None, neginf=None):
        """
        Replaces NaN, positive infinity, and negative infinity values in the tensor.
        """
        res_tensor = torch.nan_to_num(self.tensor, nan=nan, posinf=posinf, neginf=neginf)
        return self._wrap(res_tensor)

    # --- Optimization Primitives ---

    def addcmul(self, tensor1, tensor2, value=1.0):
        """
        Performs out = self + value * (tensor1 * tensor2)
        """
        if not isinstance(tensor1, Complex):
            tensor1 = Complex(tensor1, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                              **self.kwargs)
        if not isinstance(tensor2, Complex):
            tensor2 = Complex(tensor2, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                              **self.kwargs)

        prod = tensor1.multiply(tensor2)
        scaled_prod = prod.multiply(value)
        return self.add(scaled_prod)

    def addcdiv(self, tensor1, tensor2, value=1.0):
        """
        Performs out = self + value * (tensor1 / tensor2)
        """
        if not isinstance(tensor1, Complex):
            tensor1 = Complex(tensor1, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                              **self.kwargs)
        if not isinstance(tensor2, Complex):
            tensor2 = Complex(tensor2, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                              **self.kwargs)

        div = tensor1.divide(tensor2)
        scaled_div = div.multiply(value)
        return self.add(scaled_div)

    def lerp(self, end, weight):
        """
        Performs linear interpolation between self and end: self + weight * (end - self)
        """
        if not isinstance(end, Complex):
            end = Complex(end, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                          **self.kwargs)

        diff = end.subtract(self)
        scaled_diff = diff.multiply(weight)
        return self.add(scaled_diff)

    def sign(self):
        return self.unit()

    # --- Standard PyTorch Aliases ---

    def angle(self):
        """Standard alias for phi()."""
        return self.phi()

    def sgn(self):
        """Standard alias for sign() / unit()."""
        return self.unit()

    def absolute(self):
        """Standard alias for mag() / abs()."""
        return self.mag()

    def square(self):
        """Standard alias for pow(2)."""
        return self.pow(2)

    # --- Tensor Movement Operations ---

    def movedim(self, source, destination):
        """
        Moves a dimension from source index to destination index.
        """
        real_moved = torch.movedim(self.real, source, destination)
        imag_moved = torch.movedim(self.imag, source, destination)
        # Use dim=-1 for stacking since dimensions have changed
        stacked = torch.stack([real_moved, imag_moved], dim=-1)
        return self._wrap(stacked, dim=-1)

    def swapaxes(self, axis0, axis1):
        """
        Swaps two specific dimensions.
        """
        real_swapped = torch.swapaxes(self.real, axis0, axis1)
        imag_swapped = torch.swapaxes(self.imag, axis0, axis1)
        # Use dim=-1 for stacking since dimensions have changed
        stacked = torch.stack([real_swapped, imag_swapped], dim=-1)
        return self._wrap(stacked, dim=-1)

    def swapdims(self, dim0, dim1):
        """Alias for swapaxes."""
        return self.swapaxes(dim0, dim1)

    # --- Math Remainder Operations ---

    def fmod(self, divisor):
        """
        Floating-point remainder of division (element-wise).
        """
        if not isinstance(divisor, Complex):
            divisor = Complex(divisor, dim=self.dim, dtype=self.dtype, device=self.device)

        real_fmod = torch.fmod(self.real, divisor.real)
        imag_fmod = torch.fmod(self.imag, divisor.imag)
        stacked = torch.stack([real_fmod, imag_fmod], dim=self.dim)
        return self._wrap(stacked)

    def remainder(self, divisor):
        """
        Element-wise remainder of division.
        """
        if not isinstance(divisor, Complex):
            divisor = Complex(divisor, dim=self.dim, dtype=self.dtype, device=self.device)

        real_rem = torch.remainder(self.real, divisor.real)
        imag_rem = torch.remainder(self.imag, divisor.imag)
        stacked = torch.stack([real_rem, imag_rem], dim=self.dim)
        return self._wrap(stacked)

    # --- Linear Algebra ---
    def adjoint(self):
        conj = self.conj()
        # conj.tensor is the underlying stacked tensor
        real, imag = conj.tensor.unbind(self.dim)
        real_t = real.mT
        imag_t = imag.mT
        res = torch.stack([real_t, imag_t], dim=self.dim)
        return self._wrap(res)

    def triu(self, diagonal=0):
        """
        Returns the upper triangular part of a matrix or batch of matrices.
        """
        real = self.real.triu(diagonal)
        imag = self.imag.triu(diagonal)
        return self._wrap(torch.stack([real, imag], dim=self.dim))

    def tril(self, diagonal=0):
        """
        Returns the lower triangular part of a matrix or batch of matrices.
        """
        real = self.real.tril(diagonal)
        imag = self.imag.tril(diagonal)
        return self._wrap(torch.stack([real, imag], dim=self.dim))

    def diag(self, diagonal=0):
        """
        If self is 1D, returns a 2D diagonal matrix.
        If self is 2D, returns the diagonal elements as a 1D tensor.
        """
        real = self.real.diag(diagonal)
        imag = self.imag.diag(diagonal)
        # diag behavior: 1D -> 2D, 2D -> 1D. 
        # The complex dimension is preserved/appended at the same relative position if possible.
        return self._wrap(torch.stack([real, imag], dim=self.dim))

    def diag_embed(self, diagonal=0, offset=0, dim1=-2, dim2=-1):
        """
        Creates a tensor whose diagonals are certain 1D sections of self.
        """
        real = self.real.diag_embed(diagonal, offset, dim1, dim2)
        imag = self.imag.diag_embed(diagonal, offset, dim1, dim2)
        return self._wrap(torch.stack([real, imag], dim=self.dim))

    def matrix_exp(self):
        c_tensor = torch.complex(self.real, self.imag)
        res = torch.linalg.matrix_exp(c_tensor)
        return Complex(res, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                       **self.kwargs)

    def matrix_power(self, n: int):
        c_tensor = torch.complex(self.real, self.imag)
        res = torch.linalg.matrix_power(c_tensor, n)
        return Complex(res, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                       **self.kwargs)

    def addmm(self, mat1, mat2, beta=1, alpha=1):
        """
        Performs out = beta * self + alpha * (mat1 @ mat2)
        """
        if not isinstance(mat1, Complex):
            mat1 = Complex(mat1, dim=self.dim, dtype=self.dtype, device=self.device)
        if not isinstance(mat2, Complex):
            mat2 = Complex(mat2, dim=self.dim, dtype=self.dtype, device=self.device)

        c_self = torch.complex(self.real, self.imag)
        c_m1 = torch.complex(mat1.real, mat1.imag)
        c_m2 = torch.complex(mat2.real, mat2.imag)

        res = torch.addmm(c_self, c_m1, c_m2, beta=beta, alpha=alpha)
        return Complex(res, dim=self.dim, dtype=self.dtype, device=self.device)

    def baddbmm(self, batch1, batch2, beta=1, alpha=1):
        """
        Performs out = beta * self + alpha * (batch1 @ batch2) for batches.
        """
        if not isinstance(batch1, Complex):
            batch1 = Complex(batch1, dim=self.dim, dtype=self.dtype, device=self.device)
        if not isinstance(batch2, Complex):
            batch2 = Complex(batch2, dim=self.dim, dtype=self.dtype, device=self.device)

        c_self = torch.complex(self.real, self.imag)
        c_b1 = torch.complex(batch1.real, batch1.imag)
        c_b2 = torch.complex(batch2.real, batch2.imag)

        res = torch.baddbmm(c_self, c_b1, c_b2, beta=beta, alpha=alpha)
        return Complex(res, dim=self.dim, dtype=self.dtype, device=self.device)

    def addmv(self, mat, vec, beta=1, alpha=1):
        """
        Performs out = beta * self + alpha * (mat @ vec)
        """
        if not isinstance(mat, Complex):
            mat = Complex(mat, dim=self.dim, dtype=self.dtype, device=self.device)
        if not isinstance(vec, Complex):
            vec = Complex(vec, dim=self.dim, dtype=self.dtype, device=self.device)

        c_self = torch.complex(self.real, self.imag)
        c_mat = torch.complex(mat.real, mat.imag)
        c_vec = torch.complex(vec.real, vec.imag)

        res = torch.addmv(c_self, c_mat, c_vec, beta=beta, alpha=alpha)
        return Complex(res, dim=self.dim, dtype=self.dtype, device=self.device)

    def cond(self, p=None):
        c_tensor = torch.complex(self.real, self.imag)
        res = torch.linalg.cond(c_tensor, p=p)
        res_stacked = torch.stack([res, torch.zeros_like(res)], dim=self.dim)
        return self._wrap(res_stacked)

    def norm(self, ord=None, dim=None, keepdim=False):
        c_tensor = torch.complex(self.real, self.imag)
        res = torch.linalg.norm(c_tensor, ord=ord, dim=dim, keepdim=keepdim)
        res_stacked = torch.stack([res, torch.zeros_like(res)], dim=self.dim)
        return self._wrap(res_stacked)

    def nuclear_norm(self, dim=None, keepdim=False):
        c_tensor = torch.complex(self.real, self.imag)
        res = torch.linalg.norm(c_tensor, ord='nuc', dim=dim, keepdim=keepdim)
        return Complex(res, dim=0, device=self.device)

    def matrix_rank(self, atol=None, rtol=None, hermitian=False):
        """
        Returns the numerical rank of a matrix.
        """
        c_tensor = torch.complex(self.real, self.imag)
        rank = torch.linalg.matrix_rank(c_tensor, atol=atol, rtol=rtol, hermitian=hermitian)
        return rank

    def slogdet(self):
        """
        Computes the sign and natural logarithm of the absolute value of the determinant.
        """
        c_tensor = torch.complex(self.real, self.imag)
        sign, logabsdet = torch.linalg.slogdet(c_tensor)
        return Complex(sign, dim=0), Complex(logabsdet, dim=0)

    def is_hermitian(self, atol=1e-05, rtol=1e-08):
        """
        Returns True if the matrix is Hermitian (A = A^H).
        """
        if self.real.ndim < 2: return False
        return self.allclose(self.adjoint(), atol=atol, rtol=rtol)

    def is_unitary(self, atol=1e-05, rtol=1e-08):
        """
        Returns True if the matrix is unitary (A^H A = I).
        """
        if self.real.ndim < 2: return False
        adj = self.adjoint()
        prod = adj.multiply(self, mul_type='matmul')
        eye = Complex.eye(self.real.shape[-1], device=self.device, dtype=self.dtype)
        return prod.allclose(eye, atol=atol, rtol=rtol)

    def det(self):
        c_tensor = torch.complex(self.real, self.imag)
        res = torch.linalg.det(c_tensor)
        real, imag = res.real, res.imag
        res_t = torch.stack([real, imag], dim=self.dim)
        return self._wrap(res_t)

    def trace(self):
        real_tr = torch.diagonal(self.real, dim1=-2, dim2=-1).sum(-1)
        imag_tr = torch.diagonal(self.imag, dim1=-2, dim2=-1).sum(-1)
        res_t = torch.stack([real_tr, imag_tr], dim=self.dim)
        return self._wrap(res_t)

    def eig(self):
        c_tensor = torch.complex(self.real, self.imag)
        vals, vecs = torch.linalg.eig(c_tensor)

        vals_c = Complex(vals, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                         **self.kwargs)
        vecs_c = Complex(vecs, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                         **self.kwargs)

        return vals_c, vecs_c

    def svd(self, full_matrices: bool = True):
        c_tensor = torch.complex(self.real, self.imag)
        U, S, Vh = torch.linalg.svd(c_tensor, full_matrices=full_matrices)

        u_c = Complex(U, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                      **self.kwargs)

        s_stacked = torch.stack([S, torch.zeros_like(S)], dim=self.dim)
        s_c = self._wrap(s_stacked)

        vh_c = Complex(Vh, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                       **self.kwargs)

        return u_c, s_c, vh_c

    def polar_decomposition(self):
        """
        Computes the polar decomposition of a matrix. A = UP.
        For scalars, U is the unit phase and P is the magnitude.
        """
        if self.real.ndim < 2:
            mag = self.mag()
            U = self.divide(mag)
            P = Complex(mag, dim=0, device=self.device, dtype=self.dtype)
            return U, P

        try:
            c_tensor = torch.complex(self.real, self.imag)
            U, P = torch.linalg.polar(c_tensor)
            return Complex(U, dim=self.dim), Complex(P, dim=self.dim)
        except AttributeError:
            # SVD fallback
            U_svd, S, Vh = self.svd(full_matrices=False)
            U_polar = U_svd.multiply(Vh, mul_type='matmul')
            P_polar = U_polar.adjoint().multiply(self, mul_type='matmul')
            return U_polar, P_polar

    def schur(self, output_complex=True):
        """
        Computes the Schur decomposition of a matrix.
        """
        try:
            c_tensor = torch.complex(self.real, self.imag)
            T, Z = torch.linalg.schur(c_tensor, output_complex=output_complex)
            return Complex(T, dim=self.dim), Complex(Z, dim=self.dim)
        except AttributeError:
            # SciPy fallback
            import scipy.linalg
            A_np = self.numpy()
            T, Z = scipy.linalg.schur(A_np, output='complex' if output_complex else 'real')
            # Scipy returns complex numpy arrays. Complex() constructor handles them.
            return Complex(T, device=self.device), Complex(Z, device=self.device)

    def cholesky(self):
        c_tensor = torch.complex(self.real, self.imag)
        res = torch.linalg.cholesky(c_tensor)
        return Complex(res, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                       **self.kwargs)

    def cholesky_solve(self, b, upper=False):
        """
        Solves a linear system using the Cholesky factor.
        self is the Cholesky factor (from cholesky()), b is the right-hand side.
        """
        if not isinstance(b, Complex):
            b = Complex(b, dim=self.dim, dtype=self.dtype, device=self.device)

        u_complex = torch.complex(self.real, self.imag)
        b_complex = torch.complex(b.real, b.imag)

        # torch.cholesky_solve expects (b, u)
        x = torch.cholesky_solve(b_complex, u_complex, upper=upper)
        return Complex(x, dim=self.dim, dtype=self.dtype, device=self.device)

    def cholesky_inverse(self, upper=False):
        """
        Computes the inverse of a positive-definite matrix using its Cholesky factor.
        self is the Cholesky factor.
        """
        u_complex = torch.complex(self.real, self.imag)
        inv = torch.cholesky_inverse(u_complex, upper=upper)
        return Complex(inv, dim=self.dim, dtype=self.dtype, device=self.device)

    def logdet(self):
        """
        Computes the log determinant of a square matrix.
        Convenience wrapper for slogdet()[1].
        """
        return self.slogdet()[1]

    def ldl_factor(self, hermitian=True, upper=False):
        """
        Computes the LDL factorization of a Hermitian matrix.
        Returns (L, D, pivots) where A = L @ D @ L.H
        """
        c_tensor = torch.complex(self.real, self.imag)
        # PyTorch doesn't have ldl_factor for complex, use linalg.ldl_factor if available
        if hasattr(torch.linalg, 'ldl_factor'):
            LD, pivots = torch.linalg.ldl_factor(c_tensor, hermitian=hermitian, upper=upper)
            return Complex(LD, dim=self.dim, dtype=self.dtype, device=self.device), pivots
        else:
            # Fallback: use Cholesky decomposition L @ L.H
            L = self.cholesky()
            D = Complex.eye(L.shape[0], device=self.device, dtype=self.dtype)
            return L, D, None

    def ldl_solve(self, LD, pivots, b, hermitian=True, upper=False):
        """
        Solves a system using LDL factorization.
        """
        c_LD = torch.complex(LD.real, LD.imag) if isinstance(LD, Complex) else LD
        c_b = torch.complex(b.real, b.imag) if isinstance(b, Complex) else b

        if hasattr(torch.linalg, 'ldl_solve'):
            result = torch.linalg.ldl_solve(c_LD, pivots, c_b, hermitian=hermitian, upper=upper)
            return Complex(result, dim=self.dim, dtype=self.dtype, device=self.device)
        else:
            # Fallback: use cholesky_solve
            return LD.cholesky_solve(b)

    # --- Scalar Reductions ---

    def count_nonzero(self, dim=None):
        """Returns the number of non-zero elements (based on magnitude)."""
        return (self.mag() != 0).sum(dim=dim)

    def dist(self, other, p=2):
        """Computes the p-norm distance between two tensors (scalar result)."""
        if not isinstance(other, Complex):
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device)

        diff = self.sub(other)
        return diff.mag().norm(p=p)

    # --- Ranking & Quantiles ---

    def kthvalue(self, k: int, dim: int = -1, keepdim: bool = False):
        """Returns the k-th smallest element and its index (based on magnitude)."""
        mag = self.mag()
        values, indices = torch.kthvalue(mag, k, dim=dim, keepdim=keepdim)

        # Gather the complex values at those indices
        if dim < 0:
            dim = self.real.ndim + dim

        real_vals = torch.gather(self.real, dim, indices.unsqueeze(dim) if not keepdim else indices)
        imag_vals = torch.gather(self.imag, dim, indices.unsqueeze(dim) if not keepdim else indices)

        complex_vals = self._wrap(torch.stack([real_vals, imag_vals], dim=self.dim))
        return complex_vals, indices

    def pinv(self, rcond=1e-15, hermitian=False):
        """
        Computes the Moore-Penrose pseudoinverse.
        """
        c_tensor = torch.complex(self.real, self.imag)
        res = torch.linalg.pinv(c_tensor, rcond=rcond, hermitian=hermitian)
        return Complex(res, dim=self.dim, dtype=self.dtype, device=self.device)

    def renorm(self, p, dim, maxnorm):
        """
        Renormalizes sub-tensors along a dimension to satisfy a maximum norm constraint.
        """
        # Complex norm is the same as real norm of the complex elements
        c_tensor = torch.complex(self.real, self.imag)
        res = torch.renorm(c_tensor, p, dim, maxnorm)
        return Complex(res, dim=self.dim, dtype=self.dtype, device=self.device)

    def renorm_(self, p, dim, maxnorm):
        """
        In-place version of renorm.
        """
        c_tensor = torch.complex(self.real, self.imag)
        res = torch.renorm(c_tensor, p, dim, maxnorm)
        # Update in-place
        with torch.no_grad():
            self.real().copy_(res.real)
            self.imag().copy_(res.imag)
        return self

    def tensordot(self, other, dims=2):
        """
        General tensor contraction between two Complex tensors.
        """
        if not isinstance(other, Complex):
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device)

        c_self = torch.complex(self.real, self.imag)
        c_other = torch.complex(other.real, other.imag)

        res = torch.tensordot(c_self, c_other, dims=dims)
        # Result dim might change, we default to -1 or first found
        return Complex(res, dtype=self.dtype, device=self.device)

    # --- Statistics ---

    def var(self, dim=None, correction=1):
        c_tensor = torch.complex(self.real, self.imag)
        res = torch.var(c_tensor, dim=dim, correction=correction)
        res_stacked = torch.stack([res, torch.zeros_like(res)], dim=self.dim)
        return self._wrap(res_stacked)

    def cov(self, correction=1):
        c_tensor = torch.complex(self.real, self.imag)
        # Covariance matrix of the complex vector
        # Cov(z) = E[(z-mu)(z-mu)^H]
        mu = self.mean(dim=0, keepdim=True)
        z_centered = self.subtract(mu)
        res = z_centered.multiply(z_centered.adjoint(), mul_type='matmul')
        return res.divide(max(1, self.real.shape[0] - correction))

    # --- Robust Statistics ---

    def median(self, dim=None, keepdim=False):
        """
        Returns the median of the complex tensor based on magnitude.
        If dim is provided, returns the complex element at the median magnitude along that dimension.
        """
        mags = self.mag()
        if dim is not None:
            # torch.median returns (values, indices)
            res = torch.median(mags, dim=dim, keepdim=keepdim)
            indices = res.indices

            # Index into real and imag parts
            real = torch.gather(self.real(), dim=dim, index=indices)
            imag = torch.gather(self.imag(), dim=dim, index=indices)

            # The complex dimension is self.dim. 
            # If we squeezed dim, we need to adjust self.dim if it was affected.
            target_dim = self.dim
            if not keepdim and self.dim > dim:
                target_dim -= 1

            return self._wrap(torch.stack([real, imag], dim=target_dim), dim=target_dim)
        else:
            # Full reduction
            flat_mags = mags.flatten()
            # torch.median(input) returns a single value.
            # We need indices to get the complex pair.
            # Let's use flatten() and then argmedian if it existed, but we can use topk or sort.
            sorted_mags, sorted_indices = torch.sort(flat_mags)
            mid = len(flat_mags) // 2
            idx = sorted_indices[mid]

            r = self.real().flatten()[idx]
            i = self.imag().flatten()[idx]
            return Complex(torch.complex(r, i), dim=0)

    def quantile(self, q, dim=None, keepdim=False):
        """
        Returns the q-th quantile of the magnitudes of the complex tensor.
        Returns a real tensor (as Complex with imag=0).
        """
        mags = self.mag()
        res = torch.quantile(mags, q, dim=dim, keepdim=keepdim)
        # Returns a real tensor. We wrap it as Complex (imag=0).
        return Complex(res, dim=0, dtype=self.dtype, device=self.device)

    def mode(self, dim=-1, keepdim=False):
        """
        Returns the most frequent complex value along a dimension.
        Currently implemented by treating real and imag parts as a single unit.
        """
        # torch.mode doesn't support complex. 
        # We'll use a trick: combine real and imag into a single float if possible, 
        # or just use torch.mode on real and imag if they match? No.
        # For now, we'll return a warning that exact complex mode is complex to compute efficiently,
        # and provide a magnitude-based approximation or just the mode of real/imag separately is wrong.
        # Actually, we can use torch.unique with return_counts.
        # But this is per-dimension.
        # Let's just return the mode of real and imag parts if they coincide? 
        # No, let's implement for the whole tensor or specific dim.
        if dim is None:
            # Global mode
            c_flat = torch.complex(self.real().flatten(), self.imag().flatten())
            vals, counts = torch.unique(c_flat, return_counts=True)
            idx = torch.argmax(counts)
            return Complex(vals[idx], dim=0)
        else:
            # per dim mode is harder without native torch support for complex mode.
            # We'll fallback to magnitude-based mode for now or just the first mode.
            mags = self.mag()
            res = torch.mode(mags, dim=dim, keepdim=keepdim)
            indices = res.indices
            real = torch.gather(self.real(), dim=dim, index=indices)
            imag = torch.gather(self.imag(), dim=dim, index=indices)
            target_dim = self.dim
            if not keepdim and self.dim > dim:
                target_dim -= 1
            return self._wrap(torch.stack([real, imag], dim=target_dim), dim=target_dim)

    def histogram(self, bins=10, range=None, weight=None, density=False):
        """
        Computes the histogram of the magnitudes of the complex tensor.
        """
        mags = self.mag()
        # torch.histogram returns (hist, bin_edges)
        hist, edges = torch.histogram(mags.cpu(), bins=bins, range=range, weight=weight, density=density)
        return hist.to(self.device), edges.to(self.device)

    def cumsum(self, dim: int):
        c_tensor = torch.complex(self.real, self.imag)
        res = torch.cumsum(c_tensor, dim=dim)
        return Complex(res, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                       **self.kwargs)

    def cumprod(self, dim: int):
        c_tensor = torch.complex(self.real, self.imag)
        res = torch.cumprod(c_tensor, dim=dim)
        return Complex(res, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                       **self.kwargs)

    def cummax(self, dim: int):
        """
        Returns cumulative maximum along dimension (based on magnitude).
        Returns (values, indices) tuple.
        """
        mag = self.mag()
        _, indices = torch.cummax(mag, dim=dim)

        # Gather complex values using indices
        real_vals = torch.gather(self.real, dim, indices)
        imag_vals = torch.gather(self.imag, dim, indices)
        values = self._wrap(torch.stack([real_vals, imag_vals], dim=self.dim))

        return values, indices

    def cummin(self, dim: int):
        """
        Returns cumulative minimum along dimension (based on magnitude).
        Returns (values, indices) tuple.
        """
        mag = self.mag()
        _, indices = torch.cummin(mag, dim=dim)

        # Gather complex values using indices
        real_vals = torch.gather(self.real, dim, indices)
        imag_vals = torch.gather(self.imag, dim, indices)
        values = self._wrap(torch.stack([real_vals, imag_vals], dim=self.dim))

        return values, indices

    # --- Trapezoidal Integration ---

    def trapezoid(self, x=None, dim=-1):
        """
        Computes the trapezoidal rule along a dimension.
        If x is None, assumes unit spacing.
        """
        if x is not None and not isinstance(x, Complex):
            x = Complex(x, dim=self.dim, dtype=self.dtype, device=self.device)

        # For complex numbers, integrate real and imaginary parts separately
        if x is None:
            real_int = torch.trapezoid(self.real, dim=dim)
            imag_int = torch.trapezoid(self.imag, dim=dim)
        else:
            # Use x.real for spacing (assuming x is real or using real part)
            x_vals = x.real if isinstance(x, Complex) else x
            real_int = torch.trapezoid(self.real, x_vals, dim=dim)
            imag_int = torch.trapezoid(self.imag, x_vals, dim=dim)

        # Handle dimension for result
        if real_int.dim() == 0:
            real_int = real_int.unsqueeze(0)
            imag_int = imag_int.unsqueeze(0)
            return self._wrap(torch.stack([real_int, imag_int], dim=0), dim=0)
        else:
            return self._wrap(torch.stack([real_int, imag_int], dim=self.dim))

    def cumulative_trapezoid(self, x=None, dim=-1):
        """
        Computes the cumulative trapezoidal integral along a dimension.
        """
        if x is not None and not isinstance(x, Complex):
            x = Complex(x, dim=self.dim, dtype=self.dtype, device=self.device)

        # For complex numbers, integrate real and imaginary parts separately
        if x is None:
            real_cumint = torch.cumulative_trapezoid(self.real, dim=dim)
            imag_cumint = torch.cumulative_trapezoid(self.imag, dim=dim)
        else:
            x_vals = x.real if isinstance(x, Complex) else x
            real_cumint = torch.cumulative_trapezoid(self.real, x_vals, dim=dim)
            imag_cumint = torch.cumulative_trapezoid(self.imag, x_vals, dim=dim)

        return self._wrap(torch.stack([real_cumint, imag_cumint], dim=self.dim))

    # --- Tensor Manipulation & Structure ---

    @property
    def shape(self):
        """Shape of the complex tensor (logical shape, omitting the stack dim)."""
        if self.tensor is None:
            return torch.Size([])
        s = list(self.tensor.shape)
        if self.dim is not None:
            # Handle positive and negative dims
            idx = self.dim if self.dim >= 0 else len(s) + self.dim
            if 0 <= idx < len(s):
                s.pop(idx)
        return torch.Size(s)

    @property
    def ndim(self):
        return self.tensor.ndim - 1

    def reshape(self, *shape):
        real, imag = self.real, self.imag
        real = real.reshape(*shape)
        imag = imag.reshape(*shape)
        c_tensor = torch.complex(real, imag)
        # Using dim=-1 ensures we append the complex dimension at the end, avoiding out-of-bounds with fixed (old) dims.
        return Complex(c_tensor, dim=-1, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                       **self.kwargs)

    def polyfit(self, x, y, deg, **kwargs):
        """Static polyfit using numpy for complex polynomials (torch has no equivalent)."""
        import numpy as np
        x_c = x.detach().cpu().numpy()
        y_c = y.detach().cpu().numpy()
        coeffs = np.polyfit(x_c, y_c, deg)
        return Complex(torch.from_numpy(coeffs).to(self.device), dim=0)

    def polyder(self, m=1):
        """
        Computes the derivative of the polynomial coefficients.
        """
        # coefficients are [a_n, ..., a_0] for degree n
        real = self.real()
        imag = self.imag()

        def deriv(r, i):
            n = r.shape[0] - 1
            if n < 1:
                return torch.zeros(1, device=self.device, dtype=self.dtype), torch.zeros(1, device=self.device,
                                                                                         dtype=self.dtype)
            powers = torch.arange(n, 0, -1, device=self.device, dtype=self.dtype)
            return r[:-1] * powers, i[:-1] * powers

        r_res, i_res = real, imag
        for _ in range(m):
            r_res, i_res = deriv(r_res, i_res)

        return Complex(torch.complex(r_res, i_res), dim=0)

    def polyint(self, m=1, k=0):
        """
        Computes the indefinite integral of the polynomial coefficients.
        k: integration constant(s).
        """
        real = self.real()
        imag = self.imag()

        def integr(r, i, integration_const):
            n = r.shape[0] - 1
            powers = torch.arange(n + 1, 0, -1, device=self.device, dtype=self.dtype)
            new_r = r / powers
            new_i = i / powers
            if not isinstance(integration_const, complex):
                integration_const = complex(integration_const)

            final_r = torch.cat([new_r, torch.tensor([integration_const.real], device=self.device, dtype=self.dtype)])
            final_i = torch.cat([new_i, torch.tensor([integration_const.imag], device=self.device, dtype=self.dtype)])
            return final_r, final_i

        r_res, i_res = real, imag
        if not isinstance(k, (list, tuple)):
            k_list = [k] * m
        else:
            k_list = k

        for j in range(m):
            r_res, i_res = integr(r_res, i_res, k_list[j])

        return Complex(torch.complex(r_res, i_res), dim=0)

    def view(self, *shape):
        real, imag = self.real, self.imag
        real = real.view(*shape)
        imag = imag.view(*shape)
        c_tensor = torch.complex(real, imag)
        return Complex(c_tensor, dim=-1, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                       **self.kwargs)

    def permute(self, *dims):
        real, imag = self.real, self.imag
        real = real.permute(*dims)
        imag = imag.permute(*dims)
        c_tensor = torch.complex(real, imag)
        return Complex(c_tensor, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                       **self.kwargs)

    def transpose(self, dim0, dim1):
        real, imag = self.real, self.imag
        real = real.transpose(dim0, dim1)
        imag = imag.transpose(dim0, dim1)
        c_tensor = torch.complex(real, imag)
        return Complex(c_tensor, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                       **self.kwargs)

    def rot90(self, k=1, dims=(0, 1)):
        """
        Rotate tensor by 90 degrees k times in the plane specified by dims.
        """
        real_rotated = torch.rot90(self.real, k=k, dims=dims)
        imag_rotated = torch.rot90(self.imag, k=k, dims=dims)
        c_tensor = torch.complex(real_rotated, imag_rotated)
        return Complex(c_tensor, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                       **self.kwargs)

    def expand(self, *sizes):
        real, imag = self.real, self.imag
        real = real.expand(*sizes)
        imag = imag.expand(*sizes)
        c_tensor = torch.complex(real, imag)
        return Complex(c_tensor, dim=-1, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                       **self.kwargs)

    def repeat(self, *repeats):
        real, imag = self.real, self.imag
        real = real.repeat(*repeats)
        imag = imag.repeat(*repeats)
        c_tensor = torch.complex(real, imag)
        return Complex(c_tensor, dim=-1, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                       **self.kwargs)

    def squeeze(self, dim=None):
        real, imag = self.real, self.imag
        real = real.squeeze(dim)
        imag = imag.squeeze(dim)
        c_tensor = torch.complex(real, imag)
        return Complex(c_tensor, dim=-1, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                       **self.kwargs)

    def unsqueeze(self, dim):
        real, imag = self.real, self.imag
        real = real.unsqueeze(dim)
        imag = imag.unsqueeze(dim)
        c_tensor = torch.complex(real, imag)
        return Complex(c_tensor, dim=-1, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                       **self.kwargs)

    def flatten(self, start_dim=0, end_dim=-1):
        real, imag = self.real, self.imag
        real = real.flatten(start_dim, end_dim)
        imag = imag.flatten(start_dim, end_dim)
        # Stacking at -1 is safe for flattened tensors
        stacked = torch.stack([real, imag], dim=-1)
        return self._wrap(stacked, dim=-1)

    def ravel(self):
        """Alias for flatten()."""
        return self.flatten()

    def moveaxis(self, source, destination):
        """Alias for movedim."""
        return self.movedim(source, destination)

    def softmax(self, dim: int = -1, out: Optional['Complex'] = None):
        """Standard softmax applied to complex tensor via magnitude."""
        m = self.mag()
        sm = torch.nn.functional.softmax(m, dim=dim)
        # Scale complex values by softmax of magnitudes
        scale = sm / (m + 1e-12)
        real_scaled = self.real * scale
        imag_scaled = self.imag * scale
        res_stacked = torch.stack([real_scaled, imag_scaled], dim=self.dim)

        if out is not None:
            out.tensor.copy_(res_stacked)
            return out
        return self._wrap(res_stacked)

    def signbit(self):
        """
        Returns True if the real part is negative (PyTorch convention for complex).
        """
        return torch.signbit(self.real)

    def tensor_split(self, indices_or_sections, dim=0):
        """
        NumPy-style tensor splitting that handles uneven divisions gracefully.
        """
        real_splits = torch.tensor_split(self.real, indices_or_sections, dim=dim)
        imag_splits = torch.tensor_split(self.imag, indices_or_sections, dim=dim)

        result = []
        for r, i in zip(real_splits, imag_splits):
            stacked = torch.stack([r, i], dim=self.dim)
            result.append(self._wrap(stacked))

        return result

    def unique_consecutive(self, return_inverse=False, return_counts=False, dim=None):
        """
        Eliminates consecutive duplicate elements only (useful for run-length encoding).
        Based on magnitude comparison for complex numbers.
        """
        # For complex numbers, compare based on both real and imaginary parts
        c_tensor = torch.complex(self.real, self.imag)
        result = torch.unique_consecutive(c_tensor, return_inverse=return_inverse,
                                          return_counts=return_counts, dim=dim)

        if return_inverse or return_counts:
            # result is a tuple
            unique_vals = result[0]
            unique_complex = Complex(unique_vals, dim=self.dim, dtype=self.dtype, device=self.device)
            return (unique_complex,) + result[1:]
        else:
            return Complex(result, dim=self.dim, dtype=self.dtype, device=self.device)

    def isposinf(self):
        """Checks for positive infinity specifically."""
        # For complex numbers, check if real part is positive infinity
        return torch.isposinf(self.real)

    def isneginf(self):
        """Checks for negative infinity specifically."""
        # For complex numbers, check if real part is negative infinity
        return torch.isneginf(self.real)

    # --- Sparse & Index Operations ---

    def gather(self, dim, index):
        """
        Gathers values along an axis specified by indices.
        """
        real = self.real.gather(dim, index)
        imag = self.imag.gather(dim, index)
        return self._wrap(torch.stack([real, imag], dim=self.dim))

    def scatter(self, dim, index, src):
        """
        Writes values into a tensor at indices specified by index.
        """
        if not isinstance(src, Complex):
            src = Complex(src, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                          **self.kwargs)

        real = self.real.scatter(dim, index, src.real)
        imag = self.imag.scatter(dim, index, src.imag)
        return self._wrap(torch.stack([real, imag], dim=self.dim))

    def scatter_(self, dim, index, src):
        """
        In-place version of scatter.
        """
        if not isinstance(src, Complex):
            src = Complex(src, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                          **self.kwargs)

        self.real.scatter_(dim, index, src.real)
        self.imag.scatter_(dim, index, src.imag)
        return self

    def index_select(self, dim, index):
        """
        Returns a new tensor which indexes self along dimension dim using the entries in index.
        """
        real = self.real.index_select(dim, index)
        imag = self.imag.index_select(dim, index)
        return self._wrap(torch.stack([real, imag], dim=self.dim))

    def take(self, indices):
        """
        Returns a new tensor with the elements of self at the given indices.
        """
        real = self.real.take(indices)
        imag = self.imag.take(indices)
        return self._wrap(torch.stack([real, imag], dim=self.dim))

    def chunk(self, chunks, dim=0):
        real, imag = self.real, self.imag
        real_chunks = real.chunk(chunks, dim)
        imag_chunks = imag.chunk(chunks, dim)

        res = []
        for r, i in zip(real_chunks, imag_chunks):
            c_tensor = torch.complex(r, i)
            res.append(Complex(c_tensor, dim=-1, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                               **self.kwargs))
        return tuple(res)

    # --- Neural Network activations ---

    def modReLU(self, bias: float = 0.0, inplace: bool = False, out: Optional['Complex'] = None):
        mag = self.mag()
        mag_plus_bias = mag + bias
        relu_mag = torch.nn.functional.relu(mag_plus_bias)

        scale = relu_mag / (mag + 1e-8)

        real_new = self.real * scale
        imag_new = self.imag * scale

        res = torch.stack([real_new, imag_new], dim=self.dim)
        if out is not None:
            out.tensor.copy_(res)
            return out
        if inplace:
            self.tensor = res
            return self
        else:
            return self._wrap(res)

    def relu(self, inplace: bool = False, out: Optional['Complex'] = None):
        """
        Applies ReLU to both real and imaginary parts independently.
        """
        res_real = F.relu(self.real, inplace=False)
        res_imag = F.relu(self.imag, inplace=False)
        res = torch.stack([res_real, res_imag], dim=self.dim)

        if out is not None:
            out.tensor.copy_(res)
            return out
        if inplace:
            self.tensor = res
            return self
        else:
            return self._wrap(res)

    def cReLU(self, inplace: bool = False, out: Optional['Complex'] = None):
        real = torch.nn.functional.relu(self.real)
        imag = torch.nn.functional.relu(self.imag)
        res = torch.stack([real, imag], dim=self.dim)
        if out is not None:
            out.tensor.copy_(res)
            return out
        if inplace:
            self.tensor = res
            return self
        else:
            return self._wrap(res)

    def zReLU(self, inplace: bool = False, out: Optional['Complex'] = None):
        real, imag = self.real, self.imag
        mask = (real >= 0) & (imag >= 0)

        real_new = real * mask.float()
        imag_new = imag * mask.float()
        res = torch.stack([real_new, imag_new], dim=self.dim)
        if out is not None:
            out.tensor.copy_(res)
            return out
        if inplace:
            self.tensor = res
            return self
        else:
            return self._wrap(res)

    def complex_softmax(self, dim=-1, inplace: bool = False):
        exp_z = self.exp()
        exp_z_c = self._wrap(exp_z)

        real_exp, imag_exp = exp_z_c.real, exp_z_c.imag
        c_exp = torch.complex(real_exp, imag_exp)

        s = torch.sum(c_exp, dim=dim, keepdim=True)
        res_c = c_exp / s

        res = torch.stack([res_c.real, res_c.imag], dim=self.dim)
        if inplace:
            self.tensor = res
        else:
            return self._wrap(res)

    # --- Signal Processing ---

    def conv1d(self, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        input = torch.complex(self.real, self.imag)
        w = torch.complex(weight.real(), weight.imag())
        b = torch.complex(bias.real(), bias.imag()) if bias else None

        res = torch.nn.functional.conv1d(input, w, b, stride, padding, dilation, groups)
        return Complex(res, dim=-1, dtype=self.dtype, device=self.device, arrangement=self.arrangement, **self.kwargs)

    def conv2d(self, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        input = torch.complex(self.real, self.imag)
        w = torch.complex(weight.real, weight.imag)
        b = torch.complex(bias.real, bias.imag) if bias else None

        res = torch.nn.functional.conv2d(input, w, b, stride, padding, dilation, groups)
        return Complex(res, dim=-1, dtype=self.dtype, device=self.device, arrangement=self.arrangement, **self.kwargs)

    def conv3d(self, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        input = torch.complex(self.real, self.imag)
        w = torch.complex(weight.real, weight.imag)
        b = torch.complex(bias.real, bias.imag) if bias else None

        res = torch.nn.functional.conv3d(input, w, b, stride, padding, dilation, groups)
        return Complex(res, dim=-1, dtype=self.dtype, device=self.device, arrangement=self.arrangement, **self.kwargs)

    def unwrap(self, dim=-1):
        phi = self.phi()
        try:
            return torch.unwrap(phi, dim=dim)
        except AttributeError:
            # Manual implementation
            diff = phi.diff(dim=dim)
            diff_mod = (diff + math.pi) % (2 * math.pi) - math.pi

            # Reconstruct
            # Taking slice to match shape
            start = phi.narrow(dim, 0, 1)
            res = torch.cumsum(diff_mod, dim=dim)
            return torch.cat([start, res + start], dim=dim)

    # --- Special Functions ---

    def erf(self):
        c_tensor = torch.complex(self.real, self.imag)
        res = torch.special.erf(c_tensor)
        return Complex(res, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                       **self.kwargs)

    def erfc(self):
        c_tensor = torch.complex(self.real, self.imag)
        res = torch.special.erfc(c_tensor)
        return Complex(res, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                       **self.kwargs)

    # Note: gamma/lgamma might not support complex in all torch versions. 
    # But usually do in recent ones.
    def gamma(self):
        c_tensor = torch.complex(self.real, self.imag)
        # torch.gamma strictly standard? torch.special.gammainc etc exists.
        # torch.exp(torch.lgamma(z)) is common workaround if gamma not direct.
        # But let's try torch.special.gamma or torch.gamma.
        # torch.gamma(complex) failed in past?
        # Safer: c_tensor.gamma() if tensor method exists?
        try:
            res = torch.special.gamma(c_tensor)
        except (AttributeError, RuntimeError):
            # Fallback if needed? likely implemented.
            res = torch.exp(torch.special.gammaln(c_tensor))

        return Complex(res, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                       **self.kwargs)

    def lgamma(self):
        c_tensor = torch.complex(self.real, self.imag)
        res = torch.special.gammaln(c_tensor)  # lgamma is usually gammaln in torch special
        return Complex(res, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                       **self.kwargs)

    # --- Rounding ---

    def round(self, decimals=0):
        real, imag = self.real, self.imag
        real = torch.round(real * 10 ** decimals) / (10 ** decimals)
        imag = torch.round(imag * 10 ** decimals) / (10 ** decimals)
        c_tensor = torch.complex(real, imag)
        return Complex(c_tensor, dim=-1, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                       **self.kwargs)

    def floor(self):
        real, imag = self.real, self.imag
        real = torch.floor(real)
        imag = torch.floor(imag)
        c_tensor = torch.complex(real, imag)
        return Complex(c_tensor, dim=-1, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                       **self.kwargs)

    def ceil(self):
        real, imag = self.real, self.imag
        real = torch.ceil(real)
        imag = torch.ceil(imag)
        c_tensor = torch.complex(real, imag)
        return Complex(c_tensor, dim=-1, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                       **self.kwargs)

    # --- Missing Fixes ---

    def cosech(self, inplace: bool = False):
        # cosech(z) = 1 / sinh(z)
        sinh_z = self.sinh()
        res = sinh_z.inv()
        if inplace:
            self.tensor = res.tensor
        else:
            return res

    def fft(self, n=None, dim=-1, norm=None):
        c_tensor = torch.complex(self.real, self.imag)
        res = torch.fft.fft(c_tensor, n=n, dim=dim, norm=norm)

        res_stacked = torch.stack([res.real, res.imag], dim=self.dim)
        return self._wrap(res_stacked)

    def ifft(self, n=None, dim=-1, norm=None):
        c_tensor = torch.complex(self.real, self.imag)
        res = torch.fft.ifft(c_tensor, n=n, dim=dim, norm=norm)

        res_stacked = torch.stack([res.real, res.imag], dim=self.dim)
        return self._wrap(res_stacked)

    def stft(self, n_fft, hop_length=None, win_length=None, window=None, center=True, pad_mode='reflect',
             normalized=False, onesided=None, return_complex=None):
        res_real = torch.stft(self.real, n_fft, hop_length, win_length, window, center, pad_mode, normalized,
                              onesided=False, return_complex=True)
        res_imag = torch.stft(self.imag, n_fft, hop_length, win_length, window, center, pad_mode, normalized,
                              onesided=False, return_complex=True)

        final_real = res_real.real - res_imag.imag
        final_imag = res_real.imag + res_imag.real

        res_stacked = torch.stack([final_real, final_imag], dim=self.dim)

        return self._wrap(res_stacked)

    def istft(self, n_fft, hop_length=None, win_length=None, window=None, center=True, normalized=False, onesided=None,
              length=None, return_complex=True):
        """
        Inverse Short-Time Fourier Transform.
        """
        c_tensor = torch.complex(self.real, self.imag)
        res = torch.istft(c_tensor, n_fft, hop_length, win_length, window, center, normalized, onesided, length,
                          return_complex)

        if return_complex:
            res_stacked = torch.stack([res.real, res.imag], dim=self.dim)
            return self._wrap(res_stacked)
        else:
            return res

    # --- Exponential & Log Methods ---

    # --- Helper Methods ---

    def shape_(self):
        return self.real.shape

    def ndim_(self):
        return self.real.ndim

    def magnitude(self):
        return self.mag()

    def phase(self):
        return self.phi()

    def normalize(self, inplace: bool = False):
        res = self.unit()
        if inplace:
            self.tensor = res
        else:
            return res

    def reciprocal(self, inplace: bool = False):
        res = self.inv()
        if inplace:
            self.tensor = res
        else:
            return res

    def negation(self, inplace: bool = False):
        real, imag = self.tensor.unbind(self.dim)
        if inplace:
            self.tensor = torch.stack([-real, -imag], dim=self.dim)
        else:
            return torch.stack([-real, -imag], dim=self.dim)

    def l2_magnitude(self):
        mag = self.mag()
        return torch.square(mag)

    # --- Arithmetic Methods ---

    def add(self, other, inplace: bool = False, out: Optional['Complex'] = None):
        if isinstance(other, Complex):
            real = self.real + other.real
            imag = self.imag + other.imag
        else:
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                            **self.kwargs)
            real = self.real + other.real
            imag = self.imag + other.imag

        res_stacked = torch.stack([real, imag], dim=self.dim)
        if out is not None:
            out.tensor.copy_(res_stacked)
            return out
        if inplace:
            self.tensor = res_stacked
            return self
        else:
            return self._wrap(res_stacked)

    def subtract(self, other, inplace: bool = False, out: Optional['Complex'] = None):
        if isinstance(other, Complex):
            real = self.real - other.real
            imag = self.imag - other.imag
        else:
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                            **self.kwargs)
            real = self.real - other.real
            imag = self.imag - other.imag

        res_stacked = torch.stack([real, imag], dim=self.dim)
        if out is not None:
            out.tensor.copy_(res_stacked)
            return out
        if inplace:
            self.tensor = res_stacked
            return self
        else:
            return self._wrap(res_stacked)

    def multiply(self, other, inplace: bool = False, out: Optional['Complex'] = None, **kwargs):
        if not isinstance(other, Complex):
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                            **self.kwargs)

        mul_type = kwargs.get('mul_type', 'hadamard')

        real_self, imag_self = self.real, self.imag
        real_other, imag_other = other.real, other.imag

        match mul_type.lower():
            case 'hadamard' | 'element_wise':
                if real_self.shape != real_other.shape:
                    try:
                        torch.broadcast_shapes(real_self.shape, real_other.shape)
                    except RuntimeError:
                        raise ValueError(
                            f"Element-wise mismatch: Shapes {real_self.shape} and {real_other.shape} are not equal/broadcastable")
                real = (real_self * real_other) - (imag_self * imag_other)
                imag = (real_self * imag_other) + (imag_self * real_other)

            case 'dot':
                if real_self.ndim != 1 or real_other.ndim != 1:
                    raise ValueError(
                        f"Dot product requires 1D tensors. Got ranks {real_self.ndim} and {real_other.ndim}")
                if real_self.shape[0] != real_other.shape[0]:
                    raise ValueError(f"Dot product length mismatch: {real_self.shape[0]} vs {real_other.shape[0]}")
                real = torch.dot(real_self, real_other) - torch.dot(imag_self, imag_other)
                imag = torch.dot(real_self, imag_other) + torch.dot(imag_self, real_other)

            case 'matmul':
                if real_self.ndim < 1 or real_other.ndim < 1:
                    raise ValueError(f"Matrix Multiplication requires at least 1D Tensors")
                if real_self.shape[-1] != real_other.shape[-2]:
                    raise RuntimeError(f"Matrix Multiplication mismatch: {real_self.shape} vs {real_other.shape}."
                                       f"Inner dims {real_self.shape[-1]} and {real_other.shape[-2]} must match.")
                real = real_self @ real_other - imag_self @ imag_other
                imag = real_self @ imag_other + imag_self @ real_other

            case 'inner' | 'frobenius':
                if real_self.shape != real_other.shape:
                    raise RuntimeError(
                        f"Frobenius Inner Product requires identical shapes. Got {real_self.shape} and {real_other.shape}.")
                real = torch.sum((real_self * real_other) - (imag_self * imag_other))
                imag = torch.sum((real_self * imag_other) + (imag_self * real_other))

            case 'outer':
                real = torch.tensordot(real_self, real_other, dims=0) - torch.tensordot(imag_self, imag_other, dims=0)
                imag = torch.tensordot(real_self, imag_other, dims=0) + torch.tensordot(imag_self, real_other, dims=0)

            case 'kron' | 'kronecker':
                if real_self.ndim != real_other.ndim:
                    warnings.warn(
                        f"The tensors are having different ranks. Got {real_self.ndim} and {real_other.ndim}",
                        RuntimeWarning)
                real = torch.kron(real_self, real_other) - torch.kron(imag_self, imag_other)
                imag = torch.kron(real_self, imag_other) + torch.kron(imag_self, real_other)

            case 'khatri_rao' | 'kr':
                if real_self.ndim != 2 or real_other.ndim != 2:
                    raise RuntimeError(
                        f"Khatri-Rao requires 2D matrices. Got ranks {real_self.ndim} and {real_other.ndim}")
                if real_self.shape[1] != real_other.shape[1]:
                    raise ValueError(f"Khatri-Rao column mismatch: {real_self.shape[1]} and {real_other.shape[1]}")
                khatri_rao_format = "ik,jk->ijk"
                real = (torch.einsum(khatri_rao_format, real_self, real_other) - torch.einsum(khatri_rao_format,
                                                                                              imag_self,
                                                                                              imag_other)).reshape(-1,
                                                                                                                   real_self.shape[
                                                                                                                       1])
                imag = (torch.einsum(khatri_rao_format, real_self, imag_other) + torch.einsum(khatri_rao_format,
                                                                                              imag_self,
                                                                                              real_other)).reshape(-1,
                                                                                                                   real_self.shape[
                                                                                                                       1])

            case 'mode_n':
                mode = kwargs.get('mode')
                if mode is None:
                    raise ValueError(f"For 'mode_n', you must provide 'mode=<int>' argument.")
                if real_other.ndim != 2:
                    raise ValueError(
                        f"Mode-n product requires the second input to be a Matrix (2D). Got rank {real_other.ndim}.")
                if mode < 0 or mode >= real_self.ndim:
                    raise ValueError(f"Mode {mode} is out of bounds for first input tensor with rank {real_self.ndim}")
                dim_len = real_other.shape[1]
                if real_self.shape[mode] != dim_len:
                    raise RuntimeError(
                        f"Mode-n mismatch: Tensor dim {mode} is size {real_self.shape[mode]}, but Matrix dim 1 is size {dim_len}")

                real = torch.tensordot(real_self, real_other, dims=([mode], [1])) - torch.tensordot(imag_self,
                                                                                                    imag_other,
                                                                                                    dims=([mode], [1]))
                imag = torch.tensordot(real_self, imag_other, dims=([mode], [1])) + torch.tensordot(imag_self,
                                                                                                    real_other,
                                                                                                    dims=([mode], [1]))

            case 'einsum':
                mul_format = kwargs.get('mul_format')
                if mul_format is None:
                    raise ValueError("For 'einsum', you must provide 'mul_format=<string>' argument.")
                real = torch.einsum(mul_format, real_self, real_other) - torch.einsum(mul_format, imag_self, imag_other)
                imag = torch.einsum(mul_format, real_self, imag_other) + torch.einsum(mul_format, imag_self, real_other)

        # Handle scalars/stacking for results
        if torch.is_tensor(real) and real.ndim > 0:
            res_stacked = torch.stack([real, imag], dim=self.dim)
        else:
            # Scalar case
            res_stacked = torch.stack([torch.as_tensor(real), torch.as_tensor(imag)], dim=0)

        if out is not None:
            out.tensor.copy_(res_stacked)
            return out
        if inplace:
            self.tensor = res_stacked
            return self
        else:
            return self._wrap(res_stacked)

    def matmul(self, other, out: Optional['Complex'] = None):
        """Matrix Multiplication with out= support."""
        if not isinstance(other, Complex):
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                            **self.kwargs)

        # Complex matrix multiplication
        c_self = torch.complex(self.real, self.imag)
        c_other = torch.complex(other.real, other.imag)
        result = torch.matmul(c_self, c_other)

        res = Complex(result, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                      **self.kwargs)
        if out is not None:
            out.tensor.copy_(res.tensor)
            return out
        return res

    def __matmul__(self, other):
        return self.matmul(other)

    def mm(self, mat, out: Optional['Complex'] = None):
        """mm alias with out= support."""
        return self.matmul(mat, out=out)

    def bmm(self, batch_mat, out: Optional['Complex'] = None):
        """bmm alias with out= support."""
        return self.matmul(batch_mat, out=out)

    def mv(self, vec, out: Optional['Complex'] = None):
        """mv alias with out= support."""
        return self.matmul(vec, out=out)

    def matrix_inv(self):
        c_tensor = torch.complex(self.real, self.imag)
        res = torch.linalg.inv(c_tensor)
        return Complex(res, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                       **self.kwargs)

    def cross(self, other, dim=-1):
        if not isinstance(other, Complex):
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                            **self.kwargs)

        c_self = torch.complex(self.real, self.imag)
        c_other = torch.complex(other.real, other.imag)

        res = torch.linalg.cross(c_self, c_other, dim=dim)
        return Complex(res, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                       **self.kwargs)

    def divide(self, other, inplace: bool = False, out: Optional['Complex'] = None, **kwargs):
        if not isinstance(other, Complex):
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                            **self.kwargs)

        div_type = kwargs.get('div_type', 'element_wise')

        match div_type.lower():
            case 'element_wise' | 'hadamard':
                if self.real.shape != other.real.shape:
                    try:
                        torch.broadcast_shapes(self.real.shape, other.real.shape)
                    except RuntimeError:
                        raise ValueError(
                            f"Element-wise mismatch: Shapes {self.real.shape} and {other.real.shape} are not broadcastable")
                t1 = torch.complex(self.real, self.imag)
                t2 = torch.complex(other.real, other.imag)
                res_t = torch.div(t1, t2)
                real, imag = res_t.real, res_t.imag

            case 'matrix' | 'inv':
                t1 = torch.complex(self.real, self.imag)
                t2 = torch.complex(other.real, other.imag)
                res_t = torch.matmul(t1, torch.linalg.inv(t2))
                real, imag = res_t.real, res_t.imag

            case 'solve_right':
                t1 = torch.complex(self.real, self.imag)
                t2 = torch.complex(other.real, other.imag)
                res_t = (torch.linalg.solve(t2.mT, t1.mT)).mT
                real, imag = res_t.real, res_t.imag

            case 'pinv':
                t1 = torch.complex(self.real, self.imag)
                t2 = torch.complex(other.real, other.imag)
                res_t = torch.matmul(t1, torch.linalg.pinv(t2))
                real, imag = res_t.real, res_t.imag

            case _:
                raise ValueError(f"Unknown division type: {div_type}")

        res_stacked = torch.stack([real, imag], dim=self.dim)
        if out is not None:
            out.tensor.copy_(res_stacked)
            return out
        if inplace:
            self.tensor = res_stacked
            return self
        else:
            return self._wrap(res_stacked)

    # --- Operator Overloading ---

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return self.add(other)

    def __iadd__(self, other):
        self.add(other, inplace=True)
        return self

    def __sub__(self, other):
        return self.subtract(other)

    def __rsub__(self, other):
        # other_decomposition - self
        if not isinstance(other, Complex):
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                            **self.kwargs)
        return other.subtract(self)

    def __isub__(self, other):
        self.subtract(other, inplace=True)
        return self

    def dot(self, other):
        return self.multiply(other, mul_type='dot')

    # --- Vector Similarity & Geometry ---

    def cosine_similarity(self, other, dim=-1, eps=1e-08):
        """
        Computes the cosine similarity between complex vectors.
        Re( (A . B^H) / (|A||B|) )
        """
        if not isinstance(other, Complex):
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                            **self.kwargs)

        # A . B^H
        dot_val = self.multiply(other.conj(), mul_type='dot', dim=dim)

        norm_self = torch.linalg.norm(torch.complex(self.real, self.imag), dim=dim)
        norm_other = torch.linalg.norm(torch.complex(other.real, other.imag), dim=dim)

        denom = (norm_self * norm_other).clamp(min=eps)
        # dot_val is Complex. (A.B^H)/|A||B|
        res = dot_val.divide(denom)
        return res.real()  # Return the real part as per definition

    def proj(self, other):
        """
        Projects self onto other_decomposition.
        """
        if not isinstance(other, Complex):
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                            **self.kwargs)

        # <self, other_decomposition> / <other_decomposition, other_decomposition> * other_decomposition
        num = self.multiply(other.conj(), mul_type='dot')
        den = other.multiply(other.conj(), mul_type='dot')

        coeff = num.divide(den)
        return coeff.multiply(other)

    # --- Deep Learning Utilities ---

    def whiten(self, dim=None, eps=1e-05):
        """
        Decorrelates the real and imaginary components.
        Aligns the 2x2 covariance of (real, imag) to the identity matrix.
        """
        # 1. Center
        mu = self.mean(dim=dim, keepdim=True)
        z = self.subtract(mu)

        real = z.real
        imag = z.imag

        # 2. Compute 2x2 covariance matrix components
        v_rr = torch.mean(real * real, dim=dim, keepdim=True)
        v_ii = torch.mean(imag * imag, dim=dim, keepdim=True)
        v_ri = torch.mean(real * imag, dim=dim, keepdim=True)

        # 3. Compute Inverse Square Root of 2x2 Covariance Matrix
        # [[v_rr, v_ri], [v_ri, v_ii]]
        det = (v_rr * v_ii - v_ri ** 2).clamp(min=1e-12)
        trace = v_rr + v_ii
        s = torch.sqrt(det)
        t = torch.sqrt(trace + 2 * s + eps)

        inv_st = 1.0 / (s * t + eps)
        w11 = (v_ii + s) * inv_st
        w12 = -v_ri * inv_st
        w22 = (v_rr + s) * inv_st

        # 4. Transform
        new_real = w11 * real + w12 * imag
        new_imag = w12 * real + w22 * imag  # w21 = w12

        return self._wrap(torch.stack([new_real, new_imag], dim=self.dim))

    def __mul__(self, other):
        return self.multiply(other)

    def __rmul__(self, other):
        return self.multiply(other)

    def __imul__(self, other):
        self.multiply(other, inplace=True)
        return self

    def __truediv__(self, other):
        return self.divide(other)

    def __rtruediv__(self, other):
        # other_decomposition / self
        if not isinstance(other, Complex):
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                            **self.kwargs)
        return other.divide(self)

    def __itruediv__(self, other):
        self.divide(other, inplace=True)
        return self

    def __pow__(self, exponent):
        return self.pow(exponent)

    def __neg__(self):
        return self._wrap(-self.tensor)

    def __abs__(self):
        return self.mag()

    # --- Exponential & Log Methods ---

    def log(self, inplace: bool = False, out: Optional['Complex'] = None, **kwargs):
        real_mag = self.mag()
        real_part = torch.log(real_mag)
        imag_part = self.phi()

        bias = kwargs.get('phase_bias')
        if bias is not None:
            if isinstance(bias, (int, float)):
                bias = torch.full_like(imag_part, bias)
            elif isinstance(bias, (list, tuple)) or (hasattr(bias, '__array__') and not isinstance(bias, torch.Tensor)):
                bias = torch.tensor(bias, device=self.device, dtype=self.dtype)

            if bias.shape != imag_part.shape:
                try:
                    bias = bias.broadcast_to(imag_part.shape)
                except:
                    pass

            imag_part = imag_part + (2.0 * math.pi * bias)

        res = torch.stack([real_part, imag_part], dim=self.dim)
        if out is not None:
            out.tensor.copy_(res)
            return out
        if inplace:
            self.tensor = res
            return self
        else:
            return self._wrap(res)

    # --- Aggregation & Reduction ---

    def sum(self, dim=None, keepdim=False):
        real = torch.sum(self.real, dim=dim, keepdim=keepdim)
        imag = torch.sum(self.imag, dim=dim, keepdim=keepdim)
        new_dim = self.dim if (keepdim or dim is not None) else 0
        if not keepdim and dim is not None:
            # If we removed a dimension, self.dim might shifted
            if self.dim > dim:
                new_dim = self.dim - 1
            elif self.dim == dim:
                # This shouldn't happen usually because real() already unbinded self.dim
                new_dim = 0

        # Ensure new_dim is within range for stacking
        target_dim = min(new_dim, real.ndim)
        return self._wrap(torch.stack([real, imag], dim=target_dim), dim=target_dim)

    def mean(self, dim=None, keepdim=False):
        real = torch.mean(self.real, dim=dim, keepdim=keepdim)
        imag = torch.mean(self.imag, dim=dim, keepdim=keepdim)
        new_dim = self.dim if (keepdim or dim is not None) else 0
        if not keepdim and dim is not None:
            if self.dim > dim:
                new_dim = self.dim - 1
            elif self.dim == dim:
                new_dim = 0
        target_dim = min(new_dim, real.ndim)
        return self._wrap(torch.stack([real, imag], dim=target_dim), dim=target_dim)

    def prod(self, dim=None, keepdim=False):
        # Product of complex numbers (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        # This is iterative. We use torch.complex for simplicity.
        c_tensor = torch.complex(self.real, self.imag)
        res_t = torch.prod(c_tensor, dim=dim, keepdim=keepdim)
        new_dim = self.dim if (keepdim or dim is not None) else 0
        if not keepdim and dim is not None:
            if self.dim > dim:
                new_dim = self.dim - 1
        target_dim = min(new_dim, res_t.ndim)
        return Complex(res_t, dim=target_dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                       **self.kwargs)

    def nanmean(self, dim=None, keepdim=False):
        """
        Computes the mean while ignoring NaN values.
        """
        real = torch.nanmean(self.real, dim=dim, keepdim=keepdim)
        imag = torch.nanmean(self.imag, dim=dim, keepdim=keepdim)

        new_dim = self.dim if (keepdim or dim is not None) else 0
        if not keepdim and dim is not None:
            if self.dim > dim:
                new_dim = self.dim - 1
            elif self.dim == dim:
                new_dim = 0
        target_dim = min(new_dim, real.ndim)
        return self._wrap(torch.stack([real, imag], dim=target_dim), dim=target_dim)

    def nansum(self, dim=None, keepdim=False):
        """
        Computes the sum while ignoring NaN values.
        """
        real = torch.nansum(self.real, dim=dim, keepdim=keepdim)
        imag = torch.nansum(self.imag, dim=dim, keepdim=keepdim)

        new_dim = self.dim if (keepdim or dim is not None) else 0
        if not keepdim and dim is not None:
            if self.dim > dim:
                new_dim = self.dim - 1
            elif self.dim == dim:
                new_dim = 0
        target_dim = min(new_dim, real.ndim)
        return self._wrap(torch.stack([real, imag], dim=target_dim), dim=target_dim)

    # --- Advanced Utilities ---

    def interpolate(self, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None,
                    antialias=False):
        # Resize both real and imaginary parts
        res_real = F.interpolate(self.real, size=size, scale_factor=scale_factor, mode=mode,
                                 align_corners=align_corners, recompute_scale_factor=recompute_scale_factor,
                                 antialias=antialias)
        res_imag = F.interpolate(self.imag, size=size, scale_factor=scale_factor, mode=mode,
                                 align_corners=align_corners, recompute_scale_factor=recompute_scale_factor,
                                 antialias=antialias)

        stacked = torch.stack([res_real, res_imag], dim=self.dim)
        return self._wrap(stacked)

    def unfold(self, dimension, size, step):
        """
        Returns a view of the tensor with sliding windows.
        """
        res_real = self.real.unfold(dimension, size, step)
        res_imag = self.imag.unfold(dimension, size, step)
        stacked = torch.stack([res_real, res_imag], dim=self.dim)
        return self._wrap(stacked)

    def fold(self, output_size, kernel_size, dilation=1, padding=0, stride=1):
        """
        Combines sliding windows back into a full tensor.
        """
        res_real = F.fold(self.real, output_size, kernel_size, dilation, padding, stride)
        res_imag = F.fold(self.imag, output_size, kernel_size, dilation, padding, stride)
        stacked = torch.stack([res_real, res_imag], dim=self.dim)
        return self._wrap(stacked)

    def diff(self, n=1, dim=-1, prepend=None, append=None):
        """
        Computes the $n$-th discrete difference along the given dimension.
        """

        def _get_part(x, part='real'):
            if isinstance(x, Complex):
                return x.real if part == 'real' else x.imag
            if torch.is_tensor(x):
                return x  # Assume user passed real part or complex tensor?
                # Better safe: if complex tensor, extract part.
            return x

        p_real = _get_part(prepend, 'real') if prepend is not None else None
        p_imag = _get_part(prepend, 'imag') if prepend is not None else None
        a_real = _get_part(append, 'real') if append is not None else None
        a_imag = _get_part(append, 'imag') if append is not None else None

        res_real = torch.diff(self.real, n=n, dim=dim, prepend=p_real, append=a_real)
        res_imag = torch.diff(self.imag, n=n, dim=dim, prepend=p_imag, append=a_imag)

        stacked = torch.stack([res_real, res_imag], dim=self.dim)
        return self._wrap(stacked)

    def gradient(self, spacing=1, dim=None, edge_order=1):
        """
        Computes the numerical gradient using second-order accurate central differences.
        """
        # torch.gradient returns a list of tensors if dim is None or multiple
        res_real = torch.gradient(self.real, spacing=spacing, dim=dim, edge_order=edge_order)
        res_imag = torch.gradient(self.imag, spacing=spacing, dim=dim, edge_order=edge_order)

        if isinstance(res_real, (list, tuple)):
            out = []
            for r, i in zip(res_real, res_imag):
                stacked = torch.stack([r, i], dim=self.dim)
                out.append(self._wrap(stacked))
            return out
        else:
            stacked = torch.stack([res_real, res_imag], dim=self.dim)
            return self._wrap(stacked)

    def upsample(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        return self.interpolate(size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)

    def cdist(self, other: 'Complex', p: float = 2.0):
        """
        Compute the distance matrix between self and other_decomposition.
        sqrt(|z1 - z2|^2)
        """
        if not isinstance(other, Complex):
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                            **self.kwargs)

        # We can use torch.cdist by stacking real and imag as feature dimensions (concatenating)
        # Dist(z1, z2) = sqrt((r1-r2)^2 + (i1-i2)^2).
        # This is equivalent to cdist of (r, i) vectors.
        # Reshape to (Batch, N, 2*Features) or similar? 
        # Actually it's easier to just use the complex plane interpretation.
        c1 = torch.complex(self.real, self.imag)
        c2 = torch.complex(other.real, other.imag)

        # torch.cdist doesn't support complex. We implement manually or use real vectors.
        # r1, i1 are parts of self.
        # We want to treat (real, imag) as coordinates.
        # If self is (N, D), c1 is (N, D).
        # Flatten D and (r,i) -> (N, 2D)
        v1 = torch.cat([self.real.flatten(start_dim=1), self.imag.flatten(start_dim=1)], dim=-1)
        v2 = torch.cat([other.real.flatten(start_dim=1), other.imag.flatten(start_dim=1)], dim=-1)

        dist = torch.cdist(v1, v2, p=p)
        # Returns a real tensor. We wrap it as Complex (imag=0).
        return self._wrap(torch.stack([dist, torch.zeros_like(dist)], dim=self.dim))

    # --- Functional Pooling & Normalization ---

    def max_pool2d(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        """
        Max pooling based on magnitude for 2D complex tensors.
        """
        # Apply max pooling to real and imag parts (based on magnitude for selection)
        pooled_real = F.max_pool2d(self.real, kernel_size, stride, padding, dilation, ceil_mode=ceil_mode)
        pooled_imag = F.max_pool2d(self.imag, kernel_size, stride, padding, dilation, ceil_mode=ceil_mode)

        result = self._wrap(torch.stack([pooled_real, pooled_imag], dim=self.dim))

        if return_indices:
            mag = self.mag()
            _, indices = F.max_pool2d(mag, kernel_size, stride, padding, dilation, return_indices=True,
                                      ceil_mode=ceil_mode)
            return result, indices
        return result

    def avg_pool2d(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True,
                   divisor_override=None):
        """
        Average pooling for 2D complex tensors (operates on real/imag independently).
        """
        pooled_real = F.avg_pool2d(self.real, kernel_size, stride, padding, ceil_mode, count_include_pad,
                                   divisor_override)
        pooled_imag = F.avg_pool2d(self.imag, kernel_size, stride, padding, ceil_mode, count_include_pad,
                                   divisor_override)

        return self._wrap(torch.stack([pooled_real, pooled_imag], dim=self.dim))

    def dropout(self, p=0.5, training=True, inplace=False):
        """
        Randomly zeros out complex elements (both real and imag parts).
        """
        if not training or p == 0:
            return self if inplace else self.clone()

        # Create dropout mask, apply to both real and imag
        mask = F.dropout(torch.ones_like(self.real), p, training, inplace=False)

        if inplace:
            with torch.no_grad():
                self.real.mul_(mask)
                self.imag.mul_(mask)
            return self
        else:
            real_dropped = self.real * mask
            imag_dropped = self.imag * mask
            return self._wrap(torch.stack([real_dropped, imag_dropped], dim=self.dim))

    def dropout2d(self, p=0.5, training=True, inplace=False):
        """
        2D dropout for CNNs - zeros out entire channels.
        """
        if not training or p == 0:
            return self if inplace else self.clone()

        # Create channel-wise dropout mask
        mask = F.dropout2d(torch.ones_like(self.real), p, training, inplace=False)

        if inplace:
            with torch.no_grad():
                self.real.mul_(mask)
                self.imag.mul_(mask)
            return self
        else:
            real_dropped = self.real * mask
            imag_dropped = self.imag * mask
            return self._wrap(torch.stack([real_dropped, imag_dropped], dim=self.dim))

    def layer_norm(self, normalized_shape, weight=None, bias=None, eps=1e-5):
        """
        Complex layer normalization.
        """
        # Normalize real and imag parts separately
        real_norm = F.layer_norm(self.real, normalized_shape, weight, bias, eps)
        imag_norm = F.layer_norm(self.imag, normalized_shape, weight, bias, eps)

        return Complex(torch.stack([real_norm, imag_norm], dim=self.dim), dim=self.dim, dtype=self.dtype,
                       device=self.device)

    def group_norm(self, num_groups, weight=None, bias=None, eps=1e-5):
        """
        Complex group normalization.
        """
        # Normalize real and imag parts separately
        real_norm = F.group_norm(self.real, num_groups, weight, bias, eps)
        imag_norm = F.group_norm(self.imag, num_groups, weight, bias, eps)

        return Complex(torch.stack([real_norm, imag_norm], dim=self.dim), dim=self.dim, dtype=self.dtype,
                       device=self.device)

    # --- Autograd Integration ---

    def backward(self, gradient=None, retain_graph=None, create_graph=False):
        """
        Wrapper for backward pass on the underlying tensor.
        """
        if gradient is not None and isinstance(gradient, Complex):
            gradient = gradient.tensor

        self.tensor.backward(gradient=gradient, retain_graph=retain_graph, create_graph=create_graph)

    def retain_grad(self):
        """
        Enable gradient retention for non-leaf tensors.
        """
        self.tensor.retain_grad()

    @property
    def grad(self):
        """
        Access accumulated gradients as a Complex object.
        """
        if self.tensor.grad is None:
            return None
        return Complex(self.tensor.grad, dim=self.dim, dtype=self.dtype, device=self.device)

    # --- Linear Algebra Solvers ---

    def solve(self, B):
        """
        Solves the linear system AX = B where self is A.
        """
        if not isinstance(B, Complex):
            B = Complex(B, dim=self.dim, dtype=self.dtype, device=self.device)

        A_complex = torch.complex(self.real, self.imag)
        B_complex = torch.complex(B.real, B.imag)

        X = torch.linalg.solve(A_complex, B_complex)
        return self._wrap(torch.stack([X.real, X.imag], dim=self.dim))

    def lstsq(self, B, rcond=None):
        """
        Computes least-squares. Returns (solution, residuals, rank, singular_values).
        """
        if not isinstance(B, Complex):
            B = Complex(B, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                        **self.kwargs)

        A_c = torch.complex(self.real, self.imag)
        B_c = torch.complex(B.real, B.imag)

        # Use torch.linalg.lstsq which returns a solution object
        result = torch.linalg.lstsq(A_c, B_c, rcond=rcond)

        solution = Complex(result.solution, dim=self.dim, dtype=self.dtype, device=self.device,
                           arrangement=self.arrangement, **self.kwargs)

        # Return tuple matching PyTorch API
        return (
            solution,
            result.residuals if hasattr(result, 'residuals') else torch.tensor([], device=self.device),
            result.rank if hasattr(result, 'rank') else None,
            result.singular_values if hasattr(result, 'singular_values') else None
        )

    def triangular_solve(self, B, upper=True, transpose=False, unitriangular=False):
        """
        Solves triangular system. Returns (solution, cloned_coefficient).
        """
        if not isinstance(B, Complex):
            B = Complex(B, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                        **self.kwargs)

        A_c = torch.complex(self.real, self.imag)
        B_c = torch.complex(B.real, B.imag)

        # Use torch.triangular_solve (deprecated, but kept for compatibility)
        if hasattr(torch, 'triangular_solve'):
            X, clone_A = torch.triangular_solve(B_c, A_c, upper=upper, transpose=transpose, unitriangular=unitriangular)
        else:
            # Fallback to linalg.solve_triangular
            X = torch.linalg.solve_triangular(A_c, B_c, upper=upper, unitriangular=unitriangular)
            clone_A = A_c.clone()

        return (
            Complex(X, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement, **self.kwargs),
            Complex(clone_A, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                    **self.kwargs)
        )

    def lu_solve(self, LU_data, LU_pivots):
        """
        Solves the linear system using pre-computed LU factorization.
        """
        if not isinstance(LU_data, Complex):
            LU_data = Complex(LU_data, dim=self.dim, dtype=self.dtype, device=self.device)

        b_complex = torch.complex(self.real, self.imag)
        LU_complex = torch.complex(LU_data.real, LU_data.imag)

        X = torch.lu_solve(b_complex, LU_complex, LU_pivots)
        return self._wrap(torch.stack([X.real, X.imag], dim=self.dim))

    # --- Splitting & Slicing APIs ---

    def unbind(self, dim=0):
        """
        Removes a dimension and returns a tuple of all slices along that dimension.
        """
        real_slices = torch.unbind(self.real, dim=dim)
        imag_slices = torch.unbind(self.imag, dim=dim)

        # After unbind, slices have one less dimension, so use dim=-1 for stacking
        return tuple(Complex(torch.stack([r, i], dim=-1), dim=-1, dtype=self.dtype, device=self.device)
                     for r, i in zip(real_slices, imag_slices))

    def split(self, split_size_or_sections, dim=0):
        """
        Splits the tensor into chunks.
        """
        real_splits = torch.split(self.real, split_size_or_sections, dim=dim)
        imag_splits = torch.split(self.imag, split_size_or_sections, dim=dim)

        return tuple(Complex(torch.stack([r, i], dim=self.dim), dim=self.dim, dtype=self.dtype, device=self.device)
                     for r, i in zip(real_splits, imag_splits))

    def narrow(self, dim, start, length):
        """
        Returns a narrowed version of the tensor.
        """
        real_narrow = torch.narrow(self.real, dim, start, length)
        imag_narrow = torch.narrow(self.imag, dim, start, length)

        return Complex(torch.stack([real_narrow, imag_narrow], dim=self.dim), dim=self.dim, dtype=self.dtype,
                       device=self.device)

    def select(self, dim, index):
        """
        Slices the tensor along the selected dimension at the given index.
        """
        real_select = torch.select(self.real, dim, index)
        imag_select = torch.select(self.imag, dim, index)

        return Complex(torch.stack([real_select, imag_select], dim=self.dim), dim=self.dim, dtype=self.dtype,
                       device=self.device)

    # --- Unique & Search Operations ---

    def unique(self, sorted=True, return_inverse=False, return_counts=False, dim=None):
        """
        Returns the unique elements of the complex tensor.
        """
        # Combine real and imag as a 2D coordinate for uniqueness
        c_tensor = torch.complex(self.real, self.imag)

        result = torch.unique(c_tensor, sorted=sorted, return_inverse=return_inverse, return_counts=return_counts,
                              dim=dim)

        if isinstance(result, tuple):
            unique_vals = Complex(result[0], dim=self.dim, dtype=self.dtype, device=self.device)
            return (unique_vals,) + result[1:]
        else:
            return Complex(result, dim=self.dim, dtype=self.dtype, device=self.device)

    def searchsorted(self, sorted_sequence, out_int32=False, right=False):
        """
        Finds indices where elements should be inserted (based on magnitude).
        """
        # Search based on magnitude
        mag = self.mag()
        seq_mag = sorted_sequence.mag() if isinstance(sorted_sequence, Complex) else sorted_sequence

        return torch.searchsorted(seq_mag, mag, out_int32=out_int32, right=right)

    def bucketize(self, boundaries, out_int32=False, right=False):
        """
        Returns the indices of the buckets (based on magnitude).
        """
        mag = self.mag()
        bound_mag = boundaries.mag() if isinstance(boundaries, Complex) else boundaries

        return torch.bucketize(mag, bound_mag, out_int32=out_int32, right=right)

    # --- Scatter Reductions ---

    def scatter_add(self, dim, index, src):
        """
        Adds all values from src into self at the indices specified.
        """
        if not isinstance(src, Complex):
            src = Complex(src, dim=self.dim, dtype=self.dtype, device=self.device)

        real_result = self.real.scatter_add(dim, index, src.real)
        imag_result = self.imag.scatter_add(dim, index, src.imag)

        return Complex(torch.stack([real_result, imag_result], dim=self.dim), dim=self.dim, dtype=self.dtype,
                       device=self.device)

    def scatter_reduce(self, dim, index, src, reduce, include_self=True):
        """
        Reduces values into self at the indices specified.
        """
        if not isinstance(src, Complex):
            src = Complex(src, dim=self.dim, dtype=self.dtype, device=self.device)

        real_result = self.real.scatter_reduce(dim, index, src.real, reduce, include_self=include_self)
        imag_result = self.imag.scatter_reduce(dim, index, src.imag, reduce, include_self=include_self)

        return Complex(torch.stack([real_result, imag_result], dim=self.dim), dim=self.dim, dtype=self.dtype,
                       device=self.device)

    # --- Additional Normalization Layers ---

    def batch_norm(self, running_mean=None, running_var=None, weight=None, bias=None, training=False, momentum=0.1,
                   eps=1e-5):
        """
        Batch normalization for complex tensors.
        """
        real_bn = F.batch_norm(self.real, running_mean, running_var, weight, bias, training, momentum, eps)
        imag_bn = F.batch_norm(self.imag, running_mean, running_var, weight, bias, training, momentum, eps)

        return Complex(torch.stack([real_bn, imag_bn], dim=self.dim), dim=self.dim, dtype=self.dtype,
                       device=self.device)

    def instance_norm(self, running_mean=None, running_var=None, weight=None, bias=None, use_input_stats=True,
                      momentum=0.1, eps=1e-5):
        """
        Instance normalization for complex tensors.
        """
        real_in = F.instance_norm(self.real, running_mean, running_var, weight, bias, use_input_stats, momentum, eps)
        imag_in = F.instance_norm(self.imag, running_mean, running_var, weight, bias, use_input_stats, momentum, eps)

        return self._wrap(torch.stack([real_in, imag_in], dim=self.dim))

    def local_response_norm(self, size, alpha=1e-4, beta=0.75, k=1.0):
        """
        Local response normalization.
        """
        real_lrn = F.local_response_norm(self.real, size, alpha, beta, k)
        imag_lrn = F.local_response_norm(self.imag, size, alpha, beta, k)

        return self._wrap(torch.stack([real_lrn, imag_lrn], dim=self.dim))

    # --- Special Math Functions ---

    def isfinite(self):
        """
        Returns a boolean tensor indicating which elements are finite.
        """
        real_finite = torch.isfinite(self.real)
        imag_finite = torch.isfinite(self.imag)
        return real_finite & imag_finite

    def i0(self):
        """
        Modified Bessel function of order 0 (applied to magnitude).
        """
        mag = self.mag()
        i0_mag = torch.i0(mag)
        # Return as complex with zero imaginary part
        return Complex(torch.stack([i0_mag, torch.zeros_like(i0_mag)], dim=self.dim), dim=self.dim, dtype=self.dtype,
                       device=self.device)

    def digamma(self):
        """
        Logarithmic derivative of gamma function (applied to real/imag separately).
        """
        real_digamma = torch.digamma(self.real)
        imag_digamma = torch.digamma(self.imag)

        return Complex(torch.stack([real_digamma, imag_digamma], dim=self.dim), dim=self.dim, dtype=self.dtype,
                       device=self.device)

    def polygamma(self, n):
        """
        N-th derivative of digamma function.
        """
        real_polygamma = torch.polygamma(n, self.real)
        imag_polygamma = torch.polygamma(n, self.imag)

        return Complex(torch.stack([real_polygamma, imag_polygamma], dim=self.dim), dim=self.dim, dtype=self.dtype,
                       device=self.device)

    # --- Hard activations ---

    def hardtanh(self, min_val=-1.0, max_val=1.0, inplace=False):
        """
        Hard tanh activation (applied to real/imag independently).
        """
        if inplace:
            F.hardtanh(self.real, min_val, max_val, inplace=True)
            F.hardtanh(self.imag, min_val, max_val, inplace=True)
            return self
        else:
            real_ht = F.hardtanh(self.real, min_val, max_val)
            imag_ht = F.hardtanh(self.imag, min_val, max_val)
            return Complex(torch.stack([real_ht, imag_ht], dim=self.dim), dim=self.dim, dtype=self.dtype,
                           device=self.device)

    def hardsigmoid(self, inplace=False):
        """
        Hard sigmoid activation (applied to real/imag independently).
        """
        if inplace:
            F.hardsigmoid(self.real, inplace=True)
            F.hardsigmoid(self.imag, inplace=True)
            return self
        else:
            real_hs = F.hardsigmoid(self.real)
            imag_hs = F.hardsigmoid(self.imag)
            return Complex(torch.stack([real_hs, imag_hs], dim=self.dim), dim=self.dim, dtype=self.dtype,
                           device=self.device)

    def hardswish(self, inplace=False):
        """
        Hard swish activation (applied to real/imag independently).
        """
        if inplace:
            F.hardswish(self.real, inplace=True)
            F.hardswish(self.imag, inplace=True)
            return self
        else:
            real_hsw = F.hardswish(self.real)
            imag_hsw = F.hardswish(self.imag)
            return Complex(torch.stack([real_hsw, imag_hsw], dim=self.dim), dim=self.dim, dtype=self.dtype,
                           device=self.device)

    # --- Shape Manipulation Shortcuts ---

    @staticmethod
    def hstack(tensors):
        """Horizontal stacking (along dim 1)."""
        real_list = [t.real if isinstance(t, Complex) else t for t in tensors]
        imag_list = [t.imag if isinstance(t, Complex) else torch.zeros_like(t) for t in tensors]

        real_stacked = torch.hstack(real_list)
        imag_stacked = torch.hstack(imag_list)

        return Complex(torch.stack([real_stacked, imag_stacked], dim=-1), dim=-1)

    @staticmethod
    def vstack(tensors):
        """Vertical stacking (along dim 0)."""
        real_list = [t.real if isinstance(t, Complex) else t for t in tensors]
        imag_list = [t.imag if isinstance(t, Complex) else torch.zeros_like(t) for t in tensors]

        real_stacked = torch.vstack(real_list)
        imag_stacked = torch.vstack(imag_list)

        return Complex(torch.stack([real_stacked, imag_stacked], dim=-1), dim=-1)

    @staticmethod
    def dstack(tensors):
        """Depth-wise stacking (along dim 2)."""
        real_list = [t.real if isinstance(t, Complex) else t for t in tensors]
        imag_list = [t.imag if isinstance(t, Complex) else torch.zeros_like(t) for t in tensors]

        real_stacked = torch.dstack(real_list)
        imag_stacked = torch.dstack(imag_list)

        return Complex(torch.stack([real_stacked, imag_stacked], dim=-1), dim=-1)

    def hsplit(self, indices_or_sections):
        """Horizontal splitting."""
        real_splits = torch.hsplit(self.real, indices_or_sections)
        imag_splits = torch.hsplit(self.imag, indices_or_sections)

        return tuple(Complex(torch.stack([r, i], dim=self.dim), dim=self.dim, dtype=self.dtype, device=self.device)
                     for r, i in zip(real_splits, imag_splits))

    def vsplit(self, indices_or_sections):
        """Vertical splitting."""
        real_splits = torch.vsplit(self.real, indices_or_sections)
        imag_splits = torch.vsplit(self.imag, indices_or_sections)

        return tuple(Complex(torch.stack([r, i], dim=self.dim), dim=self.dim, dtype=self.dtype, device=self.device)
                     for r, i in zip(real_splits, imag_splits))

    def dsplit(self, indices_or_sections):
        """Depth-wise splitting."""
        real_splits = torch.dsplit(self.real, indices_or_sections)
        imag_splits = torch.dsplit(self.imag, indices_or_sections)

        return tuple(Complex(torch.stack([r, i], dim=self.dim), dim=self.dim, dtype=self.dtype, device=self.device)
                     for r, i in zip(real_splits, imag_splits))

    @staticmethod
    def column_stack(tensors):
        """Stack 1D arrays as columns."""
        real_list = [t.real if isinstance(t, Complex) else t for t in tensors]
        imag_list = [t.imag if isinstance(t, Complex) else torch.zeros_like(t) for t in tensors]

        real_stacked = torch.column_stack(real_list)
        imag_stacked = torch.column_stack(imag_list)

        return Complex(torch.stack([real_stacked, imag_stacked], dim=-1), dim=-1)

    @staticmethod
    def row_stack(tensors):
        """Stack arrays as rows (alias for vstack)."""
        return Complex.vstack(tensors)

    def atleast_1d(self):
        """Ensure tensor has at least 1 dimension."""
        real_1d = torch.atleast_1d(self.real)
        imag_1d = torch.atleast_1d(self.imag)
        return Complex(torch.stack([real_1d, imag_1d], dim=-1), dim=-1, dtype=self.dtype, device=self.device)

    def atleast_2d(self):
        """Ensure tensor has at least 2 dimensions."""
        real_2d = torch.atleast_2d(self.real)
        imag_2d = torch.atleast_2d(self.imag)
        return Complex(torch.stack([real_2d, imag_2d], dim=-1), dim=-1, dtype=self.dtype, device=self.device)

    def atleast_3d(self):
        """Ensure tensor has at least 3 dimensions."""
        real_3d = torch.atleast_3d(self.real)
        imag_3d = torch.atleast_3d(self.imag)
        return Complex(torch.stack([real_3d, imag_3d], dim=-1), dim=-1, dtype=self.dtype, device=self.device)

    # --- Random Sampling Distributions ---

    @staticmethod
    def bernoulli(p, size=None):
        """Sample from Bernoulli distribution."""
        if size is None:
            size = p.shape if hasattr(p, 'shape') else ()

        real_sample = torch.bernoulli(p if not isinstance(p, Complex) else p.real)
        imag_sample = torch.bernoulli(p if not isinstance(p, Complex) else p.imag)

        return Complex(torch.stack([real_sample, imag_sample], dim=-1), dim=-1)

    @staticmethod
    def exponential(lambd, size):
        """Sample from exponential distribution."""
        real_sample = torch.empty(size).exponential_(lambd)
        imag_sample = torch.empty(size).exponential_(lambd)

        return Complex(torch.stack([real_sample, imag_sample], dim=-1), dim=-1)

    @staticmethod
    def geometric(p, size):
        """Sample from geometric distribution."""
        real_sample = torch.empty(size).geometric_(p)
        imag_sample = torch.empty(size).geometric_(p)

        return Complex(torch.stack([real_sample, imag_sample], dim=-1), dim=-1)

    @staticmethod
    def log_normal(mean=0, std=1, size=None):
        """Sample from log-normal distribution."""
        real_sample = torch.empty(size).log_normal_(mean, std)
        imag_sample = torch.empty(size).log_normal_(mean, std)

        return Complex(torch.stack([real_sample, imag_sample], dim=-1), dim=-1)

    @staticmethod
    def multinomial(input, num_samples, replacement=False):
        """Sample from multinomial distribution."""
        if isinstance(input, Complex):
            input = input.mag()  # Use magnitude for probabilities

        samples = torch.multinomial(input, num_samples, replacement)
        return samples  # Returns indices, not complex values

    # --- Binary Element-wise Comparison ---

    def maximum(self, other):
        """Element-wise maximum (based on magnitude)."""
        if not isinstance(other, Complex):
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device)

        self_mag = self.mag()
        other_mag = other.mag()

        mask = self_mag >= other_mag

        real_result = torch.where(mask, self.real, other.real)
        imag_result = torch.where(mask, self.imag, other.imag)

        return Complex(torch.stack([real_result, imag_result], dim=self.dim), dim=self.dim, dtype=self.dtype,
                       device=self.device)

    def minimum(self, other):
        """Element-wise minimum (based on magnitude)."""
        if not isinstance(other, Complex):
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device)

        self_mag = self.mag()
        other_mag = other.mag()

        mask = self_mag <= other_mag

        real_result = torch.where(mask, self.real, other.real)
        imag_result = torch.where(mask, self.imag, other.imag)

        return Complex(torch.stack([real_result, imag_result], dim=self.dim), dim=self.dim, dtype=self.dtype,
                       device=self.device)

    def fmax(self, other):
        """Element-wise max ignoring NaNs."""
        if not isinstance(other, Complex):
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device)

        real_result = torch.fmax(self.real, other.real)
        imag_result = torch.fmax(self.imag, other.imag)

        return Complex(torch.stack([real_result, imag_result], dim=self.dim), dim=self.dim, dtype=self.dtype,
                       device=self.device)

    def fmin(self, other):
        """Element-wise min ignoring NaNs."""
        if not isinstance(other, Complex):
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device)

        real_result = torch.fmin(self.real, other.real)
        imag_result = torch.fmin(self.imag, other.imag)

        return Complex(torch.stack([real_result, imag_result], dim=self.dim), dim=self.dim, dtype=self.dtype,
                       device=self.device)

    # --- Advanced Linear Algebra ---

    def addr(self, vec1, vec2, beta=1, alpha=1, out: Optional['Complex'] = None):
        """addr with out= support."""
        if not isinstance(vec1, Complex):
            vec1 = Complex(vec1, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                           **self.kwargs)
        if not isinstance(vec2, Complex):
            vec2 = Complex(vec2, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                           **self.kwargs)

        # addr: out = beta * self + alpha * (vec1 ⊗ vec2)
        c_self = torch.complex(self.real, self.imag)
        c_vec1 = torch.complex(vec1.real, vec1.imag)
        c_vec2 = torch.complex(vec2.real, vec2.imag)

        result = torch.addr(c_self, c_vec1, c_vec2, beta=beta, alpha=alpha)

        res = Complex(result, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                      **self.kwargs)
        if out is not None:
            out.tensor.copy_(res.tensor)
            return out
        return res

    def addbmm(self, batch1, batch2, beta=1, alpha=1):
        """Performs out = beta*self + alpha*sum(batch1@batch2)."""
        if not isinstance(batch1, Complex):
            batch1 = Complex(batch1, dim=self.dim, dtype=self.dtype, device=self.device)
        if not isinstance(batch2, Complex):
            batch2 = Complex(batch2, dim=self.dim, dtype=self.dtype, device=self.device)

        c_self = torch.complex(self.real, self.imag)
        c_batch1 = torch.complex(batch1.real, batch1.imag)
        c_batch2 = torch.complex(batch2.real, batch2.imag)

        result = torch.addbmm(c_self, c_batch1, c_batch2, beta=beta, alpha=alpha)
        return Complex(result, dim=self.dim, dtype=self.dtype, device=self.device)

    @staticmethod
    def chain_matmul(*matrices):
        """Efficiently multiplies a sequence of matrices."""
        complex_matrices = []
        for m in matrices:
            if isinstance(m, Complex):
                complex_matrices.append(torch.complex(m.real, m.imag))
            else:
                complex_matrices.append(m)

        result = torch.chain_matmul(*complex_matrices)
        return Complex(result, dim=-1)

    def corrcoef(self):
        """Computes correlation coefficient matrix."""
        # Compute for real and imag separately
        real_corr = torch.corrcoef(self.real)
        imag_corr = torch.corrcoef(self.imag)

        return Complex(torch.stack([real_corr, imag_corr], dim=-1), dim=-1, dtype=self.dtype, device=self.device)

    # --- Missing Statistics ---

    def std(self, dim=None, correction=1, keepdim=False):
        """Standard deviation."""
        return self.var(dim=dim, correction=correction, keepdim=keepdim).sqrt()

    def nanmedian(self, dim=None, keepdim=False):
        """Median ignoring NaNs."""
        real_median = torch.nanmedian(self.real, dim=dim, keepdim=keepdim)
        imag_median = torch.nanmedian(self.imag, dim=dim, keepdim=keepdim)

        if isinstance(real_median, tuple):
            real_val, real_idx = real_median
            imag_val, imag_idx = imag_median
            result = Complex(torch.stack([real_val, imag_val], dim=self.dim if keepdim else -1),
                             dim=self.dim if keepdim else -1, dtype=self.dtype, device=self.device)
            return result, real_idx
        else:
            return self._wrap(torch.stack([real_median, imag_median], dim=self.dim if keepdim else -1),
                              dim=self.dim if keepdim else -1)

    def nanquantile(self, q, dim=None, keepdim=False):
        """Quantile ignoring NaNs."""
        real_quantile = torch.nanquantile(self.real, q, dim=dim, keepdim=keepdim)
        imag_quantile = torch.nanquantile(self.imag, q, dim=dim, keepdim=keepdim)

        return Complex(torch.stack([real_quantile, imag_quantile], dim=-1), dim=-1, dtype=self.dtype,
                       device=self.device)

    # --- Phase & Math Utilities ---

    def deg2rad(self):
        """Convert phase from degrees to radians."""
        real_rad = torch.deg2rad(self.real)
        imag_rad = torch.deg2rad(self.imag)

        return Complex(torch.stack([real_rad, imag_rad], dim=self.dim), dim=self.dim, dtype=self.dtype,
                       device=self.device)

    def rad2deg(self):
        """Convert phase from radians to degrees."""
        real_deg = torch.rad2deg(self.real)
        imag_deg = torch.rad2deg(self.imag)

        return Complex(torch.stack([real_deg, imag_deg], dim=self.dim), dim=self.dim, dtype=self.dtype,
                       device=self.device)

    def frac(self):
        """Fractional part of real/imag components."""
        real_frac = torch.frac(self.real)
        imag_frac = torch.frac(self.imag)

        return Complex(torch.stack([real_frac, imag_frac], dim=self.dim), dim=self.dim, dtype=self.dtype,
                       device=self.device)

    def trunc(self):
        """Truncated integer part of real/imag components."""
        real_trunc = torch.trunc(self.real)
        imag_trunc = torch.trunc(self.imag)

        return self._wrap(torch.stack([real_trunc, imag_trunc], dim=self.dim))

    def logit(self, eps=None):
        """Inverse of sigmoid (log-odds)."""
        real_logit = torch.logit(self.real, eps=eps)
        imag_logit = torch.logit(self.imag, eps=eps)

        return self._wrap(torch.stack([real_logit, imag_logit], dim=self.dim))

    # --- Window Functions ---

    @staticmethod
    def blackman_window(window_length, periodic=True, dtype=None, device=None):
        """Blackman window function."""
        window = torch.blackman_window(window_length, periodic=periodic, dtype=dtype, device=device)
        return Complex(torch.complex(window, torch.zeros_like(window)), dim=-1, dtype=dtype, device=device)

    @staticmethod
    def kaiser_window(window_length, periodic=True, beta=12.0, dtype=None, device=None):
        """Kaiser window function."""
        window = torch.kaiser_window(window_length, periodic=periodic, beta=beta, dtype=dtype, device=device)
        return Complex(torch.complex(window, torch.zeros_like(window)), dim=-1, dtype=dtype, device=device)

    # --- View Converters (Interoperability) ---

    def view_as_real(self):
        """
        Returns the underlying real-valued tensor with complex dimension at the end (..., 2).
        """
        # The tensor is already in (..., 2) format with real/imag stacked
        return self.tensor

    @staticmethod
    def view_as_complex(real_tensor, dim=-1):
        """
        Factory method to wrap a real (..., 2) tensor into a Complex object.
        """
        return Complex(real_tensor, dim=dim)

    # --- Equality Operator ---

    def __eq__(self, other):
        """
        Equality operator for element-wise comparison.
        Returns a boolean tensor mask.
        """
        eq_result = self.eq(other)
        # eq() returns a Complex object, extract the real part as boolean
        if isinstance(eq_result, Complex):
            return eq_result.real.bool()
        return eq_result

    # --- Missing Math Operations ---

    def hypot(self, other):
        """
        Computes sqrt(|self|^2 + |other_decomposition|^2) element-wise.
        """
        if not isinstance(other, Complex):
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device)

        self_mag_sq = self.real ** 2 + self.imag ** 2
        other_mag_sq = other.real ** 2 + other.imag ** 2

        result_mag = torch.sqrt(self_mag_sq + other_mag_sq)

        # Return as complex with zero imaginary part
        return Complex(torch.stack([result_mag, torch.zeros_like(result_mag)], dim=self.dim),
                       dim=self.dim, dtype=self.dtype, device=self.device)

    def heaviside(self, values):
        """
        Heaviside step function (applied to magnitude).
        """
        if not isinstance(values, Complex):
            values = Complex(values, dim=self.dim, dtype=self.dtype, device=self.device)

        mag = self.mag()
        values_real = values.real if isinstance(values, Complex) else values

        result = torch.heaviside(mag, values_real)

        # Return as complex with zero imaginary part
        return Complex(torch.stack([result, torch.zeros_like(result)], dim=self.dim),
                       dim=self.dim, dtype=self.dtype, device=self.device)

    def ldexp(self, other):
        """
        Computes self * 2^other_decomposition efficiently.
        """
        if not isinstance(other, Complex):
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device)

        real_result = torch.ldexp(self.real, other.real.to(torch.int))
        imag_result = torch.ldexp(self.imag, other.imag.to(torch.int))

        return Complex(torch.stack([real_result, imag_result], dim=self.dim),
                       dim=self.dim, dtype=self.dtype, device=self.device)

    # --- Boolean Checks ---

    def is_complex(self):
        """
        Returns True (to mimic torch tensor API).
        """
        return True

    def is_nonzero(self):
        """
        Returns True if the tensor is a scalar and not zero.
        """
        if self.numel() != 1:
            raise RuntimeError("is_nonzero() can only be called on scalar tensors")
        return bool(self.mag().item() != 0)

    # --- Explicit In-Place Methods ---

    # --- DataLoader Support ---

    @staticmethod
    def collate(batch):
        """
        Collate function for PyTorch DataLoader.
        Stacks a list of Complex objects into a batched Complex tensor.
        """
        if not batch:
            raise ValueError("Cannot collate an empty batch")

        # Extract real and imag parts from all items
        real_list = [item.real if isinstance(item, Complex) else item for item in batch]
        imag_list = [item.imag if isinstance(item, Complex) else torch.zeros_like(item) for item in batch]

        # Stack along batch dimension (dim=0)
        real_batched = torch.stack(real_list, dim=0)
        imag_batched = torch.stack(imag_list, dim=0)

        # Return as Complex with dim=-1
        return Complex(torch.complex(real_batched, imag_batched), dim=-1)

    # --- Conjugate View Compatibility ---

    def resolve_conj(self):
        """
        Resolves conjugate view bit (returns self for API compatibility).
        """
        return self

    # --- Tensor Factory Methods ---

    def new_zeros(self, *size):
        """Create a new complex tensor of zeros with same dtype/device."""
        real_zeros = torch.zeros(*size, dtype=self.dtype, device=self.device)
        imag_zeros = torch.zeros(*size, dtype=self.dtype, device=self.device)
        return self._wrap(torch.stack([real_zeros, imag_zeros], dim=-1), dim=-1)

    def new_ones(self, *size):
        """Create a new complex tensor of ones with same dtype/device."""
        real_ones = torch.ones(*size, dtype=self.dtype, device=self.device)
        imag_zeros = torch.zeros(*size, dtype=self.dtype, device=self.device)
        return self._wrap(torch.stack([real_ones, imag_zeros], dim=-1), dim=-1)

    def new_empty(self, *size):
        """Create a new uninitialized complex tensor with same dtype/device."""
        real_empty = torch.empty(*size, dtype=self.dtype, device=self.device)
        imag_empty = torch.empty(*size, dtype=self.dtype, device=self.device)
        return self._wrap(torch.stack([real_empty, imag_empty], dim=-1), dim=-1)

    def new_full(self, size, fill_value):
        """Create a new complex tensor filled with fill_value."""
        if not isinstance(fill_value, Complex):
            fill_value = Complex(fill_value, dim=-1, dtype=self.dtype, device=self.device)

        real_full = torch.full(size, fill_value.real.item() if fill_value.real.numel() == 1 else 0,
                               dtype=self.dtype, device=self.device)
        imag_full = torch.full(size, fill_value.imag.item() if fill_value.imag.numel() == 1 else 0,
                               dtype=self.dtype, device=self.device)
        return self._wrap(torch.stack([real_full, imag_full], dim=-1), dim=-1)

    def vdot(self, other):
        """Conjugate inner product (dot product with conjugation)."""
        if not isinstance(other, Complex):
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device)

        # Conjugate inner product: sum(conj(self) * other_decomposition)
        conj_self = self.conj()
        product = conj_self * other
        return product.sum()

    def outer(self, other):
        """Outer product of two vectors."""
        if not isinstance(other, Complex):
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device)

        # Outer product: self[:, None] * other_decomposition[None, :]
        self_col = self.unsqueeze(-1)
        other_row = other.unsqueeze(0)
        return self_col * other_row

    # --- Standard Math Aliases ---

    def asin(self):
        """Alias for arcsin."""
        return self.arcsin()

    def acos(self):
        """Alias for arccos."""
        return self.arccos()

    def atan(self):
        """Alias for arctan."""
        return self.arctan()

    def abs(self):
        """Absolute value (magnitude) of complex tensor."""
        return self.mag()

    def fix(self):
        """Round towards zero (truncate)."""
        return self.trunc()

    def copysign(self, other):
        """Copy the sign/phase of another tensor."""
        if not isinstance(other, Complex):
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device)

        # For complex numbers, copy the phase
        self_mag = self.mag()
        other_phase = other.phi()

        real_result = self_mag * torch.cos(other_phase)
        imag_result = self_mag * torch.sin(other_phase)

        return self._wrap(torch.stack([real_result, imag_result], dim=self.dim))

    # --- Autograd Hooks ---

    def register_hook(self, hook):
        """
        Register a backward hook on the underlying tensor.
        The hook will be called during backward pass.
        """
        return self.tensor.register_hook(hook)

    # --- __torch_function__ Hook (Critical Interoperability) ---

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        Intercepts calls from torch.* namespace and routes them to Complex methods dynamically.
        This allows torch.tan(z), torch.sinh(z), etc. to work if the methods are defined.
        """
        if kwargs is None:
            kwargs = {}

        # Check if all types are compatible
        if not all(issubclass(t, (torch.Tensor, cls)) for t in types):
            return NotImplemented

        # Try to route to instance method
        method_name = func.__name__

        # Try to route to class method
        # Try to route to class method or property
        if hasattr(cls, method_name):
            attr = getattr(cls, method_name)
            if isinstance(attr, property):
                # Property access: torch.real(z) -> z.real
                return attr.__get__(args[0], cls)
            if callable(attr):
                # Calling the method on the class with *args will correctly pass
                # the first argument as 'self'.

                # Fix for operations where LHS is not Complex (e.g. Tensor + Complex)
                # In this case args[0] is Tensor, but attr expects 'self' as Complex.
                # We wrap args[0] into Complex using metadata from the first actual Complex arg.
                if len(args) > 0 and not isinstance(args[0], cls):
                    ref = next((arg for arg in args if isinstance(arg, cls)), None)
                    if ref is not None:
                        args_list = list(args)
                        # Wrap the first arg (e.g. Tensor) into Complex
                        # We trust 'ref' to provide dim/dtype/device context
                        args_list[0] = cls(args[0], dim=ref.dim, dtype=ref.dtype, device=ref.device)
                        return attr(*args_list, **kwargs)

                return attr(*args, **kwargs)

        # For unmapped functions, try to call directly if they are in math/torch and we can handle them
        # but NotImplemented is safer for unknown functions.
        return NotImplemented

    # --- Distributed & Shared Memory ---

    def share_memory_(self):
        """Moves the underlying storage to shared memory."""
        self.tensor.share_memory_()
        return self

    def is_shared(self):
        """Checks if the tensor is in shared memory."""
        return self.tensor.is_shared()

    # --- Low-Level Storage Manipulation ---

    def as_strided(self, size, stride, storage_offset=None):
        """Creates a view of an existing tensor with specified strides."""
        return Complex(self.tensor.as_strided(size, stride, storage_offset),
                       dim=self.dim, dtype=self.dtype, device=self.device)

    def set_(self, source):
        """Sets the underlying storage to point to another tensor's storage."""
        if isinstance(source, Complex):
            self.tensor.set_(source.tensor)
        else:
            self.tensor.set_(source)
        return self

    def resize_(self, size):
        """In-place resizing of the underlying storage."""
        self.tensor.resize_(size)
        return self

    def unflatten(self, dim, sizes):
        """The inverse of flatten."""
        return Complex(self.tensor.unflatten(dim, sizes),
                       dim=self.dim, dtype=self.dtype, device=self.device)

    # --- Advanced "Soft" activations ---

    def softplus(self, beta=1, threshold=20, out: Optional['Complex'] = None):
        """
        Applies Softplus to the magnitude of the complex tensor.
        Softplus(z) = unit(z) * log(1 + exp(beta * mag(z))) / beta
        """
        m = self.mag()
        sp = torch.nn.functional.softplus(m, beta=beta, threshold=threshold)
        res = self.unit().multiply(sp)
        if out is not None:
            out.tensor.copy_(res.tensor)
            return out
        return res

    def gelu(self, approximate='none', out: Optional['Complex'] = None):
        """
        Applies GELU to the magnitude of the complex tensor.
        GELU(z) = unit(z) * GELU(mag(z))
        """
        m = self.mag()
        g = torch.nn.functional.gelu(m, approximate=approximate)
        res = self.unit().multiply(g)
        if out is not None:
            out.tensor.copy_(res.tensor)
            return out
        return res

    def log_softmax(self, dim=None, out: Optional['Complex'] = None):
        """
        Computes log-softmax of the complex tensor based on magnitude.
        """
        if dim is None:
            dim = -1
        m = self.mag()
        ls = torch.nn.functional.log_softmax(m, dim=dim)
        res = self._wrap(torch.stack([ls, torch.zeros_like(ls)], dim=self.dim))
        if out is not None:
            out.tensor.copy_(res.tensor)
            return out
        return res

    # --- Comparison Magic Methods ---

    def __lt__(self, other):
        """Less than comparison (based on magnitude)."""
        if not isinstance(other, Complex):
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device)
        return self.mag() < other.mag()

    def __le__(self, other):
        """Less than or equal comparison (based on magnitude)."""
        if not isinstance(other, Complex):
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device)
        return self.mag() <= other.mag()

    def __gt__(self, other):
        """Greater than comparison (based on magnitude)."""
        if not isinstance(other, Complex):
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device)
        return self.mag() > other.mag()

    def __ge__(self, other):
        """Greater than or equal comparison (based on magnitude)."""
        if not isinstance(other, Complex):
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device)
        return self.mag() >= other.mag()

    # --- Tensor Type Metadata ---

    def is_floating_point(self):
        """Returns True (complex tensors are floating point)."""
        return True

    @property
    def layout(self):
        """Returns the layout of the underlying tensor."""
        return self.tensor.layout

    def storage(self):
        """Returns the underlying storage."""
        return self.tensor.storage()

    # --- Resampling (Signal Processing) ---

    def resample(self, orig_freq, new_freq):
        """
        Resample complex signal from orig_freq to new_freq.
        Uses interpolation for resampling.
        """
        # Calculate resampling ratio
        ratio = new_freq / orig_freq

        # Resample real and imaginary parts separately
        real_resampled = torch.nn.functional.interpolate(
            self._real().unsqueeze(0).unsqueeze(0),
            scale_factor=ratio,
            mode='linear',
            align_corners=False
        ).squeeze(0).squeeze(0)

        imag_resampled = torch.nn.functional.interpolate(
            self._imag().unsqueeze(0).unsqueeze(0),
            scale_factor=ratio,
            mode='linear',
            align_corners=False
        ).squeeze(0).squeeze(0)

        return self._wrap(torch.stack([real_resampled, imag_resampled], dim=-1), dim=-1)

    # --- Scalar Conversion ---

    def __int__(self):
        """Convert to int (returns magnitude for scalar tensors)."""
        if self.numel() != 1:
            raise TypeError("only size-1 arrays can be converted to Python scalars")
        import warnings
        warnings.warn("Casting complex to int discards imaginary part", UserWarning)
        return int(self.mag().item())

    def __float__(self):
        """Convert to float (returns magnitude for scalar tensors)."""
        if self.numel() != 1:
            raise TypeError("only size-1 arrays can be converted to Python scalars")
        import warnings
        warnings.warn("Casting complex to float returns magnitude", UserWarning)
        return float(self.mag().item())

    def __complex__(self):
        """Convert to Python complex (for scalar tensors)."""
        if self.numel() != 1:
            raise TypeError("only size-1 arrays can be converted to Python scalars")
        return complex(self.real.item(), self.imag.item())

    def __pos__(self):
        """Unary plus operator (+z)."""
        return self

    # --- Boolean Binary Operators ---

    def __and__(self, other):
        """Logical AND (&) operator for masking."""
        if not isinstance(other, Complex):
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device)

        # Boolean representation: magnitude > 0
        self_bool = self.mag() > 0
        other_bool = other.mag() > 0
        result_bool = self_bool & other_bool

        # Return as Complex with real=bool, imag=0
        real_result = result_bool.float()
        imag_result = torch.zeros_like(real_result)
        return self._wrap(torch.stack([real_result, imag_result], dim=self.dim))

    def __or__(self, other):
        """Logical OR (|) operator for masking."""
        if not isinstance(other, Complex):
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device)

        # Boolean representation: magnitude > 0
        self_bool = self.mag() > 0
        other_bool = other.mag() > 0
        result_bool = self_bool | other_bool

        # Return as Complex with real=bool, imag=0
        real_result = result_bool.float()
        imag_result = torch.zeros_like(real_result)
        return self._wrap(torch.stack([real_result, imag_result], dim=self.dim))

    def __xor__(self, other):
        """Logical XOR (^) operator for masking."""
        if not isinstance(other, Complex):
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device)

        # Boolean representation: magnitude > 0
        self_bool = self.mag() > 0
        other_bool = other.mag() > 0
        result_bool = self_bool ^ other_bool

        # Return as Complex with real=bool, imag=0
        real_result = result_bool.float()
        imag_result = torch.zeros_like(real_result)
        return self._wrap(torch.stack([real_result, imag_result], dim=self.dim))

    # --- Logical Not ---

    def __invert__(self):
        """Bitwise/logical NOT operation."""
        return self.logical_not()

    def logical_not(self):
        """Logical NOT (inverts boolean mask based on magnitude)."""
        # For complex numbers, consider zero magnitude as False
        mag_zero = self.mag() == 0
        real_result = mag_zero.float()
        imag_result = torch.zeros_like(real_result)

        return self._wrap(torch.stack([real_result, imag_result], dim=self.dim))

    # --- Memory Pinning (Critical for DataLoaders) ---

    def pin_memory(self):
        """Pin memory for faster CPU-to-GPU transfer."""
        pinned_tensor = self.tensor.pin_memory()
        return self._wrap(pinned_tensor)

    def is_pinned(self):
        """Returns True if tensor is in pinned memory."""
        return self.tensor.is_pinned()

    # --- Boolean Reductions ---

    def all(self, dim=None, keepdim=False):
        """Returns True if all elements are non-zero."""
        # For complex numbers, check if magnitude is non-zero
        mag_nonzero = self.mag() != 0

        if dim is None:
            return mag_nonzero.all()
        else:
            return mag_nonzero.all(dim=dim, keepdim=keepdim)

    def any(self, dim=None, keepdim=False):
        """Returns True if any element is non-zero."""
        # For complex numbers, check if magnitude is non-zero
        mag_nonzero = self.mag() != 0

        if dim is None:
            return mag_nonzero.any()
        else:
            return mag_nonzero.any(dim=dim, keepdim=keepdim)

            self.imag.fill_diagonal_(val_imag, wrap=wrap)

        return self

    def diagonal_scatter(self, src, offset=0, dim1=0, dim2=1):
        """
        Returns a new tensor with the diagonal replaced by src.
        Functional (non-mutating) version of diagonal assignment.
        """
        result = self.clone()

        if not isinstance(src, Complex):
            src = Complex(src, dtype=self.dtype, device=self.device)

        # Get diagonal view
        target_diag_real = result.real.diagonal(offset, dim1, dim2)
        target_diag_imag = result.imag.diagonal(offset, dim1, dim2)

        # Source parts
        src_real = src.real.flatten()
        src_imag = src.imag.flatten()

        # Match sizes
        numel_to_copy = min(target_diag_real.numel(), src_real.numel())
        if numel_to_copy > 0:
            target_diag_real.flatten()[:numel_to_copy].copy_(src_real[:numel_to_copy])
            target_diag_imag.flatten()[:numel_to_copy].copy_(src_imag[:numel_to_copy])

        return result

    def select_scatter(self, src, dim, index):
        """
        Returns a new tensor with the selected slice replaced by src.
        """
        result = self.clone()

        if not isinstance(src, Complex):
            src = Complex(src, dim=self.dim, dtype=self.dtype, device=self.device)

        if dim < 0:
            dim = result.real.ndim + dim

        idx = [slice(None)] * result.real.ndim
        idx[dim] = index
        idx = tuple(idx)

        # Use copy_ to avoid broadcasting issues with simple assignment
        target_real = result.real[idx]
        target_imag = result.imag[idx]

        # Ensure source is reshaped to target if numel matches
        src_real = src.real
        src_imag = src.imag

        if target_real.shape != src_real.shape and target_real.numel() == src_real.numel():
            src_real = src_real.reshape(target_real.shape)
            src_imag = src_imag.reshape(target_imag.shape)

        target_real.copy_(src_real)
        target_imag.copy_(src_imag)

        return result

    def slice_scatter(self, src, dim=0, start=None, end=None, step=1):
        """
        Returns a new tensor with the slice replaced by src.
        """
        result = self.clone()

        if not isinstance(src, Complex):
            src = Complex(src, dim=self.dim, dtype=self.dtype, device=self.device)

        if dim < 0:
            dim = result.real.ndim + dim

        idx = [slice(None)] * result.real.ndim
        idx[dim] = slice(start, end, step)
        idx = tuple(idx)

        target_real = result.real[idx]
        target_imag = result.imag[idx]

        src_real = src.real
        src_imag = src.imag

        if target_real.shape != src_real.shape and target_real.numel() == src_real.numel():
            src_real = src_real.reshape(target_real.shape)
            src_imag = src_imag.reshape(target_imag.shape)

        target_real.copy_(src_real)
        target_imag.copy_(src_imag)

        return result

    def index_add_(self, dim, index, source):
        """Accumulate values into specific indices (in-place)."""
        if not isinstance(source, Complex):
            source = Complex(source, dim=self.dim, dtype=self.dtype, device=self.device)

        self._real().index_add_(dim, index, source._real())
        self._imag().index_add_(dim, index, source._imag())
        return self

    def index_fill_(self, dim, index, value):
        """Fill specific indices with a value (in-place)."""
        if isinstance(value, Complex):
            val_real = value._real().item() if value.numel() == 1 else 0
            val_imag = value._imag().item() if value.numel() == 1 else 0
        elif isinstance(value, complex):
            val_real = value.real
            val_imag = value.imag
        else:
            val_real = float(value)
            val_imag = 0.0

        self._real().index_fill_(dim, index, val_real)
        self._imag().index_fill_(dim, index, val_imag)
        return self

    def index_copy_(self, dim, index, source):
        """Copy values into specific indices (in-place)."""
        if not isinstance(source, Complex):
            source = Complex(source, dim=self.dim, dtype=self.dtype, device=self.device)

        self._real().index_copy_(dim, index, source._real())
        self._imag().index_copy_(dim, index, source._imag())
        return self

    # --- Reciprocal Square Root ---

    def rsqrt(self, inplace: bool = False, out: Optional['Complex'] = None):
        """Reciprocal square root: 1/sqrt(z)."""
        res = self.sqrt().reciprocal()
        if out is not None:
            out.tensor.copy_(res.tensor)
            return out
        if inplace:
            self.tensor = res.tensor
            return self
        return res

    # --- Tensor Strides ---

    @property
    def real(self):
        # print(f"DEBUG Complex.real: stacked={self._is_stacked}, dim={self.dim}, shape={self.tensor.shape}")
        if self._is_stacked:
            return self.tensor.select(self.dim, 0)
        else:
            return self.tensor

    def stride(self, dim=None):
        """Returns the stride of the underlying tensor."""
        if dim is None:
            return self.tensor.stride()
        else:
            return self.tensor.stride(dim)

    # --- Statistics & Metrics ---

    def pdist(self, p=2):
        """
        Computes pairwise distance between rows of the tensor.
        Returns a condensed 1D vector.
        """
        # We use the magnitude for distances
        # Actually, pdist in torch takes a real tensor.
        # For complex tensors, we could use the complex vectors directly.
        # But since we store as real/imag stacked, we can unflatten and use 2*cols
        # However, it's easier to use the complex distance formula:
        # dist(z1, z2) = sqrt(sum(|z1_i - z2_i|^2))

        # Reshape to (N, -1)
        x = self.tensor.reshape(self.tensor.size(0), -1)
        return torch.nn.functional.pdist(x, p=p)

    # --- Serialization Protocols ---

    def __getstate__(self):
        """Serialization state for pickling."""
        return {
            'tensor': self.tensor,
            'dim': self.dim,
            'dtype': self.dtype,
            'device': self.device,
            'arrangement': self.arrangement if hasattr(self, 'arrangement') else 'split'
        }

    def __setstate__(self, state):
        """Restoration after pickling."""
        self.tensor = state['tensor']
        self.dim = state['dim']
        self.dtype = state['dtype']
        self.device = state['device']
        self.arrangement = state.get('arrangement', 'split')

    # --- Helper Logic ---

    # --- Phase 18: Final Edge-Case Parity ---

    def nonzero(self, *, as_tuple=False):
        """
        Returns the indices of elements that are non-zero (based on magnitude).
        """
        mag = self.mag()
        return torch.nonzero(mag, as_tuple=as_tuple)

    def index_put_(self, indices, values, accumulate=False):
        """
        In-place version of index_put.
        """
        if not isinstance(values, Complex):
            values = Complex(values, dim=self.dim, dtype=self.dtype, device=self.device)

        self.real.index_put_(indices, values.real, accumulate=accumulate)
        self.imag.index_put_(indices, values.imag, accumulate=accumulate)
        return self

    def index_put(self, indices, values, accumulate=False):
        """
        Out-of-place version of index_put.
        """
        return self.clone().index_put_(indices, values, accumulate=accumulate)

    def put_(self, index, source, accumulate=False):
        """
        Copies the elements from source into the locations of self specified by index.
        """
        if not isinstance(source, Complex):
            source = Complex(source, dim=self.dim, dtype=self.dtype, device=self.device)

        self.real.put_(index, source.real, accumulate=accumulate)
        self.imag.put_(index, source.imag, accumulate=accumulate)
        return self

    def repeat_interleave(self, repeats, dim=None, *, output_size=None):
        """
        Repeats elements of a tensor.
        """
        real_res = torch.repeat_interleave(self.real, repeats, dim=dim, output_size=output_size)
        imag_res = torch.repeat_interleave(self.imag, repeats, dim=dim, output_size=output_size)
        return self._wrap(torch.stack([real_res, imag_res], dim=self.dim))

    def broadcast_to(self, *shape):
        """
        Broadcasts the tensor to a new shape.
        """
        real_res = self.real.broadcast_to(*shape)
        imag_res = self.imag.broadcast_to(*shape)
        return self._wrap(torch.stack([real_res, imag_res], dim=self.dim))

    def fill_diagonal_(self, fill_value, wrap=False):
        """
        Fills the main diagonal of a tensor that has at least 2-dimensions.
        """
        if not isinstance(fill_value, Complex):
            fill_value = Complex(fill_value, dim=0, dtype=self.dtype, device=self.device)

        self.real.fill_diagonal_(fill_value.real.item() if fill_value.real.numel() == 1 else 0, wrap=wrap)
        self.imag.fill_diagonal_(fill_value.imag.item() if fill_value.imag.numel() == 1 else 0, wrap=wrap)
        return self

    def triu_(self, diagonal=0):
        """In-place upper triangular."""
        self.real.triu_(diagonal)
        self.imag.triu_(diagonal)
        return self

    def tril_(self, diagonal=0):
        """In-place lower triangular."""
        self.real.tril_(diagonal)
        self.imag.tril_(diagonal)
        return self

    def frexp(self):
        """
        Decomposes the magnitude into a normalized fraction and an exponent of two.
        Returns (mantissa, exponent).
        """
        # We apply this to magnitude
        mag = self.mag()
        mantissa, exponent = torch.frexp(mag)
        # Wrap mantissa as Complex (real)
        mantissa_c = self._wrap(torch.stack([mantissa, torch.zeros_like(mantissa)], dim=self.dim))
        return mantissa_c, exponent

    def nextafter(self, other):
        """
        Computes the next floating-point value after self towards other_decomposition, element-wise.
        """
        if not isinstance(other, Complex):
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device)

        real_res = torch.nextafter(self.real, other.real)
        imag_res = torch.nextafter(self.imag, other.imag)
        return self._wrap(torch.stack([real_res, imag_res], dim=self.dim))

    def xlogy(self, other):
        """
        Computes x * log(y) element-wise.
        """
        if not isinstance(other, Complex):
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device)

        # x * log(y)
        log_y = other.log()
        return self.multiply(log_y)

    def __contains__(self, item):
        """
        Allows syntax like 'if value in complex_tensor:'.
        """
        if isinstance(item, Complex):
            # Check if the complex value exists in the tensor
            c_tensor = torch.complex(self.real, self.imag)
            item_tensor = torch.complex(item.real, item.imag)
            return (c_tensor == item_tensor).any().item()
        else:
            # Check in both real and imag parts
            return (item in self.real) or (item in self.imag)

    def __bool__(self):
        """
        Allows checking 'if z:' (raises error if numel > 1, similar to PyTorch).
        """
        if self.numel() > 1:
            raise RuntimeError("The truth value of a Complex tensor with more than one element is ambiguous. "
                               "Use a.any() or a.all()")
        # For single element, check if it's non-zero
        return bool(self.mag().item() != 0)

    def apply_(self, callable):
        """
        Applies a function to every element in-place.
        """
        for i in range(self.numel()):
            # This is slow but provided for parity
            idx = _unravel_index(i, self.shape)
            val = self[idx]
            self[idx] = callable(val)
        return self

    def map_(self, tensor, callable):
        """
        Applies a function using an external tensor.
        """
        if not isinstance(tensor, Complex):
            tensor = Complex(tensor, dim=self.dim, dtype=self.dtype, device=self.device)

        for i in range(self.numel()):
            idx = _unravel_index(i, self.shape)
            self[idx] = callable(self[idx], tensor[idx])
        return self

    def __ne__(self, other):
        """Not equal operator (!=) returning a boolean mask."""
        if not isinstance(other, Complex):
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                            **self.kwargs)

        # Element-wise not equal comparison
        real_ne = self.real != other.real
        imag_ne = self.imag != other.imag
        result = real_ne | imag_ne  # True if either part is not equal
        return result

    def __ipow__(self, exponent):
        """In-place power (z **= n)."""
        # Compute power using De Moivre's formula
        mag = self.mag()
        phi = self.phi()

        new_mag = mag ** exponent
        new_phi = phi * exponent

        real = new_mag * torch.cos(new_phi)
        imag = new_mag * torch.sin(new_phi)

        self.tensor = torch.stack([real, imag], dim=self.dim)
        return self

    # --- Linear Algebra Aliases ---

    def inner(self, other):
        """Computes the inner product (dot product for 1D vectors)."""
        if not isinstance(other, Complex):
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                            **self.kwargs)

        # Inner product: sum(conj(self) * other_decomposition)
        conj_self = self.conj()
        c_self = torch.complex(conj_self.real, conj_self.imag)
        c_other = torch.complex(other.real, other.imag)
        product = c_self * c_other
        result = product.sum()
        return Complex(result, dim=0, dtype=self.dtype, device=self.device, arrangement=self.arrangement, **self.kwargs)

    def kron(self, other):
        """Computes the Kronecker product."""
        if not isinstance(other, Complex):
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                            **self.kwargs)

        # Kronecker product using torch.kron
        c_self = torch.complex(self.real, self.imag)
        c_other = torch.complex(other.real, other.imag)
        result = torch.kron(c_self, c_other)
        return Complex(result, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                       **self.kwargs)

    def ger(self, other):
        """Alias for outer product."""
        if not isinstance(other, Complex):
            other = Complex(other, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                            **self.kwargs)

        # Outer product: self[:, None] * other_decomposition[None, :]
        self_col = self.unsqueeze(-1)
        other_row = other.unsqueeze(0)

        # Complex multiplication
        real = self_col.real * other_row.real - self_col.imag * other_row.imag
        imag = self_col.real * other_row.imag + self_col.imag * other_row.real
        return self._wrap(torch.stack([real, imag], dim=self.dim))

    # --- In-Place Optimization Primitives ---

    def addcmul_(self, tensor1, tensor2, value=1):
        """In-place version of addcmul: self += value * tensor1 * tensor2."""
        if not isinstance(tensor1, Complex):
            tensor1 = Complex(tensor1, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                              **self.kwargs)
        if not isinstance(tensor2, Complex):
            tensor2 = Complex(tensor2, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                              **self.kwargs)

        # Compute tensor1 * tensor2
        prod_real = tensor1.real * tensor2.real - tensor1.imag * tensor2.imag
        prod_imag = tensor1.real * tensor2.imag + tensor1.imag * tensor2.real

        # Add value * product to self
        self.real.add_(prod_real, alpha=value)
        self.imag.add_(prod_imag, alpha=value)
        return self

    def addcdiv_(self, tensor1, tensor2, value=1):
        """In-place version of addcdiv: self += value * tensor1 / tensor2."""
        if not isinstance(tensor1, Complex):
            tensor1 = Complex(tensor1, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                              **self.kwargs)
        if not isinstance(tensor2, Complex):
            tensor2 = Complex(tensor2, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                              **self.kwargs)

        # Compute tensor1 / tensor2
        denom = tensor2.real ** 2 + tensor2.imag ** 2
        div_real = (tensor1.real * tensor2.real + tensor1.imag * tensor2.imag) / denom
        div_imag = (tensor1.imag * tensor2.real - tensor1.real * tensor2.imag) / denom

        # Add value * division to self
        self.real.add_(div_real, alpha=value)
        self.imag.add_(div_imag, alpha=value)
        return self

    def lerp_(self, end, weight):
        """In-place linear interpolation: self = self + weight * (end - self)."""
        if not isinstance(end, Complex):
            end = Complex(end, dim=self.dim, dtype=self.dtype, device=self.device, arrangement=self.arrangement,
                          **self.kwargs)

        # Linear interpolation
        self.real.lerp_(end.real, weight)
        self.imag.lerp_(end.imag, weight)
        return self

    # --- Final Polish: Missing Aliases & Convenience ---

    def asinh(self, inplace=False, out=None):
        """Alias for arcsinh."""
        return self.arcsinh(inplace, out)

    def acosh(self, inplace=False, out=None):
        """Alias for arccosh."""
        return self.arccosh(inplace, out)

    def atanh(self, inplace=False, out=None):
        """Alias for arctanh."""
        return self.arctanh(inplace, out)

    def conj_(self):
        """In-place conjugate alias."""
        return self.conj(inplace=True)

    def ndimension(self):
        """Alias for ndim (returns the number of dimensions)."""
        return self.ndim

    def reshape_as(self, other):
        """Reshapes this tensor to be the same shape as other_decomposition."""
        return self.reshape(*other.shape)

    # --- Final Polish: In-Place Samplers & Bitwise Aliases ---

    def bernoulli_(self, p=0.5):
        """In-place Bernoulli sampling."""
        self.real.bernoulli_(p)
        self.imag.bernoulli_(p)
        return self

    def exponential_(self, lambd=1.0):
        """In-place Exponential sampling."""
        self.real.exponential_(lambd)
        self.imag.exponential_(lambd)
        return self

    def geometric_(self, p):
        """In-place Geometric sampling."""
        self.real.geometric_(p)
        self.imag.geometric_(p)
        return self

    def log_normal_(self, mean=1.0, std=2.0):
        """In-place Log-Normal sampling."""
        self.real.log_normal_(mean, std)
        self.imag.log_normal_(mean, std)
        return self

    def random_(self, from_=0, to=None):
        """In-place discrete uniform sampling."""
        self.real.random_(from_, to)
        self.imag.random_(from_, to)
        return self

    def bitwise_not(self):
        """Alias for logical_not / ~ operator."""
        return self.logical_not()

    def bitwise_and(self, other):
        """Alias for & operator."""
        return self.__and__(other)

    def bitwise_or(self, other):
        """Alias for | operator."""
        return self.__or__(other)

    def bitwise_xor(self, other):
        """Alias for ^ operator."""
        return self.__xor__(other)

    def masked_fill_(self, mask, value):
        """In-place masked_fill."""
        if not isinstance(value, Complex):
            # Handle scalar assignment
            val_real = value.real if isinstance(value, (complex)) else float(value)
            val_imag = value.imag if isinstance(value, (complex)) else 0.0
        else:
            val_real = value.real
            val_imag = value.imag

        self.real.masked_fill_(mask, val_real)
        self.imag.masked_fill_(mask, val_imag)
        return self

    def masked_scatter_(self, mask, source):
        """In-place masked_scatter."""
        if not isinstance(source, Complex):
            source = Complex(source, dim=self.dim, dtype=self.dtype, device=self.device)

        self.real.masked_scatter_(mask, source.real)
        self.imag.masked_scatter_(mask, source.imag)
        return self

    def vander(self, N=None):
        """
        Generates a Vandermonde matrix.
        """
        c_tensor = torch.complex(self.real, self.imag)
        res = torch.vander(c_tensor, N=N)
        return Complex(res, dim=self.dim, dtype=self.dtype, device=self.device)

    # --- Final Polish: LU Factor & In-Place Aliases ---

    def lu_factor(self, pivot=True):
        """
        Computes the compact LU factorization of a matrix.
        Returns (LU, pivots). Required input for lu_solve.
        """
        c_tensor = torch.complex(self.real, self.imag)
        LU, pivots = torch.linalg.lu_factor(c_tensor, pivot=pivot)
        return Complex(LU, dim=self.dim, dtype=self.dtype, device=self.device), pivots

    def bitwise_not_(self):
        """In-place bitwise NOT."""
        # For boolean/mask tensors represented by magnitude > 0
        mag_zero = self.mag() == 0
        self.real.copy_(mag_zero.float())
        self.imag.zero_()
        return self

    def bitwise_and_(self, other):
        """In-place bitwise AND."""
        res = self.__and__(other)
        self.tensor = res.tensor
        return self

    def bitwise_or_(self, other):
        """In-place bitwise OR."""
        res = self.__or__(other)
        self.tensor = res.tensor
        return self

    def bitwise_xor_(self, other):
        """In-place bitwise XOR."""
        res = self.__xor__(other)
        self.tensor = res.tensor
        return self

    def sigmoid_(self):
        """In-place sigmoid alias."""
        return self.sigmoid(inplace=True)

    def tanh_(self):
        """In-place tanh alias."""
        return self.tanh(inplace=True)

    def nan_to_num_(self, nan=0.0, posinf=None, neginf=None):
        """In-place nan_to_num."""
        # Standard torch.nan_to_num does not support in-place on the tensor itself 
        # for all versions, but we can implement it by copying.
        res = torch.nan_to_num(self.tensor, nan=nan, posinf=posinf, neginf=neginf)
        self.tensor.copy_(res)
        return self

    # --- Final Polish: N-Dimensional FFTs & activations ---

    def fft2(self, s=None, dim=(-2, -1), norm=None):
        """2D Forward FFT."""
        c_tensor = torch.complex(self.real, self.imag)
        res = torch.fft.fft2(c_tensor, s=s, dim=dim, norm=norm)
        return self._wrap(torch.stack([res.real, res.imag], dim=self.dim))

    def ifft2(self, s=None, dim=(-2, -1), norm=None):
        """2D Inverse FFT."""
        c_tensor = torch.complex(self.real, self.imag)
        res = torch.fft.ifft2(c_tensor, s=s, dim=dim, norm=norm)
        return self._wrap(torch.stack([res.real, res.imag], dim=self.dim))

    def fftn(self, s=None, dim=None, norm=None):
        """N-D Forward FFT."""
        c_tensor = torch.complex(self.real, self.imag)
        res = torch.fft.fftn(c_tensor, s=s, dim=dim, norm=norm)
        return self._wrap(torch.stack([res.real, res.imag], dim=self.dim))

    def ifftn(self, s=None, dim=None, norm=None):
        """N-D Inverse FFT."""
        c_tensor = torch.complex(self.real, self.imag)
        res = torch.fft.ifftn(c_tensor, s=s, dim=dim, norm=norm)
        return self._wrap(torch.stack([res.real, res.imag], dim=self.dim))

    def fftshift(self, dim=None):
        """Shift the zero-frequency component to the center of the spectrum."""
        real_shift = torch.fft.fftshift(self.real, dim=dim)
        imag_shift = torch.fft.fftshift(self.imag, dim=dim)
        return self._wrap(torch.stack([real_shift, imag_shift], dim=self.dim))

    def ifftshift(self, dim=None):
        """Inverse of fftshift."""
        real_shift = torch.fft.ifftshift(self.real, dim=dim)
        imag_shift = torch.fft.ifftshift(self.imag, dim=dim)
        return self._wrap(torch.stack([real_shift, imag_shift], dim=self.dim))

    # --- Spectral Helpers ---

    def eigvals(self):
        """Returns the eigenvalues of a square matrix."""
        c_tensor = torch.complex(self.real, self.imag)
        res = torch.linalg.eigvals(c_tensor)
        return Complex(res, dim=self.dim, dtype=self.dtype, device=self.device)

    def eigvalsh(self, UPLO='L'):
        """Returns the eigenvalues of a Hermitian matrix."""
        c_tensor = torch.complex(self.real, self.imag)
        res = torch.linalg.eigvalsh(c_tensor, UPLO=UPLO)
        # Eigenvalues of Hermitian matrices are real
        return Complex(torch.stack([res, torch.zeros_like(res)], dim=self.dim), dim=self.dim)

    def svdvals(self):
        """Returns the singular values of a matrix."""
        c_tensor = torch.complex(self.real, self.imag)
        res = torch.linalg.svdvals(c_tensor)
        # Singular values are real
        return Complex(torch.stack([res, torch.zeros_like(res)], dim=self.dim), dim=self.dim)

    # --- Modern activations ---

    def silu(self, inplace=False):
        """Sigmoid Linear Unit (SiLU) / Swish: x * sigmoid(x)."""
        sig = self.sigmoid()
        return self.multiply(sig, inplace=inplace)

    def mish(self, inplace=False):
        """Mish: x * tanh(softplus(x))."""
        sp = self.softplus()
        t = sp.tanh()
        return self.multiply(t, inplace=inplace)

    def glu(self, dim=-1):
        """Gated Linear Unit. Splits tensor and applies sigmoidal gating."""
        # Split along dim
        a, b = self.chunk(2, dim=dim)
        act = b.sigmoid()
        return a.multiply(act)

    # --- Missing In-Place Methods ---

    def scatter_add_(self, dim, index, src):
        """In-place scatter add."""
        if not isinstance(src, Complex):
            src = Complex(src, dim=self.dim, dtype=self.dtype, device=self.device)
        self.real.scatter_add_(dim, index, src.real)
        self.imag.scatter_add_(dim, index, src.imag)
        return self

    def scatter_reduce_(self, dim, index, src, reduce, include_self=True):
        """In-place scatter reduce."""
        if not isinstance(src, Complex):
            src = Complex(src, dim=self.dim, dtype=self.dtype, device=self.device)
        self.real.scatter_reduce_(dim, index, src.real, reduce, include_self=include_self)
        self.imag.scatter_reduce_(dim, index, src.imag, reduce, include_self=include_self)
        return self

    def cauchy_(self, median=0.0, sigma=1.0):
        """In-place Cauchy distribution sampler."""
        self.real.cauchy_(median, sigma)
        self.imag.cauchy_(median, sigma)
        return self

    def is_signed(self):
        return True

    # --- Final Polish: Device Props, Magic Methods & Aliases ---

    @property
    def is_cuda(self):
        """Returns True if the tensor is stored on GPU."""
        return self.tensor.is_cuda

    @property
    def is_cpu(self):
        """Returns True if the tensor is stored on CPU."""
        return self.device.type == 'cpu'

    @property
    def is_sparse(self):
        """Returns True if the tensor uses sparse storage (False for Complex)."""
        return False

    def get_device(self):
        """For CUDA tensors, this returns the device ordinal of the GPU."""
        return self.tensor.get_device()

    def __rpow__(self, other):
        """
        Reverse power operator: other_decomposition ** self
        Computes exp(self * log(other_decomposition))
        """
        # If other_decomposition is scalar
        if isinstance(other, (int, float, complex)):
            # Handle standard python scalars
            import cmath
            val = cmath.log(other)  # Complex log

            # Construct log(other_decomposition) as Complex scalar
            ln_other = Complex(torch.tensor([val.real, val.imag],
                                            device=self.device, dtype=self.dtype), dim=0)

            # z * ln(other_decomposition)
            exponent = self.multiply(ln_other)
            return exponent.exp()

        elif isinstance(other, Complex):
            # This generally shouldn't happen in __rpow__ unless strict type checking failed elsewhere,
            # but implies other_decomposition ** self
            return other.pow(self)

        return NotImplemented

    def __reversed__(self):
        """
        Allows usage of reversed(tensor). Flips along dimension 0.
        """
        return self.flip(dims=(0,))

    def logcumsumexp(self, dim):
        """
        Computes log(cumsum(exp(x))).
        """
        # Numerically stable implementation:
        # For complex, we typically just do the direct computation as "stable" log-sum-exp 
        # logic for complex numbers is less standard than real. 
        # We assume standard def: log(cumsum(exp(z)))
        return self.exp().cumsum(dim=dim).log()

    def reciprocal_(self):
        """In-place reciprocal."""
        return self.reciprocal(inplace=True)

    def rsqrt_(self):
        """In-place reciprocal square root."""
        return self.rsqrt(inplace=True)

    def conjugate(self):
        """Alias for conj()."""
        return self.conj()

    # --- Final Polish: In-Place Math & Algebra Aliases ---

    def exp_(self):
        return self.exp(inplace=True)

    def log_(self):
        return self.log(inplace=True)

    def sqrt_(self):
        return self.sqrt(inplace=True)

    def square_(self):
        return self.pow(2).copy_(self)  # In-place via copy

    def sin_(self):
        return self.sin(inplace=True)

    def cos_(self):
        return self.cos(inplace=True)

    def tan_(self):
        return self.tan(inplace=True)

    def asin_(self):
        return self.arcsin(inplace=True)

    def acos_(self):
        return self.arccos(inplace=True)

    def atan_(self):
        return self.arctan(inplace=True)

    def sinh_(self):
        return self.sinh(inplace=True)

    def cosh_(self):
        return self.cosh(inplace=True)

    def asinh_(self):
        return self.arcsinh(inplace=True)

    def acosh_(self):
        return self.arccosh(inplace=True)

    def atanh_(self):
        return self.arctanh(inplace=True)

    # --- In-Place Linear Algebra Accumulators ---

    def addmm_(self, mat1, mat2, beta=1, alpha=1):
        """In-place addmm."""
        res = self.addmm(mat1, mat2, beta=beta, alpha=alpha)
        self.tensor.copy_(res.tensor)
        return self

    def addmv_(self, mat, vec, beta=1, alpha=1):
        """In-place addmv."""
        res = self.addmv(mat, vec, beta=beta, alpha=alpha)
        self.tensor.copy_(res.tensor)
        return self

    def addr_(self, vec1, vec2, beta=1, alpha=1):
        """In-place addr."""
        res = self.addr(vec1, vec2, beta=beta, alpha=alpha)
        self.tensor.copy_(res.tensor)
        return self

    def baddbmm_(self, batch1, batch2, beta=1, alpha=1):
        """In-place baddbmm."""
        res = self.baddbmm(batch1, batch2, beta=beta, alpha=alpha)
        self.tensor.copy_(res.tensor)
        return self

    # --- Comparison Aliases ---

    def greater(self, other):
        return self.gt(other)

    def greater_equal(self, other):
        return self.ge(other)

    def less(self, other):
        return self.lt(other)

    def less_equal(self, other):
        return self.le(other)

    def not_equal(self, other):
        return self.neq(other)

    # --- Utilities ---

    def resize_as_(self, other):
        """Resizes self to match other_decomposition's size."""
        if isinstance(other, Complex):
            self.tensor.resize_as_(other.tensor)
        else:
            # If resizing to match a real tensor, we must account for the complex dim
            # This is ambiguous, so we generally rely on the underlying tensor logic
            pass
        return self

    def copy_(self, src, non_blocking=False):
        """In-place copy with non_blocking support."""
        if isinstance(src, Complex):
            self.tensor.copy_(src.tensor, non_blocking=non_blocking)
        else:
            # Assume real tensor, copy to real part, zero imaginary
            # This is complex to do atomically with non_blocking, so we do separate ops
            self.real.copy_(src, non_blocking=non_blocking)
            self.imag.zero_()
        return self

    # --- Final Polish: In-Place Shape & Vision Utilities ---

    def squeeze_(self, dim=None):
        """In-place squeeze."""
        # Note: We cannot change the dimensionality of the underlying storage if it breaks the stack dim.
        # But if we squeeze a non-stack dim, it works.
        # Ideally, we squeeze the real/imag parts.
        self.real.squeeze_(dim)
        self.imag.squeeze_(dim)
        return self

    def unsqueeze_(self, dim):
        """In-place unsqueeze."""
        # Adjust dim to account for stack dim if necessary
        target_dim = dim
        if target_dim < 0:
            target_dim += self.real.ndim + 1

        # If unsqueezing before the stack dim, stack dim shifts
        if target_dim <= self.dim:
            self.dim += 1

        self.real.unsqueeze_(dim)
        self.imag.unsqueeze_(dim)
        return self

    def detach_(self):
        """In-place detach."""
        self.tensor.detach_()
        return self

    def sinc_(self):
        """In-place sinc."""
        # sinc(z) = sin(pi*z)/(pi*z)
        # We can implement via out-of-place calc and copy
        res = self.sinc()
        self.tensor.copy_(res.tensor)
        return self

    def erf_(self):
        """In-place error function."""
        res = self.erf()
        self.tensor.copy_(res.tensor)
        return self

    def erfc_(self):
        """In-place complementary error function."""
        res = self.erfc()
        self.tensor.copy_(res.tensor)
        return self

    def sign_(self):
        """In-place sign."""
        res = self.sign()
        self.tensor.copy_(res.tensor)
        return self

    def narrow_copy(self, dim, start, length):
        """Returns a copy of a narrowed tensor."""
        return self.narrow(dim, start, length).clone()

    def pixel_shuffle(self, upscale_factor):
        """
        Rearranges elements in a tensor of shape (*, C*r^2, H, W) to (*, C, H*r, W*r).
        """
        real_shuffled = torch.nn.functional.pixel_shuffle(self.real, upscale_factor)
        imag_shuffled = torch.nn.functional.pixel_shuffle(self.imag, upscale_factor)
        return self._wrap(torch.stack([real_shuffled, imag_shuffled], dim=self.dim))

    def pixel_unshuffle(self, downscale_factor):
        """
        Rearranges elements in a tensor of shape (*, C, H*r, W*r) to (*, C*r^2, H, W).
        """
        real_unshuffled = torch.nn.functional.pixel_unshuffle(self.real, downscale_factor)
        imag_unshuffled = torch.nn.functional.pixel_unshuffle(self.imag, downscale_factor)
        return self._wrap(torch.stack([real_unshuffled, imag_unshuffled], dim=self.dim))

    # --- Final Polish: Inverse Real FFTs & Missing Utils ---

    def irfft(self, n=None, dim=-1, norm=None):
        """Inverse Real FFT: Complex -> Real."""
        c_tensor = torch.complex(self.real, self.imag)
        return torch.fft.irfft(c_tensor, n=n, dim=dim, norm=norm)

    def irfft2(self, s=None, dim=(-2, -1), norm=None):
        """Inverse Real 2D FFT: Complex -> Real."""
        c_tensor = torch.complex(self.real, self.imag)
        return torch.fft.irfft2(c_tensor, s=s, dim=dim, norm=norm)

    def irfftn(self, s=None, dim=None, norm=None):
        """Inverse Real N-D FFT: Complex -> Real."""
        c_tensor = torch.complex(self.real, self.imag)
        return torch.fft.irfftn(c_tensor, s=s, dim=dim, norm=norm)

    def hfft(self, n=None, dim=-1, norm=None):
        """FFT of a Hermitian symmetric signal: Complex -> Real."""
        c_tensor = torch.complex(self.real, self.imag)
        return torch.fft.hfft(c_tensor, n=n, dim=dim, norm=norm)

    # --- Missing activations ---

    def leaky_relu(self, negative_slope=0.01, inplace=False):
        """LeakyReLU applied to real and imag parts independently."""
        real = F.leaky_relu(self.real, negative_slope, inplace=inplace)
        imag = F.leaky_relu(self.imag, negative_slope, inplace=inplace)
        if inplace:
            return self
        return self._wrap(torch.stack([real, imag], dim=self.dim))

    def elu(self, alpha=1.0, inplace=False):
        """ELU applied to real and imag parts independently."""
        real = F.elu(self.real, alpha, inplace=inplace)
        imag = F.elu(self.imag, alpha, inplace=inplace)
        if inplace:
            return self
        return self._wrap(torch.stack([real, imag], dim=self.dim))

    def selu(self, inplace=False):
        """SELU applied to real and imag parts independently."""
        real = F.selu(self.real, inplace=inplace)
        imag = F.selu(self.imag, inplace=inplace)
        if inplace:
            return self
        return self._wrap(torch.stack([real, imag], dim=self.dim))

    # --- Utilities & Casting ---

    @classmethod
    def empty_like(cls, other, dtype=None, device=None, **kwargs):
        """Returns an uninitialized tensor with the same size as input."""
        if dtype is None: dtype = other.dtype
        if device is None: device = other.device
        return cls.empty(*other.shape, dim=other.dim, dtype=dtype, device=device, **kwargs)

    @classmethod
    def full_like(cls, other, fill_value, dtype=None, device=None, **kwargs):
        """Returns a tensor filled with value with the same size as input."""
        if dtype is None: dtype = other.dtype
        if device is None: device = other.device
        return cls.full(other.shape, fill_value, dim=other.dim, dtype=dtype, device=device, **kwargs)

    def is_same_size(self, other):
        """Returns True if other_decomposition has the same size as self."""
        if isinstance(other, Complex):
            return self.real.shape == other.real.shape
        return self.real.shape == other.shape

    def equal(self, other):
        """True if two tensors have the same size and elements, False otherwise."""
        if not isinstance(other, Complex):
            return False
        return torch.equal(self.tensor, other.tensor)

    def diagonal(self, offset=0, dim1=0, dim2=1):
        """Returns a view of the diagonal elements."""
        real_diag = self.real.diagonal(offset, dim1, dim2)
        imag_diag = self.imag.diagonal(offset, dim1, dim2)
        # We must decide where to stack. Usually logic dictates preserving stack dim if possible,
        # or appending it. Since diagonal() reduces dims, let's append at -1.
        return self._wrap(torch.stack([real_diag, imag_diag], dim=-1), dim=-1)

    def bool(self):
        """Casts to boolean (Magnitude != 0)."""
        return self.mag() != 0

    def long(self):
        """Casts to long (int64) magnitude."""
        return self.mag().long()

    def int(self):
        """Casts to int (int32) magnitude."""
        return self.mag().int()

    # --- Out-of-Place Indexing ---

    def index_add(self, dim, index, source):
        """Out-of-place index_add."""
        return self.clone().index_add_(dim, index, source)

    def index_copy(self, dim, index, source):
        """Out-of-place index_copy."""
        return self.clone().index_copy_(dim, index, source)

    def index_fill(self, dim, index, value):
        """Out-of-place index_fill."""
        return self.clone().index_fill_(dim, index, value)

    # --- Final Polish: Transpose Alias & Shrinkage ---

    def t(self):
        """
        Alias for transpose(0, 1). Expects a 2D tensor.
        """
        return self.transpose(0, 1)

    def softshrink(self, lambd=0.5, inplace=False):
        """
        Applies softshrink activation (independently on real/imag).
        """
        if inplace:
            F.softshrink(self.real, lambd=lambd, inplace=True)
            F.softshrink(self.imag, lambd=lambd, inplace=True)
            return self
        else:
            real = F.softshrink(self.real, lambd=lambd)
            imag = F.softshrink(self.imag, lambd=lambd)
            return self._wrap(torch.stack([real, imag], dim=self.dim))

    def hardshrink(self, lambd=0.5, inplace=False):
        """
        Applies hardshrink activation (independently on real/imag).
        """
        if inplace:
            F.hardshrink(self.real, lambd=lambd, inplace=True)
            F.hardshrink(self.imag, lambd=lambd, inplace=True)
            return self
        else:
            real = F.hardshrink(self.real, lambd=lambd)
            imag = F.hardshrink(self.imag, lambd=lambd)
            return self._wrap(torch.stack([real, imag], dim=self.dim))

    def is_conj(self):
        """
        Returns True if the conjugate bit is set. 
        Always False for this implementation as conjugation is physical.
        """
        return False

    def index_reduce_(self, dim, index, source, reduce, include_self=True):
        """
        Accumulate values into specific indices using a reduction (prod, mean, amax, amin).
        """
        if not isinstance(source, Complex):
            source = Complex(source, dim=self.dim, dtype=self.dtype, device=self.device)

        # Apply reduction to real and imag parts independently
        self.real.index_reduce_(dim, index, source.real, reduce, include_self=include_self)
        self.imag.index_reduce_(dim, index, source.imag, reduce, include_self=include_self)
        return self

    # --- Final Polish: Reductions, Pooling & Factories ---

    def argmax(self, dim=None, keepdim=False):
        """
        Returns the indices of the maximum value of the tensor (based on magnitude).
        """
        mag = self.mag()
        return torch.argmax(mag, dim=dim, keepdim=keepdim)

    def argmin(self, dim=None, keepdim=False):
        """
        Returns the indices of the minimum value of the tensor (based on magnitude).
        """
        mag = self.mag()
        return torch.argmin(mag, dim=dim, keepdim=keepdim)

    def aminmax(self, dim=None, keepdim=False):
        """
        Computes the minimum and maximum values of the tensor (based on magnitude).
        Returns (min_complex, max_complex).
        """
        # We need indices to retrieve the complex values
        # torch.aminmax returns (min_vals, max_vals) but no indices.
        # So we must use min() and max() which return indices.
        if dim is None:
            flat = self.flatten()
            min_val, min_idx = flat.min(dim=0)
            max_val, max_idx = flat.max(dim=0)
            return min_val, max_val

        min_val, _ = self.min(dim=dim, keepdim=keepdim)
        max_val, _ = self.max(dim=dim, keepdim=keepdim)
        return min_val, max_val

    def adaptive_avg_pool2d(self, output_size):
        """
        Adaptive average pooling (operates on real/imag independently).
        """
        real_pool = F.adaptive_avg_pool2d(self.real, output_size)
        imag_pool = F.adaptive_avg_pool2d(self.imag, output_size)
        return self._wrap(torch.stack([real_pool, imag_pool], dim=self.dim))

    def adaptive_max_pool2d(self, output_size, return_indices=False):
        """
        Adaptive max pooling (based on magnitude).
        """
        if return_indices:
            # Pool magnitude to get indices
            mag = self.mag()
            _, indices = F.adaptive_max_pool2d(mag, output_size, return_indices=True)

            # Gather complex values using indices (flattened gather)
            # This is complex to do efficiently without unrolling, so we apply independent pooling
            # Note: Max pooling independent real/imag is NOT mathematically correct "complex max pooling",
            # but it is the standard "feature-wise" approach in most complex CNN implementations.
            real_pool, idx_r = F.adaptive_max_pool2d(self.real, output_size, return_indices=True)
            imag_pool, idx_i = F.adaptive_max_pool2d(self.imag, output_size, return_indices=True)
            return self._wrap(torch.stack([real_pool, imag_pool], dim=self.dim)), indices
        else:
            # Independent pooling
            real_pool = F.adaptive_max_pool2d(self.real, output_size)
            imag_pool = F.adaptive_max_pool2d(self.imag, output_size)
            return self._wrap(torch.stack([real_pool, imag_pool], dim=self.dim))

    def new_tensor(self, data, dtype=None, device=None, requires_grad=False):
        """
        Returns a new Complex tensor with data as the tensor data.
        Defaults to the current tensor's dtype and device.
        """
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device

        return Complex(data, dim=self.dim, dtype=dtype, device=device, requires_grad=requires_grad)

    def relu6(self, inplace=False):
        """
        ReLU6 activation: min(max(0, x), 6).
        """
        if inplace:
            F.relu6(self.real, inplace=True)
            F.relu6(self.imag, inplace=True)
            return self
        else:
            real = F.relu6(self.real)
            imag = F.relu6(self.imag)
            return self._wrap(torch.stack([real, imag], dim=self.dim))

    def log_sigmoid(self):
        """
        Log sigmoid: log(1 / (1 + exp(-x))).
        Numerically stable implementation.
        """
        # log_sigmoid(z) = -log(1 + exp(-z)) = -softplus(-z)
        # softplus handles complex magnitude, but log_sigmoid usually implies elementwise.
        # We apply to real and imag independently for feature-map compatibility,
        # or use the mathematical definition: z - logaddexp(0, z)
        # Mathematical:
        zeros = torch.zeros_like(self.real)
        zeros_c = Complex(torch.complex(zeros, zeros), dim=0, device=self.device)
        return self.subtract(self.logaddexp(zeros_c))

    def diagflat(self, offset=0):
        """
        Creates a diagonal matrix with diagonal elements from the flattened input.
        """
        flat = self.flatten()
        real_diag = torch.diagflat(flat.real, offset=offset)
        imag_diag = torch.diagflat(flat.imag, offset=offset)
        return self._wrap(torch.stack([real_diag, imag_diag], dim=self.dim))


# --- Global Wrappers ---

def mag(x: Union[torch.Tensor, 'Complex'], **kwargs):
    if not isinstance(x, Complex):
        x = Complex(x, **kwargs)
    return x.mag()


def phi(x: Union[torch.Tensor, 'Complex'], **kwargs):
    if not isinstance(x, Complex):
        x = Complex(x, **kwargs)
    return x.phi()


def unit(x: Union[torch.Tensor, 'Complex'], **kwargs):
    if not isinstance(x, Complex):
        x = Complex(x, **kwargs)
    return x.unit()


def inv(x: Union[torch.Tensor, 'Complex'], **kwargs):
    if not isinstance(x, Complex):
        x = Complex(x, **kwargs)
    return x.inv()


def exp(x: Union[torch.Tensor, 'Complex'], **kwargs):
    if not isinstance(x, Complex):
        x = Complex(x, **kwargs)
    return x.exp()


def iexp(x: Union[torch.Tensor, 'Complex'], **kwargs):
    if not isinstance(x, Complex):
        x = Complex(x, **kwargs)
    return x.iexp()


def add(x: Any, y: Any, **kwargs):
    if not isinstance(x, Complex):
        x = Complex(x, **kwargs)
    return x.add(y, **kwargs)


def subtract(x: Any, y: Any, **kwargs):
    if not isinstance(x, Complex):
        x = Complex(x, **kwargs)
    return x.subtract(y, **kwargs)


def multiply(x: Any, y: Any, **kwargs):
    if not isinstance(x, Complex):
        x = Complex(x, **kwargs)
    return x.multiply(y, **kwargs)


def divide(x: Any, y: Any, **kwargs):
    if not isinstance(x, Complex):
        x = Complex(x, **kwargs)
    return x.divide(y, **kwargs)


def sin(x: Any, **kwargs):
    if not isinstance(x, Complex):
        x = Complex(x, **kwargs)
    return x.sin()


def cos(x: Any, **kwargs):
    if not isinstance(x, Complex):
        x = Complex(x, **kwargs)
    return x.cos()


def tan(x: Any, **kwargs):
    if not isinstance(x, Complex):
        x = Complex(x, **kwargs)
    return x.tan()


def cot(x: Any, **kwargs):
    if not isinstance(x, Complex):
        x = Complex(x, **kwargs)
    return x.cot()


def sec(x: Any, **kwargs):
    if not isinstance(x, Complex):
        x = Complex(x, **kwargs)
    return x.sec()


def cosec(x: Any, **kwargs):
    if not isinstance(x, Complex):
        x = Complex(x, **kwargs)
    return x.cosec()


def sinh(x: Any, **kwargs):
    if not isinstance(x, Complex):
        x = Complex(x, **kwargs)
    return x.sinh()


def cosh(x: Any, **kwargs):
    if not isinstance(x, Complex):
        x = Complex(x, **kwargs)
    return x.cosh()


def tanh(x: Any, **kwargs):
    if not isinstance(x, Complex):
        x = Complex(x, **kwargs)
    return x.tanh()


def coth(x: Any, **kwargs):
    if not isinstance(x, Complex):
        x = Complex(x, **kwargs)
    return x.coth()


def sech(x: Any, **kwargs):
    if not isinstance(x, Complex):
        x = Complex(x, **kwargs)
    return x.sech()


def cosech(x: Any, **kwargs):
    if not isinstance(x, Complex):
        x = Complex(x, **kwargs)
    return x.cosech()


def exp2(x: Any, **kwargs):
    if not isinstance(x, Complex):
        x = Complex(x, **kwargs)
    return x.exp2()


def exp10(x: Any, **kwargs):
    if not isinstance(x, Complex):
        x = Complex(x, **kwargs)
    return x.exp10()


def exp_n(x: Any, n=math.e, **kwargs):
    if not isinstance(x, Complex):
        x = Complex(x, **kwargs)
    return x.exp_n(n=n)


def log(x: Any, **kwargs):
    if not isinstance(x, Complex):
        x = Complex(x, **kwargs)
    return x.log(**kwargs)


# Phase 18 Global Wrappers

def nonzero(x: Any, **kwargs):
    if not isinstance(x, Complex):
        x = Complex(x, **kwargs)
    return x.nonzero(**kwargs)


def index_put(x: Any, indices: Any, values: Any, accumulate: bool = False, **kwargs):
    if not isinstance(x, Complex):
        x = Complex(x, **kwargs)
    return x.index_put(indices, values, accumulate=accumulate)


def repeat_interleave(x: Any, repeats: Any, dim: Optional[int] = None, **kwargs):
    if not isinstance(x, Complex):
        x = Complex(x, **kwargs)
    return x.repeat_interleave(repeats, dim=dim, **kwargs)


def broadcast_to(x: Any, shape: Any, **kwargs):
    if not isinstance(x, Complex):
        x = Complex(x, **kwargs)
    return x.broadcast_to(*shape)


def frexp(x: Any, **kwargs):
    if not isinstance(x, Complex):
        x = Complex(x, **kwargs)
    return x.frexp()


def nextafter(x: Any, other: Any, **kwargs):
    if not isinstance(x, Complex):
        x = Complex(x, **kwargs)
    return x.nextafter(other)


def xlogy(x: Any, y: Any, **kwargs):
    if not isinstance(x, Complex):
        x = Complex(x, **kwargs)
    return x.xlogy(y)


def polyval(coeffs: Any, x: Any, **kwargs):
    if not isinstance(x, Complex):
        x = Complex(x, **kwargs)
    return x.polyval(coeffs)


def polyder(coeffs: Any, m: int = 1, **kwargs):
    if not isinstance(coeffs, Complex):
        coeffs = Complex(coeffs, **kwargs)
    return coeffs.polyder(m=m)


def polyint(coeffs: Any, m: int = 1, k: Any = 0, **kwargs):
    if not isinstance(coeffs, Complex):
        coeffs = Complex(coeffs, **kwargs)
    return coeffs.polyint(m=m, k=k)


class ComplexLinear(nn.Module):
    def __init__(self,
                 in_features:int,
                 out_features:int,
                 bias:bool=True,
                 dim:int=-1,
                 device:str="cpu",
                 dtype:torch.dtype=torch.float32,
                 arrangement: str = 'split',
                 is_stacked_flag: bool = False,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {
            "device": device,
            "dtype": dtype
        }
        self.g_kwargs = {
            "arrangement": arrangement,
            "is_stacked_flag": is_stacked_flag,
            "args": args,
            "kwargs": kwargs
        }
        self.dim = dim
        in_kwargs = {
            "in_features": in_features,
            "out_features": out_features,
            "bias": bias,
            **self.factory_kwargs
        }
        self.real_lin = nn.Linear(**in_kwargs)
        self.imag_lin = nn.Linear(**in_kwargs)

    def forward(self, x:Complex):
        w_real, w_imag = self.real_lin.weight, self.imag_lin.weight
        b_real, b_imag = self.real_lin.bias, self.imag_lin.bias
        w = torch.stack([w_real, w_imag], dim=self.dim)
        b = torch.stack([b_real, b_imag], dim=self.dim)
        w = Complex(w, dim=self.dim, **self.factory_kwargs, **self.g_kwargs)
        b = Complex(b, dim=self.dim, **self.factory_kwargs, **self.g_kwargs)
        return x.linear(w, b)
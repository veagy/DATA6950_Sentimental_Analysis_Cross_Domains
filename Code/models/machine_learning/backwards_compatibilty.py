"""
Wraps sklearn and other non-PyTorch models for PyTorch save/load and gradient flow.

Limitations:
- safetensors: Only tensor parameters are saved. The wrapped sklearn object and its
  non-tensor state (e.g., classes_) are not restored. Full restore needs pickle-based
  formats (.pkl, .pt) or manual base-model reconstruction.
- Backprop: Works only for models with a differentiable forward (e.g., linear models
  with coef_/intercept_). Tree-based and most other sklearn models do not support
  gradient-based updates through the wrapper.
"""

import os
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib
from torch.func import vmap

class PyTorchWrapper(nn.Module):
    """
    Wraps an arbitrary Python class instance (e.g., a scikit-learn model, custom class) 
    such that it can be saved and loaded via PyTorch, and its tensor/array 
    attributes can receive gradients during fine-tuning.
    """
    def __init__(self, base_model, requires_grad=True):
        super(PyTorchWrapper, self).__init__()
        # Temporary flag to prevent __getattr__ from resolving from base_model
        # during parameter registration, which confuses PyTorch's hasattr() checks.
        self._is_wrapping = True
        self.base_model = base_model
        self._requires_grad = requires_grad
        self._register_attributes()
        # Finished wrapping
        self._is_wrapping = False

    def _register_attributes(self):
        """
        Dynamically discovers all array/tensor properties of the base_model
        and registers them as PyTorch parameters or buffers.
        """
        for attr_name, attr_value in list(self.base_model.__dict__.items()):
            if attr_name.startswith('__'):
                continue
            if isinstance(attr_value, np.ndarray):
                tensor = torch.from_numpy(attr_value)
                self._register_tensor(attr_name, tensor)
            elif isinstance(attr_value, torch.Tensor):
                self._register_tensor(attr_name, attr_value)
            elif isinstance(attr_value, (list, tuple)) and len(attr_value) > 0 and isinstance(attr_value[0], (int, float)):
                try:
                    tensor = torch.tensor(attr_value)
                    self._register_tensor(attr_name, tensor)
                except Exception:
                    pass
            elif isinstance(attr_value, float):
                # Float scalars might be optimized, treat as tensors
                tensor = torch.tensor(attr_value, dtype=torch.float32)
                self._register_tensor(attr_name, tensor)

    def _register_tensor(self, name, tensor):
        if tensor.is_floating_point() and self._requires_grad:
            param = nn.Parameter(tensor.clone().detach().requires_grad_(True))
            self.register_parameter(name, param)
            setattr(self.base_model, name, param)
        else:
            buffer = tensor.clone().detach()
            self.register_buffer(name, buffer)
            setattr(self.base_model, name, buffer)

    def forward(self, *args, **kwargs):
        if hasattr(self.base_model, 'forward'):
            return self.base_model.forward(*args, **kwargs)
        # Differentiable path for linear sklearn models (coef_/intercept_)
        # Enables backprop in mixed model scenarios
        coef = getattr(self, 'coef_', None)
        if coef is None:
            coef = getattr(self.base_model, 'coef_', None)
        if coef is not None:
            if isinstance(coef, np.ndarray):
                dev = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
                coef = torch.from_numpy(np.asarray(coef, dtype=np.float32)).to(dev)
            if isinstance(coef, (nn.Parameter, torch.Tensor)):
                X = args[0] if args else kwargs.get('X') or kwargs.get('x')
                if X is not None:
                    X = torch.as_tensor(X, dtype=coef.dtype, device=coef.device)
                    intercept = None
                    inc = getattr(self, 'intercept_', None)
                    if inc is None:
                        inc = getattr(self.base_model, 'intercept_', None)
                    if inc is not None:
                        if isinstance(inc, (nn.Parameter, torch.Tensor)):
                            intercept = inc
                        elif isinstance(inc, np.ndarray):
                            intercept = torch.from_numpy(np.asarray(inc, dtype=np.float32)).to(coef.device)
                        else:
                            try:
                                intercept = torch.tensor(float(inc), dtype=coef.dtype, device=coef.device)
                            except Exception:
                                pass
                    return F.linear(X, coef, intercept)
        # Fallback: non-differentiable (sklearn predict breaks gradient flow)
        if hasattr(self.base_model, 'predict'):
            return self.base_model.predict(*args, **kwargs)
        elif hasattr(self.base_model, '__call__'):
            return self.base_model(*args, **kwargs)
        else:
            raise NotImplementedError("Wrapped base model missing callable method ('forward', 'predict', or '__call__')")

    def fit(self, *args, **kwargs):
        if hasattr(self.base_model, 'fit'):
            result = self.base_model.fit(*args, **kwargs)
            # Re-discover attributes in case fit created new ones
            self._is_wrapping = True
            self._register_attributes()
            self._is_wrapping = False
            return result
        raise NotImplementedError("Wrapped base model does not implement 'fit'")

    def __getattr__(self, name):
        # Prevent recursion and PyTorch confusion during init
        if getattr(self, '_is_wrapping', False):
            raise AttributeError(name)
        
        # Try normal PyTorch attribute resolution first
        try:
            return super().__getattr__(name)
        except AttributeError:
            # Delegate to the base model
            if hasattr(self.base_model, name):
                return getattr(self.base_model, name)
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")


def _test_sklearn_save_load_pkl():
    """Fit sklearn.LogisticRegression, wrap with PyTorchWrapper, save/load via joblib."""
    try:
        from sklearn.linear_model import LogisticRegression as SklearnLR
    except ImportError:
        print("  sklearn not installed, skipping pkl test")
        return False
    np.random.seed(42)
    X = np.random.randn(50, 5).astype(np.float32)
    y = (X @ np.random.randn(5) + 0.1 > 0).astype(np.int64)
    sk = SklearnLR(max_iter=500).fit(X, y)
    wrapper = PyTorchWrapper(sk)
    pred_orig = wrapper.forward(torch.from_numpy(X)).detach().numpy()
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        path = f.name
    try:
        joblib.dump(wrapper, path)
        loaded = joblib.load(path)
        pred_loaded = loaded.forward(torch.from_numpy(X)).detach().numpy()
        ok = np.allclose(pred_orig, pred_loaded)
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass
    return ok


def _test_sklearn_save_load_pt():
    """Fit sklearn.LogisticRegression, wrap with PyTorchWrapper, save/load via torch.save."""
    try:
        from sklearn.linear_model import LogisticRegression as SklearnLR
    except ImportError:
        print("  sklearn not installed, skipping pt test")
        return False
    np.random.seed(42)
    X = np.random.randn(50, 5).astype(np.float32)
    y = (X @ np.random.randn(5) + 0.1 > 0).astype(np.int64)
    sk = SklearnLR(max_iter=500).fit(X, y)
    wrapper = PyTorchWrapper(sk)
    pred_orig = wrapper.forward(torch.from_numpy(X)).detach().numpy()
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        path = f.name
    try:
        torch.save(wrapper, path)
        loaded = torch.load(path, map_location='cpu', weights_only=False)
        pred_loaded = loaded.forward(torch.from_numpy(X)).detach().numpy()
        ok = np.allclose(pred_orig, pred_loaded)
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass
    return ok


def _test_sklearn_save_load_safetensors():
    """Save wrapper.state_dict() to safetensors, load into new instance. Note: base sklearn
    object is not in state_dict; only tensor params. Requires reconstructing base model for
    full restore; here we load params into an existing wrapper with same base."""
    try:
        from sklearn.linear_model import LogisticRegression as SklearnLR
        from safetensors.torch import save_file, load_file
    except ImportError as e:
        print(f"  sklearn or safetensors not installed, skipping safetensors test: {e}")
        return False
    np.random.seed(42)
    X = np.random.randn(50, 5).astype(np.float32)
    y = (X @ np.random.randn(5) + 0.1 > 0).astype(np.int64)
    sk = SklearnLR(max_iter=500).fit(X, y)
    wrapper = PyTorchWrapper(sk)
    pred_orig = wrapper.forward(torch.from_numpy(X)).detach().numpy()
    with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as f:
        path = f.name
    try:
        save_file(wrapper.state_dict(), path)
        sk2 = SklearnLR(max_iter=500).fit(X, y)
        wrapper2 = PyTorchWrapper(sk2)
        state = load_file(path)
        wrapper2.load_state_dict(state, strict=False)
        pred_loaded = wrapper2.forward(torch.from_numpy(X)).detach().numpy()
        ok = np.allclose(pred_orig, pred_loaded)
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass
    return ok


def _test_backprop_mixed_models():
    """Build mixed model (nn.Linear + PyTorchWrapper(sklearn)), run backward, check gradients."""
    try:
        from sklearn.linear_model import LogisticRegression as SklearnLR
    except ImportError:
        print("  sklearn not installed, skipping backprop test")
        return False
    np.random.seed(42)
    X_np = np.random.randn(20, 10).astype(np.float32)
    X_5 = X_np[:, :5]
    y_np = (X_5 @ np.random.randn(5) + 0.1 > 0).astype(np.int64)
    sk = SklearnLR(max_iter=500).fit(X_5, y_np)
    wrapper = PyTorchWrapper(sk)
    linear = nn.Linear(10, 5)
    model = nn.Sequential(linear, wrapper)
    X = torch.from_numpy(X_np).requires_grad_(True)
    logits = model(X)
    if logits.requires_grad:
        loss = logits.sum()
        loss.backward()
        has_linear_grad = linear.weight.grad is not None
        c = getattr(wrapper, 'coef_', None)
        if c is None:
            c = getattr(wrapper.base_model, 'coef_', None)
        has_wrapper_grad = c is not None and hasattr(c, 'grad') and c.grad is not None
        ok = has_linear_grad and has_wrapper_grad
        if ok:
            print("  Backprop through mixed model: OK")
        else:
            print("  Backprop through mixed model: FAILED (sklearn predict is non-differentiable)")
        return ok
    print("  Backprop through mixed model: FAILED (output has no grad)")
    return False


if __name__ == "__main__":
    print("Sklearn save-load and backprop tests (backwards_compatibilty.py)")
    results = {}
    results["pkl"] = _test_sklearn_save_load_pkl()
    results["pt"] = _test_sklearn_save_load_pt()
    results["safetensors"] = _test_sklearn_save_load_safetensors()
    results["backprop"] = _test_backprop_mixed_models()
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")


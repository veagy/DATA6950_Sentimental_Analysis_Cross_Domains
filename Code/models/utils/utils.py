import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Any, Optional, Tuple, Union, Callable, List, Dict
import math
import functools
import pickle

try:
    import bitsandbytes  # pyright: ignore[reportMissingImports]
except ImportError:
    bitsandbytes = None
import numpy as np
import pandas as pd  # pyright: ignore[reportMissingImports]
import tqdm  # pyright: ignore[reportMissingModuleSource]
import os
import json

from torch.nn.modules.module import T, _grad_t
from torch.utils.hooks import RemovableHandle
from abc import abstractmethod


class _PipelineDelegate(nn.Module):
    """Wrapper that delegates forward to the underlying module without going through __call__."""
    def __init__(self, delegate: nn.Module):
        super().__init__()
        self.delegate = delegate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.delegate.forward(x)


def _move_nested_estimators_to_device(module, device):
    """Recursively move nested estimators (not registered submodules) to device."""
    if device is None or str(device) == 'cpu':
        return
    attrs = ('estimators_', 'estimator', '_classifier', 'regressor_', '_estimators',
             'final_estimator_', 'best_estimator_', '_base_estimator')
    for attr in attrs:
        try:
            val = getattr(module, attr, None)
            if val is None:
                continue
            if hasattr(val, 'to') and callable(getattr(val, 'to')):
                try:
                    val.to(device)
                except Exception:
                    pass
            elif isinstance(val, (list, tuple)):
                for item in val:
                    if hasattr(item, 'to') and callable(getattr(item, 'to')):
                        try:
                            item.to(device)
                        except Exception:
                            pass
            elif isinstance(val, dict):
                for item in val.values():
                    if hasattr(item, 'to') and callable(getattr(item, 'to')):
                        try:
                            item.to(device)
                        except Exception:
                            pass
        except Exception:
            pass
    if hasattr(module, 'search') and hasattr(module.search, 'best_estimator_'):
        try:
            module.search.best_estimator_.to(device)
        except Exception:
            pass


__all__ = [
    "ActFuncWrapper",
    "ActFuncUtils",
    "Forward_hook",
    "Backward_hook",
    "DLModule",
    "MLModule",
    "MLRegressor",
    "MLClassifier",
    "MLCluster",
    "MLTransform",
    "save_checkpoint",
    "compute_total_grad_norm",
]


class ActFuncWrapper(nn.Module):
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        result = self.func(x, *self.args, **self.kwargs)
        return result


class ActFuncUtils(nn.Module):
    def __init__(self,
                 funcs: Union[str, nn.Module, Callable,
                 List[Union[str, nn.Module, Callable]],
                 Dict[str, Union[str, nn.Module, Callable]],
                 Tuple[Union[str, nn.Module, Callable]]
                 ],
                 *args, **kwargs):
        super().__init__()
        if isinstance(funcs, str):
            from ...models.deep_learning.activations.ActivationFunction import Activation
            self.funcs = Activation(funcs, *args, **kwargs)
            self.N = 1
        elif isinstance(funcs, Callable):
            self.funcs = ActFuncWrapper(funcs, *args, **kwargs)
            self.N = 1
        elif isinstance(funcs, nn.Module):
            self.funcs = funcs
            self.N = 1
        elif isinstance(funcs, Union[list, tuple]):
            self.funcs = nn.ModuleList([])
            for func in funcs:
                if isinstance(func, str):
                    from ...models.deep_learning.activations.ActivationFunction import Activation
                    act_func = Activation(func, *args, **kwargs)
                    self.funcs.append(act_func)
                elif isinstance(func, Callable):
                    act_func = ActFuncWrapper(func, *args, **kwargs)
                    self.funcs.append(act_func)
                elif isinstance(func, nn.Module):
                    for param in func.parameters():
                        param.requires_grad = False
                    self.funcs.append(func)
            self.N = len(self.funcs)
        elif isinstance(funcs, dict):
            self.funcs = nn.ModuleDict({})
            for key, func in funcs.items():
                if isinstance(func, str):
                    from ...models.deep_learning.activations.ActivationFunction import Activation
                    act_func = Activation(func, *args, **kwargs)
                    self.funcs[key] = act_func
                elif isinstance(func, Callable):
                    act_func = ActFuncWrapper(func, *args, **kwargs)
                    self.funcs[key] = act_func
                elif isinstance(func, nn.Module):
                    for param in func.parameters():
                        param.requires_grad = False
                    self.funcs[key] = func
            self.N = list(self.funcs)
        else:
            self.funcs = None
            self.N = 0

    def train(self, mode=True):
        super().train(mode)
        if isinstance(self.funcs, nn.ModuleList):
            for func in self.funcs:
                func.eval()
        elif isinstance(self.funcs, nn.ModuleDict):
            for key, func in self.funcs.items():
                func.eval()
        elif isinstance(self.funcs, nn.Module):
            self.funcs.eval()
        return self

    def get_len(self):
        return self.N

    def get_funcs(self):
        return self.funcs

    def get_outputs(self, x: torch.Tensor):
        if isinstance(self.funcs, nn.ModuleList):
            outputs = []
            for func in self.funcs:
                output = func(x)
                outputs.append(output)
            return outputs
        elif isinstance(self.funcs, nn.ModuleDict):
            outputs = {}
            for key, func in self.funcs.items():
                output = func(x)
                outputs[key] = output
            return outputs
        elif isinstance(self.funcs, nn.Module):
            return self.funcs(x)


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


class DLModule(nn.Module):
    """
    A robust parent class for Deep Learning models.
    Supports:
    - Generic Training Loop (Mixed Precision, Gradient Accumulation)
    - Fine-tuning (adapters via PEFT, weight decay)
    - Model I/O (Save/Load/From Pretrained)
    - Hooks
    """

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)

    def __init__(self, *args, **kwargs):
        super().__init__()
        # Placeholder for configuration
        self._quant_info = None
        self._is_quantized = None
        self.config = kwargs.get('config', {})
        # System linking: attached by default; use detach_pipeline() to opt out
        self._detached = False

    @property
    def device_param(self):
        """Returns the device of the model parameters."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return getattr(self, '_device', torch.device('cpu'))

    @device_param.setter
    def device_param(self, value):
        self._device = torch.device(value) if isinstance(value, (str, torch.device)) else value
        self.to(self._device)

    @property
    def dtype_param(self):
        """Returns the dtype of the model parameters."""
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return getattr(self, '_dtype', torch.float32)

    @dtype_param.setter
    def dtype_param(self, value):
        self._dtype = value
        self.to(dtype=self._dtype)

    def __call__(self, *args, **kwargs):
        """Standard PyTorch call, removing legacy system pipeline hooks."""
        # Auto-cast input for quantized models
        if getattr(self, '_is_quantized', False) and args and isinstance(args[0], torch.Tensor):
            try:
                param_dtype = next(self.parameters()).dtype
                if args[0].dtype != param_dtype:
                    args = (args[0].to(param_dtype),) + args[1:]
            except StopIteration:
                pass

        return super().__call__(*args, **kwargs)



    def get_class_type(self) -> str:
        """Return class name for registry and system linking."""
        return type(self).__name__

    def get_manifest(self) -> Dict[str, Any]:
        """Minimal manifest for system/checkpoint compatibility."""
        return {
            "class_type": self.get_class_type(),
            "version_stamp": getattr(self, '_version_stamp', ''),
            "graph_anchor": getattr(self, '_version_stamp', '')[:16] if hasattr(self, '_version_stamp') else '',
            "sub_model_stamps": {},
            "parent_anchor": "",
            "training_state": {"optimizer": None, "lr": None, "scheduler": None},
            "accuracy_metrics": {"loss": None, "accuracy": None, "latency_ms": None},
            "encryption_meta": {"aes_salt": None},
        }

    def fit(self,
            data: Union[torch.utils.data.DataLoader, Any],
            epochs: int = 1,
            batch_size: int = 32,
            learning_rate: float = 1e-3,
            loss: Union[Callable, nn.Module, str] = 'mse',
            optimizer: Union[Callable, nn.Module, str] = 'adamw',
            lr_scheduler: Union[nn.Module, str, Callable] = None,
            weight_decay: float = 0.0,
            warmup: int = 0,
            gradient_accumulation_steps: int = 1,
            mixed_precision: bool = False,
            use_accelerate: bool = False,
            show_progress_bar: bool = True,
            verbose: bool = True,
            save_dir: str = 'checkpoints',
            save_type: str = 'pt',
            val_split: float = 0.0,
            on_train_start: Optional[Callable] = None,
            on_epoch_end: Optional[Callable] = None,
            on_train_end: Optional[Callable] = None,
            on_error: Optional[Callable] = None):
        """
        Generic training loop with val_split, lifecycle hooks, and error recovery.
        """
        val_dataloader = None
        # 1. Setup Data
        if not isinstance(data, torch.utils.data.DataLoader):
            # (X, y) tuple must be handled first — tuples have __getitem__/__len__
            if isinstance(data, (tuple, list)) and len(data) == 2:
                from ...train.utils.data_loader import make_loader
                out = make_loader(data[0], targets=data[1], batch_size=batch_size, val_split=val_split)
                dataloader, val_dataloader = (out[0], out[1]) if isinstance(out, tuple) else (out, None)
            # Dataset-like: __getitem__ + __len__ (but not tuple/list of 2)
            elif hasattr(data, '__getitem__') and hasattr(data, '__len__'):
                from ...train.utils.data_loader import make_loader
                out = make_loader(data, batch_size=batch_size, val_split=val_split)
                dataloader, val_dataloader = (out[0], out[1]) if isinstance(out, tuple) else (out, None)
            else:
                # Raw arrays, dict — use make_loader
                from ...train.utils.data_loader import make_loader
                out = make_loader(data, batch_size=batch_size, val_split=val_split)
                dataloader, val_dataloader = (out[0], out[1]) if isinstance(out, tuple) else (out, None)
        else:
            if val_split > 0:
                raise ValueError("val_split cannot be used with an already-constructed DataLoader.")
            dataloader = data
            if val_split > 0:
                if verbose:
                    print("Warning: val_split ignored when data is already a DataLoader (cannot split).")

        if dataloader.batch_size is not None and dataloader.batch_size != batch_size:
            if verbose: print(
                f"Warning: DataLoader batch_size ({dataloader.batch_size}) != requested batch_size ({batch_size}). Using DataLoader's.")
            batch_size = dataloader.batch_size

        # 2. Setup Device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device_param = device
        if verbose: print(f"Training on {device}...")

        # 3. Setup Loss
        if isinstance(loss, str):
            # Try to get from nn directly (case insensitive search)
            loss_lower = loss.lower().replace('_', '').replace('-', '')

            # Common mappings or direct lookup
            # Create a map of all available loss classes in nn
            available_losses = {n.lower(): n for n in dir(nn) if
                                isinstance(getattr(nn, n), type) and issubclass(getattr(nn, n), nn.Module)}
            # Manual aliases for common ones
            aliases = {
                'mse': 'MSELoss',
                'l2': 'MSELoss',
                'l1': 'L1Loss',
                'mae': 'L1Loss',
                'bce': 'BCELoss',
                'bcelogits': 'BCEWithLogitsLoss',
                'ce': 'CrossEntropyLoss',
                'crossentropy': 'CrossEntropyLoss',
                'nll': 'NLLLoss'
            }

            target_loss_name = aliases.get(loss_lower, None)
            if not target_loss_name:
                # Try to find match in available losses
                # We do a fuzzy-ish match: available keys are lower cased
                if loss_lower in available_losses:
                    target_loss_name = available_losses[loss_lower]
                elif loss_lower + 'loss' in available_losses:
                    target_loss_name = available_losses[loss_lower + 'loss']

            if target_loss_name and hasattr(nn, target_loss_name):
                criterion = getattr(nn, target_loss_name)()
            else:
                raise ValueError(
                    f"Unknown loss string: {loss}. Valid options include any torch.nn loss_name (e.g. 'CrossEntropyLoss', 'MSELoss') or common aliases.")
        else:
            criterion = loss

        # 4. Setup Optimizer
        if isinstance(optimizer, str):
            optimizer_lower = optimizer.lower().replace('_', '').replace('-', '')
            opt_cls = None

            # 1. Check bitsandbytes first if requested (keywords: 8bit, 4bit, bnb)
            if '8bit' in optimizer_lower or 'bnb' in optimizer_lower:
                if bitsandbytes is None:
                    print(
                        "Warning: bitsandbytes requested but not installed/importable. Fallback to standard torch optimizers.")
                else:
                    # Map common bnb names
                    # e.g. adamw8bit -> AdamW8bit
                    bnb_opts = {n.lower(): n for n in dir(bitsandbytes.optim) if 'Optimizer' in n or '8bit' in n}

                    # Try direct match
                    if optimizer_lower in bnb_opts:
                        opt_cls = getattr(bitsandbytes.optim, bnb_opts[optimizer_lower])
                    # Try stripping 'bnb' prefix if user typed 'bnbadamw'
                    elif optimizer_lower.replace('bnb', '') in bnb_opts:
                        opt_cls = getattr(bitsandbytes.optim, bnb_opts[optimizer_lower.replace('bnb', '')])

                    if opt_cls is None:
                        print(f"Warning: Could not find specific bitsandbytes optimizer '{optimizer}'.")

            # 2. Check torch.optim if not found yet
            if opt_cls is None:
                # Map torch optim names
                torch_opts = {n.lower(): n for n in dir(optim) if
                              isinstance(getattr(optim, n), type) and issubclass(getattr(optim, n), optim.Optimizer)}

                if optimizer_lower in torch_opts:
                    opt_cls = getattr(optim, torch_opts[optimizer_lower])
                elif optimizer_lower == 'sgd':  # Explicit check just in case
                    opt_cls = optim.SGD

            if opt_cls is None:
                raise ValueError(
                    f"Unknown optimizer string: {optimizer}. Supported: All torch.optim classes and bitsandbytes 8bit classes.")

            # Simple parameter grouping for weight decay
            no_bias_decay = True  # Common practice
            if weight_decay > 0:
                param_optimizer = [(n, p) for n, p in self.named_parameters() if p.requires_grad]
                no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'norm']
                optimizer_grouped_parameters = [
                    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                     'weight_decay': weight_decay},
                    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]
                optim_instance = opt_cls(optimizer_grouped_parameters, lr=learning_rate)
            else:
                params = [p for p in self.parameters() if p.requires_grad]
                if not params:
                    if verbose:
                        print("Warning: fit() called on model with no trainable parameters. Skipping optimization steps.")
                    optim_instance = None
                else:
                    optim_instance = opt_cls(params, lr=learning_rate)

        elif isinstance(optimizer, Callable) and not isinstance(optimizer, torch.optim.Optimizer):
            optim_instance = optimizer(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            optim_instance = optimizer  # Assume it's already instantiated

        # 5. Setup Scheduler
        scheduler = lr_scheduler
        if isinstance(lr_scheduler, str):
            # Simple string based setup could be added here
            pass

        # 6. Setup Scaler / Accelerate
        _accelerator = None
        if use_accelerate:
            try:
                from accelerate import Accelerator  # pyright: ignore[reportMissingImports]
                _mp = 'fp16' if (mixed_precision and device.type == 'cuda') else 'no'
                _accelerator = Accelerator(
                    mixed_precision=_mp,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                )
                self, optim_instance, dataloader = _accelerator.prepare(self, optim_instance, dataloader)
                device = _accelerator.device
                scaler = None
            except ImportError:
                if verbose:
                    print("Warning: 'accelerate' not installed. Falling back to standard training.")
                scaler = torch.cuda.amp.GradScaler() if mixed_precision and device.type == 'cuda' else None
        else:
            scaler = torch.cuda.amp.GradScaler() if mixed_precision and device.type == 'cuda' else None

        history = []
        global_step = 0
        self.train()

        if on_train_start:
            try:
                on_train_start(self, dataloader, val_dataloader)
            except Exception as e:
                if verbose:
                    print(f"on_train_start hook error: {e}")

        try:
            for epoch in range(epochs):
                if show_progress_bar:
                    pbar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
                else:
                    pbar = dataloader

                epoch_loss = 0.0
                steps = 0

                for batch in pbar:
                    # Handle batch unpacking
                    if isinstance(batch, (tuple, list)):
                        inputs, targets = batch[0], batch[1]
                    elif isinstance(batch, dict):
                        # Basic assumption: 'input_ids' or 'x' for input, 'labels' or 'y' for target
                        if 'labels' in batch:
                            targets = batch.pop('labels')
                            inputs = batch  # Pass the rest as kwargs or similar
                        elif 'y' in batch:
                            targets = batch.pop('y')
                            inputs = batch
                        else:
                            inputs = batch
                            targets = None
                    else:
                        inputs = batch
                        targets = None  # Autoencoders?

                    # Move to device
                    if isinstance(inputs, torch.Tensor):
                        inputs = inputs.to(device)
                    elif isinstance(inputs, dict):
                        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

                    if isinstance(targets, torch.Tensor):
                        targets = targets.to(device)

                    # Reset gradients (accumulation handling below)
                    # optim_instance.zero_grad() # Moved to after step

                    # Forward
                    with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=mixed_precision):
                        if isinstance(inputs, dict):
                            outputs = self(**inputs)
                        else:
                            outputs = self(inputs)

                        if targets is not None:
                            # Handle Output classes (e.g. HuggingFace models return CausalLMOutput)
                            if hasattr(outputs, 'loss') and outputs.loss is not None:
                                loss_val = outputs.loss
                            else:
                                # Standard
                                if hasattr(outputs, 'logits'): outputs = outputs.logits
                                loss_val = criterion(outputs, targets)
                        else:
                            # Unsupervised / Self-supervised (loss computed inside model)
                            if hasattr(outputs, 'loss'):
                                loss_val = outputs.loss
                            else:
                                raise ValueError("No targets provided and model did not compute loss internally.")

                        loss_val = loss_val / gradient_accumulation_steps

                    # Backward
                    if _accelerator is not None:
                        _accelerator.backward(loss_val)
                    elif scaler:
                        scaler.scale(loss_val).backward()
                    else:
                        loss_val.backward()

                    loss_value_scalar = loss_val.item() * gradient_accumulation_steps
                    
                    if math.isnan(loss_value_scalar) or math.isinf(loss_value_scalar):
                        print(f"[FAILURE-MGMT] NaN or Inf loss detected at epoch {epoch+1}, step {global_step}. Terminating.")
                        raise ValueError(f"NaN/Inf loss detected at epoch {epoch+1}, step {global_step}")

                    epoch_loss += loss_value_scalar



                    if (global_step + 1) % gradient_accumulation_steps == 0:
                        if _accelerator is not None:
                            _accelerator.clip_grad_norm_(self.parameters(), 1.0)
                            optim_instance.step()
                        elif scaler:
                            scaler.step(optim_instance)
                            scaler.update()
                        else:
                            optim_instance.step()

                        optim_instance.zero_grad()
                        if scheduler:
                            # Assume StepLR or similar step-per-batch for now, or move to epoch end
                            if hasattr(scheduler, 'step'):
                                scheduler.step()

                    global_step += 1
                    steps += 1

                    if show_progress_bar:
                        if hasattr(pbar, "set_postfix"):
                            pbar.set_postfix({'loss': loss_val.item() * gradient_accumulation_steps})

                avg_epoch_loss = epoch_loss / steps
                val_loss_epoch = None
                if val_dataloader is not None:
                    self.eval()
                    val_loss_sum = 0.0
                    val_steps = 0
                    with torch.no_grad():
                        for vbatch in val_dataloader:
                            if isinstance(vbatch, (tuple, list)):
                                vinputs, vtargets = vbatch[0], vbatch[1]
                            elif isinstance(vbatch, dict):
                                vtargets = vbatch.get('labels') or vbatch.get('y')
                                vinputs = {k: v for k, v in vbatch.items() if k not in ('labels', 'y')} if vtargets is not None else vbatch
                                if isinstance(vinputs, dict) and not vinputs and vtargets is not None:
                                    vinputs = vbatch
                            else:
                                vinputs, vtargets = vbatch, None
                            if isinstance(vinputs, torch.Tensor):
                                vinputs = vinputs.to(device)
                            elif isinstance(vinputs, dict):
                                vinputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in vinputs.items()}
                            if isinstance(vtargets, torch.Tensor):
                                vtargets = vtargets.to(device)
                            vout = self(**vinputs) if isinstance(vinputs, dict) else self(vinputs)
                            if hasattr(vout, 'loss') and vout.loss is not None:
                                vloss = vout.loss
                            elif hasattr(vout, 'logits'):
                                vloss = criterion(vout.logits, vtargets)
                            else:
                                vloss = criterion(vout, vtargets)
                            val_loss_sum += vloss.item()
                            val_steps += 1
                    val_loss_epoch = val_loss_sum / max(1, val_steps)
                    self.train()
                    if verbose:
                        print(f"Epoch {epoch + 1} Loss: {avg_epoch_loss:.4f} Val Loss: {val_loss_epoch:.4f}")
                else:
                    if verbose:
                        print(f"Epoch {epoch + 1} Loss: {avg_epoch_loss:.4f}")
                rec = {'epoch': epoch + 1, 'loss': avg_epoch_loss}
                if val_loss_epoch is not None:
                    rec['val_loss'] = val_loss_epoch
                history.append(rec)

                if on_epoch_end:
                    try:
                        on_epoch_end(self, epoch + 1, rec, history)
                    except Exception as e:
                        if verbose:
                            print(f"on_epoch_end hook error: {e}")

                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    ckpt_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.{save_type}")
                    self.save_model(ckpt_path, save_type=save_type)

        except Exception as e:
            if on_error:
                try:
                    on_error(self, e)
                except Exception:
                    pass
            if save_dir:
                try:
                    os.makedirs(save_dir, exist_ok=True)
                    self.save_model(os.path.join(save_dir, "error_recovery.pt"), save_type=save_type)
                except Exception:
                    pass
            raise

        # Clean up accelerate hooks to allow pickling after fit
        if _accelerator is not None:
            try:
                from accelerate.hooks import remove_hook_from_module  # pyright: ignore
                remove_hook_from_module(self, recurse=True)
            except Exception:
                pass

        if on_train_end:
            try:
                on_train_end(self, history)
            except Exception as e:
                if verbose:
                    print(f"on_train_end hook error: {e}")

        return pd.DataFrame(history)

    @classmethod
    def from_pretrained(cls, model_name: str, model_path: str = None, device_map: Union[str, dict] = 'auto',
                        dtype: torch.dtype = torch.float32, quantization_config=None, low_cpu_mem_usage: bool = True,
                        trust_remote_code: bool = True, revision: str = 'main', variant: str = None,
                        token: Union[str, bool] = None,
                        cache_dir: str = None, force_download: bool = False, output_hidden_states: bool = False,
                        **kwargs):
        """
        Load model from pretrained weights (Local or Hub).
        Wrapper around transformers.AutoModel or manual torch.load handling.
        """
        # 1. Try Loading via Transformers (if it looks like a HF model)
        try:
            from transformers import AutoModel, AutoConfig, BitsAndBytesConfig  # pyright: ignore[reportMissingImports]

            # Handle Quantization config from kwargs or bitsandbytes imports
            bnb_config = None
            if quantization_config:
                bnb_config = quantization_config

            # Setup loading args
            load_args = {
                'trust_remote_code': trust_remote_code,
                'revision': revision,
                'cache_dir': cache_dir,
                'force_download': force_download,
                'output_hidden_states': output_hidden_states,
            }
            if token: load_args['token'] = token
            if device_map: load_args['device_map'] = device_map
            if bnb_config: load_args['quantization_config'] = bnb_config
            if low_cpu_mem_usage: load_args['low_cpu_mem_usage'] = True

            if model_path and os.path.exists(model_path):
                # Load local
                model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, **load_args, **kwargs)
            else:
                # Load from Hub
                model = AutoModel.from_pretrained(model_name, torch_dtype=dtype, **load_args, **kwargs)

            return model  # This returns the HF model, which might not be an instance of DLModule class if DLModule wraps it.
            # Ideally, DLModule should wrap the HF model.

        except ImportError:
            print("Transformers not installed or failed. Falling back to torch.load.")
            AutoModel = None
        except Exception as e:
            print(f"Transformers loading failed: {e}. Falling back to standard torch.load.")

        # 2. Standard Torch Load
        # Create instance
        instance = cls(**kwargs)
        path = model_path if model_path else model_name

        if os.path.exists(path):
            if os.path.isdir(path):
                # Try to find common model filenames
                for ext in ['pt', 'pth', 'safetensors']:
                    potential_path = os.path.join(path, f"model.{ext}")
                    if os.path.exists(potential_path):
                        path = potential_path
                        break

            # If path is still a directory after the loop, torch.load will fail
            # So we check again
            if os.path.isdir(path):
                raise IsADirectoryError(f"Found directory at {path}, but no 'model.pt/pth/safetensors' file inside.")

            if path.endswith('.safetensors'):
                try:
                    from safetensors.torch import load_file  # pyright: ignore[reportMissingImports]
                    state_dict = load_file(path)
                except ImportError:
                    raise ImportError("safetensors not installed.")
            else:
                state_dict = torch.load(path, map_location='cpu')

            instance.load_state_dict(state_dict)
            return instance
        else:
            raise FileNotFoundError(f"Could not find model at {path}")

    def save_pretrained(self, save_dir: str, save_type: str = 'pt', safe_serialization: bool = True,
                        is_main_process: bool = True,
                        max_shard_size: Union[str, int] = '10GB', push_to_hub: bool = False, token: str = None,
                        state_dict: Optional[dict] = None, save_config: bool = True):

        if not is_main_process: return
        os.makedirs(save_dir, exist_ok=True)

        # Try saving via transformers if applicable
        if hasattr(self, 'save_pretrained') and callable(getattr(super(DLModule, self), 'save_pretrained', None)):
            # This handles the case where DLModule inherits from a Transformers model
            super().save_pretrained(save_dir, safe_serialization=safe_serialization, max_shard_size=max_shard_size,
                                    push_to_hub=push_to_hub, token=token)
            return

        # Manual Save
        if state_dict is None:
            state_dict = self.state_dict()

        torch.save(state_dict, os.path.join(save_dir, f"model.{save_type}"))

        if save_config and hasattr(self, 'config'):
            with open(os.path.join(save_dir, "config.json"), 'w') as f:
                # Attempt to serialize config
                try:
                    json.dump(self.config, f, indent=4)
                except:
                    f.write(str(self.config))  # Fallback

    def save_model(self, save_path: str, save_type: str = 'pt', *args, **kwargs):
        """
        Save the full model object so it can be restored via load_model.
        Falls back to dill when standard pickle cannot handle all attributes.
        """
        if save_type == 'safetensors':
            try:
                from safetensors.torch import save_file  # pyright: ignore[reportMissingImports]
                save_file(self.state_dict(), save_path)
            except ImportError:
                print("Safetensors not installed. Saving as .pt")
                torch.save(self, save_path.replace('.safetensors', '.pt'))
            return
        try:
            torch.save(self, save_path)
        except Exception:
            try:
                import dill as _dill  # pyright: ignore[reportMissingImports]
                with open(save_path, 'wb') as _fh:
                    _dill.dump(self, _fh)
            except ImportError:
                raise RuntimeError(
                    "Model contains unpicklable attributes and 'dill' is not "
                    "installed.  Run: pip install dill"
                )
            except Exception as _dill_err:
                try:
                    import dill as _dill2  # pyright: ignore[reportMissingImports]
                    # Recursively remove accelerate hooks from all sub-modules
                    for _m in self.modules():
                        for _a in ['_hf_hook', '_hf_prehook', '_old_forward', '_accelerate_hook']:
                            try:
                                if hasattr(_m, _a):
                                    delattr(_m, _a)
                            except Exception:
                                pass
                    with open(save_path, 'wb') as _fh2:
                        _dill2.dump(self, _fh2)
                except Exception as _e2:
                    raise RuntimeError(
                        f"Cannot save model even after hook cleanup: {_e2}"
                    ) from _dill_err

    @classmethod
    def load_model(cls, load_path: str, *args, **kwargs):
        """
        Class-level load wrapper. Returns a fully restored model instance.
        Handles both full-model checkpoints and plain state-dict checkpoints.
        Uses dill as fallback for checkpoints saved with dill.
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model not found at {load_path}")

        if load_path.endswith('.safetensors'):
            try:
                from safetensors.torch import load_file  # pyright: ignore[reportMissingImports]
                state_dict = load_file(load_path)
                instance = cls()
                instance.load_state_dict(state_dict)
                return instance
            except ImportError:
                raise ImportError("safetensors not installed.")

        try:
            obj = torch.load(load_path, map_location='cpu', weights_only=False)
            if isinstance(obj, cls):
                # Restore to the device the model was configured for
                model_dev = getattr(obj, 'device', 'cpu') or 'cpu'
                if str(model_dev) != 'cpu' and torch.cuda.is_available():
                    try:
                        obj = obj.to(model_dev)
                        _move_nested_estimators_to_device(obj, model_dev)
                    except Exception:
                        pass
                return obj
            instance = cls()
            if hasattr(obj, 'state_dict'):
                obj = obj.state_dict()
            instance.load_state_dict(obj)
            return instance
        except Exception:
            pass
        try:
            import dill as _dill  # pyright: ignore[reportMissingImports]
            with open(load_path, 'rb') as _fh:
                obj = _dill.load(_fh)
            if isinstance(obj, torch.nn.Module):
                model_dev = getattr(obj, 'device', 'cpu') or 'cpu'
                if str(model_dev) != 'cpu' and torch.cuda.is_available():
                    try:
                        obj = obj.to(model_dev)
                        _move_nested_estimators_to_device(obj, model_dev)
                    except Exception:
                        pass
            return obj
        except ImportError:
            raise RuntimeError(
                "Could not load checkpoint with torch.load; 'dill' not installed."
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load checkpoint from {load_path}: {exc}")

    def fine_tune(self,
                  data: Union[torch.Tensor, np.ndarray, pd.Series, pd.DataFrame, list, dict, tuple],
                  fine_tune_type: str = 'lora',  # 'weight-decay', 'lora', 'q-lora', 'dora', 'q-dora'
                  epochs: int = 1, batch_size: int = 32, learning_rate: float = 1e-4,
                  **kwargs):
        """
        Fine-tuning wrapper. Works for both ML models (refit on data) and DL models (gradient loop).
        Unpacks (X, y) for ML models; safely skips PEFT for incompatible models.
        """
        fit_kwargs = dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, **kwargs)

        # 1. Unpack data for ML models expecting fit(X, y) or fit(X)
        fit_data = data
        fit_y = None
        if isinstance(data, (tuple, list)) and len(data) == 2:
            fit_data, fit_y = data[0], data[1]
            if not isinstance(fit_data, torch.Tensor):
                fit_data = torch.as_tensor(np.asarray(fit_data), dtype=torch.float32)
            if fit_y is not None and not isinstance(fit_y, torch.Tensor):
                fit_y = torch.as_tensor(np.asarray(fit_y))

        # 2. PEFT only for compatible models (skip silently on failure)
        if fine_tune_type.lower() in ['lora', 'q-lora', 'dora'] and list(self.parameters()):
            try:
                from peft import get_peft_model, LoraConfig  # pyright: ignore[reportMissingImports]
                peft_config = LoraConfig(
                    inference_mode=False,
                    r=8,
                    lora_alpha=32,
                    lora_dropout=0.1
                )
                self = get_peft_model(self, peft_config)
                if hasattr(self, 'print_trainable_parameters'):
                    self.print_trainable_parameters()
            except Exception:
                pass

        # 3. Call fit with correct signature — wrap (X, y) in DataLoader for DLModule compatibility
        if fit_y is not None:
            from ...train.utils.data_loader import make_loader
            bs = fit_kwargs.pop('batch_size', 32)
            fit_data = make_loader(fit_data, targets=fit_y, batch_size=bs)
        return self.fit(fit_data, **fit_kwargs)

    def register_forward_hook(
            self,
            hook: Union[
                Callable[[T, Tuple[Any, ...], Any], Optional[Any]],
                Callable[[T, Tuple[Any, ...], Dict[str, Any], Any], Optional[Any]],
            ],
            *,
            prepend: bool = False,
            with_kwargs: bool = False,
            always_call: bool = False,
    ) -> RemovableHandle:
        return super().register_forward_hook(hook, prepend=prepend, with_kwargs=with_kwargs, always_call=always_call)

    def register_backward_hook(
            self, hook: Callable[["nn.Module", _grad_t, _grad_t], Union[None, _grad_t]]
    ) -> RemovableHandle:
        return super().register_backward_hook(hook)

    def quantize(self, mode: str = 'int8'):
        """
        Quantizes the model weights for reduced memory usage.
        Args:
            mode (str): 'int8' or 'float16'.
        """
        if getattr(self, '_is_quantized', False):
            print("Model is already quantized.")
            return

        self._quant_info = {}
        print(f"Quantizing model weights to {mode}...")

        for name, param in self.named_parameters():
            if not param.requires_grad and 'bias' in name:
                # Often biases are small and sensitive, maybe skip?
                # But for memory reduction, we want everything.
                pass

            data = param.data
            original_dtype = data.dtype

            if mode == 'float16':
                if original_dtype == torch.float32:
                    param.data = data.half()
                    self._quant_info[name] = {'mode': 'float16', 'orig_dtype': original_dtype}

            elif mode == 'int8':
                # Determine scale
                # symmetric quantization
                abs_max = data.abs().max()
                scale = (abs_max / 127.0).to(data.device)
                if scale == 0: scale = torch.tensor(1.0, device=data.device)

                # Quantize
                quantized = (data / scale).round().clamp(-127, 127)
                param.requires_grad = False
                param.data = (quantized * scale).to(original_dtype).to(data.device)
                self._quant_info[name] = {
                    'mode': 'int8',
                    'scale': scale,
                    'orig_dtype': original_dtype
                }

        self._is_quantized = True
        print("Quantization complete.")

    def dequantize(self):
        """
        Dequantizes the model weights back to original precision.
        """
        if not getattr(self, '_is_quantized', False):
            print("Model is not quantized.")
            return

        print("Dequantizing model weights...")

        for name, param in self.named_parameters():
            if name in self._quant_info:
                info = self._quant_info[name]
                mode = info['mode']
                orig_dtype = info['orig_dtype']

                if mode == 'float16':
                    param.data = param.data.to(orig_dtype)

                elif mode == 'int8':
                    scale = info['scale']
                    if isinstance(scale, torch.Tensor):
                        scale = scale.to(param.device)

                    param.data = param.data.to(orig_dtype) * scale

        self._is_quantized = False
        self._quant_info = {}
        print("Dequantization complete.")

    def _resolve_func(self, func, *args, **kwargs):
        if func is None:
            return None
        from ...models.deep_learning.activations.Complex.complex_activations import ComplexActivation, \
            ComplexCustomStringActivationLayer
        if isinstance(func, Callable):
            return ActFuncWrapper(func, *args, **kwargs)
        if isinstance(func, Union[nn.Module, DLModule]):
            return func
        if isinstance(func, str):
            func = func.lower()
            if func == 'sigmoid':
                return self._resolve_func(torch.sigmoid, *args, **kwargs)
            if func == 'tanh':
                return self._resolve_func(torch.tanh, *args, **kwargs)
            if func == 'relu':
                return self._resolve_func(torch.relu, *args, **kwargs)
            if func == 'softplus':
                return self._resolve_func(F.softplus, *args, **kwargs)
            if func == 'elu':
                return self._resolve_func(F.elu, *args, **kwargs)
            act_type = kwargs.get("act_type", None)
            if act_type is not None and act_type == "complex":
                try:
                    return ComplexActivation(func, *args, **kwargs)
                except Exception:
                    try:
                        return ComplexCustomStringActivationLayer(func, *args, **kwargs)
                    except Exception:
                        return None
            from ...models.deep_learning.activations.ActivationFunction import Activation
            return Activation(func, *args, **kwargs)

    def _resolve_funcs(self, funcs, *args, **kwargs):
        if isinstance(funcs, Union[str, Callable, nn.Module, DLModule]):
            return self._resolve_func(funcs, *args, **kwargs)
        if isinstance(funcs, Union[list, tuple]):
            out = nn.ModuleList([])
            for func in funcs:
                out.append(self._resolve_func(func, *args, **kwargs))
            return out
        if isinstance(funcs, dict):
            funcs = list(funcs.values())
            out = nn.ModuleList([])
            for func in funcs:
                out.append(self._resolve_func(func, *args, **kwargs))
            return out


def _to_numpy(X) -> np.ndarray:
    """Convert *X* (Tensor, DataFrame, or array-like) to a numpy ndarray."""
    if isinstance(X, torch.Tensor):
        return X.detach().cpu().numpy()
    if hasattr(X, 'values'):
        return X.values
    return np.asarray(X)


class MLModule(DLModule):
    """
    Parent class for Torch-based Machine Learning algorithms (Regression, Classification, Clustering).
    Extends DLModule with Scikit-Learn like interface methods.
    Strictly PyTorch implementations.

    Additional features over DLModule
    ----------------------------------
    * ``fit_status`` — boolean flag that subclasses set to ``True`` once
      :meth:`fit` completes.  Setting it to ``True`` automatically triggers
      :meth:`_register_fitted_state`, which registers every tensor / numpy
      array instance attribute as a PyTorch named buffer so that the full
      fitted state round-trips through :meth:`state_dict` /
      :meth:`load_state_dict`.
    * ``force`` — flag consumed by :meth:`_reset_params`.  When ``False``
      (default) the method is a no-op; when ``True`` it clears all
      dynamically-registered buffers, nullifies sklearn-style fitted
      attributes, and resets ``fit_status``.
    * :meth:`get_extra_state` / :meth:`set_extra_state` — PyTorch hooks that
      serialise non-tensor fitted attributes (scalars, lists, etc.) alongside
      the normal state dict so they survive a checkpoint round-trip.
    """

    def __init_subclass__(cls, **kwargs):
        """Wrap ``fit`` defined in any subclass to automatically set
        ``fit_status = True`` after the call completes successfully.
        This makes ``fit_status`` universal without requiring every
        concrete model to set it manually.
        """
        super().__init_subclass__(**kwargs)
        if 'fit' in cls.__dict__:
            _original_fit = cls.__dict__['fit']

            @functools.wraps(_original_fit)
            def _auto_fit(self, *args, **kw):
                result = _original_fit(self, *args, **kw)
                if not self.fit_status:
                    self.fit_status = True
                return result

            cls.fit = _auto_fit

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Estimator type: 'regressor', 'classifier', 'cluster' or 'transformer'
        # Subclasses should set this in their __init__
        self.estimator_type = None
        self.fit_status = False
        # Whether _reset_params() performs an actual reset (False = no-op)
        self.force = False
        # Internal bookkeeping — use object.__setattr__ to bypass nn.Module
        # so these never appear as parameters / buffers / modules.
        object.__setattr__(self, '_dynamic_buffers', set())
        object.__setattr__(self, '_skip_set_attr_hook', False)

    def to(self, *args, **kwargs):
        """
        Extends nn.Module.to to also update the 'device' and 'dtype' attributes
        if they exist on the instance. This ensures compatibility with 
        subclasses that use these as string/dtype fields in their own methods.
        """
        # Call super().to() first to handle parameters/buffers
        result = super().to(*args, **kwargs)
        
        # Use torch._C._nn._parse_to to get the target device/dtype from args
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        
        if device is not None:
            if hasattr(self, 'device'):
                object.__setattr__(self, 'device', str(device))
            if hasattr(self, '_device'): # Support for DLModule's _device
                object.__setattr__(self, '_device', device if isinstance(device, torch.device) else torch.device(device))
        
        if dtype is not None:
            if hasattr(self, 'dtype'):
                object.__setattr__(self, 'dtype', dtype)
            if hasattr(self, '_dtype'):
                object.__setattr__(self, '_dtype', dtype)
                
        return result

    # ------------------------------------------------------------------
    # Custom _apply() — also moves plain tensor attributes (e.g. _classes)
    # nn.Module.to() uses _apply() internally, so overriding _apply ensures
    # that all tensor attributes (not just Parameters/Buffers) are moved
    # when the module or any sub-module is moved to a new device/dtype.
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_to_nested(obj, fn):
        """Recursively apply fn to every tensor inside nested containers."""
        if isinstance(obj, torch.Tensor):
            try:
                return fn(obj)
            except Exception:
                return obj
        elif isinstance(obj, dict):
            return {k: MLModule._apply_to_nested(v, fn) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [MLModule._apply_to_nested(v, fn) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(MLModule._apply_to_nested(v, fn) for v in obj)
        return obj

    def _apply(self, fn, recurse=True):
        """Extend nn.Module._apply to also transform plain tensor attributes,
        including tensors nested inside lists, dicts, and tuples."""
        result = super()._apply(fn, recurse=recurse)
        for attr_name in list(self.__dict__.keys()):
            val = self.__dict__.get(attr_name)
            if val is None:
                continue
            if isinstance(val, torch.Tensor):
                try:
                    object.__setattr__(self, attr_name, fn(val))
                except Exception:
                    pass
            elif isinstance(val, (list, dict, tuple)):
                try:
                    new_val = MLModule._apply_to_nested(val, fn)
                    object.__setattr__(self, attr_name, new_val)
                except Exception:
                    pass
        return result

    # ------------------------------------------------------------------
    # Attribute hook — trigger state registration when fit completes
    # ------------------------------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        # When fit_status transitions to True, auto-register fitted state.
        if name == 'fit_status' and value is True:
            # Guard against re-entrant calls (e.g. from within _register_fitted_state).
            guard = self.__dict__.get('_skip_set_attr_hook', False)
            if not guard:
                try:
                    object.__setattr__(self, '_skip_set_attr_hook', True)
                    self._register_fitted_state()
                except Exception:
                    pass
                finally:
                    try:
                        object.__setattr__(self, '_skip_set_attr_hook', False)
                    except Exception:
                        pass

    @staticmethod
    def _remove_accelerate_hooks(module: 'torch.nn.Module') -> None:
        """Recursively remove accelerate hooks from a module and all sub-modules."""
        for m in module.modules():
            try:
                if hasattr(m, '_hf_hook'):
                    object.__setattr__(m, '_hf_hook', None)
                    try:
                        del m._hf_hook
                    except Exception:
                        pass
            except Exception:
                pass
            # Remove any other accelerate-injected attributes
            for attr in ['_hf_prehook', '_old_forward', '_accelerate_hook']:
                try:
                    if hasattr(m, attr):
                        delattr(m, attr)
                except Exception:
                    pass

    def __getstate__(self):
        """Return picklable state, stripping accelerate hooks and other unpicklable attrs."""
        # Remove accelerate hooks from all sub-modules in-place
        try:
            MLModule._remove_accelerate_hooks(self)
        except Exception:
            pass
        state = self.__dict__.copy()
        state.pop('_hf_hook', None)
        state.pop('_hf_prehook', None)
        state.pop('_old_forward', None)
        # Scan for any remaining unpicklable attributes and set them to None.
        # This handles torch._dynamo.ConfigModuleInstance in optimizer closures, etc.
        import pickle as _pickle
        for _k in list(state.keys()):
            try:
                _pickle.dumps(state[_k])
            except Exception:
                try:
                    import dill as _dill
                    _dill.dumps(state[_k])
                except Exception:
                    state[_k] = None
        return state

    def fit(self, data_or_X, y=None, **kwargs):
        """
        Flexible fit method supporting both DataLoader/Dataset (DL style) and X, y (Sklearn style).
        Args:
            data_or_X: DataLoader, Dataset, or Feature Matrix X (Tensor).
            y: Target Vector y (Tensor), optional.
            **kwargs: Arguments passed to DLModule.fit (epochs, batch_size, etc.).
        """
        # Case 1: Sklearn style fit(X, y)
        if y is not None:
            X = data_or_X
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float32)
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y, dtype=torch.float32)
            X = X.float()

            if not list(self.parameters()) and hasattr(self, '_init_module_'):
                try:
                    self._init_module_(X, y)
                except TypeError:
                    try:
                        self._init_module_(X)
                    except TypeError:
                        in_features = X.shape[1]
                        out_features = y.shape[1] if y.ndim > 1 else 1
                        self._init_module_(in_features, out_features)

            # Wrap in TensorDataset
            from torch.utils.data import TensorDataset
            data = TensorDataset(X, y)

        # Case 2: Unsupervised Sklearn style fit(X) or just X passed as data
        elif isinstance(data_or_X, torch.Tensor):
            X = data_or_X
            X = X.float()

            if not list(self.parameters()) and hasattr(self, '_init_module_'):
                try:
                    self._init_module_(X)
                except TypeError:
                    in_features = X.shape[1]
                    self._init_module_(in_features, 1)

            # Wrap in TensorDataset (unsupervised)
            from torch.utils.data import TensorDataset
            data = TensorDataset(X)

        # Case 3: DataLoader or Dataset
        else:
            data = data_or_X

        return super().fit(data, **kwargs)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predicts output for input X.
        Delegates to the forward pass which handles device/dtype conversion.
        """
        return self(X)

    # ------------------------------------------------------------------
    # vmap / accelerate utilities — inherited by all MLModule subclasses
    # ------------------------------------------------------------------

    def vmap_predict(self, X: torch.Tensor) -> torch.Tensor:
        """Vectorized batch prediction using torch.func.vmap.

        For models whose ``forward`` handles single samples, this provides an
        efficient way to predict over a full batch via vmap.  For models that
        already process full batches in ``forward``, this simply falls back to
        the standard ``predict`` path.
        """
        self.eval()
        device = getattr(self, 'device', 'cpu') or 'cpu'
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(device)
        with torch.no_grad():
            try:
                from torch.func import vmap  # pyright: ignore[reportMissingImports]
                def _single_forward(x):
                    out = self(x.unsqueeze(0))
                    if isinstance(out, torch.Tensor):
                        return out.squeeze(0)
                    return out
                return vmap(_single_forward)(X)
            except Exception:
                return self.predict(X)

    def _parallel_predict(self, X_list: list, n_jobs: int = -1) -> list:
        """Predict in parallel over a list of input tensors using joblib."""
        try:
            import joblib  # pyright: ignore[reportMissingImports]
            return joblib.Parallel(n_jobs=n_jobs, prefer='threads')(
                joblib.delayed(self.predict)(x) for x in X_list
            )
        except ImportError:
            return [self.predict(x) for x in X_list]

    def _vmap_multi_target_predict(self, X: torch.Tensor, models: list) -> torch.Tensor:
        """Predict from multiple fitted sub-models and stack results.

        Parameters
        ----------
        X : torch.Tensor  shape (n_samples, n_features)
        models : list of callable  — one per output target

        Returns
        -------
        predictions : torch.Tensor  shape (n_samples, n_targets)
        """
        try:
            import joblib  # pyright: ignore[reportMissingImports]
            preds = joblib.Parallel(n_jobs=-1, prefer='threads')(
                joblib.delayed(m.predict)(X) for m in models
            )
        except ImportError:
            preds = [m.predict(X) for m in models]
        return torch.stack([p.flatten() for p in preds], dim=1)

    def _vmap_compute_grads(self, loss_fn: Callable, params: torch.Tensor, X: torch.Tensor,
                             y: torch.Tensor) -> torch.Tensor:
        """Compute per-sample gradients via torch.func.vmap + grad.

        Parameters
        ----------
        loss_fn : callable (params, x, y_i) -> scalar loss
        params  : 1-D parameter tensor
        X, y    : batch tensors

        Returns
        -------
        grads : mean gradient tensor, same shape as *params*
        """
        try:
            from torch.func import vmap, grad  # pyright: ignore[reportMissingImports]
            per_sample_grad = vmap(grad(loss_fn), in_dims=(None, 0, 0))
            grads = per_sample_grad(params, X, y)
            return grads.mean(dim=0)
        except Exception:
            params_req = params.detach().requires_grad_(True)
            loss = loss_fn(params_req, X, y)
            loss.backward()
            return params_req.grad if params_req.grad is not None else torch.zeros_like(params)

    def prediction_loss(self, X: torch.Tensor, y: torch.Tensor = None, criterion: str = "mse",
                        **kwargs) -> torch.Tensor:
        match criterion.lower():
            case "mse":
                loss = nn.MSELoss(**kwargs)
            case "mae":
                loss = nn.L1Loss(**kwargs)
            case "huber" | "huber_loss":
                loss = nn.HuberLoss(**kwargs)
            case "smooth_mae":
                loss = nn.SmoothL1Loss(**kwargs)
            case "gaussian_nl_loss":
                loss = nn.GaussianNLLLoss(**kwargs)
            case "poisson_nl_loss":
                loss = nn.PoissonNLLLoss(**kwargs)
            case "kl_div_loss":
                loss = nn.KLDivLoss(**kwargs)
            case "cosine_loss":
                loss = nn.CosineEmbeddingLoss(**kwargs)
            case "pairwise_distance":
                loss = nn.PairwiseDistance(**kwargs)
            case "triple_margin_loss":
                loss = nn.TripletMarginLoss(**kwargs)
            case _:
                loss = nn.MSELoss(**kwargs)
        predicts = self.predict(X)
        return loss(predicts, y)

    def score(self, X, y, sample_weight=None):
        """
        Returns the score of the prediction.
        - Regressor: R² score (coefficient of determination)
        - Classifier: Accuracy score
        - Clusterer: Not yet implemented
        
        Parameters
        ----------
        X : torch.Tensor
            Test samples
        y : torch.Tensor
            True values for X
        sample_weight : torch.Tensor, optional
            Sample weights
            
        Returns
        -------
        score : float
            Score value
        """
        if self.estimator_type is None:
            raise ValueError("estimator_type must be set by subclass ('regressor', 'classifier', or 'clusterer')")

        # Ensure tensors are on same device
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y)

        device = next(self.parameters(), torch.tensor(0.0)).device if list(self.parameters()) else 'cpu'
        X = X.to(device)
        y = y.to(device)
        if sample_weight is not None and not isinstance(sample_weight, torch.Tensor):
            sample_weight = torch.tensor(sample_weight).to(device)

        if self.estimator_type == 'regressor':
            # R² = 1 - (SS_res / SS_tot)
            y_pred = self.predict(X)

            # Ensure same shape
            if y_pred.ndim != y.ndim:
                if y.ndim == 1 and y_pred.ndim == 2 and y_pred.shape[1] == 1:
                    y_pred = y_pred.flatten()
                elif y_pred.ndim == 1 and y.ndim == 2 and y.shape[1] == 1:
                    y = y.flatten()

            if sample_weight is not None:
                ss_res = torch.sum(sample_weight * (y - y_pred) ** 2)
                ss_tot = torch.sum(sample_weight * (y - y.mean()) ** 2)
            else:
                ss_res = torch.sum((y - y_pred) ** 2)
                ss_tot = torch.sum((y - y.mean()) ** 2)

            # Avoid division by zero
            if ss_tot == 0:
                return 1.0 if ss_res == 0 else 0.0

            r2 = 1 - (ss_res / ss_tot)
            return r2.item() if isinstance(r2, torch.Tensor) else r2

        elif self.estimator_type == 'classifier':
            # Accuracy = correct predictions / total predictions
            y_pred = self.predict(X)

            # Handle multi-class: get argmax if predictions are probabilities
            if y_pred.ndim > 1 and y_pred.shape[1] > 1:
                y_pred = torch.argmax(y_pred, dim=1)

            correct = (y_pred == y).float()

            if sample_weight is not None:
                accuracy = torch.sum(sample_weight * correct) / torch.sum(sample_weight)
            else:
                accuracy = torch.mean(correct)

            return accuracy.item() if isinstance(accuracy, torch.Tensor) else accuracy

        elif self.estimator_type == 'clusterer':
            raise NotImplementedError(
                "Clusterer score not yet implemented. Consider using silhouette_score or similar metrics.")
        else:
            raise ValueError(
                f"Unknown estimator_type: {self.estimator_type}. Must be 'regressor', 'classifier', or 'clusterer'.")

    def save_pretrained(self, save_directory: str, save_type: str = 'pt', push_to_hub: bool = False, **kwargs):
        """
        Save the ML model to a directory.
        Overrides DLModule.save_pretrained to handle sklearn-like state (non-parameter attributes).
        Supports: 'pt', 'pth', 'onnx', 'safetensors' (partial).
        """
        import os
        os.makedirs(save_directory, exist_ok=True)

        # Determine strict save type
        if save_type == 'pytorch': save_type = 'pt'

        save_path = os.path.join(save_directory, f"model.{save_type}")

        if save_type in ['pt', 'pth']:
            # Full object serialization (pickle)
            # Necessary for sklearn-like estimators with dynamic attributes
            torch.save(self, save_path)

            # Also save config if present
            if hasattr(self, 'config'):
                import json
                try:
                    with open(os.path.join(save_directory, "config.json"), 'w') as f:
                        json.dump(self.config, f, indent=4)
                except Exception as e:
                    print(f"Warning: Could not save config.json: {e}")

        elif save_type == 'onnx':
            # Expert ONNX export
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = 'cpu'

            dummy_input = torch.randn(1, self.in_features or 1, device=device)
            torch.onnx.export(self, (*dummy_input.tolist(),), save_path,
                              input_names=['input'], output_names=['output'],
                              dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

        elif save_type == 'safetensors':
            # Warn about state loss
            print(
                "Warning: Saving MLModule as safetensors may lose non-parameter state (e.g. tree structures, fitted attributes).")
            # Fallback to state dict saving
            super().save_pretrained(save_directory, save_type=save_type, **kwargs)

        else:
            raise ValueError(f"Unsupported save_type: {save_type}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        """
        Load an ML model from a directory or file.
        Overrides DLModule.from_pretrained to support pickled objects.
        """
        import os

        path = pretrained_model_name_or_path

        # If directory, find model file
        if os.path.isdir(path):
            files = os.listdir(path)
            if "model.pt" in files:
                path = os.path.join(path, "model.pt")
            elif "model.pth" in files:
                path = os.path.join(path, "model.pth")
            elif "model.safetensors" in files:
                path = os.path.join(path, "model.safetensors")
            elif "model.onnx" in files:
                raise NotImplementedError("Loading from ONNX not supported directly.")
            else:
                raise FileNotFoundError(f"No model file found in {path}")

        # Load
        if path.endswith('.pt') or path.endswith('.pth'):
            try:
                # Load full object
                obj = torch.load(path, weights_only=False)
                return obj
            except Exception as e:
                print(f"Failed to load full object: {e}. Trying state_dict...")
                # Fallback to instantiation + load_state_dict if it was just state dict
                model = cls(*args, **kwargs)
                state_dict = torch.load(path, map_location='cpu')
                model.load_state_dict(state_dict)
                return model

        elif path.endswith('.safetensors'):
            # Create new instance and load weights
            model = cls(*args, **kwargs)
            from safetensors.torch import load_file  # pyright: ignore[reportMissingImports]
            state_dict = load_file(path)
            model.load_state_dict(state_dict)
            return model

        else:
            return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    def save_model(self, save_path: str, save_type: str = 'pt', *args, **kwargs):
        """
        Save the full model object directly to save_path so that load_model
        can restore it from the same path without directory indirection.
        Falls back to save_pretrained when save_type is 'safetensors'.
        Uses dill when standard pickle cannot serialise all attributes
        (e.g. local lambda functions stored on the model).
        """
        if save_type == 'safetensors':
            directory = os.path.dirname(save_path) or '.'
            filename = os.path.basename(save_path)
            if '.' in filename:
                save_type = filename.split('.')[-1]
            self.save_pretrained(directory, save_type=save_type)
            return
        try:
            torch.save(self, save_path)
        except Exception:
            # Fall back to dill for models that contain unpicklable objects
            # (e.g. locally-defined lambda functions, accelerate hooks).
            try:
                import dill as _dill  # pyright: ignore[reportMissingImports]
                with open(save_path, 'wb') as _fh:
                    _dill.dump(self, _fh)
            except ImportError:
                raise RuntimeError(
                    "Model contains unpicklable attributes and 'dill' is not "
                    "installed.  Run: pip install dill"
                )
            except Exception as _dill_err:
                # Last resort: strip accelerate hooks recursively and retry with dill
                try:
                    import dill as _dill2  # pyright: ignore[reportMissingImports]
                    MLModule._remove_accelerate_hooks(self)
                    with open(save_path, 'wb') as _fh2:
                        _dill2.dump(self, _fh2)
                except Exception as _e2:
                    raise RuntimeError(
                        f"Cannot save model even after hook cleanup: {_e2}"
                    ) from _dill_err

    @classmethod
    def load_model(cls, load_path: str, *args, **kwargs):
        """
        Class-level load wrapper. Returns a fully restored model instance.
        Handles both direct file paths and directory paths.
        Uses dill when the checkpoint was saved with dill (lambda-containing
        models).
        """
        if os.path.isdir(load_path):
            return cls.from_pretrained(load_path, *args, **kwargs)
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model not found at {load_path}")
        # Try torch.load first (fast path for standard checkpoints)
        try:
            obj = torch.load(load_path, map_location='cpu', weights_only=False)
            if isinstance(obj, cls):
                # Restore to the device the model was configured for
                model_dev = getattr(obj, 'device', 'cpu') or 'cpu'
                if str(model_dev) != 'cpu' and torch.cuda.is_available():
                    try:
                        obj = obj.to(model_dev)
                        _move_nested_estimators_to_device(obj, model_dev)
                    except Exception:
                        pass
                return obj
            # Saved as state_dict — create fresh instance and restore
            instance = cls(*args, **kwargs)
            if hasattr(obj, 'state_dict'):
                obj = obj.state_dict()
            instance.load_state_dict(obj)
            return instance
        except Exception:
            pass
        # Fall back to dill (for checkpoints saved with dill)
        try:
            import dill as _dill  # pyright: ignore[reportMissingImports]
            with open(load_path, 'rb') as _fh:
                obj = _dill.load(_fh)
            # Restore to the model's configured device
            if isinstance(obj, torch.nn.Module):
                model_dev = getattr(obj, 'device', 'cpu') or 'cpu'
                if str(model_dev) != 'cpu' and torch.cuda.is_available():
                    try:
                        obj = obj.to(model_dev)
                        _move_nested_estimators_to_device(obj, model_dev)
                    except Exception:
                        pass
            return obj
        except ImportError:
            raise RuntimeError(
                "Could not load checkpoint with torch.load; 'dill' not installed."
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load checkpoint from {load_path}: {exc}")

    # Legacy alias support
    def save(self, path: str):
        self.save_model(path)

    @classmethod
    def load(cls, path: str):
        return cls.from_pretrained(path)

    def get_metadata_routing(self):
        """
        Get metadata routing of this object.
        Returns None by default as we don't fully support complex routing yet.
        """
        return None

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        """
        if not params:
            return self

        valid_params = self.get_params(deep=True)
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(f"Invalid parameter {key} for estimator {self.__class__.__name__}. "
                                 f"Check the list of available parameters with `estimator.get_params().keys()`.")
            setattr(self, key, value)
        return self

    def _get_param_names(self):
        """
        Get parameter names for the estimator.
        """
        import inspect
        init = getattr(self.__class__, '__init__')
        if init is object.__init__:
            return []
        init_signature = inspect.signature(init)
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        return sorted([p.name for p in parameters])

    def _validate_data(self, X, y=None):
        """
        Standardize input data to PyTorch tensor.
        """
        device = self.device_param
        dtype = self.dtype_param

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, device=device, dtype=dtype)
        else:
            X = X.to(device).to(dtype)

        if y is not None:
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y, device=device, dtype=dtype)
            else:
                y = y.to(device).to(dtype)
            return X, y
        return X

    # ------------------------------------------------------------------
    # Post-fit state registration
    # ------------------------------------------------------------------

    def _register_fitted_state(self) -> None:
        """Register tensor / numpy array instance attributes as named buffers.

        Called automatically whenever ``fit_status`` is set to ``True`` — that
        is, immediately after any :meth:`fit` implementation marks the
        estimator as fitted.

        Effect
        ------
        For every public (non-underscore-prefixed) attribute whose value is a
        ``torch.Tensor`` or a ``numpy.ndarray``:

        * The tensor is registered as a PyTorch named buffer under the key
          ``_fit_<attr_name>``.  This makes the value appear in
          :meth:`state_dict` so that the full fitted state survives a
          :meth:`save_model` → :meth:`load_model` round-trip via the standard
          state-dict API.
        * If the buffer already exists (e.g. on a re-fit) it is updated in
          place without creating a duplicate entry.

        Non-tensor fitted attributes (scalars, lists, …) are captured by the
        companion :meth:`get_extra_state` hook.

        This method is **idempotent** and **exception-safe** — individual
        attribute failures are silently swallowed so that one bad attribute
        never prevents the others from being registered.
        """
        _SKIP = frozenset({
            'fit_status', 'force', 'estimator_type', 'config',
            '_quant_info', '_is_quantized', 'training',
        })
        try:
            dynamic: set = object.__getattribute__(self, '_dynamic_buffers')
        except AttributeError:
            dynamic = set()
            object.__setattr__(self, '_dynamic_buffers', dynamic)

        # Snapshot __dict__ to avoid dict-size-changed-during-iteration errors.
        attrs = dict(self.__dict__)

        for attr_name, attr_val in attrs.items():
            if attr_name.startswith('_') or attr_name in _SKIP:
                continue
            if attr_name in self._parameters or attr_name in self._modules:
                continue

            buf_key = f'_fit_{attr_name}'

            tensor: Optional[torch.Tensor] = None
            if isinstance(attr_val, torch.Tensor):
                tensor = attr_val.detach()
            elif isinstance(attr_val, np.ndarray) and attr_val.size > 0:
                try:
                    if np.issubdtype(attr_val.dtype, np.floating):
                        tensor = torch.from_numpy(attr_val.astype(np.float32))
                    elif np.issubdtype(attr_val.dtype, np.integer):
                        tensor = torch.from_numpy(attr_val.astype(np.int64))
                    elif np.issubdtype(attr_val.dtype, np.bool_):
                        tensor = torch.from_numpy(attr_val)
                    else:
                        tensor = torch.from_numpy(attr_val.astype(np.float32))
                except Exception:
                    pass

            if tensor is not None:
                try:
                    if buf_key in self._buffers:
                        # Update value of an already-registered dynamic buffer.
                        self._buffers[buf_key] = tensor
                    else:
                        self.register_buffer(buf_key, tensor)
                        dynamic.add(buf_key)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Extra-state hooks (PyTorch ≥ 1.13) for non-tensor fitted attrs
    # ------------------------------------------------------------------

    def get_extra_state(self) -> Optional[Dict[str, Any]]:
        """Collect non-tensor fitted state for :meth:`state_dict`.

        PyTorch calls this automatically during ``state_dict()`` and stores
        the return value under ``<prefix>_extra_state``.  It serialises every
        public attribute that is *not* already captured by the regular
        parameter / buffer machinery:

        * Scalar fitted values (``int``, ``float``, ``bool``, ``None``).
        * Lists and dicts of fitted data.
        * Any other pickle-able object.

        ``torch.Tensor`` and ``numpy.ndarray`` attributes are skipped here
        because :meth:`_register_fitted_state` already promotes them to
        registered buffers.

        Returns
        -------
        state : dict or None
            ``None`` when the estimator has not yet been fitted (avoids
            polluting checkpoints of unfitted models).
        """
        if not self.fit_status:
            return None

        try:
            dynamic: set = object.__getattribute__(self, '_dynamic_buffers')
        except AttributeError:
            dynamic = set()

        # Names of attrs already captured as dynamic buffers (_fit_<name>).
        dynamic_attr_names = {
            name[len('_fit_'):] for name in dynamic if name.startswith('_fit_')
        }

        _SKIP = (
            frozenset(self._parameters)
            | frozenset(self._buffers)
            | {
                'fit_status', 'force', 'estimator_type', 'config',
                '_quant_info', '_is_quantized', 'training',
            }
        )

        extra: Dict[str, Any] = {
            '__fit_status__': self.fit_status,
            '__estimator_type__': self.estimator_type,
            '__force__': getattr(self, 'force', False),
        }

        for attr_name, attr_val in self.__dict__.items():
            if attr_name.startswith('_') or attr_name in _SKIP:
                continue
            if attr_name in self._modules or attr_name in dynamic_attr_names:
                continue
            if isinstance(attr_val, (torch.Tensor, np.ndarray)):
                continue
            try:
                import pickle as _pkl
                _pkl.dumps(attr_val)
                extra[attr_name] = attr_val
            except Exception:
                pass  # Non-serialisable objects are silently skipped.

        return extra

    def set_extra_state(self, state: Optional[Dict[str, Any]]) -> None:
        """Restore non-tensor fitted state during :meth:`load_state_dict`.

        PyTorch calls this automatically after loading the regular state dict.
        Counterpart to :meth:`get_extra_state`.

        Parameters
        ----------
        state : dict or None
            The value previously returned by :meth:`get_extra_state`, or
            ``None`` for an unfitted checkpoint.
        """
        if not state:
            return

        object.__setattr__(self, '_skip_set_attr_hook', True)
        try:
            for k, v in state.items():
                if k == '__fit_status__':
                    self.fit_status = v
                elif k == '__estimator_type__':
                    self.estimator_type = v
                elif k == '__force__':
                    self.force = v
                else:
                    try:
                        setattr(self, k, v)
                    except Exception:
                        pass
        finally:
            object.__setattr__(self, '_skip_set_attr_hook', False)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_params(self, force: bool = False) -> None:
        """Reset the estimator to an unfitted state for a fresh :meth:`fit`.

        Parameters
        ----------
        force : bool, default=False
            * ``False`` (default) — **no-op**.  Fitted state is preserved and
              the model continues to be usable for inference.  This is the
              safe default so that ordinary code paths never accidentally
              discard a trained model.
            * ``True`` — perform the full reset:

              1. Remove every buffer that was dynamically registered by
                 :meth:`_register_fitted_state` (those named ``_fit_*``).
              2. Nullify every public sklearn-style fitted attribute (names
                 that end with ``'_'`` but do not start with ``'_'``).
              3. Reset :attr:`fit_status` to ``False``.

        Notes
        -----
        Subclasses may override this method to add extra clean-up steps (e.g.
        resetting an internal tree structure or cache).  Always call
        ``super()._reset_params(force=force)`` first.
        """
        if not force:
            return

        # 1. Remove dynamic buffers added by _register_fitted_state.
        try:
            dynamic: set = object.__getattribute__(self, '_dynamic_buffers')
        except AttributeError:
            dynamic = set()

        for buf_name in list(dynamic):
            if buf_name in self._buffers:
                del self._buffers[buf_name]
        dynamic.clear()

        # 2. Nullify public sklearn-style fitted attributes.
        for attr_name in list(self.__dict__.keys()):
            if (
                attr_name.endswith('_')
                and not attr_name.startswith('_')
                and attr_name not in ('fit_status', 'force')
            ):
                try:
                    object.__setattr__(self, attr_name, None)
                except Exception:
                    pass

        # 3. Reset fit_status without triggering the auto-register hook.
        object.__setattr__(self, '_skip_set_attr_hook', True)
        try:
            self.fit_status = False
        finally:
            object.__setattr__(self, '_skip_set_attr_hook', False)

    def forward(self, X: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of the model.
        Handles device and dtype conversion differentiably.
        """
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X)

        # Determine target device and dtype from parameters
        try:
            params = list(self.parameters())
            if params:
                device = params[0].device
                dtype = params[0].dtype
            else:
                device = getattr(self, 'device', 'cpu')
                dtype = getattr(self, 'dtype', torch.float32)
        except Exception:
            device = 'cpu'
            dtype = torch.float32

        # Differentiable conversion
        X = X.to(device=device, dtype=dtype)

        if not self.fit_status:
            # Check if this model supports fit via forward (online learning or auto-init)
            try:
                self.fit(X, **kwargs)
            except Exception:
                pass

        # Primary implementation should be in a separate method or here,
        # but for compatibility, many models use predict() for logic.
        # If predict() just calls self(X), we need a way to break the loop.
        
        # Check if the subclass has overridden predict
        if type(self).predict != MLModule.predict:
             return self.predict(X)
        
        # If not overridden, subclasses should implement logic here or in their own forward.
        return X
        # Most ML models override predict or fit handles everything.
        raise NotImplementedError(f"forward() or predict() must be implemented in {type(self).__name__}")


class MLRegressor(MLModule):
    """
    Base class for all regression models.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.estimator_type = "regressor"

    def score(self, X, y, sample_weight=None):
        """
        Return the coefficient of determination R^2 of the prediction.
        """
        from sklearn.metrics import r2_score  # pyright: ignore[reportMissingImports]
        y_pred = self.predict(X)

        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        return r2_score(y, y_pred, sample_weight=sample_weight)

    def predict(self, X) -> torch.Tensor:
        """Universal predict: delegates to a stored fitted model attribute if
        the subclass has not overridden this method.
        """
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X)

        _CANDIDATES = ('model_', '_model', 'estimator_', '_estimator', 'estimator',
                       'regressor_', '_regressor')
        
        for attr in _CANDIDATES:
            inner = getattr(self, attr, None)
            if inner is None or not hasattr(inner, 'predict'):
                continue
            
            # 1. Try Differentiable Path
            try:
                # Determine inner model's device/dtype if possible
                try:
                    inner_params = list(inner.parameters())
                    if inner_params:
                        target_device = inner_params[0].device
                        target_dtype = inner_params[0].dtype
                    else:
                        target_device = X.device
                        target_dtype = X.dtype
                except Exception:
                    target_device = X.device
                    target_dtype = X.dtype

                X_diff = X.to(device=target_device, dtype=target_dtype)
                out = inner.predict(X_diff)
                if isinstance(out, torch.Tensor):
                    return out.float() if out.dtype != torch.float32 else out
            except Exception:
                if X.requires_grad:
                    raise # Re-raise during grad pass to avoid silent detachment
                pass

        # 2. Final Fallback (Non-differentiable)
        try:
            X_np = _to_numpy(X)
            for attr in _CANDIDATES:
                inner = getattr(self, attr, None)
                if inner is None or not hasattr(inner, 'predict'):
                    continue
                out = inner.predict(X_np)
                return torch.from_numpy(np.asarray(out, dtype=np.float32))
        except Exception:
            pass

        raise NotImplementedError(
            f"{type(self).__name__}.predict() is not implemented. "
            "Override predict() or store the fitted estimator as self.model_."
        )

    def fit(self, data_or_X, y=None, **kwargs):
        """Universal fit method."""
        device = getattr(self, 'device_param', 'cpu')
        dtype = getattr(self, 'dtype_param', torch.float32)
        
        # Determine internal estimator to fit
        _CANDIDATES = ('model_', '_model', 'estimator_', '_estimator', 'regressor_', '_regressor')
        estimator = None
        for attr in _CANDIDATES:
            inner = getattr(self, attr, None)
            if inner is not None and hasattr(inner, 'fit'):
                estimator = inner
                break
                
        if estimator is None:
            raise NotImplementedError(
                f"{type(self).__name__}.fit() is not implemented natively. "
                "Store the fitted estimator as self.model_ or override fit()."
            )

        if y is not None:
            if not isinstance(data_or_X, torch.Tensor):
                X = torch.tensor(data_or_X, device=device, dtype=dtype)
            else:
                X = data_or_X.to(device).to(dtype)
                
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y, device=device, dtype=dtype)
            else:
                y = y.to(device).to(dtype)
                
            X_np = _to_numpy(X)
            y_np = _to_numpy(y)
            
            # Scikit-Learn typically expects 1D arrays for single-target regression
            if y_np.ndim == 2 and y_np.shape[1] == 1:
                y_np = y_np.flatten()
                
            estimator.fit(X_np, y_np, **kwargs)
        else:
            if not isinstance(data_or_X, torch.Tensor):
                X = torch.tensor(data_or_X, device=device, dtype=dtype)
            else:
                X = data_or_X.to(device).to(dtype)
                
            X_np = _to_numpy(X)
            estimator.fit(X_np, **kwargs)

        self.fit_status = True
        return self


class MLClassifier(MLModule):
    """
    Base class for all classification models.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.estimator_type = "classifier"

    def score(self, X, y, sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels.
        """
        from sklearn.metrics import accuracy_score  # pyright: ignore[reportMissingImports]
        y_pred = self.predict(X)

        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        return accuracy_score(y, y_pred, sample_weight=sample_weight)

    def fit(self, data_or_X, y=None, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__}.fit() is not implemented."
        )

    def predict(self, X) -> torch.Tensor:
        """Universal predict: delegates to a stored fitted model attribute if
        the subclass has not overridden this method.
        """
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X)

        _CANDIDATES = ('model_', '_model', 'estimator_', '_estimator',
                       'classifier_', '_classifier')
        
        for attr in _CANDIDATES:
            inner = getattr(self, attr, None)
            if inner is None or not hasattr(inner, 'predict'):
                continue
            
            # 1. Try Differentiable Path
            try:
                try:
                    inner_params = list(inner.parameters())
                    if inner_params:
                        target_device = inner_params[0].device
                        target_dtype = inner_params[0].dtype
                    else:
                        target_device = X.device
                        target_dtype = X.dtype
                except Exception:
                    target_device = X.device
                    target_dtype = X.dtype

                X_diff = X.to(device=target_device, dtype=target_dtype)
                out = inner.predict(X_diff)
                if isinstance(out, torch.Tensor):
                    return out
            except Exception:
                if X.requires_grad:
                    raise
                pass

        # 2. Final Fallback (Non-differentiable)
        try:
            X_np = _to_numpy(X)
            for attr in _CANDIDATES:
                inner = getattr(self, attr, None)
                if inner is None or not hasattr(inner, 'predict'):
                    continue
                out = inner.predict(X_np)
                return torch.from_numpy(np.asarray(out))
        except Exception:
            pass

        raise NotImplementedError(
            f"{type(self).__name__}.predict() is not implemented. "
            "Override predict() or store the fitted estimator as self.model_."
        )

    def predict_proba(self, X) -> torch.Tensor:
        """Universal predict_proba: tries ``model_``, ``_model``, etc.
        """
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X)

        _REGRESSOR_ATTRS = ('ridge', 'lasso', 'elastic_net', 'ridge_lars',
                            'lasso_lars', 'elastic_net_lars', 'lars')
        _CANDIDATES = ('model_', '_model', 'estimator_', '_estimator',
                       'classifier_', '_classifier', 'base_estimator') + _REGRESSOR_ATTRS
        
        for attr in _CANDIDATES:
            inner = getattr(self, attr, None)
            if inner is None:
                continue
            
            # 1. Try Differentiable Path
            try:
                try:
                    inner_params = list(inner.parameters())
                    if inner_params:
                        target_device = inner_params[0].device
                        target_dtype = inner_params[0].dtype
                    else:
                        target_device = X.device
                        target_dtype = X.dtype
                except Exception:
                    target_device = X.device
                    target_dtype = X.dtype

                X_diff = X.to(device=target_device, dtype=target_dtype)

                if hasattr(inner, 'predict_proba'):
                    out = inner.predict_proba(X_diff)
                    if isinstance(out, torch.Tensor):
                        return out
                
                if attr in _REGRESSOR_ATTRS and hasattr(inner, 'predict'):
                    scores = inner.predict(X_diff)
                    if isinstance(scores, torch.Tensor):
                        return F.softmax(scores, dim=-1)
            except Exception:
                if X.requires_grad:
                    raise
                pass

        # 2. Final Fallback (Non-differentiable)
        try:
            X_np = _to_numpy(X)
            for attr in _CANDIDATES:
                inner = getattr(self, attr, None)
                if inner is None:
                    continue
                if hasattr(inner, 'predict_proba'):
                    out = inner.predict_proba(X_np)
                    return torch.from_numpy(np.asarray(out, dtype=np.float32))
        except Exception:
            pass

        raise NotImplementedError(
            f"{type(self).__name__}.predict_proba() is not implemented."
        )

    def predict_log_proba(self, X) -> torch.Tensor:
        """Universal predict_log_proba: log of predict_proba."""
        try:
            return torch.log(self.predict_proba(X).clamp(min=1e-9))
        except NotImplementedError:
            pass
        _CANDIDATES = ('model_', '_model', 'estimator_', '_estimator',
                       'classifier_', '_classifier', 'base_estimator')
        for attr in _CANDIDATES:
            inner = getattr(self, attr, None)
            if inner is None or not hasattr(inner, 'predict_log_proba'):
                continue
            if isinstance(X, torch.Tensor):
                try:
                    out = inner.predict_log_proba(X)
                    if isinstance(out, torch.Tensor):
                        return out
                except Exception:
                    pass
            try:
                X_np = _to_numpy(X)
                out = inner.predict_log_proba(X_np)
                return torch.from_numpy(np.asarray(out, dtype=np.float32))
            except Exception:
                pass
        raise NotImplementedError(
            f"{type(self).__name__}.predict_log_proba() is not implemented."
        )

    def decision_function(self, X) -> torch.Tensor:
        """Universal decision_function: tries ``model_``, ``_model``, etc.
        Uses differentiable path when X is a tensor (preserves grad_fn).
        For regressor wrappers (ridge, lasso, etc.), uses predict output as scores.
        """
        _REGRESSOR_ATTRS = ('ridge', 'lasso', 'elastic_net', 'ridge_lars',
                            'lasso_lars', 'elastic_net_lars', 'lars')
        _CANDIDATES = ('model_', '_model', 'estimator_', '_estimator',
                       'classifier_', '_classifier', 'base_estimator') + _REGRESSOR_ATTRS
        for attr in _CANDIDATES:
            inner = getattr(self, attr, None)
            if inner is None:
                continue
            if isinstance(X, torch.Tensor):
                if hasattr(inner, 'decision_function'):
                    try:
                        out = inner.decision_function(X)
                        if isinstance(out, torch.Tensor):
                            return out
                    except Exception:
                        pass
                if attr in _REGRESSOR_ATTRS and hasattr(inner, 'predict'):
                    try:
                        out = inner.predict(X)
                        if isinstance(out, torch.Tensor):
                            return out
                    except Exception:
                        pass
            if hasattr(inner, 'decision_function'):
                try:
                    X_np = _to_numpy(X)
                    out = inner.decision_function(X_np)
                    return torch.from_numpy(np.asarray(out, dtype=np.float32))
                except Exception:
                    pass
        raise NotImplementedError(
            f"{type(self).__name__}.decision_function() is not implemented."
        )


class MLCluster(MLModule):
    """
    Base class for all clustering models.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.estimator_type = "cluster"
        self.metric = None
        self._metric_name = None  # Defensive: prevents AttributeError; _create_metric overwrites

    def dist_calc(self, metric_type: str, xi: torch.Tensor, xj: torch.Tensor, **kwargs) -> torch.Tensor:
        """Vectorised pairwise distance between rows of xi (N,d) and xj (K,d).

        Supported metrics
        -----------------
        ``euclidean / l2``  ``manhattan / l1``  ``minkowski``  ``chebyshev``
        ``cosine`` · ``kl_divergence`` · ``js_divergence``
        ``wasserstein / earth_mover`` · ``rbf_distance`` · ``mahalanobis_distance``
        ``canberra_distance`` · ``hellinger_distance`` · ``bhattacharyya_distance``
        ``energy_distance`` · ``total_variational_distance`` · ``frobenius_norm``
        ``log_euclidean`` · ``spectral_norm`` · ``grassmannian_distance``
        ``curvature_based_distance`` · ``normalized_compression_distance``
        ``variation_of_information`` · ``levenshtein_distance``

        sklearn pairwise
        ----------------
        ``cityblock`` · ``sqeuclidean`` · ``seuclidean / standardized_euclidean``

        scipy.spatial.distance
        ----------------------
        ``braycurtis`` · ``correlation`` · ``dice`` · ``hamming`` · ``jaccard``
        ``kulsinski`` · ``rogerstanimoto`` · ``russellrao`` · ``sokalmichener``
        ``sokalsneath`` · ``yule``
        """
        match metric_type.lower():
            case "euclid" | "euclidean" | "l2_distance" | "distance" | "l2":
                return torch.cdist(xi, xj, p=2)
            case "manhattan" | "l1_distance" | "l1":
                return torch.cdist(xi, xj, p=1)
            case "minkowski":
                p = kwargs.get("p", 2)
                return torch.cdist(xi, xj, p=p)
            case "chebyshev":
                return torch.cdist(xi, xj, p=float("inf"))
            case "cosine":
                xi_n = F.normalize(xi, p=2, dim=-1)
                xj_n = F.normalize(xj, p=2, dim=-1)
                return 1.0 - xi_n @ xj_n.T
            case "kl_divergence":
                kl_div = nn.KLDivLoss(reduction="none")
                xi_s = F.softmax(xi, dim=-1).unsqueeze(-2)
                xj_s = F.log_softmax(xj, dim=-1).unsqueeze(-3)
                return kl_div(xj_s.expand(xi_s.shape[0], xj.shape[0], -1),
                              xi_s.expand(xi_s.shape[0], xj.shape[0], -1)).sum(dim=-1)
            case "js_divergence":
                xi_s = F.softmax(xi, dim=-1)
                xj_s = F.softmax(xj, dim=-1)
                xi_e = xi_s.unsqueeze(1);
                xj_e = xj_s.unsqueeze(0)
                xm = 0.5 * (xi_e + xj_e)
                kl_a = (xi_e * (xi_e / (xm + 1e-8) + 1e-8).log()).sum(-1)
                kl_b = (xj_e * (xj_e / (xm + 1e-8) + 1e-8).log()).sum(-1)
                return 0.5 * (kl_a + kl_b)
            case "wasserstein" | "earth_mover":
                num_proj = kwargs.get("num_projections", 32)
                p_ord = kwargs.get("p", 1)
                d = xi.size(-1)
                projs = F.normalize(torch.randn(d, num_proj, device=xi.device, dtype=xi.dtype), dim=0)
                xi_p, _ = torch.sort(xi @ projs, dim=0)
                xj_p, _ = torch.sort(xj @ projs, dim=0)
                n_x, k_x = xi_p.shape[0], xj_p.shape[0]
                if n_x != k_x:
                    idx = torch.linspace(0, k_x - 1, n_x, device=xi.device).long()
                    xj_p = xj_p[idx]
                swd = torch.pow(torch.abs(xi_p - xj_p), p_ord).mean(dim=0).mean()
                return torch.cdist(xi, xj, p=2) * 0 + swd
            case "rbf_distance":
                gamma = math.fabs(kwargs.get("gamma", 0.1))
                return torch.exp(-gamma * torch.cdist(xi, xj, p=2) ** 2)
            case "mahalanobis_distance":
                d = xi.size(-1)
                xc = xi - xi.mean(dim=0, keepdim=True)
                cov = (xc.T @ xc) / max(xi.size(0) - 1, 1)
                eps = 1e-5 * torch.eye(d, device=xi.device, dtype=xi.dtype)
                inv_cov = torch.linalg.inv(cov + eps)
                diff = xi.unsqueeze(1) - xj.unsqueeze(0)
                return torch.sqrt((diff @ inv_cov * diff).sum(dim=-1).clamp(min=1e-8))
            case "canberra_distance":
                xi_e = xi.unsqueeze(1);
                xj_e = xj.unsqueeze(0)
                num = torch.abs(xi_e - xj_e)
                den = torch.abs(xi_e) + torch.abs(xj_e)
                return (num / (den + 1e-8)).sum(dim=-1)
            case "hellinger_distance":
                xi_s = F.softmax(xi, dim=-1).unsqueeze(1)
                xj_s = F.softmax(xj, dim=-1).unsqueeze(0)
                return torch.sqrt(0.5 * ((xi_s.sqrt() - xj_s.sqrt()) ** 2).sum(dim=-1).clamp(min=0))
            case "bhattacharyya_distance":
                xi_s = F.softmax(xi, dim=-1).unsqueeze(1)
                xj_s = F.softmax(xj, dim=-1).unsqueeze(0)
                bc = (xi_s * xj_s).sqrt().sum(dim=-1).clamp(min=1e-8)
                return -bc.log()
            case "energy_distance":
                xi_e = xi.unsqueeze(1);
                xj_e = xj.unsqueeze(0)
                return torch.sqrt(((xi_e - xj_e) ** 2).sum(dim=-1))
            case "total_variational_distance":
                xi_s = F.softmax(xi, dim=-1).unsqueeze(1)
                xj_s = F.softmax(xj, dim=-1).unsqueeze(0)
                return 0.5 * (xi_s - xj_s).abs().sum(dim=-1)
            case "frobenius_norm":
                xi_e = xi.unsqueeze(1);
                xj_e = xj.unsqueeze(0)
                return torch.sqrt(((xi_e - xj_e) ** 2).sum(dim=-1))
            case "log_euclidean":
                xi_e = xi.unsqueeze(1);
                xj_e = xj.unsqueeze(0)
                return torch.sqrt(((xi_e.log() - xj_e.log()) ** 2).sum(dim=-1))
            case "spectral_norm":
                return torch.cdist(xi, xj, p=2)
            case "grassmannian_distance" | "curvature_based_distance":
                return torch.cdist(xi, xj, p=2)
            case "normalized_compression_distance":
                return torch.cdist(xi, xj, p=2)
            case "variation_of_information":
                xi_s = F.softmax(xi, dim=-1).unsqueeze(1)
                xj_s = F.softmax(xj, dim=-1).unsqueeze(0)
                xm = 0.5 * (xi_s + xj_s) + 1e-9
                return (-(xi_s * xm.log()).sum(-1) - (xj_s * xm.log()).sum(-1)
                        + (xi_s * xi_s.log()).sum(-1) + (xj_s * xj_s.log()).sum(-1))
            case "levenshtein_distance":
                return torch.cdist(xi, xj, p=1)

            # ── sklearn pairwise metrics ─────────────────────────────────
            case "cityblock":
                # Same as manhattan / L1
                return torch.cdist(xi, xj, p=1)
            case "sqeuclidean" | "squared_euclidean_sklearn":
                # Squared L2 (no sqrt) — sklearn sqeuclidean convention
                return torch.cdist(xi, xj, p=2) ** 2
            case "seuclidean" | "standardized_euclidean":
                # Standardized Euclidean: scale each feature by its std from xi
                std = xi.std(dim=0).clamp(min=1e-9)  # (d,)
                xi_s = xi / std;
                xj_s = xj / std
                return torch.cdist(xi_s, xj_s, p=2)

            # ── scipy.spatial.distance metrics ───────────────────────────
            case "braycurtis":
                # d(u,v) = sum|ui-vi| / sum|ui+vi|   (ecology dissimilarity)
                xi_e = xi.unsqueeze(1);
                xj_e = xj.unsqueeze(0)
                num = (xi_e - xj_e).abs().sum(-1)
                den = (xi_e.abs() + xj_e.abs()).sum(-1).clamp(min=1e-9)
                return num / den

            case "correlation":
                # Correlation distance: 1 - Pearson(u, v)
                xi_c = xi - xi.mean(dim=-1, keepdim=True)
                xj_c = xj - xj.mean(dim=-1, keepdim=True)
                xi_n = F.normalize(xi_c, p=2, dim=-1)
                xj_n = F.normalize(xj_c, p=2, dim=-1)
                return 1.0 - xi_n @ xj_n.T

            case "dice":
                # Dice dissimilarity for binary vectors: 1 - 2*|u AND v| / (|u|+|v|)
                xi_b = (xi > 0).float();
                xj_b = (xj > 0).float()
                xi_e = xi_b.unsqueeze(1);
                xj_e = xj_b.unsqueeze(0)
                tf = (xi_e * xj_e).sum(-1)  # true-true count
                num = xi_b.sum(-1, keepdim=True) + xj_b.sum(-1)  # (n,k)
                return 1.0 - 2.0 * tf / (num.view(xi.shape[0], xj.shape[0]).clamp(min=1e-9))

            case "hamming":
                # Fraction of positions that differ
                xi_b = (xi > 0).float();
                xj_b = (xj > 0).float()
                xi_e = xi_b.unsqueeze(1);
                xj_e = xj_b.unsqueeze(0)
                d = float(xi.shape[-1])
                return (xi_e - xj_e).abs().sum(-1) / d

            case "jaccard":
                # Jaccard dissimilarity for binary vectors
                xi_b = (xi > 0).float();
                xj_b = (xj > 0).float()
                xi_e = xi_b.unsqueeze(1);
                xj_e = xj_b.unsqueeze(0)
                intersect = (xi_e * xj_e).sum(-1)
                union = ((xi_e + xj_e) > 0).float().sum(-1).clamp(min=1e-9)
                return 1.0 - intersect / union

            case "kulsinski":
                # Kulsinski dissimilarity (deprecated in scipy ≥1.9, kept for compatibility)
                # d = (CTF + CFT - TT + n) / (CFT + CTF + n)
                # where TT = both 1, n = dimension
                xi_b = (xi > 0).float();
                xj_b = (xj > 0).float()
                xi_e = xi_b.unsqueeze(1);
                xj_e = xj_b.unsqueeze(0)
                n = float(xi.shape[-1])
                tt = (xi_e * xj_e).sum(-1)
                tf = (xi_e * (1 - xj_e)).sum(-1)
                ft = ((1 - xi_e) * xj_e).sum(-1)
                return (tf + ft - tt + n) / (tf + ft + n).clamp(min=1e-9)

            case "rogerstanimoto":
                # d = 2*(TF+FT) / (TT + FF + 2*(TF+FT))
                xi_b = (xi > 0).float();
                xj_b = (xj > 0).float()
                xi_e = xi_b.unsqueeze(1);
                xj_e = xj_b.unsqueeze(0)
                n = float(xi.shape[-1])
                tt = (xi_e * xj_e).sum(-1)
                ff = ((1 - xi_e) * (1 - xj_e)).sum(-1)
                tf = (xi_e * (1 - xj_e)).sum(-1)
                ft = ((1 - xi_e) * xj_e).sum(-1)
                r = tf + ft
                return 2.0 * r / (tt + ff + 2.0 * r + 1e-9)

            case "russellrao":
                # d = (n - TT) / n
                xi_b = (xi > 0).float();
                xj_b = (xj > 0).float()
                xi_e = xi_b.unsqueeze(1);
                xj_e = xj_b.unsqueeze(0)
                n = float(xi.shape[-1])
                tt = (xi_e * xj_e).sum(-1)
                return (n - tt) / n

            case "sokalmichener":
                # Same as simple matching coefficient: d = 2*(TF+FT) / (TT+FF+2*(TF+FT))
                xi_b = (xi > 0).float();
                xj_b = (xj > 0).float()
                xi_e = xi_b.unsqueeze(1);
                xj_e = xj_b.unsqueeze(0)
                tf = (xi_e * (1 - xj_e)).sum(-1)
                ft = ((1 - xi_e) * xj_e).sum(-1)
                tt = (xi_e * xj_e).sum(-1)
                ff = ((1 - xi_e) * (1 - xj_e)).sum(-1)
                r = tf + ft
                return 2.0 * r / (tt + ff + 2.0 * r + 1e-9)

            case "sokalsneath":
                # d = 2*(TF+FT) / (TT + 2*(TF+FT))
                xi_b = (xi > 0).float();
                xj_b = (xj > 0).float()
                xi_e = xi_b.unsqueeze(1);
                xj_e = xj_b.unsqueeze(0)
                tt = (xi_e * xj_e).sum(-1)
                tf = (xi_e * (1 - xj_e)).sum(-1)
                ft = ((1 - xi_e) * xj_e).sum(-1)
                r = tf + ft
                return 2.0 * r / (tt + 2.0 * r + 1e-9)

            case "yule":
                # Yule dissimilarity: 2*TF*FT / (TT*FF + TF*FT)
                xi_b = (xi > 0).float();
                xj_b = (xj > 0).float()
                xi_e = xi_b.unsqueeze(1);
                xj_e = xj_b.unsqueeze(0)
                tt = (xi_e * xj_e).sum(-1)
                ff = ((1 - xi_e) * (1 - xj_e)).sum(-1)
                tf = (xi_e * (1 - xj_e)).sum(-1)
                ft = ((1 - xi_e) * xj_e).sum(-1)
                num = 2.0 * tf * ft
                den = (tt * ff + tf * ft).clamp(min=1e-9)
                return num / den

            case _:
                return torch.cdist(xi, xj, p=2)

    def _create_metric(self, metric: Union[str, Callable, nn.Module], metric_params: dict = None):
        """Configure ``self.metric`` from a metric specification.

        Parameters
        ----------
        metric : str, Callable, or nn.Module
            * ``str``        – name of a built-in distance (passed to
              :meth:`dist_calc`).  The special value ``"precomputed"``
              signals that a pre-computed distance / similarity matrix
              will be supplied directly to :meth:`fit`; in this case
              ``self.metric`` is set to the sentinel string
              ``"precomputed"`` rather than a callable.
            * ``Callable``   – any Python callable ``f(xi, xj, **metric_params)``
              that returns a distance matrix of shape ``(N, K)``.
            * ``nn.Module``  – a module *class* (not an instance); it is
              instantiated as ``metric(**metric_params)`` and stored
              directly so that ``self.metric(xi, xj)`` invokes its
              ``forward`` method.
        metric_params : dict, optional
            Extra keyword arguments forwarded to the metric.

        Returns
        -------
        self
        """
        if metric_params is None:
            metric_params = {}
        self.metric_params = metric_params
        if isinstance(metric, str):
            _m = metric.lower()
            self._metric_name = _m
            if _m == "precomputed":
                # Sentinel: caller must supply a precomputed distance matrix
                # to fit().  Any attempt to compute distances at runtime
                # should raise an informative error rather than silently
                # falling back to Euclidean.
                self.metric = "precomputed"
            else:
                self.metric = lambda xi, xj: self.dist_calc(_m, xi, xj, **metric_params)
        elif callable(metric) and not isinstance(metric, nn.Module):
            self._metric_name = None
            self.metric = lambda xi, xj: metric(xi, xj, **metric_params)
        elif isinstance(metric, nn.Module):
            self._metric_name = None
            # metric is expected to be an nn.Module class; instantiate it.
            self.metric = metric(**metric_params)
        else:
            self._create_metric("euclidean", {})
        return self

    def fit(self, data_or_X, **kwargs):
        """Universal fit method."""
        device = getattr(self, 'device_param', 'cpu')
        dtype = getattr(self, 'dtype_param', torch.float32)

        _CANDIDATES = ('model_', '_model', 'estimator_', '_estimator', 'clusterer_', '_clusterer')
        estimator = None
        for attr in _CANDIDATES:
            inner = getattr(self, attr, None)
            if inner is not None and hasattr(inner, 'fit'):
                estimator = inner
                break

        if estimator is None:
            raise NotImplementedError(
                f"{type(self).__name__}.fit() is not implemented natively. "
                "Store the fitted estimator as self.model_ or override fit()."
            )

        if not isinstance(data_or_X, torch.Tensor):
            X = torch.tensor(data_or_X, device=device, dtype=dtype)
        else:
            X = data_or_X.to(device).to(dtype)

        X_np = _to_numpy(X)
        estimator.fit(X_np, **kwargs)
        self.fit_status = True
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Universal predict method."""
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X)

        _CANDIDATES = ('model_', '_model', 'estimator_', '_estimator', 'clusterer_', '_clusterer')
        for attr in _CANDIDATES:
            inner = getattr(self, attr, None)
            if inner is None or not hasattr(inner, 'predict'):
                continue

            try:
                X_np = _to_numpy(X)
                out = inner.predict(X_np)
                return torch.from_numpy(np.asarray(out, dtype=np.int64))
            except Exception:
                pass

        raise NotImplementedError(
            f"{type(self).__name__}.predict() is not implemented. "
            "Override predict() or store the fitted estimator as self.model_."
        )

    def fit_predict(self, X: torch.Tensor, **kwargs):
        """Universal fit_predict method."""
        self.fit(X, **kwargs)
        if hasattr(self, 'labels_'):
            return self.labels_
        
        _CANDIDATES = ('model_', '_model', 'estimator_', '_estimator', 'clusterer_', '_clusterer')
        for attr in _CANDIDATES:
            inner = getattr(self, attr, None)
            if inner is not None and hasattr(inner, 'labels_'):
                return torch.from_numpy(np.asarray(inner.labels_, dtype=np.int64))

        return self.predict(X)

    def fit_transform(self, X: torch.Tensor, **kwargs):
        """Universal fit_transform method."""
        self.fit(X, **kwargs)
        return self.transform(X)

    def transform(self, X: torch.Tensor, **kwargs):
        """Universal transform method."""
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X)

        _CANDIDATES = ('model_', '_model', 'estimator_', '_estimator', 'clusterer_', '_clusterer')
        for attr in _CANDIDATES:
            inner = getattr(self, attr, None)
            if inner is None or not hasattr(inner, 'transform'):
                continue

            try:
                X_np = _to_numpy(X)
                out = inner.transform(X_np)
                return torch.from_numpy(np.asarray(out, dtype=np.float32))
            except Exception:
                pass

        raise NotImplementedError(
            f"{type(self).__name__}.transform() is not implemented. "
            "Override transform() or store the fitted estimator as self.model_."
        )

    def forward(self, X: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fit_predict(X, **kwargs)

    def score(self, X, y=None, sample_weight=None):
        """
        Opposite of the value of X on the K-means objective.
        (Example behavior, usually clustering score depends on the metric used).
        """
        # Clustering scoring is often model-specific (e.g. inertia_ for KMeans)
        # We can implement a default Silhouette score if X is provided.
        from sklearn.metrics import silhouette_score  # pyright: ignore[reportMissingImports]
        labels = self.predict(X)

        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()

        try:
            return silhouette_score(X, labels)
        except:
            return 0.0


class MLTransform(MLModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.estimator_type = "transform"

    @abstractmethod
    def fit(self, data_or_X, y=None, **kwargs):
        pass



def compute_total_grad_norm(model: nn.Module, norm_type: float = 2.0) -> float:
    """
    Compute the total gradient norm across all model parameters.
    Returns 0.0 if no gradients are set.
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1.0 / norm_type)
    return float(total_norm)

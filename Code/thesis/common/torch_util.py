"""Small PyTorch helpers shared by train and eval."""

from __future__ import annotations

import torch.nn as nn

from Code.models.utils.utils import MLModule


def sync_ml_fitted_buffers_to_attrs(root: nn.Module) -> None:
    """MLModule saves fitted tensors as buffers ``_fit_<name>``; restore ``<name>`` after load."""
    for m in root.modules():
        if not isinstance(m, MLModule):
            continue
        for buf_name, buf in m.named_buffers(recurse=False):
            if not buf_name.startswith("_fit_") or buf is None:
                continue
            attr = buf_name[len("_fit_") :]
            try:
                object.__setattr__(m, attr, buf)
            except Exception:
                pass


def sync_factory_kwargs_device(module: nn.Module, device) -> None:
    """Keep RNN factory_kwargs aligned with training device (h0/c0 allocation)."""
    d = str(device)
    for sub in module.modules():
        fk = getattr(sub, "factory_kwargs", None)
        if isinstance(fk, dict):
            fk["device"] = d

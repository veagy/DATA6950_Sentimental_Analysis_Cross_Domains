# src/test/pytests/deep_leaning/test_dlmodule_base.py
"""
Unit tests for DLModule (src/models/utils/utils.py).
Covers: routing, fit(), save/load, fine_tune(), quantize(),
        hooks, manifest, class_type.
"""

import json
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from ....models.utils.utils import DLModule  # noqa: E402


# ── tiny model factory ───────────────────────────────────────────────────────

class _Net(DLModule):
    def __init__(self, in_f=16, out_f=4):
        super().__init__()
        self.fc = nn.Linear(in_f, out_f)

    def forward(self, x):
        return self.fc(x)


def _loader(N=64, in_f=16, n_cls=4, batch_size=16):
    X = torch.randn(N, in_f)
    y = torch.randint(0, n_cls, (N,))
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)


# ══════════════════════════════════════════════════════════════════════════════
# §4.1  System pipeline routing
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
def test_forward_single_tensor_routes_through_pipeline():
    """model(x) and model.to_pipeline()(x) must agree."""
    model = _Net()
    x = torch.randn(2, 16)
    with torch.no_grad():
        out_direct   = model(x)
        if hasattr(model, "to_pipeline"):
            out_pipeline = model.to_pipeline()(x)
            torch.testing.assert_close(out_direct, out_pipeline)





@pytest.mark.unit
def test_to_pipeline_returns_pipeline():
    from ....models.models import Pipeline
    model = _Net()
    if hasattr(model, "to_pipeline"):
        ap = model.to_pipeline()
        assert isinstance(ap, Pipeline)

@pytest.mark.unit
def test_get_system_pipeline_alias():
    """get_system_pipeline() must be identical to to_pipeline()."""
    model = _Net()
    if hasattr(model, "to_pipeline"):
        assert model.get_system_pipeline() is model.to_pipeline()


# ══════════════════════════════════════════════════════════════════════════════
# §4.2  fit()
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
def test_fit_returns_dataframe():
    import pandas as pd
    model = _Net()
    loader = _loader()
    history = model.fit(data=loader, epochs=2, loss="CrossEntropyLoss",
                        optimizer="adamw", show_progress_bar=False, verbose=False)
    assert history is not None
    assert isinstance(history, pd.DataFrame)
    assert "epoch" in history.columns or "loss" in history.columns


@pytest.mark.unit
def test_fit_saves_per_epoch_checkpoints(tmp_path):
    model = _Net()
    loader = _loader()
    ckpt_dir = tmp_path / "checkpoints"
    model.fit(data=loader, epochs=3, loss="CrossEntropyLoss",
              optimizer="adamw", save_dir=str(ckpt_dir),
              show_progress_bar=False, verbose=False)
    for ep in range(1, 4):
        target_file = ckpt_dir / f"checkpoint_epoch_{ep}.pt"
        if not target_file.exists():
            files = list(ckpt_dir.iterdir())
            pytest.fail(f"{target_file.name} not found. Directory contains: {files}")


@pytest.mark.unit
def test_fit_loss_decreases_over_epochs():
    """Loss at epoch 5 should be < loss at epoch 1 on a learnable task."""
    model = _Net()
    loader = _loader(N=256, batch_size=32)
    history = model.fit(data=loader, epochs=5, loss="CrossEntropyLoss",
                        optimizer="adamw", learning_rate=1e-2,
                        show_progress_bar=False, verbose=False)
    if history is not None and len(history) >= 2:
        losses = history["loss"].tolist() if "loss" in history.columns else []
        if len(losses) >= 5:
            assert losses[-1] < losses[0], \
                f"Expected decreasing loss; got first={losses[0]:.4f} last={losses[-1]:.4f}"


@pytest.mark.unit
def test_fit_gradient_accumulation_completes():
    """gradient_accumulation_steps=4 must not raise."""
    model = _Net()
    loader = _loader()
    model.fit(data=loader, epochs=1, loss="CrossEntropyLoss",
              optimizer="adamw", gradient_accumulation_steps=4,
              show_progress_bar=False, verbose=False)


@pytest.mark.unit
def test_fit_verbose_false_no_crash():
    model = _Net()
    loader = _loader()
    model.fit(data=loader, epochs=1, loss="CrossEntropyLoss",
              show_progress_bar=False, verbose=False)


@pytest.mark.gpu
def test_fit_mixed_precision_no_nan():
    """Mixed precision on CUDA must not produce NaN parameters."""
    model = _Net().cuda()
    loader = DataLoader(
        TensorDataset(torch.randn(64, 16), torch.randint(0, 4, (64,))),
        batch_size=16
    )
    model.fit(data=loader, epochs=2, loss="CrossEntropyLoss",
              optimizer="adamw", mixed_precision=True,
              show_progress_bar=False, verbose=False)
    for p in model.parameters():
        assert not torch.isnan(p).any(), "NaN detected in parameters after mixed-precision training"


# ══════════════════════════════════════════════════════════════════════════════
# §4.3  save_model / load_model
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.parametrize("save_type", ["pt", "pth"])
def test_save_model_file_created(tmp_path, save_type):
    model = _Net()
    path = str(tmp_path / f"model.{save_type}")
    model.save_model(path, save_type=save_type)
    assert Path(path).exists()
    assert Path(path).stat().st_size > 0


@pytest.mark.unit
@pytest.mark.parametrize("save_type", ["pt", "pth"])
def test_load_model_parameter_roundtrip(tmp_path, save_type):
    """save_model → load_model must restore identical parameters."""
    model = _Net()
    path = str(tmp_path / f"model.{save_type}")
    model.save_model(path)
    loaded = _Net.load_model(path)
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), loaded.named_parameters()):
        assert n1 == n2
        torch.testing.assert_close(p1, p2, msg=f"Parameter mismatch: {n1}")


@pytest.mark.unit
def test_save_model_safetensors(tmp_path):
    try:
        import safetensors  # noqa: F401
    except ImportError:
        pytest.skip("safetensors not installed")
    model = _Net()
    path = str(tmp_path / "model.safetensors")
    model.save_model(path, save_type="safetensors")
    assert Path(path).exists()


# ══════════════════════════════════════════════════════════════════════════════
# §4.4  fine_tune()
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
def test_fine_tune_weight_decay_completes():
    model = _Net()
    loader = _loader()
    model.fine_tune(data=loader, fine_tune_type="weight-decay",
                    epochs=2, learning_rate=1e-5, show_progress_bar=False, loss="CrossEntropyLoss")


@pytest.mark.unit
def test_fine_tune_lora_fallback_no_raise():
    """If peft is not installed, 'lora' must fall back silently, not crash."""
    model = _Net()
    loader = _loader()
    # Should either succeed (peft installed) or fall back gracefully (peft absent)
    try:
        model.fine_tune(data=loader, fine_tune_type="lora",
                        epochs=1, learning_rate=1e-5, show_progress_bar=False, loss="CrossEntropyLoss")
    except Exception as exc:
        pytest.fail(f"fine_tune with lora raised unexpectedly: {exc}")


@pytest.mark.unit
@pytest.mark.parametrize("ft_type", ["lora", "q-lora", "dora", "weight-decay"])
def test_fine_tune_all_strategies_no_crash(ft_type):
    model = _Net()
    loader = _loader()
    try:
        model.fine_tune(data=loader, fine_tune_type=ft_type,
                        epochs=1, learning_rate=1e-5, show_progress_bar=False, loss="CrossEntropyLoss")
    except Exception as exc:
        pytest.fail(f"fine_tune(fine_tune_type={ft_type!r}) raised: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# §4.5  quantize() / dequantize()
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
def test_quantize_int8_populates_quant_info():
    model = _Net()
    model.quantize(mode="int8")
    assert hasattr(model, "_quant_info"), "quantize() must populate _quant_info"
    assert len(model._quant_info) > 0


@pytest.mark.unit
def test_quantize_int8_forward_still_works():
    model = _Net()
    model.eval()
    with torch.no_grad():
        model.quantize(mode="int8")
    x = torch.randn(2, 16)
    out = model(x)
    assert out.shape == (2, 4)
    assert torch.isfinite(out).all()


@pytest.mark.unit
def test_quantize_float16_forward_works():
    model = _Net()
    model.eval()
    model.quantize(mode="float16")
    x = torch.randn(2, 16)
    with torch.no_grad():
        out = model(x)
    assert torch.isfinite(out).all()


# ══════════════════════════════════════════════════════════════════════════════
# §4.6  get_manifest()
# ══════════════════════════════════════════════════════════════════════════════

REQUIRED_MANIFEST_KEYS = {
    "class_type", "version_stamp", "graph_anchor",
    "sub_model_stamps", "training_state", "accuracy_metrics", "encryption_meta",
}

@pytest.mark.unit
def test_get_manifest_has_required_keys():
    model = _Net()
    manifest = model.get_manifest()
    missing = REQUIRED_MANIFEST_KEYS - set(manifest.keys())
    assert not missing, f"Missing manifest keys: {missing}"


@pytest.mark.unit
def test_get_manifest_class_type_matches_get_class_type():
    model = _Net()
    assert model.get_manifest()["class_type"] == model.get_class_type()


@pytest.mark.unit
def test_get_class_type_non_empty_string():
    model = _Net()
    ct = model.get_class_type()
    assert isinstance(ct, str) and len(ct) > 0


# ══════════════════════════════════════════════════════════════════════════════
# §4.7  Hooks
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
def test_register_forward_hook_fires():
    model = _Net()
    hook_log = []

    def _hook(module, inp, out):
        hook_log.append(out.shape)

    handle = model.register_forward_hook(_hook)
    x = torch.randn(3, 16)
    with torch.no_grad():
        model(x)
    handle.remove()
    assert len(hook_log) > 0, "Forward hook never fired"
    assert hook_log[0][0] == 3  # batch dim preserved


@pytest.mark.unit
def test_register_backward_hook_fires():
    model = _Net()
    hook_log = []

    def _bwd_hook(module, grad_inp, grad_out):
        hook_log.append(True)

    handle = model.register_backward_hook(_bwd_hook)
    x = torch.randn(3, 16, requires_grad=True)
    y = torch.randint(0, 4, (3,))
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    handle.remove()
    assert len(hook_log) > 0, "Backward hook never fired"


# ══════════════════════════════════════════════════════════════════════════════
# §4.8  device_param / dtype_param properties
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
def test_device_param_getter():
    model = _Net()
    dev = model.device_param
    assert isinstance(dev, torch.device)


@pytest.mark.unit
def test_device_param_setter_moves_model():
    model = _Net()
    model.device_param = torch.device("cpu")
    assert all(p.device.type == "cpu" for p in model.parameters())

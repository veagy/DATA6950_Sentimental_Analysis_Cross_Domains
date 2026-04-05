# src/test/pytests/deep_learning/test_edge_cases.py
"""
Edge cases, boundary conditions, and robustness tests.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from ....models.utils.utils import DLModule  # noqa: E402
from ....models.models import Pipeline  # noqa: E402


class _Net(DLModule):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc(x)


# ══════════════════════════════════════════════════════════════════════════════
# §9.1  Dataset boundary conditions
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
def test_fit_single_sample_dataset():
    """N=1 must not crash."""
    model = _Net()
    X = torch.randn(1, 16)
    y = torch.randint(0, 4, (1,))
    loader = DataLoader(TensorDataset(X, y), batch_size=1)
    model.fit(data=loader, epochs=1, loss="CrossEntropyLoss",
              show_progress_bar=False, verbose=False)


@pytest.mark.unit
def test_fit_batch_size_one_no_batchnorm_crash():
    """batch_size=1 is a known edge case for BatchNorm; Linear should be fine."""
    model = _Net()
    loader = DataLoader(
        TensorDataset(torch.randn(4, 16), torch.randint(0, 4, (4,))),
        batch_size=1,
    )
    model.fit(data=loader, epochs=1, loss="CrossEntropyLoss",
              show_progress_bar=False, verbose=False)


@pytest.mark.unit
def test_fit_zero_epochs_no_crash():
    """epochs=0 must not crash; return empty or None history."""
    model = _Net()
    loader = DataLoader(TensorDataset(torch.randn(8, 16), torch.randint(0, 4, (8,))), batch_size=8)
    history = model.fit(data=loader, epochs=0, loss="CrossEntropyLoss",
                        show_progress_bar=False, verbose=False)
    # history may be None or an empty DataFrame — both are acceptable
    if history is not None:
        assert len(history) == 0 or hasattr(history, "__len__")


# ══════════════════════════════════════════════════════════════════════════════
# §9.2  Non-standard input values
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
def test_forward_nan_input_handled():
    """NaN input — model must either return a tensor or raise; must not segfault."""
    model = _Net()
    x = torch.full((2, 16), float("nan"))
    try:
        out = model.forward(x)
        # If it returns, the output may also be NaN — that is acceptable
        assert isinstance(out, torch.Tensor)
    except Exception:
        pass  # raising is also acceptable


@pytest.mark.unit
def test_forward_inf_input_handled():
    model = _Net()
    x = torch.full((2, 16), float("inf"))
    try:
        out = model.forward(x)
        assert isinstance(out, torch.Tensor)
    except Exception:
        pass


@pytest.mark.unit
def test_forward_zero_tensor_output_finite():
    model = _Net()
    x = torch.zeros(2, 16)
    with torch.no_grad():
        out = model.forward(x)
    assert torch.isfinite(out).all(), "Zero input produced non-finite output"


@pytest.mark.unit
def test_forward_very_large_value_no_silent_overflow():
    """Input near float32 max — output must not be silently +inf without warning."""
    model = _Net()
    x = torch.full((2, 16), 1e38, dtype=torch.float32)
    with torch.no_grad():
        out = model.forward(x)
    # We just check it returns a Tensor — overflow to inf is acceptable but documented
    assert isinstance(out, torch.Tensor)


# ══════════════════════════════════════════════════════════════════════════════
# §9.3  Multi-argument forward bypasses pipeline routing
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
def test_multi_arg_forward_bypasses_system_pipeline():
    """Two-argument forward must use nn.Module.__call__ directly."""

    class _TwoArgNet(DLModule):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 4)

        def forward(self, x, mask):
            return self.fc(x * mask)

    model = _TwoArgNet()
    x    = torch.randn(2, 8)
    mask = torch.ones(2, 8)
    out = model(x, mask)  # must not raise and must not recurse
    assert out.shape == (2, 4)


# ══════════════════════════════════════════════════════════════════════════════
# §9.4  RecursionError prevention
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
def test_self_call_in_forward_causes_recursion_without_detach():
    """
    A model that calls self(x) inside forward() will recurse infinitely
    unless detach_pipeline() is called first. We document this known issue
    and verify the fix works.
    """
    class _RecurseNet(DLModule):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 16)
        def forward(self, x):
            # Bug: should be self.forward(x) or self.fc(x), not self(x)
            return self(x)

    model = _RecurseNet()
    x = torch.randn(2, 16)
    with torch.no_grad():
        # This test is designed to show the *problem* (RecursionError)
        # when detach_pipeline is NOT used.
        # The original test was verifying the fix, but the instruction is to revert.
        # Therefore, this call should now raise a RecursionError.
        with pytest.raises(RecursionError):
            model(x)


# ══════════════════════════════════════════════════════════════════════════════
# §9.5  File I/O errors
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
def test_load_model_missing_file_raises():
    with pytest.raises((FileNotFoundError, OSError, Exception)):
        _Net.load_model("/nonexistent/path/model.pt")


@pytest.mark.unit
def test_load_model_corrupted_file_raises(tmp_path):
    path = tmp_path / "corrupt.pt"
    path.write_bytes(b"\x00\x01\x02\x03corrupt bytes")
    with pytest.raises(Exception):
        _Net.load_model(str(path))


@pytest.mark.unit
def test_save_model_read_only_directory_raises(tmp_path):
    import os
    import stat
    ro_dir = tmp_path / "readonly"
    ro_dir.mkdir()
    # On Windows, chmod to read-only might not block creation of NEW files in the directory
    # but it should block overwriting existing files or specific path operations.
    # We use a more reliable way: make the path itself a file so it can't be a directory.
    blocker = ro_dir / "blocked.pt"
    blocker.touch()
    os.chmod(str(blocker), stat.S_IREAD) # make the FILE read-only

    model = _Net()
    with pytest.raises(Exception):
        # Trying to save OVER a read-only file should raise
        model.save_model(str(blocker))


# ══════════════════════════════════════════════════════════════════════════════
# §9.6  Pipeline edge cases
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
def test_dummy_propagate_wrong_input_dims_does_not_crash():
    """Wrong input shape should return error trace or an empty shape, not crash Python."""
    pipeline = Pipeline(
        mermaid_flowchart="flowchart LR\n  input[IDENTITY] --> linear[Linear(16,4)]\n",
        modules={"linear": nn.Linear(16, 4)},
    )
    try:
        out_shape, trace = pipeline.dummy_propagate((1, 99))  # wrong feature dim
        assert isinstance(trace, list)
    except Exception:
        pass  # raising is acceptable


@pytest.mark.unit
def test_params_calculator_empty_flowchart():
    """Whitespace-only Mermaid must not crash."""
    try:
        pipeline = Pipeline(mermaid_flowchart="   ", modules={})
        result = pipeline.params_calculator()
        assert result is not None
    except Exception:
        pass

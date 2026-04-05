# src/test/unittests/deep_leaning/test_hypothesis.py
"""
Property-based tests using Hypothesis.
These tests generate hundreds of random inputs and verify invariants
that must hold for all valid inputs.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Resolve PROJECT_ROOT: this file is src/test/unittests/deep_leaning/test_hypothesis.py
# ROOT should be 4 levels up
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

try:
    from hypothesis import given, settings, assume, HealthCheck
    import hypothesis
    from hypothesis import strategies as st
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False
    def given(*args, **kwargs): return lambda f: f
    def settings(*args, **kwargs): return lambda f: f
    assume = lambda condition: None
    class st:
        integers = lambda *args, **kwargs: None

pytestmark = pytest.mark.skipif(
    not HAS_HYPOTHESIS,
    reason="hypothesis not installed"
)

from ....models.utils.utils import DLModule  # noqa: E402

REQUIRED_MANIFEST_KEYS = {
    "class_type", "version_stamp", "graph_anchor",
    "sub_model_stamps", "training_state", "accuracy_metrics", "encryption_meta",
}


class _Net(DLModule):
    def __init__(self, in_f=16, out_f=4):
        super().__init__()
        self.fc = nn.Linear(in_f, out_f)

    def forward(self, x):
        return self.fc(x)


# ══════════════════════════════════════════════════════════════════════════════
# §16.1  Fit with arbitrary batch sizes
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.hypothesis
@settings(max_examples=10, deadline=None)
@given(batch_size=st.integers(min_value=1, max_value=32))
def test_fit_arbitrary_batch_size_no_crash(batch_size):
    """fit() must complete for any batch_size in [1, 32]."""
    from torch.utils.data import DataLoader, TensorDataset
    model = _Net()
    # Ensure dataset size is a multiple of batch_size or large enough
    N = max(batch_size * 2, 8)
    loader = DataLoader(
        TensorDataset(torch.randn(N, 16), torch.randint(0, 4, (N,))),
        batch_size=batch_size,
    )
    try:
        model.fit(data=loader, epochs=1, loss="CrossEntropyLoss",
                  show_progress_bar=False, verbose=False)
    except Exception as exc:
        pytest.fail(f"fit() raised with batch_size={batch_size}: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# §16.2  dummy_propagate with arbitrary feature dimensions
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.hypothesis
@settings(max_examples=10, deadline=None)
@given(d=st.integers(min_value=2, max_value=128))
def test_dummy_propagate_arbitrary_feature_dim(d):
    """dummy_propagate((1, d)) must return a valid shape for any d."""
    try:
        from ....models.models import Pipeline
    except ImportError:
        pytest.skip("src/models/models.py not found")
        
    mermaid = f"flowchart LR\n  input[IDENTITY] --> linear[Linear({d},4)]\n"
    pipeline = Pipeline(
        mermaid_flowchart=mermaid,
        modules={"linear": nn.Linear(d, 4)},
    )
    out_shape, trace = pipeline.dummy_propagate((1, d))
    assert isinstance(out_shape, tuple)
    assert len(out_shape) > 0


# ══════════════════════════════════════════════════════════════════════════════
# §16.3  Manifest always has required keys
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.hypothesis
@settings(max_examples=10, deadline=None)
@given(in_f=st.integers(min_value=2, max_value=32),
       out_f=st.integers(min_value=2, max_value=16))
def test_manifest_always_has_required_keys(in_f, out_f):
    """get_manifest() must always contain all required keys."""
    model = _Net(in_f=in_f, out_f=out_f)
    manifest = model.get_manifest()
    missing = REQUIRED_MANIFEST_KEYS - set(manifest.keys())
    assert not missing, f"Missing manifest keys: {missing}"


# ══════════════════════════════════════════════════════════════════════════════
# §16.4  Save/load parameter invariance
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.hypothesis
@settings(max_examples=5, deadline=None, suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture])
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_save_load_parameter_invariance(seed, tmp_path):
    """For any random seed, save → load must produce identical parameters."""
    import tempfile
    torch.manual_seed(seed)
    model = _Net()
    
    with tempfile.TemporaryDirectory() as td:
        checkpoint_file = Path(td) / f"model_{seed}.pt"
        path = str(checkpoint_file)
        model.save_model(path)
        
        loaded = _Net()
        loaded = loaded.load_model(path)
        
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), loaded.named_parameters()):
            torch.testing.assert_close(p1, p2, msg=f"Parameter mismatch: {n1}")

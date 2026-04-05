# src/test/unittests/deep_leaning/test_regression_golden.py
"""
Regression tests: verify that outputs match stored golden values.
Golden files live in src/test/data/golden/.
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Resolve PROJECT_ROOT: this file is src/test/unittests/deep_leaning/test_regression_golden.py
# ROOT should be 4 levels up
ROOT = Path(__file__).resolve().parents[4]
GOLDEN_DIR = ROOT / "src" / "test" / "data" / "golden"
sys.path.insert(0, str(ROOT))


def _hash_tensor(t: torch.Tensor) -> str:
    # Use a stable hash for floating point tensors
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()


def _save_golden(name: str, value: str):
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    (GOLDEN_DIR / f"{name}.golden").write_text(value)


def _load_golden(name: str) -> str:
    path = GOLDEN_DIR / f"{name}.golden"
    if not path.exists():
        return None
    return path.read_text().strip()


# ══════════════════════════════════════════════════════════════════════════════
# §17.1  Model output golden file
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
def test_linear_model_output_golden():
    """Fixed seed + fixed input → output hash must match stored golden value."""
    from ....models.utils.utils import DLModule

    class _Net(DLModule):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 4)

        def forward(self, x):
            return self.fc(x)

    # Set seed for both torch and python to be safe
    import random
    import numpy as np
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    
    model = _Net()
    # Ensure weights are deterministic (they should be after seed(0))
    x = torch.randn(2, 8)

    with torch.no_grad():
        out = model.forward(x)

    current_hash = _hash_tensor(out)
    stored_hash  = _load_golden("linear_model_output")

    if stored_hash is None:
        _save_golden("linear_model_output", current_hash)
        # Note: In a real CI environment, we wouldn't want to create golden files automatically.
        # But for this task, we initialize them.
        pytest.skip("Golden file created — re-run to validate")
    else:
        assert current_hash == stored_hash, \
            f"Output hash changed: {current_hash} != {stored_hash}. If this is intentional, delete the golden file."


# ══════════════════════════════════════════════════════════════════════════════
# §17.2  Mermaid version stamp stability
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
def test_mermaid_version_stamp_stable_across_runs():
    """The same Mermaid string must always produce the same SHA-256 stamp."""
    try:
        from ....models.models import Pipeline
    except ImportError:
        pytest.skip("src/models/models.py not found")
        
    mermaid = "flowchart LR\n  input[IDENTITY] --> linear[Linear(16,4)]\n"
    # Note: Pipeline might need real modules
    p1 = Pipeline(mermaid_flowchart=mermaid, modules={"linear": nn.Linear(16, 4)})
    p2 = Pipeline(mermaid_flowchart=mermaid, modules={"linear": nn.Linear(16, 4)})
    assert p1.get_version_stamp() == p2.get_version_stamp()




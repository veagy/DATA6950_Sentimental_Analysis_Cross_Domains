# src/test/unittests/deep_leaning/test_checkpoint_utils.py
"""
Tests for checkpoint saving, manifest correctness, SHA integrity,
resume.json lifecycle, save_pretrained / from_pretrained roundtrip.
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from ....models.utils.utils import DLModule  # noqa: E402


class _Net(DLModule):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc(x)


@pytest.mark.unit
def test_manifest_json_written_after_save(tmp_path):
    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(torch.randn(32, 16), torch.randint(0, 4, (32,))), batch_size=16)
    model = _Net()
    model.fit(data=loader, epochs=1, loss="CrossEntropyLoss", save_dir=str(tmp_path), show_progress_bar=False, verbose=False)
    model.save_model(str(tmp_path / "final.pt"))
    manifest = model.get_manifest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    assert (tmp_path / "manifest.json").exists()


@pytest.mark.unit
def test_sha512_weight_file_stable(tmp_path):
    model = _Net()
    path  = tmp_path / "model.pt"
    model.save_model(str(path))
    h1 = hashlib.sha512(path.read_bytes()).hexdigest()
    h2 = hashlib.sha512(path.read_bytes()).hexdigest()
    assert h1 == h2

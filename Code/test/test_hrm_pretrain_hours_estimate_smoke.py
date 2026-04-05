"""Smoke: wall-time helper exits 0 (no full parquet required)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def test_hrm_pretrain_hours_estimate_rows_only() -> None:
    tool = _REPO / "Code" / "thesis" / "tools" / "hrm_pretrain_hours_estimate.py"
    r = subprocess.run(
        [
            sys.executable,
            str(tool),
            "--rows",
            "9500000",
            "--gpus",
            "2",
            "--batch",
            "64",
            "--t-step",
            "0.5",
            "--epochs",
            "2",
        ],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    assert "steps" in r.stdout.lower() or "epoch" in r.stdout.lower()

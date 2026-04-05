"""
Checkpoint manager: saves and restores model weights with manifest + SHA-512 hash verification.
Integrates with sentinel's existing checkpoint tree under PATH_CHECKPOINTS.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import torch


class CheckpointManager:
    """
    Manages checkpoint saves and resumes.

    Directory layout::

        PATH_CHECKPOINTS/
            manifest.json          # top-level manifest with all version stamps
            <run_id>/
                model_epoch_N.pth  # weight file
                resume.json        # epoch, metrics, sha512, timestamp
    """

    def __init__(self, config: dict):
        import os
        self.ckpt_dir = Path("./checkpoints")
        # SENTINEL_RUN_ID env var allows per-model checkpoint dirs (e.g. set to model class name)
        self.run_id = (
            os.environ.get("SENTINEL_RUN_ID")
            or config.get("training", {}).get("run_id", "default")
        )
        self.run_dir = self.ckpt_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model: torch.nn.Module, epoch: int, metrics: dict | None = None) -> Path:
        """Atomic save: write to temp, compute SHA-512, rename into place, update manifest."""
        filename = f"model_epoch_{epoch}.pth"
        dest = self.run_dir / filename
        metrics = metrics or {}

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pth", dir=self.run_dir) as tmp:
            torch.save(model.state_dict(), tmp.name)
            tmp_path = Path(tmp.name)

        sha512 = hashlib.sha512(tmp_path.read_bytes()).hexdigest()
        shutil.move(str(tmp_path), str(dest))

        resume = {
            "epoch": epoch,
            "metrics": metrics,
            "file": filename,
            "sha512": sha512,
            "saved_at": datetime.utcnow().isoformat() + "Z",
        }
        (self.run_dir / "resume.json").write_text(json.dumps(resume, indent=2))
        self._update_manifest(filename, sha512, epoch, metrics)
        return dest

    def resume(self, model: torch.nn.Module) -> int:
        """Load latest checkpoint and verify integrity. Return next epoch number."""
        resume_path = self.run_dir / "resume.json"
        if not resume_path.exists():
            return 0

        meta = json.loads(resume_path.read_text())
        weight_file = self.run_dir / meta["file"]

        if not weight_file.exists():
            raise FileNotFoundError(f"Checkpoint file missing: {weight_file}")

        actual_hash = hashlib.sha512(weight_file.read_bytes()).hexdigest()
        if actual_hash != meta["sha512"]:
            raise ValueError(
                f"SHA-512 mismatch for {weight_file}. "
                "The file may be corrupted or tampered."
            )

        model.load_state_dict(torch.load(str(weight_file), map_location="cpu"))
        return meta["epoch"] + 1

    def _update_manifest(self, filename: str, sha512: str, epoch: int, metrics: dict) -> None:
        manifest_path = self.ckpt_dir / "manifest.json"
        manifest: dict = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
            except json.JSONDecodeError:
                pass
        stamps: list = manifest.setdefault("version_stamps", [])
        stamps.append(
            {
                "run_id": self.run_id,
                "file": filename,
                "sha512": sha512,
                "epoch": epoch,
                "metrics": metrics,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2))

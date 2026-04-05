"""
Periodic training snapshots under a temp (or custom) directory for resume after crash/SIGTERM.

Final deliverable weights still go to checkpoints/ as .safetensors per docs/task.txt.
Resume bundles use model.safetensors + train_state.pt (optimizer, AMP scaler) + meta.json.
"""
from __future__ import annotations

import json
import os
import re
import signal
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from Code.thesis.common.checkpoint_io import load_safetensors_state, save_safetensors
from Code.thesis.common.distributed import is_dist_avail, is_main_process


def default_resume_temp_root(repo_root: Path | None = None) -> Path:
    if os.environ.get("THESIS_RESUME_TEMP"):
        return Path(os.environ["THESIS_RESUME_TEMP"]).resolve()
    if repo_root is not None:
        p = (Path(repo_root) / "logs" / "resume").resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    base = os.environ.get("TMPDIR") or os.environ.get("TEMP") or "/tmp"
    return Path(base).resolve()


def sanitize_slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)[:220]


def _class_names_resume_compatible(a: str, b: str, segment: str) -> bool:
    """Live resume may have been saved under a different wrapper name (same weights)."""
    if a == b:
        return True
    if segment == "hrm_supervised":
        if {a, b} == {"HierarchicalReasoningModel", "HRMClassifierWrapper"}:
            return True
    return False


def resume_run_dir(root: Path, cfg_stem: str, dataset_stem: str, n_labels: int) -> Path:
    slug = sanitize_slug(f"{cfg_stem}__{dataset_stem}__{n_labels}l")
    return (root / slug).resolve()


@dataclass
class ResumeMeta:
    version: int = 2
    cfg_path: str = ""
    dataset_stem: str = ""
    n_labels: int = 0
    phase: str = ""
    segment: str = ""
    epoch_next: int = 0
    epochs_segment: int = 0
    lr: float = 0.0
    class_name: str = ""
    head_only: Optional[bool] = None
    # Per-rank optimizer steps finished in epoch ``epoch_next`` when mid-epoch; 0 after a full epoch save.
    steps_completed_in_epoch: int = 0

    def matches_run(self, other: "ResumeMeta") -> bool:
        """Match training identity. ``epochs_segment`` is excluded so resume works if you change
        ``--epochs_finetune`` (e.g. 3 on disk vs 2 on CLI); ``try_restore_training`` still requires
        ``disk.epoch_next < expected.epochs_segment`` so finished runs do not reload."""
        return (
            self.cfg_path == other.cfg_path
            and self.dataset_stem == other.dataset_stem
            and self.n_labels == other.n_labels
            and self.phase == other.phase
            and self.segment == other.segment
            and abs(self.lr - other.lr) < 1e-12
            and _class_names_resume_compatible(self.class_name, other.class_name, self.segment)
            and self.head_only == other.head_only
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "ResumeMeta":
        d = json.loads(s)
        if d.get("head_only") is not None:
            d["head_only"] = bool(d["head_only"])
        if "steps_completed_in_epoch" in d:
            d["steps_completed_in_epoch"] = int(d["steps_completed_in_epoch"])
        names = {f.name for f in fields(cls)}
        return cls(**{k: d[k] for k in names if k in d})


def cleanup_stale_partials(resume_dir: Path) -> None:
    """Remove leftover ``*.part`` staging files from an interrupted save."""
    if not resume_dir.is_dir():
        return
    for p in resume_dir.glob("*.part"):
        try:
            p.unlink()
        except OSError:
            pass


def _fsync_file(path: Path) -> None:
    try:
        with open(path, "rb+") as f:
            f.flush()
            os.fsync(f.fileno())
    except OSError:
        pass


def read_resume_meta(resume_dir: Path) -> Optional[ResumeMeta]:
    meta_p = resume_dir / "meta.json"
    if not meta_p.is_file():
        return None
    try:
        return ResumeMeta.from_json(meta_p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, TypeError, KeyError):
        return None


class InterruptSave:
    def __init__(self) -> None:
        self.requested = False

    def _handler(self, signum: int, frame: Any) -> None:
        self.requested = True

    def install(self) -> None:
        signal.signal(signal.SIGTERM, self._handler)
        signal.signal(signal.SIGINT, self._handler)


class LiveResumeDir:
    """Rank 0 writes; all ranks barrier around IO."""

    def __init__(self, directory: Path, use_ddp: bool) -> None:
        self.dir = directory
        self.use_ddp = use_ddp

    def _barrier(self) -> None:
        if self.use_ddp and is_dist_avail() and dist.is_initialized():
            dist.barrier()

    def model_path(self) -> Path:
        return self.dir / "model.safetensors"

    def state_path(self) -> Path:
        return self.dir / "train_state.pt"

    def meta_path(self) -> Path:
        return self.dir / "meta.json"

    def is_complete(self) -> bool:
        return self.meta_path().is_file() and self.model_path().is_file() and self.state_path().is_file()

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[Any],
        meta: ResumeMeta,
        *,
        use_ddp_module: bool,
    ) -> None:
        self._barrier()
        if is_main_process():
            self.dir.mkdir(parents=True, exist_ok=True)
            cleanup_stale_partials(self.dir)
            mp = self.dir / "model.safetensors.part"
            sp = self.dir / "train_state.pt.part"
            jp = self.dir / "meta.json.part"
            raw_sd = model.module.state_dict() if use_ddp_module else model.state_dict()
            save_safetensors(raw_sd, mp)
            payload = {
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
            }
            torch.save(payload, sp)
            jp.write_text(meta.to_json(), encoding="utf-8")
            _fsync_file(mp)
            _fsync_file(sp)
            _fsync_file(jp)
            os.replace(mp, self.model_path())
            os.replace(sp, self.state_path())
            os.replace(jp, self.meta_path())
        self._barrier()

    def load_weights(self, model: nn.Module, *, use_ddp_module: bool, device: torch.device) -> None:
        state = load_safetensors_state(self.model_path(), map_location=device)
        if "_empty" in state and len(state) == 1:
            return
        state.pop("_empty", None)
        target = model.module if use_ddp_module else model
        target.load_state_dict(state, strict=True)

    def load_optimizer_scaler(
        self,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[Any],
    ) -> None:
        blob = torch.load(self.state_path(), map_location="cpu", weights_only=False)
        optimizer.load_state_dict(blob["optimizer"])
        if scaler is not None and blob.get("scaler") is not None:
            scaler.load_state_dict(blob["scaler"])

    def try_restore_training(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[Any],
        expected: ResumeMeta,
        *,
        use_ddp_module: bool,
        device: torch.device,
    ) -> Optional[tuple[int, int]]:
        """Return ``(epoch_next, skip_batches)`` for ``range(epoch_next, epochs)`` and batch skip, or None."""
        if not self.is_complete():
            return None
        try:
            disk = ResumeMeta.from_json(self.meta_path().read_text(encoding="utf-8"))
        except (json.JSONDecodeError, TypeError, KeyError):
            return None
        if not disk.matches_run(expected):
            return None
        if disk.epoch_next >= expected.epochs_segment:
            return None
        self.load_weights(model, use_ddp_module=use_ddp_module, device=device)
        self.load_optimizer_scaler(optimizer, scaler)
        if is_main_process():
            print(
                f"[resume] Restored live checkpoint from {self.dir}: "
                f"epoch_next={disk.epoch_next} skip_batches={disk.steps_completed_in_epoch} "
                f"(disk epochs_segment={disk.epochs_segment}, this run epochs_finetune={expected.epochs_segment})",
                flush=True,
            )
        return (disk.epoch_next, disk.steps_completed_in_epoch)

"""LiveResumeDir atomic save / restore (CPU, tiny module)."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from Code.thesis.common.resume_checkpoint import (
    LiveResumeDir,
    ResumeMeta,
)


@pytest.fixture
def tiny_meta(tmp_path: Path) -> ResumeMeta:
    return ResumeMeta(
        cfg_path=str(tmp_path / "cfg.json"),
        dataset_stem="dummy",
        n_labels=0,
        phase="pretrain",
        segment="hrm_mlm",
        epoch_next=0,
        epochs_segment=1,
        lr=1e-4,
        class_name="HierarchicalReasoningModel",
        head_only=None,
        steps_completed_in_epoch=0,
    )


def test_live_resume_save_restore_roundtrip(tmp_path: Path, tiny_meta: ResumeMeta) -> None:
    d = tmp_path / "resume"
    live = LiveResumeDir(d, use_ddp=False)
    m = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    x = torch.randn(2, 4)
    loss = m(x).sum()
    loss.backward()
    opt.step()

    live.save(m, opt, scaler, tiny_meta, use_ddp_module=False)
    assert live.is_complete()

    m2 = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    scaler2 = torch.cuda.amp.GradScaler(enabled=False)

    r = live.try_restore_training(
        m2,
        opt2,
        scaler2,
        tiny_meta,
        use_ddp_module=False,
        device=torch.device("cpu"),
    )
    assert r is not None
    epoch_next, skip = r
    assert epoch_next == 0 and skip == 0

    torch.testing.assert_close(m(x), m2(x))


def test_live_resume_meta_mismatch_returns_none(tmp_path: Path, tiny_meta: ResumeMeta) -> None:
    d = tmp_path / "resume2"
    live = LiveResumeDir(d, use_ddp=False)
    m = nn.Linear(3, 2)
    opt = torch.optim.AdamW(m.parameters())
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    live.save(m, opt, scaler, tiny_meta, use_ddp_module=False)

    m2 = nn.Linear(3, 2)
    opt2 = torch.optim.AdamW(m2.parameters())
    scaler2 = torch.cuda.amp.GradScaler(enabled=False)
    bad = replace(tiny_meta, lr=9.99e-3)

    r = live.try_restore_training(
        m2,
        opt2,
        scaler2,
        bad,
        use_ddp_module=False,
        device=torch.device("cpu"),
    )
    assert r is None

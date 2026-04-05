"""
HRM encoder-only MLM: short train_single subprocess with periodic resume saves.

Creates a tiny ``all-data.parquet`` under a temp ``data_root`` (no DUMMY repo state).
Requires HF tokenizer cache when offline env is set.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def tiny_all_data_parquet(tmp_path_factory: pytest.TempPathFactory) -> Path:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    root = tmp_path_factory.mktemp("hrm_integ_data")
    proc = root / "processed"
    proc.mkdir(parents=True)
    # Short repeated text — enough rows for several optimizer steps at batch 1
    texts = [f"sample tweet number {i} for hrm mlm pretrain test. " * 3 for i in range(48)]
    df = pd.DataFrame({"text": texts, "sentiment_value": [0] * len(texts)})
    pq = proc / "all-data.parquet"
    df.to_parquet(pq, index=False)
    return root


def test_hrm_encoder_pretrain_periodic_saves_and_final_ckpt(tiny_all_data_parquet: Path, tmp_path: Path) -> None:
    """``save_every_steps=2`` + full epoch; expect ``checkpoints/pretrain/all-data/*.safetensors``."""
    ck = tmp_path / "ckpt"
    log = tmp_path / "logs"
    res = tmp_path / "resume"
    for p in (ck, log, res):
        p.mkdir(parents=True)

    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    if "CUDA_VISIBLE_DEVICES" not in env:
        env["CUDA_VISIBLE_DEVICES"] = ""

    cmd = [
        sys.executable,
        str(_REPO / "Code" / "thesis" / "train" / "train_single.py"),
        "--config",
        str(_REPO / "Code" / "thesis" / "config" / "hrm" / "E_HRM1_4Level.json"),
        "--dataset_stem",
        "all-data",
        "--data_root",
        str(tiny_all_data_parquet),
        "--checkpoint_root",
        str(ck),
        "--log_dir",
        str(log),
        "--resume_temp_root",
        str(res),
        "--phase",
        "pretrain",
        "--pretrain_text_source",
        "all_data_parquet",
        "--epochs_pretrain",
        "1",
        "--epochs_finetune",
        "0",
        "--max_samples",
        "32",
        "--batch_size",
        "1",
        "--lr",
        "3e-5",
        "--num_workers",
        "0",
        "--gc_every",
        "0",
        "--save_every_steps",
        "2",
        "--min_save_interval_sec",
        "1",
    ]
    r = subprocess.run(cmd, cwd=str(_REPO), env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout + "\n" + r.stderr

    pre = ck / "pretrain" / "all-data" / "E_HRM1_4Level.safetensors"
    assert pre.is_file(), f"missing final encoder ckpt {pre}"
    assert pre.stat().st_size > 10_000, "checkpoint suspiciously small"

    slug_dirs = list(res.glob("*"))
    assert not any(
        (d / "meta.json").is_file() for d in slug_dirs if d.is_dir()
    ), "live resume should be cleared after successful pretrain"


def test_hrm_encoder_pretrain_resume_after_sigterm(tiny_all_data_parquet: Path, tmp_path: Path) -> None:
    """SIGTERM mid-epoch leaves a resume bundle; second run completes (skip_batches path)."""
    ck = tmp_path / "ckpt_resume"
    log = tmp_path / "logs_resume"
    res = tmp_path / "resume_resume"
    for p in (ck, log, res):
        shutil.rmtree(p, ignore_errors=True)
        p.mkdir(parents=True)

    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    if "CUDA_VISIBLE_DEVICES" not in env:
        env["CUDA_VISIBLE_DEVICES"] = ""

    base_cmd = [
        sys.executable,
        str(_REPO / "Code" / "thesis" / "train" / "train_single.py"),
        "--config",
        str(_REPO / "Code" / "thesis" / "config" / "hrm" / "E_HRM1_4Level.json"),
        "--dataset_stem",
        "all-data",
        "--data_root",
        str(tiny_all_data_parquet),
        "--checkpoint_root",
        str(ck),
        "--log_dir",
        str(log),
        "--resume_temp_root",
        str(res),
        "--phase",
        "pretrain",
        "--pretrain_text_source",
        "all_data_parquet",
        "--epochs_pretrain",
        "1",
        "--epochs_finetune",
        "0",
        "--max_samples",
        "48",
        "--batch_size",
        "1",
        "--lr",
        "3e-5",
        "--num_workers",
        "0",
        "--gc_every",
        "0",
        "--save_every_steps",
        "1",
        "--min_save_interval_sec",
        "1",
    ]

    p1 = subprocess.Popen(
        base_cmd,
        cwd=str(_REPO),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        try:
            p1.wait(timeout=18)
        except subprocess.TimeoutExpired:
            p1.send_signal(signal.SIGTERM)
            try:
                p1.wait(timeout=120)
            except subprocess.TimeoutExpired:
                p1.kill()
                p1.wait()
    finally:
        if p1.stdout:
            p1.stdout.close()

    meta_any = any(
        (d / "meta.json").is_file() for d in res.iterdir() if d.is_dir()
    )
    assert meta_any, "expected a resume bundle after SIGTERM under resume_temp_root"

    r2 = subprocess.run(base_cmd, cwd=str(_REPO), env=env, capture_output=True, text=True, timeout=900)
    assert r2.returncode == 0, r2.stdout + "\n" + r2.stderr

    pre = ck / "pretrain" / "all-data" / "E_HRM1_4Level.safetensors"
    assert pre.is_file(), f"missing final encoder ckpt {pre}"


@pytest.mark.skip(
    reason=(
        "Skipped by default (cheap runs): needs a long partial epoch before SIGTERM. "
        "Resume I/O: test_live_resume_dir.py. Cloud SIGTERM: scripts/smoke_hrm_ddp_sigterm_resume.sh"
    )
)
def test_hrm_sigterm_leaves_resumable_bundle(tiny_all_data_parquet: Path, tmp_path: Path) -> None:
    """First run: SIGTERM mid-epoch should write a resume bundle; second run completes."""
    ck = tmp_path / "ckpt2"
    log = tmp_path / "logs2"
    res = tmp_path / "resume2"
    for p in (ck, log, res):
        shutil.rmtree(p, ignore_errors=True)
        p.mkdir(parents=True)

    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    if "CUDA_VISIBLE_DEVICES" not in env:
        env["CUDA_VISIBLE_DEVICES"] = ""

    base_cmd = [
        sys.executable,
        str(_REPO / "Code" / "thesis" / "train" / "train_single.py"),
        "--config",
        str(_REPO / "Code" / "thesis" / "config" / "hrm" / "E_HRM1_4Level.json"),
        "--dataset_stem",
        "all-data",
        "--data_root",
        str(tiny_all_data_parquet),
        "--checkpoint_root",
        str(ck),
        "--log_dir",
        str(log),
        "--resume_temp_root",
        str(res),
        "--phase",
        "pretrain",
        "--pretrain_text_source",
        "all_data_parquet",
        "--epochs_pretrain",
        "1",
        "--epochs_finetune",
        "0",
        "--max_samples",
        "128",
        "--batch_size",
        "1",
        "--lr",
        "3e-5",
        "--num_workers",
        "0",
        "--gc_every",
        "0",
        "--save_every_steps",
        "50",
        "--min_save_interval_sec",
        "1",
    ]

    p1 = subprocess.Popen(
        base_cmd,
        cwd=str(_REPO),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        try:
            p1.wait(timeout=25)
        except subprocess.TimeoutExpired:
            p1.send_signal(signal.SIGTERM)
            try:
                p1.wait(timeout=120)
            except subprocess.TimeoutExpired:
                p1.kill()
                p1.wait()
    finally:
        if p1.stdout:
            p1.stdout.close()

    meta_any = any(
        (d / "meta.json").is_file() for d in res.iterdir() if d.is_dir()
    )
    assert meta_any, "expected a resume bundle after SIGTERM under resume_temp_root"

    r2 = subprocess.run(base_cmd, cwd=str(_REPO), env=env, capture_output=True, text=True, timeout=900)
    assert r2.returncode == 0, r2.stdout + "\n" + r2.stderr

    pre = ck / "pretrain" / "all-data" / "E_HRM1_4Level.safetensors"
    assert pre.is_file(), f"missing final encoder ckpt {pre}"

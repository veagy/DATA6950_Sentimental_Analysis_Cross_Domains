"""
Unified logging for training scripts.
Integrates with Sentinel logging (COMMAND_HISTORY, PERFORMANCE_PROFILE).
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests

try:
    import torch
except ImportError:
    torch = None


def _get_log_dir() -> Path:
    """Training logs directory."""
    return Path("./logs")


def log_train_event(
    event: str,
    run_id: str = "default",
    epoch: Optional[int] = None,
    batch: Optional[int] = None,
    metrics: Optional[dict] = None,
    message: str = "",
) -> None:
    """
    Log a training event to stdout and optionally to training.log.

    Args:
        event: Event type (e.g. "epoch_end", "train_start", "checkpoint")
        run_id: Training run identifier
        epoch: Current epoch
        batch: Current batch index
        metrics: Dict of metric name -> value
        message: Free-form message
    """
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    parts = [f"[{ts}]", f"[{event}]"]
    if run_id:
        parts.append(f"run={run_id}")
    if epoch is not None:
        parts.append(f"epoch={epoch}")
    if batch is not None:
        parts.append(f"batch={batch}")
    if metrics:
        m_str = " ".join(f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in metrics.items())
        parts.append(m_str)
    if message:
        parts.append(message)
    line = " ".join(parts)
    print(line, flush=True)

    log_dir = _get_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    train_log = log_dir / run_id / "training.log"
    train_log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(train_log, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def get_logger(name: str = "train", level: int = logging.INFO) -> logging.Logger:
    """Return a logger for training modules."""
    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(level)
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        log.addHandler(h)
    return log


def trace(msg: str) -> None:
    """
    Write a line to STABILITY_TRACE.log and print to stdout.
    """
    log_dir = Path("./logs") / "default"
        
    log_path = log_dir / "STABILITY_TRACE.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().isoformat()
    line = f"[{ts}Z] {msg}\n"
    
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)
    except OSError:
        pass
        
    print(msg, flush=True)


def push_batch_loss(
    loss_val: float, 
    step: int,
    endpoint: str = "http://localhost:8000/api/metrics"
) -> None:
    """Push training batch loss telemetry to the dashboard backend."""
    try:
        requests.post(endpoint, json={"loss": loss_val, "step": step}, timeout=0.5)
    except Exception:
        pass  # Silently fail if dashboard backend is unreachable


def push_vram_telemetry(endpoint: str = "http://localhost:8000/api/vram") -> None:
    """Push VRAM snapshot telemetry to the dashboard backend for the Hardware Monitor."""
    if torch is None or not torch.cuda.is_available():
        return
        
    data = []
    for i in range(torch.cuda.device_count()):
        try:
            props = torch.cuda.get_device_properties(i)
            data.append({
                "gpu_id": i,
                "name": props.name,
                "allocated_gb": torch.cuda.memory_allocated(i) / 1e9,
                "reserved_gb": torch.cuda.memory_reserved(i) / 1e9,
                "total_gb": props.total_memory / 1e9,
            })
        except Exception:
            continue
            
    if not data:
        return
        
    try:
        requests.post(endpoint, json={"devices": data}, timeout=1.0)
    except Exception:
        pass  # Silently fail if dashboard backend is unreachable


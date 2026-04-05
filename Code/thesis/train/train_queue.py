"""
Sequential feature-model training queue: CNN/RNN via ``torchrun`` (multi-GPU); optional classical ML
via single-process train_single.py (use --include-ml). Transformers/HRM are not queued here.

Optionally runs **five** ``FeaturePretrainAutoencoder`` jobs on ``transformed/all-data.parquet`` first
(``checkpoints/pretrain/pretrain_{ffnn,cnn,lstm,gru,rnn}.safetensors``), then schedules
``feature_encoder/{2,3}_labels/*.json`` finetune jobs with ``--encoder_pretrain_ckpt`` unless
``--skip-feature-encoder-pretrain`` is set.

With ``--feature-encoder-only`` and ``--include-ml-bc-after-feature-encoder``, appends docs/ml
Tracks B and C jobs (``train_ml_processed_embed_meta.py`` on ``processed/{stem}.parquet``) after
all feature-encoder finetunes. See ``Code/thesis/config/ml_queue/track_*.json``.

Waits optionally for PIDs and/or stable data/transformed/*.parquet mtimes, then runs
each (config, dataset_stem) pair. Checkpoints follow docs/task.txt (2-labels/3-labels + dataset).

By default the merged stem ``all-data`` is excluded from dataset stems; set
``THESIS_QUEUE_INCLUDE_ALL_DATA=1`` to include it.

Progress is persisted atomically to queue_state.json under the run log directory so runs can
resume after termination (--resume-run-dir).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

_REPO = Path(__file__).resolve().parents[3]

QUEUE_STATE_VERSION = 1
QUEUE_STATE_NAME = "queue_state.json"


def _run_subprocess_tee_stdout(
    cmd: list[str],
    cwd: str,
    job_log: Path,
    *,
    preamble: str = "",
) -> int:
    """Run ``cmd``; copy merged stdout+stderr to ``job_log`` and ``sys.stdout`` by chunks (tqdm \\r-safe)."""
    parts: list[str] = []
    if preamble:
        parts.append(preamble if preamble.endswith("\n") else preamble + "\n")
    parts.append(f"CMD: {' '.join(cmd)}\n\n")
    bhdr = "".join(parts).encode("utf-8")
    job_log.parent.mkdir(parents=True, exist_ok=True)
    with open(job_log, "wb") as lf:
        lf.write(bhdr)
        lf.flush()
        sys.stdout.buffer.write(bhdr)
        sys.stdout.buffer.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            bufsize=0,
        )
        assert proc.stdout is not None
        try:
            while True:
                chunk = proc.stdout.read(65536)
                if not chunk:
                    break
                lf.write(chunk)
                lf.flush()
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
        finally:
            proc.stdout.close()
        return int(proc.wait())


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_pids(pids: list[int], poll: float) -> None:
    pids = list(dict.fromkeys(pids))  # unique, order preserved
    while True:
        alive = [p for p in pids if _pid_exists(p)]
        if not alive:
            return
        time.sleep(poll)


def _read_pids_from_files(paths: list[Path]) -> list[int]:
    out: list[int] = []
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                out.append(int(line.split()[0]))
            except ValueError:
                continue
    return out


def _wait_transformed_stable(transformed_dir: Path, stable_seconds: int, poll: float) -> None:
    if stable_seconds <= 0:
        return
    last_sig: tuple[tuple[str, float, int], ...] | None = None
    stable_elapsed = 0.0
    while True:
        if not transformed_dir.is_dir():
            time.sleep(poll)
            continue
        files = sorted(transformed_dir.glob("*.parquet"))
        if not files:
            time.sleep(poll)
            continue
        sig = tuple((f.name, f.stat().st_mtime, f.stat().st_size) for f in files)
        if sig == last_sig:
            stable_elapsed += poll
            if stable_elapsed >= stable_seconds:
                return
        else:
            last_sig = sig
            stable_elapsed = 0.0
        time.sleep(poll)


def _n_labels_from_config_path(cfg: Path) -> int:
    p = str(cfg).replace("\\", "/")
    if "/2_labels/" in p:
        return 2
    if "/3_labels/" in p:
        return 3
    raise ValueError(f"Cannot infer label count from path: {cfg}")


def _collect_cnn_rnn_configs(cfg_root: Path) -> list[Path]:
    out: list[Path] = []
    for sub in ("cnn", "rnn"):
        d = cfg_root / sub
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*.json")):
            norm = str(p).replace("\\", "/")
            if "/moe/" in norm or p.name == "example_experts.json":
                continue
            out.append(p)
    return sorted(out, key=lambda x: str(x))


def _collect_ml_configs(cfg_root: Path) -> list[Path]:
    d = cfg_root / "ml"
    if not d.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(d.rglob("*.json")):
        norm = str(p).replace("\\", "/")
        if "/moe/" in norm or p.name == "example_experts.json":
            continue
        out.append(p)
    return sorted(out, key=lambda x: str(x))


def _collect_feature_encoder_configs(cfg_root: Path) -> list[Path]:
    out: list[Path] = []
    for sub in ("2_labels", "3_labels"):
        d = cfg_root / "feature_encoder" / sub
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.json")):
            out.append(p)
    return sorted(out, key=lambda x: str(x))


def _feature_pretrain_ckpt_path(checkpoint_root: Path, architecture: str) -> Path:
    return checkpoint_root / "pretrain" / f"pretrain_{architecture.lower().strip()}.safetensors"


def _architecture_from_feature_encoder_json(cfg: Path) -> str:
    data = json.loads(cfg.read_text(encoding="utf-8"))
    block = data.get("FeatureEncoderClassifier") or {}
    return str(block.get("architecture", "ffnn")).lower().strip()


def _architecture_from_pretrain_ae_json(cfg: Path) -> str:
    data = json.loads(cfg.read_text(encoding="utf-8"))
    block = data.get("FeaturePretrainAutoencoder") or {}
    return str(block.get("architecture", "")).lower().strip()


def _transformed_stems(data_root: Path) -> list[str]:
    t = data_root / "transformed"
    if not t.is_dir():
        return []
    stems = sorted(p.stem for p in t.glob("*.parquet") if p.is_file())
    inc = os.environ.get("THESIS_QUEUE_INCLUDE_ALL_DATA", "").strip().lower()
    if inc not in ("1", "true", "yes"):
        stems = [s for s in stems if s != "all-data"]
    return stems


def _stems_from_file(path: Path) -> list[str]:
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


def _job_key(cfg: Path, stem: str) -> tuple[str, str]:
    return (str(cfg.resolve()), stem)


def _ml_bc_skip_if_done(checkpoint_root: Path, queue_cfg: Path, stem: str) -> bool:
    """Tracks B/C outputs from ``train_ml_processed_embed_meta.py`` (see docs/ml)."""
    try:
        data = json.loads(queue_cfg.read_text(encoding="utf-8"))
        track = str(data.get("_ml_queue_track", "")).lower()
        n = int(data.get("_ml_queue_n_labels", 0))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False
    d = checkpoint_root / "moe" / "ml_stack" / f"{n}-labels" / stem
    if track == "b":
        return (d / "trackB_DistilBERT_Linear.safetensors").is_file()
    if track == "c":
        return (
            (d / "proc_LogisticRegression_DistilBERT.joblib").is_file()
            and (d / "proc_LinearSVC_DistilBERT.joblib").is_file()
        )
    return False


def _build_train_ml_bc_cmd(
    repo: Path,
    queue_cfg: Path,
    stem: str,
    data_root: Path,
    ckpt_root: Path,
    *,
    default_batch_size: int | None = None,
    default_epochs_b: int | None = None,
) -> list[str]:
    cmd: list[str] = [
        sys.executable,
        str(repo / "Code/thesis/train/train_ml_processed_embed_meta.py"),
        "--queue_config",
        str(queue_cfg.resolve()),
        "--dataset_stem",
        stem,
        "--data_root",
        str(data_root.resolve()),
        "--checkpoint_root",
        str(ckpt_root.resolve()),
    ]
    bs = (os.environ.get("THESIS_ML_BC_BATCH_SIZE") or "").strip()
    if bs:
        cmd.extend(["--batch_size", bs])
    elif default_batch_size is not None:
        cmd.extend(["--batch_size", str(int(default_batch_size))])
    pc = (os.environ.get("THESIS_ML_BC_PARQUET_CHUNK") or "").strip()
    if pc:
        cmd.extend(["--parquet_chunk", pc])
    eb = (os.environ.get("THESIS_ML_BC_EPOCHS_B") or "").strip()
    if eb:
        cmd.extend(["--epochs_b", eb])
    elif default_epochs_b is not None:
        cmd.extend(["--epochs_b", str(int(default_epochs_b))])
    mc = (os.environ.get("THESIS_ML_BC_MAX_SAMPLES_C") or "").strip()
    if mc:
        cmd.extend(["--max_samples_c", mc])
    ec = (os.environ.get("THESIS_ML_BC_ENCODER_CONFIG") or "").strip()
    if ec:
        cmd.extend(["--encoder_config", str(Path(ec).expanduser().resolve())])
    ek = (os.environ.get("THESIS_ML_BC_ENCODER_CKPT") or "").strip()
    if ek:
        cmd.extend(["--encoder_ckpt", str(Path(ek).expanduser().resolve())])
    return cmd


def _final_checkpoint_path(checkpoint_root: Path, cfg: Path, stem: str) -> Path:
    n = _n_labels_from_config_path(cfg)
    return checkpoint_root / f"{n}-labels" / stem / f"{cfg.stem}.safetensors"


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, indent=2)
    fd, tmp_name = tempfile.mkstemp(suffix=".json.tmp", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _load_queue_state(path: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], str | None]:
    if not path.is_file():
        return {}, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}, None
    if data.get("version") != QUEUE_STATE_VERSION:
        return {}, data.get("created")
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for j in data.get("jobs", []):
        try:
            k = (j["config"], j["stem"])
            out[k] = {
                "status": j.get("status", "pending"),
                "returncode": j.get("returncode"),
                "finished_at": j.get("finished_at"),
            }
        except (KeyError, TypeError):
            continue
    return out, data.get("created")


def _build_queue_state_payload(
    jobs: list[tuple[Path, str, str, Optional[Path]]],
    statuses: dict[tuple[str, str], dict[str, Any]],
    created: str,
) -> dict[str, Any]:
    job_rows = []
    for cfg, stem, _kind, _enc in jobs:
        k = _job_key(cfg, stem)
        st = statuses.get(
            k,
            {"status": "pending", "returncode": None, "finished_at": None},
        )
        job_rows.append(
            {
                "config": k[0],
                "stem": k[1],
                "status": st["status"],
                "returncode": st["returncode"],
                "finished_at": st["finished_at"],
            }
        )
    return {
        "version": QUEUE_STATE_VERSION,
        "created": created,
        "jobs": job_rows,
    }


def _torchrun_extra_args_from_env() -> list[str]:
    """Avoid EADDRINUSE across sequential queue jobs: ephemeral port unless user sets THESIS_TORCH_MASTER_PORT."""
    port = (os.environ.get("THESIS_TORCH_MASTER_PORT") or "").strip()
    if not port:
        import socket

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        port = str(int(s.getsockname()[1]))
        s.close()
    return ["--master_port", port]


def _train_single_resume_extra_from_env() -> list[str]:
    """Optional periodic live-resume flags for train_single (see resume_checkpoint / THESIS_RESUME_TEMP)."""
    extra: list[str] = []
    ses = (os.environ.get("THESIS_SAVE_EVERY_STEPS") or "").strip()
    if ses:
        extra.extend(["--save_every_steps", ses])
    sem = (os.environ.get("THESIS_SAVE_EVERY_MINUTES") or "").strip()
    if sem:
        extra.extend(["--save_every_minutes", sem])
    mis = (os.environ.get("THESIS_MIN_SAVE_INTERVAL_SEC") or "").strip()
    if mis:
        extra.extend(["--min_save_interval_sec", mis])
    rtp = (os.environ.get("THESIS_RESUME_TEMP_ROOT") or os.environ.get("THESIS_RESUME_TEMP") or "").strip()
    if rtp:
        extra.extend(["--resume_temp_root", str(Path(rtp).expanduser().resolve())])
    gce = (os.environ.get("THESIS_GC_EVERY") or "").strip()
    if gce:
        extra.extend(["--gc_every", gce])
    ms = (os.environ.get("THESIS_MAX_SAMPLES") or "").strip()
    if ms:
        extra.extend(["--max_samples", ms])
    return extra


def _build_train_2gpu_cmd(
    repo: Path,
    cfg: Path,
    stem: str,
    data_root: Path,
    checkpoint_root: Path,
    log_dir: Path,
    epochs_finetune: int,
    batch_size: int,
    num_workers: int,
    phase: str,
    epochs_pretrain: int,
    *,
    encoder_pretrain_ckpt: Optional[Path] = None,
) -> list[str]:
    py = sys.executable
    nproc = os.environ.get("THESIS_NPROC_PER_NODE", "2")
    train_py = repo / "Code" / "thesis" / "train" / "train_single.py"
    cmd: list[str] = [
        py,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        *_torchrun_extra_args_from_env(),
        str(train_py),
        "--config",
        str(cfg.resolve()),
        "--dataset_stem",
        stem,
        "--data_root",
        str(data_root.resolve()),
        "--checkpoint_root",
        str(checkpoint_root.resolve()),
        "--log_dir",
        str(log_dir.resolve()),
        "--epochs_finetune",
        str(epochs_finetune),
        "--epochs_pretrain",
        str(epochs_pretrain),
        "--batch_size",
        str(batch_size),
        "--num_workers",
        str(num_workers),
        "--phase",
        phase,
    ]
    if encoder_pretrain_ckpt is not None:
        cmd.extend(["--encoder_pretrain_ckpt", str(encoder_pretrain_ckpt.resolve())])
    cmd.extend(_train_single_resume_extra_from_env())
    return cmd


def _build_train_single_cmd(
    repo: Path,
    cfg: Path,
    stem: str,
    data_root: Path,
    checkpoint_root: Path,
    log_dir: Path,
    epochs_finetune: int,
    batch_size: int,
    num_workers: int,
    phase: str,
    epochs_pretrain: int,
    *,
    encoder_pretrain_ckpt: Optional[Path] = None,
) -> list[str]:
    """Single-process train_single (MLModule / sklearn path); do not use with train_2gpu.sh."""
    cmd: list[str] = [
        sys.executable,
        str(repo / "Code" / "thesis" / "train" / "train_single.py"),
        "--config",
        str(cfg.resolve()),
        "--dataset_stem",
        stem,
        "--data_root",
        str(data_root.resolve()),
        "--checkpoint_root",
        str(checkpoint_root.resolve()),
        "--log_dir",
        str(log_dir.resolve()),
        "--epochs_finetune",
        str(epochs_finetune),
        "--epochs_pretrain",
        str(epochs_pretrain),
        "--batch_size",
        str(batch_size),
        "--num_workers",
        str(num_workers),
        "--phase",
        phase,
    ]
    if encoder_pretrain_ckpt is not None:
        cmd.extend(["--encoder_pretrain_ckpt", str(encoder_pretrain_ckpt.resolve())])
    cmd.extend(_train_single_resume_extra_from_env())
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Queue CNN/RNN (2× GPU) and optionally ML (single-GPU) on transformed parquets.",
    )
    ap.add_argument("--data_root", type=Path, default=None, help="Default: <repo>/data")
    ap.add_argument("--checkpoint_root", type=Path, default=None, help="Default: <repo>/checkpoints")
    ap.add_argument("--log_dir", type=Path, default=None, help="Default: <repo>/logs")
    ap.add_argument("--epochs_finetune", type=int, default=8)
    ap.add_argument("--epochs_pretrain", type=int, default=1, help="Unused for CNN/RNN tensor path; passed for CLI parity.")
    ap.add_argument("--batch_size", type=int, default=None, help="Default: THESIS_BATCH_SIZE env or 24")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--phase", default="finetune", choices=("pretrain", "finetune", "all"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-wait", action="store_true", help="Skip PID and stability waits.")
    ap.add_argument("--wait-pid", type=int, action="append", default=[], help="Repeatable; wait until each PID exits.")
    ap.add_argument(
        "--wait-pid-file",
        type=Path,
        action="append",
        default=[],
        help="Repeatable; file with one PID per line (# comments ok).",
    )
    ap.add_argument(
        "--stable-seconds",
        type=int,
        default=0,
        help="After PID wait: require data/transformed/*.parquet mtimes+sizes stable for this many seconds (0=off).",
    )
    ap.add_argument(
        "--stems-file",
        type=Path,
        default=None,
        help="If set, only these dataset stems (one per line); must still have transformed parquet.",
    )
    ap.add_argument("--poll-seconds", type=float, default=10.0, help="PID wait poll interval.")
    ap.add_argument(
        "--resume-run-dir",
        type=Path,
        default=None,
        help="Continue a prior run: use this logs/queue_cnn_rnn_* directory; skip jobs marked ok or with existing checkpoint.",
    )
    ap.add_argument(
        "--no-skip-if-checkpoint",
        action="store_true",
        help="Run jobs even when final .safetensors already exists (default is to skip).",
    )
    ap.add_argument(
        "--include-ml",
        action="store_true",
        help="After CNN/RNN jobs, run Code/thesis/config/ml/*.json via single-process train_single.py.",
    )
    ap.add_argument(
        "--skip-feature-encoder-pretrain",
        action="store_true",
        help="Do not run five FeaturePretrainAutoencoder jobs on transformed/all-data.parquet before the queue.",
    )
    ap.add_argument(
        "--feature-encoder-only",
        action="store_true",
        help="Only queue Code/thesis/config/feature_encoder/**/*.json (100-D FeatEnc CNN/LSTM/GRU/RNN/FFN); skip cnn/, rnn/, and ml/.",
    )
    ap.add_argument(
        "--include-ml-bc-after-feature-encoder",
        action="store_true",
        help=(
            "After feature-encoder jobs, queue docs/ml Tracks B and C (frozen DistilBERT embeddings from "
            "processed parquet + linear head or sklearn LR/LinearSVC); see Code/thesis/config/ml_queue/."
        ),
    )
    args = ap.parse_args()
    skip_if_ckpt = not args.no_skip_if_checkpoint

    if args.include_ml_bc_after_feature_encoder and not args.feature_encoder_only:
        print(
            "ERROR: --include-ml-bc-after-feature-encoder requires --feature-encoder-only "
            "(ML B/C jobs are appended after the feature-encoder queue only).",
            file=sys.stderr,
        )
        return 1

    if args.resume_run_dir is None:
        env_resume = (os.environ.get("THESIS_QUEUE_RESUME_DIR") or os.environ.get("THESIS_RESUME_RUN_DIR") or "").strip()
        if env_resume:
            args.resume_run_dir = Path(env_resume)

    repo = _REPO
    data_root = (args.data_root or (repo / "data")).resolve()
    ckpt_root = (args.checkpoint_root or (repo / "checkpoints")).resolve()
    log_dir = (args.log_dir or (repo / "logs")).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    batch_size = args.batch_size
    if batch_size is None:
        env_bs = os.environ.get("THESIS_BATCH_SIZE")
        batch_size = int(env_bs) if env_bs else 24

    if not args.skip_wait:
        pids = list(args.wait_pid)
        pids.extend(_read_pids_from_files(args.wait_pid_file))
        if pids:
            print(f"[train_queue] Waiting for PIDs to exit: {pids}", flush=True)
            _wait_pids(pids, args.poll_seconds)
            print("[train_queue] All watched PIDs have exited.", flush=True)
        if args.stable_seconds > 0:
            trans = data_root / "transformed"
            print(
                f"[train_queue] Waiting for {trans} parquets stable for {args.stable_seconds}s...",
                flush=True,
            )
            _wait_transformed_stable(trans, args.stable_seconds, min(5.0, args.poll_seconds))
            print("[train_queue] Transformed parquets stable.", flush=True)

    cfg_root = repo / "Code" / "thesis" / "config"
    if args.feature_encoder_only:
        cnn_rnn_cfgs = []
        ml_cfgs = []
    else:
        cnn_rnn_cfgs = _collect_cnn_rnn_configs(cfg_root)
        ml_cfgs = _collect_ml_configs(cfg_root) if args.include_ml else []
    feat_enc_cfgs = _collect_feature_encoder_configs(cfg_root)
    stems = _transformed_stems(data_root)
    if args.stems_file is not None:
        allow = set(_stems_from_file(args.stems_file.resolve()))
        stems = [s for s in stems if s in allow]

    if not args.skip_feature_encoder_pretrain:
        pre_dir = cfg_root / "pretrain" / "2_labels"
        all_data_pq = data_root / "transformed" / "all-data.parquet"
        if pre_dir.is_dir() and all_data_pq.is_file():
            for cfg_pre in sorted(pre_dir.glob("Pretrain_*.json")):
                try:
                    blob = json.loads(cfg_pre.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                if "FeaturePretrainAutoencoder" not in blob:
                    continue
                arch = _architecture_from_pretrain_ae_json(cfg_pre)
                if not arch:
                    continue
                ck_out = _feature_pretrain_ckpt_path(ckpt_root, arch)
                if skip_if_ckpt and ck_out.is_file():
                    print(f"[train_queue] SKIP feature AE pretrain (checkpoint exists) {ck_out}", flush=True)
                    continue
                cmd_pre = _build_train_2gpu_cmd(
                    repo,
                    cfg_pre,
                    "all-data",
                    data_root,
                    ckpt_root,
                    log_dir,
                    0,
                    batch_size,
                    args.num_workers,
                    "pretrain",
                    args.epochs_pretrain,
                )
                print(f"[train_queue] RUN feature AE pretrain ({arch}) {' '.join(cmd_pre)}", flush=True)
                if not args.dry_run:
                    rc_pre = subprocess.run(cmd_pre, cwd=str(repo), check=False)
                    if rc_pre.returncode != 0:
                        print(
                            f"[train_queue] Feature AE pretrain failed rc={rc_pre.returncode} ({cfg_pre.name})",
                            file=sys.stderr,
                            flush=True,
                        )
                        return 1
        elif pre_dir.is_dir():
            print(
                f"[train_queue] Skipping feature AE pretrain: missing {all_data_pq}",
                flush=True,
            )

    jobs: list[tuple[Path, str, str, Optional[Path]]] = []
    for cfg in feat_enc_cfgs:
        try:
            _n_labels_from_config_path(cfg)
        except ValueError:
            continue
        arch = _architecture_from_feature_encoder_json(cfg)
        enc_ckpt = _feature_pretrain_ckpt_path(ckpt_root, arch)
        for stem in stems:
            pq = data_root / "transformed" / f"{stem}.parquet"
            if pq.is_file():
                jobs.append((cfg, stem, "2gpu", enc_ckpt))
    for cfg in cnn_rnn_cfgs:
        try:
            _n_labels_from_config_path(cfg)
        except ValueError:
            continue
        for stem in stems:
            pq = data_root / "transformed" / f"{stem}.parquet"
            if pq.is_file():
                jobs.append((cfg, stem, "2gpu", None))
    for cfg in ml_cfgs:
        try:
            _n_labels_from_config_path(cfg)
        except ValueError:
            continue
        for stem in stems:
            pq = data_root / "transformed" / f"{stem}.parquet"
            if pq.is_file():
                jobs.append((cfg, stem, "single", None))

    if args.include_ml_bc_after_feature_encoder:
        mq = cfg_root / "ml_queue"
        if mq.is_dir():
            qcfgs = sorted(p for p in mq.glob("track_*.json") if p.is_file())
            if not qcfgs:
                print("[train_queue] ml_queue/ has no track_*.json; skipping ML B/C.", flush=True)
            for stem in stems:
                proc_pq = data_root / "processed" / f"{stem}.parquet"
                if not proc_pq.is_file():
                    print(
                        f"[train_queue] ML B/C: skip stem {stem!r} (missing processed {proc_pq})",
                        flush=True,
                    )
                    continue
                for qc in qcfgs:
                    jobs.append((qc.resolve(), stem, "ml_bc", None))
        else:
            print("[train_queue] --include-ml-bc-after-feature-encoder but ml_queue/ missing.", flush=True)

    if args.resume_run_dir is not None:
        run_log_dir = args.resume_run_dir.resolve()
        if not run_log_dir.is_dir():
            print(f"[train_queue] --resume-run-dir is not a directory: {run_log_dir}", file=sys.stderr)
            return 1
        ts = run_log_dir.name.replace("queue_cnn_rnn_", "", 1) if "queue_cnn_rnn_" in run_log_dir.name else run_log_dir.name
    else:
        run_id = (os.environ.get("THESIS_QUEUE_RUN_ID") or "").strip()
        ts = run_id if run_id else time.strftime("%Y%m%d_%H%M%S")
        run_log_dir = log_dir / f"queue_cnn_rnn_{ts}"
        run_log_dir.mkdir(parents=True, exist_ok=True)

    state_path = run_log_dir / QUEUE_STATE_NAME
    summary_path = run_log_dir / "summary.txt"

    loaded, created_prev = _load_queue_state(state_path)
    created = created_prev or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    statuses: dict[tuple[str, str], dict[str, Any]] = {k: dict(v) for k, v in loaded.items()}

    lines = [f"train_queue run {ts}", f"jobs={len(jobs)}", ""]
    for cfg, stem, kind, _enc in jobs:
        lines.append(f"{cfg.name}\t{stem}\t{kind}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    _atomic_write_json(state_path, _build_queue_state_payload(jobs, statuses, created))

    print(f"[train_queue] Planned {len(jobs)} jobs; manifest {summary_path} state {state_path}", flush=True)

    train_single = repo / "Code" / "thesis" / "train" / "train_single.py"
    if not train_single.is_file():
        print(f"Missing {train_single}", file=sys.stderr)
        return 1

    any_failed = False
    for i, (cfg, stem, kind, enc_ckpt) in enumerate(jobs, start=1):
        key = _job_key(cfg, stem)
        rec = statuses.get(
            key,
            {"status": "pending", "returncode": None, "finished_at": None},
        )
        if rec.get("status") == "ok":
            print(f"[train_queue] [{i}/{len(jobs)}] SKIP (state ok) {stem} {cfg.name}", flush=True)
            continue
        if skip_if_ckpt:
            if kind == "ml_bc":
                done = _ml_bc_skip_if_done(ckpt_root, cfg, stem)
            else:
                ckpt_p = _final_checkpoint_path(ckpt_root, cfg, stem)
                done = ckpt_p.is_file()
            if done:
                print(
                    f"[train_queue] [{i}/{len(jobs)}] SKIP (checkpoint exists) {stem} {cfg.name}",
                    flush=True,
                )
                rec = {
                    "status": "ok",
                    "returncode": 0,
                    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                statuses[key] = rec
                _atomic_write_json(state_path, _build_queue_state_payload(jobs, statuses, created))
                continue

        job_log = run_log_dir / f"{stem}__{cfg.stem}.log"
        if kind == "2gpu":
            cmd = _build_train_2gpu_cmd(
                repo,
                cfg,
                stem,
                data_root,
                ckpt_root,
                log_dir,
                args.epochs_finetune,
                batch_size,
                args.num_workers,
                args.phase,
                args.epochs_pretrain,
                encoder_pretrain_ckpt=enc_ckpt,
            )
        elif kind == "ml_bc":
            cmd = _build_train_ml_bc_cmd(
                repo,
                cfg,
                stem,
                data_root,
                ckpt_root,
                default_batch_size=batch_size,
                default_epochs_b=args.epochs_finetune,
            )
        else:
            cmd = _build_train_single_cmd(
                repo,
                cfg,
                stem,
                data_root,
                ckpt_root,
                log_dir,
                args.epochs_finetune,
                batch_size,
                args.num_workers,
                args.phase,
                args.epochs_pretrain,
                encoder_pretrain_ckpt=enc_ckpt,
            )
        print(f"[train_queue] [{i}/{len(jobs)}] RUN ({kind}) {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        banner = f"\n========== {stem}__{cfg.stem} ({kind}) ==========\n"
        rc_code = _run_subprocess_tee_stdout(cmd, str(repo), job_log, preamble=banner)
        fin = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if rc_code == 0:
            statuses[key] = {"status": "ok", "returncode": 0, "finished_at": fin}
        else:
            any_failed = True
            statuses[key] = {"status": "failed", "returncode": rc_code, "finished_at": fin}
            print(
                f"[train_queue] Job failed rc={rc_code} log={job_log}",
                file=sys.stderr,
                flush=True,
            )
        _atomic_write_json(state_path, _build_queue_state_payload(jobs, statuses, created))

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

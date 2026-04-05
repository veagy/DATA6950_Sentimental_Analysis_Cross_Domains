"""
DDP fine-tune **B11 stacked pipeline**: frozen pretrained **CNN** then **LSTM** (each 100→100 from
``checkpoints/pretrain``), plus trainable ``100 -> 400 -> GELU -> K`` on transformed parquet.

Configs: ``Code/thesis/config/b11_cnn_lstm_stack/{2_labels,3_labels}/B11_CNN_LSTM_stack.json``
(key ``B11StackedCNNLSTMGeLUHeadDDP``).

Checkpoints (rank 0):
  ``checkpoints/b11_cnn_lstm_stack_gelu_ddp/combined/{K}-labels/{stem}/``
    - ``stacked_encoders_cnn_lstm_head.safetensors``
    - ``trainable_head.safetensors``
  ``checkpoints/b11_cnn_lstm_stack_gelu_ddp/heads_only/{K}-labels/{stem}/``
    - ``trainable_head.safetensors``

Launch: ``torchrun --nproc_per_node=3 ...`` (e.g. ``CUDA_VISIBLE_DEVICES=1,2,3``).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from Code.thesis.common.pkg_bootstrap import install_lazy_code_models

install_lazy_code_models(_REPO)

from Code.thesis.common.b11_cnn_lstm_stack_gelu_model import FrozenCNNLSTMStackGeLUHeadClassifier
from Code.thesis.common.checkpoint_io import save_safetensors
from Code.thesis.common.datasets import (
    StreamingParquetFeaturesIterable,
    features_collate,
)
from Code.thesis.common.distributed import (
    get_rank,
    init_distributed_from_env,
    is_main_process,
    make_dataloader_worker_init_fn,
    wrap_ddp,
)

_CFG_KEY = "B11StackedCNNLSTMGeLUHeadDDP"


def _default_pretrain_path(pretrain_dir: Path, arch: str, n_classes: int) -> Path:
    base = arch.lower().strip()
    suffix = "_3labels" if int(n_classes) == 3 else ""
    return pretrain_dir / f"pretrain_{base}{suffix}.safetensors"


def _dataset_stem_from_parquet(path: Path) -> str:
    return path.stem


def _load_json_config(path: Path) -> Mapping[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    inner = data.get(_CFG_KEY)
    if not isinstance(inner, dict):
        raise ValueError(f"{path}: expected top-level {_CFG_KEY!r} object")
    return inner


def _resolve_optional_ckpt(
    ckpt_root: Path,
    pretrain_dir: Path,
    cfg_val: Any,
    arch: str,
    n_classes: int,
) -> Path:
    if isinstance(cfg_val, str) and cfg_val.strip():
        p = Path(cfg_val.strip())
        return (ckpt_root / p).resolve() if not p.is_absolute() else p.resolve()
    return _default_pretrain_path(pretrain_dir, arch, n_classes)


def main() -> int:
    ap = argparse.ArgumentParser(description="B11: frozen CNN→LSTM pretrain stack + GeLU head (DDP).")
    ap.add_argument("--config", type=Path, default=None, help=f"Thesis JSON with {_CFG_KEY}.")
    ap.add_argument("--data_parquet", type=Path, default=None)
    ap.add_argument("--checkpoint_root", type=Path, default=None)
    ap.add_argument("--log_dir", type=Path, default=None)
    ap.add_argument("--pretrain_cnn_ckpt", type=Path, default=None)
    ap.add_argument("--pretrain_lstm_ckpt", type=Path, default=None)
    ap.add_argument("--n_classes", type=int, default=None, choices=(2, 3))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--batch_read", type=int, default=None)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--hidden_dim", type=int, default=None)
    args = ap.parse_args()

    file_cfg: Mapping[str, Any] = {}
    config_path: Path | None = None
    if args.config is not None:
        config_path = args.config.resolve()
        file_cfg = _load_json_config(config_path)

    def cfg_int(key: str, arg: int | None, default: int) -> int:
        if arg is not None:
            return int(arg)
        v = file_cfg.get(key)
        return int(v) if v is not None else default

    def cfg_float(key: str, arg: float | None, default: float) -> float:
        if arg is not None:
            return float(arg)
        v = file_cfg.get(key)
        return float(v) if v is not None else default

    ckpt_root = (args.checkpoint_root or (_REPO / "checkpoints")).resolve()
    log_dir = (args.log_dir or (_REPO / "logs")).resolve()
    pretrain_dir = ckpt_root / "pretrain"

    n_classes = args.n_classes
    if n_classes is None and file_cfg.get("n_labels") is not None:
        n_classes = int(file_cfg["n_labels"])
    if n_classes is None:
        print("Missing n_classes: use --n_classes or config n_labels.", file=sys.stderr)
        return 1

    epochs = cfg_int("epochs", args.epochs, 2)
    batch_size = cfg_int("batch_size", args.batch_size, 512)
    lr = cfg_float("lr", args.lr, 1e-3)
    num_workers = cfg_int("num_workers", args.num_workers, 4)
    batch_read = cfg_int("batch_read", args.batch_read, 8192)
    hidden_dim = cfg_int("hidden_dim", args.hidden_dim, 400)
    max_samples = args.max_samples if args.max_samples is not None else file_cfg.get("max_samples")
    if max_samples is not None:
        max_samples = int(max_samples)

    dataset_stem = str(file_cfg.get("dataset_stem", "all-data"))
    if args.data_parquet is not None:
        data_pq = args.data_parquet.resolve()
    else:
        data_pq = (_REPO / "data" / "transformed" / f"{dataset_stem}.parquet").resolve()

    if not data_pq.is_file():
        print(f"Missing parquet: {data_pq}", file=sys.stderr)
        return 1

    if args.pretrain_cnn_ckpt is not None:
        ckpt_cnn = Path(args.pretrain_cnn_ckpt).resolve()
    else:
        ckpt_cnn = _resolve_optional_ckpt(
            ckpt_root, pretrain_dir, file_cfg.get("pretrain_cnn_ckpt"), "cnn", n_classes
        )
    if args.pretrain_lstm_ckpt is not None:
        ckpt_lstm = Path(args.pretrain_lstm_ckpt).resolve()
    else:
        ckpt_lstm = _resolve_optional_ckpt(
            ckpt_root, pretrain_dir, file_cfg.get("pretrain_lstm_ckpt"), "lstm", n_classes
        )

    if not ckpt_cnn.is_file():
        print(f"Missing CNN pretrain: {ckpt_cnn}", file=sys.stderr)
        return 1
    if not ckpt_lstm.is_file():
        print(f"Missing LSTM pretrain: {ckpt_lstm}", file=sys.stderr)
        return 1

    use_ddp, local_rank, _ws = init_distributed_from_env()
    if local_rank is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        print("CUDA required for this script.", file=sys.stderr)
        return 1

    rank = get_rank()
    stream_ds = StreamingParquetFeaturesIterable(
        str(data_pq),
        rank=rank,
        world_size=int(dist.get_world_size()) if use_ddp else 1,
        max_samples=max_samples,
        input_dim=100,
        batch_read=int(batch_read),
        n_classes=int(n_classes),
    )
    if len(stream_ds) == 0:
        if is_main_process():
            print("No rows in shard.", file=sys.stderr)
        return 1

    model = FrozenCNNLSTMStackGeLUHeadClassifier(
        n_classes,
        ckpt_cnn,
        ckpt_lstm,
        hidden_dim=int(hidden_dim),
    ).to(device)

    if use_ddp:
        model = wrap_ddp(model, int(local_rank), find_unused_parameters=False)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=float(lr))
    crit = nn.CrossEntropyLoss()

    _wi = make_dataloader_worker_init_fn(rank) if num_workers > 0 else None
    _gen = torch.Generator()
    _gen.manual_seed(3407 + rank)
    loader = DataLoader(
        stream_ds,
        batch_size=int(batch_size),
        shuffle=False,
        sampler=None,
        collate_fn=features_collate,
        num_workers=int(num_workers),
        pin_memory=True,
        worker_init_fn=_wi,
        generator=_gen,
        drop_last=use_ddp,
    )

    steps_per_epoch = len(stream_ds) // int(batch_size) if use_ddp else (len(stream_ds) + batch_size - 1) // batch_size
    if use_ddp and len(stream_ds) < batch_size:
        if is_main_process():
            print("Shard smaller than batch_size with drop_last=True.", file=sys.stderr)
        return 1

    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    stem = _dataset_stem_from_parquet(data_pq)
    log_path = log_dir / f"b11_cnn_lstm_stack_gelu_ddp_{stem}_{n_classes}l_{ts}.log"
    log_fp = open(log_path, "w", encoding="utf-8") if is_main_process() else None

    def log(msg: str) -> None:
        print(msg, flush=True)
        if log_fp is not None:
            log_fp.write(msg + "\n")
            log_fp.flush()

    if is_main_process():
        log(
            f"DDP world={dist.get_world_size() if use_ddp else 1} device={device} "
            f"n_classes={n_classes} cnn_pretrain={ckpt_cnn} lstm_pretrain={ckpt_lstm}"
        )
        if config_path is not None:
            log(f"config={config_path}")
        log(f"data={data_pq} rows/rank={len(stream_ds)} batch={batch_size} steps/epoch≈{steps_per_epoch}")

    raw = model.module if hasattr(model, "module") else model
    assert isinstance(raw, FrozenCNNLSTMStackGeLUHeadClassifier)

    for epoch in range(int(epochs)):
        model.train()
        raw.encoder_cnn.eval()
        raw.encoder_lstm.eval()

        it = tqdm(
            loader,
            total=steps_per_epoch,
            desc=f"b11 ep{epoch + 1}/{epochs}",
            disable=not is_main_process(),
            dynamic_ncols=True,
        )
        for xb, yb in it:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            it.set_postfix(loss=float(loss.item()))

        if use_ddp:
            dist.barrier()

    if is_main_process():
        base = ckpt_root / "b11_cnn_lstm_stack_gelu_ddp"
        combined_dir = base / "combined" / f"{n_classes}-labels" / stem
        heads_dir = base / "heads_only" / f"{n_classes}-labels" / stem
        combined_dir.mkdir(parents=True, exist_ok=True)
        heads_dir.mkdir(parents=True, exist_ok=True)
        head_sd = raw.trainable_state_dict()
        stacked_sd = raw.stacked_export_state_dict()
        save_safetensors(head_sd, combined_dir / "trainable_head.safetensors")
        save_safetensors(stacked_sd, combined_dir / "stacked_encoders_cnn_lstm_head.safetensors")
        save_safetensors(head_sd, heads_dir / "trainable_head.safetensors")
        meta_body = (
            f"model=B11 frozen CNN→LSTM (pretrain) + trainable GeLU MLP head\n"
            f"combined_dir={combined_dir}\n"
            f"heads_only_dir={heads_dir}\n"
            f"config={config_path}\n"
            f"cnn_pretrain={ckpt_cnn}\n"
            f"lstm_pretrain={ckpt_lstm}\n"
            f"data_parquet={data_pq}\n"
            f"n_classes={n_classes}\n"
            f"epochs={epochs}\n"
            f"batch_size={batch_size}\n"
            f"lr={lr}\n"
            f"hidden_dim={hidden_dim}\n"
        )
        (combined_dir / "run_meta.txt").write_text(meta_body, encoding="utf-8")
        (heads_dir / "run_meta.txt").write_text(meta_body, encoding="utf-8")
        log(f"Saved {combined_dir / 'stacked_encoders_cnn_lstm_head.safetensors'}")
        log(f"Saved {combined_dir / 'trainable_head.safetensors'}")
        log(f"Saved heads_only {heads_dir / 'trainable_head.safetensors'}")
        if log_fp is not None:
            log_fp.close()

    if use_ddp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

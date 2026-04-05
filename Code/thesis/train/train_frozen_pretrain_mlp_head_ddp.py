"""
DDP fine-tune: frozen pretrained feature encoder (``checkpoints/pretrain/*.safetensors``)
+ trainable MLP ``100 -> 400 -> GELU -> n_classes`` on transformed parquet (100-D features).

Configs: ``Code/thesis/config/mlp_gelu_head_ddp/{2_labels,3_labels}/FeatEnc_*.json``
(body key ``FrozenPretrainGeLUHeadDDP``).

Checkpoints (rank 0):
  - ``checkpoints/mlp_geLU_head_ddp/combined/{K}-labels/{stem}/{arch}/`` —
    ``stacked_encoder_head.safetensors`` + ``trainable_head.safetensors``
  - ``checkpoints/mlp_geLU_head_ddp/heads_only/{K}-labels/{stem}/{arch}/`` —
    ``trainable_head.safetensors`` (same head weights; flat tree for head-only workflows)

Launch with ``torchrun`` (e.g. 3 processes on physical GPUs 1,2,3 via ``CUDA_VISIBLE_DEVICES``).
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

from Code.thesis.common.checkpoint_io import load_safetensors_state, save_safetensors
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
from Code.thesis.common.feature_pretrain_models import FeaturePretrainAutoencoder


class FrozenPretrainGeLUHeadClassifier(nn.Module):
    """Frozen encoder (100-D latent) + trainable ``latent -> 400 -> GELU -> K``."""

    latent_dim: int = 100

    def __init__(
        self,
        architecture: str,
        n_classes: int,
        pretrain_ckpt: Path,
        *,
        hidden_dim: int = 400,
        input_dim: int = 100,
    ) -> None:
        super().__init__()
        self.architecture = architecture.lower().strip()
        self.n_classes = int(n_classes)
        self.input_dim = int(input_dim)

        ae = FeaturePretrainAutoencoder(self.architecture, input_dim=self.input_dim, latent_dim=self.latent_dim)
        blob = load_safetensors_state(pretrain_ckpt, map_location="cpu")
        enc_sd = {k[len("encoder.") :]: v for k, v in blob.items() if k.startswith("encoder.")}
        ae.encoder.load_state_dict(enc_sd, strict=True)
        self.encoder = ae.encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        self.head = nn.Sequential(
            nn.Linear(self.latent_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.head(z)

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v for k, v in self.state_dict().items() if k.startswith("head.")}

    def stacked_export_state_dict(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for k, v in self.encoder.state_dict().items():
            out[f"encoder.{k}"] = v.detach().cpu().contiguous()
        for k, v in self.head.state_dict().items():
            out[f"head.{k}"] = v.detach().cpu().contiguous()
        return out


def _default_pretrain_path(pretrain_dir: Path, arch: str, n_classes: int) -> Path:
    base = "ffnn" if arch.lower() == "ffnn" else arch.lower().strip()
    suffix = "_3labels" if int(n_classes) == 3 else ""
    return pretrain_dir / f"pretrain_{base}{suffix}.safetensors"


def _dataset_stem_from_parquet(path: Path) -> str:
    return path.stem


_CFG_KEY = "FrozenPretrainGeLUHeadDDP"


def _load_json_config(path: Path) -> Mapping[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    inner = data.get(_CFG_KEY)
    if not isinstance(inner, dict):
        raise ValueError(f"{path}: expected top-level {_CFG_KEY!r} object")
    return inner


def main() -> int:
    ap = argparse.ArgumentParser(description="Frozen pretrain encoder + GeLU MLP head (DDP).")
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Thesis JSON with {_CFG_KEY} (paths default under repo).",
    )
    ap.add_argument("--data_parquet", type=Path, default=None, help="Transformed parquet (default: data/transformed/{{dataset_stem}}.parquet)")
    ap.add_argument("--checkpoint_root", type=Path, default=None, help="Default: <repo>/checkpoints")
    ap.add_argument("--log_dir", type=Path, default=None, help="Default: <repo>/logs")
    ap.add_argument("--pretrain_ckpt", type=Path, default=None, help="encoder.* safetensors; default from arch + n_classes")
    ap.add_argument("--pretrain_arch", type=str, default=None, choices=("ffnn", "cnn", "lstm", "gru", "rnn"))
    ap.add_argument("--n_classes", type=int, default=None, choices=(2, 3))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--batch_read", type=int, default=None, help="Parquet row batch for streaming iterator")
    ap.add_argument("--max_samples", type=int, default=None, help="Cap rows per rank (debug)")
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

    def cfg_str(key: str, arg: str | None, default: str | None = None) -> str | None:
        if arg is not None:
            return arg
        v = file_cfg.get(key)
        if v is not None:
            return str(v)
        return default

    ckpt_root = (args.checkpoint_root or (_REPO / "checkpoints")).resolve()
    log_dir = (args.log_dir or (_REPO / "logs")).resolve()
    pretrain_dir = ckpt_root / "pretrain"

    n_classes = args.n_classes
    if n_classes is None:
        nl = file_cfg.get("n_labels")
        if nl is not None:
            n_classes = int(nl)
    if n_classes is None:
        print("Missing n_classes: set --n_classes or config n_labels.", file=sys.stderr)
        return 1

    pretrain_arch = cfg_str("pretrain_arch", args.pretrain_arch, "ffnn")
    assert pretrain_arch is not None
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

    pretrain_ckpt = args.pretrain_ckpt
    cfg_ckpt = file_cfg.get("pretrain_ckpt")
    if pretrain_ckpt is None and isinstance(cfg_ckpt, str) and cfg_ckpt.strip():
        p = Path(cfg_ckpt.strip())
        pretrain_ckpt = (ckpt_root / p).resolve() if not p.is_absolute() else p.resolve()
    if pretrain_ckpt is None:
        pretrain_ckpt = _default_pretrain_path(pretrain_dir, pretrain_arch, n_classes)
    else:
        pretrain_ckpt = Path(pretrain_ckpt).resolve()
    if not pretrain_ckpt.is_file():
        print(f"Missing pretrain checkpoint: {pretrain_ckpt}", file=sys.stderr)
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

    model = FrozenPretrainGeLUHeadClassifier(
        pretrain_arch,
        n_classes,
        pretrain_ckpt,
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
            print("Shard smaller than batch_size with drop_last=True; increase data or lower batch_size.", file=sys.stderr)
        return 1

    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    stem = _dataset_stem_from_parquet(data_pq)
    log_path = log_dir / f"mlp_geLU_head_ddp_{stem}_{n_classes}l_{pretrain_arch}_{ts}.log"
    log_fp = open(log_path, "w", encoding="utf-8") if is_main_process() else None

    def log(msg: str) -> None:
        print(msg, flush=True)
        if log_fp is not None:
            log_fp.write(msg + "\n")
            log_fp.flush()

    if is_main_process():
        log(
            f"DDP world={dist.get_world_size() if use_ddp else 1} device={device} "
            f"pretrain={pretrain_ckpt} arch={pretrain_arch} n_classes={n_classes}"
        )
        if config_path is not None:
            log(f"config={config_path}")
        log(f"data={data_pq} rows/rank={len(stream_ds)} batch={batch_size} steps/epoch≈{steps_per_epoch}")

    raw = model.module if hasattr(model, "module") else model
    assert isinstance(raw, FrozenPretrainGeLUHeadClassifier)

    for epoch in range(int(epochs)):
        if hasattr(model, "train"):
            model.train()
        raw.encoder.eval()

        it = tqdm(
            loader,
            total=steps_per_epoch,
            desc=f"ep{epoch + 1}/{epochs}",
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
        base = ckpt_root / "mlp_geLU_head_ddp"
        combined_dir = base / "combined" / f"{n_classes}-labels" / stem / pretrain_arch
        heads_dir = base / "heads_only" / f"{n_classes}-labels" / stem / pretrain_arch
        combined_dir.mkdir(parents=True, exist_ok=True)
        heads_dir.mkdir(parents=True, exist_ok=True)
        head_sd = raw.trainable_state_dict()
        stacked_sd = raw.stacked_export_state_dict()
        save_safetensors(head_sd, combined_dir / "trainable_head.safetensors")
        save_safetensors(stacked_sd, combined_dir / "stacked_encoder_head.safetensors")
        save_safetensors(head_sd, heads_dir / "trainable_head.safetensors")
        meta_body = (
            f"layout=combined: encoder+head stacked + trainable_head; heads_only: trainable_head copy\n"
            f"combined_dir={combined_dir}\n"
            f"heads_only_dir={heads_dir}\n"
            f"config={config_path}\n"
            f"pretrain_ckpt={pretrain_ckpt}\n"
            f"pretrain_arch={pretrain_arch}\n"
            f"data_parquet={data_pq}\n"
            f"n_classes={n_classes}\n"
            f"epochs={epochs}\n"
            f"batch_size={batch_size}\n"
            f"lr={lr}\n"
            f"hidden_dim={hidden_dim}\n"
        )
        (combined_dir / "run_meta.txt").write_text(meta_body, encoding="utf-8")
        (heads_dir / "run_meta.txt").write_text(meta_body, encoding="utf-8")
        log(f"Saved combined stacked: {combined_dir / 'stacked_encoder_head.safetensors'}")
        log(f"Saved combined head: {combined_dir / 'trainable_head.safetensors'}")
        log(f"Saved heads_only: {heads_dir / 'trainable_head.safetensors'}")
        if log_fp is not None:
            log_fp.close()

    if use_ddp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Train **one** model from a single thesis JSON config (one `--config` per invocation).

Data:
  - Transformers / HRM: processed text under ``data/processed/`` (one stem or merged; see ``--pretrain_text_source``).
  - Merged pretrain (``all_processed``): lazy per-rank shards + ``shuffle=False``; HRM uses 2/3-label sentiment filter (pos/neg vs pos/neu/neg) on merged rows; ``--gc_every`` for periodic gc / CUDA cache clear.
  - CNN, RNN, classical ML: transformed feature parquet under ``data/transformed/``.

Phases:
  - Transformers (LLMModule): pretrain = classifier head only; finetune = classifier head only (backbone frozen).
  - HRM: pretrain = MLM on bare encoder (``checkpoints/pretrain/...``); finetune = ``HRMClassifierWrapper`` with frozen encoder + trainable K-way head (``checkpoints/K-labels/...``). Optional ``--hrm_encoder_ckpt`` loads MLM weights for finetune-only.
  - CNN / RNN / CNN-LSTM filenames: single supervised phase (use ``--phase finetune`` or ``all``).

Multi-model workflows (do **not** use this script for those):
  - Mixture-of-experts: ``Code/thesis/train/train_moe.py``
  - Stacking / meta-learner: ``Code/thesis/train/train_stack.py``
  - Bulk matrix of configs: ``Code/thesis/train/train_all.py --sweep-all`` (opt-in only).
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Callable, Optional

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
from Code.thesis.common.resume_checkpoint import (
    InterruptSave,
    LiveResumeDir,
    ResumeMeta,
    default_resume_temp_root,
    resume_run_dir,
)
from Code.thesis.common.datasets import (
    ParquetFeaturesDataset,
    ParquetTextDataset,
    StreamingParquetFeaturesIterable,
    features_collate,
    text_collate,
)
from Code.thesis.common.multi_parquet_text import LazyShardedMergedParquetTextDataset
from Code.thesis.common.distributed import (
    get_rank,
    init_distributed_from_env,
    is_main_process,
    loader_with_sampler,
    make_dataloader_worker_init_fn,
    make_sampler,
    set_sampler_epoch,
    wrap_ddp,
)
from Code.thesis.common.feature_pretrain_models import FeatureEncoderClassifier, FeaturePretrainAutoencoder
from Code.thesis.common.model_factory import (
    build_model_from_config_dict,
    is_feature_encoder_classifier_config,
    is_text_model_config_path,
    load_config,
)
from Code.thesis.common.torch_util import sync_factory_kwargs_device
from Code.thesis.common.wrappers import RNNClassifier
from Code.models.deep_learning.llm.llm_models import LLMModule
from Code.models.deep_learning.hrm.hrm_model import (
    HierarchicalReasoningModel,
    HRMClassifierWrapper,
)
from Code.models.utils.utils import MLModule


def _resolve_n_classes(config_path: Path, override: int | None, cfg: dict) -> int:
    """Wrapper head size for HRMClassifierWrapper / RNNClassifier. Path, CLI, or JSON."""
    if override is not None:
        return int(override)
    p = str(config_path).replace("\\", "/")
    if "/2_labels/" in p:
        return 2
    if "/3_labels/" in p:
        return 3
    inner = cfg.get("HierarchicalReasoningModel") or {}
    if isinstance(inner.get("n_classes"), int):
        return int(inner["n_classes"])
    ch = inner.get("classification_head")
    if isinstance(ch, dict) and isinstance(ch.get("num_classes"), int):
        return int(ch["num_classes"])
    return 2


def _parquet_for_dataset(data_root: Path, dataset_stem: str, text_mode: bool) -> Path:
    sub = "processed" if text_mode else "transformed"
    return data_root / sub / f"{dataset_stem}.parquet"


def _checkpoint_path(checkpoint_root: Path, n_labels: int, dataset_stem: str, cfg_stem: str) -> Path:
    return checkpoint_root / f"{n_labels}-labels" / dataset_stem / f"{cfg_stem}.safetensors"


# Tree / sklearn-forest state is not fully represented in safetensors-only checkpoints (see checkpoint_io.save_safetensors).
_ML_JOBLIB_CHECKPOINT_CLASSES = frozenset({"DecisionTreeClassifier", "RandomForestClassifier"})


def _checkpoint_joblib_path(checkpoint_root: Path, n_labels: int, dataset_stem: str, cfg_stem: str) -> Path:
    return checkpoint_root / f"{n_labels}-labels" / dataset_stem / f"{cfg_stem}.joblib"


def _hrm_pretrain_checkpoint_path(checkpoint_root: Path, dataset_stem: str, cfg_stem: str) -> Path:
    """K-agnostic HRM MLM checkpoint (bare ``HierarchicalReasoningModel`` weights)."""
    return checkpoint_root / "pretrain" / dataset_stem / f"{cfg_stem}.safetensors"


def _hrm_finetune_checkpoint_path(
    checkpoint_root: Path, dataset_stem: str, n_labels: int, cfg_stem: str
) -> Path:
    """HRM supervised finetune (wrapper) under fine-tune/{stem}/{K-labels}/."""
    return (
        checkpoint_root
        / "fine-tune"
        / dataset_stem
        / f"{n_labels}-labels"
        / f"{cfg_stem}.safetensors"
    )


def _feature_encoder_pretrain_ckpt_path(checkpoint_root: Path, architecture: str) -> Path:
    return checkpoint_root / "pretrain" / f"pretrain_{architecture.lower().strip()}.safetensors"


def _prepare_features_batch(x: torch.Tensor, raw: nn.Module, class_name: str) -> torch.Tensor:
    """Shape feature vectors for CNN/RNN/LSTM/GRU vs flat FFNN models."""
    if isinstance(raw, (FeatureEncoderClassifier, FeaturePretrainAutoencoder)):
        arch = raw.architecture  # type: ignore[attr-defined]
        if arch == "cnn":
            return x
        if arch in ("lstm", "gru", "rnn"):
            # _SeqEncoder: nn.LSTM/GRU/RNN with input_size=1 expects [B, seq_len, 1]
            return x.unsqueeze(-1)
        return x
    cn = class_name.lower()
    if "cnn" in cn and "lstm" not in cn:
        return x.unsqueeze(-1).expand(-1, -1, 3)
    if isinstance(raw, RNNClassifier) or "lstm" in cn or "gru" in cn or "rnn" in cn:
        return x.unsqueeze(1)
    return x


def _hrm_mlm_step(
    model,
    texts,
    device,
    tokenizer,
    pad_id,
    mask_id,
    cls_id,
    sep_id,
    criterion,
    vocab_size: int,
    max_seq_len: int,
):
    tokens = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )
    input_ids = tokens["input_ids"].to(device)
    lm_labels = input_ids.clone()
    rand = torch.rand(input_ids.shape, device=device)
    mask_arr = (rand < 0.15) & (input_ids != pad_id)
    if cls_id is not None:
        mask_arr = mask_arr & (input_ids != cls_id)
    if sep_id is not None:
        mask_arr = mask_arr & (input_ids != sep_id)
    masked = input_ids.clone()
    masked[mask_arr] = mask_id
    lm_labels[~mask_arr] = -100
    logits = model(masked, pretrain=True)
    loss = criterion(logits.view(-1, vocab_size), lm_labels.view(-1))
    return loss


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model


def _hrm_encoder_module(inner: nn.Module) -> HierarchicalReasoningModel | None:
    if isinstance(inner, HRMClassifierWrapper):
        return inner.encoder
    if isinstance(inner, HierarchicalReasoningModel):
        return inner
    return None


def _freeze_hrm_encoder_train_head_only(inner: nn.Module) -> None:
    if isinstance(inner, HRMClassifierWrapper):
        for p in inner.encoder.parameters():
            p.requires_grad = False
        for p in inner.head.parameters():
            p.requires_grad = True


ALL_DATA_PARQUET = "all-data.parquet"

MetaFn = Callable[[int, int], ResumeMeta]


def _amp_grad_scaler(*, enabled: bool):
    """Prefer ``torch.amp.GradScaler('cuda')`` (PyTorch 2.x); fall back for older builds."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _amp_autocast(
    device: torch.device,
    *,
    enabled: bool,
    dtype: Optional[torch.dtype] = None,
):
    if device.type != "cuda":
        return contextlib.nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        if dtype is not None:
            return torch.amp.autocast("cuda", enabled=enabled, dtype=dtype)
        return torch.amp.autocast("cuda", enabled=enabled)
    if dtype is not None:
        return torch.cuda.amp.autocast(enabled=enabled, dtype=dtype)
    return torch.cuda.amp.autocast(enabled=enabled)


def _maybe_periodic_resume_save(
    live: LiveResumeDir | None,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    scaler: Optional[object],
    make_meta: MetaFn | None,
    epoch: int,
    steps_completed_in_epoch: int,
    *,
    use_ddp_module: bool,
    periodic: dict,
    save_every_steps: int,
    save_every_minutes: float,
    min_save_interval_sec: float,
) -> None:
    if live is None or make_meta is None:
        return
    if save_every_steps <= 0 and save_every_minutes <= 0:
        return
    periodic["steps_since_save"] = int(periodic["steps_since_save"]) + 1
    now = time.monotonic()
    due_s = save_every_steps > 0 and periodic["steps_since_save"] >= save_every_steps
    due_t = save_every_minutes > 0 and (now - float(periodic["last_save_mono"])) >= (save_every_minutes * 60.0)
    if not due_s and not due_t:
        return
    last = float(periodic["last_save_mono"])
    if last > 0.0 and (now - last) < min_save_interval_sec:
        return
    live.save(
        model,
        opt,
        scaler,
        make_meta(epoch, steps_completed_in_epoch),
        use_ddp_module=use_ddp_module,
    )
    periodic["last_save_mono"] = now
    periodic["steps_since_save"] = 0


def _clear_live_resume(live: LiveResumeDir | None, use_ddp: bool) -> None:
    if live is None:
        return
    if use_ddp and dist.is_available() and dist.is_initialized():
        dist.barrier()
    if is_main_process():
        shutil.rmtree(live.dir, ignore_errors=True)
    if use_ddp and dist.is_available() and dist.is_initialized():
        dist.barrier()


def train_loop_llm(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    head_only: bool,
    gc_every: int = 50,
    *,
    live: LiveResumeDir | None = None,
    make_meta: MetaFn | None = None,
    interrupt: InterruptSave | None = None,
    use_ddp_module: bool = False,
    save_every_steps: int = 0,
    save_every_minutes: float = 0.0,
    min_save_interval_sec: float = 45.0,
) -> None:
    raw = _unwrap(model)
    raw.set_backbone_trainable(not head_only)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    crit = nn.CrossEntropyLoss()
    scaler = _amp_grad_scaler(enabled=device.type == "cuda")
    st, skip_batches = 0, 0
    if live is not None and make_meta is not None:
        r = live.try_restore_training(
            model,
            opt,
            scaler,
            make_meta(0, 0),
            use_ddp_module=use_ddp_module,
            device=device,
        )
        if r is not None:
            st, skip_batches = r
    model.train()
    global_step = 0
    periodic = {"last_save_mono": 0.0, "steps_since_save": 0}
    for epoch in range(st, epochs):
        set_sampler_epoch(loader.sampler, epoch)
        if interrupt is not None and interrupt.requested:
            if live is not None and make_meta is not None:
                steps_done = 0
                live.save(
                    model,
                    opt,
                    scaler,
                    make_meta(epoch, steps_done),
                    use_ddp_module=use_ddp_module,
                )
            return
        bar = tqdm(loader, desc="LLM train", disable=not is_main_process())
        for i, (texts, y) in enumerate(bar):
            if skip_batches > 0 and epoch == st and i < skip_batches:
                continue
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            with _amp_autocast(device, enabled=device.type == "cuda"):
                logits = model(list(texts), return_type="logits")  # type: ignore[operator]
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            bar.set_postfix(loss=float(loss.item()))
            global_step += 1
            steps_completed = i + 1
            if gc_every > 0 and global_step % gc_every == 0:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            _maybe_periodic_resume_save(
                live,
                model,
                opt,
                scaler,
                make_meta,
                epoch,
                steps_completed,
                use_ddp_module=use_ddp_module,
                periodic=periodic,
                save_every_steps=save_every_steps,
                save_every_minutes=save_every_minutes,
                min_save_interval_sec=min_save_interval_sec,
            )
            if interrupt is not None and interrupt.requested:
                if live is not None and make_meta is not None:
                    live.save(
                        model,
                        opt,
                        scaler,
                        make_meta(epoch, steps_completed),
                        use_ddp_module=use_ddp_module,
                    )
                return
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if live is not None and make_meta is not None:
            live.save(model, opt, scaler, make_meta(epoch + 1, 0), use_ddp_module=use_ddp_module)
            periodic["last_save_mono"] = time.monotonic()
            periodic["steps_since_save"] = 0


def train_loop_hrm_supervised(
    model,
    loader,
    device,
    epochs,
    lr,
    gc_every: int = 50,
    *,
    live: LiveResumeDir | None = None,
    make_meta: MetaFn | None = None,
    interrupt: InterruptSave | None = None,
    use_ddp_module: bool = False,
    save_every_steps: int = 0,
    save_every_minutes: float = 0.0,
    min_save_interval_sec: float = 45.0,
) -> None:
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    crit = nn.CrossEntropyLoss()
    scaler = _amp_grad_scaler(enabled=device.type == "cuda")
    st, skip_batches = 0, 0
    if live is not None and make_meta is not None:
        r = live.try_restore_training(
            model,
            opt,
            scaler,
            make_meta(0, 0),
            use_ddp_module=use_ddp_module,
            device=device,
        )
        if r is not None:
            st, skip_batches = r
    model.train()
    global_step = 0
    periodic = {"last_save_mono": 0.0, "steps_since_save": 0}
    if is_main_process():
        print(
            f"[finetune] HRM cls: starting epochs {st}..{epochs - 1} "
            f"({epochs - st} epoch(s)), {len(loader)} optimizer steps per epoch (tqdm below).",
            flush=True,
        )
    for epoch in range(st, epochs):
        set_sampler_epoch(loader.sampler, epoch)
        if interrupt is not None and interrupt.requested:
            if live is not None and make_meta is not None:
                live.save(
                    model,
                    opt,
                    scaler,
                    make_meta(epoch, 0),
                    use_ddp_module=use_ddp_module,
                )
            return
        bar = tqdm(
            loader,
            desc="HRM cls",
            disable=not is_main_process(),
            file=sys.stdout,
            ascii=True,
            mininterval=1.0,
            dynamic_ncols=False,
        )
        for i, (texts, y) in enumerate(bar):
            if skip_batches > 0 and epoch == st and i < skip_batches:
                continue
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            with _amp_autocast(device, enabled=device.type == "cuda"):
                logits = model(list(texts), pretrain=False)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            bar.set_postfix(loss=float(loss.item()))
            global_step += 1
            if gc_every > 0 and global_step % gc_every == 0:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            steps_completed = i + 1
            _maybe_periodic_resume_save(
                live,
                model,
                opt,
                scaler,
                make_meta,
                epoch,
                steps_completed,
                use_ddp_module=use_ddp_module,
                periodic=periodic,
                save_every_steps=save_every_steps,
                save_every_minutes=save_every_minutes,
                min_save_interval_sec=min_save_interval_sec,
            )
            if interrupt is not None and interrupt.requested:
                if live is not None and make_meta is not None:
                    live.save(
                        model,
                        opt,
                        scaler,
                        make_meta(epoch, steps_completed),
                        use_ddp_module=use_ddp_module,
                    )
                return
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if live is not None and make_meta is not None:
            live.save(model, opt, scaler, make_meta(epoch + 1, 0), use_ddp_module=use_ddp_module)
            periodic["last_save_mono"] = time.monotonic()
            periodic["steps_since_save"] = 0


def train_loop_hrm_mlm(
    model,
    loader,
    device,
    epochs,
    lr,
    gc_every: int = 50,
    *,
    live: LiveResumeDir | None = None,
    make_meta: MetaFn | None = None,
    interrupt: InterruptSave | None = None,
    use_ddp_module: bool = False,
    save_every_steps: int = 0,
    save_every_minutes: float = 0.0,
    min_save_interval_sec: float = 45.0,
    amp_bf16: bool = False,
) -> None:
    enc = _hrm_encoder_module(_unwrap(model))
    if enc is None:
        raise TypeError("HRM MLM expects HRMClassifierWrapper or HierarchicalReasoningModel")
    tok = enc.tokenizer
    vs = enc.hrm_config.vocab_size
    max_seq_len = int(enc.hrm_config.seq_len)
    pad_id = tok.pad_token_id or 0
    mask_id = getattr(tok, "mask_token_id", 103) or 103
    cls_id = getattr(tok, "cls_token_id", None)
    sep_id = getattr(tok, "sep_token_id", None)
    crit = nn.CrossEntropyLoss(ignore_index=-100)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    use_cuda_amp = device.type == "cuda"
    scaler = _amp_grad_scaler(enabled=use_cuda_amp and not amp_bf16)
    st, skip_batches = 0, 0
    if live is not None and make_meta is not None:
        r = live.try_restore_training(
            model,
            opt,
            scaler,
            make_meta(0, 0),
            use_ddp_module=use_ddp_module,
            device=device,
        )
        if r is not None:
            st, skip_batches = r
    model.train()
    global_step = 0
    periodic = {"last_save_mono": 0.0, "steps_since_save": 0}
    for epoch in range(st, epochs):
        set_sampler_epoch(loader.sampler, epoch)
        if interrupt is not None and interrupt.requested:
            if live is not None and make_meta is not None:
                live.save(
                    model,
                    opt,
                    scaler,
                    make_meta(epoch, 0),
                    use_ddp_module=use_ddp_module,
                )
            return
        bar = tqdm(
            loader,
            desc=f"HRM_MLM ep{epoch + 1}/{epochs}",
            disable=not is_main_process(),
            file=sys.stdout,
            ascii=True,
            mininterval=1.0,
            dynamic_ncols=False,
        )
        for i, (texts, _) in enumerate(bar):
            if skip_batches > 0 and epoch == st and i < skip_batches:
                continue
            opt.zero_grad(set_to_none=True)
            _bf16 = torch.bfloat16 if (use_cuda_amp and amp_bf16) else None
            with _amp_autocast(device, enabled=use_cuda_amp, dtype=_bf16):
                loss = _hrm_mlm_step(
                    model,
                    texts,
                    device,
                    tok,
                    pad_id,
                    mask_id,
                    cls_id,
                    sep_id,
                    crit,
                    vs,
                    max_seq_len,
                )
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            bar.set_postfix(loss=float(loss.item()))
            global_step += 1
            steps_completed = i + 1
            if gc_every > 0 and global_step % gc_every == 0:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            _maybe_periodic_resume_save(
                live,
                model,
                opt,
                scaler,
                make_meta,
                epoch,
                steps_completed,
                use_ddp_module=use_ddp_module,
                periodic=periodic,
                save_every_steps=save_every_steps,
                save_every_minutes=save_every_minutes,
                min_save_interval_sec=min_save_interval_sec,
            )
            if interrupt is not None and interrupt.requested:
                if live is not None and make_meta is not None:
                    live.save(
                        model,
                        opt,
                        scaler,
                        make_meta(epoch, steps_completed),
                        use_ddp_module=use_ddp_module,
                    )
                return
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if live is not None and make_meta is not None:
            live.save(model, opt, scaler, make_meta(epoch + 1, 0), use_ddp_module=use_ddp_module)
            periodic["last_save_mono"] = time.monotonic()
            periodic["steps_since_save"] = 0


def train_loop_tensor(
    model,
    loader,
    device,
    epochs,
    lr,
    class_name: str,
    *,
    live: LiveResumeDir | None = None,
    make_meta: MetaFn | None = None,
    interrupt: InterruptSave | None = None,
    use_ddp_module: bool = False,
    save_every_steps: int = 0,
    save_every_minutes: float = 0.0,
    min_save_interval_sec: float = 45.0,
) -> None:
    raw = _unwrap(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    scaler = _amp_grad_scaler(enabled=device.type == "cuda")
    st, skip_batches = 0, 0
    if live is not None and make_meta is not None:
        r = live.try_restore_training(
            model,
            opt,
            scaler,
            make_meta(0, 0),
            use_ddp_module=use_ddp_module,
            device=device,
        )
        if r is not None:
            st, skip_batches = r
    model.train()
    periodic = {"last_save_mono": 0.0, "steps_since_save": 0}
    for epoch in range(st, epochs):
        set_sampler_epoch(loader.sampler, epoch)
        if interrupt is not None and interrupt.requested:
            if live is not None and make_meta is not None:
                live.save(
                    model,
                    opt,
                    scaler,
                    make_meta(epoch, 0),
                    use_ddp_module=use_ddp_module,
                )
            return
        bar = tqdm(loader, desc=f"{class_name}", disable=not is_main_process())
        for i, (x, y) in enumerate(bar):
            if skip_batches > 0 and epoch == st and i < skip_batches:
                continue
            x = x.to(device)
            y = y.to(device)
            x = _prepare_features_batch(x, raw, class_name)
            opt.zero_grad(set_to_none=True)
            with _amp_autocast(device, enabled=device.type == "cuda"):
                logits = model(x)
                if isinstance(logits, tuple):
                    logits = logits[0]
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            bar.set_postfix(loss=float(loss.item()))
            steps_completed = i + 1
            _maybe_periodic_resume_save(
                live,
                model,
                opt,
                scaler,
                make_meta,
                epoch,
                steps_completed,
                use_ddp_module=use_ddp_module,
                periodic=periodic,
                save_every_steps=save_every_steps,
                save_every_minutes=save_every_minutes,
                min_save_interval_sec=min_save_interval_sec,
            )
            if interrupt is not None and interrupt.requested:
                if live is not None and make_meta is not None:
                    live.save(
                        model,
                        opt,
                        scaler,
                        make_meta(epoch, steps_completed),
                        use_ddp_module=use_ddp_module,
                    )
                return
        if live is not None and make_meta is not None:
            live.save(model, opt, scaler, make_meta(epoch + 1, 0), use_ddp_module=use_ddp_module)
            periodic["last_save_mono"] = time.monotonic()
            periodic["steps_since_save"] = 0


def train_loop_feature_ae(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    *,
    live: LiveResumeDir | None = None,
    make_meta: MetaFn | None = None,
    interrupt: InterruptSave | None = None,
    use_ddp_module: bool = False,
    save_every_steps: int = 0,
    save_every_minutes: float = 0.0,
    min_save_interval_sec: float = 45.0,
    gc_every: int = 0,
) -> None:
    raw = _unwrap(model)
    if not isinstance(raw, FeaturePretrainAutoencoder):
        raise TypeError("train_loop_feature_ae expects FeaturePretrainAutoencoder")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    scaler = _amp_grad_scaler(enabled=device.type == "cuda")
    st, skip_batches = 0, 0
    if live is not None and make_meta is not None:
        r = live.try_restore_training(
            model,
            opt,
            scaler,
            make_meta(0, 0),
            use_ddp_module=use_ddp_module,
            device=device,
        )
        if r is not None:
            st, skip_batches = r
    model.train()
    periodic = {"last_save_mono": 0.0, "steps_since_save": 0}
    global_step = 0
    for epoch in range(st, epochs):
        set_sampler_epoch(loader.sampler, epoch)
        if interrupt is not None and interrupt.requested:
            if live is not None and make_meta is not None:
                live.save(
                    model,
                    opt,
                    scaler,
                    make_meta(epoch, 0),
                    use_ddp_module=use_ddp_module,
                )
            return
        bar = tqdm(loader, desc=f"FeatureAE-{raw.architecture}", disable=not is_main_process())
        for i, (x, _) in enumerate(bar):
            if skip_batches > 0 and epoch == st and i < skip_batches:
                continue
            x = x.to(device)
            target = x
            x_in = _prepare_features_batch(x, raw, "FeaturePretrainAutoencoder")
            opt.zero_grad(set_to_none=True)
            with _amp_autocast(device, enabled=device.type == "cuda"):
                _, recon = model(x_in)
                loss = crit(recon, target)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            bar.set_postfix(loss=float(loss.item()))
            global_step += 1
            steps_completed = i + 1
            if gc_every > 0 and global_step % gc_every == 0:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            _maybe_periodic_resume_save(
                live,
                model,
                opt,
                scaler,
                make_meta,
                epoch,
                steps_completed,
                use_ddp_module=use_ddp_module,
                periodic=periodic,
                save_every_steps=save_every_steps,
                save_every_minutes=save_every_minutes,
                min_save_interval_sec=min_save_interval_sec,
            )
            if interrupt is not None and interrupt.requested:
                if live is not None and make_meta is not None:
                    live.save(
                        model,
                        opt,
                        scaler,
                        make_meta(epoch, steps_completed),
                        use_ddp_module=use_ddp_module,
                    )
                return
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if live is not None and make_meta is not None:
            live.save(model, opt, scaler, make_meta(epoch + 1, 0), use_ddp_module=use_ddp_module)
            periodic["last_save_mono"] = time.monotonic()
            periodic["steps_since_save"] = 0


def main() -> None:
    for _log in ("httpx", "httpcore"):
        logging.getLogger(_log).setLevel(logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument(
        "--n_classes",
        type=int,
        default=None,
        choices=(2, 3),
        help="Classification head width for finetune (HRM/transformers with wrapper). "
        "Omit for HRM MLM-only pretrain (encoder-only; no K-way head).",
    )
    ap.add_argument("--dataset_stem", type=str, required=True, help="Parquet basename without .parquet")
    ap.add_argument("--data_root", type=Path, default=None, help="Contains processed/ and transformed/")
    ap.add_argument("--checkpoint_root", type=Path, default=None)
    ap.add_argument("--log_dir", type=Path, default=None)
    ap.add_argument("--epochs_pretrain", type=int, default=1)
    ap.add_argument("--epochs_finetune", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--phase", choices=("pretrain", "finetune", "all"), default="all")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument(
        "--gc_every",
        type=int,
        default=50,
        help="Run gc (and CUDA cache clear) every N train steps; 0 disables.",
    )
    ap.add_argument(
        "--pretrain_text_source",
        choices=("dataset", "all_processed", "all_data_parquet"),
        default="dataset",
        help="Pretrain: dataset stem parquet | merge processed/*.parquet | only data/processed/all-data.parquet text (HRM MLM). Finetune uses --dataset_stem unless --hrm-finetune-sharded-processed.",
    )
    ap.add_argument(
        "--hrm-finetune-sharded-processed",
        action="store_true",
        help="HRM finetune: rank-sharded lazy load of data/processed/*.parquet (same style as --pretrain_text_source all_processed). "
        "Lower peak RAM than loading the stem parquet into memory; training can be slower (re-reads whole file columns when crossing shards). "
        "Incompatible with --no_hrm_exclude_neutral for 2-class.",
    )
    ap.add_argument(
        "--resume_temp_root",
        type=Path,
        default=None,
        help="Directory for live resume bundles (overrides THESIS_RESUME_TEMP / TMPDIR).",
    )
    ap.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not load or save temp resume state between epochs.",
    )
    ap.add_argument(
        "--save_every_steps",
        type=int,
        default=0,
        help="Write resume bundle every N optimizer steps (0 disables).",
    )
    ap.add_argument(
        "--save_every_minutes",
        type=float,
        default=0.0,
        help="Write resume bundle every N minutes (0 disables).",
    )
    ap.add_argument(
        "--min_save_interval_sec",
        type=float,
        default=45.0,
        help="Minimum seconds between periodic resume saves (avoids IO thrash).",
    )
    ap.add_argument(
        "--encoder_pretrain_ckpt",
        type=Path,
        default=None,
        help="FeatureEncoderClassifier: load encoder.* weights from this safetensors before finetune.",
    )
    ap.add_argument(
        "--hrm_encoder_ckpt",
        type=Path,
        default=None,
        help="HRM finetune: load bare encoder weights from this safetensors (e.g. checkpoints/pretrain/.../E_HRM1_4Level.safetensors) into HRMClassifierWrapper.encoder.",
    )
    ap.add_argument(
        "--hrm_finetune_checkpoint_layout",
        action="store_true",
        help="Save HRMClassifierWrapper under checkpoint_root/fine-tune/{dataset_stem}/{n_labels}-labels/{config_stem}.safetensors.",
    )
    ap.add_argument(
        "--no_hrm_exclude_neutral",
        action="store_true",
        help="For HRM 2-class text finetune, keep rows labeled neutral (default: drop neutral rows).",
    )
    ap.add_argument(
        "--amp_bf16",
        action="store_true",
        help="HRM MLM: autocast bfloat16 on CUDA (GradScaler off). Typical on Ada/Blackwell.",
    )
    ap.add_argument(
        "--no_save_hrm_tokenizer",
        action="store_true",
        help="Skip saving the HF tokenizer under checkpoint_root/tokenizer after HRM pretrain.",
    )
    ap.add_argument(
        "--dataloader_persistent_workers",
        action="store_true",
        help="Use persistent_workers=True when num_workers > 0.",
    )
    ap.add_argument(
        "--dataloader_prefetch_factor",
        type=int,
        default=4,
        help="DataLoader prefetch_factor per worker when num_workers > 0 (ignored if num_workers is 0).",
    )
    args = ap.parse_args()
    if os.environ.get("THESIS_AMP_BF16", "").strip().lower() in ("1", "true", "yes"):
        args.amp_bf16 = True
    if os.environ.get("THESIS_DATALOADER_PERSISTENT", "").strip().lower() in ("1", "true", "yes"):
        args.dataloader_persistent_workers = True
    if os.environ.get("THESIS_HRM_FINETUNE_CKPT_LAYOUT", "").strip().lower() in ("1", "true", "yes"):
        args.hrm_finetune_checkpoint_layout = True
    if os.environ.get("THESIS_HRM_FINETUNE_SHARDED", "").strip().lower() in ("1", "true", "yes"):
        args.hrm_finetune_sharded_processed = True

    data_root = args.data_root or (_REPO / "data")
    ckpt_root = args.checkpoint_root or (_REPO / "checkpoints")
    log_dir = args.log_dir or (_REPO / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    use_ddp, local_rank, _world_sz = init_distributed_from_env()
    try:
        if use_ddp:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Distributed training requires CUDA but torch.cuda.is_available() is False. "
                    "Use THESIS_PYTHON with a CUDA PyTorch build, fix drivers, and set CUDA_VISIBLE_DEVICES "
                    "before the process starts."
                )
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")

        cfg_path = args.config.resolve()
        cfg = load_config(cfg_path)
        n_classes = _resolve_n_classes(cfg_path, args.n_classes, cfg)
        text_mode = is_text_model_config_path(cfg_path)
        pq = _parquet_for_dataset(data_root, args.dataset_stem, text_mode)

        hrm_in_cfg = "HierarchicalReasoningModel" in cfg
        hrm_encoder_only_first = (
            hrm_in_cfg
            and text_mode
            and args.phase in ("pretrain", "all")
            and args.epochs_pretrain > 0
        )
        hrm_pretrain_only_for_resume = (
            hrm_encoder_only_first
            and args.phase == "pretrain"
            and args.epochs_finetune == 0
        )
        resume_n_labels_for_live = 0 if hrm_pretrain_only_for_resume else n_classes

        model, class_name = build_model_from_config_dict(
            cfg, n_classes, "", hrm_encoder_only=hrm_encoder_only_first
        )
        model = model.to(device)
        if not isinstance(model, MLModule):
            sync_factory_kwargs_device(model, device)

        if (
            args.hrm_encoder_ckpt is not None
            and hrm_in_cfg
            and args.phase == "finetune"
            and isinstance(model, HRMClassifierWrapper)
        ):
            ep = args.hrm_encoder_ckpt.resolve()
            if ep.is_file():
                sd_enc = load_safetensors_state(str(ep), map_location=str(device))
                model.encoder.load_state_dict(sd_enc, strict=True)
            elif is_main_process():
                print(
                    f"Warning: --hrm_encoder_ckpt not found at {ep}; training from init.",
                    flush=True,
                )

        if (
            args.encoder_pretrain_ckpt is not None
            and is_feature_encoder_classifier_config(cfg)
            and isinstance(model, FeatureEncoderClassifier)
        ):
            ep = args.encoder_pretrain_ckpt.resolve()
            if ep.is_file():
                model.load_encoder_from_safetensors(str(ep), map_location=str(device))
            elif is_main_process():
                print(
                    f"Warning: encoder checkpoint not found at {ep}; training encoder from init.",
                    flush=True,
                )

        # LazyLinear / uninitialized heads break DDP until a forward pass; materialize on every rank.
        if isinstance(model, LLMModule):
            model.train()
            with torch.no_grad():
                _ = model(["hello"], return_type="logits")

        if use_ddp and not isinstance(model, MLModule):
            # HRM paths may have unused params in some forwards; FeatEnc / AE / materialized LLM do not.
            # find_unused_parameters=True adds an autograd graph pass every step (see PyTorch warning).
            raw_ddp = _unwrap(model)
            find_unused = not isinstance(
                raw_ddp,
                (FeatureEncoderClassifier, FeaturePretrainAutoencoder, LLMModule),
            )
            model = wrap_ddp(model, local_rank, find_unused_parameters=find_unused)
        use_ddp_module = isinstance(model, nn.parallel.DistributedDataParallel)
        # Shard iterators/maps from the process group (not only env) so every DDP model path agrees.
        stream_shard_rank = (
            int(dist.get_rank())
            if use_ddp and dist.is_available() and dist.is_initialized()
            else 0
        )
        stream_shard_world = (
            int(dist.get_world_size())
            if use_ddp and dist.is_available() and dist.is_initialized()
            else 1
        )

        if text_mode:
            exclude_neutral_ft = (
                hrm_in_cfg and n_classes == 2 and not args.no_hrm_exclude_neutral
            )
            want_sharded_hrm_ft = (
                hrm_in_cfg
                and args.phase in ("finetune", "all")
                and args.hrm_finetune_sharded_processed
                and not (n_classes == 2 and args.no_hrm_exclude_neutral)
            )
            finetune_ds: ParquetTextDataset | None
            if want_sharded_hrm_ft:
                finetune_ds = None
            else:
                if (
                    is_main_process()
                    and args.phase in ("finetune", "all")
                    and args.epochs_finetune > 0
                ):
                    print(
                        f"[finetune] Loading parquet into RAM (read + per-row scan; "
                        f"runs on every DDP rank, no tqdm yet): {pq}",
                        flush=True,
                    )
                finetune_ds = ParquetTextDataset(
                    str(pq),
                    max_samples=args.max_samples,
                    exclude_neutral=exclude_neutral_ft,
                    n_classes=n_classes,
                )
            if (
                finetune_ds is not None
                and exclude_neutral_ft
                and is_main_process()
            ):
                print(
                    f"[finetune] HRM 2-class: excluding neutral-labeled rows from {pq} "
                    f"({len(finetune_ds)} samples).",
                    flush=True,
                )
            elif (
                finetune_ds is not None
                and (not exclude_neutral_ft)
                and is_main_process()
                and args.phase in ("finetune", "all")
                and args.epochs_finetune > 0
            ):
                print(
                    f"[finetune] Loaded {len(finetune_ds)} samples from {pq}.",
                    flush=True,
                )
        else:
            # FeatureEncoderClassifier finetune uses StreamingParquetFeaturesIterable (sharded, no full-RAM load).
            finetune_ds = None

        inner = model.module if use_ddp else model

        if isinstance(inner, MLModule):
            if text_mode:
                raise ValueError("ML models require transformed features.")
            if finetune_ds is None:
                finetune_ds = ParquetFeaturesDataset(
                    str(pq), max_samples=args.max_samples, n_classes=n_classes
                )
            ds = finetune_ds
            if len(ds) == 0:
                if is_main_process():
                    print(f"No samples in {pq}")
                sys.exit(1)
            xs = torch.stack([ds[i][0] for i in range(len(ds))])
            ys = torch.tensor([ds[i][1] for i in range(len(ds))], dtype=torch.float32)
            inner.fit(xs, ys)
            if is_main_process():
                out_dir = _checkpoint_path(ckpt_root, n_classes, args.dataset_stem, cfg_path.stem).parent
                out_dir.mkdir(parents=True, exist_ok=True)
                if class_name in _ML_JOBLIB_CHECKPOINT_CLASSES:
                    import joblib

                    out_j = _checkpoint_joblib_path(
                        ckpt_root, n_classes, args.dataset_stem, cfg_path.stem
                    )
                    joblib.dump(inner, out_j)
                    print("Saved", out_j, "(joblib; full estimator state)", flush=True)
                else:
                    out = _checkpoint_path(ckpt_root, n_classes, args.dataset_stem, cfg_path.stem)
                    sd = inner.state_dict() if callable(getattr(inner, "state_dict", None)) else {}
                    save_safetensors(sd, out)
                    print("Saved", out, flush=True)
            return

        interrupt = InterruptSave()
        interrupt.install()
        resume_root = (
            args.resume_temp_root.resolve() if args.resume_temp_root else default_resume_temp_root(_REPO)
        )
        live: LiveResumeDir | None
        if args.no_resume:
            live = None
        else:
            live = LiveResumeDir(
                resume_run_dir(resume_root, cfg_path.stem, args.dataset_stem, resume_n_labels_for_live),
                use_ddp,
            )

        if not text_mode:
            if isinstance(inner, FeaturePretrainAutoencoder):
                if args.phase == "finetune":
                    raise ValueError(
                        "FeaturePretrainAutoencoder config is pretrain-only; use --phase pretrain or all."
                    )
                pq_ft = _parquet_for_dataset(data_root, args.dataset_stem, text_mode=False)
                if not pq_ft.is_file():
                    if is_main_process():
                        print(f"Missing transformed parquet: {pq_ft}")
                    sys.exit(1)
                input_dim = int(inner.input_dim)
                try:
                    stream_ds = StreamingParquetFeaturesIterable(
                        str(pq_ft),
                        rank=stream_shard_rank,
                        world_size=stream_shard_world,
                        max_samples=args.max_samples,
                        input_dim=input_dim,
                        batch_read=4096,
                    )
                except RuntimeError as e:
                    if is_main_process():
                        print(e, file=sys.stderr)
                    sys.exit(1)
                if len(stream_ds) == 0:
                    if is_main_process():
                        print(f"No rows in {pq_ft}")
                    sys.exit(1)
                _wi_ae = make_dataloader_worker_init_fn(get_rank()) if args.num_workers > 0 else None
                _gen_ae = torch.Generator()
                _gen_ae.manual_seed(3407 + int(get_rank()))
                ae_loader = DataLoader(
                    stream_ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    sampler=None,
                    collate_fn=features_collate,
                    num_workers=args.num_workers,
                    pin_memory=torch.cuda.is_available(),
                    worker_init_fn=_wi_ae,
                    generator=_gen_ae,
                    drop_last=use_ddp,
                )
                make_ae_meta = lambda epoch_next, steps_done=0: ResumeMeta(
                    cfg_path=str(cfg_path),
                    dataset_stem=args.dataset_stem,
                    n_labels=n_classes,
                    phase=args.phase,
                    segment="feature_encoder_pretrain",
                    epoch_next=epoch_next,
                    epochs_segment=args.epochs_pretrain,
                    lr=args.lr,
                    class_name=class_name,
                    head_only=None,
                    steps_completed_in_epoch=steps_done,
                )
                train_loop_feature_ae(
                    model,
                    ae_loader,
                    device,
                    args.epochs_pretrain,
                    args.lr,
                    live=live,
                    make_meta=make_ae_meta,
                    interrupt=interrupt,
                    use_ddp_module=use_ddp_module,
                    save_every_steps=args.save_every_steps,
                    save_every_minutes=args.save_every_minutes,
                    min_save_interval_sec=args.min_save_interval_sec,
                    gc_every=args.gc_every,
                )
                if interrupt.requested:
                    return
                _clear_live_resume(live, use_ddp)
                if use_ddp and dist.is_available() and dist.is_initialized():
                    dist.barrier()
                if is_main_process():
                    raw_u = _unwrap(model)
                    assert isinstance(raw_u, FeaturePretrainAutoencoder)
                    out_enc = _feature_encoder_pretrain_ckpt_path(ckpt_root, raw_u.architecture)
                    save_safetensors(raw_u.encoder_checkpoint_tensors(), out_enc)
                    print("Saved encoder-only pretrain checkpoint", out_enc, flush=True)
                if use_ddp and dist.is_available() and dist.is_initialized():
                    dist.barrier()
                return

            raw_fe = _unwrap(model)
            if isinstance(raw_fe, FeatureEncoderClassifier):
                pq_fe = Path(str(pq))
                if not pq_fe.is_file():
                    if is_main_process():
                        print(f"No samples in {pq_fe}")
                    sys.exit(1)
                input_dim = int(raw_fe.input_dim)
                try:
                    stream_ft = StreamingParquetFeaturesIterable(
                        str(pq_fe),
                        rank=stream_shard_rank,
                        world_size=stream_shard_world,
                        max_samples=args.max_samples,
                        input_dim=input_dim,
                        batch_read=8192,
                        n_classes=n_classes,
                    )
                except RuntimeError as e:
                    if is_main_process():
                        print(e, file=sys.stderr)
                    sys.exit(1)
                if len(stream_ft) == 0:
                    if is_main_process():
                        print(f"No rows in {pq_fe}")
                    sys.exit(1)
                if is_main_process():
                    cap = (
                        f" [capped: --max_samples={args.max_samples}]"
                        if args.max_samples is not None
                        else " [all rows: shard per rank]"
                    )
                    print(
                        f"[finetune] Streaming transformed features: {len(stream_ft)} samples/rank "
                        f"× {stream_shard_world} ranks from {pq_fe}{cap}",
                        flush=True,
                    )
                _wi_ft = make_dataloader_worker_init_fn(get_rank()) if args.num_workers > 0 else None
                _gen_ft = torch.Generator()
                _gen_ft.manual_seed(3407 + int(get_rank()))
                loader = DataLoader(
                    stream_ft,
                    batch_size=args.batch_size,
                    shuffle=False,
                    sampler=None,
                    collate_fn=features_collate,
                    num_workers=args.num_workers,
                    pin_memory=torch.cuda.is_available(),
                    worker_init_fn=_wi_ft,
                    generator=_gen_ft,
                    drop_last=use_ddp,
                )
            else:
                if finetune_ds is None:
                    finetune_ds = ParquetFeaturesDataset(
                        str(pq), max_samples=args.max_samples, n_classes=n_classes
                    )
                ds = finetune_ds
                if len(ds) == 0:
                    if is_main_process():
                        print(f"No samples in {pq}")
                    sys.exit(1)
                sampler = make_sampler(ds, shuffle=True, use_ddp=use_ddp)
                _wi = make_dataloader_worker_init_fn(get_rank()) if args.num_workers > 0 else None
                _gen = torch.Generator()
                _gen.manual_seed(3407 + int(get_rank()))
                _fpw = args.num_workers > 0 and args.dataloader_persistent_workers
                _fpf = args.dataloader_prefetch_factor if args.num_workers > 0 else None
                loader = loader_with_sampler(
                    ds,
                    batch_size=args.batch_size,
                    collate_fn=features_collate,
                    sampler=sampler,
                    num_workers=args.num_workers,
                    worker_init_fn=_wi,
                    generator=_gen,
                    persistent_workers=_fpw,
                    prefetch_factor=_fpf,
                    drop_last=use_ddp,
                )
            make_tensor_meta = lambda epoch_next, steps_done=0: ResumeMeta(
                cfg_path=str(cfg_path),
                dataset_stem=args.dataset_stem,
                n_labels=n_classes,
                phase=args.phase,
                segment="finetune",
                epoch_next=epoch_next,
                epochs_segment=args.epochs_finetune,
                lr=args.lr,
                class_name=class_name,
                head_only=None,
                steps_completed_in_epoch=steps_done,
            )
            train_loop_tensor(
                model,
                loader,
                device,
                args.epochs_finetune,
                args.lr,
                class_name,
                live=live,
                make_meta=make_tensor_meta,
                interrupt=interrupt,
                use_ddp_module=use_ddp_module,
                save_every_steps=args.save_every_steps,
                save_every_minutes=args.save_every_minutes,
                min_save_interval_sec=args.min_save_interval_sec,
            )
            if interrupt.requested:
                return
            _clear_live_resume(live, use_ddp)
        else:

            def _text_loader(text_ds):
                if len(text_ds) == 0:
                    return None
                sam = make_sampler(text_ds, shuffle=True, use_ddp=use_ddp)
                wi = make_dataloader_worker_init_fn(get_rank()) if args.num_workers > 0 else None
                gen = torch.Generator()
                gen.manual_seed(3407 + int(get_rank()))
                _pw = args.num_workers > 0 and args.dataloader_persistent_workers
                _pf = args.dataloader_prefetch_factor if args.num_workers > 0 else None
                return loader_with_sampler(
                    text_ds,
                    batch_size=args.batch_size,
                    collate_fn=text_collate,
                    sampler=sam,
                    num_workers=args.num_workers,
                    worker_init_fn=wi,
                    generator=gen,
                    persistent_workers=_pw,
                    prefetch_factor=_pf,
                    drop_last=use_ddp,
                )

            if args.phase in ("finetune", "all") and finetune_ds is not None:
                if len(finetune_ds) == 0:
                    if is_main_process():
                        print(f"No samples in {pq}")
                    sys.exit(1)

            pre_loader = None
            if args.phase in ("pretrain", "all"):
                if args.pretrain_text_source == "all_data_parquet":
                    ad_path = data_root / "processed" / ALL_DATA_PARQUET
                    pre_ds = ParquetTextDataset(
                        str(ad_path),
                        max_samples=args.max_samples,
                        allow_missing_label=True,
                        dummy_label=0,
                    )
                    if len(pre_ds) == 0:
                        if is_main_process():
                            print(f"No pretrain samples from {ad_path}")
                        sys.exit(1)
                    if is_main_process():
                        cap = (
                            f" [capped: --max_samples={args.max_samples}]"
                            if args.max_samples is not None
                            else " [all rows in file]"
                        )
                        print(
                            f"[pretrain] all_data_parquet: {len(pre_ds)} samples from {ad_path}{cap}"
                        )
                    pre_loader = _text_loader(pre_ds)
                elif args.pretrain_text_source == "all_processed":
                    pretrain_nc = (
                        n_classes
                        if _hrm_encoder_module(inner) is not None and n_classes in (2, 3)
                        else None
                    )
                    pre_ds = LazyShardedMergedParquetTextDataset(
                        data_root / "processed",
                        rank=stream_shard_rank,
                        world_size=stream_shard_world,
                        max_samples_per_file=None,
                        max_total=args.max_samples,
                        pretrain_num_classes=pretrain_nc,
                    )
                    if len(pre_ds) == 0:
                        if is_main_process():
                            print(f"No pretrain samples from {data_root / 'processed'}")
                        sys.exit(1)
                    _gen_ap = torch.Generator()
                    _gen_ap.manual_seed(3407 + stream_shard_rank)
                    _wi_ap = (
                        make_dataloader_worker_init_fn(stream_shard_rank)
                        if args.num_workers > 0
                        else None
                    )
                    pre_loader = DataLoader(
                        pre_ds,
                        batch_size=args.batch_size,
                        shuffle=False,
                        sampler=None,
                        collate_fn=text_collate,
                        num_workers=args.num_workers,
                        pin_memory=torch.cuda.is_available(),
                        worker_init_fn=_wi_ap,
                        generator=_gen_ap,
                        drop_last=use_ddp,
                    )
                else:
                    pre_ds = ParquetTextDataset(str(pq), max_samples=args.max_samples)
                    if len(pre_ds) == 0:
                        if is_main_process():
                            print(f"No pretrain samples from {pq}")
                        sys.exit(1)
                    pre_loader = _text_loader(pre_ds)

            ft_loader = None
            if args.phase in ("finetune", "all"):
                if want_sharded_hrm_ft:
                    pretrain_nc = n_classes if n_classes in (2, 3) else None
                    ft_lazy = LazyShardedMergedParquetTextDataset(
                        data_root / "processed",
                        rank=stream_shard_rank,
                        world_size=stream_shard_world,
                        max_samples_per_file=None,
                        max_total=args.max_samples,
                        text_col=None,
                        pretrain_num_classes=pretrain_nc,
                        return_supervised_labels=True,
                    )
                    if len(ft_lazy) == 0:
                        if is_main_process():
                            print(
                                f"No finetune samples from sharded processed dir {data_root / 'processed'}",
                                flush=True,
                            )
                        sys.exit(1)
                    if is_main_process():
                        cap = (
                            f" [capped: --max_samples={args.max_samples}]"
                            if args.max_samples is not None
                            else ""
                        )
                        print(
                            f"[finetune] HRM sharded processed: {len(ft_lazy)} samples per rank "
                            f"× {stream_shard_world} ranks (merged data/processed/*.parquet){cap}",
                            flush=True,
                        )
                    _gen_ft = torch.Generator()
                    _gen_ft.manual_seed(3407 + stream_shard_rank)
                    _wi_ft = (
                        make_dataloader_worker_init_fn(stream_shard_rank)
                        if args.num_workers > 0
                        else None
                    )
                    _fpw_ft = args.num_workers > 0 and args.dataloader_persistent_workers
                    _fpf_ft = args.dataloader_prefetch_factor if args.num_workers > 0 else None
                    ft_loader = DataLoader(
                        ft_lazy,
                        batch_size=args.batch_size,
                        shuffle=False,
                        sampler=None,
                        collate_fn=text_collate,
                        num_workers=args.num_workers,
                        pin_memory=torch.cuda.is_available(),
                        worker_init_fn=_wi_ft,
                        generator=_gen_ft,
                        persistent_workers=_fpw_ft,
                        prefetch_factor=_fpf_ft,
                        drop_last=use_ddp,
                    )
                else:
                    ft_loader = _text_loader(finetune_ds)

            if isinstance(inner, LLMModule):
                if pre_loader is not None:
                    make_pre = lambda epoch_next, steps_done=0: ResumeMeta(
                        cfg_path=str(cfg_path),
                        dataset_stem=args.dataset_stem,
                        n_labels=n_classes,
                        phase=args.phase,
                        segment="pretrain",
                        epoch_next=epoch_next,
                        epochs_segment=args.epochs_pretrain,
                        lr=args.lr,
                        class_name=class_name,
                        head_only=True,
                        steps_completed_in_epoch=steps_done,
                    )
                    train_loop_llm(
                        model,
                        pre_loader,
                        device,
                        args.epochs_pretrain,
                        args.lr,
                        head_only=True,
                        gc_every=args.gc_every,
                        live=live,
                        make_meta=make_pre,
                        interrupt=interrupt,
                        use_ddp_module=use_ddp_module,
                        save_every_steps=args.save_every_steps,
                        save_every_minutes=args.save_every_minutes,
                        min_save_interval_sec=args.min_save_interval_sec,
                    )
                    if interrupt.requested:
                        return
                    _clear_live_resume(live, use_ddp)
                if ft_loader is not None:
                    make_ft = lambda epoch_next, steps_done=0: ResumeMeta(
                        cfg_path=str(cfg_path),
                        dataset_stem=args.dataset_stem,
                        n_labels=n_classes,
                        phase=args.phase,
                        segment="finetune",
                        epoch_next=epoch_next,
                        epochs_segment=args.epochs_finetune,
                        lr=args.lr,
                        class_name=class_name,
                        head_only=True,
                        steps_completed_in_epoch=steps_done,
                    )
                    train_loop_llm(
                        model,
                        ft_loader,
                        device,
                        args.epochs_finetune,
                        args.lr,
                        head_only=True,
                        gc_every=args.gc_every,
                        live=live,
                        make_meta=make_ft,
                        interrupt=interrupt,
                        use_ddp_module=use_ddp_module,
                        save_every_steps=args.save_every_steps,
                        save_every_minutes=args.save_every_minutes,
                        min_save_interval_sec=args.min_save_interval_sec,
                    )
                    if interrupt.requested:
                        return
                    _clear_live_resume(live, use_ddp)
            elif _hrm_encoder_module(inner) is not None:
                if pre_loader is not None:
                    make_mlm = lambda epoch_next, steps_done=0: ResumeMeta(
                        cfg_path=str(cfg_path),
                        dataset_stem=args.dataset_stem,
                        n_labels=resume_n_labels_for_live,
                        phase=args.phase,
                        segment="hrm_mlm",
                        epoch_next=epoch_next,
                        epochs_segment=args.epochs_pretrain,
                        lr=args.lr,
                        class_name=class_name,
                        head_only=None,
                        steps_completed_in_epoch=steps_done,
                    )
                    train_loop_hrm_mlm(
                        model,
                        pre_loader,
                        device,
                        args.epochs_pretrain,
                        args.lr,
                        gc_every=args.gc_every,
                        live=live,
                        make_meta=make_mlm,
                        interrupt=interrupt,
                        use_ddp_module=use_ddp_module,
                        save_every_steps=args.save_every_steps,
                        save_every_minutes=args.save_every_minutes,
                        min_save_interval_sec=args.min_save_interval_sec,
                        amp_bf16=args.amp_bf16,
                    )
                    if interrupt.requested:
                        return
                    _clear_live_resume(live, use_ddp)
                    if (
                        args.phase == "all"
                        and ft_loader is not None
                        and isinstance(_unwrap(model), HierarchicalReasoningModel)
                    ):
                        if use_ddp and dist.is_available() and dist.is_initialized():
                            dist.barrier()
                        if is_main_process():
                            pre_out = _hrm_pretrain_checkpoint_path(
                                ckpt_root, args.dataset_stem, cfg_path.stem
                            )
                            pre_out.parent.mkdir(parents=True, exist_ok=True)
                            save_safetensors(_unwrap(model).state_dict(), pre_out)
                            print("Saved HRM encoder MLM pretrain", pre_out, flush=True)
                        if use_ddp and dist.is_available() and dist.is_initialized():
                            dist.barrier()
                        inner_enc = _unwrap(model)
                        wrapped = HRMClassifierWrapper(inner_enc, n_classes)
                        wrapped = wrapped.to(device)
                        if use_ddp:
                            model = wrap_ddp(wrapped, local_rank, find_unused_parameters=True)
                        else:
                            model = wrapped
                        use_ddp_module = isinstance(model, nn.parallel.DistributedDataParallel)
                        inner = model.module if use_ddp_module else model
                if ft_loader is not None:
                    _freeze_hrm_encoder_train_head_only(_unwrap(model))
                    make_cls = lambda epoch_next, steps_done=0: ResumeMeta(
                        cfg_path=str(cfg_path),
                        dataset_stem=args.dataset_stem,
                        n_labels=n_classes,
                        phase=args.phase,
                        segment="hrm_supervised",
                        epoch_next=epoch_next,
                        epochs_segment=args.epochs_finetune,
                        lr=args.lr,
                        class_name=class_name,
                        head_only=None,
                        steps_completed_in_epoch=steps_done,
                    )
                    train_loop_hrm_supervised(
                        model,
                        ft_loader,
                        device,
                        args.epochs_finetune,
                        args.lr,
                        gc_every=args.gc_every,
                        live=live,
                        make_meta=make_cls,
                        interrupt=interrupt,
                        use_ddp_module=use_ddp_module,
                        save_every_steps=args.save_every_steps,
                        save_every_minutes=args.save_every_minutes,
                        min_save_interval_sec=args.min_save_interval_sec,
                    )
                    if interrupt.requested:
                        return
                    _clear_live_resume(live, use_ddp)
            else:
                if ft_loader is None:
                    if is_main_process():
                        print(
                            "Text-mode config expects LLMModule, HRMClassifierWrapper, or HierarchicalReasoningModel."
                        )
                    sys.exit(1)
                make_ft_tensor = lambda epoch_next, steps_done=0: ResumeMeta(
                    cfg_path=str(cfg_path),
                    dataset_stem=args.dataset_stem,
                    n_labels=n_classes,
                    phase=args.phase,
                    segment="finetune",
                    epoch_next=epoch_next,
                    epochs_segment=args.epochs_finetune,
                    lr=args.lr,
                    class_name=class_name,
                    head_only=None,
                    steps_completed_in_epoch=steps_done,
                )
                train_loop_tensor(
                    model,
                    ft_loader,
                    device,
                    args.epochs_finetune,
                    args.lr,
                    class_name,
                    live=live,
                    make_meta=make_ft_tensor,
                    interrupt=interrupt,
                    use_ddp_module=use_ddp_module,
                    save_every_steps=args.save_every_steps,
                    save_every_minutes=args.save_every_minutes,
                    min_save_interval_sec=args.min_save_interval_sec,
                )
                if interrupt.requested:
                    return
                _clear_live_resume(live, use_ddp)

        if is_main_process():
            ufin = _unwrap(model)
            if isinstance(ufin, HierarchicalReasoningModel):
                out = _hrm_pretrain_checkpoint_path(ckpt_root, args.dataset_stem, cfg_path.stem)
            elif isinstance(ufin, HRMClassifierWrapper) and args.hrm_finetune_checkpoint_layout:
                out = _hrm_finetune_checkpoint_path(
                    ckpt_root, args.dataset_stem, n_classes, cfg_path.stem
                )
            else:
                out = _checkpoint_path(ckpt_root, n_classes, args.dataset_stem, cfg_path.stem)
            out.parent.mkdir(parents=True, exist_ok=True)
            state = model.module.state_dict() if use_ddp else model.state_dict()
            save_safetensors(state, out)
            print("Saved", out)
            if isinstance(ufin, HierarchicalReasoningModel) and not args.no_save_hrm_tokenizer:
                tok_dir = (ckpt_root / "tokenizer").resolve()
                tok_dir.mkdir(parents=True, exist_ok=True)
                tok = getattr(ufin, "tokenizer", None)
                if tok is not None and hasattr(tok, "save_pretrained"):
                    tok.save_pretrained(str(tok_dir))
                    print("Saved HRM tokenizer", tok_dir, flush=True)
        if use_ddp and dist.is_available() and dist.is_initialized():
            dist.barrier()
    finally:
        if use_ddp and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

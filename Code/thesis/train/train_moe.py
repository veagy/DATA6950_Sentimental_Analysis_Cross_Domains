"""
Fine-tune the MoE gate (and optional tiny PEFT LoRA on one LLM expert) while experts stay frozen.

Experts JSON: list of {config, checkpoint, modality}; see Code/thesis/config/moe/README.md.

Distributed: launch with ``torchrun --nproc_per_node=2 Code/thesis/train/train_moe.py ...`` (NCCL, CUDA).

Periodic checkpoints (``--checkpoint-dir`` + ``--save-every-steps``) write ``checkpoint_latest.pt`` for resume
(``--resume``). Optional ``--auto-batch-vram-target 0.8`` binary-searches batch size using CUDA peak memory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from Code.thesis.common.pkg_bootstrap import install_lazy_code_models

install_lazy_code_models(_REPO)

from Code.models.deep_learning.llm.llm_models import LLMModule
from Code.thesis.common.checkpoint_io import load_safetensors_state, save_safetensors
from Code.thesis.common.datasets import ParquetFeaturesDataset, ParquetTextDataset
from Code.thesis.common.distributed import (
    init_distributed_from_env,
    is_main_process,
    make_sampler,
    set_sampler_epoch,
    wrap_ddp,
)
from Code.thesis.common.model_factory import build_model_from_config_dict, load_config
from Code.thesis.train.moe_facade import FeatureGatedMoE, HeterogeneousMoE


def _moe_raw(moe: nn.Module) -> nn.Module:
    return moe.module if hasattr(moe, "module") else moe


def _gate_backend_name(raw: nn.Module) -> str:
    return "features" if isinstance(raw, FeatureGatedMoE) else "distilbert"


def _lora_trainable_count(module: nn.Module) -> int:
    return sum(p.numel() for n, p in module.named_parameters() if p.requires_grad and "lora_" in n.lower())


def _attach_lora_peft(
    llm: LLMModule,
    *,
    rank: int,
    lora_preset: str,
    lora_max_params: int | None,
) -> int:
    """Attach PEFT LoRA to first LLM expert; return LoRA trainable param count."""
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as e:
        raise RuntimeError("[train_moe] peft required for LoRA") from e
    if llm.model is None:
        return 0

    if lora_preset == "tiny10k":
        cfg = LoraConfig(
            r=max(1, rank) if rank > 0 else 3,
            lora_alpha=max(6, rank * 2) if rank > 0 else 6,
            target_modules=["q_lin", "v_lin"],
            layers_to_transform=[5],
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
    else:
        cfg = LoraConfig(
            r=max(1, rank),
            lora_alpha=max(rank * 2, 8),
            target_modules=["q_lin", "v_lin"],
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )

    llm.model = get_peft_model(llm.model, cfg)
    llm.set_backbone_trainable(True)
    for n, p in llm.named_parameters():
        if "lora_" not in n.lower():
            p.requires_grad = False

    n_lora = _lora_trainable_count(llm)
    cap = lora_max_params or (10_000 if lora_preset == "tiny10k" else None)
    if cap is not None and n_lora > cap and is_main_process():
        print(
            f"[train_moe] warning: LoRA trainable {n_lora} exceeds cap {cap} (try --lora-rank 2 or 1)",
            flush=True,
        )
    return n_lora


def _replace_distilbert_with_4bit(llm: LLMModule, device_index: int) -> bool:
    """Swap backbone for NF4 DistilBERT (QLoRA-style base). Classifier stays from checkpoint."""
    try:
        from transformers import BitsAndBytesConfig, DistilBertModel
    except ImportError:
        print("[train_moe] transformers missing; skip 4-bit", file=sys.stderr)
        return False
    if not torch.cuda.is_available():
        print("[train_moe] --load-in-4bit needs CUDA; skip", file=sys.stderr)
        return False
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        print("[train_moe] bitsandbytes not installed; skip 4-bit", file=sys.stderr)
        return False
    try:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        dm = DistilBertModel.from_pretrained(
            "distilbert-base-uncased",
            quantization_config=bnb,
            device_map={"": int(device_index)},
        )
    except Exception as e:
        print(f"[train_moe] 4-bit DistilBERT load failed: {e}", file=sys.stderr)
        return False
    llm.model = dm
    llm.set_backbone_trainable(False)
    try:
        from peft import prepare_model_for_kbit_training

        llm.model = prepare_model_for_kbit_training(llm.model)
    except Exception:
        pass
    return True


class DualDataset(torch.utils.data.Dataset):
    """Aligned processed (text, y) + transformed (features) rows."""

    def __init__(
        self,
        stem: str,
        data_root: Path,
        max_samples: int | None,
        label_n_classes: int | None = None,
    ):
        self.text_ds = ParquetTextDataset(
            str(data_root / "processed" / f"{stem}.parquet"),
            max_samples=max_samples,
            n_classes=label_n_classes,
        )
        self.feat_ds = ParquetFeaturesDataset(
            str(data_root / "transformed" / f"{stem}.parquet"),
            max_samples=max_samples,
            n_classes=label_n_classes,
        )
        n = min(len(self.text_ds), len(self.feat_ds))
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        t, y = self.text_ds[i]
        x, _y2 = self.feat_ds[i]
        return t, x, y


def dual_collate(batch):
    texts = [b[0] for b in batch]
    feats = torch.stack([b[1] for b in batch])
    y = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return texts, feats, y


def _lora_state_dict(expert: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in expert.state_dict().items() if "lora_" in k.lower()}


def _save_moe_training_checkpoint(
    path: Path,
    *,
    moe: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    batch_idx: int,
    global_step: int,
    lora_expert_idx: int | None,
    batch_size: int,
) -> None:
    raw = _moe_raw(moe)
    payload = {
        "epoch": int(epoch),
        "batch_idx": int(batch_idx),
        "global_step": int(global_step),
        "batch_size": int(batch_size),
        "num_experts": len(raw.experts),
        "n_classes": int(raw.n_classes),
        "gate_backend": _gate_backend_name(raw),
        "lora_expert_idx": lora_expert_idx,
        "gate": {k: v.detach().cpu() for k, v in raw.gate.state_dict().items()},
        "optimizer": optimizer.state_dict(),
    }
    if lora_expert_idx is not None:
        payload["lora"] = _lora_state_dict(raw.experts[lora_expert_idx])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _load_moe_training_checkpoint(
    path: Path,
    moe: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, int, int, int]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    raw = _moe_raw(moe)
    saved_ne = payload.get("num_experts")
    if saved_ne is not None and int(saved_ne) != len(raw.experts):
        raise ValueError(
            f"Resume checkpoint num_experts={saved_ne} but current MoE has {len(raw.experts)} experts. "
            "Delete the checkpoint dir or run without --resume."
        )
    saved_gb = payload.get("gate_backend", "distilbert")
    cur_gb = _gate_backend_name(raw)
    if str(saved_gb) != cur_gb:
        raise ValueError(
            f"Resume checkpoint gate_backend={saved_gb!r} but current run uses {cur_gb!r}. "
            "Remove checkpoint or use the same gate mode."
        )
    raw.gate.load_state_dict(payload["gate"])
    li = payload.get("lora_expert_idx")
    if li is not None and payload.get("lora"):
        expert = raw.experts[int(li)]
        cur = expert.state_dict()
        for k, v in payload["lora"].items():
            if k in cur:
                cur[k] = v.to(device)
        expert.load_state_dict(cur, strict=False)
    optimizer.load_state_dict(payload["optimizer"])
    return (
        int(payload["epoch"]),
        int(payload.get("batch_idx", 0)),
        int(payload.get("global_step", 0)),
        int(payload.get("batch_size", 4)),
    )


def _broadcast_int(value: int, device: torch.device, use_ddp: bool) -> int:
    if not use_ddp:
        return value
    import torch.distributed as dist

    t = torch.tensor([value], device=device, dtype=torch.long)
    dist.broadcast(t, src=0)
    return int(t.item())


def _probe_batch_size_vram(
    moe: nn.Module,
    ds: torch.utils.data.Dataset,
    modalities: list[str],
    device: torch.device,
    *,
    min_bs: int,
    max_bs: int,
    target_frac: float,
) -> int:
    if not torch.cuda.is_available():
        return max(1, min_bs)
    crit = nn.CrossEntropyLoss()
    total = torch.cuda.get_device_properties(device).total_memory
    target_bytes = int(total * float(target_frac))

    def try_bs(bs: int) -> tuple[bool, int]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        moe.train()
        n_take = min(bs, len(ds))
        if n_take < 1:
            return True, 0
        subset = torch.utils.data.Subset(ds, list(range(n_take)))
        loader = DataLoader(
            subset,
            batch_size=bs,
            shuffle=False,
            collate_fn=dual_collate,
            num_workers=0,
            drop_last=False,
        )
        texts, feats, y = next(iter(loader))
        feats = feats.to(device)
        y = y.to(device)
        try:
            trainable = [p for p in moe.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(trainable, lr=1e-4)
            opt.zero_grad(set_to_none=True)
            logits = moe(texts, feats, expert_modalities=modalities)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            peak = torch.cuda.max_memory_allocated(device)
            return True, peak
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                return False, 0
            raise

    lo = max(1, min_bs)
    hi = max(max_bs, lo)
    best = 0  # largest batch that succeeded in the exponential phase
    cur = lo
    while cur <= hi:
        ok, peak = try_bs(cur)
        if not ok:
            break
        best = cur
        if peak >= target_bytes * 0.82:
            break
        if cur >= hi:
            break
        cur = min(hi, max(cur * 2, cur + 1))

    if best == 0:
        low, high = 1, lo - 1
        ans = 1
    else:
        low, high = lo, min(hi, best * 2 + 32)
        ans = best
    while low <= high:
        mid = (low + high) // 2
        ok, peak = try_bs(mid)
        if ok:
            ans = mid
            low = mid + 1
        else:
            high = mid - 1
    torch.cuda.empty_cache()
    if is_main_process():
        print(f"[train_moe] auto batch_size={ans} (target_vram_frac={target_frac})", flush=True)
    return max(1, ans)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts_json", type=Path, required=True)
    ap.add_argument("--dataset_stem", type=str, required=True)
    ap.add_argument("--data_root", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out_path", type=Path, default=None)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--sparse_top_k", type=int, default=2)
    ap.add_argument(
        "--gate-hidden-dim",
        type=int,
        default=256,
        help="Gating MLP hidden width; use 0 for a single linear gate (fewer trainable params).",
    )
    ap.add_argument("--lora-rank", type=int, default=0)
    ap.add_argument(
        "--lora-preset",
        choices=("none", "default", "tiny10k"),
        default="none",
        help="tiny10k: last DistilBERT layer, q_lin+v_lin, budget ~10k trainable LoRA params.",
    )
    ap.add_argument("--lora-max-params", type=int, default=None)
    ap.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Replace DistilBERT backbone with NF4 weights before LoRA (QLoRA-style; CUDA+bnb).",
    )
    ap.add_argument("--sync-labels", action="store_true")
    ap.add_argument(
        "--text-gate",
        action="store_true",
        help="Use frozen DistilBERT for gate embeddings even when every expert is dense "
        "(default: feature-only gate on feats—no transformer in the forward pass).",
    )
    ap.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory for checkpoint_latest.pt (periodic + resume).",
    )
    ap.add_argument(
        "--save-every-steps",
        type=int,
        default=0,
        help="If >0, save checkpoint_latest.pt every N optimizer steps (rank 0 only).",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Load checkpoint_latest.pt from --checkpoint-dir (or --resume-path).",
    )
    ap.add_argument("--resume-path", type=Path, default=None)
    ap.add_argument(
        "--auto-batch-vram-target",
        type=float,
        default=None,
        help="If set (e.g. 0.8), binary-search batch size on CUDA to approach this VRAM fraction.",
    )
    ap.add_argument("--auto-batch-min", type=int, default=1)
    ap.add_argument("--auto-batch-max", type=int, default=4096)
    args = ap.parse_args()

    use_ddp, local_rank, world_sz = init_distributed_from_env()
    if use_ddp and local_rank is not None:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = args.data_root or (_REPO / "data")

    with open(args.experts_json, encoding="utf-8") as f:
        spec = json.load(f)

    experts: list[nn.Module] = []
    modalities: list[str] = []
    n_classes: int | None = None
    lora_llm_idx: int | None = None
    for i, entry in enumerate(spec):
        cfg_p = Path(entry["config"])
        if not cfg_p.is_absolute():
            cfg_p = _REPO / cfg_p
        cfg = load_config(cfg_p)
        stem = cfg_p.as_posix()
        if "/2_labels/" in stem:
            nc = 2
        elif "/3_labels/" in stem:
            nc = 3
        else:
            raise ValueError("Expert config must live under 2_labels or 3_labels")
        if n_classes is None:
            n_classes = nc
        elif n_classes != nc:
            raise ValueError("All experts must share label count")
        model, _ = build_model_from_config_dict(cfg, nc, "")
        ck = Path(entry["checkpoint"])
        if not ck.is_absolute():
            ck = _REPO / ck
        if ck.is_file():
            model.load_state_dict(load_safetensors_state(ck, map_location="cpu"), strict=False)

        want_lora = args.lora_preset != "none" or args.lora_rank > 0
        if want_lora and isinstance(model, LLMModule) and lora_llm_idx is None:
            if args.load_in_4bit:
                di = int(local_rank) if use_ddp and local_rank is not None else 0
                if torch.cuda.is_available():
                    _replace_distilbert_with_4bit(model, di)
            preset = args.lora_preset if args.lora_preset != "none" else "default"
            n_lora = _attach_lora_peft(
                model,
                rank=args.lora_rank,
                lora_preset=preset,
                lora_max_params=args.lora_max_params,
            )
            if is_main_process():
                print(f"[train_moe] LoRA trainable params (expert {i}): {n_lora}", flush=True)
            lora_llm_idx = i

        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        if lora_llm_idx == i:
            for n, p in model.named_parameters():
                if "lora_" in n.lower():
                    p.requires_grad = True
        experts.append(model.to(device))
        modalities.append(entry.get("modality", "text"))

    assert n_classes is not None
    label_nc = n_classes if args.sync_labels else None
    rk = ""
    if use_ddp and local_rank is not None:
        rk = f" rank={local_rank}"
    print(
        f"[train_moe] loading dual dataset stem={args.dataset_stem}{rk} "
        f"(processed+transformed parquets; can take many minutes on full data)...",
        flush=True,
    )
    ds = DualDataset(args.dataset_stem, data_root, args.max_samples, label_n_classes=label_nc)
    print(f"[train_moe] dataset ready len={len(ds)}{rk}", flush=True)
    if len(ds) < 1:
        raise SystemExit("[train_moe] dataset is empty (check parquets and --sync-labels).")

    gh = None if args.gate_hidden_dim <= 0 else int(args.gate_hidden_dim)
    all_dense = bool(modalities) and all(m == "dense" for m in modalities)
    if all_dense and not args.text_gate:
        feat_dim = int(ds[0][1].shape[0])
        moe = FeatureGatedMoE(
            n_classes=n_classes,
            expert_modules=nn.ModuleList(experts),
            feat_dim=feat_dim,
            sparse_top_k=args.sparse_top_k,
            gate_hidden_dim=gh,
        )
        if is_main_process():
            print(
                f"[train_moe] gate=features only (feat_dim={feat_dim}); "
                f"no DistilBERT/transformer/HRM in forward",
                flush=True,
            )
    else:
        moe = HeterogeneousMoE(
            n_classes=n_classes,
            expert_modules=nn.ModuleList(experts),
            sparse_top_k=args.sparse_top_k,
            gate_hidden_dim=gh,
        )
        if is_main_process():
            print("[train_moe] gate=DistilBERT text encoder (frozen) + trainable gate MLP", flush=True)
    moe = moe.to(device)

    ckpt_dir = args.checkpoint_dir
    if ckpt_dir is None and (args.save_every_steps > 0 or args.resume):
        ckpt_dir = _REPO / "checkpoints" / "moe" / "training_state"
    if ckpt_dir is not None:
        ckpt_dir = Path(ckpt_dir).resolve()

    resume_file = args.resume_path
    if args.resume and resume_file is None and ckpt_dir is not None:
        resume_file = ckpt_dir / "checkpoint_latest.pt"
    do_resume = bool(resume_file and resume_file.is_file() and args.resume)

    if args.auto_batch_vram_target is not None and torch.cuda.is_available() and not do_resume:
        if not use_ddp or is_main_process():
            args.batch_size = _probe_batch_size_vram(
                moe,
                ds,
                modalities,
                device,
                min_bs=args.auto_batch_min,
                max_bs=args.auto_batch_max,
                target_frac=float(args.auto_batch_vram_target),
            )
        args.batch_size = _broadcast_int(int(args.batch_size), device, use_ddp)
    elif use_ddp:
        args.batch_size = _broadcast_int(int(args.batch_size), device, use_ddp)

    sampler = make_sampler(ds, shuffle=True, use_ddp=use_ddp)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=dual_collate,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=bool(use_ddp),
    )

    if use_ddp:
        moe = wrap_ddp(moe, int(local_rank), find_unused_parameters=True)

    train_params = [p for p in moe.parameters() if p.requires_grad]
    if is_main_process():
        n_tr = sum(p.numel() for p in train_params)
        print(f"[train_moe] total trainable params: {n_tr}", flush=True)
    opt = torch.optim.AdamW(train_params, lr=args.lr)
    crit = nn.CrossEntropyLoss()

    start_epoch = 0
    start_batch = 0
    global_step = 0
    if do_resume and resume_file is not None:
        se, sb, gs, saved_bs = _load_moe_training_checkpoint(resume_file, moe, opt, device)
        start_epoch = se
        start_batch = sb
        global_step = gs
        if saved_bs != args.batch_size and is_main_process():
            print(
                f"[train_moe] resume: checkpoint batch_size={saved_bs}, current={args.batch_size} (keeping current)",
                flush=True,
            )
        if is_main_process():
            print(
                f"[train_moe] resumed from {resume_file} epoch={start_epoch} batch_idx={start_batch} step={global_step}",
                flush=True,
            )

    latest_path = (ckpt_dir / "checkpoint_latest.pt") if ckpt_dir is not None else None

    moe.train()
    for epoch in range(start_epoch, args.epochs):
        set_sampler_epoch(sampler, epoch)
        bar = tqdm(
            loader,
            desc=f"moe ep{epoch + 1}/{args.epochs}",
            disable=not is_main_process(),
        )
        for batch_idx, (texts, feats, y) in enumerate(bar):
            if epoch == start_epoch and batch_idx < start_batch:
                continue
            feats = feats.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = moe(texts, feats, expert_modalities=modalities)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            global_step += 1
            if is_main_process():
                bar.set_postfix(loss=float(loss.item()))
            next_batch = batch_idx + 1
            if (
                latest_path is not None
                and args.save_every_steps > 0
                and is_main_process()
                and global_step % args.save_every_steps == 0
            ):
                _save_moe_training_checkpoint(
                    latest_path,
                    moe=moe,
                    optimizer=opt,
                    epoch=epoch,
                    batch_idx=next_batch,
                    global_step=global_step,
                    lora_expert_idx=lora_llm_idx,
                    batch_size=args.batch_size,
                )
                print(f"[train_moe] checkpoint step={global_step} -> {latest_path}", flush=True)
        start_batch = 0

    if latest_path is not None and is_main_process() and args.save_every_steps > 0:
        _save_moe_training_checkpoint(
            latest_path,
            moe=moe,
            optimizer=opt,
            epoch=args.epochs,
            batch_idx=0,
            global_step=global_step,
            lora_expert_idx=lora_llm_idx,
            batch_size=args.batch_size,
        )

    if is_main_process():
        raw = _moe_raw(moe)
        out = args.out_path or (_REPO / "checkpoints" / "moe" / f"gate_{args.dataset_stem}.safetensors")
        out.parent.mkdir(parents=True, exist_ok=True)
        save_safetensors(raw.gate.state_dict(), out)
        print("Saved gate to", out, flush=True)

    if use_ddp:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

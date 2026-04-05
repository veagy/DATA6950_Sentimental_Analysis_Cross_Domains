#!/usr/bin/env python3
"""
Load HRM encoder-only weights (config + safetensors), run mean-pooled embeddings on sample text.

Run from repository root:
  python Code/thesis/tools/hrm_embed_smoke.py
  python Code/thesis/tools/hrm_embed_smoke.py \\
    --config Code/thesis/config/hrm/E_HRM1_4Level.json \\
    --checkpoint checkpoints/hrm/pretrain/all-data/E_HRM1_4Level.safetensors \\
    --tokenizer-dir checkpoints/hrm/tokenizer
  python Code/thesis/tools/hrm_embed_smoke.py \\
    --parquet data/processed/all-data.parquet --parquet-rows 8

Uses ``HierarchicalReasoningModel(..., pretrain=False)`` → [B, output_embed_dim] (see hrm_model.py).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import torch

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

from Code.thesis.common.pkg_bootstrap import install_lazy_code_models

install_lazy_code_models(_REPO)

from Code.models.deep_learning.hrm.hrm_model import HierarchicalReasoningModel
from Code.thesis.common.checkpoint_io import load_safetensors_state
from Code.thesis.common.model_factory import build_model_from_config_dict, load_config

# Some Python/env combinations fail the first HRM/EncoderLM init (activation import graph) then succeed.
_hrm_import_prime_done = False


def ensure_hrm_encoder_imports_primed() -> None:
    global _hrm_import_prime_done
    if _hrm_import_prime_done:
        return
    _hrm_import_prime_done = True
    try:
        from Code.models.deep_learning.hrm.hrm_model import HierarchicalReasoningModel, HRMConfig

        cfg = HRMConfig(
            batch_size=1,
            seq_len=8,
            vocab_size=64,
            hidden_size=32,
            output_embed_dim=8,
            H_cycles=1,
            L_cycles=1,
            h_level_model="EncoderLM",
            l_level_model="EncoderLM",
            model_kwargs={"num_layers": 1, "num_heads": 4},
            tokenizer_name=None,
        )
        HierarchicalReasoningModel(config=cfg)
    except Exception:
        pass


DEFAULT_SAMPLES = (
    "Short review: great product.",
    "Another sample: disappointing experience overall.",
    "Neutral comment about the service.",
)


def _strip_module_prefix(sd: dict) -> dict:
    if not sd or not any(k.startswith("module.") for k in sd):
        return sd
    out: dict = {}
    for k, v in sd.items():
        if k.startswith("module."):
            out[k[len("module.") :]] = v
        else:
            out[k] = v
    return out


def load_hrm_encoder(
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    tokenizer_dir: Path | None,
) -> HierarchicalReasoningModel:
    ensure_hrm_encoder_imports_primed()
    cfg = load_config(config_path)
    model, class_name = build_model_from_config_dict(cfg, 2, "", hrm_encoder_only=True)
    if class_name != "HierarchicalReasoningModel" or not isinstance(
        model, HierarchicalReasoningModel
    ):
        raise TypeError(f"Expected bare HierarchicalReasoningModel, got {class_name!r}")

    sd = _strip_module_prefix(load_safetensors_state(str(checkpoint_path), map_location="cpu"))
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError:
        incomp = model.load_state_dict(sd, strict=False)
        miss = getattr(incomp, "missing_keys", None) or []
        unexp = getattr(incomp, "unexpected_keys", None) or []
        if miss:
            raise RuntimeError(f"Checkpoint missing keys ({len(miss)}): {miss[:12]}") from None
        if unexp:
            print(f"Warning: unexpected checkpoint keys ({len(unexp)}): {unexp[:12]}")

    if tokenizer_dir is not None and tokenizer_dir.is_dir():
        from transformers import AutoTokenizer

        model.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))

    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def embed_texts(model: HierarchicalReasoningModel, texts: Sequence[str]) -> torch.Tensor:
    """Return float tensor [B, output_embed_dim]."""
    dev = next(model.parameters()).device
    with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda", dtype=torch.bfloat16):
        out = model(list(texts), pretrain=False)
    return out.float().cpu()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="HRM encoder embedding smoke test")
    ap.add_argument(
        "--config",
        type=Path,
        default=_REPO / "Code" / "thesis" / "config" / "hrm" / "E_HRM1_4Level.json",
    )
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=_REPO / "checkpoints" / "hrm" / "pretrain" / "all-data" / "E_HRM1_4Level.safetensors",
    )
    ap.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=None,
        help="Optional: local save_pretrained dir (e.g. checkpoints/hrm/tokenizer); else HF cache from config",
    )
    ap.add_argument("--device", type=str, default=None, help="cpu | cuda | cuda:0 (default: cuda if available)")
    ap.add_argument(
        "--samples-json",
        type=Path,
        default=None,
        help="JSON array of strings; default: built-in short samples",
    )
    ap.add_argument("--print-json", action="store_true", help="Print embeddings as JSON (rounded)")
    ap.add_argument(
        "--parquet",
        type=Path,
        default=None,
        help="Read first --parquet-rows rows from this parquet (text column); overrides built-in samples unless --samples-json is set",
    )
    ap.add_argument(
        "--parquet-text-column",
        type=str,
        default="text",
        help="Column name for text when using --parquet (default: text)",
    )
    ap.add_argument(
        "--parquet-rows",
        type=int,
        default=8,
        help="Number of rows to take from the start of --parquet (default: 8)",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    cfg_p = args.config.resolve()
    ckpt_p = args.checkpoint.resolve()
    if not cfg_p.is_file():
        print(f"ERROR: config not found: {cfg_p}", file=sys.stderr)
        return 2
    if not ckpt_p.is_file():
        print(f"ERROR: checkpoint not found: {ckpt_p}", file=sys.stderr)
        return 2

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok_dir = args.tokenizer_dir.resolve() if args.tokenizer_dir else None
    if args.tokenizer_dir is not None and not tok_dir.is_dir():
        print(f"WARNING: --tokenizer-dir not a directory: {tok_dir}", file=sys.stderr)

    if args.samples_json is not None:
        texts = json.loads(args.samples_json.read_text(encoding="utf-8"))
    elif args.parquet is not None:
        pq_p = args.parquet.resolve()
        if not pq_p.is_file():
            print(f"ERROR: parquet not found: {pq_p}", file=sys.stderr)
            return 2
        try:
            import pandas as pd
        except ImportError as e:
            print("ERROR: --parquet requires pandas", file=sys.stderr)
            raise SystemExit(2) from e
        n = max(1, int(args.parquet_rows))
        df = pd.read_parquet(pq_p, columns=[args.parquet_text_column]).head(n)
        col = args.parquet_text_column
        if col not in df.columns:
            print(f"ERROR: column {col!r} not in {pq_p} (have {list(df.columns)})", file=sys.stderr)
            return 2
        texts = [str(x) for x in df[col].tolist()]
    else:
        texts = list(DEFAULT_SAMPLES)

    print(f"Repo:      {_REPO}")
    print(f"Config:    {cfg_p}")
    print(f"Checkpoint:{ckpt_p}")
    print(f"Device:    {device}")
    print(f"Samples:   {len(texts)} strings")
    print("Loading model...", flush=True)

    model = load_hrm_encoder(cfg_p, ckpt_p, device, tok_dir)
    d = int(model.hrm_config.output_embed_dim)
    print(f"output_embed_dim={d}", flush=True)

    emb = embed_texts(model, texts)
    assert emb.shape == (len(texts), d), emb.shape
    print(f"Embeddings shape: {tuple(emb.shape)}", flush=True)
    for i, t in enumerate(texts):
        row = emb[i]
        print(f"  [{i}] norm={row.norm().item():.4f}  preview={row[:5].tolist()}  text={t[:60]!r}")

    if args.print_json:
        print(json.dumps(emb.tolist(), indent=0))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

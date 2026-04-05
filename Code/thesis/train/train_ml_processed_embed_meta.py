"""
docs/ml Tracks B and C (practical subset): frozen DistilBERT mean-pooled embeddings from
``data/processed/{stem}.parquet``, then either a trainable linear head (B) or sklearn
LogisticRegression + LinearSVC (C). Aligns with TRAINING_PIPELINES.md §4–5; full multi-encoder
MoE stacks remain future work.

Outputs (under ``checkpoint_root``):
  Track B: ``moe/ml_stack/{n}-labels/{stem}/trackB_DistilBERT_Linear.safetensors``
  Track C: ``moe/ml_stack/{n}-labels/{stem}/proc_LogisticRegression_DistilBERT.joblib``
           ``moe/ml_stack/{n}-labels/{stem}/proc_LinearSVC_DistilBERT.joblib``
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from Code.thesis.common.pkg_bootstrap import install_lazy_code_models

install_lazy_code_models(_REPO)

from Code.thesis.common.checkpoint_io import load_safetensors_state, save_safetensors  # noqa: E402
from Code.thesis.common.datasets import (  # noqa: E402
    _infer_label_column,
    _infer_text_column,
    coerce_label_int,
    normalize_label_for_n_classes,
)
from Code.thesis.common.model_factory import build_model_from_config_dict, load_config  # noqa: E402

try:
    import pyarrow.parquet as pq
except ImportError as e:  # pragma: no cover
    raise SystemExit("pyarrow is required for train_ml_processed_embed_meta.py") from e


def _parquet_num_rows(path: Path) -> int:
    return int(pq.ParquetFile(str(path)).metadata.num_rows)


def _load_queue_spec(path: Path) -> tuple[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    track = str(data.get("_ml_queue_track", "")).lower().strip()
    n_labels = int(data.get("_ml_queue_n_labels", 0))
    if track not in ("b", "c") or n_labels not in (2, 3):
        raise ValueError(f"Invalid queue config {path}: expected _ml_queue_track b|c and n_labels 2|3")
    return track, n_labels


def _default_encoder_config(repo: Path, n_labels: int) -> Path:
    sub = "2_labels" if n_labels == 2 else "3_labels"
    p = repo / "Code/thesis/config/transformers" / sub / "B3_E_DL1_DistilBERT.json"
    if not p.is_file():
        p = repo / "Code/thesis/config/transformers" / sub / "B3_E_DL1_DistilBERT_mlp768_1024.json"
    return p


def _default_encoder_ckpt(checkpoint_root: Path, n_labels: int, stem: str) -> Path | None:
    d = checkpoint_root / f"{n_labels}-labels" / stem
    if not d.is_dir():
        return None
    for name in (
        "B3_E_DL1_DistilBERT_mlp768_1024.safetensors",
        "B3_E_DL1_DistilBERT.safetensors",
    ):
        p = d / name
        if p.is_file():
            return p
    return None


def _out_dir(checkpoint_root: Path, n_labels: int, stem: str) -> Path:
    return (checkpoint_root / "moe" / "ml_stack" / f"{n_labels}-labels" / stem).resolve()


def _iter_text_label_batches(
    parquet_path: Path,
    n_classes: int,
    batch_rows: int,
    text_col: str | None,
    label_col: str | None,
):
    pf = pq.ParquetFile(str(parquet_path))
    schema_names = list(pf.schema_arrow.names)
    tc = text_col or _infer_text_column(schema_names)
    lc = label_col or _infer_label_column(schema_names)
    stem_col = "source_stem" if "source_stem" in schema_names else None
    if not tc or not lc:
        raise ValueError(f"{parquet_path}: need text + label columns (got tc={tc!r} lc={lc!r})")
    cols = [c for c in (tc, lc, stem_col) if c is not None]
    for batch in pf.iter_batches(batch_size=batch_rows, columns=cols):
        df = batch.to_pandas()
        texts: list[str] = []
        labels: list[int] = []
        for _, row in df.iterrows():
            t = row[tc]
            if not isinstance(t, str):
                t = str(t) if t is not None else ""
            y_raw = coerce_label_int(row[lc]) if lc in row.index else 0
            stem: str | None = None
            if stem_col and stem_col in row.index:
                sv = row[stem_col]
                if sv is not None and not (isinstance(sv, float) and np.isnan(sv)):
                    stem = str(sv).strip()
            yn = normalize_label_for_n_classes(int(y_raw), n_classes, source_stem=stem)
            if yn is None:
                continue
            texts.append(t)
            labels.append(int(yn))
        if texts:
            yield texts, torch.tensor(labels, dtype=torch.long)


def _normalize_llm_checkpoint_dir(cfg: dict, repo: Path) -> None:
    """Force a repo-local, writable HF cache dir for Tracks B/C (configs may ship stale absolute paths)."""
    block = cfg.get("LLMModule")
    if not isinstance(block, dict):
        return
    fallback = (repo / "checkpoints" / "deep_learning" / "llm").resolve()
    raw = block.get("checkpoint_dir")
    s = str(raw).strip() if raw is not None else "checkpoints/deep_learning/llm"
    # Broken / other-machine paths from checked-in configs
    if "Rohan_Ravula" in s or "capstone-2" in s.replace("\\", "/"):
        fallback.mkdir(parents=True, exist_ok=True)
        block["checkpoint_dir"] = str(fallback)
        return
    p = Path(s)
    if not p.is_absolute():
        block["checkpoint_dir"] = str((repo / p).resolve())
        return
    try:
        p.mkdir(parents=True, exist_ok=True)
        test = p / ".write_test"
        test.write_text("ok", encoding="utf-8")
        test.unlink(missing_ok=True)
        block["checkpoint_dir"] = str(p.resolve())
    except OSError:
        fallback.mkdir(parents=True, exist_ok=True)
        block["checkpoint_dir"] = str(fallback)


def _build_frozen_llm(
    repo: Path,
    encoder_cfg: Path,
    n_classes: int,
    encoder_ckpt: Path | None,
    device: torch.device,
) -> nn.Module:
    cfg = load_config(encoder_cfg)
    _normalize_llm_checkpoint_dir(cfg, repo)
    model, _ = build_model_from_config_dict(cfg, n_classes, "")
    if encoder_ckpt is not None and encoder_ckpt.is_file():
        model.load_state_dict(load_safetensors_state(encoder_ckpt, map_location="cpu"), strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)


def run_track_b(
    repo: Path,
    processed_pq: Path,
    out_dir: Path,
    n_classes: int,
    encoder_cfg: Path,
    encoder_ckpt: Path | None,
    batch_size: int,
    parquet_chunk: int,
    epochs: int,
    device: torch.device,
) -> None:
    from Code.models.deep_learning.llm.llm_models import LLMModule

    llm = _build_frozen_llm(repo, encoder_cfg, n_classes, encoder_ckpt, device)
    if not isinstance(llm, LLMModule):
        raise TypeError("Track B/C expect LLMModule (DistilBERT) for embedding extraction.")

    with torch.no_grad():
        z0 = llm.get_embeddings(["hello"], pooling_strategy="mean", layer_strategy="last")
    in_dim = int(z0.shape[-1])
    head = nn.Linear(in_dim, n_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    n_rows = _parquet_num_rows(processed_pq)
    steps_per_epoch = max(1, (n_rows + batch_size - 1) // batch_size)

    for ep in range(epochs):
        bar = tqdm(
            total=steps_per_epoch,
            desc=f"ml_track_B ep{ep+1}/{epochs}",
            unit="step",
            mininterval=0.5,
            file=sys.stderr,
        )
        try:
            for texts, y_cpu in _iter_text_label_batches(
                processed_pq, n_classes, parquet_chunk, None, None
            ):
                for start in range(0, len(texts), batch_size):
                    chunk = texts[start : start + batch_size]
                    y = y_cpu[start : start + batch_size].to(device)
                    with torch.no_grad():
                        z = llm.get_embeddings(chunk, pooling_strategy="mean", layer_strategy="last")
                    logits = head(z)
                    loss = crit(logits, y)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    bar.update(1)
                    bar.set_postfix(loss=float(loss.item()))
        finally:
            bar.close()

    out = out_dir / "trackB_DistilBERT_Linear.safetensors"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_safetensors(head.state_dict(), out)
    print("Saved", out)


def run_track_c(
    repo: Path,
    processed_pq: Path,
    out_dir: Path,
    n_classes: int,
    encoder_cfg: Path,
    encoder_ckpt: Path | None,
    batch_size: int,
    parquet_chunk: int,
    max_samples: int,
    device: torch.device,
) -> None:
    from Code.models.deep_learning.llm.llm_models import LLMModule
    from joblib import dump
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC

    llm = _build_frozen_llm(repo, encoder_cfg, n_classes, encoder_ckpt, device)
    if not isinstance(llm, LLMModule):
        raise TypeError("Track B/C expect LLMModule (DistilBERT) for embedding extraction.")

    xs: list[np.ndarray] = []
    ys: list[int] = []
    n = 0
    n_rows = _parquet_num_rows(processed_pq)
    cap = min(max_samples, n_rows)
    embed_steps = max(1, (cap + batch_size - 1) // batch_size)
    bar = tqdm(
        total=embed_steps,
        desc="ml_track_C embed",
        unit="batch",
        mininterval=0.5,
        file=sys.stderr,
    )
    try:
        for texts, y_cpu in _iter_text_label_batches(
            processed_pq, n_classes, parquet_chunk, None, None
        ):
            for start in range(0, len(texts), batch_size):
                if n >= max_samples:
                    break
                chunk = texts[start : start + batch_size]
                y = y_cpu[start : start + batch_size]
                take = min(len(chunk), max_samples - n)
                chunk = chunk[:take]
                y = y[:take]
                with torch.no_grad():
                    z = llm.get_embeddings(chunk, pooling_strategy="mean", layer_strategy="last")
                xs.append(z.detach().cpu().numpy().astype(np.float32))
                ys.extend(int(t) for t in y.tolist())
                n += take
                bar.update(1)
                bar.set_postfix(collected=n)
            if n >= max_samples:
                break
    finally:
        bar.close()
    if n == 0:
        raise RuntimeError("Track C: no labeled rows collected (check processed parquet and n_classes).")
    X = np.vstack(xs)
    y = np.array(ys, dtype=np.int64)
    out_dir.mkdir(parents=True, exist_ok=True)

    lr = LogisticRegression(max_iter=2000, solver="lbfgs", n_jobs=-1, multi_class="auto")
    lr.fit(X, y)
    dump(lr, out_dir / "proc_LogisticRegression_DistilBERT.joblib")

    if n_classes == 2:
        svc = LinearSVC(max_iter=3000, dual="auto")
    else:
        svc = LinearSVC(max_iter=3000, dual=False)
    svc.fit(X, y)
    dump(svc, out_dir / "proc_LinearSVC_DistilBERT.joblib")
    print("Saved", out_dir / "proc_LogisticRegression_DistilBERT.joblib")
    print("Saved", out_dir / "proc_LinearSVC_DistilBERT.joblib")


def main() -> int:
    ap = argparse.ArgumentParser(description="docs/ml Tracks B/C: frozen DistilBERT embeddings + meta learner.")
    ap.add_argument("--queue_config", type=Path, required=True, help="ml_queue/track_{b,c}_{2,3}_labels.json")
    ap.add_argument("--dataset_stem", type=str, required=True)
    ap.add_argument("--data_root", type=Path, default=None)
    ap.add_argument("--checkpoint_root", type=Path, default=None)
    ap.add_argument("--encoder_config", type=Path, default=None)
    ap.add_argument("--encoder_ckpt", type=Path, default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--parquet_chunk", type=int, default=2048)
    ap.add_argument("--epochs_b", type=int, default=2)
    ap.add_argument("--max_samples_c", type=int, default=400_000)
    args = ap.parse_args()

    track, n_labels = _load_queue_spec(args.queue_config.resolve())
    repo = _REPO
    data_root = (args.data_root or (repo / "data")).resolve()
    ckpt_root = (args.checkpoint_root or (repo / "checkpoints")).resolve()
    processed_pq = data_root / "processed" / f"{args.dataset_stem}.parquet"
    if not processed_pq.is_file():
        print(f"ERROR: missing processed parquet: {processed_pq}", file=sys.stderr)
        return 1

    enc_cfg = args.encoder_config
    if enc_cfg is None:
        enc_cfg = _default_encoder_config(repo, n_labels)
    if not enc_cfg.is_file():
        print(f"ERROR: encoder config not found: {enc_cfg}", file=sys.stderr)
        return 1

    enc_ckpt = args.encoder_ckpt
    if enc_ckpt is None:
        cand = _default_encoder_ckpt(ckpt_root, n_labels, args.dataset_stem)
        enc_ckpt = cand
    elif enc_ckpt is not None and not enc_ckpt.is_file():
        print(f"[ml_bc] encoder checkpoint not found (using pretrained backbone only): {enc_ckpt}", flush=True)
        enc_ckpt = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = _out_dir(ckpt_root, n_labels, args.dataset_stem)

    if track == "b":
        run_track_b(
            repo,
            processed_pq,
            out_dir,
            n_labels,
            enc_cfg.resolve(),
            enc_ckpt.resolve() if enc_ckpt is not None else None,
            int(args.batch_size),
            int(args.parquet_chunk),
            int(args.epochs_b),
            device,
        )
    else:
        run_track_c(
            repo,
            processed_pq,
            out_dir,
            n_labels,
            enc_cfg.resolve(),
            enc_ckpt.resolve() if enc_ckpt is not None else None,
            int(args.batch_size),
            int(args.parquet_chunk),
            int(args.max_samples_c),
            device,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

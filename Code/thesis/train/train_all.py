"""
Thesis training orchestrator.

- **Default (no args):** Prints usage. Train **one model at a time** with
  `Code/thesis/train/train_single.py` (see `scripts/README.md` → Core training).
- **Single run:** Pass `--config` and `--dataset_stem` to spawn one `train_single` process.
- **Full sweep:** Pass `--sweep-all` to run every thesis config × every dataset stem that has
  the required parquet (processed for transformers/HRM, transformed for others).

HRM / transformer MLM on the merged file: use `train_single.py --pretrain_text_source all_data_parquet`.

MoE and stacking are **not** handled here; use `train_moe.py` and `train_stack.py` per `docs/task.txt`.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]


def _build_single_cmd(
    py: str,
    single: Path,
    cfg: Path,
    stem: str,
    data_root: Path,
    checkpoint_root: Path | None,
    log_dir: Path | None,
    max_samples: int | None,
    epochs_pretrain: int,
    epochs_finetune: int,
    batch_size: int,
    phase: str,
) -> list[str]:
    cmd = [
        py,
        str(single),
        "--config",
        str(cfg),
        "--dataset_stem",
        stem,
        "--data_root",
        str(data_root),
        "--epochs_pretrain",
        str(epochs_pretrain),
        "--epochs_finetune",
        str(epochs_finetune),
        "--batch_size",
        str(batch_size),
        "--phase",
        phase,
    ]
    if checkpoint_root is not None:
        cmd += ["--checkpoint_root", str(checkpoint_root)]
    if log_dir is not None:
        cmd += ["--log_dir", str(log_dir)]
    if max_samples is not None:
        cmd += ["--max_samples", str(max_samples)]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Thesis training orchestrator (single run or explicit full sweep).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # One model / one dataset (preferred default workflow):
  python Code/thesis/train/train_all.py --config Code/thesis/config/transformers/2_labels/B3_E_DL1_DistilBERT.json \\
      --dataset_stem IMDB_Dataset --data_root data

  # Full matrix (all configs × all stems) — opt-in only:
  python Code/thesis/train/train_all.py --sweep-all --data_root data

  # Same as above but print commands only:
  python Code/thesis/train/train_all.py --sweep-all --dry_run

For dual GPU, use torch.distributed.run with train_single.py (see scripts/README.md → Dual GPU).
""",
    )
    ap.add_argument("--data_root", type=Path, default=None)
    ap.add_argument("--checkpoint_root", type=Path, default=None)
    ap.add_argument("--log_dir", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None, help="Single thesis JSON (requires --dataset_stem)")
    ap.add_argument("--dataset_stem", type=str, default=None, help="Parquet basename without .parquet")
    ap.add_argument("--config_glob", type=str, default="**/*.json")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--epochs_pretrain", type=int, default=1)
    ap.add_argument("--epochs_finetune", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--phase", default="all", choices=("pretrain", "finetune", "all"))
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument(
        "--sweep-all",
        action="store_true",
        help="Run every thesis config against every compatible dataset stem (expensive).",
    )
    args = ap.parse_args()

    py = sys.executable
    single = _REPO / "Code" / "thesis" / "train" / "train_single.py"
    data_root = args.data_root or (_REPO / "data")

    single_mode = args.config is not None or args.dataset_stem is not None
    if single_mode:
        if args.config is None or args.dataset_stem is None:
            ap.error("--config and --dataset_stem must be passed together for a single run.")
        cfg = args.config.resolve()
        if not cfg.is_file():
            ap.error(f"Config not found: {cfg}")
        cmd = _build_single_cmd(
            py,
            single,
            cfg,
            args.dataset_stem,
            data_root,
            args.checkpoint_root,
            args.log_dir,
            args.max_samples,
            args.epochs_pretrain,
            args.epochs_finetune,
            args.batch_size,
            args.phase,
        )
        print("RUN", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=False, cwd=str(_REPO))
        return

    if not args.sweep_all:
        ap.print_help()
        print(
            "\nNo action: specify --config and --dataset_stem for one training job, "
            "or --sweep-all to run all configs × datasets.\n"
            "See also: scripts/README.md (Core training) and Code/thesis/train/train_single.py",
            file=sys.stderr,
        )
        sys.exit(2)

    proc = data_root / "processed"
    trans = data_root / "transformed"

    stems = set()
    for folder in (proc, trans):
        if folder.is_dir():
            for p in folder.glob("*.parquet"):
                stems.add(p.stem)

    cfg_dir = _REPO / "Code" / "thesis" / "config"
    configs = sorted(cfg_dir.glob(args.config_glob))

    for cfg in configs:
        p = str(cfg).replace("\\", "/")
        if "/config/moe/" in p or p.endswith("example_experts.json"):
            continue
        text = "transformer" in str(cfg).lower() or "hrm" in str(cfg).lower()
        sub = proc if text else trans
        for stem in sorted(stems):
            if not (sub / f"{stem}.parquet").is_file():
                continue
            cmd = _build_single_cmd(
                py,
                single,
                cfg,
                stem,
                data_root,
                args.checkpoint_root,
                args.log_dir,
                args.max_samples,
                args.epochs_pretrain,
                args.epochs_finetune,
                args.batch_size,
                args.phase,
            )
            print("RUN", " ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=False, cwd=str(_REPO))


if __name__ == "__main__":
    main()

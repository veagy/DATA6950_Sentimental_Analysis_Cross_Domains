#!/usr/bin/env python3
"""
Generate matplotlib/seaborn charts from TEMP/output metrics, models, dataset EDA, and path inventory.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Writable config for CI / restricted home
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-charts")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def safe_filename(s: str, max_len: int = 100) -> str:
    s = re.sub(r"[^\w.\-]+", "_", str(s), flags=re.UNICODE)
    s = s.strip("_") or "unnamed"
    return s[:max_len]


def read_csv_optional(path: Path) -> Optional[pd.DataFrame]:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def model_family(model_id: str) -> str:
    mid = str(model_id)
    if mid.startswith("transformers_"):
        return "transformers"
    if mid.startswith("feature_encoder_"):
        return "feature_encoder"
    if mid.startswith("ml_"):
        return "ml"
    if mid.startswith("mlp_gelu_head_ddp_"):
        return "mlp_gelu_head_ddp"
    if mid.startswith("moe_"):
        return "moe"
    if "hrm" in mid.lower():
        return "hrm"
    return "other"


def normalize_model_key(model_id: str) -> str:
    """Key for matching 2-label vs 3-label rows."""
    mid = str(model_id)
    mid = re.sub(r"__(all_data|stem)_ckpt.*$", "", mid, flags=re.I)
    mid = re.sub(r"_2_labels_", "__L__", mid)
    mid = re.sub(r"_3_labels_", "__L__", mid)
    return mid


def parse_confusion_cell(val: Any) -> Optional[np.ndarray]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return None
        try:
            data = json.loads(val)
        except json.JSONDecodeError:
            return None
    else:
        data = val
    if not isinstance(data, list) or not data:
        return None
    try:
        return np.asarray(data, dtype=float)
    except Exception:
        return None


def _numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def plot_metrics_tables(
    repo: Path,
    charts: Path,
    dpi: int,
    ext: str,
    manifest: list[str],
) -> None:
    metrics_dir = repo / "output" / "metrics"
    for label_name, csv_name in (("2label", "2label_metrics_table.csv"), ("3label", "3label_metrics_table.csv")):
        path = metrics_dir / csv_name
        df = read_csv_optional(path)
        if df is None or df.empty:
            print(f"[warn] missing or empty {path}")
            continue

        for c in (
            "accuracy",
            "balanced_accuracy",
            "f1_macro",
            "f1_weighted",
            "matthews_corrcoef",
            "cohen_kappa",
            "roc_auc_ovr_macro",
        ):
            if c in df.columns:
                df[c] = _numeric_series(df[c])

        df = df.copy()
        df["_family"] = df["model_id"].astype(str).map(model_family)
        stem_col = "safe_stem" if "safe_stem" in df.columns else None
        stems = sorted(df[stem_col].dropna().unique()) if stem_col else ["all"]

        out_root = charts / "metrics" / label_name
        out_root.mkdir(parents=True, exist_ok=True)

        for stem in stems:
            sub = df[df[stem_col] == stem] if stem_col else df
            if sub.empty:
                continue
            ok = sub[sub["f1_macro"].notna() | sub["accuracy"].notna()].copy()
            if ok.empty:
                continue
            ok = ok.sort_values("f1_macro", ascending=False, na_position="last")

            # A1 ranking by F1
            fig, ax = plt.subplots(figsize=(12, max(4, 0.35 * len(ok))))
            y = ok["model_id"].astype(str)
            x = ok["f1_macro"].fillna(ok["accuracy"])
            fams = ok["_family"].unique().tolist()
            pal = dict(zip(fams, sns.color_palette("husl", max(len(fams), 1))))
            colors = [pal[f] for f in ok["_family"]]
            ax.barh(range(len(ok)), x, color=colors)
            ax.set_yticks(range(len(ok)))
            ax.set_yticklabels([t[:70] + "…" if len(t) > 70 else t for t in y], fontsize=7)
            ax.invert_yaxis()
            ax.set_xlabel("F1 macro (fallback: accuracy)")
            ax.set_title(f"Model ranking — {label_name} — {stem}")
            fig.tight_layout()
            fn = out_root / f"ranking_f1_{safe_filename(stem)}.{ext}"
            fig.savefig(fn, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            manifest.append(str(fn.relative_to(charts)))

            # A2 grouped macro vs weighted F1 (top 15)
            top = ok.head(15)
            if "f1_weighted" in top.columns and top["f1_weighted"].notna().any():
                fig, ax = plt.subplots(figsize=(10, 5))
                m = top.melt(
                    id_vars=["model_id"],
                    value_vars=["f1_macro", "f1_weighted"],
                    var_name="metric",
                    value_name="value",
                )
                m["model_short"] = m["model_id"].astype(str).str.slice(0, 40)
                sns.barplot(data=m, x="model_short", y="value", hue="metric", ax=ax)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
                ax.set_title(f"F1 macro vs weighted — {label_name} — {stem}")
                fig.tight_layout()
                fn2 = out_root / f"grouped_f1_macro_weighted_{safe_filename(stem)}.{ext}"
                fig.savefig(fn2, dpi=dpi, bbox_inches="tight")
                plt.close(fig)
                manifest.append(str(fn2.relative_to(charts)))

            # A3 heatmap models x scalars
            metric_cols = [
                c
                for c in (
                    "accuracy",
                    "balanced_accuracy",
                    "f1_macro",
                    "f1_weighted",
                    "matthews_corrcoef",
                    "cohen_kappa",
                    "hamming_loss",
                    "jaccard_macro",
                )
                if c in ok.columns
            ]
            heat = ok.set_index("model_id")[metric_cols].dropna(how="all")
            heat = heat.dropna(axis=0, how="all")
            if not heat.empty and len(heat) <= 40:
                fig, ax = plt.subplots(figsize=(10, max(4, 0.25 * len(heat))))
                sns.heatmap(heat, ax=ax, cmap="viridis", annot=False)
                ax.set_title(f"Metrics heatmap — {label_name} — {stem}")
                fig.tight_layout()
                fn3 = out_root / f"heatmap_metrics_{safe_filename(stem)}.{ext}"
                fig.savefig(fn3, dpi=dpi, bbox_inches="tight")
                plt.close(fig)
                manifest.append(str(fn3.relative_to(charts)))

            # A5 ROC AUC bar (numeric only)
            roc = ok[ok["roc_auc_ovr_macro"].notna()].copy()
            if not roc.empty:
                fig, ax = plt.subplots(figsize=(10, max(3, 0.3 * len(roc))))
                roc = roc.sort_values("roc_auc_ovr_macro", ascending=True)
                ax.barh(roc["model_id"].astype(str).str.slice(0, 60), roc["roc_auc_ovr_macro"])
                ax.set_xlabel("roc_auc_ovr_macro")
                ax.set_title(f"ROC AUC (macro OVR) — {label_name} — {stem}")
                fig.tight_layout()
                fn4 = out_root / f"roc_auc_{safe_filename(stem)}.{ext}"
                fig.savefig(fn4, dpi=dpi, bbox_inches="tight")
                plt.close(fig)
                manifest.append(str(fn4.relative_to(charts)))

        # A6 eval status
        has_err = "error" in df.columns
        if has_err:
            err_nonempty = df["error"].astype(str).str.strip().ne("") & df["error"].notna()
            counts = pd.Series(
                {"success": (~err_nonempty).sum(), "error": err_nonempty.sum()}
            )
            fig, ax = plt.subplots(figsize=(4, 4))
            counts.plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"])
            ax.set_title(f"Eval status — {label_name}")
            ax.set_ylabel("rows")
            fig.tight_layout()
            fn5 = out_root / f"eval_status.{ext}"
            fig.savefig(fn5, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            manifest.append(str(fn5.relative_to(charts)))

    # A7 scatter from summary.csv
    summary_path = metrics_dir / "summary.csv"
    sdf = read_csv_optional(summary_path)
    if sdf is not None and not sdf.empty:
        for c in ("accuracy", "f1_macro", "balanced_accuracy", "roc_auc_ovr_macro"):
            if c in sdf.columns:
                sdf[c] = _numeric_series(sdf[c])
        sub = sdf[sdf["accuracy"].notna() & sdf["f1_macro"].notna()]
        if not sub.empty:
            fig, ax = plt.subplots(figsize=(7, 6))
            hue = sdf["n_classes"] if "n_classes" in sdf.columns else None
            sns.scatterplot(
                data=sub,
                x="accuracy",
                y="f1_macro",
                hue=hue,
                ax=ax,
                alpha=0.8,
            )
            ax.set_title("Accuracy vs F1 macro (summary.csv)")
            fig.tight_layout()
            out_sc = charts / "metrics" / f"scatter_accuracy_f1_summary.{ext}"
            out_sc.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_sc, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            manifest.append(str(out_sc.relative_to(charts)))


def plot_confusion_jsons(
    repo: Path,
    charts: Path,
    dpi: int,
    ext: str,
    manifest: list[str],
) -> None:
    metrics_root = repo / "output" / "metrics"
    if not metrics_root.is_dir():
        return
    out_dir = charts / "metrics" / "confusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    for jpath in sorted(metrics_root.rglob("metrics.json")):
        try:
            data = json.loads(jpath.read_text(encoding="utf-8"))
        except Exception:
            continue
        m = data.get("metrics") or {}
        cm = m.get("confusion_matrix")
        if cm is None:
            continue
        arr = np.asarray(cm, dtype=float)
        if arr.size == 0:
            continue
        rel = jpath.relative_to(metrics_root)
        parts = rel.parts
        label_mode = parts[0] if parts else "unknown"
        stem = parts[1] if len(parts) > 1 else "unknown"
        model_slug = safe_filename("_".join(parts[2:-1]) if len(parts) > 2 else "model")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(arr, annot=True, fmt=".0f", cmap="Blues", ax=ax)
        ax.set_title(f"Confusion — {label_mode}/{stem}\n{data.get('model_id', '')[:50]}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fn = out_dir / f"{label_mode}_{safe_filename(stem)}_{model_slug}.{ext}"
        fig.tight_layout()
        fig.savefig(fn, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        manifest.append(str(fn.relative_to(charts)))


def plot_combined_label_modes(
    repo: Path,
    charts: Path,
    dpi: int,
    ext: str,
    manifest: list[str],
) -> None:
    p2 = repo / "output" / "metrics" / "2label_metrics_table.csv"
    p3 = repo / "output" / "metrics" / "3label_metrics_table.csv"
    d2 = read_csv_optional(p2)
    d3 = read_csv_optional(p3)
    if d2 is None or d3 is None or d2.empty or d3.empty:
        return
    if "f1_macro" in d2.columns:
        d2["f1_macro"] = _numeric_series(d2["f1_macro"])
    if "f1_macro" in d3.columns:
        d3["f1_macro"] = _numeric_series(d3["f1_macro"])
    d2 = d2[d2["f1_macro"].notna()].copy()
    d3 = d3[d3["f1_macro"].notna()].copy()
    d2["_key"] = d2["model_id"].astype(str).map(normalize_model_key)
    d3["_key"] = d3["model_id"].astype(str).map(normalize_model_key)
    stem_col = "safe_stem" if "safe_stem" in d2.columns else None
    if not stem_col:
        return
    out_dir = charts / "metrics" / "combined"
    out_dir.mkdir(parents=True, exist_ok=True)

    for stem in sorted(set(d2[stem_col]) & set(d3[stem_col])):
        a = d2[d2[stem_col] == stem][["_key", "model_id", "f1_macro"]].rename(
            columns={"f1_macro": "f1_2", "model_id": "id_2"}
        )
        b = d3[d3[stem_col] == stem][["_key", "model_id", "f1_macro"]].rename(
            columns={"f1_macro": "f1_3", "model_id": "id_3"}
        )
        merged = a.merge(b, on="_key", how="inner")
        if merged.empty:
            continue
        merged["_fmin"] = merged[["f1_2", "f1_3"]].min(axis=1)
        merged = merged.nlargest(min(20, len(merged)), "_fmin")
        merged = merged.drop(columns=["_fmin"])
        if merged.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(merged))))
        x = np.arange(len(merged))
        w = 0.35
        ax.barh(x - w / 2, merged["f1_2"], w, label="2-label")
        ax.barh(x + w / 2, merged["f1_3"], w, label="3-label")
        ax.set_yticks(x)
        labels = merged["_key"].astype(str).str.slice(0, 55)
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("F1 macro")
        ax.set_title(f"Matched 2-label vs 3-label F1 — {stem}")
        ax.legend()
        fig.tight_layout()
        fn = out_dir / f"paired_f1_{safe_filename(stem)}.{ext}"
        fig.savefig(fn, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        manifest.append(str(fn.relative_to(charts)))


def plot_dataset_analysis(
    repo: Path,
    charts: Path,
    dpi: int,
    ext: str,
    manifest: list[str],
) -> None:
    eda = repo / "output" / "dataset analysis"
    if not eda.is_dir():
        print(f"[warn] missing dataset analysis dir: {eda}")
        return
    out_root = charts / "dataset_analysis"
    out_root.mkdir(parents=True, exist_ok=True)

    for profile_path in sorted(eda.rglob("profile.json")):
        try:
            prof = json.loads(profile_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        slug = safe_filename(profile_path.parent.name)

        # C1 label distribution (3-class normalized preferred)
        counts = prof.get("label_distribution_normalized_3") or prof.get(
            "label_raw_value_counts_top20"
        )
        if isinstance(counts, dict) and counts:
            labels = [str(k) for k in counts.keys() if not str(k).startswith("_")]
            vals = [float(counts[k]) for k in labels]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(labels, vals, color=sns.color_palette("muted", len(labels)))
            ax.set_title(f"Label distribution — {slug}")
            ax.set_xlabel("label")
            ax.set_ylabel("count")
            fig.tight_layout()
            fn = out_root / f"labels_{slug}.{ext}"
            fig.savefig(fn, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            manifest.append(str(fn.relative_to(charts)))

        # C2 text length quantiles
        tls = prof.get("text_length_stats_sample")
        if isinstance(tls, dict):
            keys = [
                "char_len_p50",
                "char_len_p90",
                "char_len_p95",
                "char_len_p99",
                "char_len_mean",
            ]
            present = {k: tls.get(k) for k in keys if tls.get(k) is not None}
            if present:
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(list(present.keys()), list(present.values()), marker="o")
                ax.set_title(f"Text length stats — {slug}")
                ax.set_ylabel("characters")
                ax.tick_params(axis="x", rotation=25)
                fig.tight_layout()
                fn2 = out_root / f"text_length_{slug}.{ext}"
                fig.savefig(fn2, dpi=dpi, bbox_inches="tight")
                plt.close(fig)
                manifest.append(str(fn2.relative_to(charts)))

    # C3 per-stem label CSVs
    for csv_path in sorted(eda.rglob("label_distribution_by_source_stem.csv")):
        df = read_csv_optional(csv_path)
        if df is None or df.empty:
            continue
        if not {"source_stem", "count"}.issubset(df.columns):
            continue
        slug = safe_filename(csv_path.parent.name)
        fig, ax = plt.subplots(figsize=(8, 4))
        label_col = "raw_label_coerced" if "raw_label_coerced" in df.columns else None
        if label_col:
            pivot = df.pivot_table(
                index="source_stem",
                columns=label_col,
                values="count",
                aggfunc="sum",
            )
            pivot.plot(kind="bar", ax=ax, stacked=False)
        else:
            sns.barplot(data=df, x="source_stem", y="count", ax=ax)
        ax.set_title(f"Label counts by source_stem — {slug}")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fn = out_root / f"stem_labels_{slug}.{ext}"
        fig.savefig(fn, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        manifest.append(str(fn.relative_to(charts)))


def plot_models_charts(
    repo: Path,
    charts: Path,
    dpi: int,
    ext: str,
    manifest: list[str],
) -> None:
    models_dir = repo / "output" / "models"
    out_m = charts / "models"
    out_m.mkdir(parents=True, exist_ok=True)

    # D1 checkpoints top 25
    ckpt = read_csv_optional(models_dir / "checkpoints_inventory.csv")
    if ckpt is not None and not ckpt.empty and "size_bytes" in ckpt.columns:
        ckpt = ckpt.copy()
        ckpt["size_bytes"] = _numeric_series(ckpt["size_bytes"])
        top = ckpt.nlargest(25, "size_bytes")
        col = "weight_relative" if "weight_relative" in top.columns else top.columns[0]
        top["name"] = top[col].astype(str).str.split("/").str[-1]
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(data=top, y="name", x="size_bytes", ax=ax, color="steelblue")
        ax.set_title("Largest checkpoints (top 25)")
        ax.set_xlabel("size_bytes")
        fig.tight_layout()
        fn = out_m / f"checkpoints_top25.{ext}"
        fig.savefig(fn, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        manifest.append(str(fn.relative_to(charts)))

    # D2 configs by model_class
    cfg = read_csv_optional(models_dir / "configs_catalog.csv")
    if cfg is not None and "model_class" in cfg.columns:
        vc = cfg["model_class"].dropna().astype(str).value_counts().head(25)
        if not vc.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            vc.plot(kind="bar", ax=ax, color="steelblue")
            ax.set_title("Config count by model_class (top 25)")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            fig.tight_layout()
            fn2 = out_m / f"configs_by_model_class.{ext}"
            fig.savefig(fn2, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            manifest.append(str(fn2.relative_to(charts)))

    # E1 logs size distribution
    logs = read_csv_optional(models_dir / "logs_index.csv")
    if logs is not None and not logs.empty and "size_bytes" in logs.columns:
        logs = logs.copy()
        logs["size_bytes"] = _numeric_series(logs["size_bytes"])
        sub = logs[logs["size_bytes"].notna() & (logs["size_bytes"] > 0)]
        if not sub.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(np.log10(sub["size_bytes"] + 1), bins=20, color="coral", edgecolor="white")
            ax.set_xlabel("log10(size_bytes + 1)")
            ax.set_title("Log file size distribution")
            fig.tight_layout()
            fn3 = out_m / f"logs_size_hist.{ext}"
            fig.savefig(fn3, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            manifest.append(str(fn3.relative_to(charts)))

    # E2 MoE manifests
    moe_path = models_dir / "moe_manifests.json"
    if moe_path.is_file():
        try:
            manifests = json.loads(moe_path.read_text(encoding="utf-8"))
        except Exception:
            manifests = []
        if isinstance(manifests, list) and manifests:
            names = [Path(m.get("path", str(i))).name[:40] for i, m in enumerate(manifests)]
            nex = [int(m.get("n_experts", 0)) for m in manifests]
            fig, ax = plt.subplots(figsize=(10, max(3, 0.25 * len(names))))
            ax.barh(names, nex, color="teal")
            ax.set_xlabel("n_experts")
            ax.set_title("MoE manifests — expert count")
            fig.tight_layout()
            fn4 = out_m / f"moe_manifest_experts.{ext}"
            fig.savefig(fn4, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            manifest.append(str(fn4.relative_to(charts)))


def plot_path_charts(
    repo: Path,
    charts: Path,
    dpi: int,
    ext: str,
    manifest: list[str],
) -> None:
    inv_path = repo / "output" / "path" / "inventory_all.csv"
    inv = read_csv_optional(inv_path)
    out_p = charts / "path"
    out_p.mkdir(parents=True, exist_ok=True)
    if inv is not None and not inv.empty:
        files = inv[inv["kind"].astype(str) == "file"].copy()
        if not files.empty and "relative_path" in files.columns:
            files["top"] = (
                files["relative_path"]
                .astype(str)
                .str.split("/", n=1)
                .str[0]
                .replace(".", "_root")
            )
            files["size_bytes"] = _numeric_series(files.get("size_bytes", 0))
            g = files.groupby("top", as_index=False).agg(
                n_files=("relative_path", "count"),
                total_bytes=("size_bytes", "sum"),
            )
            g = g.sort_values("total_bytes", ascending=False).head(20)
            fig, ax = plt.subplots(figsize=(9, 5))
            sns.barplot(data=g, x="top", y="total_bytes", ax=ax, color="darkslateblue")
            ax.set_title("Total file bytes by top-level path segment (top 20)")
            plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
            fig.tight_layout()
            fn = out_p / f"topdir_bytes.{ext}"
            fig.savefig(fn, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            manifest.append(str(fn.relative_to(charts)))

            fig2, ax2 = plt.subplots(figsize=(9, 5))
            sns.barplot(data=g, x="top", y="n_files", ax=ax2, color="coral")
            ax2.set_title("File count by top-level segment (top 20)")
            plt.setp(ax2.get_xticklabels(), rotation=35, ha="right")
            fig2.tight_layout()
            fn2 = out_p / f"topdir_file_count.{ext}"
            fig2.savefig(fn2, dpi=dpi, bbox_inches="tight")
            plt.close(fig2)
            manifest.append(str(fn2.relative_to(charts)))


def write_index(
    charts: Path,
    manifest: list[str],
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Generated charts",
        "",
        f"- **Generated:** {datetime.now(timezone.utc).isoformat()}",
        f"- **Repo root:** `{args.repo_root}`",
        f"- **DPI:** {args.dpi}",
        f"- **Format:** {args.format}",
        f"- **Only:** {args.only}",
        "",
        "## Figures",
        "",
    ]
    for p in sorted(set(manifest)):
        lines.append(f"- `{p}`")
    lines.append("")
    (charts / "index.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate charts from TEMP/output artifacts.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="TEMP repo root (directory containing output/). Default: parent of output/scripts/..",
    )
    parser.add_argument(
        "--charts-dir",
        type=Path,
        default=None,
        help="Output charts directory (default: REPO/output/charts)",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--format", choices=("png", "pdf"), default="png")
    parser.add_argument(
        "--only",
        choices=("all", "metrics", "dataset", "models", "path"),
        default="all",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = args.repo_root or script_dir.parent.parent
    charts_dir = args.charts_dir or (repo_root / "output" / "charts")
    charts_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="notebook")
    manifest: list[str] = []
    ext = args.format

    if args.only in ("all", "metrics"):
        plot_metrics_tables(repo_root, charts_dir, args.dpi, ext, manifest)
        plot_confusion_jsons(repo_root, charts_dir, args.dpi, ext, manifest)
        plot_combined_label_modes(repo_root, charts_dir, args.dpi, ext, manifest)
    if args.only in ("all", "dataset"):
        plot_dataset_analysis(repo_root, charts_dir, args.dpi, ext, manifest)
    if args.only in ("all", "models"):
        plot_models_charts(repo_root, charts_dir, args.dpi, ext, manifest)
    if args.only in ("all", "path"):
        plot_path_charts(repo_root, charts_dir, args.dpi, ext, manifest)

    write_index(charts_dir, manifest, args)
    print(f"Wrote {len(manifest)} figures under {charts_dir}")


if __name__ == "__main__":
    main()

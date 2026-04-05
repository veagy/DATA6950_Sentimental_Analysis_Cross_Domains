#!/usr/bin/env python3
"""
Exploratory analysis for thesis Parquet datasets under data/processed and data/transformed.

Writes reports under output/dataset analysis/ (default). See index.md for methodology notes.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as e:
    raise SystemExit("pyarrow is required") from e

from Code.thesis.common.datasets import (
    _infer_features_column,
    _infer_label_column,
    _infer_text_column,
    coerce_label_int,
    normalize_label_for_n_classes,
)


def path_slug(rel: Path) -> str:
    return str(rel).replace("/", "__").replace("\\", "__")


def _to_scalar_cell(v: Any) -> Any:
    """Flatten single-element array/list cells (common in some Parquet exports)."""
    if isinstance(v, np.ndarray):
        if v.size == 0:
            return None
        if v.size == 1:
            return v.flat[0]
        return v.tolist()
    if isinstance(v, (list, tuple)):
        if len(v) == 1:
            return v[0]
        return v
    return v


def discover_parquets(
    data_root: Path,
    only_glob: Optional[str],
    *,
    include_by_source_stem: bool = True,
) -> list[Path]:
    files = sorted(data_root.rglob("*.parquet"))
    if not include_by_source_stem:
        files = [p for p in files if "by_source_stem" not in p.parts]
    if only_glob:
        import fnmatch

        pat = only_glob
        files = [p for p in files if fnmatch.fnmatch(str(p.relative_to(data_root)), pat)]
    return files


def read_parquet_row_limit(path: Path, max_rows: int, columns: Optional[list[str]] = None) -> pd.DataFrame:
    """Read up to max_rows using row groups (does not load entire file into memory at once)."""
    pf = pq.ParquetFile(path)
    chunks: list[pa.Table] = []
    n = 0
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=columns)
        chunks.append(t)
        n += t.num_rows
        if n >= max_rows:
            break
    if not chunks:
        return pd.DataFrame()
    tab = pa.concat_tables(chunks)
    if tab.num_rows > max_rows:
        tab = tab.slice(0, max_rows)
    return tab.to_pandas()


def source_stem_counts_full(path: Path) -> dict[str, int]:
    """Stream source_stem column for exact counts (single column)."""
    pf = pq.ParquetFile(path)
    if "source_stem" not in pf.schema_arrow.names:
        return {}
    c: Counter[str] = Counter()
    for batch in pf.iter_batches(batch_size=65536, columns=["source_stem"]):
        col = batch.column(0)
        for i in range(batch.num_rows):
            v = col[i].as_py()
            if v is None or (isinstance(v, float) and pd.isna(v)):
                key = "__null__"
            else:
                key = str(v).strip()
            c[key] += 1
    return dict(c)


def column_null_stats(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for col in df.columns:
        s = df[col]
        n = len(s)
        nnull = int(s.isna().sum())
        out[col] = {
            "null_count": nnull,
            "null_rate": float(nnull / n) if n else 0.0,
        }
        if pd.api.types.is_numeric_dtype(s) and n - nnull > 0:
            sn = s.dropna()
            out[col]["min"] = float(sn.min()) if len(sn) else None
            out[col]["max"] = float(sn.max()) if len(sn) else None
            out[col]["mean"] = float(sn.mean()) if len(sn) else None
        elif s.dtype == object or pd.api.types.is_string_dtype(s):
            nn = s.dropna()
            if len(nn) == 0:
                out[col]["n_distinct_sample"] = 0
            else:
                try:
                    out[col]["n_distinct_sample"] = int(nn.nunique())
                except TypeError:
                    out[col]["n_distinct_sample"] = None
                    out[col]["n_distinct_note"] = "skipped (unhashable / nested values)"
    return out


def text_length_stats(texts: pd.Series) -> dict[str, Any]:
    lens = texts.fillna("").astype(str).str.len()
    words = texts.fillna("").astype(str).str.split().str.len()
    empty = int((lens == 0).sum())
    return {
        "n_empty_text": empty,
        "char_len_min": int(lens.min()) if len(lens) else 0,
        "char_len_max": int(lens.max()) if len(lens) else 0,
        "char_len_mean": float(lens.mean()) if len(lens) else 0.0,
        "char_len_std": float(lens.std()) if len(lens) else 0.0,
        "char_len_p50": float(np.percentile(lens, 50)) if len(lens) else 0.0,
        "char_len_p90": float(np.percentile(lens, 90)) if len(lens) else 0.0,
        "char_len_p95": float(np.percentile(lens, 95)) if len(lens) else 0.0,
        "char_len_p99": float(np.percentile(lens, 99)) if len(lens) else 0.0,
        "word_count_mean": float(words.mean()) if len(words) else 0.0,
    }


def label_distribution_normalized(
    df: pd.DataFrame,
    label_col: str,
    stem_col: Optional[str],
    n_classes: int,
) -> dict[str, int]:
    raw = df[label_col].map(lambda x: coerce_label_int(_to_scalar_cell(x)))
    if stem_col and stem_col in df.columns:
        def _stem_cell(x: Any) -> Optional[str]:
            x = _to_scalar_cell(x)
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return None
            return str(x).strip()

        stems = df[stem_col].map(_stem_cell)
    else:
        stems = pd.Series([None] * len(df), index=df.index, dtype=object)
    counts: Counter[str] = Counter()
    skipped = 0
    for r, s in zip(raw.tolist(), stems.tolist()):
        yn = normalize_label_for_n_classes(int(r), n_classes, source_stem=s)
        if yn is None:
            skipped += 1
            continue
        counts[str(yn)] += 1
    out = {str(k): int(v) for k, v in sorted(counts.items(), key=lambda x: int(x[0]))}
    out["_skipped_normalization"] = skipped
    return out


def dedupe_rate_sample(
    df: pd.DataFrame,
    text_col: Optional[str],
    label_col: Optional[str],
    dedupe_key_cols: Optional[list[str]] = None,
) -> Optional[float]:
    cols = dedupe_key_cols
    if cols:
        if not all(c in df.columns for c in cols):
            return None
        keys = df[cols[0]].astype(str)
        for c in cols[1:]:
            keys = keys + "\x00" + df[c].astype(str)
    elif text_col and label_col and text_col in df.columns and label_col in df.columns:
        keys = df[text_col].fillna("").astype(str) + "\x00" + df[label_col].astype(str)
    else:
        return None
    n = len(keys)
    if n == 0:
        return None
    return float(1.0 - keys.nunique() / n)


def _first_column_compression(meta: Any) -> Optional[str]:
    if meta is None or meta.num_row_groups == 0 or meta.row_group(0).num_columns == 0:
        return None
    try:
        return str(meta.row_group(0).column(0).compression)
    except Exception:
        return None


def analyze_one_file(
    path: Path,
    data_root: Path,
    max_rows: int,
    dedupe_key_cols: Optional[list[str]] = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rel = path.relative_to(data_root)
    pf = pq.ParquetFile(path)
    meta = pf.metadata
    names = list(pf.schema_arrow.names)

    profile: dict[str, Any] = {
        "path": str(path.resolve()),
        "relative_to_data_root": str(rel.as_posix()),
        "file_size_bytes": path.stat().st_size,
        "num_rows_metadata": int(meta.num_rows),
        "num_row_groups": int(meta.num_row_groups),
        "compression": _first_column_compression(meta),
        "column_names": names,
        "sampled_for_deep_stats": False,
        "sample_max_rows_cap": max_rows,
        "dedupe_key_columns": dedupe_key_cols,
    }

    n_meta = int(meta.num_rows)
    sample_n = min(max_rows, n_meta) if max_rows > 0 else n_meta
    profile["deep_stats_row_count"] = sample_n
    profile["sampled_for_deep_stats"] = sample_n < n_meta
    profile["deep_stats_mode"] = "sampled" if sample_n < n_meta else "exact_window"

    df = read_parquet_row_limit(path, sample_n, columns=None)
    profile["columns_profile"] = column_null_stats(df)

    text_c = _infer_text_column(list(df.columns))
    label_c = _infer_label_column(list(df.columns))
    feat_c = _infer_features_column(list(df.columns))
    profile["inferred_text_column"] = text_c
    profile["inferred_label_column"] = label_c
    profile["inferred_features_column"] = feat_c

    stem_col = "source_stem" if "source_stem" in df.columns else None
    profile["source_stem_column"] = stem_col

    if label_c and label_c in df.columns:
        raw_vals = df[label_c].map(lambda x: coerce_label_int(_to_scalar_cell(x)))
        try:
            profile["label_raw_value_counts_top20"] = dict(
                raw_vals.value_counts().head(20).astype(int)
            )
        except TypeError:
            profile["label_raw_value_counts_top20"] = {}
            profile["label_raw_value_counts_note"] = "skipped (unhashable label cells)"
        profile["label_distribution_normalized_3"] = label_distribution_normalized(
            df, label_c, stem_col, 3
        )
        profile["label_distribution_normalized_2"] = label_distribution_normalized(
            df, label_c, stem_col, 2
        )

    if stem_col:
        profile["source_stem_value_counts_sample"] = (
            df[stem_col].astype(str).value_counts().head(50).to_dict()
        )

    if text_c and text_c in df.columns:
        profile["text_length_stats_sample"] = text_length_stats(df[text_c])

    profile["dedupe_rate_key_sample"] = dedupe_rate_sample(
        df, text_c, label_c, dedupe_key_cols=dedupe_key_cols
    )

    return profile, df


def write_profile_json(out_dir: Path, profile: dict[str, Any]) -> None:
    def ser(o: Any) -> Any:
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, dict):
            return {str(k): ser(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [ser(x) for x in o]
        return o

    with open(out_dir / "profile.json", "w", encoding="utf-8") as f:
        json.dump(ser(profile), f, indent=2)


def write_report_md(out_dir: Path, profile: dict[str, Any]) -> None:
    lines = [
        f"# Dataset profile: `{profile.get('relative_to_data_root', '')}`",
        "",
        f"- **Rows (metadata):** {profile.get('num_rows_metadata')}",
        f"- **Deep stats rows:** {profile.get('deep_stats_row_count')} "
        f"({'sampled' if profile.get('sampled_for_deep_stats') else 'full sample window'})",
        f"- **File size (bytes):** {profile.get('file_size_bytes')}",
        "",
        "## Inferred columns",
        "",
        f"- Text: `{profile.get('inferred_text_column')}`",
        f"- Label: `{profile.get('inferred_label_column')}`",
        f"- Features: `{profile.get('inferred_features_column')}`",
        f"- source_stem: `{profile.get('source_stem_column')}`",
        "",
    ]
    tls = profile.get("text_length_stats_sample")
    if tls:
        lines += ["## Text length (sample)", "", "```json", json.dumps(tls, indent=2), "```", ""]
    l3 = profile.get("label_distribution_normalized_3")
    if l3:
        lines += ["## Normalized labels (3-class, sample)", "", "```json", json.dumps(l3, indent=2), "```", ""]
    l2 = profile.get("label_distribution_normalized_2")
    if l2:
        lines += ["## Normalized labels (2-class, sample)", "", "```json", json.dumps(l2, indent=2), "```", ""]
    stem_csv = out_dir / "label_distribution_by_source_stem.csv"
    if profile.get("source_stem_column") and profile.get("inferred_label_column"):
        lines += [f"- Per-stem raw label counts (sample): see `{stem_csv.name}`", ""]
    with open(out_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_label_by_stem_csv(
    df: pd.DataFrame,
    stem_col: str,
    label_col: str,
    out_csv: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for stem, g in df.groupby(stem_col, dropna=False):
        st = str(stem) if stem is not None and not (isinstance(stem, float) and pd.isna(stem)) else "__null__"
        vc = g[label_col].map(lambda x: coerce_label_int(_to_scalar_cell(x))).value_counts()
        for lab, cnt in vc.items():
            rows.append({"source_stem": st, "raw_label_coerced": int(lab), "count": int(cnt)})
    if not rows:
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source_stem", "raw_label_coerced", "count"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["source_stem"], r["raw_label_coerced"])))


def run_alignment(
    data_root: Path,
    out_analysis: Path,
    pairs: list[tuple[Path, Path]],
) -> None:
    lines = ["# Processed vs transformed alignment", ""]
    for proc, trans in pairs:
        if not proc.is_file() or not trans.is_file():
            continue
        rp = proc.relative_to(data_root)
        rt = trans.relative_to(data_root)
        np_ = pq.ParquetFile(proc).metadata.num_rows
        nt = pq.ParquetFile(trans).metadata.num_rows
        lines.append(f"## `{rp}` vs `{rt}`")
        lines.append(f"- Row count processed: **{np_}**")
        lines.append(f"- Row count transformed: **{nt}**")
        if np_ != nt:
            lines.append("- **Mismatch** in row counts.")
        if np_ == nt and np_ > 0:
            if "source_stem" in pq.ParquetFile(proc).schema_arrow.names and "source_stem" in pq.ParquetFile(
                trans
            ).schema_arrow.names:
                cp = source_stem_counts_full(proc)
                ct = source_stem_counts_full(trans)
                if cp != ct:
                    lines.append("- **Mismatch** in source_stem distribution (full column scan).")
                    lines.append("```")
                    lines.append(f"processed keys: {len(cp)}, transformed keys: {len(ct)}")
                    lines.append("```")
                else:
                    lines.append("- source_stem counts match (full column scan).")
        lines.append("")
    p = out_analysis / "alignment_report.md"
    p.write_text("\n".join(lines), encoding="utf-8")


def find_alignment_pairs(data_root: Path) -> list[tuple[Path, Path]]:
    proc_root = data_root / "processed"
    trans_root = data_root / "transformed"
    pairs: list[tuple[Path, Path]] = []
    if not proc_root.is_dir() or not trans_root.is_dir():
        return pairs
    for p in proc_root.rglob("*.parquet"):
        rel = p.relative_to(proc_root)
        t = trans_root / rel
        if t.is_file():
            pairs.append((p, t))
    return sorted(pairs, key=lambda x: str(x[0]))


def main() -> None:
    ap = argparse.ArgumentParser(description="EDA for thesis Parquet datasets.")
    ap.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Default: <repo>/data",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help='Default: <repo>/output/dataset analysis',
    )
    ap.add_argument(
        "--max-rows-per-file",
        type=int,
        default=500_000,
        help="Cap rows for deep stats (nulls, labels sample, text lengths). 0 = all rows (may OOM).",
    )
    ap.add_argument("--plots", action="store_true", help="Write PNG histograms if matplotlib is installed")
    ap.add_argument("--only-glob", type=str, default=None, help="fnmatch on path relative to data root")
    ap.add_argument(
        "--no-include-by-source-stem",
        dest="include_by_source_stem",
        action="store_false",
        default=True,
        help="Exclude Parquet files under any path segment named by_source_stem",
    )
    ap.add_argument(
        "--dedupe-key",
        type=str,
        default=None,
        help="Comma-separated column names for duplicate-rate estimate (default: inferred text + label)",
    )
    ap.add_argument("--no-alignment", action="store_true", help="Skip processed/transformed alignment report")
    args = ap.parse_args()

    data_root = (args.data_root or (_REPO / "data")).resolve()
    out_dir = (
        args.output_dir
        if args.output_dir is not None
        else (_REPO / "output" / "dataset analysis")
    )
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    per_root = out_dir / "per_file"
    per_root.mkdir(parents=True, exist_ok=True)
    fig_root = out_dir / "figures"
    if args.plots:
        fig_root.mkdir(parents=True, exist_ok=True)

    max_rows = args.max_rows_per_file
    if max_rows == 0:
        max_rows = 10**15

    dedupe_cols = (
        [c.strip() for c in args.dedupe_key.split(",") if c.strip()] if args.dedupe_key else None
    )

    files = discover_parquets(
        data_root,
        args.only_glob,
        include_by_source_stem=args.include_by_source_stem,
    )
    summary_rows: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()

    for path in files:
        rel = path.relative_to(data_root)
        slug = path_slug(rel)
        fdir = per_root / slug
        fdir.mkdir(parents=True, exist_ok=True)
        df: Optional[pd.DataFrame] = None
        try:
            profile, df = analyze_one_file(path, data_root, max_rows, dedupe_key_cols=dedupe_cols)
            profile["analysis_error"] = None
        except Exception as e:
            profile = {
                "path": str(path.resolve()),
                "relative_to_data_root": str(rel.as_posix()),
                "analysis_error": str(e),
            }
        write_profile_json(fdir, profile)
        if not profile.get("analysis_error") and df is not None:
            write_report_md(fdir, profile)
            sc = profile.get("source_stem_column")
            lc = profile.get("inferred_label_column")
            if sc and lc and sc in df.columns and lc in df.columns:
                write_label_by_stem_csv(df, sc, lc, fdir / "label_distribution_by_source_stem.csv")

            if args.plots:
                try:
                    import matplotlib

                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    tc = profile.get("inferred_text_column")
                    if tc and tc in df.columns:
                        lens = df[tc].fillna("").astype(str).str.len()
                        plt.figure(figsize=(8, 4))
                        plt.hist(lens.clip(upper=lens.quantile(0.99)), bins=50, color="steelblue", edgecolor="white")
                        plt.title(f"Text char length (<= p99), {rel}")
                        plt.xlabel("Characters")
                        plt.tight_layout()
                        plt.savefig(fig_root / f"{slug}__text_len.png", dpi=120)
                        plt.close()
                except Exception:
                    pass

        summary_rows.append(
            {
                "relative_path": str(rel.as_posix()),
                "num_rows_metadata": profile.get("num_rows_metadata", ""),
                "file_size_bytes": profile.get("file_size_bytes", path.stat().st_size if path.is_file() else ""),
                "n_columns": len(profile.get("column_names", [])),
                "inferred_text": profile.get("inferred_text_column", ""),
                "inferred_label": profile.get("inferred_label_column", ""),
                "sampled": profile.get("sampled_for_deep_stats", ""),
                "deep_stats_rows": profile.get("deep_stats_row_count", ""),
                "error": profile.get("analysis_error", ""),
                "profile_json": str((fdir / "profile.json").resolve()),
            }
        )

    sum_path = out_dir / "summary_all_files.csv"
    with open(sum_path, "w", newline="", encoding="utf-8") as f:
        fn = [
            "relative_path",
            "num_rows_metadata",
            "file_size_bytes",
            "n_columns",
            "inferred_text",
            "inferred_label",
            "sampled",
            "deep_stats_rows",
            "error",
            "profile_json",
        ]
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        w.writerows(summary_rows)

    if not args.no_alignment:
        pairs = find_alignment_pairs(data_root)
        if pairs:
            run_alignment(data_root, out_dir, pairs)

    index_lines = [
        "# Dataset analysis run",
        "",
        f"- **UTC time:** {now}",
        f"- **Data root:** `{data_root}`",
        f"- **Output:** `{out_dir}`",
        f"- **Max rows per file (deep stats):** {args.max_rows_per_file}",
        f"- **Include by_source_stem paths:** {args.include_by_source_stem}",
        f"- **Plots:** {args.plots}",
        f"- **Alignment report:** {not args.no_alignment}",
        f"- **Dedupe key columns:** {args.dedupe_key or '(inferred text+label)'}",
        "",
        "## References (methodology)",
        "",
        "- [Google ML: Explore your data (text classification)](https://developers.google.com/machine-learning/guides/text-classification/step-2)",
        "- [Data checklists / dataset quality (arXiv)](https://arxiv.org/html/2408.02919v1)",
        "",
        "Optional: install `ydata-profiling` for interactive HTML profiles (not run by this script).",
        "",
        "## Artifacts",
        "",
        f"- [`summary_all_files.csv`](summary_all_files.csv)",
        "- `per_file/<path_slug>/profile.json`, `report.md`, optional `label_distribution_by_source_stem.csv`",
        "- `alignment_report.md` (if processed/transformed pairs exist)",
        "- `figures/` (if `--plots`)",
        "",
        "## Files analyzed",
        "",
        *[f"- `{r['relative_path']}`" for r in summary_rows],
        "",
    ]
    (out_dir / "index.md").write_text("\n".join(index_lines), encoding="utf-8")
    print(f"[analyze] wrote {len(files)} profiles under {out_dir}", flush=True)


if __name__ == "__main__":
    main()

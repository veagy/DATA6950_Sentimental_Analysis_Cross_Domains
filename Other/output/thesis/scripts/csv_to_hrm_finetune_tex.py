"""Emit tab_hrm_finetune_{2,3}label.tex from output/metrics/hrm_finetune_*.csv."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _esc(s: str) -> str:
    return (
        str(s)
        .replace("\\", "/")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
    )


def _fmt_num(v) -> str:
    if pd.isna(v) or v == "":
        return "---"
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return "---"


def _emit_one(df: pd.DataFrame, caption: str, label: str, out_path: Path) -> None:
    cols = ["n_samples", "accuracy", "balanced_accuracy", "f1_macro", "roc_auc_ovr_macro"]
    have = [c for c in cols if c in df.columns]
    header = "Dataset & " + " & ".join(c.replace("_", r"\_") for c in have) + r" & Error / note \\"
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"{\footnotesize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}l " + "c" * len(have) + " p{3.2cm}@{}}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for _, r in df.iterrows():
        ds = _esc(r.get("dataset_stem", ""))
        cells = [_fmt_num(r.get(c)) for c in have]
        err = ""
        if pd.notna(r.get("error")) and str(r["error"]).strip():
            raw_e = str(r["error"]).strip()
            if raw_e.startswith("missing_checkpoint:"):
                err = r"missing\_checkpoint"
            elif raw_e.startswith("missing_config:"):
                err = r"missing\_config"
            elif raw_e.startswith("empty_dataset"):
                err = _esc(raw_e.split(":")[0])
            else:
                err = _esc(raw_e)
                err = " ".join(err.split())
                if len(err) > 72:
                    err = err[:69] + r"\ldots"
        row = f"{ds} & " + " & ".join(cells) + f" & {err} \\\\"
        lines.append(row)
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"}",
        r"\end{table*}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote", out_path)


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    mdir = root / "output" / "metrics"
    out_dir = root / "output" / "thesis" / "latex" / "tables and visualizations" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    p2 = mdir / "hrm_finetune_2label.csv"
    p3 = mdir / "hrm_finetune_3label.csv"
    if p2.is_file():
        _emit_one(
            pd.read_csv(p2),
            r"HRM fine-tuned classifier (2-label head): metrics per \texttt{data/processed} dataset (excluding \texttt{all-data}). Source: \texttt{output/metrics/hrm\_finetune\_2label.csv}.",
            "tab:hrm_finetune_2label",
            out_dir / "tab_hrm_finetune_2label.tex",
        )
    if p3.is_file():
        _emit_one(
            pd.read_csv(p3),
            r"HRM fine-tuned classifier (3-label head): metrics per \texttt{data/processed} dataset (excluding \texttt{all-data}). Source: \texttt{output/metrics/hrm\_finetune\_3label.csv}.",
            "tab:hrm_finetune_3label",
            out_dir / "tab_hrm_finetune_3label.tex",
        )


if __name__ == "__main__":
    main()

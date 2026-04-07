"""Emit tab_hrast_summary.tex from output/metrics/summary.csv.

Only rows with computable accuracy and macro-F1 are included (failed loads are
omitted from the paper table but remain in summary.csv for engineering audit).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    df = pd.read_csv(root / "output" / "metrics" / "summary.csv")
    # Keep rows where both primary metrics were computed
    acc_ok = df["accuracy"].notna()
    f1_ok = df["f1_macro"].notna()
    df = df.loc[acc_ok & f1_ok].copy()
    out_dir = root / "output" / "thesis" / "latex" / "tables and visualizations" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{HRAST stem: models with successfully computed accuracy and macro-F1 (from \texttt{output/metrics/summary.csv}). Rows with load or state errors are omitted here but retained in the CSV for debugging.}",
        r"\label{tab:hrast_summary}",
        r"{\footnotesize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}c l c c c@{}}",
        r"\toprule",
        r"$K$ & Model ID & Acc. & F1\textsubscript{macro} & ROC-AUC OVR \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        k = int(r["n_classes"]) if pd.notna(r["n_classes"]) else 0
        mid = str(r["model_id"]).replace("_", r"\_")
        acc = f"{float(r['accuracy']):.4f}"
        f1 = f"{float(r['f1_macro']):.4f}"
        roc = f"{float(r['roc_auc_ovr_macro']):.4f}" if pd.notna(r["roc_auc_ovr_macro"]) else "---"
        lines.append(f"{k} & {mid} & {acc} & {f1} & {roc} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"}",
        r"\end{table*}",
    ]
    (out_dir / "tab_hrast_summary.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote", out_dir / "tab_hrast_summary.tex", f"({len(df)} rows)")


if __name__ == "__main__":
    main()

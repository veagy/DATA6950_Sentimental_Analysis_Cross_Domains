# IEEE LaTeX report — build guide

This guide explains how the capstone **IEEEtran conference** paper under `output/thesis/latex/` is produced, which inputs under `output/` it uses, and how to refresh figures and tables after metrics change.

## 1. Document class and packages

- **Class:** `IEEEtran` with the `conference` option in **two-column** mode (default; no `onecolumn` in `main.tex`). Wide thesis and HRAST tables use `table*` / `figure*` so key assets span the text width; the PDF is tuned to stay near **10–15 pages** via float placement and body depth, not via `onecolumn`.
- **Column balance:** `\usepackage{balance}` and `\balance` **after** `\bibliography{references}` (end of `main.tex`) even out the references page; placing it before the bibliography can trigger a “called in second column” warning when the appendix ends mid-page.
- **Abstract discipline:** `sections/abstract.tex` stays qualitative—no numeric scores, no `\ref{...}` / appendix pointers (IEEE-style abstracts avoid cross-references; quantitative detail lives in Results, HRAST, and Appendix).
- **Bibliography:** BibTeX with `IEEEtran.bst` and `references.bib` (keys `ref01`–`ref67`, matching `\cite{ref01}` … in the narrative).
- **Packages in `main.tex`:** `graphicx`, `amsmath`, `booktabs`, `xurl` (path/URL line breaks), `microtype`, `cite`, `import`, `balance`.

Install a full TeX distribution (TeX Live or MiKTeX) that includes **IEEEtran** (`IEEEtran.cls` and `IEEEtran.bst`). If compilation fails with “File `IEEEtran.cls` not found”, install the `ieeetran` / `ieee` package for your distribution.

## 2. Repository layout

```text
output/thesis/docs/                    ← this documentation (do not scatter copies elsewhere)
  PAPER_SOURCE_INDEX.md               ← which repo paths feed each LaTeX section
output/thesis/markdown/
  THESIS_IMPLEMENTATION.md             ← primary narrative for LaTeX body
  THESIS_REPORT_NEW.md               ← PDF-aligned verbatim extract; secondary prose only (clean line breaks, map [n]→`\cite{refNN}` if key exists)
  THESIS_REPORT.md                   ← same role as NEW; do not paste wholesale into LaTeX
output/thesis/latex/
  main.tex
  sections/*.tex
  references.bib
  tables and visualizations/
    figures/                           ← copied PNGs (curated list: FIGURE_SELECTION.md)
    tables/                            ← \input fragments (e.g. HRAST metrics)
```

**Spaces in `tables and visualizations`:** `main.tex` sets:

```latex
\graphicspath{{"tables and visualizations/figures/"}}
```

Use only filenames inside `\includegraphics{...}` (no unquoted spaces in the argument).

## 3. Prerequisite pipeline (figures)

1. Ensure `output/metrics/` and `output/dataset analysis/` exist (from your eval / analysis pipeline).
2. Install chart dependencies: `pip install -r output/scripts/requirements-charts.txt`
3. **Windows:** set a writable Matplotlib config dir, e.g.  
   `$env:MPLCONFIGDIR = "$env:TEMP\mplconfig-charts"`
4. From **repository root:**  
   `python output/scripts/generate_all_charts.py --repo-root .`
5. Confirm PNGs appear under `output/charts/` and that `output/charts/index.md` lists them.

Copy the curated set into `latex/tables and visualizations/figures/` (see [FIGURE_SELECTION.md](FIGURE_SELECTION.md)). A one-shot PowerShell example:

```powershell
$dst = "output/thesis/latex/tables and visualizations/figures"
New-Item -ItemType Directory -Force -Path $dst | Out-Null
Copy-Item "output/charts/metrics/2label/ranking_f1_HRAST.png" "$dst/fig_ranking_2label.png"
Copy-Item "output/charts/metrics/3label/ranking_f1_HRAST.png" "$dst/fig_ranking_3label.png"
Copy-Item "output/charts/metrics/combined/paired_f1_HRAST.png" "$dst/fig_paired_f1.png"
Copy-Item "output/charts/dataset_analysis/labels_processed__by_source_stem__HRAST.parquet.png" "$dst/fig_labels_hrast.png"
Copy-Item "output/charts/metrics/confusion/2label_HRAST_transformers_2_labels_B5_E_DL2_RoBERTa_mlp768_1024__all_data_ckpt.png" "$dst/fig_confusion_roberta_2label.png"
Copy-Item "output/charts/metrics/2label/heatmap_metrics_HRAST.png" "$dst/fig_heatmap_2label.png"
Copy-Item "output/charts/metrics/3label/heatmap_metrics_HRAST.png" "$dst/fig_heatmap_3label.png"
Copy-Item "output/charts/metrics/2label/eval_status.png" "$dst/fig_eval_status_2label.png"
Copy-Item "output/charts/metrics/3label/eval_status.png" "$dst/fig_eval_status_3label.png"
Copy-Item "output/charts/dataset_analysis/labels_processed__IMDB_Dataset.parquet.png" "$dst/fig_labels_imdb.png"
Copy-Item "output/charts/dataset_analysis/labels_processed__all-data.parquet.png" "$dst/fig_labels_alldata.png"
```

## 4. Tables from CSV

- Rollup: `output/metrics/summary.csv` — one row per model × label mode for HRAST stem eval.
- Detail: `output/metrics/2label_metrics_table.csv`, `3label_metrics_table.csv`.

LaTeX fragments live in `latex/tables and visualizations/tables/`. The **full** HRAST rollup is included from **`sections/appendix.tex`** (not the main results body). After large CSV changes, regenerate it from `summary.csv`:

```powershell
python output/thesis/scripts/csv_to_hrast_summary_tex.py
```

Edit `tab_hrast_top2.tex` and `tab_hrast_top3.tex` manually if you need different top-$k$ rows; spot-check Section IX in `THESIS_IMPLEMENTATION.md`.

## 5. Compiling the PDF

From `output/thesis/latex/`:

```powershell
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or: `latexmk -pdf main.tex` (if `latexmk` is configured to run BibTeX).

Expected output: `main.pdf`.

## 6. Citations

- In-text: `\cite{ref01}` … `\cite{ref67}` aligned with `references.bib`.
- The Markdown implementation summary uses bracket numbers [1]–[67]; keep keys consistent with the `.bib` file shipped in `latex/references.bib`.

## 7. Maintenance checklist

- [ ] Regenerate metrics → refresh `summary.csv` / metric tables.
- [ ] Re-run `generate_all_charts.py` → re-copy curated figures.
- [ ] Update `sections/results.tex`, `sections/thesis_results.tex`, or table fragments if narrative numbers change.
- [ ] Recompile LaTeX; resolve undefined references.
- [ ] Record any thesis-vs-checkout mismatches in this doc or in `THESIS_IMPLEMENTATION.md` Section IX.

## 8. Related indexes

- [SOURCE_MAP.md](SOURCE_MAP.md) — section-to-folder map.
- [FIGURE_SELECTION.md](FIGURE_SELECTION.md) — figure copy list.
- `output/charts/index.md` — full chart manifest.

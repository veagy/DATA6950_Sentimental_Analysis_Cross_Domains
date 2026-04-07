# Source map: outputs → IEEE report

| Report section | Primary text | Numbers / tables | Figures |
|----------------|--------------|------------------|---------|
| Abstract, I–VIII | `output/thesis/markdown/THESIS_IMPLEMENTATION.md` | Cross-check vs `output/metrics/*.csv` | Optional `output/thesis/imgs/` thesis page PNGs |
| Dataset / EDA | Section IX + `output/dataset analysis/` | `summary_all_files.csv`, `per_file/*/report.md` | `output/charts/dataset_analysis/` |
| Models / pipeline | `output/models/*.md`, `output/path/*.md` | `output/models/*.csv`, `output/path/inventory_all.csv` | As needed from `output/charts/` |
| Results (HRAST stem) | Section IX | `summary.csv`, `2label_metrics_table.csv`, `3label_metrics_table.csv` | `output/charts/metrics/`, `confusion/` |
| Reproducibility | `output/scripts/` | — | Regenerate via `generate_all_charts.py` |

Paths are relative to the repository root unless noted.

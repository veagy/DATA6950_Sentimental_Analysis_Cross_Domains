# Curated figures for the LaTeX report

These PNGs are copied from `output/charts/` into `output/thesis/latex/tables and visualizations/figures/` (flattened filenames) for inclusion in `main.tex`.

| LaTeX filename | Source (under `output/charts/`) |
|----------------|----------------------------------|
| `fig_ranking_2label.png` | `metrics/2label/ranking_f1_HRAST.png` |
| `fig_ranking_3label.png` | `metrics/3label/ranking_f1_HRAST.png` |
| `fig_paired_f1.png` | `metrics/combined/paired_f1_HRAST.png` |
| `fig_labels_hrast.png` | `dataset_analysis/labels_processed__by_source_stem__HRAST.parquet.png` |
| `fig_confusion_roberta_2label.png` | `metrics/confusion/2label_HRAST_transformers_2_labels_B5_E_DL2_RoBERTa_mlp768_1024__all_data_ckpt.png` |
| `fig_heatmap_2label.png` | `metrics/2label/heatmap_metrics_HRAST.png` |
| `fig_heatmap_3label.png` | `metrics/3label/heatmap_metrics_HRAST.png` |
| `fig_eval_status_2label.png` | `metrics/2label/eval_status.png` |
| `fig_eval_status_3label.png` | `metrics/3label/eval_status.png` |
| `fig_labels_imdb.png` | `dataset_analysis/labels_processed__IMDB_Dataset.parquet.png` |
| `fig_labels_alldata.png` | `dataset_analysis/labels_processed__all-data.parquet.png` |

## Capstone midterm PNGs (`output/thesis/imgs/new`)

Copied into the same LaTeX `figures/` folder for `sections/capstone_visuals.tex` (documentation only; numeric claims follow `THESIS_IMPLEMENTATION.md` and thesis tables).

**After editing any file in `output/thesis/imgs/new`,** overwrite the `fig_capstone_*.png` targets (same mapping as the table below), then rebuild `main.pdf` so the IEEE document shows the new pixels. Wide figures use `figure*` in `capstone_visuals.tex` for two-column layout.

| LaTeX filename | Source |
|----------------|--------|
| `fig_capstone_moe_framework.png` | `output/thesis/imgs/new/Screenshot 2026-04-06 011829.png` |
| `fig_capstone_bert_pipeline.png` | `output/thesis/imgs/new/Screenshot 2026-04-06 011847.png` |
| `fig_capstone_ensemble_eval_pipeline.png` | `output/thesis/imgs/new/Screenshot 2026-04-06 011858.png` |
| `fig_capstone_dataset_inventory.png` | `output/thesis/imgs/new/Screenshot 2026-04-06 011911.png` |
| `fig_capstone_ml_binary_ternary_f1.png` | `output/thesis/imgs/new/Screenshot 2026-04-06 011917.png` |
| `fig_capstone_robustness_acc_f1.png` | `output/thesis/imgs/new/Screenshot 2026-04-06 011922.png` |

Regenerate all charts from repo root:

```powershell
$env:MPLCONFIGDIR = "$env:TEMP\mplconfig-charts"
pip install -r output/scripts/requirements-charts.txt
python output/scripts/generate_all_charts.py --repo-root .
```

Then re-copy PNGs if paths change; see `output/charts/index.md` for the full manifest.

---
name: TEMP output charts pipeline
overview: Add reproducible Python tooling under [TEMP/output/scripts](TEMP/output/scripts) that reads metrics, model catalogs, dataset EDA profiles, and path inventory, then writes matplotlib/seaborn figures into [TEMP/output/charts](TEMP/output/charts) with a clear subdirectory layout per data source and label mode.
todos:
  - id: deps-and-skeleton
    content: Add requirements-charts.txt, generate_all_charts.py argparse + path resolution + index.md writer
    status: pending
  - id: metrics-plots
    content: Implement CSV/JSON loaders and charts A1–A7 + B8 (metrics/, metrics/combined/, metrics/confusion/)
    status: pending
  - id: eda-models-path
    content: Implement dataset profile + label CSV charts; checkpoints/configs; inventory aggregation charts
    status: pending
  - id: shell-runner
    content: Add run_generate_charts.sh; run once and verify PNGs + index.md
    status: pending
isProject: false
---

# Chart generation from TEMP/output artifacts

## Current data (what we will plot)


| Source            | Key files                                                                                                                                                                                                                                        | Plot-ready fields                                                                                                                                |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Metrics           | [TEMP/output/metrics/2label_metrics_table.csv](TEMP/output/metrics/2label_metrics_table.csv), [3label_metrics_table.csv](TEMP/output/metrics/3label_metrics_table.csv), [summary.csv](TEMP/output/metrics/summary.csv), nested `**/metrics.json` | `safe_stem`, `model_id`, `accuracy`, `f1_macro`, `balanced_accuracy`, `roc_auc_ovr_macro`, `error`, `confusion_matrix` / `confusion_matrix_json` |
| Dataset EDA       | [TEMP/output/dataset analysis/](TEMP/output/dataset%20analysis/) — `per_file/**/profile.json`, `label_distribution_by_source_stem.csv`                                                                                                           | Label counts, `text_length_stats_sample` quantiles, `source_stem_value_counts`                                                                   |
| Models export     | [TEMP/output/models/checkpoints_inventory.csv](TEMP/output/models/checkpoints_inventory.csv), [configs_catalog.csv](TEMP/output/models/configs_catalog.csv), [logs_index.csv](TEMP/output/models/logs_index.csv)                                 | `size_bytes`, `model_class`, log sizes                                                                                                           |
| Path inventory    | [TEMP/output/path/inventory_all.csv](TEMP/output/path/inventory_all.csv)                                                                                                                                                                         | `relative_path`, `kind`, `size_bytes` (aggregate by top-level segment)                                                                           |
| Consolidated docs | [TEMP/output/docs/](TEMP/output/docs/)                                                                                                                                                                                                           | No numeric time series; optional skip or a single “doc rollup” placeholder — **low priority** vs tabular sources                                 |


**Reality check:** Today most metric rows are **HRAST-only**; the code must **discover** `safe_stem` values from CSVs/JSON so when more stems appear, charts scale without edits.

## Output layout under `TEMP/output/charts/`

Proposed structure (flat enough to browse, grouped by theme):

- `metrics/{2label,3label}/` — per-stem and global comparison PNGs
- `metrics/combined/` — 2-vs-3 label comparisons where join keys exist
- `metrics/confusion/` — heatmaps from `metrics.json` (one file per model, or batched facets with a sane max grid)
- `dataset_analysis/` — label bars, text-length summaries per profile
- `models/` — checkpoint sizes, config counts by class
- `path/` — top-level directory counts / bytes from inventory CSV
- `index.md` — auto-generated list of figures + how they were produced

Use **150–200 DPI** PNG (and optional `--format pdf`) for thesis-friendly export.

## Chart catalog (“all possible” within this repo)

**A. Metrics (primary)**

1. **Bar — model ranking by F1 (or accuracy)** — one chart per `(label_mode, safe_stem)`; color by `class_name` or parsed family from `model_id` (e.g. `transformers`_, `feature_encoder_`, `ml_`).
2. **Grouped bar — macro vs weighted F1** — `f1_macro` vs `f1_weighted` for top-N models per stem.
3. **Heatmap — models × scalar metrics** — columns: accuracy, balanced_accuracy, f1_macro, matthews_corrcoef, cohen_kappa, etc.; filter rows with any metric non-null.
4. **Confusion matrix heatmaps** — `seaborn.heatmap` from `metrics.metrics.confusion_matrix` in each `metrics.json` (skip if missing or load error).
5. **Bar — ROC AUC** — where `roc_auc_ovr_macro` is numeric; annotate skips using `roc_auc_skip_reason` / `metrics.roc_auc_skip_reason`.
6. **Status strip — eval errors** — bar or table-style chart: count of rows with non-empty `error` vs successful evals (from CSV).
7. **Scatter — accuracy vs f1_macro** — from `summary.csv` or wide tables, hue=`n_classes` or label_mode.

**B. Cross label-mode**

1. **Matched comparison** — derive a short key from `model_id` (strip `2_labels`/`3_labels` and stem suffix) and plot paired bars for F1 when the same logical model exists in both modes for the same `safe_stem`.

**C. Dataset analysis**

1. **Bar — label distribution** — from `label_raw_value_counts_top20` or `label_distribution_normalized_{2,3}` in each `profile.json`.
2. **Bar or line — text length quantiles** — from `text_length_stats_sample` (p50/p90/p95/p99 vs char_len_mean).
3. **Bar — per-stem label counts** — from `label_distribution_by_source_stem.csv` when present.

**D. Models / path**

1. **Horizontal bar — largest checkpoints** — top 25 by `size_bytes` from `checkpoints_inventory.csv`; label with basename of path.
2. **Bar — config count by `model_class`** — from `configs_catalog.csv` (drop nulls).
3. **Bar — file count and total bytes by top-level repo segment** — parse first path component of `relative_path` in `inventory_all.csv` (files only, optional dirs separately).

**E. Optional / future**

1. **Logs size distribution** — from `logs_index.csv` if non-empty.
2. **MoE manifest summary** — if useful fields exist in [moe_manifests.json](TEMP/output/models/moe_manifests.json) after inspection.

`output/docs` markdown is narrative-only; **do not** spend complexity parsing it for charts unless you add a trivial “file count by rollup doc” — defer unless requested.

## Implementation design

- **Single entrypoint:** [TEMP/output/scripts/generate_all_charts.py](TEMP/output/scripts/generate_all_charts.py)
  - `argparse`: `--repo-root` (default: parent of `output/`, i.e. `TEMP`), `--charts-dir` (default `output/charts`), `--dpi`, optional `--only metrics|dataset|models|path`.
  - Resolve paths with `pathlib`; handle the **literal space** in `output/dataset analysis` via `repo_root / "output" / "dataset analysis"`.
  - Shared helpers: `safe_filename(s)`, `parse_confusion(obj)`, `load_metrics_json_glob`, `read_csv_optional`.
- **Idempotent runs:** overwrite PNGs; write `index.md` with timestamp and CLI args.
- **Dependencies:** add [TEMP/output/scripts/requirements-charts.txt](TEMP/output/scripts/requirements-charts.txt) listing `pandas`, `matplotlib`, `seaborn` (versions loose, e.g. `>=2.0`).
- **Runner:** [TEMP/output/scripts/run_generate_charts.sh](TEMP/output/scripts/run_generate_charts.sh) — `cd` to `TEMP`, `python output/scripts/generate_all_charts.py` (or `python3`).

## Verification

- Run script on current tree; confirm `TEMP/output/charts/**.png` exist for HRAST 2-label and 3-label at minimum.
- Open `index.md` and spot-check: ranking bar, one confusion heatmap, one dataset label bar, one checkpoint-size bar.
- Handle empty/missing folders gracefully (log warning, continue).

## Non-goals

- Fixing metric pipeline bugs (e.g. ROC AUC skip reasons).
- Reading raw Parquet for charts (use existing EDA `profile.json` only).
- Interactive Plotly/HTML unless you add a follow-up.


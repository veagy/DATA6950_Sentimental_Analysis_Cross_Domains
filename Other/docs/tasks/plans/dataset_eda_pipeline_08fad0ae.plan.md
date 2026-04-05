---
name: Dataset EDA pipeline
overview: Add a reproducible Python tool that scans all Parquet datasets under [TEMP/data](TEMP/data), runs a structured EDA aligned with common ML/NLP practice (informed by public guidance on text-classification EDA and dataset quality checklists), and writes reports under `TEMP/output/dataset analysis` (literal path with a space, per your choice).
todos:
  - id: tool-script
    content: "Implement analyze_thesis_datasets.py: discover parquets, schema/nulls/labels/text-length/source_stem, optional alignment + plots, write index.md + summary CSV + per-file profile.json/report.md"
    status: completed
  - id: shell-wrapper
    content: Add scripts/run_dataset_analysis.sh with quoted path to TEMP/output/dataset analysis
    status: completed
  - id: smoke-verify
    content: Run smoke with max-rows cap; confirm outputs and sampled-vs-exact flags in profile.json
    status: completed
isProject: false
---

# Dataset analysis for TEMP/data

## Scope and reality check

- **"All possible"** is not a finite set. The implementation will cover a **fixed, documented checklist** of analyses that matter for your thesis parquets (processed text + transformed features, merged and per-`source_stem`), with CLI flags to **sample rows** or **cap work** on multi-million-row files so runs finish on one machine.
- **Internet use:** The checklist will explicitly align with widely cited practice for **text classification exploration** (e.g. [Google ML: Explore your data](https://developers.google.com/machine-learning/guides/text-classification/step-2)) and **dataset quality / checklist-style thinking** (e.g. [Data checklists / usable information](https://arxiv.org/html/2408.02919v1)). The script itself will not scrape the web at runtime; the plan encodes those practices in code.
- **Data discovery:** Recursively find `**/*.parquet` under [TEMP/data](TEMP/data) (e.g. `processed/`, `transformed/`, `processed/by_source_stem/`, `transformed/by_source_stem/`). If a tree is missing in a given clone, the tool still runs and reports "no files."

## Output layout (literal directory with space)

Create and write under:

`TEMP/output/dataset analysis/`

Suggested artifacts:


| Artifact                                       | Purpose                                                                                        |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `index.md`                                     | Run timestamp, data root, CLI args, list of files analyzed, links/paths to children            |
| `summary_all_files.csv`                        | One row per parquet: path, rows (exact or estimated), columns, size on disk, key column stats  |
| `per_file/<relative_path_sanitized>/report.md` | Human-readable narrative + tables for that file                                                |
| `per_file/.../profile.json`                    | Machine-readable stats (schema, dtypes, null rates, label counts, text length quantiles, etc.) |
| `figures/.../*.png`                            | Optional histograms (text length, label counts) if `--plots` and matplotlib available          |


Use a **safe slug** for nested folders derived from relative path (replace `/` with `__`).

## Analyses to implement (checklist)

**Global / file-level**

- Path, file size, Parquet metadata: row count, schema (names, types), compression.
- Per-column: null count / null rate, n_distinct (exact for small cardinality; sample-based or skipped for huge string columns).
- Duplicate row rate on a **key** if identifiable (e.g. hash of text column + label) — optional `--dedupe-key` or auto-detect text column.

**Sentiment / thesis-specific (when columns exist)**

- Infer **text column** and **label column** using the same heuristics as [TEMP/Code/thesis/common/datasets.py](TEMP/Code/thesis/common/datasets.py) (`_infer_text_column`, `_infer_label_column`) — import or duplicate minimal logic to avoid drift.
- If `source_stem` exists: counts per stem, label distribution per stem (CSV: `label_distribution_by_source_stem.csv` per file or merged section in JSON).
- **Label distribution:** raw `sentiment_value` (or inferred label col) and, for 2- and 3-class views, counts after applying `normalize_label_for_n_classes` from the same module (reuses star-rating rules).
- **Text lengths:** char length and word count (whitespace split): min, max, mean, std, quantiles (p50/p90/p95/p99); flag empty or very short texts.

**Processed vs transformed alignment (optional pass)**

- If both `processed/...` and `transformed/...` exist for the same logical stem (e.g. `all-data` or matching `by_source_stem` shards): compare **row counts**; if equal and both have `source_stem`, compare stem-wise counts; document mismatches (feeds training/MoE alignment debugging).

**Privacy / scale**

- No external API calls; no uploading data.
- For very large files: default `**--max-rows-per-file`** (e.g. 500k) for expensive passes (length stats, distinct); always record whether stats are **exact** or **sampled** in `profile.json`.

## Implementation

- **New script:** [TEMP/Code/thesis/tools/analyze_thesis_datasets.py](TEMP/Code/thesis/tools/analyze_thesis_datasets.py)  
  - Args: `--data-root`, `--output-dir` (default `.../output/dataset analysis`), `--max-rows-per-file`, `--plots`, `--include-by-source-stem` (default true), optional `--only-glob`.
- **New shell wrapper:** [TEMP/scripts/run_dataset_analysis.sh](TEMP/scripts/run_dataset_analysis.sh) — `cd` to TEMP, `PYTHONPATH=.`, quoted output path.
- **Dependencies:** `pandas`, `pyarrow` (already used in thesis). Optional: `matplotlib` for plots; do **not** require `ydata-profiling` by default (heavy); document optional install in `index.md` if you want full HTML profiles later.

## Verification

- Run on a small parquet (or `--max-rows-per-file 10000`) and confirm `index.md`, one `profile.json`, and `summary_all_files.csv` appear under `TEMP/output/dataset analysis`.
- Manually spot-check label counts against a known `value_counts()` from pandas on the same file.

## Non-goals (out of scope for v1)

- Training or model inference.
- Automatic "fixing" of data; only reporting.
- Full CheckList / TextAttack behavioral test suites (could be a follow-up).


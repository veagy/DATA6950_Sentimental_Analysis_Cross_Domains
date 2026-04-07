# Cursor plans digest (and `docs/plans` copies)

All paths below are relative to **`TEMP/`**.

## `docs/plans/` vs `.cursor/plans/`

`cmp` on representative pairs shows **docs copies are not byte-identical** to `.cursor/plans` files:

- `docs/plans/feature_models_all-data_pretrain_e9470305.plan.md` ≠ `.cursor/plans/temp_path_and_code_inventory_831c6a62.plan.md`
- `docs/plans/pretrain_scripts_blackwell_resume_3c0bbcf9.plan.md` ≠ `.cursor/plans/hrm_pretrain_gpu_validation_3e7152b9.plan.md`

**Thematic overlap:** “Feature models all-data pretrain” is an encoder-pretrain product plan; “TEMP path and Code inventory” is tooling/export. “Blackwell resume” focuses on `run_thesis_pretrain.sh` and resume meta; “HRM pretrain GPU validation” focuses on `run_hrm_encoder_pretrain_only.sh` and MLM validation. Keep both locations in mind when searching.

---

## `.cursor/plans/model_documentation_export_d8e5c801.plan.md`

- **Overview:** Python exporter + shell wrapper → `output/models/` with config/checkpoint/`run_meta.txt` catalogs, training entrypoints, scripts index, optional docs mirror.
- **Todos:** export script, shell wrapper, verify run — **all completed**.
- **Key outputs:** `index.md`, `configs_catalog.csv/json`, `checkpoints_inventory.csv`, `run_meta_parsed.jsonl`, `training_entrypoints.md`, `documentation_sources.md`, logs/DUMMY indexes.

## `.cursor/plans/dataset_eda_pipeline_08fad0ae.plan.md`

- **Overview:** `analyze_thesis_datasets.py` scans Parquets under `data/`, structured EDA, writes to **`output/dataset analysis/`** (literal space in path).
- **Todos:** tool, `run_dataset_analysis.sh`, smoke verify — **all completed**.
- **Analyses:** Schema, nulls, label distributions, text length stats, optional `source_stem` breakdowns, optional processed/transformed alignment, sampling caps for huge files.

## `.cursor/plans/per-stem_metrics_script_449c9c30.plan.md`

- **Overview:** Split merged all-data parquets by `source_stem`; evaluate each checkpoint on **per-stem shards** only; JSON/CSV under `output/metrics`; MoE uniform-mixture eval path.
- **Todos:** split parquets, checkpoint index, eval core, MoE uniform, shell/docs — **all completed**.
- **Policy:** Prefer `checkpoints/{K}-labels/{stem}/…`, fallback `all-data`; parse `run_meta.txt` for nested layouts; exclude pretrain/backbone dirs by default.

## `.cursor/plans/hrm_pretrain_gpu_validation_3e7152b9.plan.md`

- **Overview:** Validate HRM encoder MLM pretrain and resume—local 8GB smoke vs dual-GPU cloud benchmarks; encoder-only checkpoints; logs under `logs/`.
- **Todos:** encoder-only code path, local fwd/bwd smoke, live resume unit tests, periodic integration, DDP SIGTERM resume, final safetensors, cloud timing — **all completed**.
- **Ops:** Foreground/tmux/nohup patterns for `run_hrm_encoder_pretrain_only.sh`; links to `train_single.py`, `resume_checkpoint.py`.

## `.cursor/plans/temp_path_and_code_inventory_831c6a62.plan.md`

- **Overview:** Add `export_temp_path_inventory.py` → `output/path/` (recursive inventory with default excludes); extend `export_model_documentation.py` with Python inventory, `validation_entrypoints.md`, `code_tools_index.md`.
- **Todos:** path-inventory script **in_progress**; extend model export **pending**; verify both **pending** (status as recorded in plan frontmatter).

## `.cursor/plans/temp_docs_consolidation_4b09a4b2.plan.md`

- **Overview:** Meta-plan driving this folder: consolidate `docs/` + `.cursor/plans` summaries into `output/docs/`.
- **Note:** Implementation lives in the markdown rollups here; do not edit the plan file in the agent workflow per user instruction.

---

## `docs/plans/feature_models_all-data_pretrain_e9470305.plan.md` (separate from path-inventory plan)

- **Overview:** Pretrain **five** encoders (FFNN, CNN, LSTM, vanilla RNN, GRU) on `data/transformed/all-data.parquet`; **100-D** embedding output; **50k–80k** trainable param budget; no K-way head in pretrain.
- **Todos:** canonical JSONs, model encoder implementation, `train_single` pretrain/finetune split, lazy parquet dataset, queue orchestration, shell/docs — **mixed** (canonical configs marked in progress in frontmatter).

## `docs/plans/pretrain_scripts_blackwell_resume_3c0bbcf9.plan.md` (separate from HRM validation plan)

- **Overview:** Tune `run_thesis_pretrain.sh` for 96GB GPUs using `train-optimize.txt` ideas; **mid-epoch** resume with `steps_completed_in_epoch`, dual time/step save triggers, deterministic batch skip on DDP.
- **Todos:** resume meta, train loops, determinism, shell script, smoke resume — **all completed** in frontmatter.

## `docs/ml/docs_ml_training_spec_6d99ff52.plan.md`

- **Overview:** Plan that introduced `docs/ml/README.md` and `TRAINING_PIPELINES.md` (three tracks, mermaid, gaps).
- **Todos:** **completed** — see live docs in [`docs/ml/`](../../docs/ml/).

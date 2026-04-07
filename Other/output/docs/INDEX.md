# TEMP documentation index (consolidated)

This folder summarizes material under [`docs/`](../../docs) and [`.cursor/plans/`](../../.cursor/plans). Authoritative prose remains in those paths; use the rollups below for navigation and quick recall.

## Rollup documents

| Document | Contents |
|----------|----------|
| [PROJECT_AND_MODELS.md](PROJECT_AND_MODELS.md) | Capstone README highlights, model families, config inventory, parameters, stacking/MoE, `task.txt` rules |
| [ARCHITECTURE_PIPELINE.md](ARCHITECTURE_PIPELINE.md) | Preprocessing, embedding/UMAP, final normalization (three architecture notes) |
| [TRAINING_ML_AND_RUNBOOKS.md](TRAINING_ML_AND_RUNBOOKS.md) | ML training tracks, HRM pretrain runbook, embed_reduce plan, fine-tuning/data pipeline, training optimization notes |
| [IMPLEMENTATION_SUMMARIES.md](IMPLEMENTATION_SUMMARIES.md) | HRM and transformer MLP head + smoke summaries |
| [PROGRESS_AND_ORCHESTRATOR.md](PROGRESS_AND_ORCHESTRATOR.md) | Progress reports, dataset analysis notes, PreprocessText orchestrator session |
| [CURSOR_PLANS_DIGEST.md](CURSOR_PLANS_DIGEST.md) | Cursor plan files: goals, deliverables, todo status; relation to `docs/plans/` |
| [SOURCE_MANIFEST.txt](SOURCE_MANIFEST.txt) | Every scanned path with line count (or `binary`) and SHA-256; fields are tab-separated: `path`, `lines|binary`, `sha256` |

## Source manifest (all files under `docs/` and `.cursor/plans/`)

Paths are relative to the `TEMP/` directory.

| Path | Category | One-line purpose |
|------|----------|------------------|
| `docs/README.md` | Project | Long-form capstone README: 59 models, HRM two-stage training, PowerShell setup/train/eval, directory layout, backup |
| `docs/task.txt` | Rules | Data roots (`processed` vs `transformed`), checkpoint/log conventions, tqdm, training queue and fine-tune roadmap |
| `docs/train-optimize.txt` | Notes | HRM pretrain throughput knobs (batch, DDP, TF32, num_workers, tokenization vs GPU bound) |
| `docs/thesis_config_inventory.md` | Config | Maps `Code/thesis/config/**/*.json` to classes, doc IDs, data modality; gaps (B11/B12/B13, attention vs plain RNN, HRM ~105M) |
| `docs/thesis_parameter_counts.md` | Config | Per-config parameter table from `count_model_params.py` (incl. build errors for some rows) |
| `docs/models_overview.md` | Models | Conceptual overview: B1–B13, transformers, HRM E-HRM1, experts, MoE |
| `docs/Model_Parameters_and_Stacking.md` | Models | Param tables by modality; stacking groups; MoE gating and expert pool |
| `docs/mode-architecture_DataPreprocessing.md` | Architecture | Raw → clean → parquet in `data/processed`; `preprocess.py` |
| `docs/mode-architecture_FeatureExtraction.md` | Architecture | SentenceTransformer + UMAP → 100D; VRAM/RAM limits; batched transform |
| `docs/mode-architecture_FinalNormalization.md` | Architecture | Unified schema `text` + `sentiment_value`; per-file column mapping |
| `docs/implementation_plan.md` | Data | `embed_reduce.py` plan: ST local cache, 100k UMAP fit sample, chunked transform |
| `docs/Fine_tuning_and_data_pipeline_implementation_summary.md` | Training | End-to-end data prep mermaid, processed vs transformed, HRM/transformer MLP finetune, scripts index |
| `docs/hrm_encoder_pretrain_runbook.md` | Training | HRM MLM pretrain on `processed/all-data.parquet`, resume/DDP, smoke tool, tests |
| `docs/ml/README.md` | ML docs | Index to `TRAINING_PIPELINES.md`; track vs data file table |
| `docs/ml/TRAINING_PIPELINES.md` | ML docs | Tracks A/B/C: tabular, frozen-encoder meta-ML, proc+LR/SVC only; checkpoints, gaps |
| `docs/ml/docs_ml_training_spec_6d99ff52.plan.md` | Plan copy | Original plan that added `docs/ml/` (completed todos) |
| `docs/summary/transformer_mlp_head_finetune_implementation.md` | Implementation | Frozen HF backbone + MLP head; configs B3–B6; checkpoints; scripts |
| `docs/summary/transformer_mlp_finetune_smoke_implementation.md` | Implementation | Sandbox finetune under `DUMMY/transformer-mlp-finetune/` |
| `docs/summary/hrm_mlp_head_finetune_implementation.md` | Implementation | Frozen HRM + deep MLP head; 2-label neutral drop; checkpoint layout |
| `docs/summary/hrm_mlp_finetune_smoke_implementation.md` | Implementation | HRM finetune smoke under `DUMMY/fine-tune/` |
| `docs/plans/feature_models_all-data_pretrain_e9470305.plan.md` | Plan copy | Five 100D encoders on `transformed/all-data`; 50k–80k param cap (in progress todos) |
| `docs/plans/pretrain_scripts_blackwell_resume_3c0bbcf9.plan.md` | Plan copy | Blackwell pretrain script tuning + mid-epoch resume (mostly completed todos) |
| `docs/progress_reports/Progress_Report_01.md` | Progress | Week of 2026-01-26: failure analysis, datasets, next-week plan |
| `docs/progress_reports/Progress_Report_01.txt` | Progress | 2026-02-09: pipeline + model updates |
| `docs/progress_reports/Progress_Report_02.txt` | Progress | 2026-02-17: deep learning modules, attention library, LLM module |
| `docs/progress_reports/Progress_Report_03.txt` | Progress | 2026-03-25: thesis training runs, ML baselines, HRM in progress, MoE next |
| `docs/progress_reports/dataset_analysis_points.txt` | Progress | Bullet notes for Yelp, Financial PhraseBank, healthcare datasets |
| `docs/progress_reports/Progress_Report_01.docx` | Progress | Word export (binary) |
| `docs/progress_reports/Progress_Report_02.docx` | Progress | Word export (binary) |
| `docs/progress_reports/Progress_Report_03.docx` | Progress | Word export (binary) |
| `docs/progress_reports/Progress_Report_04.docx` | Progress | Word export (binary) |
| `docs/progress_reports/Progress_Report_01.pdf` | Progress | PDF export (binary) |
| `docs/progress_reports/Progress_Report_02.pdf` | Progress | PDF export (binary) |
| `docs/progress_reports/Progress_Report_03.pdf` | Progress | PDF export (binary) |
| `docs/progress_reports/~$ogress_Report_04.docx` | Temp | Office lock file for Progress_Report_04 (binary) |
| `docs/tasks/.../master_plan.md` | Orchestrator | PreprocessText session task table (completed) |
| `docs/tasks/.../Orchestrator_Summary.md` | Orchestrator | Verification and scope for preprocessing session |
| `docs/tasks/.../01_preprocess_text.task.md` | Orchestrator | Task spec: `preprocess.py` from raw to parquet |
| `.cursor/plans/model_documentation_export_d8e5c801.plan.md` | Cursor | Exporter for `output/models/` (completed) |
| `.cursor/plans/dataset_eda_pipeline_08fad0ae.plan.md` | Cursor | EDA tool → `output/dataset analysis/` (completed) |
| `.cursor/plans/per-stem_metrics_script_449c9c30.plan.md` | Cursor | Per-`source_stem` eval → `output/metrics` (completed) |
| `.cursor/plans/hrm_pretrain_gpu_validation_3e7152b9.plan.md` | Cursor | HRM encoder MLM validation local + cloud (completed) |
| `.cursor/plans/temp_path_and_code_inventory_831c6a62.plan.md` | Cursor | Path inventory + extend model export (in progress / pending todos) |
| `.cursor/plans/temp_docs_consolidation_4b09a4b2.plan.md` | Cursor | Meta-plan: this consolidation into `output/docs` |

**Note:** [`docs/plans/`](../../docs/plans) files are **not** byte-identical to any single `.cursor/plans/` file (`cmp` differed). Themes overlap (feature pretrain vs path inventory; Blackwell resume vs HRM GPU validation) but filenames and bodies differ—see [CURSOR_PLANS_DIGEST.md](CURSOR_PLANS_DIGEST.md).

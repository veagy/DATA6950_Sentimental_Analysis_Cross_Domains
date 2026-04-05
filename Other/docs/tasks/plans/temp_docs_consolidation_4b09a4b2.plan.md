---
name: TEMP docs consolidation
overview: Consolidate and summarize everything under [TEMP/docs](TEMP/docs) and [TEMP/.cursor/plans](TEMP/.cursor/plans) into a small, navigable documentation set under [TEMP/output/docs](TEMP/output/docs), with a master index, topic rollups, and per-plan digests (without losing pointers back to originals).
todos:
  - id: read-all-sources
    content: Read every file under TEMP/docs and TEMP/.cursor/plans (including nested dirs); note overlap between docs/plans and .cursor/plans
    status: pending
  - id: write-index
    content: Create TEMP/output/docs/INDEX.md with full source manifest table and links
    status: pending
  - id: write-topic-md
    content: Author PROJECT_AND_MODELS, ARCHITECTURE_PIPELINE, TRAINING_ML_AND_RUNBOOKS, IMPLEMENTATION_SUMMARIES, PROGRESS_AND_ORCHESTRATOR, CURSOR_PLANS_DIGEST
    status: pending
  - id: optional-manifest
    content: Add SOURCE_MANIFEST.txt (paths + line counts or hashes) if useful
    status: pending
  - id: verify-coverage
    content: Cross-check INDEX against glob lists; fix any missing files
    status: pending
isProject: false
---

# Consolidate TEMP docs and Cursor plans into `output/docs`

## Source inventory (readonly scan)


| Area                             | Location                                                                                                                                                                                                                                                                                                                                                | Role                                                                                     |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Project / thesis                 | [TEMP/docs/README.md](TEMP/docs/README.md), [thesis_config_inventory.md](TEMP/docs/thesis_config_inventory.md), [thesis_parameter_counts.md](TEMP/docs/thesis_parameter_counts.md), [models_overview.md](TEMP/docs/models_overview.md), [Model_Parameters_and_Stacking.md](TEMP/docs/Model_Parameters_and_Stacking.md)                                  | Long-form README (~800+ lines), model and thesis metadata                                |
| Architecture                     | [mode-architecture_DataPreprocessing.md](TEMP/docs/mode-architecture_DataPreprocessing.md), [mode-architecture_FeatureExtraction.md](TEMP/docs/mode-architecture_FeatureExtraction.md), [mode-architecture_FinalNormalization.md](TEMP/docs/mode-architecture_FinalNormalization.md)                                                                    | Design specs per pipeline stage                                                          |
| ML / training                    | [TEMP/docs/ml/](TEMP/docs/ml/) (README, TRAINING_PIPELINES, training spec plan), [hrm_encoder_pretrain_runbook.md](TEMP/docs/hrm_encoder_pretrain_runbook.md), [implementation_plan.md](TEMP/docs/implementation_plan.md), [Fine_tuning_and_data_pipeline_implementation_summary.md](TEMP/docs/Fine_tuning_and_data_pipeline_implementation_summary.md) | Pipelines, runbooks, implementation notes                                                |
| Summaries (fine-tune smoke/head) | [TEMP/docs/summary/](TEMP/docs/summary/)                                                                                                                                                                                                                                                                                                                | Four implementation summaries                                                            |
| Progress / misc                  | [TEMP/docs/progress_reports/](TEMP/docs/progress_reports/), [task.txt](TEMP/docs/task.txt), [train-optimize.txt](TEMP/docs/train-optimize.txt)                                                                                                                                                                                                          | Status text and scratch notes                                                            |
| Orchestrator session             | [TEMP/docs/tasks/orchestrator-sessions/PreprocessText/](TEMP/docs/tasks/orchestrator-sessions/PreprocessText/)                                                                                                                                                                                                                                          | `master_plan.md`, task file, `Orchestrator_Summary.md`                                   |
| Plans inside docs                | [TEMP/docs/plans/](TEMP/docs/plans/) (2 files)                                                                                                                                                                                                                                                                                                          | Same naming pattern as some Cursor plans — **compare and dedupe** in the written summary |
| Cursor plans                     | [TEMP/.cursor/plans/](TEMP/.cursor/plans/) (5 files)                                                                                                                                                                                                                                                                                                    | YAML frontmatter + goals, context, steps, mermaid                                        |


Target directory [TEMP/output/docs](TEMP/output/docs) is currently empty; implementation will create it.

## Recommended output shape (readable, not a raw dump)

Avoid pasting full copies of very large files (e.g. the main README). Instead:

1. **[TEMP/output/docs/INDEX.md](TEMP/output/docs/INDEX.md)** — Canonical entry point: table of **every** source file with category, **one-line purpose**, and relative path from `TEMP/` for quick opening in the repo.
2. **[TEMP/output/docs/PROJECT_AND_MODELS.md](TEMP/output/docs/PROJECT_AND_MODELS.md)** — Synthesized overview: capstone metadata, model families/counts, stacking/parameters highlights, setup/train/eval flow (bullets + key paths), all anchored with links to the underlying files.
3. **[TEMP/output/docs/ARCHITECTURE_PIPELINE.md](TEMP/output/docs/ARCHITECTURE_PIPELINE.md)** — Merged narrative of the three `mode-architecture_*.md` docs: goals, components, data flow, acceptance-style outcomes; short mermaid optional if it clarifies stage order.
4. **[TEMP/output/docs/TRAINING_ML_AND_RUNBOOKS.md](TEMP/output/docs/TRAINING_ML_AND_RUNBOOKS.md)** — Unified summary of `ml/`, HRM pretrain runbook, implementation plan, fine-tuning/pipeline summary: what runs where, which scripts/docs matter, dependencies/risks called out in sources.
5. **[TEMP/output/docs/IMPLEMENTATION_SUMMARIES.md](TEMP/output/docs/IMPLEMENTATION_SUMMARIES.md)** — One subsection per file under `docs/summary/`: objective, what was implemented, verification/smoke notes.
6. **[TEMP/output/docs/PROGRESS_AND_ORCHESTRATOR.md](TEMP/output/docs/PROGRESS_AND_ORCHESTRATOR.md)** — Condense progress reports + PreprocessText orchestrator session (objectives, task table, completion status).
7. **[TEMP/output/docs/CURSOR_PLANS_DIGEST.md](TEMP/output/docs/CURSOR_PLANS_DIGEST.md)** — For each of the five `[.cursor/plans/*.plan.md](TEMP/.cursor/plans)` files: title/overview from frontmatter, intent, main deliverables, todo status if present, risks/decisions, and “related code paths” called out in the plan body. Append a short note for [TEMP/docs/plans/](TEMP/docs/plans) if content duplicates a Cursor plan (state “identical” or “differs in …”).
8. **[TEMP/output/docs/SOURCE_MANIFEST.txt](TEMP/output/docs/SOURCE_MANIFEST.txt)** (optional but useful) — Flat list of all scanned paths and SHA256 or `wc -l` for change detection later.

## Execution approach (after plan approval)

- Read every file under the two source roots (skipping nothing).
- Draft the seven markdown files above in one pass for consistency (cross-references between sections).
- If `docs/plans/`* matches `.cursor/plans/*` byte-for-byte or nearly, document that once in the digest to avoid duplicate reading for humans.

## Out of scope (unless you ask)

- Updating originals under `TEMP/docs` or `TEMP/.cursor/plans`.
- Automating regeneration via script (could be a follow-up if you want repeatable exports).

## Verification

- `TEMP/output/docs/` contains `INDEX.md` plus the topic files listed above.
- `INDEX.md` lists all 30 `docs` files and all 5 `.cursor/plans` files (and notes the 2 under `docs/plans`).
- Spot-check: every major topic from README sections (models, training, evaluation, backup) appears in `PROJECT_AND_MODELS.md` or `TRAINING_ML_AND_RUNBOOKS.md`.


---
name: docs/ml training spec
overview: Add structured documentation under `docs/ml/` that specifies three related training tracks (tabular classical ML on transformed data, frozen-encoder embedding stacks, processed-data meta-ML), aligned with existing thesis paths, data columns, and code references—without implementing training code in this step.
todos:
  - id: add-ml-readme
    content: Create docs/ml/README.md with index, links to TRAINING_PIPELINES.md and existing docs (task.txt, Model_Parameters, scripts/README).
    status: completed
  - id: add-training-pipelines-md
    content: Create docs/ml/TRAINING_PIPELINES.md with three tracks, source_stem convention, checkpoint/log paths, config gaps, safetensors caveat, and one mermaid diagram.
    status: completed
isProject: false
---

# Documentation plan: `docs/ml` training specification

## Context (from repo)

- **No `docs/ml/` folder exists yet** — create it with one primary spec plus a short index.
- **Data splits:** `[scripts/README.md](scripts/README.md)` and merge scripts describe `source_stem` on `[data/transformed/all-data.parquet](data/transformed/all-data.parquet)` and `[data/processed/all-data.parquet](data/processed/all-data.parquet)` for grouping rows back into per-dataset shards (`groupby("source_stem")`); document this as the canonical way to train “per dataset” from merged files.
- **Checkpoint layout (your choice):** Classical ML follows existing thesis convention from `[Code/thesis/train/train_single.py](Code/thesis/train/train_single.py)` — `_checkpoint_path` → `checkpoints/{2,3}-labels/{dataset_stem}/{config_stem}.safetensors`. The doc will describe this explicitly and use **“machine learning (tabular) family”** as terminology rather than a `machine_learning/` directory segment.
- **MoE / stack outputs:** `[Code/thesis/train/train_moe.py](Code/thesis/train/train_moe.py)` currently defaults gate checkpoints under `checkpoints/moe/`; the spec will define the **target** subtree `checkpoints/moe/ml_stack/` for the frozen-encoder + meta-classifier artifacts you described (implementation of that path can be a later code task).
- **Configs today:** `[Code/thesis/config/ml/](Code/thesis/config/ml/)` only has `E_ML1_LogisticRegression` and `E_ML2_LinearSVC` per label mode (`[Code/thesis/generate_thesis_configs.py](Code/thesis/generate_thesis_configs.py)`). Decision tree and random forest require **new JSON entries** whose top-level keys must match discoverable class names in `[Code/thesis/common/model_factory.py](Code/thesis/common/model_factory.py)` (e.g. `DecisionTreeClassifier`, `RandomForestClassifier` under `[Code/models/machine_learning/classification/](Code/models/machine_learning/classification/)`).
- **Safetensors + sklearn:** `[Code/models/utils/utils.py](Code/models/utils/utils.py)` warns that `MLModule` safetensors may lose non-parameter state; the doc should cite this and state acceptance criteria: only use safetensors where `state_dict()` after `fit` is sufficient, or require `.pt` for problematic estimators (transparently, as a project rule).
- **Normative project rules:** Cross-link `[docs/task.txt](docs/task.txt)` (processed vs transformed, 2- vs 3-label folders, logs under `logs/`, tqdm) and `[docs/Model_Parameters_and_Stacking.md](docs/Model_Parameters_and_Stacking.md)` (E-ML1/E-ML2, transformer expert IDs, MoE framing).

## Deliverables (files to add)


| File                                                             | Purpose                                                                                         |
| ---------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `[docs/ml/README.md](docs/ml/README.md)`                         | Index: links to the spec, related repo docs, and quick “which data file for which track” table. |
| `[docs/ml/TRAINING_PIPELINES.md](docs/ml/TRAINING_PIPELINES.md)` | Single source of truth for all three tracks below (sections + checklists + diagrams).           |


Optional later (only if you want finer split): `docs/ml/APPENDIX_paths_and_env.md` — defer unless the main doc grows too large.

## Content outline for `TRAINING_PIPELINES.md`

1. **Scope and goals** — Three pipelines; all checkpoints `.safetensors` where feasible; CPU emphasis for tabular sklearn-style training; logs only under `[logs/](logs/)`.
2. **Shared conventions**
  - Label modes: binary (2-class) vs ternary (3-class); neutral handling consistent with existing HRM/transformer notes in `[docs/spheron/docs/summary/](docs/spheron/docs/summary/)` where relevant.
  - **Per-dataset training from merged parquet:** filter or group on `source_stem`; `dataset_stem` argument to training scripts matches each distinct `source_stem` value (and optionally `all-data` when env/queue allows — reference `THESIS_QUEUE_INCLUDE_ALL_DATA` from `[scripts/README.md](scripts/README.md)`).
  - Logging: naming pattern e.g. `logs/ml_tabular_{stem}_{2|3}label_{model}.log` (recommendation, not enforced until scripts exist).
3. **Track A — Tabular classical ML (transformed features)**
  - **Input:** `[data/transformed/all-data.parquet](data/transformed/all-data.parquet)` (100D features + labels); split by `source_stem`.
  - **Models:** Logistic regression, SVM (project uses `LinearSVC`-style configs today), decision tree, random forest — each trained **separately** per `(source_stem, label_mode)`.
  - **Output path:** `checkpoints/{2,3}-labels/{dataset_stem}/{ConfigStem}.safetensors` (current `[train_single.py](Code/thesis/train/train_single.py)` behavior).
  - **Execution note:** Classical path in `train_single` loads full dataset into memory for `fit` — document “CPU-only, batching N/A for sklearn fit”; for speed, recommend `n_jobs` / solver choices in JSON where applicable.
  - **Gap callout:** Add four config files per label folder for the four algorithms (two exist).
4. **Track B — Frozen transformer + HRM embeddings → meta-ML (MoE stack area)**
  - **Input:** `[data/processed/all-data.parquet](data/processed/all-data.parquet)` (text + labels); split by `source_stem`.
  - **Encoders (inference only, weights frozen):** BERT, RoBERTa, BART, DistilBERT (per `[docs/Model_Parameters_and_Stacking.md](docs/Model_Parameters_and_Stacking.md)`); HRM encoder weights from `[checkpoints/hrm](checkpoints/hrm)` (document that exact checkpoint files are versioned by your training runs).
  - **Procedure (spec level):** run forward passes to materialize embedding tensors (or a cached embedding parquet — optional optimization); concatenate or stack features; train **only** shallow heads / sklearn wrappers; no backprop through encoders.
  - **Output path (target):** `checkpoints/moe/ml_stack/{2,3}-labels/{dataset_stem}/...safetensors` with a clear filename convention (e.g. include encoder id + meta-model id).
  - **Cross-reference:** `[Code/models](Code/models)` for wrapper classes; `[Code/thesis/train/train_stack.py](Code/thesis/train/train_stack.py)` for prior art on combining frozen experts (adapt conceptually for embedding features).
5. **Track C — Processed pipeline, SVC + logistic regression only**
  - **Input:** same as Track B; **models restricted** to SVC and logistic regression (align wording with `[docs/task.txt](docs/task.txt)` and E-ML1 / E-ML2 in stacking doc).
  - **Same freezing rule:** transformers and HRM frozen; only meta models train.
  - **Output:** either same `moe/ml_stack` subtree with a naming suffix (`_proc_lr`, `_proc_svc`) or parallel folder — pick one in the doc and stick to it for reproducibility.
6. **Mermaid diagram (one flowchart)** — Data source → split by `source_stem` → feature type (100D vs embeddings) → trainable component → checkpoint leaf.
7. **Definition of done (documentation)** — Reader can answer: which parquet, which column for splits, which configs, which checkpoint path, which logs, and what is not yet implemented in code.

## Out of scope for this task

- Implementing new training scripts, config JSON files, or changing checkpoint paths in Python.
- Running training or producing real checkpoints.

## Suggested follow-up (after doc approval)

- Add missing ML JSON configs and, if needed, extend checkpoint path logic for `checkpoints/moe/ml_stack/`.
- Add shell wrappers mirroring existing patterns in `[scripts/](scripts/)` for the three tracks.


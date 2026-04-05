# Machine learning documentation (`docs/ml`)

This folder holds specifications for classical and meta-learning training tracks in the thesis codebase. The detailed contract lives in one place:

| Document | Description |
|----------|-------------|
| [TRAINING_PIPELINES.md](TRAINING_PIPELINES.md) | Three training tracks (tabular on transformed data, frozen-encoder embedding stacks, processed-data SVC/LR), `source_stem` splits, checkpoint and log conventions, and implementation gaps. |

## Related project docs

| Topic | Location |
|-------|----------|
| Global rules (data roots, checkpoints, logs, tqdm) | [docs/task.txt](../task.txt) |
| Expert IDs, stacking groups, MoE framing | [docs/Model_Parameters_and_Stacking.md](../Model_Parameters_and_Stacking.md) |
| Merged parquets, `source_stem`, queue env vars | [scripts/README.md](../../scripts/README.md) |

## Which data file for which track?

| Track | Primary data | Feature type |
|-------|----------------|--------------|
| **A — Tabular classical ML** | `data/transformed/all-data.parquet` | 100D dense vectors (`features_100d`) + labels |
| **B — Frozen encoders + meta-ML** | `data/processed/all-data.parquet` | Text + labels; embeddings from frozen transformers + HRM |
| **C — Processed, SVC + logistic regression only** | `data/processed/all-data.parquet` | Same as B; meta-models restricted to SVC and logistic regression |

For all tracks that use merged `all-data.parquet`, **per-dataset** runs use the **`source_stem`** column to filter or `groupby` rows so each checkpoint corresponds to one originating dataset stem.

## Code entrypoints (reference)

- Single-config training: `Code/thesis/train/train_single.py` (classical ML on transformed features uses in-memory `fit` and `_checkpoint_path`).
- Stacking / frozen experts: `Code/thesis/train/train_stack.py`
- MoE gate training: `Code/thesis/train/train_moe.py`
- Model configs: `Code/thesis/config/` (ML baselines under `ml/2_labels/` and `ml/3_labels/`)
- Model implementations: `Code/models/`

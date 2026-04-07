# Implementation summaries (`docs/summary/`)

Each subsection condenses one file under [`docs/summary/`](../../docs/summary/). Full command tables and env vars remain in the originals.

## Transformer MLP head fine-tune — [`transformer_mlp_head_finetune_implementation.md`](../../docs/summary/transformer_mlp_head_finetune_implementation.md)

- **Purpose:** Frozen Hugging Face transformer + trainable MLP head for 2- or 3-way sentiment on **processed** parquet.
- **Head:** `single_linear_head=false` → Linear(768,1024) → GELU → LazyLinear(K); `CrossEntropyLoss` on logits.
- **Configs:** B3 DistilBERT, B4 BERT, B5 RoBERTa, B6 BART under `Code/thesis/config/transformers/{2_labels,3_labels}/*_mlp768_1024.json`.
- **Checkpoints:** `{checkpoint_root}/{K}-labels/{dataset_stem}/{config_stem}.safetensors` (no HRM layout flag).
- **Scripts:** `scripts/run_transformer_mlp_finetune.sh` plus 2/3-label and 4×GPU wrappers.
- **2-label nuance:** `exclude_neutral` applies to **HRM only**; transformer 2-class keeps neutral-labeled rows unless filtered elsewhere.

## Transformer MLP finetune smoke — [`transformer_mlp_finetune_smoke_implementation.md`](../../docs/summary/transformer_mlp_finetune_smoke_implementation.md)

- **Purpose:** ~1000-row sandbox using production launcher with isolated paths under `DUMMY/transformer-mlp-finetune/`.
- **Mechanisms:** `THESIS_MAX_SAMPLES`, sandbox `THESIS_CHECKPOINT_ROOT` / `THESIS_RESUME_TEMP`, aggressive periodic save env vars, `THESIS_DETACH=none`.
- **Artifacts:** Smoke shell + interrupt demo script; logs and checkpoints stay out of production trees unless overridden.

## HRM MLP head fine-tune — [`hrm_mlp_head_finetune_implementation.md`](../../docs/summary/hrm_mlp_head_finetune_implementation.md)

- **Purpose:** Frozen `HierarchicalReasoningModel` encoder + trainable MLP head (100-D pooled → deep MLP → K logits).
- **MLP topology:** 100 → 320 → 640 → 1250 → 640 → 320 → K with ReLU/GELU mix as implemented in `build_sentiment_mlp_head`.
- **Data:** `ParquetTextDataset`; **2-label** runs **drop neutral (`2`) by default**; `--no_hrm_exclude_neutral` to keep.
- **Weights:** Load encoder from `checkpoints/hrm/pretrain/{stem}/E_HRM1_4Level.safetensors`; save finetune under `checkpoints/hrm/fine-tune/...` when layout flag/env set.
- **Status table** in source maps plan items (configs, factory, train loop, scripts) to done/pending.

## HRM MLP finetune smoke — [`hrm_mlp_finetune_smoke_implementation.md`](../../docs/summary/hrm_mlp_finetune_smoke_implementation.md)

- **Purpose:** Same as transformer smoke pattern: `DUMMY/fine-tune/`, exec’s `run_hrm_finetune_mlp.sh`, interrupt/resume demo, single-GPU default for demo.
- **Prereqs:** Processed parquet + production encoder checkpoint path unless overridden.
- **Artifacts:** `run_hrm_mlp_finetune_smoke.sh`, `demo_resume_interrupt.sh`, sandbox logs/resume/checkpoints mirroring production layout flag.

## Parent rollup

The four summaries are cross-linked from [`Fine_tuning_and_data_pipeline_implementation_summary.md`](../../docs/Fine_tuning_and_data_pipeline_implementation_summary.md) (see [TRAINING_ML_AND_RUNBOOKS.md](TRAINING_ML_AND_RUNBOOKS.md)).

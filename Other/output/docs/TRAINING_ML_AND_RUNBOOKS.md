# Training, ML specs, and runbooks (synthesized)

Sources: [`docs/ml/README.md`](../../docs/ml/README.md), [`docs/ml/TRAINING_PIPELINES.md`](../../docs/ml/TRAINING_PIPELINES.md), [`docs/hrm_encoder_pretrain_runbook.md`](../../docs/hrm_encoder_pretrain_runbook.md), [`docs/implementation_plan.md`](../../docs/implementation_plan.md), [`docs/Fine_tuning_and_data_pipeline_implementation_summary.md`](../../docs/Fine_tuning_and_data_pipeline_implementation_summary.md), [`docs/train-optimize.txt`](../../docs/train-optimize.txt).

## Three ML training tracks ([`TRAINING_PIPELINES.md`](../../docs/ml/TRAINING_PIPELINES.md))

| Track | Data | Trainable parts | Checkpoint target (as documented) |
|-------|------|-----------------|-------------------------------------|
| **A — Tabular classical** | `data/transformed/all-data.parquet` (`features_100d` + labels) | Sklearn-style `fit` in memory (CPU) | `checkpoints/{2,3}-labels/{dataset_stem}/{ConfigStem}.safetensors` |
| **B — Frozen encoders + meta-ML** | `data/processed/all-data.parquet` | Shallow/meta on embeddings; encoders frozen | Target `checkpoints/moe/ml_stack/...` (code may still default to `checkpoints/moe/`) |
| **C — Processed + LR/SVC only** | Same as B | Only logistic regression / `LinearSVC`-style heads on embeddings | Same subtree as B with `proc_` filename convention suggested |

**Shared conventions:** Split merged parquets by **`source_stem`** for per-dataset training; 2-label vs 3-label runs separate; logs under `logs/` with suggested filename patterns; safetensors caveat for sklearn via `MLModule.save_pretrained` (non-tensor state).

**Documented gaps:** Decision tree / random forest JSONs under `config/ml/`; aligning on-disk `ml_stack` paths with `train_moe.py` defaults.

## HRM encoder MLM pretrain ([`hrm_encoder_pretrain_runbook.md`](../../docs/hrm_encoder_pretrain_runbook.md))

- **Data:** `data/processed/all-data.parquet` (text column); built via `merge_all_data_parquet.py`.
- **Launcher:** `scripts/run_hrm_encoder_pretrain_only.sh`.
- **Smoke:** `Code/thesis/tools/hrm_embed_smoke.py` on encoder safetensors (mean-pooled **100D** for E_HRM1).
- **Resume:** `LiveResumeDir`, `ResumeMeta`, periodic saves (`--save_every_steps`, `--save_every_minutes`), SIGINT handling in `train_loop_hrm_mlm`; live bundles under `logs/resume/`.
- **DDP:** `torch.distributed.run`, `distributed.py`, `find_unused_parameters=True` for HRM MLM.
- **Tests:** `Code/test/test_hrm_encoder_pretrain_integration_local.py` (periodic saves, resume after SIGTERM).

## Embed / reduce implementation plan ([`implementation_plan.md`](../../docs/implementation_plan.md))

- Adds **`embed_reduce.py`**: local MiniLM cache, iterate processed Parquets, PyArrow streaming, output `features_100d` + `sentiment_value` to `data/transformed/`.
- **Safety:** `batch_size=128` for encoding; UMAP fit on **100k** random rows per dataset (same theme as architecture doc).

## Fine-tuning and data pipeline summary ([`Fine_tuning_and_data_pipeline_implementation_summary.md`](../../docs/Fine_tuning_and_data_pipeline_implementation_summary.md))

- **Mermaid** links raw → normalized processed → merge/rewrite → `embed_reduce` → transformed merge/rewrite/`source_stem`.
- **Default finetune input:** `data/processed/{dataset_stem}.parquet` via `ParquetTextDataset`—transformed merge does not replace this unless explicitly changed.
- **Key scripts:** `normalize_datasets.py`, `preprocess.py`, `merge_all_data_parquet.py`, label rewrites, `embed_reduce.py`, `merge_all_transformed_parquet.py`, `add_source_stem_to_transformed_parquet.py`, etc.; **UMAP caveat** when merging per-stem transformed shards (different fits)—prefer merge processed then single `embed_reduce` on `all-data` for one space.
- **HRM MLP finetune:** Frozen encoder from MLM pretrain; trainable deep MLP head; configs `E_HRM1_4Level_ft_mlp_{2,3}label.json`; optional `--no_hrm_exclude_neutral`; checkpoint layout flag `hrm_finetune_checkpoint_layout`.
- **Transformer MLP finetune:** Frozen HF backbone; head 768→1024→GELU→LazyLinear(K); B3–B6 `*_mlp768_1024.json`; 2-class keeps neutral rows (unlike HRM); sequential all-8 script and queue-after-HRM script names documented.
- **Shared:** `train_single.py` CLI, DDP, resume env vars; log file name patterns under `logs/`.

## Training optimization notes ([`train-optimize.txt`](../../docs/train-optimize.txt))

- HRM memory scales with `seq_len`; tune `THESIS_HRM_BATCH` and DDP world size.
- Practical speedups: larger stable batch, `num_workers` 4–8, `--gc_every 0` if no fragmentation OOM, TF32 on Ampere+, optional bf16, `torch.compile`, future tokenization in dataloader vs hot loop, streaming parquet for huge tables.
- Explains **pre-tokenizing** tradeoffs (disk/RAM vs CPU overhead).

## ML folder index ([`docs/ml/README.md`](../../docs/ml/README.md))

- Points readers to `TRAINING_PIPELINES.md`, `task.txt`, `Model_Parameters_and_Stacking.md`, `scripts/README.md` for queue env vars.

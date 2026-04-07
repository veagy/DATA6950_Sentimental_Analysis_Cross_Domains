# Validation-related entrypoints

| Path | Summary |
| --- | --- |
| `Code/thesis/test/validate_all.py` | Evaluate checkpoints produced by train_single (safetensors). |
| `Code/thesis/tools/analyze_thesis_datasets.py` | Exploratory analysis for thesis Parquet datasets under data/processed and data/transformed. Writes reports under output/dataset analysis/ (default). See index.md for methodology notes. |
| `Code/thesis/tools/eval_per_source_stem_metrics.py` | Split merged all-data parquets by ``source_stem``, then evaluate each checkpointed model separately on split files only (under processed/transformed/{split_subdir}/). See TEMP/docs/ml/TRAINING_PIPELINES.md for data conventions. |

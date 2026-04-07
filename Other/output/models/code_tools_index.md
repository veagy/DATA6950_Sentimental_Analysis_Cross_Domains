# `Code/thesis/tools/` Python index

| Script | Summary |
| --- | --- |
| `analyze_thesis_datasets.py` | Exploratory analysis for thesis Parquet datasets under data/processed and data/transformed. Writes reports under output/dataset analysis/ (default). See index.md for methodology notes. |
| `count_model_params.py` | Count trainable/total parameters for each thesis JSON config (same build path as training). Run from repository root: python Code/thesis/tools/count_model_params.py python Code/thesis/tools/count_model_params.py --markdown-out docs/thesis_parameter_counts.md Skips: Code/thesis/config/moe/example_exp |
| `download_transformers.py` | Download Hugging Face base models + tokenizers into ``checkpoints/transformer/<name>``. Run from repository root (requires ``transformers``, ``torch``):: python Code/thesis/tools/download_transformers.py |
| `eval_per_source_stem_metrics.py` | Split merged all-data parquets by ``source_stem``, then evaluate each checkpointed model separately on split files only (under processed/transformed/{split_subdir}/). See TEMP/docs/ml/TRAINING_PIPELINES.md for data conventions. |
| `export_model_documentation.py` | Scan thesis configs, checkpoints (run_meta), training scripts, docs/logs/DUMMY indexes; write a model documentation catalog under output/models/. See output index.md after each run. |
| `export_temp_path_inventory.py` | Walk the repository tree and write a CSV + index under output/path/. Default excludes skip large or regenerated subtrees; use --no-default-excludes for a full tree. |
| `hrm_embed_smoke.py` | Load HRM encoder-only weights (config + safetensors), run mean-pooled embeddings on sample text. Run from repository root: python Code/thesis/tools/hrm_embed_smoke.py python Code/thesis/tools/hrm_embed_smoke.py \\ --config Code/thesis/config/hrm/E_HRM1_4Level.json \\ --checkpoint checkpoints/hrm/pre |
| `hrm_pretrain_hours_estimate.py` | Rough wall-time estimate for HRM MLM epochs on dual-GPU (or any) setup. T ≈ num_epochs * ceil(N / (R * B)) * t_step N = parquet rows (text), R = GPU count, B = per-GPU batch, t_step = seconds per optimizer step **measured on the target hardware** (not a laptop 4070). Usage: python Code/thesis/tools/ |
| `parquet_text_token_stats.py` | Compute max and mean token counts for a parquet text column using the HRM tokenizer (google-bert/bert-base-uncased by default). Full encode, no truncation. Run from repository root: python Code/thesis/tools/parquet_text_token_stats.py python Code/thesis/tools/parquet_text_token_stats.py --parquet pa |
| `transformer_mlp_head_smoke.py` | Load one transformer MLP-head JSON and run a single forward pass (head + embeddings path). |

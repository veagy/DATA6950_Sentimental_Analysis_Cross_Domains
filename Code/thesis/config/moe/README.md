# MoE expert manifests

These JSON files are **lists** consumed by `Code/thesis/train/train_moe.py --experts_json`.

Each entry:

- `name`: optional label
- `config`: path to thesis JSON (under `2_labels` or `3_labels`)
- `checkpoint`: path to `.safetensors` (repo-relative or absolute)
- `modality`: `text` for `LLMModule` (DistilBERT etc.), or `dense` for `FeatureEncoderClassifier` on 100-D features

**Default `all_data` manifests** (`experts_all_data_{2,3}label.json`) list **dense experts only**. In that case `train_moe.py` builds **`FeatureGatedMoE`**: the gate reads **100-D `feats` only**—**no DistilBERT, no tokenizer, no HRM** in the forward pass (true opt-out of transformers for routing). Pass `--text-gate` if you want the old **frozen DistilBERT** gate even when all experts are dense.

To train **DistilBERT as an expert** again, use `MOE_MATRIX_PROFILE=with_distilbert` or `--experts_json experts_all_data_*_with_distilbert.json` (then the gate is DistilBERT-based unless you only have dense experts and pass `--text-gate`—mixed text+dense always uses `HeterogeneousMoE`).

**Note:** `HeterogeneousMoE` calls `ex(..., return_type="logits")` for `text` experts; only LLM-style modules support that today.

Smoke examples assume checkpoints under `checkpoints/{2,3}-labels/all-data/` and dual parquets `processed/{stem}.parquet` + `transformed/{stem}.parquet` with aligned rows.

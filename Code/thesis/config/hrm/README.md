# HRM (E-HRM1) configs

- **Single encoder JSON:** `E_HRM1_4Level.json` in this directory (no `2_labels` / `3_labels` copies). **MLM pretrain** (`--phase pretrain`, `epochs_pretrain>0`) builds the **bare** `HierarchicalReasoningModel` and writes **`checkpoints/pretrain/{stem}/E_HRM1_4Level.safetensors`** (no K-way head). **Finetune** uses **`--n_classes 2`** or **`3`** with `HRMClassifierWrapper` and saves under **`checkpoints/K-labels/...`**. Load MLM weights into finetune with **`--hrm_encoder_ckpt`**.
- **Tokenizer:** `google-bert/bert-base-uncased` (vocab 30522).
- **Sequence length:** `seq_len` 512 (MLM tokenization `max_length` matches via `train_single.py`).
- **Inner stack:** `EncoderLM` for both H and L levels; `model_kwargs.num_layers` is per stack (4 → eight transformer blocks per full H/L cycle schedule).
- **Size:** ~105M parameters with `hidden_size` 800, `num_heads` 16. After changing JSON, run `python Code/thesis/tools/count_model_params.py`.

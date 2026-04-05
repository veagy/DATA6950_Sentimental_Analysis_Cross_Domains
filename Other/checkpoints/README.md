# Model checkpoints

Trained weights and related artifacts for this project live under this directory. The same layout is published on Google Drive for download or sharing when the repo does not include full binary assets.

## Google Drive

**All checkpoints can be accessed here:**

https://drive.google.com/drive/folders/1XptLnqd2ycvWgQM_u37c7Leop10xkpKk?usp=sharing

Download the folders you need and place them alongside this README so local paths match the training and evaluation code.

## Top-level folders (mirror Drive)

| Folder | Description |
|--------|-------------|
| `2-labels` | Checkpoints / runs for binary (2-class) sentiment setups |
| `3-labels` | Checkpoints / runs for three-class sentiment setups |
| `b11_cnn_lstm_stack_gelu_ddp` | CNN–LSTM stack with GELU, DDP training |
| `deep_learning` | Deep-learning experiments (e.g. encoder configs under `llm/`) |
| `hrm` | HRM-related tokenizer and assets |
| `mlp_geLU_head_ddp` | MLP + GELU head models trained with DDP |
| `moe` | Mixture-of-experts checkpoints |
| `pretrain` | Pretraining checkpoints |
| `transformer` | Transformer / sentence-transformer assets (e.g. MiniLM) |

If a subfolder is missing locally, fetch it from the Drive link above.

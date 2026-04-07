# Project and models (synthesized)

Sources: [`docs/README.md`](../../docs/README.md), [`docs/models_overview.md`](../../docs/models_overview.md), [`docs/Model_Parameters_and_Stacking.md`](../../docs/Model_Parameters_and_Stacking.md), [`docs/thesis_config_inventory.md`](../../docs/thesis_config_inventory.md), [`docs/thesis_parameter_counts.md`](../../docs/thesis_parameter_counts.md), [`docs/task.txt`](../../docs/task.txt).

## Capstone narrative (README)

- **Author / program:** Rohan Pratap Reddy Ravula; MS Data Science, Wentworth; DATA-6900 Capstone.
- **Stated scope:** Large sentiment stack: traditional ML, CNN/RNN, transformers, HRM, ensembles, MoE—README describes **59** model implementations, two-stage HRM (unsupervised pretrain + supervised fine-tune), PowerShell automation (`SETUP.ps1`, `TRAIN_SETUP.ps1`, `EVAL_SETUP.ps1`), backup with checksums, pytest, logging split by phase.
- **Path caveat:** README’s tree centers on `D:\CAPSTONE-I\Mixed_Models\mixed_models\` and `src/`; the active thesis layout referenced elsewhere in this repo uses `Code/thesis/`, `data/processed`, `data/transformed`, and `checkpoints/` as in [`task.txt`](../../docs/task.txt). Treat README as high-level product description; align operations with `task.txt` and `Code/thesis/`.

## Model families ([`models_overview.md`](../../docs/models_overview.md))

- **Baselines B1–B13:** Traditional ML (B1–B2), CNN (B9, B11–B13 hybrids), RNN (B10 LSTM, B8 GRU+attention, B7 BiLSTM+attention).
- **Transformers:** B3 DistilBERT, B4 BERT, B5 RoBERTa, B6 BART.
- **HRM:** E-HRM1 four-level stack (~105M params in inventory).
- **Experts:** E-ML1/E-ML2, E-DL1–E-DL4, E-HRM1.
- **MoE / ensembles:** Dense vs sparse routing, load balancing; README also lists simple/weighted/stacked ensembles.

## Data modality and parameters ([`Model_Parameters_and_Stacking.md`](../../docs/Model_Parameters_and_Stacking.md))

- **Transformed tabular (100D UMAP):** Classical ML, CNN/RNN-style experts used with dense features—E-ML1, E-ML2, B9–B13 family, E-DL4 as BiLSTM+attention expert.
- **Processed text:** Transformer and HRM experts; B6 as seq2seq-class style in table.
- **Stacking groups (documented):** Fast statistical; transformer envoys; cognitive (HRM); hybrid diversity.
- **MoE:** DistilBERT as feature extractor for gating; nine frozen experts in pool; sparse top-k routing called out.

## Config inventory decisions ([`thesis_config_inventory.md`](../../docs/thesis_config_inventory.md))

- JSON under `Code/thesis/config/` mapped to classes (`LLMModule`, `HierarchicalReasoningModel`, `CNNetworks`, `LSTMModule`/`GRUModule`/`RNNClassifier`, sklearn wrappers, `FeatureEncoderClassifier`, pretrain autoencoders, etc.).
- **B12/B13:** Documented in overview but no separate thesis JSON; B11_CNN_LSTM_v1 uses `CNNetworks` same as B9 until true hybrid exists.
- **B7/B8 “+ Attention” in docs vs code:** Implementation is plain BiLSTM/GRU in configs unless extended later.
- **HRM:** Single shared encoder JSON; `train_single.py` uses `--n_classes` 2 or 3 for head; ~105M trainable params with `n_classes=2` per parameter table.
- **Training entrypoints:** `train_single.py`, `train_all.py --sweep-all`, `train_moe.py`, `train_stack.py`; HRM/transformer pretrain options documented there.

## Parameter count table ([`thesis_parameter_counts.md`](../../docs/thesis_parameter_counts.md))

- Populated rows for feature encoders, HRM, RNNs, transformers (mostly frozen backbone + tiny head trainable counts for LLMs), pretrain autoencoders.
- **Build errors** recorded for some CNN/B11 rows, sklearn ML JSONs (`Unknown model class`), etc.—treat as live diagnostics, not final truth for those lines.

## Project rules ([`task.txt`](../../docs/task.txt))

- **Locations:** Models in `Code/models`; `data/processed` for cleaned text; `data/transformed` for SentenceTransformer + UMAP 100D features; checkpoints under `checkpoints/`.
- **Transformers/HRM:** Use processed data; embedding → dense layer → sigmoid/softmax by label count for training and inference (stated rule; implementation details in code and fine-tune summaries).
- **Other models:** Use transformed data for classification.
- **Checkpoints:** Prefer `.safetensors`; separate `2-labels` vs `3-labels` trees; per-dataset subfolders.
- **Logs:** Under `logs/` only; use `tqdm` in training.
- **Roadmap:** Config generation, training/test code, HRM unsupervised pretrain on combined text, per-dataset queues, resume metadata, DUMMY mock runs, MoE stacking, Q-LoRA-style fine-tune (later), detachment via tmux/screen/nohup.

## Cross-links

- Training pipeline detail: [TRAINING_ML_AND_RUNBOOKS.md](TRAINING_ML_AND_RUNBOOKS.md).
- Data-stage architecture: [ARCHITECTURE_PIPELINE.md](ARCHITECTURE_PIPELINE.md).

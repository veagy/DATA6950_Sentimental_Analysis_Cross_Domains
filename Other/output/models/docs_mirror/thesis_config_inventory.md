# Thesis config inventory (config ↔ model ↔ documentation)

This table maps each JSON under `Code/thesis/config/` to the implementation class, literature/doc ID, data modality per [`docs/task.txt`](task.txt), and known gaps versus [`docs/models_overview.md`](models_overview.md) / [`docs/Model_Parameters_and_Stacking.md`](docs/Model_Parameters_and_Stacking.md).

**Parameter counts** are generated separately: run `python Code/thesis/tools/count_model_params.py --markdown-out docs/thesis_parameter_counts.md` from the repo root (see [`docs/thesis_parameter_counts.md`](thesis_parameter_counts.md)).

## Summary

| Config path pattern | Class | Doc ID | Data (`task.txt`) |
|---------------------|-------|--------|-------------------|
| `transformers/*/B3_*.json` | `LLMModule` | E-DL1 DistilBERT | `data/processed` |
| `transformers/*/B4_*.json` | `LLMModule` | E-DL3 BERT | processed |
| `transformers/*/B5_*.json` | `LLMModule` | E-DL2 RoBERTa | processed |
| `transformers/*/B6_*.json` | `LLMModule` | B6 BART | processed |
| `hrm/E_HRM1_4Level.json` | `HierarchicalReasoningModel` | E-HRM1 | processed |
| `cnn/*/B9_*.json` | `CNNetworks` | B9 | `data/transformed` |
| `rnn/*/B7_*.json` | `LSTMModule` + `RNNClassifier` | B7 BiLSTM | transformed |
| `rnn/*/B8_*.json` | `GRUModule` + `RNNClassifier` | B8 GRU | transformed |
| `rnn/*/B10_*.json` | `LSTMModule` + `RNNClassifier` | B10 LSTM | transformed |
| `rnn/*/B11_*.json` | `CNNetworks` | B11 (filename only) | transformed |
| `ml/*/E_ML1_*.json` | `LogisticRegression` | E-ML1 | transformed |
| `ml/*/E_ML2_*.json` | `LinearSVC` | E-ML2 | transformed |
| `moe/example_experts.json` | (JSON list, not `model_factory`) | MoE routing example | mixed |

## Decisions on doc vs code gaps

1. **B12, B13 (CNN–LSTM hybrids)**  
   Documented in `models_overview.md` but **there are no separate thesis JSON files** for B12/B13. Only **B11_CNN_LSTM_v1** exists, and it currently instantiates **`CNNetworks` with the same architecture as B9** (no LSTM in the graph).  
   **Decision:** Treat **B11** as the single hybrid *placeholder* config until a true CNN→LSTM module exists in `Code/models`. Add B12/B13 JSONs only when matching implementations land.

2. **B7 / B8 “+ Attention” in docs**  
   Docs describe attention-augmented RNNs; thesis configs use **plain** `LSTMModule` / `GRUModule` via `RNNClassifier` without a separate attention block in JSON.  
   **Decision:** Document as **implementation = plain RNN**; align doc copy or add attention layers in a future model variant.

3. **HRM parameter count vs docs**  
   E-HRM1 is a **single** shared encoder JSON (`hrm/E_HRM1_4Level.json`); `train_single.py` takes **`--n_classes 2`** or **`3`** for the `Linear(100, K)` head. `seq_len` 512, `hidden_size` 800, dual `EncoderLM` stacks; `count_model_params.py` reports **~105M** trainable params with `n_classes=2` (**+101** if `3`). See [`docs/thesis_parameter_counts.md`](thesis_parameter_counts.md).

4. **Training policy**  
   Per `docs/task.txt`, **one expert model per run** uses `Code/thesis/train/train_single.py`. **Bulk sweeps** require `--sweep-all` on `Code/thesis/train/train_all.py`. **MoE** uses `Code/thesis/train/train_moe.py`; **stacking** uses `Code/thesis/train/train_stack.py`. Example one-shot invocation: **Core training** → **Single process** in [`scripts/README.md`](../scripts/README.md).  
   For **HRM (and transformer) pretrain only**, `train_single.py` supports `--pretrain_text_source {dataset,all_processed}`: `dataset` uses `data/processed/{--dataset_stem}.parquet`; `all_processed` merges every `data/processed/*.parquet`. **Finetune** always uses `--dataset_stem`’s parquet.

# Model Parameters & Stacking Blueprint

This document specifies the internal parameter sizes for all 59 generated configurations and explicitly scopes how individual experts should be aggregated to form Neural Ensembles, Meta-Stacked Learners, and Mixture of Expert (MoE) architectures in this thesis.

---

## 1. Classical ML & Recurrent/Convolutional Models
**Data Source:** `data/transformed` (Strictly 100D dense UMAP features)

| Model Name | Expert ID | Architecture Type | Approximate Params |
|------------|-----------|-------------------|--------------------|
| `logistic_regression` | **E-ML1** | Traditional ML | N/A (Statistical) |
| `linear_svm` | **E-ML2** | Traditional ML | N/A (Statistical) |
| `cnn_text_classifier` | B9 | 1D Convolutions | ~180,000 |
| `lstm_standalone` | B10 | LSTM | ~420,000 |
| `gru_attention` | B8 | GRU + Attention | ~550,000 |
| `cnn_lstm_hybrid_v1-v3` | B11-B13 | CNN feature maps into LSTM | ~550K - 800K |
| `bilstm_attention` | **E-DL4** | Bidirectional LSTM + Attn | ~750,000 |

---

## 2. Massive Pre-Trained Transformers
**Data Source:** `data/processed` (Raw String Vectors)

| Model Name | Expert ID | Architecture Type | Approximate Params |
|------------|-----------|-------------------|--------------------|
| `distilbert` | **E-DL1** | Transformer Encoder | ~66,000,000 |
| `bert_base` | **E-DL3** | Transformer Encoder | ~110,000,000 |
| `roberta_base` | **E-DL2** | Transformer Encoder | ~125,000,000 |
| `bart` | B6 | Transformer Seq2Seq | ~139,000,000 |

---

## 3. Hierarchical Reasoning Models (HRMs)
**Data Source:** `data/processed` (Raw String Vectors)

| Model Name | Expert ID | Cognitive Reasoning Pathway | Approximate Params |
| `e_hrm1` | **E-HRM1** | 4-Level (Lex → Syn → Sem → Pragmatic), seq_len 512 | ~105,000,000 |

---

## 4. Neural Stacking & MoE Topologies
Certain models generate the final classification by actively clustering the outputs of "frozen" Expert sub-models. 

### Stacking Architectures (`stack1` through `stack7`)
A neural stack freezes base models and passes their logits/probabilities into a "Meta-Classifier" (like Logistic Regression or an MLP) to learn which model handles which edge-cases best.

**Required Stacking Combinations:**
- **Stack Group 1 (Fast Statistical):** Combine `[E-ML1, E-ML2, cnn_text_classifier]`.
- **Stack Group 2 (Transformer Envoys):** Combine `[E-DL1, E-DL2, E-DL3]`.
- **Stack Group 3 (Cognitive Reasoning):** Combine `[E-HRM1]`.
- **Stack Group 4 (Hybrid Diversity):** Combine `[E-DL1, E-DL4, E-HRM1, E-ML2]`.

### Mixture of Experts (`moe1` through `moe5`)
MoE models ditch the static Meta-Classifier. Instead, a "Gating Network" looks at the raw text, calculates which models are best suited for the specific string dynamically, and mathematically routes the query exclusively to those models.

**Architecture Configuration for MoE:**
1. **Feature Extractor:** Use **`E-DL1` (DistilBERT)**. Its small 66M layout makes it exceptionally fast at embedding language context directly for the Gating Mechanism without slowing down the pipeline.
2. **The 9 Frozen Experts (The Router Pool):**
   - *Statistical Experts:* `E-ML1`, `E-ML2`
   - *Deep Learning Experts:* `E-DL1`, `E-DL2`, `E-DL3`, `E-DL4`
   - *Reasoning Experts:* `E-HRM1`
3. **Execution Masking (Sparsity):**
   - By enforcing `sparse_top_k=2` in the configs, the Gating Network will block 7 of the experts for every string, activating and computing only against the smartest 2 sub-models per inference, preserving immense VRAM on large datasets.

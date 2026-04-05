# Project Models Overview

This document synthesizes the 59 sentiment analysis models defined across the source code into their respective conceptual categories.

## 1. Baselines & Literature Review Models (B1 - B13)

_These models represent standard benchmarks and established architectures thoroughly studied in sentiment analysis literature._

### Traditional Machine Learning

- **B1 (Transformer + Logistic Regression)**
- **B2 (transformer + Linear SVM)**:

### Convolutional Neural Networks (CNN)

- **B9 (CNN Text Classifier)**: ~180K parameters. Uses parallel 1D convolutions with varying filter sizes (e.g., 3, 4, 5) to capture local n-gram temporal patterns simultaneously.
- **B11, B12, B13 (CNN-LSTM Hybrids)**: ~550K-800K parameters. Advanced variants that feed parallel convolution outputs into recurrent layers to model both local structure and global sequence dependencies.

### Recurrent Neural Networks (RNN)

- **B10 (Standalone LSTM)**: ~420K parameters. A standard Long Short-Term Memory network without attention mechanics.
- **B8 (GRU + Attention)**: ~550K parameters. A Gated Recurrent Unit model with a scaled dot-product attention mechanism overlay.
- **B7 (BiLSTM + Attention)**: ~750K parameters. Captures past and future context synchronously, paired with customized Multi-Head Attention.

---

## 2. Transformer Models

_Pre-trained deep bidirectional transformers used for transfer learning._

- **B3 / E-DL1 (DistilBERT)**: ~66M parameters. A smaller, faster, cheaper version of BERT. Often used as the feature extractor for MoE gating logic.
- **B4 / E-DL3 (BERT)**: ~110M parameters. The industry standard `bert-base-uncased`.
- **B5 / E-DL2 (RoBERTa)**: ~125M parameters. The `roberta-base` model, pre-trained with dynamic masking on a larger corpus.
- **B6 (BART)**: ~139M parameters. Bidirectional and Auto-Regressive Transformer.

---

## 3. Hierarchical Reasoning Models (HRM)

_Custom, highly interpretable architectures built specifically for this thesis that mimic cognitive processing steps from words to pragmatics._

- **E-HRM1 (4-Level)**: ~105M parameters (`hidden_size` 800, `seq_len` 512, 16 heads, dual `EncoderLM` stacks). Full implementation utilizing a BiLSTM for Lexical and Transformers for Syntactic, Semantic, and Pragmatic stages. Generates unique "reasoning chains" explaining inferences at each level.

---

## 4. Expert Models

_Strongly tuned individual models marked with an "E-" prefix. These are purposefully frozen and clustered into larger ensemble meta-architectures._

- **ML Experts**: `E-ML1` (LogReg) and `E-ML2` (SVM).
- **DL Experts**: `E-DL1` (DistilBERT), `E-DL2` (RoBERTa), `E-DL3` (BERT), and `E-DL4` (BiLSTM+Attention).
   - *Reasoning Experts:* `E-HRM1`

---

## 5. Mixture of Experts (MoE) Models

_The thesis implements a true neural routing architecture ranging from `MOE1` to `MOE5`._

- **Mechanics**: Instead of rigidly averaging outputs across everyone, these models inject text into a "Feature Extractor" (e.g., DistilBERT) connected to a learned dense **Gating Network**.
- **Dense vs Sparse**: Supports Dense gating (Softmax routing to _all_ experts simultaneously) and Sparse gating (Top-K routing like `sparse_top_k=2`, where only the best 2 experts are queried per sample).
- **Load Balancing**: Includes specific loss penalizations to ensure the routing uniformally queries various experts instead of heavily favoring one.

_(Averaging and voting ensembles (`ENS1-ENS3`) as well as neural meta-learner stacking (`STACK1-STACK7`) approaches are also fully populated.)_

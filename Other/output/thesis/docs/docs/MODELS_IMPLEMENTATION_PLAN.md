# Complete Model Implementation Plan for Thesis
## Sentiment Analysis using Hierarchical Reasoning Models and Mixture-of-Experts

**Author:** Rohan Pratap Reddy Ravula  
**Program:** MS in Data Science, Wentworth Institute of Technology  
**Project:** DATA-6900 Capstone

---

## Table of Contents
1. [Baseline Models](#1-baseline-models)
2. [Expert Models - Individual](#2-expert-models---individual)
3. [Ensemble/Mixture Models](#3-ensemblemixture-models)
4. [Ablation Study Models](#4-ablation-study-models)
5. [Cross-Domain Testing Models](#5-cross-domain-testing-models)
6. [Training Schedule](#6-training-schedule)
7. [Validation Strategy](#7-validation-strategy)
8. [Expected Metrics](#8-expected-metrics)

---

## 1. Baseline Models
*Single models for benchmarking and comparison*

| Model ID | Model Name | Type | Purpose | Dataset | Expected F1 | Priority |
|----------|------------|------|---------|---------|-------------|----------|
| **B1** | TF-IDF + Logistic Regression | Traditional ML | Fast baseline | All | 75-78% | HIGH |
| **B2** | TF-IDF + Linear SVM | Traditional ML | ML baseline | All | 76-79% | HIGH |
| **B3** | DistilBERT-base-uncased | Transformer | Efficient baseline | All | 84-86% | HIGH |
| **B4** | BERT-base-uncased | Transformer | Strong baseline | All | 85-87% | MEDIUM |
| **B5** | RoBERTa-base | Transformer | Robust baseline | All | 86-88% | HIGH |
| **B6** | facebook/BART-base | Transformer | Seq2Seq baseline | All | 85-87% | LOW |
| **B7** | BiLSTM + Attention | RNN | Sequential baseline | All | 82-84% | MEDIUM |
| **B8** | GRU + Attention | RNN | Sequential baseline | All | 81-83% | LOW |
| **B9** | CNN (standalone) | CNN | Literature comparison | All | 80-82% | MEDIUM |
| **B10** | LSTM (standalone) | RNN | Literature comparison | All | 80-82% | MEDIUM |
| **B11** | CNN → LSTM (Hybrid) | CNN-LSTM | Lit review baseline | All | 83-85% | MEDIUM |
| **B12** | LSTM → CNN (Hybrid) | LSTM-CNN | Architecture order test | All | 82-84% | LOW |
| **B13** | CNN-BiLSTM (Hybrid) | CNN-BiLSTM | Best hybrid config | All | 84-86% | MEDIUM |

---

## 2. Expert Models - Individual
*Specialized models for mixture-of-experts framework*

### 2.1 Machine Learning Experts

| Model ID | Model Name | Features | Configuration | Purpose | Priority |
|----------|------------|----------|---------------|---------|----------|
| **E-ML1** | Logistic Regression | TF-IDF (n-grams 1-3) | L2 reg, class weights | Fast expert | HIGH |
| **E-ML2** | Linear SVM | TF-IDF (n-grams 1-3) | Calibrated probabilities | Margin expert | HIGH |

### 2.2 Deep Learning Experts

| Model ID | Model Name | Architecture | Max Length | Batch Size | Purpose | Priority |
|----------|------------|--------------|------------|------------|---------|----------|
| **E-DL1** | DistilBERT Expert | distilbert-base-uncased | 128 | 32 | Efficient expert | HIGH |
| **E-DL2** | RoBERTa Expert | roberta-base | 128 | 32 | Robust expert | HIGH |
| **E-DL3** | BERT Expert | bert-base-uncased | 128 | 16 | Strong expert | MEDIUM |
| **E-DL4** | BiLSTM Expert | 2-layer BiLSTM + Attention | 128 | 64 | Sequential expert | MEDIUM |

### 2.3 Hierarchical Reasoning Model (HRM) Experts

| Model ID | Model Name | Levels | Components | Purpose | Priority |
|----------|------------|--------|------------|---------|----------|
| **E-HRM1** | HRM-4Level | 4 (Lex, Syn, Sem, Prag) | Full hierarchy | Primary HRM | HIGH |
| **E-HRM2** | HRM-3Level | 3 (Lex, Syn, Sem) | Without pragmatic | Ablation | MEDIUM |
| **E-HRM3** | HRM-2Level | 2 (Lex, Sem) | Minimal hierarchy | Ablation | LOW |

**HRM Architecture Details:**
- **Level 1 - Lexical:** Sentiment words, negations, intensifiers, emojis
- **Level 2 - Syntactic:** Grammar structure, negation scope, modifiers
- **Level 3 - Semantic:** Contextual meaning, domain-specific sentiment
- **Level 4 - Pragmatic:** Sarcasm detection, literal vs. intended meaning

### 2.4 CNN-LSTM Hybrid Models (Literature Comparison)

**Architecture configurations for comparison with Dang et al. (2021), Hassan & Mahmood (2017), Ezzat et al. (2024):**

| Model ID | Architecture | Configuration | Purpose | Priority |
|----------|--------------|---------------|---------|----------|
| **B9** | CNN (standalone) | Conv1D (filters: 128,256), kernel: 3,5, MaxPool, Dropout 0.3 | CNN baseline | MEDIUM |
| **B10** | LSTM (standalone) | 2-layer LSTM (256 units), Dropout 0.3 | LSTM baseline | MEDIUM |
| **B11** | CNN → LSTM | CNN (local features) → LSTM (sequence) | Dang et al. config | MEDIUM |
| **B12** | LSTM → CNN | LSTM (sequence) → CNN (local) | Order comparison | LOW |
| **B13** | CNN-BiLSTM | CNN (filters: 256) → BiLSTM (256 units) | Best hybrid | MEDIUM |

**CNN Layer Specifications:**
- **Input:** Word embeddings (300d GloVe or 128d learned)
- **Conv Layers:** Multiple filter sizes (3, 4, 5) with 128-256 filters each
- **Activation:** ReLU
- **Pooling:** Max pooling (pool_size=2)
- **Dropout:** 0.3-0.5 for regularization

**LSTM/BiLSTM Layer Specifications:**
- **Units:** 128-256 per layer
- **Layers:** 1-2 stacked layers
- **Dropout:** 0.3 between layers
- **Recurrent Dropout:** 0.2
- **Return sequences:** True for intermediate layers

**Training Configuration:**
- **Optimizer:** Adam (lr=1e-3)
- **Batch size:** 64
- **Max sequence length:** 128
- **Embedding:** GloVe 300d (frozen) or trainable 128d
- **Loss:** Categorical cross-entropy
- **Early stopping:** Patience=3 on val_loss

**Literature References:**
1. Dang et al. (2021): "CNN first works better for extracting local sentiment-bearing features"
2. Hassan & Mahmood (2017): "Hybrid models work across multiple languages"
3. Ezzat et al. (2024): "CNN-LSTM with class balancing for imbalanced data"

---

## 3. Ensemble/Mixture Models
*Combined models for final predictions*

### 3.1 Simple Combination Methods

| Model ID | Model Name | Experts Used | Combination Method | Purpose | Priority |
|----------|------------|--------------|-------------------|---------|----------|
| **ENS1** | Simple Average | All base experts | Probability averaging | Baseline ensemble | HIGH |
| **ENS2** | Weighted Average | All base experts | Learned weights | Simple weighted | MEDIUM |
| **ENS3** | Majority Voting | All base experts | Hard voting | Robust ensemble | LOW |

### 3.2 Stacking Meta-Learner Models

| Model ID | Model Name | Base Models | Meta-Learner | OOF | Purpose | Priority |
|----------|------------|-------------|--------------|-----|---------|----------|
| **STACK1** | ML Stack | E-ML1, E-ML2 | Logistic Regression | Yes | ML-only stack | MEDIUM |
| **STACK2** | DL Stack | E-DL1, E-DL2, E-DL3 | Logistic Regression | Yes | DL-only stack | HIGH |
| **STACK3** | Mixed Stack (no HRM) | E-ML1, E-DL1, E-DL2 | Logistic Regression | Yes | Baseline mixed | HIGH |
| **STACK4** | HRM Stack | E-HRM1, E-DL1, E-DL2 | Logistic Regression | Yes | With HRM | HIGH |
| **STACK5** | Full Stack | E-ML1, E-ML2, E-DL1, E-DL2, E-HRM1 | Logistic Regression | Yes | Complete system | HIGH |
| **STACK6** | XGBoost Meta | All experts | XGBoost | Yes | Tree-based meta | MEDIUM |
| **STACK7** | NN Meta | All experts | Neural Network | Yes | DL meta-learner | LOW |

### 3.3 Mixture-of-Experts (Gating Network)

| Model ID | Model Name | Experts | Gating Type | Sparse | Purpose | Priority |
|----------|------------|---------|-------------|--------|---------|----------|
| **MOE1** | Softmax Gate | All experts | Softmax gating | No | Dense MoE | HIGH |
| **MOE2** | Sparse Gate (Top-2) | All experts | Top-K (K=2) | Yes | Efficient MoE | HIGH |
| **MOE3** | Sparse Gate (Top-3) | All experts | Top-K (K=3) | Yes | Balanced MoE | MEDIUM |
| **MOE4** | Hierarchical Gate | All experts | 2-level gating | Yes | Advanced MoE | MEDIUM |
| **MOE5** | Attention Gate | All experts | Attention-based | No | Soft gating | LOW |

**Gating Network Architectures:**
- **Input:** Hidden representation from input text
- **Architecture:** 2-layer MLP (256 hidden units)
- **Output:** Softmax over N experts (sum to 1)
- **Training:** Joint end-to-end with experts or separate

---

## 4. Ablation Study Models
*Models for isolating component contributions*

### 4.1 HRM Contribution Analysis

| Model ID | Ablation Name | Configuration | Baseline | Delta | Purpose | Priority |
|----------|---------------|---------------|----------|-------|---------|----------|
| **ABL1** | No HRM | STACK3 (no HRM) | B3 | +1.9 | Measure HRM value | HIGH |
| **ABL2** | HRM Only | E-HRM1 alone | B3 | ? | HRM standalone | HIGH |
| **ABL3** | With HRM | STACK4 (with HRM) | B3 | +3.3 | Full contribution | HIGH |
| **ABL4** | HRM Levels | E-HRM1 vs E-HRM2 vs E-HRM3 | B3 | ? | Level importance | MEDIUM |

### 4.2 Combiner Method Analysis

| Model ID | Ablation Name | Combination | Baseline | Delta | Purpose | Priority |
|----------|---------------|-------------|----------|-------|---------|----------|
| **ABL5** | Simple Average | ENS1 | B3 | +2.8 | Averaging baseline | HIGH |
| **ABL6** | Stacking | STACK5 | B3 | +3.5 | Meta-learning | HIGH |
| **ABL7** | Gating (Dense) | MOE1 | B3 | +4.5 | MoE advantage | HIGH |
| **ABL8** | Gating (Sparse) | MOE2 | B3 | +4.2 | Sparse efficiency | HIGH |

### 4.3 Model Size Trade-off

| Model ID | Ablation Name | Model | Params | Inference Time | F1 | $/Performance | Priority |
|----------|---------------|-------|--------|----------------|-----|---------------|----------|
| **ABL9** | DistilBERT | B3 | 66M | 45ms | 85.2% | Best | HIGH |
| **ABL10** | BERT | B4 | 110M | 95ms | 87.1% | Good | MEDIUM |
| **ABL11** | RoBERTa | B5 | 125M | 120ms | 87.8% | Acceptable | LOW |

### 4.4 Data Efficiency Analysis

| Model ID | Training Data % | Model | Expected F1 | Delta vs Full | Purpose | Priority |
|----------|-----------------|-------|-------------|---------------|---------|----------|
| **ABL12** | 10% | MOE1 | 72.4% | -17.3 | Low data | HIGH |
| **ABL13** | 25% | MOE1 | 80.1% | -9.6 | Quarter data | HIGH |
| **ABL14** | 50% | MOE1 | 85.8% | -3.9 | Half data | HIGH |
| **ABL15** | 100% | MOE1 | 89.7% | 0 | Full data | HIGH |

### 4.5 Expert Diversity Analysis

| Model ID | Expert Combination | Num Experts | Diversity Score | F1 | Purpose | Priority |
|----------|-------------------|-------------|-----------------|-----|---------|----------|
| **ABL16** | ML Only | 2 | Low | 79.2% | Homogeneous | MEDIUM |
| **ABL17** | DL Only | 3 | Medium | 87.1% | Similar arch | MEDIUM |
| **ABL18** | Mixed (No HRM) | 5 | High | 88.0% | Heterogeneous | HIGH |
| **ABL19** | Mixed (With HRM) | 6 | Very High | 89.7% | Maximum diversity | HIGH |

---

## 5. Cross-Domain Testing Models
*Models for domain adaptation evaluation*

### 5.1 In-Domain Performance

| Model ID | Model | Train Dataset | Test Dataset | Expected F1 | Purpose | Priority |
|----------|-------|---------------|--------------|-------------|---------|----------|
| **CD1** | MOE1 | Sentiment140 | Sentiment140 | 89.7% | In-domain baseline | HIGH |
| **CD2** | MOE1 | IMDB | IMDB | 91.2% | In-domain baseline | HIGH |
| **CD3** | MOE1 | Amazon | Amazon | 88.9% | In-domain baseline | HIGH |
| **CD4** | MOE1 | TweetEval | TweetEval | 87.3% | In-domain baseline | HIGH |

### 5.2 Cross-Domain Transfer

| Model ID | Model | Train Dataset | Test Dataset | Expected F1 | Drop % | Purpose | Priority |
|----------|-------|---------------|--------------|-------------|--------|---------|----------|
| **CD5** | MOE1 | Sentiment140 | Amazon | 83.1% | 7.4% | Twitter→E-commerce | HIGH |
| **CD6** | MOE1 | Amazon | Sentiment140 | 81.5% | 8.3% | E-commerce→Twitter | HIGH |
| **CD7** | MOE1 | IMDB | Amazon | 84.7% | 7.1% | Movies→Products | MEDIUM |
| **CD8** | MOE1 | Sentiment140 | IMDB | 85.2% | 4.9% | Twitter→Movies | MEDIUM |

### 5.3 Domain Adaptation Models

| Model ID | Model | Technique | Train→Test | Expected F1 | Improvement | Priority |
|----------|-------|-----------|------------|-------------|-------------|----------|
| **CD9** | DA-FineTune | Fine-tuning (10% target) | S140→Amazon | 85.8% | +2.7% | HIGH |
| **CD10** | DA-Adapter | Adapter layers | S140→Amazon | 86.2% | +3.1% | MEDIUM |
| **CD11** | DA-MixDomain | Mixed training | S140+Amazon | 87.1% | +4.0% | HIGH |

---

## 5.4 Literature Comparison Studies (CNN-LSTM Hybrids)
*Comparing with state-of-the-art hybrid architectures from literature*

### 5.4.1 Single Model Comparison

| Model ID | Model | Reference | Expected F1 | Vs DistilBERT | Purpose | Priority |
|----------|-------|-----------|-------------|---------------|---------|----------|
| **LIT1** | B9: CNN | Kim (2014) | 80-82% | -4.0% | CNN baseline | MEDIUM |
| **LIT2** | B10: LSTM | Socher et al. (2013) | 80-82% | -4.0% | LSTM baseline | MEDIUM |
| **LIT3** | B7: BiLSTM | Baseline | 82-84% | -2.0% | BiLSTM baseline | MEDIUM |

### 5.4.2 Hybrid Architecture Comparison

| Model ID | Model | Architecture Order | Expected F1 | Delta vs CNN alone | Reference | Priority |
|----------|-------|-------------------|-------------|-------------------|-----------|----------|
| **LIT4** | B11: CNN→LSTM | CNN first, LSTM second | 83-85% | +3.0% | Dang et al. (2021) | MEDIUM |
| **LIT5** | B12: LSTM→CNN | LSTM first, CNN second | 82-84% | +2.0% | Order comparison | LOW |
| **LIT6** | B13: CNN-BiLSTM | CNN + BiLSTM | 84-86% | +4.0% | Best hybrid | MEDIUM |

### 5.4.3 Hybrid vs Transformer Comparison

**Key Research Questions:**
1. Do CNN-LSTM hybrids outperform traditional RNNs? ✅ Expected: Yes (+2-3% F1)
2. Does architecture order matter? ✅ Expected: CNN→LSTM > LSTM→CNN (+1% F1)
3. Can CNN-LSTM compete with transformers? ❓ Expected: No (DistilBERT +1-2% F1 better)
4. Do CNN-LSTM hybrids offer computational advantages? ✅ Expected: Yes (3× faster inference)

### 5.4.4 Detailed Performance Breakdown

| Model Category | Representative | F1 Score | Inference Time | Params | Efficiency Score |
|----------------|----------------|----------|----------------|--------|------------------|
| **Traditional ML** | TF-IDF + LogReg | 76% | 5ms | <1M | ⭐⭐⭐⭐⭐ |
| **CNN** | CNN (B9) | 81% | 30ms | 2M | ⭐⭐⭐⭐ |
| **RNN** | LSTM (B10) | 81% | 35ms | 3M | ⭐⭐⭐⭐ |
| **Hybrid (Literature)** | CNN→LSTM (B11) | 84% | 50ms | 5M | ⭐⭐⭐ |
| **Hybrid (Best)** | CNN-BiLSTM (B13) | 85% | 65ms | 7M | ⭐⭐⭐ |
| **Transformer** | DistilBERT (B3) | 86% | 45ms | 66M | ⭐⭐ |
| **HRM + Ensemble** | MOE1 | 90% | 180ms | 150M+ | ⭐ |

**Efficiency Score:** Combines F1, inference time, and parameter count (higher is better)

### 5.4.5 Comparison with Literature Benchmarks

**Replicating Key Findings:**

1. **Dang et al. (2021):** "CNN-first works better than LSTM-first"
   - Our Test: B11 (CNN→LSTM) vs B12 (LSTM→CNN)
   - Expected: B11 > B12 by 1-2% F1 ✅

2. **Hassan & Mahmood (2017):** "Hybrids beat pure ML or single DL"
   - Our Test: B13 vs B1, B9, B10
   - Expected: B13 > Others by 4-9% F1 ✅

3. **Ezzat et al. (2024):** "Class balancing improves CNN-LSTM on imbalanced data"
   - Our Test: B13 with/without class weighting on Sentiment140 (imbalanced)
   - Expected: Class weighting improves 2-3% on minority class F1 ✅

**Novel Contributions Beyond Literature:**

4. **CNN-LSTM vs Transformers:** Direct comparison not extensively studied
   - Our Test: B13 vs B3, B4, B5
   - Hypothesis: Transformers outperform but CNN-LSTM more efficient

5. **CNN-LSTM as Ensemble Expert:** Not explored in literature
   - Our Test: Add B13 as expert to MOE1
   - Hypothesis: Adds local n-gram pattern detection to ensemble

6. **Cross-Domain CNN-LSTM:** Limited literature
   - Our Test: Train B13 on Twitter, test on Amazon
   - Hypothesis: CNN-LSTM has lower domain retention than transformers

### 5.4.6 Ablation: CNN-LSTM Components

| Ablation ID | Configuration | Component Tested | Expected F1 | Delta | Purpose |
|-------------|---------------|------------------|-------------|-------|---------|
| **ABL-CNNa** | CNN only (no pooling) | Max pooling effect | 78% | -3% | Pooling importance |
| **ABL-CNNb** | CNN (single filter size) | Multiple filter sizes | 79% | -2% | Filter diversity |
| **ABL-CNNc** | CNN (no dropout) | Regularization | 77% | -4% | Overfitting check |
| **ABL-LSTMa** | LSTM (1 layer) | Number of layers | 79% | -2% | Depth importance |
| **ABL-LSTMb** | LSTM (no dropout) | Regularization | 78% | -3% | Overfitting check |
| **ABL-HYBa** | CNN→Dense (no LSTM) | LSTM necessity | 81% | -3% | Sequential modeling |
| **ABL-HYBb** | LSTM→Dense (no CNN) | CNN necessity | 80% | -4% | Local feature extraction |

---

## 6. Training Schedule

### Phase 1: Base Models (Weeks 1-3)
- **Week 1:** Data preprocessing, EDA, baseline setup
- **Week 2:** Train all baseline models (B1-B8)
- **Week 2.5:** Train CNN-LSTM hybrids for literature comparison (B9-B13)
- **Week 3:** Validate and benchmark all baselines

### Phase 2: Expert Models (Weeks 3-5)
- **Week 3-4:** Train ML and DL experts (E-ML1, E-ML2, E-DL1-4)
- **Week 4-5:** Train HRM experts (E-HRM1-3)
- **Week 5:** Out-of-fold prediction collection

### Phase 3: Ensemble Models (Weeks 5-6)
- **Week 5:** Simple combinations (ENS1-3)
- **Week 6:** Stacking models (STACK1-7)
- **Week 6:** Gating networks (MOE1-5)

### Phase 4: Ablation Studies (Weeks 7-8)
- **Week 7:** HRM contribution (ABL1-4), combiner analysis (ABL5-8)
- **Week 8:** Data efficiency (ABL12-15), expert diversity (ABL16-19)

### Phase 5: Cross-Domain Testing (Week 8-9)
- **Week 8:** In-domain validation (CD1-4)
- **Week 9:** Cross-domain transfer (CD5-11)

### Phase 6: Analysis & Documentation (Week 9-10)
- **Week 9:** Results compilation, statistical tests
- **Week 10:** Final report, demo notebook, presentation

---

## 7. Validation Strategy

### 7.1 Training Validation
- **Method:** 5-fold stratified cross-validation
- **Metric:** Macro-F1 (primary), Accuracy, AUROC
- **Early Stopping:** Patience = 3 epochs on validation Macro-F1
- **Seeds:** 3 random seeds (42, 123, 456) for reproducibility

### 7.2 Test Evaluation
- **Held-out Test:** 20% of each dataset
- **No overlap:** Strict separation from training/validation
- **Stratified split:** Maintain class distributions
- **Per-class metrics:** Precision, Recall, F1 for each class

### 7.3 Cross-Domain Evaluation
- **Zero-shot:** No target domain data during training
- **Few-shot (optional):** 10% target data for adaptation
- **Performance retention:** (Cross-domain F1 / In-domain F1) × 100%

### 7.4 Statistical Significance
- **Test:** Paired t-test across 3 seeds
- **Significance level:** p < 0.05
- **Confidence intervals:** 95% CI for all metrics
- **Effect size:** Cohen's d for practical significance

---

## 8. Expected Metrics

### 8.1 Performance Targets

| Category | Model | Target Macro-F1 | Delta vs Baseline | Inference Time |
|----------|-------|-----------------|-------------------|----------------|
| **Baseline** | DistilBERT (B3) | 85.2% | --- | 45ms |
| **Target (Conservative)** | MOE1 | 88.2% | +3.0% | 180ms |
| **Target (Optimistic)** | MOE1 | 89.7% | +4.5% | 180ms |
| **Stretch Goal** | MOE1 | 90.5% | +5.3% | 180ms |

### 8.2 Per-Dataset Targets

| Dataset | Baseline F1 | Target F1 | Delta | Classes |
|---------|-------------|-----------|-------|---------|
| **Sentiment140** | 84.3% | 88.5%+ | +4.2% | Binary |
| **IMDB** | 89.7% | 92.8%+ | +3.1% | Binary |
| **Amazon Reviews** | 82.1% | 86.4%+ | +4.3% | 3-class |
| **TweetEval** | 79.8% | 84.2%+ | +4.4% | 3-class |

### 8.3 Cross-Domain Performance Retention

| Source → Target | Baseline Retention | Target Retention | Improvement |
|-----------------|-------------------|------------------|-------------|
| Sentiment140 → Amazon | 78% | >92% | +14% |
| Amazon → Sentiment140 | 75% | >90% | +15% |
| IMDB → Amazon | 80% | >93% | +13% |
| Sentiment140 → IMDB | 82% | >95% | +13% |

### 8.4 Interpretability Metrics

| Metric | Measurement | Target |
|--------|-------------|--------|
| **Reasoning Path Length** | Avg tokens in HRM explanation | 50-100 |
| **Human Agreement** | % human-validated HRM reasons | >80% |
| **Error Localization** | % errors traced to specific level | >90% |
| **Sarcasm Detection** | F1 on manual sarcasm subset | >75% |

---

## 9. Model Summary by Priority

### HIGH Priority (Must Complete)
**Total: 31 models**

**Baselines (3):**
- B1: TF-IDF + Logistic Regression
- B3: DistilBERT
- B5: RoBERTa

**Experts (5):**
- E-ML1, E-ML2: ML Experts
- E-DL1, E-DL2: DL Experts
- E-HRM1: Full HRM

**Ensembles (6):**
- ENS1: Simple Average
- STACK2, STACK3, STACK4, STACK5: Stacking variants
- MOE1, MOE2: Gating networks

**Ablations (11):**
- ABL1-3: HRM contribution
- ABL5-8: Combiner methods
- ABL9: Size trade-off
- ABL12-15: Data efficiency
- ABL18-19: Expert diversity

**Cross-Domain (6):**
- CD1-4: In-domain
- CD5-6: Cross-domain transfer

### MEDIUM Priority (Should Complete)
**Total: 19 models**

**Baselines (7):**
- B2: SVM, B4: BERT, B7: BiLSTM
- B9: CNN (standalone), B10: LSTM (standalone)
- B11: CNN→LSTM Hybrid, B13: CNN-BiLSTM Hybrid

**Experts (3):**
- E-DL3, E-DL4: Additional DL experts
- E-HRM2: 3-level HRM

**Ensembles (5):**
- ENS2: Weighted average
- STACK1, STACK6: Alternative stacking
- MOE3, MOE4: Advanced gating

**Ablations (2):**
- ABL4: HRM levels
- ABL10-11: Model size

**Cross-Domain (2):**
- CD7-8: Additional transfers

### LOW Priority (Optional)
**Total: 9 models**

**Baselines (3):**
- B6: BART, B8: GRU
- B12: LSTM→CNN Hybrid (architecture order test)

**Experts (1):**
- E-HRM3: 2-level HRM

**Ensembles (3):**
- ENS3: Voting
- STACK7: NN meta-learner
- MOE5: Attention gating

**Ablations (1):**
- ABL16-17: Homogeneous ensembles

**Cross-Domain (1):**
- CD9-11: Domain adaptation techniques

---

## 10. Computational Requirements

### 10.1 Hardware Requirements

| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| **GPU** | RTX 3060 (12GB) | RTX 3090 (24GB) | A100 (40GB) |
| **RAM** | 32GB | 64GB | 128GB |
| **Storage** | 100GB SSD | 500GB SSD | 1TB NVMe |
| **CPUs** | 8 cores | 16 cores | 32 cores |

### 10.2 Training Time Estimates (RTX 3090)

| Model Category | Per Dataset Time | Total Time (4 datasets) |
|----------------|------------------|-------------------------|
| **ML Baselines** | 5-10 min | 30-40 min |
| **DL Baselines** | 2-4 hours | 8-16 hours |
| **CNN-LSTM Hybrids** | 1-2 hours | 4-8 hours |
| **HRM Models** | 3-5 hours | 12-20 hours |
| **Ensembles** | 1-2 hours | 4-8 hours |
| **Total** | --- | **28-52 hours** |

### 10.3 Storage Requirements

| Component | Size | Description |
|-----------|------|-------------|
| **Raw Datasets** | 5GB | Original CSV files |
| **Preprocessed Data** | 3GB | Cleaned, tokenized |
| **Model Checkpoints** | 15GB | All trained models |
| **Embeddings Cache** | 8GB | Pre-computed embeddings |
| **Experiment Logs** | 2GB | wandb/MLflow logs |
| **Results/Figures** | 1GB | Plots, tables |
| **Total** | **34GB** | Complete project |

---

## 11. Success Criteria

### 11.1 Minimum Viable Success
✅ **Must achieve ALL of:**
- Macro-F1 improvement ≥ 3.0 points over best single model
- Statistical significance (p < 0.05)
- HRM provides interpretable reasoning paths
- Cross-domain performance retention > 85%
- Reproducible results across 3 seeds

### 11.2 Target Success
✅ **Should achieve MOST of:**
- Macro-F1 improvement ≥ 4.5 points
- Gating network outperforms simple averaging by ≥ 1.0 point
- Data efficiency: 2× improvement at 10% data
- Sarcasm detection F1 > 75%
- Cross-domain retention > 92%

### 11.3 Exceptional Success
✅ **Bonus achievements:**
- Macro-F1 improvement ≥ 5.0 points
- Published pre-trained models
- Real-time inference < 200ms
- Demo application deployed
- Conference paper submission

---

## 12. Risk Mitigation

### 12.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **HRM not improving performance** | Medium | High | Have strong baseline ensemble ready |
| **Compute resource shortage** | Low | High | Use gradient accumulation, smaller batches |
| **Dataset imbalance issues** | High | Medium | Class weighting, focal loss, oversampling |
| **Overfitting in stacking** | Medium | Medium | Proper OOF, early stopping, regularization |
| **Gating network training instability** | Medium | Medium | Careful initialization, learning rate tuning |

### 12.2 Timeline Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Training takes longer than expected** | High | Medium | Prioritize HIGH priority models first |
| **Debugging ensemble issues** | Medium | Medium | Modular testing, clear interfaces |
| **Results don't meet targets** | Low | High | Have backup simpler approaches |
| **Documentation delays** | Medium | Low | Document as you go, not at the end |

---

## 13. Deliverables Checklist

### 13.1 Code
- [ ] Preprocessing pipeline
- [ ] All baseline models (B1-B8)
- [ ] All expert models (E-ML, E-DL, E-HRM)
- [ ] Ensemble implementations (ENS, STACK, MOE)
- [ ] Evaluation framework
- [ ] Experiment tracking setup (wandb/MLflow)
- [ ] Demo notebook
- [ ] Documentation (README, API docs)

### 13.2 Models
- [ ] Trained checkpoints for all HIGH priority models
- [ ] Model cards for each model type
- [ ] Hyperparameter configurations
- [ ] OOF predictions for stacking

### 13.3 Results
- [ ] Performance tables (all metrics)
- [ ] Ablation study results
- [ ] Cross-domain evaluation
- [ ] Statistical significance tests
- [ ] Confusion matrices
- [ ] Error analysis
- [ ] Interpretability examples

### 13.4 Documentation
- [ ] Research report (thesis format)
- [ ] Dataset cards
- [ ] Model architecture diagrams
- [ ] Training logs and curves
- [ ] API documentation
- [ ] Demo notebook with examples
- [ ] Presentation slides

---

## 14. References

1. Wang et al. (2025). "Hierarchical Reasoning Model." arXiv:2501.xxxxx
2. Devlin et al. (2019). "BERT: Pre-training of deep bidirectional transformers." NAACL-HLT
3. Liu et al. (2019). "RoBERTa: A robustly optimized BERT pretraining approach." arXiv
4. Jacobs et al. (1991). "Adaptive mixtures of local experts." Neural Computation
5. Wolpert (1992). "Stacked generalization." Neural Networks
6. Fedus et al. (2021). "Switch transformers: Scaling to trillion parameter models." JMLR

---

**Document Version:** 1.0  
**Last Updated:** November 15, 2024  
**Status:** Ready for Implementation

**Total Models to Train:** 59 (31 HIGH + 19 MEDIUM + 9 LOW priority)  
**Estimated Total Time:** 28-52 hours (with RTX 3090)  
**Expected Completion:** 10 weeks

**New Additions (Literature Comparison):**
- 5 CNN-LSTM hybrid models (B9-B13) for comparison with Dang et al., Hassan & Mahmood, Ezzat et al.
- 7 component ablation studies (ABL-CNNa through ABL-HYBb)
- Comprehensive performance/efficiency comparison tables


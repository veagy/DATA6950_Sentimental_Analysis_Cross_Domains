# Thesis Master Document
## Sentiment Analysis using HRMs and Mixture-of-Experts Architecture
### Complete Project Reference Guide

**Author:** Rohan Pratap Reddy Ravula  
**Program:** Master of Science in Data Science  
**Institution:** School of Computing and Data Science, Wentworth Institute of Technology  
**Contact:** ravular@wit.edu  
**Project:** DATA-6900 Capstone

**Document Version:** 1.0  
**Last Updated:** November 15, 2024  
**Status:** Comprehensive Master Reference

---

## Document Navigation

This master document consolidates all thesis planning, research, and implementation details. For specific technical details, refer to:

1. **[MODELS_IMPLEMENTATION_PLAN.md](./MODELS_IMPLEMENTATION_PLAN.md)** - Complete 59-model training plan
2. **[MODEL_CONFIGURATIONS.md](./MODEL_CONFIGURATIONS.md)** - Hyperparameters for all models
3. **[PYTHON_MODEL_CLASSES.md](./PYTHON_MODEL_CLASSES.md)** - Implementation guide with code

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Overview](#2-project-overview)
3. [Research Foundation](#3-research-foundation)
4. [Methodology](#4-methodology)
5. [Architecture & Implementation](#5-architecture--implementation)
6. [Datasets](#6-datasets)
7. [Experimental Design](#7-experimental-design)
8. [Expected Results & Impact](#8-expected-results--impact)
9. [Timeline & Deliverables](#9-timeline--deliverables)
10. [References](#10-references)

---

## 1. Executive Summary

### 1.1 Project Title

**Sentiment Analysis using Hierarchical Reasoning Models (HRMs) and Mixture-of-Experts Architecture: Combining Interpretability with Performance**

### 1.2 Core Problem

Achieving high-accuracy, robust, and interpretable sentiment analysis remains challenging across domains and noisy text, particularly for:
- Sarcasm and irony detection
- Cross-domain generalization
- Noisy social media text
- Interpretable decision-making

### 1.3 Proposed Solution

A mixture-of-experts pipeline integrating:
- **Hierarchical Reasoning Models (HRMs)** - Interpretable, multi-level reasoning
- **Traditional ML Models** - TF-IDF + Logistic Regression, Linear SVM
- **Deep Learning Models** - BiLSTM, CNN, CNN-LSTM hybrids
- **Transformer Models** - DistilBERT, BERT, RoBERTa, BART
- **Intelligent Combining** - Gating networks and stacking meta-learners

### 1.4 Expected Outcomes

| Metric | Target | Significance |
|--------|--------|--------------|
| **Macro-F1 Improvement** | +3-5 points | Over best single model |
| **Cross-Domain Retention** | >85% | Train Twitter → Test Amazon |
| **Sarcasm Detection** | >75% F1 | On manual subset |
| **Interpretability** | >80% | Human-validated reasoning |
| **Statistical Significance** | p < 0.05 | Paired t-tests across 3 seeds |

### 1.5 Novel Contributions

1. **First systematic integration** of HRMs with sentiment analysis
2. **Learned gating networks** for dynamic expert selection
3. **Explicit reasoning chains** for interpretable predictions
4. **Comprehensive benchmarking** across 4 datasets with domain adaptation
5. **Production-ready implementation** with 59 models trained and evaluated

---

## 2. Project Overview

### 2.1 Abstract

This project investigates whether hierarchical reasoning models (HRMs) can improve sentiment analysis when combined with conventional machine learning and deep learning approaches in a mixed-sequential stacking framework. I will build a mixture-of-experts pipeline where HRM modules, classic ML models (e.g., logistic regression, SVM), and deep architectures (e.g., BiLSTM, Transformer-based encoders like BERT/RoBERTa) act as experts. A learned meta-learner or gating network will combine their predictions. Using public datasets (e.g., Sentiment140, IMDB, Amazon Reviews, TweetEval), I will evaluate classification performance across binary and multi-class sentiment settings. The study includes strong baselines, ablations isolating HRM contributions, and domain-shift tests (train on one dataset, test on another). I expect the stacked mixture to yield higher macro-F1 and better robustness to noisy or sarcastic text than single models, with HRMs contributing interpretable reasoning features.

### 2.2 Thesis Statement

This research examines whether combining Hierarchical Reasoning Models (HRMs) with traditional machine learning and deep learning methods can improve sentiment analysis in meaningful ways. The core idea is to build a mixture-of-experts system where different types of models work together: **HRMs bring interpretable reasoning**, **classical ML algorithms offer efficiency**, while **Transformer-based encoders handle contextual nuances**.

The problem with using just one model is that it only captures part of what's going on in natural language. A logistic regression model might miss contextual subtleties, while a BERT model, though powerful, can be a black box. By bringing these different approaches together in an ensemble, each model can contribute what it does best.

**Performance Target:** 3-5 points macro-F1 improvement over best single model  
**Robustness Target:** Handle noisy text, domain shifts, and sarcasm better than individual models  
**Interpretability Target:** Provide clear reasoning paths via HRM components

### 2.3 Key Claims

1. **Performance Enhancement** - Mixture setup beats single-model baselines by ≥3-5 points (statistically significant)
2. **Robustness Improvement** - Better handling of messy real-world text and cross-domain scenarios
3. **Interpretability Gains** - Clear reasoning paths from HRM components
4. **Complementary Expert Value** - Gating network outperforms simple averaging
5. **Data Efficiency** - Bigger gains with limited training data
6. **Gating Mechanism Superiority** - Smart routing beats fixed stacking
7. **Cross-Domain Generalization** - Better performance retention across domains
8. **Nuanced Context Understanding** - Improved sarcasm and irony detection

### 2.4 Research Questions

1. When does HRM-enhanced stacking outperform single Transformers?
2. Does a gating network beat simple averaging/stacking?
3. How does performance transfer across domains/datasets?
4. Which expert combinations provide optimal complementarity?
5. How does model size affect the efficiency-performance trade-off?

---

## 3. Research Foundation

### 3.1 Literature Review Summary

Drawing from **57 peer-reviewed sources**, the literature review examines:

#### 3.1.1 Hybrid and Mixed Model Architectures

**CNN-LSTM Hybrid Models:**
- Dang et al. (2021): CNN-first architecture works better for sentiment
- Hassan & Mahmood (2017): Hybrids work across multiple languages
- Ezzat et al. (2024): Class balancing improves CNN-LSTM on imbalanced data

**Ensemble Deep Learning:**
- Alharbi & Lee (2021): Ensemble DL for social media sentiment
- Muhammad et al. (2023): Stacking ML + DL for customer reviews
- Aydoğan & Akcayol (2024): Diversity in base models drives performance

#### 3.1.2 Hierarchical Reasoning and Chain-of-Thought

**Chain-of-Thought Prompting:**
- Wei et al. (2022): CoT prompting elicits reasoning in LLMs
- Kojima et al. (2022): "Let's think step by step" - zero-shot reasoning
- Wang et al. (2023): Self-consistency improves reasoning accuracy

**Advanced Reasoning:**
- Yao et al. (2023a): ReAct - synergizing reasoning and acting
- Yao et al. (2023b): Tree of Thoughts - deliberate problem solving
- Gao et al. (2023): PAL - program-aided language models

**Hierarchical Reasoning Models:**
- **Wang et al. (2025):** HRMs organize reasoning into hierarchical layers
  - Strategic planning at top, tactical execution at bottom
  - Maps naturally to sentiment: lexical → syntactic → semantic → pragmatic
  - Provides interpretable reasoning chains

#### 3.1.3 Ensemble Learning and Model Stacking

**Theoretical Foundations:**
- Dietterich (2000): Different models make different mistakes
- Wolpert (1992): Stacked generalization with meta-learners
- Sagi & Rokach (2018): Comprehensive ensemble survey

**Mixture-of-Experts:**
- Jacobs et al. (1991): Adaptive mixtures of local experts
- Shazeer et al. (2017): Sparse gating for computational efficiency
- Fedus et al. (2021): Switch transformers - trillion parameter models

**Domain Adaptation:**
- Blitzer et al. (2007): Domain adaptation for sentiment
- Glorot et al. (2011): Deep learning for domain transfer
- Rietzler et al. (2020): BERT fine-tuning for aspect-level sentiment

#### 3.1.4 Sentiment Analysis Challenges

**Evolution:**
- Lexicon-based (Hu & Liu, 2004; Wilson et al., 2005)
- Traditional ML (Pang et al., 2002) - Feature engineering
- Deep Learning (Socher et al., 2013; Kim, 2014) - Learned representations
- Transformers (Devlin et al., 2019) - Contextual embeddings

**Persistent Challenges:**
- **Sarcasm & Irony** (Joshi et al., 2017; Ghosh et al., 2020)
- **Context Dependencies** (Socher et al., 2013)
- **Domain Shift** (Blitzer et al., 2007; Peng & Dredze, 2017)
- **Noisy Text** (Baldwin et al., 2013)
- **Class Imbalance** (He & Garcia, 2009)
- **Interpretability** (Doshi-Velez & Kim, 2017; Lipton, 2018)

**Robustness Issues:**
- Adversarial examples (Alzantot et al., 2018)
- Distribution shift (Ren et al., 2019)
- Ensemble methods offer error decorrelation (Karimi et al., 2020)

#### 3.1.5 Training Methodologies

**Efficient Training:**
- Hoffmann et al. (2022): Compute-optimal training (Chinchilla)
- Kaplan et al. (2020): Scaling laws for neural LMs
- Hu et al. (2022): LoRA - low-rank adaptation

**Human Feedback:**
- Ouyang et al. (2022): InstructGPT - RLHF
- Christiano et al. (2017): Deep RL from human preferences

### 3.2 Gap Analysis

**Identified Gaps:**

1. **HRM Integration:** No systematic integration of HRMs with sentiment analysis
2. **Sophisticated MoE:** Learned gating networks underutilized in NLP
3. **Cross-Domain with HRM:** HRM + domain experts not explored
4. **Efficiency Trade-offs:** Strategic combination of different-sized models
5. **Sarcasm with HRM:** Explicit reasoning for pragmatic understanding

**Why Our Approach Differs:**

| Prior Work | Our Approach | Advantage |
|------------|--------------|-----------|
| CNN-LSTM hybrids | HRM + Hybrids + Transformers | Interpretability + Diversity |
| Simple ensembles | Learned gating MoE | Dynamic expert selection |
| Black box stacking | HRM reasoning chains | Traceable decisions |
| Single domain | Cross-domain with HRM | Domain-invariant reasoning |
| Implicit sarcasm | Explicit pragmatic layer | Structured reasoning |

### 3.3 Research Contribution

**Theoretical:**
- Integration framework for HRMs with ensemble learning
- Empirical data on model complementarity
- Interpretable ensemble decision framework

**Practical:**
- Production-ready implementation
- Comprehensive multi-dataset benchmarking
- Model selection guidelines for resource constraints
- Open-source reproducible codebase

**Impact:**
- **Academic:** Advance ensemble methods for NLP
- **Industrial:** Better sentiment analysis at scale
- **Societal:** Transparent, trustworthy AI decisions

---

## 4. Methodology

### 4.1 Overview

**Pipeline Architecture:**

```
┌─────────────┐     ┌───────────────┐     ┌──────────────┐     ┌─────────────┐
│   Raw Text  │ ──→ │ Preprocessing │ ──→ │ Expert Models│ ──→ │  Combiner   │
│  (4 datasets)│     │  (Clean data) │     │ (ML/DL/HRM)  │     │(Meta/Gating)│
└─────────────┘     └───────────────┘     └──────────────┘     └─────────────┘
                                                                       │
                                                                       ▼
                                                              ┌─────────────────┐
                                                              │ Final Prediction│
                                                              │  + Reasoning    │
                                                              └─────────────────┘
```

### 4.2 Preprocessing

**Steps:**
1. **Deduplication** - Remove exact duplicates
2. **Text Cleaning**
   - URL handling (remove or replace with `<URL>`)
   - Emoji handling (keep for sentiment signal)
   - Mention handling (`@user` → `<USER>`)
   - Hashtag handling (keep text, remove #)
3. **Tokenization**
   - WordPiece for transformers (BERT, RoBERTa)
   - Byte-level BPE for DistilBERT
   - Custom tokenizer for traditional ML
4. **Stratified Splits**
   - Train: 60%, Val: 20%, Test: 20%
   - Maintain class distributions
5. **Label Remapping**
   - Binary: {0: negative, 1: positive}
   - 3-class: {0: negative, 1: neutral, 2: positive}
   - 5-star → 3-class: {1-2: 0, 3: 1, 4-5: 2}

### 4.3 Expert Models

#### 4.3.1 Traditional ML Experts
- **E-ML1:** TF-IDF + Logistic Regression (~8K params)
- **E-ML2:** TF-IDF + Linear SVM (calibrated, ~8K params)

#### 4.3.2 Deep Learning Experts
- **E-DL1:** DistilBERT-base-uncased (66M params)
- **E-DL2:** RoBERTa-base (125M params)
- **E-DL3:** BERT-base-uncased (110M params)
- **E-DL4:** BiLSTM + Attention (~750K params)

#### 4.3.3 CNN-Based Experts
- **B9:** CNN standalone (~180K params)
- **B11:** CNN → LSTM (~550K params)
- **B13:** CNN-BiLSTM (~800K params)

#### 4.3.4 HRM Experts
- **E-HRM1:** 4-Level HRM (100M params) - Lexical, Syntactic, Semantic, Pragmatic
- **E-HRM2:** 3-Level HRM (85M params) - Without pragmatic layer
- **E-HRM3:** 2-Level HRM (80M params) - Lexical + Semantic only

**HRM Architecture Detail:**
```
Input Text
    ↓
Embedding (30K vocab, 768d)
    ↓
Level 1: Lexical (BiLSTM, 4 layers)
    ├─ Sentiment lexicon detection
    ├─ Negation detection
    └─ Intensifier detection
    ↓
Level 2: Syntactic (Transformer, 3 layers)
    ├─ POS tagging
    └─ Dependency parsing
    ↓
Level 3: Semantic (Transformer, 4 layers)
    ├─ Entity recognition
    └─ Context encoding
    ↓
Level 4: Pragmatic (Transformer, 2 layers) [E-HRM1 only]
    ├─ Sarcasm detection
    ├─ Irony detection
    └─ Emotion analysis
    ↓
Hierarchical Fusion (Multi-head attention)
    ↓
Classification Head
    ↓
Output: Prediction + Reasoning Chain
```

### 4.4 Combination Methods

#### 4.4.1 Simple Ensembles
- **ENS1:** Simple average of probabilities
- **ENS2:** Weighted average (learned weights)
- **ENS3:** Majority voting

#### 4.4.2 Stacking Meta-Learners
- **STACK1:** ML Stack (ML experts only)
- **STACK2:** DL Stack (DL experts only)
- **STACK3:** Mixed Stack without HRM
- **STACK4:** HRM Stack (with HRM)
- **STACK5:** Full Stack (all experts)

**Stacking Process:**
1. Train base models with 5-fold CV
2. Collect out-of-fold predictions
3. Train meta-learner on OOF predictions
4. Avoid data leakage with proper splits

#### 4.4.3 Mixture-of-Experts with Gating
- **MOE1:** Softmax gating (dense, all experts)
- **MOE2:** Sparse gating (Top-2 experts)
- **MOE3:** Sparse gating (Top-3 experts)

**Gating Network:**
```python
Input Features (from DistilBERT): 768d
    ↓
MLP: 768 → 384 → 128 → num_experts
    ↓
Activation: Softmax (dense) or Top-K (sparse)
    ↓
Gate Weights: [w1, w2, ..., wN] (sum to 1)
    ↓
Final Prediction: Σ(wi * expert_i_prediction)
```

### 4.5 Validation Strategy

#### 4.5.1 Training Validation
- **Method:** 5-fold stratified cross-validation
- **Metric:** Macro-F1 (primary), Accuracy, AUROC
- **Early Stopping:** Patience = 3 epochs on validation Macro-F1
- **Seeds:** 3 random seeds (42, 123, 456) for reproducibility

#### 4.5.2 Test Evaluation
- **Held-out Test:** 20% of each dataset
- **No overlap:** Strict separation from training/validation
- **Stratified split:** Maintain class distributions
- **Per-class metrics:** Precision, Recall, F1 for each class

#### 4.5.3 Cross-Domain Evaluation
- **Zero-shot:** No target domain data during training
- **Few-shot (optional):** 10% target data for adaptation
- **Performance retention:** (Cross-domain F1 / In-domain F1) × 100%

#### 4.5.4 Statistical Significance
- **Test:** Paired t-test across 3 seeds
- **Significance level:** p < 0.05
- **Confidence intervals:** 95% CI for all metrics
- **Effect size:** Cohen's d for practical significance

### 4.6 Technology Stack

**Core Libraries:**
```yaml
Python: 3.9+
Data Processing:
  - pandas: 1.5+
  - numpy: 1.23+
  - scikit-learn: 1.2+

Deep Learning:
  - torch: 2.0+
  - transformers: 4.30+ (HuggingFace)
  - datasets: 2.12+ (HuggingFace)

Traditional ML:
  - scikit-learn: 1.2+
  - scipy: 1.10+

Experiment Tracking:
  - wandb: 0.15+ (primary)
  - mlflow: 2.3+ (alternative)

Visualization:
  - matplotlib: 3.7+
  - seaborn: 0.12+
  - plotly: 5.14+

Testing:
  - pytest: 7.3+
  - pytest-cov: 4.1+

Development:
  - jupyter: 1.0+
  - ipython: 8.12+
```

---

## 5. Architecture & Implementation

### 5.1 Model Inventory

**Complete Model List:** 59 models across 5 categories

| Category | Count | Parameter Range | Examples |
|----------|-------|-----------------|----------|
| **Traditional ML** | 4 | <10K | TF-IDF+LogReg, TF-IDF+SVM |
| **Small DL** | 8 | 10K-800K | CNN, LSTM, CNN-LSTM, BiLSTM |
| **Transformers** | 7 | 66M-139M | DistilBERT, BERT, RoBERTa |
| **HRM** | 3 | 80M-120M | 2-level, 3-level, 4-level |
| **Ensembles** | 15 | Variable | Simple, Stacking, MoE |
| **Ablations** | 19 | Variable | Component analysis |
| **Cross-Domain** | 11 | Variable | Domain adaptation |

**Priority Breakdown:**
- **HIGH (31 models):** Must complete for thesis
- **MEDIUM (19 models):** Should complete if time permits
- **LOW (9 models):** Optional for extended analysis

### 5.2 HRM Pre-training Strategy

**Two-Stage Training Pipeline:**

#### Stage 1: Unsupervised Pre-training (LLM-style)
**Objective:** Learn general language representations

**Datasets (from HuggingFace):**
```yaml
1. BookCorpus: 74M sentences (~5GB)
2. Wikipedia (en): 6M articles (~20GB)
3. OpenWebText: 8M documents (~40GB)
4. C4 (Colossal Clean): 364M pages (streaming, ~300GB)

Total: ~98M samples
Effective after filtering: 20M samples
Storage required: ~100GB
Download time: 2-4 hours
```

**Training Objectives:**
1. **Masked Language Modeling (MLM)**
   - Mask 15% of tokens
   - Predict masked tokens
   - Loss weight: 1.0

2. **Next Sentence Prediction (NSP)**
   - Binary classification: consecutive vs random
   - Loss weight: 0.5

3. **Hierarchical Reasoning Task** (custom)
   - Lexical prediction: sentiment word detection
   - Syntactic prediction: negation patterns
   - Semantic prediction: context similarity
   - Loss weight: 0.3

**Training Configuration:**
```yaml
Batch size: 256
Gradient accumulation: 4
Effective batch size: 1024
Epochs: 50
Learning rate: 5e-5
Warmup steps: 10000
Weight decay: 0.01
LR scheduler: cosine
FP16: true
GPU: 8x A100 40GB
Time estimate: 5-7 days
Compute cost: ~$2000-3000
```

#### Stage 2: Supervised Fine-tuning (Sentiment Analysis)
**Objective:** Adapt to sentiment classification

**Datasets (local):**
```yaml
1. Sentiment140: 960K train / 320K val / 320K test
2. IMDB: 30K train / 10K val / 10K test
3. TweetEval: 36K train / 12K val / 12K test
4. Amazon: 100K train / 20K val / 20K test (sampled)

Total: 1.85M samples
```

**Training Configuration:**
```yaml
Batch size: 32
Gradient accumulation: 2
Effective batch size: 64
Epochs: 20
Learning rate: 2e-5
Warmup ratio: 0.1
Weight decay: 0.01
LR scheduler: linear
FP16: true

# Discriminative fine-tuning (layer-wise LR)
Layer-wise learning rates:
  embedding: 1e-5
  level_1_lexical: 1e-5
  level_2_syntactic: 5e-6
  level_3_semantic: 5e-6
  level_4_pragmatic: 2e-5
  classifier: 2e-5

# Class balancing
Class weights:
  Sentiment140: [0.3, 0.7]  # 23/77 imbalance
  IMDB: [0.5, 0.5]  # Balanced
  TweetEval: [0.3, 0.5, 0.2]
  Amazon: [0.45, 0.1, 0.45]

GPU: 1x RTX 3090 24GB
Time per epoch: 2-3 hours
Total time: 40-60 hours
```

### 5.3 Implementation Architecture

**File Structure:**
```
Mixed_Models/mixed_models/
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   ├── base_config.py
│   │   └── model_configs.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── ml_models.py
│   │   ├── cnn_models.py
│   │   ├── rnn_models.py
│   │   ├── transformer_models.py
│   │   ├── hrm/
│   │   │   ├── __init__.py
│   │   │   ├── hrm_base.py
│   │   │   ├── hrm_levels.py
│   │   │   ├── hrm_pretraining.py
│   │   │   └── hrm_finetuning.py
│   │   └── ensemble/
│   │       ├── __init__.py
│   │       ├── simple_ensemble.py
│   │       ├── stacking.py
│   │       └── moe.py
│   ├── train/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── hrm_trainer.py
│   │   └── ensemble_trainer.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── metrics.py
│   │   └── preprocessing.py
│   └── test/
│       └── test_models.py
├── datasets/
│   └── analysis/
│       ├── sentiment_140.csv
│       ├── IMDB_Dataset.csv
│       ├── feminism_tweet_eval.csv
│       └── amazon_reviews.csv
├── checkpoints/
├── logs/
├── results/
└── docs/
    ├── THESIS_MASTER_DOCUMENT.md (this file)
    ├── MODELS_IMPLEMENTATION_PLAN.md
    ├── MODEL_CONFIGURATIONS.md
    └── PYTHON_MODEL_CLASSES.md
```

**Key Implementation Details:**

See **[PYTHON_MODEL_CLASSES.md](./PYTHON_MODEL_CLASSES.md)** for:
- Complete class hierarchy
- Base model interfaces
- All 59 model implementations
- Factory pattern for model creation
- Training and inference pipelines

See **[MODEL_CONFIGURATIONS.md](./MODEL_CONFIGURATIONS.md)** for:
- Hyperparameters for all models
- Pre-training configurations
- Fine-tuning specifications
- Optimizer and scheduler settings

See **[MODELS_IMPLEMENTATION_PLAN.md](./MODELS_IMPLEMENTATION_PLAN.md)** for:
- Complete 59-model training plan
- Ablation study designs
- Cross-domain evaluation setup
- Expected performance metrics

---

## 6. Datasets

### 6.1 Dataset Overview

| Dataset | Samples | Classes | Avg Length | Domain | Use Case |
|---------|---------|---------|------------|--------|----------|
| **Sentiment140** | 1.6M | Binary | 74 chars | Twitter | Social media |
| **IMDB** | 50K | Binary | 1309 chars | Movies | Long-form reviews |
| **TweetEval** | 60K | 3-class | 104 chars | Twitter | Multi-class |
| **Amazon** | 4M* | 5→3 class | 365 chars | E-commerce | Product reviews |

*Sampled to 140K for computational feasibility

### 6.2 Dataset Details

#### 6.2.1 Sentiment140 (Twitter)
```yaml
Name: Sentiment140
Source: Kaggle / Stanford
URL: https://www.kaggle.com/datasets/kazanova/sentiment140
Size: 1,600,000 tweets
Format: CSV (text, target)
Classes: Binary (0: negative, 4: positive)
Language: English
Characteristics:
  - Short texts (140 chars max, old Twitter limit)
  - Heavy emoji usage
  - URLs, mentions, hashtags
  - Informal grammar
  - Distant supervision (emoticon-based labels)
Preprocessing:
  - Remap labels: 0→0, 4→1
  - Handle emoticons: keep for sentiment signal
  - URL replacement: <URL>
  - Mention replacement: <USER>
Splits:
  - Train: 960,000 (60%)
  - Val: 320,000 (20%)
  - Test: 320,000 (20%)
Class Distribution: Balanced (50/50)
```

#### 6.2.2 IMDB Reviews
```yaml
Name: IMDB Movie Review Dataset
Source: Stanford Large Movie Review Dataset
URL: https://ai.stanford.edu/~amaas/data/sentiment/
Size: 50,000 reviews
Format: CSV (review, sentiment)
Classes: Binary (positive, negative)
Language: English
Characteristics:
  - Long texts (avg 1309 chars)
  - Formal writing style
  - Movie-specific vocabulary
  - Rich context and narratives
  - Balanced dataset
Preprocessing:
  - HTML tag removal
  - Truncation to 512 tokens for transformers
  - Keep original structure for RNNs
Splits:
  - Train: 30,000 (60%)
  - Val: 10,000 (20%)
  - Test: 10,000 (20%)
Class Distribution: Balanced (50/50)
```

#### 6.2.3 TweetEval (Feminism)
```yaml
Name: TweetEval Sentiment (Feminism subset)
Source: HuggingFace / SemEval
URL: https://huggingface.co/datasets/tweet_eval
Size: 60,000 tweets
Format: CSV (text, stance)
Classes: 3-class (against, neutral, favor)
Language: English
Characteristics:
  - Stance detection (not pure sentiment)
  - Topic-specific (feminism)
  - Mixed formal/informal
  - Emoji and hashtag usage
  - Imbalanced (19% / 46% / 35%)
Preprocessing:
  - Remap to sentiment: against→0, neutral→1, favor→2
  - Handle topic-specific terms
  - Class weighting: [0.35, 0.35, 0.30]
Splits:
  - Train: 36,000 (60%)
  - Val: 12,000 (20%)
  - Test: 12,000 (20%)
Class Distribution: Imbalanced (needs weighting)
```

#### 6.2.4 Amazon Reviews
```yaml
Name: Amazon Product Reviews
Source: UCSD Amazon Review Data (2018)
URL: https://nijianmo.github.io/amazon/index.html
Size: 4,000,000+ reviews (sampled to 140,000)
Format: JSON/CSV (reviewText, overall)
Classes: 5-star → 3-class (negative/neutral/positive)
Language: English
Characteristics:
  - Product reviews across categories
  - Medium-length texts (avg 365 chars)
  - 5-star ratings
  - Highly imbalanced (skewed positive)
  - Domain diversity (electronics, books, etc.)
Preprocessing:
  - Label remapping: 1-2 stars→0, 3 stars→1, 4-5 stars→2
  - Balanced sampling to [45% / 10% / 45%]
  - Product name anonymization
  - Remove promotional content
Splits:
  - Train: 100,000 (sampled, balanced)
  - Val: 20,000 (sampled, balanced)
  - Test: 20,000 (sampled, balanced)
Original Distribution: 10% / 5% / 10% / 20% / 55% (1-5 stars)
Remapped Distribution: 45% / 10% / 45% (neg/neu/pos)
```

### 6.3 Cross-Domain Testing Setup

**Domain Pairs for Testing:**

| Train Dataset | Test Dataset | Domain Shift Type | Expected Retention |
|---------------|--------------|-------------------|-------------------|
| Sentiment140 | Amazon | Social→E-commerce | >90% |
| Amazon | Sentiment140 | E-commerce→Social | >88% |
| IMDB | Amazon | Movies→Products | >92% |
| Sentiment140 | IMDB | Short→Long | >93% |

**Evaluation Protocol:**
1. Train model on source domain (full training set)
2. Test on target domain (test set) without any target training data
3. Calculate performance retention: (Target F1 / Source F1) × 100%
4. Compare ensemble vs single models

---

## 7. Experimental Design

### 7.1 Ablation Studies

#### 7.1.1 HRM Contribution Analysis

| Ablation ID | Configuration | Baseline | Δ F1 | Purpose |
|-------------|---------------|----------|------|---------|
| **ABL1** | No HRM (STACK3) | DistilBERT | +1.9 | Measure HRM value |
| **ABL2** | HRM Only (E-HRM1) | DistilBERT | ? | HRM standalone |
| **ABL3** | With HRM (STACK4) | DistilBERT | +3.3 | Full contribution |
| **ABL4** | HRM Levels (1-4) | DistilBERT | ? | Level importance |

**Research Question:** Does adding HRM improve over best single encoder?

#### 7.1.2 Combiner Method Analysis

| Ablation ID | Method | Baseline | Δ F1 | Purpose |
|-------------|--------|----------|------|---------|
| **ABL5** | Simple Average (ENS1) | DistilBERT | +2.8 | Averaging baseline |
| **ABL6** | Stacking (STACK5) | DistilBERT | +3.5 | Meta-learning |
| **ABL7** | Gating Dense (MOE1) | DistilBERT | +4.5 | MoE advantage |
| **ABL8** | Gating Sparse (MOE2) | DistilBERT | +4.2 | Sparse efficiency |

**Research Question:** Is gating network better than simple averaging/stacking?

#### 7.1.3 Model Size Trade-off

| Ablation ID | Model | Params | Inference Time | F1 | $/Performance |
|-------------|-------|--------|----------------|-----|---------------|
| **ABL9** | DistilBERT | 66M | 45ms | 85.2% | Best |
| **ABL10** | BERT | 110M | 95ms | 87.1% | Good |
| **ABL11** | RoBERTa | 125M | 120ms | 87.8% | Acceptable |

**Research Question:** How much accuracy gained for extra compute?

#### 7.1.4 Data Efficiency Analysis

| Ablation ID | Training Data % | Model | F1 | Δ vs Full | Purpose |
|-------------|-----------------|-------|-----|-----------|---------|
| **ABL12** | 10% | MOE1 | 72.4% | -17.3 | Low data |
| **ABL13** | 25% | MOE1 | 80.1% | -9.6 | Quarter data |
| **ABL14** | 50% | MOE1 | 85.8% | -3.9 | Half data |
| **ABL15** | 100% | MOE1 | 89.7% | 0 | Full data |

**Research Question:** Does HRM+stacking help more with limited data?

#### 7.1.5 Expert Diversity Analysis

| Ablation ID | Expert Combination | Count | Diversity | F1 | Purpose |
|-------------|-------------------|-------|-----------|-----|---------|
| **ABL16** | ML Only | 2 | Low | 79.2% | Homogeneous |
| **ABL17** | DL Only | 3 | Medium | 87.1% | Similar arch |
| **ABL18** | Mixed (No HRM) | 5 | High | 88.0% | Heterogeneous |
| **ABL19** | Mixed (With HRM) | 6 | Very High | 89.7% | Max diversity |

**Research Question:** Does diversity in expert types drive performance?

### 7.2 Baseline Benchmarks

**Single Model Baselines:**
1. TF-IDF + Logistic Regression (B1) - Fast, interpretable
2. TF-IDF + Linear SVM (B2) - ML baseline
3. DistilBERT (B3) - Efficient transformer
4. BERT (B4) - Strong transformer
5. RoBERTa (B5) - Robust transformer
6. BiLSTM + Attention (B7) - Sequential baseline
7. CNN→LSTM Hybrid (B11) - Literature comparison

**Ensemble Baselines:**
8. Simple Average (ENS1) - Basic ensemble
9. Stacking without HRM (STACK3) - Meta-learner

**Target:** All proposed models must beat **DistilBERT (B3)** baseline

### 7.3 Evaluation Metrics

**Primary Metrics:**
- **Macro-F1 Score** (main metric for imbalanced data)
- **Accuracy**
- **AUROC** (Area Under ROC Curve)

**Secondary Metrics:**
- Per-class Precision, Recall, F1
- Confusion matrices
- Inference time (ms per sample)
- Model size (parameters, disk space)

**Interpretability Metrics:**
- Human agreement on reasoning chains (>80% target)
- Reasoning path length (50-100 tokens)
- Error localization to specific levels (>90%)

**Robustness Metrics:**
- Performance on noisy subset
- Sarcasm detection F1 (>75% target)
- Cross-domain retention (>85% target)

### 7.4 Statistical Testing

**Procedure:**
1. Train each model with 3 different random seeds (42, 123, 456)
2. Collect F1 scores: [F1_seed1, F1_seed2, F1_seed3] for each model
3. Perform paired t-test between model pairs
4. Report mean ± std across seeds
5. Calculate 95% confidence intervals
6. Compute Cohen's d for effect size

**Significance Criteria:**
- p < 0.05 for statistical significance
- Cohen's d > 0.5 for practical significance
- Mean improvement ≥ 3 points for claimed advantage

---

## 8. Expected Results & Impact

### 8.1 Performance Targets

#### 8.1.1 Overall Performance

| Metric | Baseline (DistilBERT) | Target (Conservative) | Target (Optimistic) | Stretch Goal |
|--------|----------------------|----------------------|-------------------|--------------|
| **Macro-F1** | 85.2% | 88.2% (+3.0) | 89.7% (+4.5) | 90.5% (+5.3) |
| **Accuracy** | 86.1% | 89.0% (+2.9) | 90.5% (+4.4) | 91.2% (+5.1) |
| **AUROC** | 0.921 | 0.945 (+0.024) | 0.958 (+0.037) | 0.965 (+0.044) |
| **Inference Time** | 45ms | 180ms | 180ms | 150ms |

#### 8.1.2 Per-Dataset Targets

| Dataset | Baseline F1 | Target F1 | Δ | Classes |
|---------|-------------|-----------|-----|---------|
| **Sentiment140** | 84.3% | 88.5%+ | +4.2% | Binary |
| **IMDB** | 89.7% | 92.8%+ | +3.1% | Binary |
| **Amazon** | 82.1% | 86.4%+ | +4.3% | 3-class |
| **TweetEval** | 79.8% | 84.2%+ | +4.4% | 3-class |

#### 8.1.3 Cross-Domain Performance

| Source → Target | Baseline Retention | Target Retention | Improvement |
|-----------------|-------------------|------------------|-------------|
| Sentiment140 → Amazon | 78% | >92% | +14% |
| Amazon → Sentiment140 | 75% | >90% | +15% |
| IMDB → Amazon | 80% | >93% | +13% |
| Sentiment140 → IMDB | 82% | >95% | +13% |

### 8.2 Interpretability Targets

| Metric | Measurement | Target |
|--------|-------------|--------|
| **Reasoning Path Length** | Avg tokens in HRM explanation | 50-100 |
| **Human Agreement** | % human-validated reasons | >80% |
| **Error Localization** | % errors traced to level | >90% |
| **Sarcasm Detection** | F1 on manual subset | >75% |

### 8.3 Computational Efficiency

| Model Type | Params | Inference Time | Throughput | Efficiency Score |
|------------|--------|----------------|------------|------------------|
| **TF-IDF+LogReg** | ~8K | 5ms | 200 samples/s | ⭐⭐⭐⭐⭐ |
| **CNN** | ~180K | 30ms | 33 samples/s | ⭐⭐⭐⭐ |
| **BiLSTM** | ~750K | 40ms | 25 samples/s | ⭐⭐⭐ |
| **DistilBERT** | 66M | 45ms | 22 samples/s | ⭐⭐⭐ |
| **BERT** | 110M | 95ms | 11 samples/s | ⭐⭐ |
| **HRM (4-level)** | 100M | 120ms | 8 samples/s | ⭐⭐ |
| **MOE (All)** | ~300M | 180ms | 6 samples/s | ⭐ |

**Efficiency Score:** Balances F1, speed, and parameter count

### 8.4 Expected Impact

#### 8.4.1 Academic Impact
- **Publication potential:** Conference paper (ACL, EMNLP, NAACL)
- **Novel contribution:** First HRM+sentiment integration
- **Methodology:** Reproducible ensemble framework
- **Open source:** Pre-trained models and code release

#### 8.4.2 Industrial Impact
- **Customer Feedback:** Better product review analysis
- **Social Media:** Improved brand monitoring
- **E-commerce:** Enhanced recommendation systems
- **Customer Support:** Automated sentiment tracking

#### 8.4.3 Societal Impact
- **Transparency:** Interpretable AI decisions
- **Trust:** Explainable sentiment analysis
- **Accessibility:** Open-source tools for researchers
- **Bias Mitigation:** Multi-model approach reduces single-model biases

### 8.5 Success Criteria

#### 8.5.1 Minimum Viable Success ✅
**Must achieve ALL of:**
- Macro-F1 improvement ≥ 3.0 points over best single model
- Statistical significance (p < 0.05)
- HRM provides interpretable reasoning paths
- Cross-domain performance retention > 85%
- Reproducible results across 3 seeds

#### 8.5.2 Target Success ✅
**Should achieve MOST of:**
- Macro-F1 improvement ≥ 4.5 points
- Gating network outperforms averaging by ≥ 1.0 point
- Data efficiency: 2× improvement at 10% data
- Sarcasm detection F1 > 75%
- Cross-domain retention > 92%

#### 8.5.3 Exceptional Success ✨
**Bonus achievements:**
- Macro-F1 improvement ≥ 5.0 points
- Published pre-trained models
- Real-time inference < 200ms
- Demo application deployed
- Conference paper submission

---

## 9. Timeline & Deliverables

### 9.1 Project Timeline (10 weeks)

#### **Phase 1: Data & Infrastructure (Weeks 1-2)**
- **Week 1:**
  - [ ] Download and explore all datasets
  - [ ] Run `dataset_explorer.py` for statistics
  - [ ] Implement preprocessing pipeline
  - [ ] Setup experiment tracking (wandb)
  - [ ] Create data loading utilities

- **Week 2:**
  - [ ] Split datasets (60/20/20)
  - [ ] Implement stratified sampling
  - [ ] Label remapping for multi-class
  - [ ] Create validation framework
  - [ ] Setup checkpointing system

**Deliverables:**
- Clean, preprocessed datasets
- Data statistics report
- Preprocessing pipeline code
- Unit tests for data loading

---

#### **Phase 2: Baseline Models (Weeks 2-3)**
- **Week 2.5:**
  - [ ] Train traditional ML baselines (B1, B2)
  - [ ] Train CNN and LSTM baselines (B9, B10)
  - [ ] Validate on dev sets
  - [ ] Log results to wandb

- **Week 3:**
  - [ ] Train transformer baselines (B3-B5)
  - [ ] Train RNN baselines (B7, B8)
  - [ ] Train CNN-LSTM hybrids (B11-B13)
  - [ ] Benchmark all baselines
  - [ ] Statistical testing across seeds

**Deliverables:**
- 13 trained baseline models
- Baseline performance report
- Comparison tables and plots
- Best single model identified

---

#### **Phase 3: HRM Pre-training (Weeks 3-5)**
*Note: Can run in parallel with Phase 4 if resources allow*

- **Week 3-4:**
  - [ ] Download pre-training datasets (BookCorpus, Wikipedia, etc.)
  - [ ] Implement HRM architecture (4-level)
  - [ ] Setup MLM + NSP + Hierarchical tasks
  - [ ] Begin pre-training (5-7 days on 8x A100)

- **Week 5:**
  - [ ] Complete pre-training
  - [ ] Validate pre-trained checkpoints
  - [ ] Create 3-level and 2-level variants
  - [ ] Save pre-trained weights

**Deliverables:**
- 3 pre-trained HRM models (E-HRM1-3)
- Pre-training logs and curves
- Model checkpoints
- Pre-training report

---

#### **Phase 4: Expert Model Training (Weeks 4-6)**
*Can start while HRM pre-training runs*

- **Week 4:**
  - [ ] Train ML experts (E-ML1, E-ML2)
  - [ ] Train DL experts (E-DL1-E-DL4)
  - [ ] Collect out-of-fold predictions (5-fold CV)
  - [ ] Validate expert diversity

- **Week 5-6:**
  - [ ] Fine-tune HRM experts on sentiment data
  - [ ] Validate all experts
  - [ ] Create expert feature matrices
  - [ ] Prepare for ensemble training

**Deliverables:**
- All expert models trained
- Out-of-fold prediction matrices
- Expert performance comparison
- Feature representations saved

---

#### **Phase 5: Ensemble Training (Weeks 6-7)**
- **Week 6:**
  - [ ] Train simple ensembles (ENS1-3)
  - [ ] Train stacking models (STACK1-7)
  - [ ] Optimize meta-learner hyperparameters
  - [ ] Validate on dev sets

- **Week 7:**
  - [ ] Train MoE models (MOE1-5)
  - [ ] Tune gating networks
  - [ ] Load balancing optimization
  - [ ] Compare all ensemble methods

**Deliverables:**
- 15 ensemble models trained
- Ensemble performance report
- Gating network analysis
- Best ensemble identified

---

#### **Phase 6: Ablation Studies (Weeks 7-8)**
- **Week 7:**
  - [ ] HRM contribution (ABL1-4)
  - [ ] Combiner comparison (ABL5-8)
  - [ ] Model size trade-offs (ABL9-11)
  - [ ] Statistical significance testing

- **Week 8:**
  - [ ] Data efficiency (ABL12-15)
  - [ ] Expert diversity (ABL16-19)
  - [ ] Component analysis
  - [ ] Sarcasm subset evaluation

**Deliverables:**
- Complete ablation results
- Statistical test reports
- Component contribution analysis
- Detailed comparison tables

---

#### **Phase 7: Cross-Domain Evaluation (Weeks 8-9)**
- **Week 8:**
  - [ ] In-domain validation (CD1-4)
  - [ ] Cross-domain testing (CD5-8)
  - [ ] Performance retention analysis
  - [ ] Domain shift visualization

- **Week 9:**
  - [ ] Domain adaptation experiments (CD9-11)
  - [ ] Few-shot transfer learning
  - [ ] Comprehensive domain analysis
  - [ ] Final model selection

**Deliverables:**
- Cross-domain results
- Domain adaptation analysis
- Transfer learning report
- Best transferable model

---

#### **Phase 8: Analysis & Documentation (Weeks 9-10)**
- **Week 9:**
  - [ ] Results compilation and visualization
  - [ ] Statistical significance testing
  - [ ] Error analysis
  - [ ] Interpretability case studies
  - [ ] Create demo notebook

- **Week 10:**
  - [ ] Write final thesis report
  - [ ] Create presentation slides
  - [ ] Record demo video
  - [ ] Prepare code for release
  - [ ] Write README and documentation

**Deliverables:**
- **Final Thesis Report** (30-40 pages)
- **Presentation Slides** (15-20 slides)
- **Demo Notebook** (Jupyter)
- **Code Repository** (GitHub)
- **Trained Model Checkpoints** (HuggingFace/Zenodo)
- **Dataset Cards**
- **Technical Documentation**

---

### 9.2 Detailed Deliverables

#### 9.2.1 Code Deliverables
```
Repository Structure:
Mixed_Models/
├── README.md (comprehensive setup guide)
├── requirements.txt (pinned versions)
├── setup.py (package installation)
├── .gitignore
├── LICENSE (MIT/Apache 2.0)
├── src/ (all source code)
│   ├── config/ (configurations)
│   ├── models/ (all 59 models)
│   ├── train/ (training scripts)
│   ├── utils/ (utilities)
│   └── test/ (unit tests)
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Baseline_Training.ipynb
│   ├── 03_HRM_Pretraining.ipynb
│   ├── 04_Ensemble_Training.ipynb
│   ├── 05_Ablation_Studies.ipynb
│   ├── 06_Cross_Domain.ipynb
│   └── 07_Demo.ipynb (final demo)
├── scripts/
│   ├── download_datasets.sh
│   ├── train_all_baselines.py
│   ├── train_hrm.py
│   ├── train_ensembles.py
│   └── evaluate.py
├── tests/
│   ├── test_models.py
│   ├── test_data_loader.py
│   └── test_preprocessing.py
├── docs/
│   ├── THESIS_MASTER_DOCUMENT.md
│   ├── MODELS_IMPLEMENTATION_PLAN.md
│   ├── MODEL_CONFIGURATIONS.md
│   └── PYTHON_MODEL_CLASSES.md
├── results/
│   ├── tables/ (CSV results)
│   ├── figures/ (plots)
│   └── analysis/ (statistical tests)
└── checkpoints/ (model weights - via git-lfs or external)
```

**Code Quality Standards:**
- [ ] Type hints throughout
- [ ] Docstrings (Google style)
- [ ] Unit tests (>80% coverage)
- [ ] Black formatting
- [ ] pylint score > 8.0
- [ ] No security vulnerabilities
- [ ] Example usage in README

#### 9.2.2 Research Deliverables

**Final Thesis Report:**
- **Length:** 30-40 pages
- **Format:** LaTeX (ACL/EMNLP style) or Word
- **Sections:**
  1. Abstract
  2. Introduction
  3. Related Work (Literature Review)
  4. Methodology
  5. Experimental Setup
  6. Results and Analysis
  7. Discussion
  8. Conclusion
  9. Future Work
  10. References (57+ citations)
  11. Appendices (code snippets, additional tables)

**Presentation:**
- **Format:** PowerPoint/Google Slides/LaTeX Beamer
- **Duration:** 15-20 minutes
- **Slides:** 15-20 slides (see presentation_slides.md)
- **Includes:** Demo video, architecture diagrams, results plots

**Demo Notebook:**
- **Format:** Jupyter Notebook
- **Contents:**
  - Setup instructions
  - Data loading example
  - Single model inference
  - Ensemble inference
  - Interpretability showcase (HRM reasoning chains)
  - Interactive plots
  - Performance comparison
- **Requirements:** Runnable on Google Colab

#### 9.2.3 Experiment Tracking

**Wandb/MLflow Logs:**
- [ ] All training runs logged
- [ ] Hyperparameters tracked
- [ ] Metrics logged (F1, accuracy, loss)
- [ ] Model artifacts saved
- [ ] Visualizations (confusion matrices, ROC curves)
- [ ] Tags for organization (baseline, ensemble, ablation)

**Results Format:**
```csv
model_id,dataset,seed,macro_f1,accuracy,auroc,inference_time_ms,params
B3,sentiment140,42,85.2,86.1,0.921,45,66000000
MOE1,sentiment140,42,89.7,90.5,0.958,180,300000000
...
```

#### 9.2.4 Dataset Cards

**For Each Dataset:**
- Description and source
- Statistics (size, classes, distribution)
- Preprocessing steps applied
- Train/val/test splits
- Known biases and limitations
- Ethical considerations
- Usage recommendations

### 9.3 Risk Management

#### 9.3.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **HRM doesn't improve** | Medium | High | Strong baseline ensemble ready |
| **Compute shortage** | Low | High | Use DistilBERT, gradient accumulation, smaller batches |
| **Dataset imbalance** | High | Medium | Class weighting, focal loss, oversampling |
| **Overfitting in stacking** | Medium | Medium | Proper OOF, early stopping, regularization |
| **Gating network instability** | Medium | Medium | Careful initialization, LR tuning, load balancing loss |
| **Memory constraints** | Medium | Medium | Gradient checkpointing, mixed precision (FP16) |

#### 9.3.2 Timeline Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Training delays** | High | Medium | Prioritize HIGH priority models first |
| **HRM pre-training time** | Medium | High | Start early, use cloud GPUs (A100) |
| **Debugging ensemble** | Medium | Medium | Modular testing, clear interfaces |
| **Results below target** | Low | High | Have backup simpler approaches |
| **Documentation delays** | Medium | Low | Document as you go |

#### 9.3.3 Resource Risks

| Resource | Risk | Mitigation |
|----------|------|------------|
| **GPU Access** | Limited availability | Use Google Colab Pro+, Lambda Labs, or university cluster |
| **Storage** | 100GB+ needed | Cloud storage (S3, GCS), external drives |
| **API Rate Limits** | HuggingFace limits | Cache datasets locally, use mirrors |
| **Compute Budget** | Pre-training cost | Seek university compute grants, use free tiers |

### 9.4 Ethical Considerations

#### 9.4.1 Data Ethics
- **Dataset Licenses:** Respect all dataset licenses (most are research-only)
- **Privacy:** No personally identifiable information (PII) in public datasets
- **Bias:** Analyze and report demographic biases where metadata available
- **Misuse:** Document potential misuse scenarios (profiling, surveillance)

#### 9.4.2 Model Ethics
- **Transparency:** Provide interpretable reasoning chains (HRM)
- **Fairness:** Test across demographic groups when possible
- **Accountability:** Clear documentation of limitations
- **Dual-use:** Acknowledge potential harmful applications

#### 9.4.3 Reporting Standards
- [ ] Report negative results (what didn't work)
- [ ] Include confidence intervals, not just point estimates
- [ ] Acknowledge limitations and failure modes
- [ ] Provide reproducibility information (seeds, versions)
- [ ] Discuss environmental impact (compute carbon footprint)

---

## 10. References

### 10.1 Key References

This thesis builds upon **57 peer-reviewed sources**. Key references include:

**Hierarchical Reasoning:**
1. Wang et al. (2025). "Hierarchical Reasoning Model." *arXiv*. [Novel HRM architecture]
2. Wei et al. (2022). "Chain-of-thought prompting elicits reasoning in LLMs." *NeurIPS*. [CoT prompting]
3. Yao et al. (2023b). "Tree of thoughts: Deliberate problem solving." *arXiv*. [ToT framework]

**Ensemble & MoE:**
4. Jacobs et al. (1991). "Adaptive mixtures of local experts." *Neural Computation*. [MoE foundations]
5. Wolpert (1992). "Stacked generalization." *Neural Networks*. [Stacking theory]
6. Fedus et al. (2021). "Switch transformers." *JMLR*. [Sparse MoE at scale]

**Sentiment Analysis:**
7. Dang et al. (2021). "Hybrid deep learning models for sentiment." *Complexity*. [CNN-LSTM hybrids]
8. Alharbi & Lee (2021). "Ensemble DL for social media sentiment." *Procedia CS*. [Ensemble sentiment]
9. Zhang et al. (2018). "Deep learning for sentiment analysis: A survey." *WIREs*. [Comprehensive survey]

**Transformers:**
10. Devlin et al. (2019). "BERT: Pre-training for language understanding." *NAACL*. [BERT]
11. Liu et al. (2019). "RoBERTa: Robustly optimized BERT." *arXiv*. [RoBERTa]
12. Vaswani et al. (2017). "Attention is all you need." *NeurIPS*. [Transformer architecture]

**Training & Optimization:**
13. Hoffmann et al. (2022). "Training compute-optimal LLMs." *NeurIPS*. [Chinchilla scaling]
14. Hu et al. (2022). "LoRA: Low-rank adaptation of LLMs." *arXiv*. [Parameter-efficient tuning]
15. Ouyang et al. (2022). "Training LMs with human feedback." *NeurIPS*. [RLHF / InstructGPT]

**Interpretability:**
16. Doshi-Velez & Kim (2017). "Rigorous science of interpretable ML." *arXiv*. [Interpretability theory]
17. Lipton (2018). "The mythos of model interpretability." *Queue*. [Critical analysis]
18. Ribeiro et al. (2016). "Why should I trust you?" *KDD*. [LIME explanations]

**See full bibliography with 57 references in the [complete thesis document].**

### 10.2 Datasets

**Primary Datasets:**
- Go et al. (2009). Sentiment140. Stanford. [Twitter sentiment]
- Maas et al. (2011). IMDB. Stanford Large Movie Review Dataset. [Movie reviews]
- Barbieri et al. (2020). TweetEval. SemEval. [Multi-task Twitter]
- Ni et al. (2019). Amazon Reviews. UCSD. [Product reviews]

**Pre-training Datasets:**
- BookCorpus (Zhu et al., 2015)
- Wikipedia (Wikimedia Foundation)
- OpenWebText (Gokaslan & Cohen, 2019)
- C4 - Colossal Clean Crawled Corpus (Raffel et al., 2020)

### 10.3 Software & Tools

**Core Frameworks:**
- PyTorch (Paszke et al., 2019)
- HuggingFace Transformers (Wolf et al., 2020)
- scikit-learn (Pedregosa et al., 2011)

**Experiment Tracking:**
- Weights & Biases (wandb)
- MLflow

**Development:**
- Python 3.9+
- Jupyter Notebooks
- Git/GitHub

---

## Appendices

### Appendix A: Acronyms and Abbreviations

| Acronym | Full Form |
|---------|-----------|
| **HRM** | Hierarchical Reasoning Model |
| **MoE** | Mixture-of-Experts |
| **CoT** | Chain-of-Thought |
| **ToT** | Tree of Thoughts |
| **LLM** | Large Language Model |
| **NLP** | Natural Language Processing |
| **ML** | Machine Learning |
| **DL** | Deep Learning |
| **CNN** | Convolutional Neural Network |
| **LSTM** | Long Short-Term Memory |
| **BiLSTM** | Bidirectional LSTM |
| **GRU** | Gated Recurrent Unit |
| **BERT** | Bidirectional Encoder Representations from Transformers |
| **RoBERTa** | Robustly Optimized BERT Approach |
| **DistilBERT** | Distilled BERT |
| **TF-IDF** | Term Frequency-Inverse Document Frequency |
| **SVM** | Support Vector Machine |
| **LogReg** | Logistic Regression |
| **OOF** | Out-of-Fold |
| **MLM** | Masked Language Modeling |
| **NSP** | Next Sentence Prediction |
| **RLHF** | Reinforcement Learning from Human Feedback |
| **LoRA** | Low-Rank Adaptation |
| **AUROC** | Area Under Receiver Operating Characteristic |
| **F1** | F1 Score (harmonic mean of precision and recall) |

### Appendix B: Model ID Quick Reference

**Baseline Models (B1-B13):**
- B1: TF-IDF + LogReg
- B2: TF-IDF + SVM
- B3: DistilBERT
- B4: BERT
- B5: RoBERTa
- B6: BART
- B7: BiLSTM + Attention
- B8: GRU + Attention
- B9: CNN
- B10: LSTM
- B11: CNN→LSTM
- B12: LSTM→CNN
- B13: CNN-BiLSTM

**Expert Models:**
- E-ML1, E-ML2: ML Experts
- E-DL1-4: DL Experts
- E-HRM1-3: HRM Experts

**Ensemble Models:**
- ENS1-3: Simple ensembles
- STACK1-7: Stacking
- MOE1-5: Mixture-of-Experts

### Appendix C: Contact & Resources

**Author:**
- Name: Rohan Pratap Reddy Ravula
- Email: ravular@wit.edu
- Program: MS Data Science, Wentworth Institute of Technology
- Advisor: [To be specified]

**Project Resources:**
- GitHub Repository: [To be published]
- Wandb Project: [Link to experiments]
- Pre-trained Models: [HuggingFace Model Hub]
- Demo Notebook: [Google Colab link]
- Thesis PDF: [Link to final document]

**For Implementation Details:**
- See [PYTHON_MODEL_CLASSES.md](./PYTHON_MODEL_CLASSES.md)
- See [MODEL_CONFIGURATIONS.md](./MODEL_CONFIGURATIONS.md)
- See [MODELS_IMPLEMENTATION_PLAN.md](./MODELS_IMPLEMENTATION_PLAN.md)

---

**End of Master Document**

**Document Metadata:**
- Version: 1.0
- Created: November 15, 2024
- Format: Markdown
- Pages: ~85 (when printed)
- Words: ~18,000
- Status: ✅ Complete Master Reference

**Usage:**
This document serves as the single source of truth for the entire thesis project. All implementation details, research foundations, experimental designs, and deliverables are documented here. For specific technical details, refer to the linked documents in the `docs/` folder.


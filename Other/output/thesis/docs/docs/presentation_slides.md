# Sentiment Analysis with HRMs and Mixture Models
## Presentation Slides Content & Visual Schema

---

## SLIDE 1: Title Slide

### Content:
**Title:** Sentiment Analysis using Hierarchical Reasoning Models and Mixture-of-Experts Architecture

**Subtitle:** Combining HRMs with ML/DL Models for Enhanced Robustness and Interpretability

**Author:** Rohan Pratap Reddy Ravula  
**Program:** Master of Science in Data Science  
**Institution:** School of Computing and Data Science, Wentworth Institute of Technology  
**Contact:** ravular@wit.edu

### Visual Schema:
- **Layout:** Center-aligned, gradient background (deep blue to purple)
- **Typography:** Large bold title (48pt), subtitle (24pt), author info (18pt)
- **Graphics:** Abstract network/neural connections pattern in background

### Image Suggestions:
- Background: `https://images.unsplash.com/photo-1639322537228-f710d846310a` (AI/ML abstract visualization)
- Icon overlay: Brain + neural network hybrid icon
- Color scheme: Blues (#1e3a8a → #6366f1) with white text

---

## SLIDE 2: Project Overview

### Content:
**Research Focus:**
- Investigating whether Hierarchical Reasoning Models (HRMs) improve sentiment analysis when combined with ML/DL approaches
- Building a mixture-of-experts pipeline with sequential stacking framework

**Key Innovation:**
- HRM modules + Classic ML (Logistic Regression, SVM) + Deep Learning (BiLSTM, BERT, RoBERTa, DistilBERT)
- Learned meta-learner/gating network combines predictions

**Target Improvement:**
- 3-5 point increase in macro-F1 score
- Enhanced robustness to noisy/sarcastic text
- Improved cross-domain generalization

### Visual Schema:
- **Layout:** Split screen - text left (60%), visual right (40%)
- **Components:** 
  - Bullet points with icons
  - Simple architecture diagram showing model combination

### Image/Diagram:
```
[Traditional ML] ──┐
[Deep Learning]  ──┤──→ [Meta-Learner/Gating] ──→ [Final Prediction]
[HRM Module]    ──┘
```
- Reference: `https://miro.medium.com/max/1400/1*ZPPw2FJsWGoLE-rYE4YAKw.png` (ensemble learning concept)

---

## SLIDE 3: The Problem Statement

### Content:
**Challenge:**
❌ Single models struggle with:
- Noisy social media text
- Sarcasm and irony detection
- Cross-domain generalization
- Lack of interpretability

**Current Limitations:**
- Traditional ML: Misses contextual subtleties
- Deep Learning: Black box, no explainability
- Domain-specific models: Poor transfer learning

**Desired Solution:**
✓ High accuracy across domains
✓ Robust to noise and sarcasm
✓ Interpretable decision-making
✓ Efficient resource utilization

### Visual Schema:
- **Layout:** Problem-Solution format with visual contrast
- **Left side:** Red/orange theme showing problems
- **Right side:** Green theme showing solutions
- **Center:** Large arrow/transformation symbol

### Image Suggestions:
- Problem icons: `https://cdn-icons-png.flaticon.com/512/5974/5974633.png` (confusion icon)
- Solution icons: `https://cdn-icons-png.flaticon.com/512/190/190411.png` (checkmark)
- Background: Subtle gradient transition from red to green

---

## SLIDE 4: Thesis Statement

### Content:
**Core Hypothesis:**
Combining Hierarchical Reasoning Models (HRMs) with traditional ML and deep learning methods in a mixture-of-experts framework significantly improves sentiment analysis performance.

**Why This Matters:**

| Model Type | Strength | Limitation |
|------------|----------|------------|
| **Classical ML** | Efficient, baseline features | Misses context |
| **Transformers** | Contextual understanding | Black box, computationally expensive |
| **HRMs** | Interpretable reasoning | Unexplored in sentiment analysis |

**Expected Outcomes:**
- **3-5 point macro-F1 improvement** over best single model
- **Better cross-domain transfer** (Twitter → Amazon reviews)
- **Enhanced interpretability** with reasoning chains

### Visual Schema:
- **Layout:** Three-column comparison + hypothesis banner at top
- **Style:** Clean table with color coding
- **Bottom:** Metrics visualization with target indicators

### Image/Formula:
```
Performance_Ensemble = f(HRM_contribution, ML_baseline, DL_context)
                      > max(P_HRM, P_ML, P_DL)
```
- Venn diagram showing model overlap: `https://miro.medium.com/max/1400/1*8rNMMmVd-rPZv4WFLUKm5w.png`

---

## SLIDE 5: Key Claims (Part 1)

### Content:
**1. Performance Enhancement**
- 3-5 point macro-F1 improvement (statistically significant)
- Validated across multiple seeds and benchmarks

**2. Robustness Improvement**
- Better handling of sarcastic tweets
- Resilient to noisy product reviews
- Superior cross-domain performance

**3. Interpretability Gains**
- Clear reasoning paths from HRM components
- Traceable decision-making process
- Debugging-friendly architecture

**4. Complementary Expert Value**
- Gating network learns expert specialization
- Outperforms simple averaging
- Dynamic model selection per input

### Visual Schema:
- **Layout:** 2x2 grid with numbered sections
- **Icons:** Performance chart, shield, magnifying glass, network
- **Style:** Cards with icons, short descriptions, and mini-graphs

### Image Suggestions:
- Performance: `https://cdn-icons-png.flaticon.com/512/2906/2906229.png` (upward trend)
- Robustness: `https://cdn-icons-png.flaticon.com/512/2913/2913133.png` (shield)
- Interpretability: `https://cdn-icons-png.flaticon.com/512/2920/2920277.png` (search/analyze)
- Expert network: `https://cdn-icons-png.flaticon.com/512/3281/3281307.png` (network nodes)

---

## SLIDE 6: Key Claims (Part 2)

### Content:
**5. Data Efficiency**
- Larger gains with limited training data (10% vs 100%)
- Important for low-resource domains

**6. Gating Mechanism Superiority**
- Smart routing beats fixed stacking
- Input-dependent expert activation

**7. Cross-Domain Generalization**
- Train: Twitter sentiment → Test: Amazon reviews
- Ensemble maintains better performance

**8. Nuanced Context Understanding**
- Sarcasm detection improvement
- Irony and context-dependent sentiment
- Negation handling

### Visual Schema:
- **Layout:** 2x2 grid (continuation of previous slide)
- **Style:** Consistent with Slide 5
- **Emphasis:** Data charts showing comparative performance

### Image/Diagram:
```
Training Data Size vs Performance Gain
     │
 F1  │     ╱── Ensemble
     │    ╱
     │   ╱ ── Single Model
     │  ╱
     └─────────────────
       10%    50%    100%
```
- Reference: Line chart showing data efficiency

---

## SLIDE 7: Research Objectives

### Content:
**Primary Objectives:**

1. **Architecture Development**
   - Build modular mixture-of-experts pipeline
   - Components: HRM + ML (LogReg, SVM) + DL (BiLSTM, BERT, RoBERTa)
   - Swappable configuration for testing

2. **Performance Benchmark**
   - Target: ≥3-5 macro-F1 improvement
   - Statistical validation (paired tests, multiple seeds)

3. **Robustness Validation**
   - Domain shift testing
   - Noisy input handling (URLs, emojis, informal text)

4. **Interpretability Framework**
   - Extract HRM reasoning artifacts
   - Document decision paths

### Visual Schema:
- **Layout:** Vertical flow with 4 main sections
- **Design:** Timeline/roadmap style with checkpoints
- **Icons:** Target, architecture, test tube, documentation

### Image Suggestions:
- Roadmap template: `https://cdn-icons-png.flaticon.com/512/2920/2920349.png`
- Architecture: `https://cdn-icons-png.flaticon.com/512/3281/3281289.png`

---

## SLIDE 8: Methodology Overview

### Content:
**Pipeline Architecture:**

**Phase 1: Preprocessing**
- Deduplication
- URL/emoji handling
- Tokenization
- Stratified splits

**Phase 2: Expert Models**
- **ML Experts:** Logistic Regression, Linear SVM (TF-IDF features)
- **DL Experts:** BiLSTM/GRU, Transformer encoders (BERT, RoBERTa, DistilBERT)
- **HRM Experts:** Hierarchical reasoning modules

**Phase 3: Combination**
- Stacking meta-learner (out-of-fold predictions)
- Gating network (mixture-of-experts)

**Phase 4: Validation**
- k-fold CV on training
- Held-out test set
- Cross-domain evaluation

### Visual Schema:
- **Layout:** Horizontal flow diagram with 4 phases
- **Style:** Pipeline visualization with arrows
- **Colors:** Different color for each phase

### Diagram:
```
[Raw Text] → [Preprocessing] → [Expert Models] → [Combiner] → [Prediction]
                   ↓                  ↓              ↓
              [Clean Data]     [ML/DL/HRM]    [Meta-Learner]
```

### Image Reference:
- Pipeline: `https://miro.medium.com/max/1400/1*oB3S5yHHhvougJkPXuc8og.png` (ML pipeline)

---

## SLIDE 9: Architecture Deep Dive

### Content:
**Mixture-of-Experts Framework:**

```
                    Input Text
                        │
        ┌───────────────┼───────────────┐
        │               │               │
    [ML Expert]    [DL Expert]    [HRM Expert]
    LogReg/SVM     BERT/RoBERTa   Hierarchical
    TF-IDF         Contextual     Reasoning
                   Embeddings     Layers
        │               │               │
        └───────────────┼───────────────┘
                        │
              [Gating Network]
               (Dynamic Weighting)
                        │
                [Final Prediction]
                with Confidence
```

**Key Features:**
- **Sparse Activation:** Not all experts for all inputs
- **Learned Gating:** Network decides expert weights
- **Interpretable Path:** HRM provides reasoning trace

### Visual Schema:
- **Layout:** Large central architecture diagram
- **Style:** Clean boxes and arrows with color coding
- **Legend:** Color-coded expert types

### Image Suggestions:
- Use custom architecture diagram tool or draw.io export
- Reference: `https://miro.medium.com/max/1400/1*ZPPw2FJsWGoLE-rYE4YAKw.png`

---

## SLIDE 10: Dataset Overview - Part 1

### Content:
**Datasets for Training & Evaluation:**

**1. Sentiment140 (Twitter)**
- **Size:** 1.6M tweets
- **Type:** Binary sentiment (positive/negative)
- **Source:** Distant supervision via emoticons
- **Characteristics:** Short, noisy, informal, emojis, URLs
- **Use Case:** Social media monitoring

**2. IMDB Reviews**
- **Size:** 50K movie reviews
- **Type:** Binary sentiment (positive/negative)
- **Characteristics:** Longer texts, formal writing
- **Split:** 25K train, 25K test
- **Use Case:** Long-form content analysis

### Visual Schema:
- **Layout:** Two-column comparison
- **Design:** Dataset cards with key statistics
- **Icons:** Twitter bird for Sentiment140, film icon for IMDB

### Statistics Box (from Python script):
```
┌─────────────────────────────────┐
│ Run dataset_explorer.py to get:│
│ - Sample counts                 │
│ - Class distribution            │
│ - Text length statistics        │
│ - Visualization plots           │
└─────────────────────────────────┘
```

### Image Suggestions:
- Twitter logo: `https://cdn-icons-png.flaticon.com/512/733/733579.png`
- IMDB/Film: `https://cdn-icons-png.flaticon.com/512/3031/3031698.png`
- Include actual distribution plots from `dataset_explorer.py` output

---

## SLIDE 11: Dataset Overview - Part 2

### Content:
**3. Amazon Reviews**
- **Size:** Multi-million reviews (using subset)
- **Type:** Multi-class (1-5 stars → 3-class: neg/neu/pos)
- **Characteristics:** Product-specific, varied domains
- **Use Case:** E-commerce feedback analysis

**4. TweetEval**
- **Size:** Curated Twitter dataset
- **Type:** Multi-task benchmark (sentiment track)
- **Characteristics:** Emotion, offensive, irony subtasks
- **Use Case:** Comprehensive Twitter understanding

**Cross-Domain Testing:**
- Train on Sentiment140 → Test on Amazon Reviews
- Measure domain adaptation capability
- Evaluate robustness to distribution shift

### Visual Schema:
- **Layout:** Two dataset cards + cross-domain flow diagram
- **Bottom:** Cross-domain testing visualization
- **Style:** Consistent with Slide 10

### Diagram:
```
[Sentiment140] ──train──→ [Model] ──test──→ [Amazon Reviews]
                                              ↓
                                        Performance Drop?
```

### Code Reference Box:
```python
# Generate dataset statistics:
python dataset_explorer.py

# Outputs:
# - sentiment_dist.png
# - text_length.png  
# - wordcount_by_sentiment.png
```

### Image Suggestions:
- Amazon: `https://cdn-icons-png.flaticon.com/512/825/825464.png`
- Tweet: `https://cdn-icons-png.flaticon.com/512/733/733635.png`
- Transfer learning diagram

---

## SLIDE 12: Literature Review Highlights

### Content:
**Key Research Areas Examined: (57 peer-reviewed sources)**

**1. Hybrid Architectures**
- CNN-LSTM combinations show promise (Dang et al., 2021)
- Complementary strengths: local features + sequential context

**2. Hierarchical Reasoning**
- Chain-of-Thought prompting (Wei et al., 2022)
- Tree of Thoughts (Yao et al., 2023)
- HRMs for explicit reasoning (Wang et al., 2025) ✨ **NEW**

**3. Ensemble Methods**
- Mixture-of-Experts (Shazeer et al., 2017)
- Stacking meta-learners (Muhammad et al., 2023)
- Diversity drives performance (Aydoğan & Akcayol, 2024)

**4. Persistent Challenges**
- Sarcasm detection remains hard
- Domain shift degrades performance
- Interpretability gap in deep learning

### Visual Schema:
- **Layout:** Four quadrants with research themes
- **Design:** Academic-style with citations
- **Highlight:** HRMs as the novel contribution

### Image/Formula:
```
Research Evolution:
Lexicon → ML → DL → Transformers → HRM+Ensemble
 2004    2008   2014    2019          2025 ✨
```

### Image Suggestions:
- Timeline visualization
- Research paper stack icon: `https://cdn-icons-png.flaticon.com/512/3074/3074767.png`

---

## SLIDE 13: Gap Analysis

### Content:
**Identified Gaps in Current Research:**

| Gap | Current State | Our Approach |
|-----|---------------|--------------|
| **HRM in Sentiment** | Not explored | First systematic integration |
| **MoE with Gating** | Rare in NLP | Learned dynamic gating |
| **Cross-Domain Robustness** | Limited | HRM + domain experts |
| **Interpretability** | Black boxes | Explicit reasoning chains |
| **Sarcasm Detection** | Implicit learning | HRM pragmatic reasoning |

**Why This Matters:**
- Real-world applications need robust, interpretable models
- Single-model approaches hit performance ceiling
- Combining diverse experts with HRM reasoning is unexplored territory

### Visual Schema:
- **Layout:** Comparison table with traffic light colors
- **Design:** Red (gap) → Yellow (current) → Green (solution)
- **Style:** Professional research summary

### Image Suggestions:
- Gap/bridge metaphor: `https://images.unsplash.com/photo-1527689368864-3a821dbccc34`
- Research opportunity icon

---

## SLIDE 14: Expected Contributions & Impact

### Content:
**Theoretical Contributions:**
✓ Integration framework for HRMs with ensemble learning
✓ Empirical data on model complementarity
✓ Interpretable ensemble decision framework

**Practical Contributions:**
✓ Production-ready implementation
✓ Comprehensive benchmarking across datasets
✓ Model selection guidelines for resource constraints
✓ Open-source reproducible codebase

**Impact Areas:**

**Academic:**
- Advance ensemble methods for NLP
- Understanding neural architecture complementarity

**Industrial:**
- Better customer feedback analysis at scale
- Improved social media monitoring tools
- Enhanced product review analytics

**Societal:**
- Transparent AI decision-making
- Trustworthy sentiment analysis
- Responsible AI deployment

### Visual Schema:
- **Layout:** Three sections (Theoretical/Practical/Impact)
- **Design:** Impact ripple visualization
- **Icons:** Academic (book), Industry (building), Society (people)

### Image Suggestions:
- Impact ripples: `https://images.unsplash.com/photo-1558494949-ef010cbdcc31`
- Contribution tree diagram

---

## SLIDE 15: Project Plan & Next Steps

### Content:
**Deliverables:**
✅ Code repository (modular, documented)
✅ Experiment logs (wandb/MLflow)
✅ Trained checkpoints (where feasible)
✅ Research report with analyses
✅ Demo notebook (reproducible examples)

**Evaluation Metrics:**
- **Primary:** Macro-F1 score
- **Secondary:** Accuracy, AUROC
- **Target:** ≥3-5 point improvement (statistically significant)

**Ablation Studies:**
1. HRM contribution analysis
2. Combiner comparison (averaging vs stacking vs gating)
3. Model size trade-offs
4. Data efficiency experiments

**Risk Mitigation:**
- Use DistilBERT for compute efficiency
- Robust preprocessing for noisy data
- Proper OOF stacking to prevent overfitting
- Dataset caching for API limitations

### Visual Schema:
- **Layout:** Project timeline with milestones
- **Design:** Gantt-style chart or roadmap
- **Colors:** Progress indicators (green = complete, blue = in progress)

### Diagram:
```
Phase 1: Data Prep → Phase 2: Model Training → Phase 3: Evaluation → Phase 4: Analysis
  (Week 1-2)            (Week 3-6)              (Week 7-8)          (Week 9-10)
```

### Image Suggestions:
- Project management: `https://cdn-icons-png.flaticon.com/512/3281/3281301.png`
- Timeline template

---

## Additional Resources

### Image Sources Summary:
1. **Unsplash** (free, high-quality): https://unsplash.com/
   - Search terms: "artificial intelligence", "data visualization", "network"
2. **Flaticon** (icons, free with attribution): https://www.flaticon.com/
3. **Canva** (design templates): https://www.canva.com/
4. **Draw.io** (architecture diagrams): https://app.diagrams.net/

### Color Palette Recommendations:
- **Primary:** #1e3a8a (Deep Blue) - Trust, Intelligence
- **Secondary:** #6366f1 (Indigo) - Innovation, Technology  
- **Accent:** #10b981 (Emerald) - Success, Growth
- **Warning:** #f59e0b (Amber) - Attention, Caution
- **Background:** #f8fafc (Light Gray) - Clean, Professional

### Font Recommendations:
- **Headings:** Montserrat Bold / Poppins Bold
- **Body:** Open Sans / Roboto
- **Code:** Fira Code / Source Code Pro

### Presentation Software:
- **PowerPoint:** Use master slides for consistency
- **Google Slides:** Collaborative, cloud-based
- **Reveal.js:** HTML-based, code-friendly
- **Beamer (LaTeX):** Academic presentations

---

## Notes for Presenter:

1. **Slide 10-11:** Before presenting, run `dataset_explorer.py` to generate actual statistics and embed those visualizations

2. **Interactive Elements:** Consider adding QR codes linking to:
   - GitHub repository
   - Live demo notebook
   - Supplementary materials

3. **Backup Slides:** Prepare additional slides for:
   - Detailed architecture specifications
   - Extended results tables
   - Additional ablation studies
   - Q&A common questions

4. **Timing:** Allocate ~1-2 minutes per slide for 15-20 minute presentation

5. **Emphasis Points:**
   - The novelty of HRM integration (Slide 4, 12, 13)
   - Concrete performance targets (Slide 5, 15)
   - Real-world applicability (Slide 14)


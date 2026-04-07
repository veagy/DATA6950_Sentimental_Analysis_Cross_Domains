# DATA-6900 Capstone I: Sentiment Analysis with HRMs and Mixture Models

* [cite_start]**Author:** Rohan Pratap Reddy Ravula [cite: 3, 551, 705]
* [cite_start]**Program:** Master of Science in Data Science [cite: 3, 551, 705]
* [cite_start]**Institution:** School of Computing and Data Science, Wentworth Institute of Technology, Boston, MA [cite: 3, 551, 705]
* [cite_start]**Contact:** ravular@wit.edu [cite: 3, 552, 706]

---

## 1. Project Overview

[cite_start]**Project Title:** Doing Sentimental Analysis using mixture models based on sequential stacking of HRMs (Hierarchical Reasoning Models), common machine-learning and deep-learning models architecture. [cite: 4, 553, 707]

### Abstract

[cite_start]This project investigates whether hierarchical reasoning models (HRMs) can improve sentiment analysis when combined with conventional machine learning and deep learning approaches in a mixed-sequential stacking framework. [cite: 555] [cite_start]I will build a mixture-of-experts pipeline where HRM modules, classic ML models (e.g., logistic regression, SVM), and deep architectures (e.g., BiLSTM, Transformer-based encoders like BERT/RoBERTa) act as experts. [cite: 556] [cite_start]A learned meta-learner or gating network will combine their predictions. [cite: 557] [cite_start]Using public datasets (e.g., Sentiment140, IMDB, Amazon Reviews, TweetEval), I will evaluate classification performance across binary and multi-class sentiment settings. [cite: 558] [cite_start]The study includes strong baselines, ablations isolating HRM contributions, and domain-shift tests (train on one dataset, test on another). [cite: 559] [cite_start]I expect the stacked mixture to yield higher macro-F1 and better robustness to noisy or sarcastic text than single models, with HRMs contributing interpretable reasoning features. [cite: 560] [cite_start]Deliverables include a reproducible codebase, a dataset card, a report with results and analyses, and a lightweight demo notebook. [cite: 561] [cite_start]This work aims to clarify when and how HRM-style reasoning adds value to standard NLP sentiment pipelines. [cite: 562]

### Problem Statement

* [cite_start]**Problem:** Achieving high-accuracy, robust, and interpretable sentiment analysis remains challenging across domains and noisy text. [cite: 564]
* [cite_start]**Impact:** Better sentiment models support product feedback mining, social media monitoring, and customer support analytics. [cite: 565]
* [cite_start]**Goal:** Design and evaluate a mixed-sequential stacking (mixture-of-experts) approach that integrates HRMs with ML/DL models to improve sentiment accuracy and robustness. [cite: 566]

### Thesis Statement

[cite_start]This research looks at whether combining Hierarchical Reasoning Models (HRMs) with traditional machine learning and deep learning methods can improve sentiment analysis in meaningful ways. [cite: 709] [cite_start]The core idea is to build a mixture-of-experts system where different types of models work together: HRMs bring interpretable reasoning, classical ML algorithms offer efficiency and solid baseline features, while Transformer-based encoders handle contextual nuances. [cite: 710]

[cite_start]The problem with using just one model is that it only captures part of what's going on in natural language. [cite: 711] [cite_start]A logistic regression model might miss contextual subtleties, while a BERT model, though powerful, can be a black box. [cite: 712] [cite_start]By bringing these different approaches together in an ensemble, each model can contribute to what it does best. [cite: 713]

[cite_start]The goal is to show improvements of at least 3-5 points in macro-F1 score compared to the best single model, while also handling tough cases like noisy text, domain shifts (training on Twitter, testing on product reviews), and tricky linguistic stuff like sarcasm. [cite: 714] [cite_start]Through ablation studies and cross-domain testing, this thesis will demonstrate not just that the mixture approach works, but also when and why HRMs add value to standard NLP pipelines. [cite: 715] [cite_start]Beyond the performance gains, this work aims to provide practical insights for deploying sentiment analysis systems in real-world scenarios where interpretability and robustness matter. [cite: 716]

### Key Claims

1.  [cite_start]**Performance Enhancement:** The mixture-of-experts setup combining HRMs with ML/DL models should beat single-model baselines by at least 3-5 points in macro-F1 on standard sentiment benchmarks. [cite: 718] [cite_start]This improvement needs to be statistically significant, not just a fluke. [cite: 719]
2.  [cite_start]**Robustness Improvement:** When dealing with messy real-world text - sarcastic tweets, noisy product reviews, cross-domain scenarios - the ensemble should hold up better than individual models. [cite: 720] [cite_start]This is crucial for practical applications. [cite: 721]
3.  [cite_start]**Interpretability Gains:** Unlike pure deep learning models that act like black boxes, the HRM components should provide clear reasoning paths. [cite: 722] [cite_start]You can see why the model made a particular decision, which matters for trust and debugging. [cite: 723]
4.  [cite_start]**Complementary Expert Value:** Different architectures capture different patterns in sentiment. [cite: 724] [cite_start]A gating network or meta-learner that intelligently combines these experts should outperform simple averaging. [cite: 725] [cite_start]The models aren't just redundant - they're complementary. [cite: 726]
5.  [cite_start]**Data Efficiency:** When training data is limited (say, 10% vs. 100%), the HRM-enhanced approach should show bigger gains. [cite: 727] [cite_start]This matters for domains where labeled data is expensive or scarce. [cite: 728]
6.  [cite_start]**Gating Mechanism Superiority:** A smart gating network that learns to route different inputs to the right expert models should beat both simple averaging and fixed stacking methods. [cite: 731] [cite_start]Not all inputs need all experts equally. [cite: 732]
7.  [cite_start]**Cross-Domain Generalization:** Train on Twitter sentiment, test on Amazon reviews - the ensemble should maintain better performance than single models. [cite: 732] [cite_start]Real applications often need this kind of flexibility. [cite: 733]
8.  [cite_start]**Nuanced Context Understanding:** Things like sarcasm, irony, and context-dependent sentiment trip up traditional classifiers. [cite: 734] [cite_start]The HRM-enhanced models should handle these cases more reliably. [cite: 735]

---

## 2. Research Objectives and Questions

### Primary Objectives

1.  [cite_start]**Architecture Development:** Build a working mixture-of-experts pipeline that brings together HRM modules, classical ML models (Logistic Regression, SVM), and deep learning architectures (BiLSTM, BERT, ROBERTa, DistilBERT). [cite: 738] [cite_start]The system needs to be modular enough to swap components and test different configurations. [cite: 739]
2.  [cite_start]**Performance Benchmark:** Hit that 3-5-point macro-F1 improvement target across multiple datasets. [cite: 740] [cite_start]Use proper statistical testing (paired tests across multiple seeds) to make sure the gains are real and not just variance. [cite: 741]
3.  [cite_start]**Robustness Validation:** Test the system under stress - domain shifts, noisy inputs with URLS and emojis, informal text. [cite: 742] [cite_start]Show that improvements aren't just on clean benchmark data but also in realistic conditions. [cite: 743]
4.  [cite_start]**Interpretability Framework:** Extract and document the reasoning artifacts from HRM components. [cite: 744] [cite_start]Make it clear how the model arrives at its predictions, not just that it works. [cite: 745]

### Secondary Objectives

5.  **Comprehensive Ablation Analysis:** Figure out what's actually helping. [cite_start]Remove HRM - does performance drop? [cite: 747] Try different combinations (averaging vs. stacking vs. gating). [cite_start]Look at per-class performance to see if certain models specialize. [cite: 748]
6.  **Multi-Dataset Evaluation:** Don't just test on one dataset. [cite_start]Use Sentiment140 (Twitter), IMDB Reviews, Amazon Reviews, and TweetEval. [cite: 749] [cite_start]Cover both binary and multi-class sentiment scenarios. [cite: 750]
7.  **Computational Efficiency Analysis:** Compare DistilBERT vs. BERT. [cite_start]How much accuracy do you gain for the extra computational cost? [cite: 751] [cite_start]This matters for deployment decisions. [cite: 752]
8.  [cite_start]**Reproducibility Standards:** Deliver clean, documented code with experiment tracking (wandb or MLflow). [cite: 753] [cite_start]Including trained checkpoints where feasible, dataset cards, and demo notebooks so others can use and build on this work. [cite: 754]

### Research Questions

1.  [cite_start]When does HRM-enhanced stacking outperform single Transformers? [cite: 579]
2.  [cite_start]Does a gating network beat simple averaging/stacking? [cite: 580]
3.  [cite_start]How does performance transfer across domains/datasets? [cite: 581]

---

## 3. Methodology

### Preprocessing
[cite_start]Deduplication, URL/emoji handling, tokenization; stratified splits; label remapping for multi-class where needed. [cite: 616]

### Experts
1.  [cite_start]**ML:** Logistic Regression / Linear SVM [cite: 618]
2.  **DL:** BILSTM/GRU; [cite_start]Transformer encoder [cite: 619]
3.  [cite_start]**HRM:** Hierarchical reasoning features/modules feeding a small classifier [cite: 620]

### Combiner
1.  [cite_start]Stacking meta-learner on out-of-fold predictions. [cite: 621]
2.  [cite_start]Gating network for mixture-of-experts [cite: 622]

### Validation
1.  [cite_start]k-fold CV on training [cite: 624]
2.  [cite_start]held-out test [cite: 626]
3.  [cite_start]cross-domain test [cite: 627]

### Stack
[cite_start]Python, pandas, scikit-learn, PyTorch, Hugging Face Transformers, wandb/MLflow (tracking), Jupyter [cite: 629]

---

## 4. Literature Review

### Abstract
[cite_start]This review explores how hierarchical reasoning models (HRMs) might transform sentiment analysis when combined with traditional machine learning and deep learning approaches. [cite: 6] [cite_start]Drawing from 57 peer-reviewed sources across NLP, machine learning, and deep learning research, I examine several interconnected themes: how foundation models have evolved, what reasoning capabilities emerge in large language models, various ensemble and meta-learning approaches, transformer architectures, and the ongoing challenges in sentiment analysis. [cite: 7] [cite_start]What becomes clear is that researchers haven't adequately explored how HRMs could integrate with conventional ML/DL methods for sentiment tasks. [cite: 8] [cite_start]This gap matters because mixture-of-experts frameworks that combine different model types could potentially improve accuracy, work better across domains, and provide more interpretable results. [cite: 9] [cite_start]The synthesis here lays groundwork for investigating these hybrid architectures in practical sentiment analysis applications. [cite: 10]

### Introduction
[cite_start]Sentiment analysis has become one of NLP's most active research areas, and for good reasons. [cite: 12] [cite_start]Companies need to monitor social media, analyze customer feedback, process product reviews, and track brand reputation all at scale. [cite: 13] [cite_start]Yet despite impressive progress with deep learning architectures, we still struggle with fundamental issues. [cite: 14] [cite_start]Models that work well in one domain often fail in another. [cite: 15] [cite_start]Noisy text from social media breaks many systems. [cite: 15] [cite_start]And perhaps most frustratingly, even when models get the right answer, we often can't explain why. [cite: 16] [cite_start]This is where hierarchical reasoning models and hybrid ensemble approaches become interesting - they might offer ways forward on these persistent problems by strategically combining different model types. [cite: 17]

I've organized this review around five main themes. [cite_start]First, I examine hybrid and mixed model architectures - how combining CNNs with LSTMs, or traditional ML with deep learning, creates systems with complementary strengths. [cite: 18] [cite_start]Second, I look at hierarchical reasoning and chain-of-thought models, which represent newer ways of thinking about how models can perform explicit reasoning. [cite: 19] [cite_start]Third, I cover ensemble learning and model stacking in depth, exploring sophisticated ways to combine different model types. [cite: 20] [cite_start]Fourth, I dive into sentiment analysis specifically - its evolution, persistent challenges, and robustness issues. [cite: 21] [cite_start]Finally, I will discuss training methodologies and optimization techniques that make these hybrid systems practical to deploy. [cite: 22] [cite_start]The goal is to see whether combining HRMs with traditional ML and DL approaches in a mixture-of-experts setup could move the needle on sentiment analysis performance, especially for challenging cases like sarcasm or cross-domain applications. [cite: 23]

### 4.1 Hybrid and Mixed Model Architectures

**CNN-LSTM Hybrid Models**
[cite_start]Hybrid models combining CNNs and LSTMs have emerged as powerful architectures for sentiment analysis. [cite: 28] [cite_start]CNNs excel at extracting local features and patterns from text, while LSTMs capture sequential dependencies and long-range context. [cite: 29] [cite_start]Combining these complementary strengths makes intuitive sense, and the empirical results back this up. [cite: 30] Dang et al. (2021) [cite_start]systematically investigated different configurations of CNN-LSTM hybrids, testing both CNN-followed-by-LSTM and LSTM-followed-by-CNN architectures across multiple datasets. [cite: 31] [cite_start]Their results showed that the order matters: CNN first works better for extracting local sentiment-bearing features before feeding them to LSTMs for sequence modeling. [cite: 32] [cite_start]What's interesting is that the hybrid consistently outperformed either architecture alone, suggesting genuine complementarity rather than just adding capacity. [cite: 33]

Ezzat et al. (2024) [cite_start]took this further with COVID-19 tweet sentiment analysis, adding class balancing techniques to handle imbalanced data. [cite: 34] [cite_start]Their hybrid CNN-LSTM model addressed a practical problem: real-world sentiment data rarely has evenly distributed classes. [cite: 35] [cite_start]By combining architectural innovation with sampling strategies, they achieved more robust performance than standard approaches. [cite: 36] [cite_start]This matters because class imbalance is pervasive in sentiment analysis applications. [cite: 37]

**Ensemble Deep Learning Approaches**
[cite_start]Moving beyond single hybrid architectures, ensemble methods combine multiple complete models to leverage their collective intelligence. [cite: 39] [cite_start]The idea is straightforward but powerful: different models make different mistakes, and aggregating predictions can reduce errors. [cite: 40] [cite_start]Alharbi and Lee (2021) developed an ensemble deep learning model specifically for social media sentiment analysis. [cite: 41] [cite_start]They combined multiple neural architectures - each with different inductive biases - and showed significant improvements over single-model baselines. [cite: 42] [cite_start]Social media text is particularly challenging due to noise, slang, and non-standard grammar, so having multiple perspectives helps. [cite: 43]

[cite_start]Hassan and Mahmood (2017) explored ensemble methods for multilingual sentiment analysis, comparing traditional ML with hybrid DL approaches across different languages. [cite: 44] [cite_start]Their key finding: hybrid models that combine CNNs and LSTMs work better than pure ML or single DL architecture when dealing with linguistic diversity. [cite: 45] [cite_start]Context matters differently across languages, and hybrid models seem better equipped to handle these variations. [cite: 46]

**Stacking and Meta-Learning Approaches**
[cite_start]Stacking takes ensembles a step further by training a meta-learner to optimally combine base models. [cite: 48] [cite_start]Instead of simple averaging or voting, the meta-learner learns which models to trust under what circumstances. [cite: 49] Muhammad et al. (2023) [cite_start]applied stacking to customer review sentiment analysis, combining traditional ML algorithms (SVM, logistic regression) with DL models (CNN, LSTM). [cite: 50] [cite_start]Their meta-learner - itself a simple logistic regression - learned to weight predictions based on input characteristics. [cite: 51] [cite_start]The stacked model outperformed any individual base model, validating the meta-learning approach. [cite: 52]

[cite_start]Aydoğan and Akcayol (2024) investigated stacking for large-scale Turkish sentiment analysis, comparing various ML models as both base learners and meta-learners. [cite: 53] [cite_start]They found that diverse base models (combining tree-based, linear, and neural approaches) produced better stacking results than homogeneous ensembles. [cite: 54] [cite_start]Diversity matters: it's not just about having multiple models but having models that bring different perspectives. [cite: 55]

### 4.2 Hierarchical Reasoning and Chain-of-Thought Models

**Chain-of-Thought Prompting**
Here's where things get interesting for sentiment analysis. [cite_start]Wei and colleagues discovered something clever in 2022: if you ask language models to show their work - to generate intermediate reasoning steps - they perform much better on complex tasks. [cite: 60] [cite_start]Arithmetic, commonsense reasoning, symbolic logic all improved when models "thought out loud." [cite: 61] [cite_start]For sentiment analysis, this matters because we don't just want correct classifications; [cite: 62] [cite_start]we want to understand why a model made a particular decision (Wei et al., 2022). [cite: 63]

What Kojima et al. (2022) [cite_start]found next was even more surprising. [cite: 64] [cite_start]Just adding "Let's think step by step" to your prompt unlocks reasoning capabilities. [cite: 65] [cite_start]No special training, no task-specific examples needed. [cite: 65] [cite_start]The reasoning ability was already there, latent in the model, waiting for the right trigger. [cite: 66] [cite_start]This suggests that large language models have learned more general reasoning patterns than we initially realized. [cite: 67]

**Advanced Reasoning Frameworks**
Yao et al. (2023a) [cite_start]proposed ReAct, which synergizes reasoning and acting in language models by interleaving thought, action, and observation steps. [cite: 69] [cite_start]This framework demonstrated improvements in decision-making tasks and multi-step problem solving. [cite: 70] [cite_start]The ability to combine internal reasoning with external actions provides a model for how sentiment analysis systems might integrate contextual information retrieval with classification decisions. [cite: 71] Yao et al. (2023b) [cite_start]further advanced reasoning capabilities with Tree of Thoughts (ToT), which enables deliberate exploration of multiple reasoning paths. [cite: 72] [cite_start]By maintaining and evaluating multiple solution trajectories, ToT achieves better performance on tasks requiring strategic lookahead and backtracking. [cite: 73] [cite_start]This approach is relevant to sentiment analysis of complex texts where initial interpretations may need revision based on later context. [cite: 74]

**Self-Consistency and Program-Aided Methods**
Wang et al. (2023) [cite_start]introduced self-consistency, a decoding strategy that samples multiple reasoning paths and selects the most consistent answer. [cite: 76] [cite_start]This approach improved reasoning accuracy by aggregating diverse solution paths, effectively creating an ensemble of reasoning chains. [cite: 77] [cite_start]The technique is particularly promising for sentiment analysis where ambiguous or sarcastic texts might benefit from consideration of multiple interpretations. [cite: 78]

Gao et al. (2023) [cite_start]proposed Program-Aided Language models (PAL), which delegate computational tasks to programmatic runtime. [cite: 79] [cite_start]By separating reasoning (language model) from computation (code execution), PAL achieved higher accuracy on quantitative reasoning tasks. [cite: 80] [cite_start]This modular approach aligns with mixture-of-experts architectures where different components handle different aspects of the classification task. [cite: 81]

**Hierarchical Reasoning Models**
The most recent development here is Hierarchical Reasoning Models from Wang et al. (2025)[cite_start]. [cite: 83] [cite_start]HRMs organize reasoning into hierarchical layers, breaking complex tasks into subtasks at different levels of abstraction. [cite: 84] [cite_start]Think strategic planning at the top, tactical execution at the bottom. [cite: 85] [cite_start]For sentiment analysis, this structure maps naturally onto how humans actually process sentiment. [cite: 86] [cite_start]We operate at multiple levels simultaneously: lexical (what do individual words mean?), syntactic (how does negation change things?), semantic (what does this mean in context?), and pragmatic (is this person being sarcastic?). [cite: 87] [cite_start]HRMs could capture this multi-level processing explicitly. [cite: 88]

What really matters is interpretability. [cite_start]One of the biggest criticisms of deep learning sentiment classifiers is that they're black boxes. [cite: 89] [cite_start]HRMs, by design, show their reasoning process. [cite: 90] [cite_start]You can see what the model is thinking at each level. [cite: 90] [cite_start]For debugging, for trust, for understanding failure modes - this could be huge. [cite: 91]

### 4.3 Ensemble Learning and Model Stacking

**Theoretical Foundations of Ensemble Methods**
[cite_start]Ensemble methods rest on a beautifully simple idea: different models make different mistakes. [cite: 96] [cite_start]If you combine them strategically, the errors can cancel out while the correct predictions reinforce each other. [cite: 97] [cite_start]Dietterich laid out this logic back in 2000, and it's held up remarkably well. [cite: 98] [cite_start]For sentiment analysis, this means mixing different types of models - maybe a linear classifier, a recurrent network, and a transformer - each bringing different strengths to the table (Sagi & Rokach, 2018). [cite: 99]

The classic approaches are bagging, boosting, and stacking. [cite_start]Stacking is particularly interesting for this research because it uses a meta-learner - essentially a model that learns which models to trust under different circumstances. [cite: 100] [cite_start]Wolpert introduced this idea in 1992 as "stacked generalization." [cite: 101] [cite_start]The meta-learner looks at predictions from multiple base models and learns patterns about when each one tends to be right or wrong. [cite: 101] [cite_start]Done well, this beats simple voting or averaging by a significant margin (Zhou, 2012). [cite: 102] [cite_start]It's not just about combining models; [cite: 102] [cite_start]it's about combining them intelligently. [cite: 103]

**Mixture-of-Experts Architectures**
[cite_start]Mixture-of-Experts (MoE) architectures advance beyond traditional ensemble methods by incorporating gating networks that dynamically route inputs to specialized expert models (Jacobs et al., 1991; Jordan & Jacobs, 1994). [cite: 105] [cite_start]Contrasting with static ensemble approaches, MoE systems enable expert specialization for distinct input regions, potentially enhancing both computational efficiency and prediction accuracy (Shazeer et al., 2017). [cite: 106] [cite_start]In sentiment analysis contexts, experts could specialize along multiple dimensions: linguistic register (formal versus informal discourse), domain characteristics (product reviews versus social media posts), or affective dimensions (positive, negative, neutral polarities). [cite: 107]

[cite_start]Contemporary transformer-based MoE implementations have demonstrated scalability to massive parameter counts while preserving computational efficiency through sparse expert activation - engaging only relevant experts for each input instance (Fedus et al., 2021). [cite: 108] [cite_start]This sparse activation paradigm could enable sentiment analysis systems to maintain specialized expert modules for challenging linguistic phenomena such as sarcasm detection, negation processing, or domain-specific terminology interpretation, without incurring proportional inference cost increases. [cite: 109]

**Transfer Learning and Domain Adaptation**
[cite_start]Ensemble methodologies offer substantial advantages for mitigating domain shift - a pervasive challenge in sentiment analysis where models trained on source domains (e.g., movie reviews) frequently exhibit degraded performance on target domains (e.g., product reviews or social media content) (Blitzer et al., 2007; Glorot et al., 2011). [cite: 111] [cite_start]By aggregating models trained across diverse domains or employing varied adaptation strategies, ensemble frameworks can achieve enhanced cross-domain robustness (Rietzler et al., 2020). [cite: 112]

[cite_start]Integrating HRMs into ensemble architectures presents a novel research direction: whereas end-to-end neural models may overfit to domain-specific lexical and syntactic patterns, hierarchical reasoning models could extract more abstract sentiment reasoning mechanisms that generalize across domains (Wang et al., 2025). [cite: 113] [cite_start]This architectural complementarity - combining domain-specific pattern recognition with domain-invariant reasoning capabilities - provides strong motivation for investigating HRM-enhanced stacking approaches to achieve robust cross-domain sentiment analysis. [cite: 114]

### 4.4 Sentiment Analysis: Methods, Challenges and Robustness

**Evolution of Sentiment Analysis Approaches**
[cite_start]Sentiment analysis has gone through several distinct eras, each with its own strengths and frustrating limitations. [cite: 119] [cite_start]Early systems relied on sentiment lexicons - basically dictionaries of positive and negative words (Hu & Liu, 2004; Wilson et al., 2005). [cite: 120] Simple, interpretable, but terrible with context. [cite_start]"Not good" looks positive if you're just counting words. [cite: 121] Then came traditional machine learning. Pang et al. (2002) [cite_start]and others showed you could get better results with features like n-grams and part-of-speech tags, but someone had to engineer those features. [cite: 122] [cite_start]Deep learning changed this by learning representations automatically (Socher et al., 2013; Kim, 2014). [cite: 123] [cite_start]Great for accuracy, but now we couldn't explain what the model was doing - the "black box" problem. [cite: 124] [cite_start]More recently, hybrid approaches combining CNNs and LSTMs have shown promise (Dang et al., 2021; Hassan & Mahmood, 2017), and ensemble methods that mix multiple model types are gaining traction (Alharbi & Lee, 2021). [cite: 125] [cite_start]Each generation solved some problems while creating new ones (Ruder et al., 2019). [cite: 126]

**Challenges in Modern Sentiment Analysis**
Even with all our fancy models, sentiment analysis still struggles with some basic problems. Zhang et al. (2018) [cite_start]provides a good overview, but let me highlight what I see as the most critical issues: [cite: 128, 129]

* **Sarcasm and irony** remain brutally hard. [cite_start]"Oh great, another meeting" has positive words but negative intent. [cite: 131] [cite_start]Models need to catch this gap between what's said and what's meant, and most still fail at this regularly (Joshi et al., 2017; Ghosh et al., 2020). [cite: 132]
* [cite_start]**Context changes everything.** The word "small" is negative for a hotel room but positive for a tumor. [cite: 133] [cite_start]Long-distance dependencies make this worse - something at the start of a paragraph might flip the sentiment of something at the end (Socher et al., 2013). [cite: 134]
* **Domain shift** kills many otherwise good models. [cite_start]Train on movie reviews, test on product reviews, and watch performance craters. [cite: 135] [cite_start]Vocabulary changes, writing styles differ, and models trained on one often can't handle the other (Blitzer et al., 2007; Peng & Dredze, 2017). [cite: 136]
* [cite_start]**Social media text** is messy - misspellings, slang, emojis, grammar that would make your English teacher cry. [cite: 137] [cite_start]Standard NLP pipelines often choke on this kind of noisy input (Baldwin et al., 2013). [cite: 138]
* [cite_start]**Class imbalance** biases models toward whatever class appears most often in training data. [cite: 139] [cite_start]If 90% of your examples are positive, your model will predict "positive" way too often (He & Garcia, 2009). [cite: 140]
* [cite_start]**The interpretability problem** might be the most important for real applications. [cite: 141] [cite_start]When a model misclassifies something, can you figure out why? [cite: 142] [cite_start]Often no, and that makes debugging and improvement much harder (Doshi-Velez & Kim, 2017; Lipton, 2018). [cite: 143]

**Robustness and Generalization**
[cite_start]Here's the thing about deploying sentiment analysis in the real world: your model needs to work consistently across all kinds of messy conditions (Huang et al., 2019). [cite: 145] But recent work has shown just how brittle even our best models can be. [cite_start]Adversarial examples trip them up. [cite: 146] [cite_start]Distribution shift degrades performance. [cite: 147] [cite_start]Low-resource scenarios expose their limitations (Alzantot et al., 2018; Ren et al., 2019). [cite: 147]

Ensemble methods offer a potential solution, and the reason is straightforward. [cite_start]Different models have different failure modes; [cite: 148] [cite_start]what breaks one model might not break another. [cite: 149] [cite_start]When you combine diverse model types, you reduce sensitivity to any kind of perturbation (Karimi et al., 2020). [cite: 149] [cite_start]It's error decorrelation in action: because the models fail in different ways, their aggregate prediction tends to be more stable than any individual model (Wang et al., 2021). [cite: 150] [cite_start]Not a silver bullet, but a meaningful improvement. [cite: 151]

### 4.5 Training Methodologies and Optimization

**Efficient Training Strategies**
[cite_start]Hoffman and colleagues made an important discovery in 2022 about compute-optimal training. [cite: 156] [cite_start]Turns out many large models aren't trained enough for their size. [cite: 157] [cite_start]You'd often get better results training a smaller model on more data than training a massive model on less data. [cite: 158] Their Chinchilla work laid this out clearly. [cite_start]For sentiment analysis projects with budget constraints - which is most of them - these matters (Hoffmann et al., 2022). [cite: 159]

Earlier, Kaplan et al. (2020) [cite_start]worked out scaling laws that show predictable relationships between model size, data size, computer, and performance. [cite: 160] These aren't just academic curiosities. [cite_start]When you're planning a sentiment analysis project, these laws help you figure out where to put your resources. [cite: 161] Should you get more data? Make the model bigger? [cite_start]Train longer? [cite: 162] [cite_start]The scaling laws give you principled ways to make these trade-offs. [cite: 163]

**Parameter-Efficient Fine-Tuning**
LoRA, from Hu et al. (2022)[cite_start], changed how we think about fine-tuning large models. [cite: 165] [cite_start]Instead of updating all the model weights, LoRA learns low-rank updates. [cite: 166] [cite_start]This cuts the number of trainable parameters dramatically while keeping performance competitive. [cite: 167] [cite_start]What this means practically: you can now fine-tune big pre-trained models for sentimental analysis without needing a data center. [cite: 168] [cite_start]This has made a lot of research and applications feasible that simply weren't before. [cite: 169]

**Human Feedback and Alignment**
Ouyang et al. (2022) [cite_start]demonstrated the effectiveness of training language models to follow instructions using human feedback. [cite: 171] [cite_start]Their Instruct-GPT approach aligned model outputs with human preferences through reinforcement learning from human feedback (RLHF). [cite: 172] Christiano et al. (2017) [cite_start]laid the theoretical groundwork for learning from human preferences using deep reinforcement learning. [cite: 173] [cite_start]These techniques could be adapted for sentiment analysis to better align model predictions with human judgment, particularly for nuanced cases. [cite: 174]

### 4.6 Evaluation Benchmarks and Datasets

**General NLP Benchmarks**
Wang et al. (2018) [cite_start]introduced GLUE (General Language Understanding Evaluation), a multi-task benchmark that includes sentiment analysis among other NLP tasks. [cite: 177] Wang et al. (2019) [cite_start]extended this with SuperGLUE, providing more challenging tasks to drive further progress. [cite: 178] [cite_start]These standardized benchmarks enable rigorous comparison of different approaches. [cite: 179] Hendrycks et al. (2021) [cite_start]proposed MMLU (Massive Multitask Language Understanding), measuring knowledge across 57 subjects. [cite: 180] [cite_start]While not sentiment-specific, MMLU evaluates models' broad capabilities that may transfer to complex sentiment analysis scenarios. [cite: 181]

**Reasoning and Problem-Solving Benchmarks**
Cobbe et al. (2021) [cite_start]created a dataset for training verifiers to solve math word problems, demonstrating that verifying solutions can be easier than generating them. [cite: 183] [cite_start]This insight could apply to sentiment analysis: verifying sentiment explanations might be easier than generating initial classifications. [cite: 184] Yang et al. (2018) [cite_start]introduced HotpotQA for multi-hop question answering, requiring reasoning over multiple documents. [cite: 185] [cite_start]The multi-step reasoning required resembles the interpretive processes needed for complex sentiment analysis, particularly for longer texts where sentiment evolves. [cite: 186]

Dua et al. (2019) [cite_start]presented DROP, requiring discrete reasoning over paragraphs. [cite: 189] [cite_start]This benchmark emphasizes numerical and logical reasoning, capabilities that could enhance sentiment analysis of texts containing comparative or quantitative information. [cite: 190] Rajpurkar et al. (2018) [cite_start]extended SQuAD with unanswerable questions, forcing models to recognize when they lack sufficient information. [cite: 191] [cite_start]This capability is valuable for sentiment analysis systems that should abstain from classification when confidence is low. [cite: 192]

### 4.7 Foundation Model Capabilities and Risks

**Opportunities and Capabilities**
Bommasani et al. (2021) [cite_start]provided a comprehensive analysis of foundation models, discussing their opportunities across diverse applications. [cite: 194] [cite_start]Their work highlighted how pre-trained models can be adapted for numerous downstream tasks, including sentiment analysis, with limited task-specific data. [cite: 195] [cite_start]They also emphasized emergent capabilities that appear on the scale, suggesting that larger models might better handle nuanced sentiment understanding. [cite: 196]

**Risks and Limitations**
The same foundation model analysis by Bommasani et al. (2021) [cite_start]extensively documented risks including bias, fairness concerns, environmental impact, and potential for misuse. [cite: 198] [cite_start]For sentiment analysis applications, these concerns are particularly acute: biased sentiment classifiers could perpetuate stereotypes, unfairly characterize demographic groups, or be weaponized for surveillance or manipulation. [cite: 199] [cite_start]Any sentiment analysis system must consider these ethical dimensions. [cite: 200]

**Recent State-of-the-Art Models**
[cite_start]OpenAI (2023) released GPT-4, demonstrating significant improvements in reasoning, factuality, and instruction-following. [cite: 202] Chowdhery et al. (2022) [cite_start]introduced PaLM, showing continued scaling benefits up to 540 billion parameters. [cite: 203] [cite_start]These models represent the current state-of-the-art, though their size makes them challenging to deploy for many sentiment analysis applications, reinforcing the value of efficient alternatives like mixture-of-experts architectures. [cite: 204]

---

## 5. Gap Analysis and Research Contribution

### 5.1 Identified Gaps in Current Literature

[cite_start]After reviewing all this research, several gaps stand out - places where we should be investigating but largely aren't: [cite: 206]

* [cite_start]**First,** nobody's really tried combining HRMs with sentiment analysis in a serious way. [cite: 207] [cite_start]HRMs have shown promise for reasoning tasks, but sentiment analysis researchers haven't picked up on this yet. [cite: 208] [cite_start]Given how much we need interpretability in sentiment analysis, and given that HRMs provide interpretable reasoning structures, this seems like an obvious avenue worth exploring. [cite: 209]
* **Second,** mixture-of-experts architectures are surprisingly underused. [cite_start]Most work either uses a single model or does simple ensembling. [cite: 210] [cite_start]Sophisticated approaches with learned gating networks - where the system learns which expert to trust for which input - remain rare in sentiment analysis literature. [cite: 211] [cite_start]This feels like a missed opportunity. [cite: 212]
* [cite_start]**Third,** cross-domain robustness needs more attention, especially combining different types of models. [cite: 213] [cite_start]Domain adaptation has been studied extensively, but the specific idea of using HRMs alongside domain-specialized experts hasn't been systematically explored. [cite: 214] [cite_start]Could HRMs capture domain-invariant reasoning while other models handle domain-specific patterns? [cite: 215]
* [cite_start]**Fourth,** we need better understanding of efficiency-performance trade-offs in ensemble settings. [cite: 218] [cite_start]Research tends to focus on either small efficient models or large powerful ones but rarely explores how to strategically combine different-sized models in an ensemble. [cite: 219] [cite_start]Maybe you don't need all large models if some smaller ones can handle easier cases. [cite: 220]
* **Finally,** sarcasm detection needs fresh approaches. [cite_start]Despite years of work, models still struggle with irony and sarcasm. [cite: 221] [cite_start]The explicit reasoning paths that HRMs provide might help here - after all, detecting sarcasm requires reasoning about the gap between literal and intended meaning. [cite: 222]

### 5.2 Comparative Analysis: Prior Work vs. Proposed Approach

#### CNN-LSTM Hybrid Models: Strengths and Limitations

* [cite_start]**What They Do Well:** The CNN-LSTM hybrids (Dang et al., 2021; Ezzat et al., 2024) represent a significant step forward. [cite: 237] [cite_start]CNNs extract local n-gram features effectively - they're great at spotting phrases like "not bad" or "very good" as single units. [cite: 238] [cite_start]LSTMs then process these features sequentially, maintaining context across longer spans. [cite: 239] [cite_start]This two-stage process makes intuitive sense, and the performance gains over single architectures are real. [cite: 240] [cite_start]Hassan and Mahmood (2017) showed these hybrids work across multiple languages, suggesting they capture something fundamental about text structure rather than just English-specific patterns. [cite: 241]
* [cite_start]**Where They Fall Short:** But here's what bothers me about pure CNN-LSTM approaches: they still operate as black boxes. [cite: 244] [cite_start]When Ezzat et al.'s model misclassifies a sarcastic tweet, you can't trace why. [cite: 245] [cite_start]The representations are distributed across thousands of neurons, and there's no explicit reasoning chain to follow. [cite: 246] Domain shift remains problematic. [cite_start]Train a CNN-LSTM on movie reviews, test on product reviews, and performance drops. [cite: 247] Architecture learns domain-specific feature combinations rather than general sentiment reasoning strategies. [cite_start]Each new domain might need retraining. [cite: 248]

#### Ensemble and Stacking Approaches: Advances and Gaps

* **What They Contribute:** Alharbi and Lee (2021) and Muhammad et al. (2023) [cite_start]showed that combining diverse models beats single architectures. [cite: 259] [cite_start]The logic is sound: different models make different errors, and aggregation reduces variance. [cite: 260] [cite_start]Muhammad et al.'s stacking approach, where a meta-learner decides which base model to trust, is particularly clever. [cite: 261] [cite_start]It's not just voting; it's learned trust. [cite: 262] The diversity principle works. [cite_start]Aydoğan and Akcayol (2024) confirmed that mixing tree-based, linear, and neural models outperforms homogeneous ensembles. [cite: 263]
* **What's Missing:** Yet these approaches don't address the core interpretability problem. [cite_start]When Alharbi and Lee's ensemble gets it wrong, you have multiple black boxes instead of one. [cite: 266] [cite_start]The meta-learner in stacking adds another layer of opacity - you're learning to trust models you don't understand based on patterns you can't see. [cite: 267] [cite_start]None of these papers systematically investigate sarcasm or pragmatic reasoning. [cite: 268] [cite_start]Cross-domain performance gets limited attention. [cite: 270]

#### How Our Proposed Approach Differs

[cite_start]The HRM-enhanced mixture-of-experts setup I'm proposing addresses these limitations directly. [cite: 252] [cite_start]Instead of just stacking architectures, we'd add an HRM component that performs explicit reasoning at multiple levels (lexical, syntactic, semantic, pragmatic). [cite: 253] [cite_start]When the system classifies text, you can see what the HRM layer "thought" at each level. [cite: 254] [cite_start]This interpretability matters for debugging and trust. [cite: 255]

More importantly, HRMs might capture domain-invariant reasoning patterns. [cite_start]While CNN-LSTM learns "very good appears often in positive movie reviews," HRM could learn the abstract pattern "intensifier + positive adjective suggests strong positive sentiment" - a pattern that transfers across domains. [cite: 256]

[cite_start]The mixture-of-experts framework retains the diversity benefits of ensembles while adding two critical elements. [cite: 273]
1.  [cite_start]**Interpretability:** HRM as an expert provides interpretable reasoning chains - you can see one component's logic even if others remain opaque. [cite: 274]
2.  [cite_start]**Adaptive Specialization:** The gating network can learn *when* to trust the HRM versus statistical models. [cite: 276] [cite_start]For straightforward sentiment ("This movie is great!"), statistical models suffice. [cite: 277] [cite_start]For sarcasm or complex negation, the system could learn to weigh the HRM more heavily. [cite: 278] [cite_start]This adaptive specialization goes beyond static stacking. [cite: 279]

#### Traditional Stacking vs. Mixture-of-Experts with Gating

* [cite_start]**Stacking's Approach (Muhammad et al., 2023; Aydoğan & Akcayol, 2024):** Traditional stacking trains all base models on the same data, collects their predictions, then trains a meta-learner on these predictions. [cite: 284] [cite_start]It's a two-stage process. [cite: 285] [cite_start]All base models process every input fully. [cite: 286] [cite_start]That's computationally expensive. [cite: 287]
* **Our Mixture-of-Experts with Dynamic Gating:** The mixture-of-experts architecture differs fundamentally. [cite_start]A gating network examines each input and dynamically decides which experts to activate and how much to weigh them. [cite: 289] For simple inputs, it might activate only lightweight models. [cite_start]For complex cases, it activates the HRM. [cite: 290] This has two benefits:
    1.  [cite_start]**Computational efficiency:** You don't run every model on every input. [cite: 291]
    2.  [cite_start]**Specialization:** Experts can focus on specific input types. [cite: 292]

#### Interpretability: The Persistent Gap

* **Current State (All Prior Work):** Look across Dang et al. (2021), Alharbi and Lee (2021), Muhammad et al. (2023), Ezzat et al. (2024) [cite_start]- none seriously address interpretability. [cite: 298, 299] [cite_start]They report accurate metrics... That's necessary but insufficient. [cite: 299] [cite_start]When models fail, you can't diagnose why. [cite: 300]
* **Our HRM Integration:** Wang et al. (2025) [cite_start]designed HRMs specifically for interpretable reasoning. [cite: 304] [cite_start]For sentiment analysis, the HRM might output: [cite: 305]
    * **Lexical level:** "Identified positive words: 'great', 'excellent'; negative words: 'not'"
    * **Syntactic level:** "Negation 'not' inverts sentiment of subsequent phrase"
    * **Semantic level:** "Overall semantic orientation: negative"
    * [cite_start]**Pragmatic level:** "Checking for sarcasm markers... none detected" [cite: 305-307]
    This explicit chain-of-thought matters practically. [cite_start]When the system misclassifies, you can trace which level failed. [cite: 308]

#### Domain Robustness: From Single-Domain to Cross-Domain

* [cite_start]**Prior Work's Domain Limitations:** Most papers test within single domains (e.g., movie reviews, COVID tweets). [cite: 315, 316] [cite_start]The problem: real applications need robustness across wildly different contexts. [cite: 318] [cite_start]Current approaches would need separate models. [cite: 320] [cite_start]They overfit surface patterns rather than learning abstract sentiment reasoning. [cite: 323]
* [cite_start]**How HRMs Could Help:** Hierarchical reasoning models separate abstract reasoning from domain-specific pattern matching. [cite: 325] [cite_start]The hypothesis: while CNN-LSTM components learn domain-specific features, the HRM learns domain-invariant reasoning strategies. [cite: 326] [cite_start]The pattern "negation inverts subsequent sentiment" works across domains. [cite: 327] [cite_start]HRMs could learn these abstract patterns. [cite: 329]

#### Computational Efficiency: The Practical Trade-off

* [cite_start]**Ensemble Approaches' Computational Cost:** Running multiple models is expensive. [cite: 334] [cite_start]Each input gets processed by every model. [cite: 335] [cite_start]For large-scale applications, this burden could be prohibitive. [cite: 336]
* [cite_start]**Mixture-of-Experts' Efficiency Advantage:** The MoE framework with sparse activation offers a solution. [cite: 340] The gating network could learn to:
    1.  [cite_start]Use only lightweight models for obvious cases. [cite: 341]
    2.  [cite_start]Activate medium models (CNN-LSTM) for moderate complexity. [cite: 343]
    3.  [cite_start]Engage HRM only for genuinely hard cases (sarcasm, etc.). [cite: 344]
    [cite_start]This selective activation means computational cost scales with input difficulty. [cite: 347]

#### Sarcasm and Pragmatic Reasoning: The Unsolved Problem

* [cite_start]**Prior Work's Implicit Approach:** None of the hybrid or ensemble papers directly address sarcasm detection. [cite: 353] [cite_start]They train on labeled data and *hope* the models learn appropriate patterns. [cite: 354] [cite_start]This implicit learning is brittle. [cite: 356] [cite_start]Sarcasm detection fundamentally requires reasoning about the gap between literal meaning and intended meaning. [cite: 356]
* [cite_start]**HRM's Explicit Reasoning Advantage:** HRMs could address sarcasm through explicit pragmatic reasoning layers. [cite: 363] [cite_start]For "Oh great, another meeting," the HRM might reason: "'great' suggests positive sentiment at literal level, but context 'another meeting' suggests negative situation... infer negative sentiment." [cite: 366] [cite_start]This explicit reasoning chain could handle novel sarcasm forms better than learned statistical patterns. [cite: 367]

### 5.3 Research Contribution

[cite_start]This work pushes forward sentiment analysis by testing whether hierarchical reasoning models add value to standard NLP pipelines, rather than just assuming they do. [cite: 758] [cite_start]It provides a practical framework for combining different types of models in a way that's reproducible and deployable in real applications. [cite: 759]

#### Theoretical Contributions
* [cite_start]Shows how to integrate hierarchical reasoning with ensemble learning for NLP - this hasn't been thoroughly explored before. [cite: 765]
* [cite_start]Provides empirical data on model complementarity across different architectures in sentiment analysis. [cite: 766]
* [cite_start]Develops a framework for interpretable ensemble decisions in text classification tasks. [cite: 767]

#### Practical Contributions
* [cite_start]Delivers a production-ready implementation that people can use for sentiment analysis. [cite: 769]
* [cite_start]Offers comprehensive benchmarking across datasets and domain-transfer scenarios with clear metrics. [cite: 770]
* [cite_start]Give concrete guidelines for model selection and ensemble design when you're working with limited compute resources. [cite: 771]
* [cite_start]Releases open-source code so others can reproduce results and adapt the approach. [cite: 772]

### 5.4 Expected Impact

* [cite_start]**Academic Impact:** Pushes ensemble learning methods forward for NLP applications [cite: 775] [cite_start]and helps us understand how different neural architectures complement each other. [cite: 776]
* [cite_start]**Industrial Impact:** Better sentiment analysis systems for processing customer feedback at scale [cite: 780][cite_start], improved social media monitoring tools [cite: 780][cite_start], and more accurate product review analysis. [cite: 780]
* [cite_start]**Societal Impact:** More effective tools for gauging public opinion [cite: 783] [cite_start]and increased transparency in AI decisions. [cite: 783] [cite_start]Lays groundwork for responsible sentiment analysis that people can trust and understand. [cite: 783]

---

## 6. Project Plan and Scope

### Ablation Studies
* [cite_start]**Baseline/Benchmarking Models:** [cite: 583]
    * [cite_start]`distilbert-base-uncased` [cite: 586]
    * [cite_start]`roberta-base` [cite: 587]
    * [cite_start]`facebook/bart-base` [cite: 588]
* [cite_start]**Ablation and Benchmarking:** [cite: 589]
    1.  [cite_start]TF-IDF + LogisticRegression [cite: 590]
    2.  [cite_start]`distilbert-base-uncased` [cite: 591]
    3.  [cite_start]`distilbert-base-uncased` + HRM (probability averaging) [cite: 592]
    4.  [cite_start]`distilbert-base-uncased` + HRM (logistic-regression stacker) [cite: 593]
* **Study:**
    1.  [cite_start]**HRM contribution:** Does adding HRM to the best single encoder improve macro-F1 vs the encoder alone? [cite: 595]
    2.  [cite_start]**Combiner choice:** Is a simple average of expert probabilities competitive with a logistic-regression stacker? [cite: 596]
    3.  [cite_start]**Classic ML baseline:** How does TF-IDF + Logistic Regression compare? [cite: 597]
    4.  [cite_start]**Model size trade-off:** `distilbert-base-uncased` vs `bert-base-uncased` - how much accuracy is gained for extra compute? [cite: 598]
    5.  [cite_start]**Training data size sensitivity:** Performance at 10% vs 100% of training data - does HRM+stacking help more with low data? [cite: 599]
* **Reporting:**
    1.  [cite_start]**Metrics:** Macro-F1 (primary), Accuracy [cite: 601]
    2.  [cite_start]**Settings:** 3 seeds, max seq length 128, early stopping on Macro-F1 [cite: 602]
    3.  [cite_start]**Dataset:** 100k subset of Sentiment140 + IMDB [cite: 603]

### Scope and Deliverables

* [cite_start]**Scope:** English text sentiment; binary and 3-class labels; public datasets. [cite: 605]
* [cite_start]**Deliverables:** Code repo, experiment logs, trained checkpoints (if small), report, demo notebook. [cite: 606]

### Data & Sources

* [cite_start]**Sentiment140 (Twitter):** Kaggle - Sentiment140 [cite: 608]
* [cite_start]**IMDB Reviews:** Stanford IMDB Large Movie Review [cite: 609]
* [cite_start]**Amazon Reviews:** UCSD Amazon Review Data [cite: 610]
* [cite_start]**TweetEval:** (Mentioned in Thesis Statement [cite: 749])

### Evaluation

* [cite_start]**Metrics:** Macro F1, Accuracy, AUROC [cite: 631]
* [cite_start]**Targets:** ≥3-5 macro-F1 over best single model; statistically significant via paired tests [cite: 632]
* [cite_start]**Ablations:** Remove HRM; change combiner; per-class and length-binned analyses [cite: 633]

### Risks & Mitigations

* [cite_start]**Noisy/imbalanced data:** Use robust preprocessing, class weighting, focal loss [cite: 636]
* [cite_start]**Compute limits:** Use smaller Transformers (DistilBERT), gradient accumulation, subset sampling [cite: 637]
* [cite_start]**API access (Twitter):** Prefer static datasets (Sentiment140/TweetEval); cache samples [cite: 638]
* [cite_start]**Overfitting ensembles:** Proper OOF stacking, nested CV, early stopping [cite: 639]

### Ethical & Legal Considerations

* [cite_start]Respect dataset licenses; avoid re-identification. [cite: 641]
* [cite_start]Analyze bias across demographics where metadata permits; report limitations. [cite: 642]
* [cite_start]Document potential misuse and mitigation (e.g., harmful profiling). [cite: 643]

---

## 7. Merged References

This is a consolidated list of all references cited across the Project Proposal, Literature Review, and Thesis Statement documents.

1.  Alharbi, A. S. M., & Lee, M. (2021). Improving sentiment analysis for social media applications using an ensemble deep learning language model. [cite_start]*Procedia Computer Science, 189*, 135-142. [cite: 419, 420]
2.  Alzantot, M., Sharma, Y., Elgohary, A., Ho, B.-J., Srivastava, M., & Chang, K.-W. (2018). Generating natural language adversarial examples. [cite_start]*In Proceedings of EMNLP*. [cite: 421, 422]
3.  Aydoğan, M., & Akcayol, M. A. (2024). Comparison of machine learning models for sentiment analysis of big Turkish web-based data. [cite_start]*Applied Sciences, 15*(5), 2297. [cite: 423, 424]
4.  Baldwin, T., Cook, P., Lui, M., MacKinlay, A., & Wang, L. (2013). How noisy social media text, how diffrnt social media sources? [cite_start]*In Proceedings of the Sixth International Joint Conference on Natural Language Processing*. [cite: 425-427]
5.  Blitzer, J., Dredze, M., & Pereira, F. (2007). Biographies, bollywood, boom-boxes and blenders: Domain adaptation for sentiment classification. [cite_start]*In Proceedings of the 45th Annual Meeting of ACL*. [cite: 428, 429]
6.  Bommasani, R., Hudson, D. A., Adeli, E., Altman, R., Arora, S., von Arx, S., ... Liang. P. (2021). On the opportunities and risks of foundation models. [cite_start]*arXiv*. [cite: 430, 431, 658, 659, 841, 842]
7.  Breiman, L. (1996). Bagging predictors. [cite_start]*Machine Learning, 24*(2), 123-140. [cite: 812]
8.  Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., ... Amodei, D. (2020). Language models are few-shot learners. [cite_start]*In Advances in Neural Information Processing Systems (NeurIPS)*. [cite: 672, 673, 804, 805]
9.  Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A., ... Dean, J. (2022). PaLM: Scaling language modeling with pathways. [cite_start]*arXiv*. [cite: 432, 433, 678, 679]
10. Christiano, P. F., Leike, J., Brown, T., Martic, M., Legg, S., & Amodei, D. (2017). Deep reinforcement learning from human preferences. [cite_start]*In Advances in Neural Information Processing Systems (NeurIPS)*. [cite: 436, 437, 662, 663]
11. Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., ... Amodei, D. (2021). Training verifiers to solve math word problems. [cite_start]*arXiv*. [cite: 438, 439, 691, 692]
12. Dang, N. C., Moreno-García, M. N., & De la Prieta, F. (2021). Hybrid deep learning models for sentiment analysis. [cite_start]*Complexity, 2021*, Article 9986920. [cite: 434, 435]
13. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. [cite_start]*In Proceedings of NAACL-HLT*. [cite: 674, 675, 801, 803]
14. Dietterich, T. G. (2000). Ensemble methods in machine learning. *In Multiple Classifier Systems*. [cite_start]Springer. [cite: 440]
15. Doshi-Velez, F., & Kim, B. (2017). Towards a rigorous science of interpretable machine learning. [cite_start]*arXiv*. [cite: 441]
16. Dua, D., Wang, Y., Dasigi, P., Stanovsky, G., Singh, S., & Gardner, M. (2019). DROP: A reading comprehension benchmark requiring discrete reasoning over paragraphs. [cite_start]*In Proceedings of NAACL-HLT*. [cite: 442, 443, 695, 696]
17. Ezzat, M., El-Bakry, H. M., Darwish, A., & Hassanien, A. E. (2024). A hybrid deep learning model for sentiment analysis of COVID-19 tweets with class balancing. [cite_start]*Multimedia Tools and Applications, 83*, 21897-21919. [cite: 444-446]
18. Fedus, W., Zoph, B., & Shazeer, N. (2021). Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. [cite_start]*Journal of Machine Learning Research, 23*(120), 1-39. [cite: 447, 448]
19. Freund, Y., & Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning and an application to boosting. [cite_start]*Journal of Computer and System Sciences, 55*(1), 119-139. [cite: 813, 814]
20. Gao, L., Yang, K., & Chen, D. (2023). PAL: Program-aided language models. [cite_start]*In Proceedings of the 40th International Conference on Machine Learning (ICML)*. [cite: 451, 452, 656, 657]
21. Ghosh, D., Vajpayee, A., & Muresan, S. (2020). A report on the 2020 sarcasm detection shared task. [cite_start]*In Proceedings of the Second Workshop on Figurative Language Processing*. [cite: 453, 454]
22. Glorot, X., Bordes, A., & Bengio, Y. (2011). Domain adaptation for large-scale sentiment classification: A deep learning approach. [cite_start]*In Proceedings of the 28th International Conference on Machine Learning (ICML)*. [cite: 455, 456]
23. Go, A., Bhayani, R., & Huang, L. (2009). Twitter sentiment classification using distant supervision. [cite_start]*CS224N Project Report, Stanford, 1*(12), 2009. [cite: 820, 821]
24. Hassan, A., & Mahmood, A. (2017). Sentiment analysis in multilingual context: Comparative analysis of machine learning and hybrid deep learning models. [cite_start]*IEEE Access, 5*, 26696-26706. [cite: 457, 458]
25. He, H., & Garcia, E. A. (2009). Learning from imbalanced data. [cite_start]*IEEE Transactions on Knowledge and Data Engineering, 21*(9), 1263-1284. [cite: 459, 460]
26. Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., & Steinhardt, J. (2021). Measuring massive multitask language understanding. [cite_start]*arXiv*. [cite: 461, 462, 689, 690]
27. Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., ... Sifre, L. (2022). Training compute-optimal large language models. [cite_start]*In Advances in Neural Information Processing Systems (NeurIPS)*. [cite: 468, 469, 664, 665, 845, 846]
28. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, L., ... Chen, W. (2022). LoRA: Low-rank adaptation of large language models. [cite_start]*arXiv*. [cite: 470, 471, 668, 669, 830, 831]
29. Hu, M., & Liu, B. (2004). Mining and summarizing customer reviews. [cite_start]*In Proceedings of the Tenth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*. [cite: 463, 464]
30. Huang, P.-S., Stanforth, R., Welbl, J., Dyer, C., Yogatama, D., Gowal, S., Dvijotham, K., & Kohli, P. (2019). Achieving verified robustness to symbol substitutions via interval bound propagation. [cite_start]*In Proceedings of EMNLP*. [cite: 465-467]
31. Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991). Adaptive mixtures of local experts. [cite_start]*Neural Computation, 3*(1), 79-87. [cite: 472, 473, 815, 816]
32. Jordan, M. I., & Jacobs, R. A. (1994). Hierarchical mixtures of experts and the EM algorithm. [cite_start]*Neural Computation, 6*(2), 181-214. [cite: 474, 475]
33. Joshi, A., Bhattacharyya, P., & Carman, M. J. (2017). Automatic sarcasm detection: A survey. [cite_start]*ACM Computing Surveys, 50*(5), 73:1-73:22. [cite: 476, 477]
34. Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., ... Amodei, D. (2020). Scaling laws for neural language models. [cite_start]*arXiv*. [cite: 478, 479, 666, 667, 843, 844]
35. Karimi, A., Rossi, L., & Prati, A. (2020). Adversarial training for aspect-based sentiment analysis with BERT. [cite_start]*arXiv*. [cite: 480]
36. Kim, Y. (2014). Convolutional neural networks for sentence classification. [cite_start]*In Proceedings of EMNLP*. [cite: 481, 482]
37. Kojima, T., Sato, S., Li, R., Iwasawa, Y., & Matsuo, Y. (2022). Large language models are zero-shot reasoners. [cite_start]*arXiv*. [cite: 483, 484, 649, 792]
38. Lipton, Z. C. (2018). The mythos of model interpretability. [cite_start]*Queue, 16*(3), 31-57. [cite: 485, 840]
39. Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). RoBERTa: A robustly optimized BERT pretraining approach. [cite_start]*arXiv*. [cite: 486, 487]
40. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. [cite_start]*In Advances in Neural Information Processing Systems (NeurIPS)*. [cite: 838, 839]
41. Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning word vectors for sentiment analysis. [cite_start]*In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics*. [cite: 818, 819]
42. Muhammad, P. F., Kusumaningrum, R., & Wibowo, A. (2023). A hybrid deep learning approach for enhanced sentiment classification and consistency analysis in customer reviews. [cite_start]*Mathematics, 11*(23), 3856. [cite: 490, 491]
43. OpenAI. (2023). GPT-4 technical report. [cite_start]*arXiv*. [cite: 492, 682, 810]
44. Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., ... Lowe, R. (2022). Training language models to follow instructions with human feedback. [cite_start]*In Advances in Neural Information Processing Systems (NeurIPS)*. [cite: 497, 498, 660, 661]
45. Pan, S. J., & Yang, Q. (2010). A survey on transfer learning. [cite_start]*IEEE Transactions on Knowledge and Data Engineering, 22*(10), 1345-1359. [cite: 832, 833]
46. Pang, B., Lee, L., & Vaithyanathan, S. (2002). Thumbs up? Sentiment classification using machine learning techniques. [cite_start]*In Proceedings of EMNLP*. [cite: 493, 494]
47. Peng, N., & Dredze, M. (2017). Multi-task domain adaptation for sequence tagging. [cite_start]*In Proceedings of the 2nd Workshop on Representation Learning for NLP*. [cite: 495, 496]
48. Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., ... Liu, P. J. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. [cite_start]*Journal of Machine Learning Research, 21*(140), 1-67. [cite: 499, 500, 676, 677, 806, 807]
49. Rajpurkar, P., Jia, R., & Liang, P. (2018). Know what you don't know: Unanswerable questions for SQuAD. [cite_start]*In Proceedings of ACL*. [cite: 501, 502, 697, 698]
50. Ren, S., Deng, Y., He, K., & Che, W. (2019). Generating natural language adversarial examples through probability weighted word saliency. [cite_start]*In Proceedings of ACL*. [cite: 503-505]
51. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?" Explaining the predictions of any classifier. [cite_start]*In Proceedings of the 22nd ACM SIGKDD*. [cite: 836, 837]
52. Rietzler, A., Stabinger, S., Opitz, P., & Engl, S. (2020). Adapt or get left behind: Domain adaptation through BERT language model finetuning for aspect-target sentiment classification. [cite_start]*In Proceedings of LREC*. [cite: 506-508]
53. Ruder, S., Peters, M. E., Swayamdipta, S., & Wolf, T. (2019). Transfer learning in natural language processing. [cite_start]*In Proceedings of NAACL: Tutorials*. [cite: 834, 835]
54. Ruder, S., Søgaard, A., & Vulić, I. (2019). Unsupervised cross-lingual representation learning. [cite_start]*In Proceedings of ACL: Tutorial Abstracts*. [cite: 509, 510]
55. Sagi, O., & Rokach, L. (2018). Ensemble learning: A survey. [cite_start]*Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 8*(4), e1249. [cite: 511, 512]
56. Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. [cite_start]*In Proceedings of ICLR*. [cite: 513, 514]
57. Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A. Y., & Potts, C. (2013). Recursive deep models for semantic compositionality over a sentiment treebank. [cite_start]*In Proceedings of EMNLP*. [cite: 515-517]
58. Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., ... Lample, G. (2023). LLaMA: Open and efficient foundation language models. [cite_start]*arXiv*. [cite: 680, 681, 808, 809]
59. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. [cite_start]*In Advances in Neural Information Processing Systems (NeurIPS)*. [cite: 670, 671, 799, 800]
60. Wang, A., Pruksachatkun, Y., Nangia, N., Singh, A., Michael, J., Hill, F., ... Bowman, S. R. (2019). SuperGLUE: A stickier benchmark for general-purpose language understanding systems. [cite_start]*In Advances in Neural Information Processing Systems (NeurIPS)*. [cite: 518, 519, 687, 688, 828, 829]
61. Wang, A., Singh, A., Michael, J., Hill, F., Levy, O., & Bowman, S. R. (2018). GLUE: A multi-task benchmark and analysis platform for natural language understanding. [cite_start]*arXiv*. [cite: 520, 521, 685, 686, 824, 825]
62. Wang, D., Liu, P., Zheng, Y., Qiu, X., & Huang, X. (2021). Heterogeneous graph neural networks for extractive document summarization. [cite_start]*In Proceedings of ACL*. [cite: 524-526]
63. Wang, G., Li, J., Sun, Y., Chen, X., Liu, C., Wu, Y., Lu, M., Song, S., & Abbasi Yadkori, Y. (2025). Hierarchical reasoning model. [cite_start]*arXiv*. [cite: 522, 523, 699, 700, 788, 789]
64. Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., ... Zhou, D. (2023). Self-consistency improves chain of thought reasoning in language models. [cite_start]*arXiv*. [cite: 527, 528, 654, 655, 797, 798]
65. Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., ... Zhou, D. (2022). Chain-of-thought prompting elicits reasoning in large language models. [cite_start]*In Advances in Neural Information Processing Systems (NeurIPS)*. [cite: 531, 532, 647, 648, 790, 791]
66. Wilson, T., Wiebe, J., & Hoffmann, P. (2005). Recognizing contextual polarity in phrase-level sentiment analysis. [cite_start]*In Proceedings of HLT-EMNLP*. [cite: 533, 534]
67. Wolpert, D. H. (1992). Stacked generalization. [cite_start]*Neural Networks, 5*(2), 241-259. [cite: 535, 811]
68. Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W., Salakhutdinov, R., & Manning, C. (2018). HotpotQA: A dataset for diverse, explainable multi-hop question answering. [cite_start]*In Proceedings of EMNLP*. [cite: 536, 537, 693, 694]
69. Yao, S., Shafran, I., Narasimhan, K., & Cao, Y. (2023a). ReAct: Synergizing reasoning and acting in language models. [cite_start]*In The Eleventh International Conference on Learning Representations (ICLR)*. [cite: 538, 539, 650, 651, 793, 794]
70. Yao, S., Yu, D., Zhao, S., Shafran, I., Griffiths, T., Cao, Y., & Narasimhan, K. (2023b). Tree of thoughts: Deliberate problem solving with large language models. [cite_start]*arXiv*. [cite: 540, 541, 652, 653, 795, 796]
71. Zhang, L., Wang, S., & Liu, B. (2018). Deep learning for sentiment analysis: A survey. [cite_start]*Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 8*(4), e1253. [cite: 542, 543]
72. Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level convolutional networks for text classification. [cite_start]*In Advances in Neural Information Processing Systems (NeurIPS)*. [cite: 822, 823]
73. Zhou, Z.-H. (2012). *Ensemble methods: Foundations and algorithms*. [cite_start]CRC Press. [cite: 544, 545, 817]
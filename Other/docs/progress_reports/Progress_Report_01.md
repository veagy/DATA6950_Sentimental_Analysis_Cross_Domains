# Progress Report

**Project Name:** Sentinel Analysis with HRMs and Mixture Models
**Student Name:** Rohan Pratap Reddy Ravula
**Date:** Week of January 26, 2026

## Tasks Completed

| Task Description                                                                                                                                                                                                                                                                                     | Hours Spent |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------- |
| **Analysis of Model Failures & Performance Evaluation**<br>- Verified Traditional ML superiority over basic DL/Transformers in current settings.<br>- Validated STACK1 ensemble success (+3.5 F1 target met).<br>- Identified root causes for Transformer (frozen weights) and HRM underperformance. | 15          |
| **Dataset Characterization Analysis**<br>- Quantified difficulty gap between IMDB (Easy) and Sentiment140 (Hard).<br>- Documented statistical significance of ML dominance.                                                                                                                          | 8           |
| **New Dataset Research & Selection**<br>- Identified and documented 4 domain-specific datasets:<br> 1. Financial PhraseBank (Finance)<br> 2. HRAST (Hospitality)<br> 3. Medical Sentiment Analysis (Healthcare)<br> 4. Yelp Open Dataset (Local Business)                                            | 5           |
| **Documentation**<br>- Progress report generation and findings synthesis.                                                                                                                                                                                                                            | 2           |

**Total Hours:** 30

## Plan for Next Week

1.  **Dataset Acquisition & Pipeline Setup**
    - Download and preprocess the four new datasets (Financial, HRAST, Medical, Yelp).
    - Adapting existing data loaders for the new formats.

2.  **Transformer Optimization**
    - Implement Full Fine-Tuning and LoRA adjustments (higher rank) to address "frozen weight" issues.
    - Run new baselines for Transformers on the original datasets.

3.  **Cross-Domain Experimentation**
    - Begin initial cross-domain transfer tests (e.g., Train on IMDB -> Test on Yelp).

4.  **HRM Refinement**
    - Investigate extended training epochs and architecture adjustments for the HRM component.

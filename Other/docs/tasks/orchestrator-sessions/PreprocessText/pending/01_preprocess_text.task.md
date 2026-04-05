## 🔧 Agent Setup (DO THIS FIRST)

### Workflow to Follow
> Follow `/mode-code` workflow to implement this processing script.

### Required Skills
> **python**, **pandas**

## Objective
Write code to preprocess the text datasets (`IMDB_Dataset.csv`, `sentiment_140.csv`, `yelp_academic_dataset_review.json`) from `data/raw`. Use the functions provided in `d:\CAPSTONE\capstone-2\Code\data\clean_text.py`.

## Scope
- Read datasets batch-by-batch appropriately.
- Apply `clean_text(text)`.
- Export to `.parquet` format in `data/processed`.
- Construct `d:\CAPSTONE\capstone-2\Code\thesis\data\preprocess.py`.

## Definition of Done
Script successfully reads at least one of the raw files and writes to `data/processed` without memory crashing.

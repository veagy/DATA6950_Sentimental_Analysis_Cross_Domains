# Dataset analysis run

- **UTC time:** 2026-04-05T04:39:06.282173+00:00
- **Data root:** `D:\CAPSTONE\new\TEMP\data`
- **Output:** `D:\CAPSTONE\new\TEMP\output\dataset analysis`
- **Max rows per file (deep stats):** 10000
- **Include by_source_stem paths:** True
- **Plots:** False
- **Alignment report:** True
- **Dedupe key columns:** (inferred text+label)

## References (methodology)

- [Google ML: Explore your data (text classification)](https://developers.google.com/machine-learning/guides/text-classification/step-2)
- [Data checklists / dataset quality (arXiv)](https://arxiv.org/html/2408.02919v1)

Optional: install `ydata-profiling` for interactive HTML profiles (not run by this script).

## Artifacts

- [`summary_all_files.csv`](summary_all_files.csv)
- `per_file/<path_slug>/profile.json`, `report.md`, optional `label_distribution_by_source_stem.csv`
- `alignment_report.md` (if processed/transformed pairs exist)
- `figures/` (if `--plots`)

## Files analyzed

- `processed/all-data.parquet`
- `processed/amazon_reviews.parquet`
- `processed/HRAST.parquet`
- `processed/IMDB_Dataset.parquet`
- `processed/MedicalSentimentAnalysis.parquet`
- `processed/PatientStatements.parquet`
- `processed/sentiment_140.parquet`
- `processed/tweet_eval.parquet`
- `processed/yelp_business.parquet`
- `processed/yelp_review.parquet`
- `raw/tweet_eval.parquet`
- `raw/yelp_business.parquet`
- `raw/yelp_review.parquet`
- `transformed/all-data.parquet`
- `transformed/amazon_reviews.parquet`
- `transformed/HRAST.parquet`
- `transformed/IMDB_Dataset.parquet`
- `transformed/MedicalSentimentAnalysis.parquet`
- `transformed/PatientStatements.parquet`
- `transformed/sentiment_140.parquet`
- `transformed/tweet_eval.parquet`
- `transformed/yelp_business.parquet`
- `transformed/yelp_review.parquet`

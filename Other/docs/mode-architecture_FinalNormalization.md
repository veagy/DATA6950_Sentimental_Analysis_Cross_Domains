# Architecture: Final Dataset Normalization

## Overview
All raw datasets will be structurally normalized into identically schemed Parquet files within `data/processed/`. The unified schema will exclusively contain two columns: `text` and `sentiment_value`.

## Requirements Tracker
- **Data Source:** `d:\CAPSTONE\capstone-2\data\raw\`
- **Data Destination:** `d:\CAPSTONE\capstone-2\data\processed\`
- **Column Schema:** Strictly `['text', 'sentiment_value']`.
- **Text Cleaning:** Must drop bad utf-8 bytes, apply regex filters (HTML/URl dropping), and NEVER drop the whole row if the value clears.
- **Positional Mapping:**
  - `all-data.csv`: Text (Col 2), Sentiment (Col 1)
  - `amazon_reviews.csv`: Text (Col 1), Sentiment (Col 2)
  - `tweets_eval_merged.parquet`: Text (Col 1), Sentiment (Col 2)
  - `HRAST.csv`: Text (Col 1), Sentiment (Col 2)
  - `IMDB_Dataset.csv`: Text (Col 1), Sentiment (Col 2)
  - `MedicalSentimentAnalysis_merged.csv`: Text (Col 1), Sentiment (Col 2)
  - `PatientStatements_merged.csv`: Text (Col 1), Sentiment (Col 2)
  - `sentiment_140.csv`: Text (Col 1), Sentiment (Col 2)
  - `yelp_business_merged.parquet`: Text (Col 1), Sentiment (Col 2)
  - `yelp_review_merged.parquet`: Text (Col 1), Sentiment (Col 2)

## Implementation Plan
1. Delete old outputs in `data/processed`.
2. Construct the pipeline using Pandas and PyArrow.
3. Handle chunked processing for massive files (Yelp and Amazon).

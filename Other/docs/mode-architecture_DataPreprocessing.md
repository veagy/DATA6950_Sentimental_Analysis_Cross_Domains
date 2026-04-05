# Architecture: Text Data Preprocessing

**Date:** 2026-03-18
**Architect:** VibeCode Architect

## Overview
A scalable preprocessing pipeline built to normalize, clean, and tokenize raw text datasets (IMDB, Sentiment140, Yelp) using existing capabilities defined in `Code/data` (`clean_text.py`, `nlp_preprocessing.py`).

## Goals
- Load raw `.csv` and `.json` text datasets reliably from `data/raw`.
- Clean text artifacts (HTML, encoding issues) using `clean_text`.
- Tokenize text appropriately using HuggingFace integration in `nlp_preprocessing`.
- Save cleansed and chunked data to `data/processed`.
- Generate modular script `Code/thesis/data/preprocess.py`.

## Non-Goals
- Training models.
- Deep exploratory data analysis (EDA).
- Feature engineering beyond textual data cleaning.

## Architecture

**1. Data Loading**
- Use Pandas for `.csv` parsing and JSON stream loading for large Yelp `.json` files.
- Drop irrelevant columns to save memory.

**2. Transformation**
- Clean: `clean_text` module functions (lowercase, HTML decode).
- Tokenize: Optionally apply `tokenize_texts` if pre-tokenization is desired, or simply strip and save normalized string representations for subsequent PyTorch datasets.

**3. Data Sinking**
- Serialize processed blocks into `.parquet` format in `data/processed` to preserve schemas and optimize disk I/O.

## Data Models
N/A - Direct tabular/parquet representations.

## API Specification
N/A - Command Line Execution only.

## Implementation Plan
**Phase 1:** Setup Orchestrator Tasks
**Phase 2:** Implement `preprocess.py` in `Code/thesis/data`
**Phase 3:** Execute preprocessing script to generate processed datasets.

## Open Questions
- Do we need HuggingFace token IDs saved directly into Parquet buffers, or simply the cleaned strings? (Defaulting to strings for flexibility).

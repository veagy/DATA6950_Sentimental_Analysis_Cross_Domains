# Preprocessing Script Implementation Plan

This plan details the code changes required to implement text feature extraction and dimensionality reduction dynamically leveraging standard VRAM mapping.

## User Review Required
> [!NOTE]
> Please review the UMAP fitting strategy. Standard UMAP cannot process 7+ million rows simultaneously during topology construction without exhausting 100GB+ of system RAM. 
> To enforce absolute safety for your 8GB RTX 4070 and basic System RAM: 
> 1) The *Sentence Transformer* will encode arrays securely with a fixed `batch_size=128`.
> 2) The *UMAP Reducer* will be locally trained (fitted) exclusively on a **100,000-row randomized sample** per dataset.
> 3) We will then iteratively stream your dataset in chunks through the encoder, and securely `.transform()` them through that fitted UMAP model to generate strictly exactly 100 features. 

## Proposed Changes

### Projection Script
#### [NEW] [embed_reduce.py](file:///d:/CAPSTONE/capstone-2/Code/thesis/data/embed_reduce.py)
This script will:
- Store `all-MiniLM-L6-v2` locally in `d:\CAPSTONE\capstone-2\checkpoints\transformer\`.
- Iterate over all processed Parquets residing in `d:\CAPSTONE\capstone-2\data\processed\`.
- Use PyArrow array streaming to map text sentences concurrently while capping physical device memory limits directly out-of-core.
- Format the outputs directly to `['features_100d', 'sentiment_value']`.
- Serialize everything safely into `d:\CAPSTONE\capstone-2\data\transformed\`.

## Verification Plan

### Automated Tests
Verification defaults to monitoring local Windows GPU resource usage. The process should visibly limit RTX VRAM to ~2 GB concurrently. 

### Manual Verification
Execute:
`python d:\CAPSTONE\capstone-2\Code\thesis\data\embed_reduce.py`
Verify that `features_100d` column dynamically possesses 100 dense numbers uniformly.

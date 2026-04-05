# Architecture: Feature Extraction & Manifold Projection

**Date:** 2026-03-19
**Architect:** VibeCode Architect

## Overview
Generate dense semantic embeddings for normalized NLP datasets using Sentence Transformers, and project these 384-dimensional embeddings down to exactly 100 dimensions using UMAP, guaranteeing that strict VRAM and RAM constraints are not exceeded on consumer hardware.

## Goals
- Download and checkpoint `sentence-transformers/all-MiniLM-L6-v2` strictly locally at `checkpoints/transformer/`.
- Translate texts to `[100]` float features via Neural Network + UMAP.
- Prevent Nvidia RTX 4070 8GB VRAM OOM (Out Of Memory) crashes.
- Prevent system RAM crashes during UMAP topological calculations on 7-million-row subsets.
- Export entirely new Datasets to `data/transformed/`.

## Non-Goals
- Generating full 384D raw embeddings for explicit persistent storage (we store only the 100D targets to save disk space).
- Fitting a Single Global UMAP across all 10 datasets merged (we fit a dedicated UMAP space per specific dataset to preserve localized task topologies).

## Architecture

**1. Inference Generation (`sentence-transformers`)**
- `all-MiniLM-L6-v2` is roughly 90 Megabytes inside VRAM. To absolutely secure the 8GB buffer during inference of extremely long texts, the `batch_size` parameter inside `.encode()` will be locked to `256` or `128`. Memory spikes will be impossible.

**2. Dimensionality Reduction (`umap-learn`)**
- **RAM Crash Solution:** Standard UMAP builds KNN graphs in memory. For 5.3GB Yelp files (~7.5M records), building a 7.5M node KNN graph requires ~120GB of RAM.
- **Batched Mapping:** For each `.parquet`, we will ingest a randomized sample of at most **100,000** rows. We will encode these, and call `umap.UMAP(n_components=100).fit()` on them exclusively.
- We then iteratively stream the dataset in chunks, calling `.encode()` followed sequentially by `umap_model.transform()`, keeping RAM purely free.

## Implementation Plan
**Phase 1:** Ask user for algorithmic strategy approval.
**Phase 2:** Delegate to Sub-Agent to draft Script.
**Phase 3:** Execution on local hardware.

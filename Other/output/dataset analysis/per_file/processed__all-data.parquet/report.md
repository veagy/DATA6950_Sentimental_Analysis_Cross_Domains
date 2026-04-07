# Dataset profile: `processed/all-data.parquet`

- **Rows (metadata):** 9514666
- **Deep stats rows:** 10000 (sampled)
- **File size (bytes):** 3021655663

## Inferred columns

- Text: `text`
- Label: `sentiment_value`
- Features: `None`
- source_stem: `source_stem`

## Text length (sample)

```json
{
  "n_empty_text": 0,
  "char_len_min": 6,
  "char_len_max": 483,
  "char_len_mean": 54.813,
  "char_len_std": 36.10726525867676,
  "char_len_p50": 45.0,
  "char_len_p90": 100.0,
  "char_len_p95": 124.0,
  "char_len_p99": 181.0,
  "word_count_mean": 9.669
}
```

## Normalized labels (3-class, sample)

```json
{
  "0": 4041,
  "1": 5558,
  "2": 401,
  "_skipped_normalization": 0
}
```

## Normalized labels (2-class, sample)

```json
{
  "0": 4041,
  "1": 5558,
  "_skipped_normalization": 401
}
```

- Per-stem raw label counts (sample): see `label_distribution_by_source_stem.csv`

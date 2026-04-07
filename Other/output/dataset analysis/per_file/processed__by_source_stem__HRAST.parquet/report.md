# Dataset profile: `processed/by_source_stem/HRAST.parquet`

- **Rows (metadata):** 4000
- **Deep stats rows:** 4000 (full sample window)
- **File size (bytes):** 129588

## Inferred columns

- Text: `text`
- Label: `sentiment_value`
- Features: `None`
- source_stem: `source_stem`

## Text length (sample)

```json
{
  "n_empty_text": 0,
  "char_len_min": 7,
  "char_len_max": 439,
  "char_len_mean": 54.6745,
  "char_len_std": 36.301707993575356,
  "char_len_p50": 44.0,
  "char_len_p90": 100.0,
  "char_len_p95": 124.0,
  "char_len_p99": 179.01999999999953,
  "word_count_mean": 9.467
}
```

## Normalized labels (3-class, sample)

```json
{
  "0": 1721,
  "1": 2042,
  "2": 237,
  "_skipped_normalization": 0
}
```

## Normalized labels (2-class, sample)

```json
{
  "0": 1721,
  "1": 2042,
  "_skipped_normalization": 237
}
```

- Per-stem raw label counts (sample): see `label_distribution_by_source_stem.csv`

# Dataset profile: `processed/yelp_review.parquet`

- **Rows (metadata):** 6990280
- **Deep stats rows:** 10000 (sampled)
- **File size (bytes):** 2621322956

## Inferred columns

- Text: `text`
- Label: `sentiment_value`
- Features: `None`
- source_stem: `None`

## Text length (sample)

```json
{
  "n_empty_text": 0,
  "char_len_min": 70,
  "char_len_max": 5019,
  "char_len_mean": 607.9393,
  "char_len_std": 500.54217850522474,
  "char_len_p50": 455.5,
  "char_len_p90": 1175.0,
  "char_len_p95": 1544.0,
  "char_len_p99": 2587.140000000003,
  "word_count_mean": 111.2418
}
```

## Normalized labels (3-class, sample)

```json
{
  "0": 1842,
  "1": 7019,
  "2": 1139,
  "_skipped_normalization": 0
}
```

## Normalized labels (2-class, sample)

```json
{
  "0": 1842,
  "1": 7019,
  "_skipped_normalization": 1139
}
```

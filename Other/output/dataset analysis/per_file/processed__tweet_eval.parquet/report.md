# Dataset profile: `processed/tweet_eval.parquet`

- **Rows (metadata):** 200721
- **Deep stats rows:** 10000 (sampled)
- **File size (bytes):** 12123346

## Inferred columns

- Text: `text`
- Label: `sentiment_value`
- Features: `None`
- source_stem: `None`

## Text length (sample)

```json
{
  "n_empty_text": 0,
  "char_len_min": 1,
  "char_len_max": 130,
  "char_len_mean": 56.5919,
  "char_len_std": 22.467541775488996,
  "char_len_p50": 56.0,
  "char_len_p90": 90.0,
  "char_len_p95": 93.0,
  "char_len_p99": 95.0,
  "word_count_mean": 10.7033
}
```

## Normalized labels (3-class, sample)

```json
{
  "0": 4233,
  "1": 3125,
  "2": 2642,
  "_skipped_normalization": 0
}
```

## Normalized labels (2-class, sample)

```json
{
  "0": 4233,
  "1": 3125,
  "_skipped_normalization": 2642
}
```

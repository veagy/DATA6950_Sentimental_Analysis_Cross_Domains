# Dataset profile: `raw/yelp_review.parquet`

- **Rows (metadata):** 6990280
- **Deep stats rows:** 10000 (sampled)
- **File size (bytes):** 2693319000

## Inferred columns

- Text: `text`
- Label: `sentiment_values`
- Features: `None`
- source_stem: `None`

## Text length (sample)

```json
{
  "n_empty_text": 0,
  "char_len_min": 89,
  "char_len_max": 5059,
  "char_len_mean": 612.0482,
  "char_len_std": 504.4530737017237,
  "char_len_p50": 458.0,
  "char_len_p90": 1182.0,
  "char_len_p95": 1559.0,
  "char_len_p99": 2606.0200000000004,
  "word_count_mean": 111.2285
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

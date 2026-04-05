# Datasets

Project datasets live under this directory in three trees: **raw**, **processed**, and **transformed**. They are not checked into Git (see the repo `.gitignore`); use local copies or download from Google Drive so paths match the code.

## Google Drive

**All dataset folders can be accessed here:**

https://drive.google.com/drive/folders/1M3TfrFmJBExkmh8eJynQHVevgX5IP_82?usp=sharing

Download **processed**, **raw**, and **transformed** from that folder and place them here:

```text
Dataset/
  raw/
  processed/
  transformed/
```

## Folder overview

| Folder | Purpose |
|--------|---------|
| `raw/` | Source extracts and original CSVs (e.g. per-domain or combined inputs) |
| `processed/` | Cleaned or intermediate artifacts derived from raw data |
| `transformed/` | Featurized or model-ready outputs (embeddings, splits, etc., as used by your pipeline) |

If a subfolder is missing locally, sync it from the Drive link above.

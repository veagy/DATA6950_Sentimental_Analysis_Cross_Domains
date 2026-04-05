import pandas as pd
from pathlib import Path

DIR = Path(r"d:\CAPSTONE\capstone-2\data\tweets eval")
OUT_FILE = Path(r"d:\CAPSTONE\capstone-2\data\raw\tweets_eval_merged.parquet")

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

all_dfs = []

def map_label(val, unique_labels):
    # Depending on types, check as ints
    labels_set = set()
    for l in unique_labels:
        try:
            labels_set.add(int(l))
        except:
            pass
            
    val_int = None
    try:
        val_int = int(val)
    except:
        pass

    # Rule 1: exactly {0, 1, 2}
    if labels_set == {0, 1, 2}:
        mapping = {0: "negative", 1: "neutral", 2: "positive"}
        return mapping.get(val_int, str(val))
    # Rule 2: exactly {0, 1}
    elif labels_set == {0, 1}:
        mapping = {0: "negative", 1: "positive"}
        return mapping.get(val_int, str(val))
    # Fallback to string representation of the label
    return str(val)

for filepath in DIR.glob("*.csv"):
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading {filepath.name}: {e}")
        continue
        
    print(f"Processing {filepath.name}...")
    
    # Verify standard columns
    if 'text' in df.columns and 'label' in df.columns:
        df = df.dropna(subset=['label', 'text'])
        
        unique_labels = df['label'].unique()
        df['mapped_label'] = df['label'].apply(lambda x: map_label(x, unique_labels))
        
        clean_df = pd.DataFrame({
            'text': df['text'].astype(str),
            'label': df['mapped_label'],
            'source': filepath.name
        })
        all_dfs.append(clean_df)
    else:
        print(f"[SKIP] {filepath.name} - 'text' or 'label' column missing.")

if all_dfs:
    merged_df = pd.concat(all_dfs, ignore_index=True)
    # Write to Parquet efficiently using pure string format for labels to avoid PyArrow mixed errors
    merged_df['label'] = merged_df['label'].astype(str)
    
    merged_df.to_parquet(OUT_FILE, engine='pyarrow', index=False)
    print(f"\n[SUCCESS] Merged {len(all_dfs)} files into {OUT_FILE} with {len(merged_df)} total rows.")
else:
    print("[ERROR] No data extracted.")

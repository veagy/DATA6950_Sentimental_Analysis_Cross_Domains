import sys
import os
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import importlib.util

_REPO = Path(__file__).resolve().parents[3]
# Load clean_text without triggering package initialization
code_path = _REPO / "Code" / "data" / "clean_text.py"
spec = importlib.util.spec_from_file_location("clean_text", str(code_path))
clean_text_module = importlib.util.module_from_spec(spec)
sys.modules["clean_text"] = clean_text_module
spec.loader.exec_module(clean_text_module)
clean_text = clean_text_module.clean_text

RAW_DIR = _REPO / "data" / "raw"
PROCESSED_DIR = _REPO / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def find_text_column(columns, df_peak=None):
    """Heuristic to find the text column to clean."""
    candidates = ['text', 'review', 'statement', 'tweet', 'sentence', 'body']
    cols_lower = {str(c).lower().strip(): c for c in columns}
    
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
            
    for c in cols_lower:
        if 'text' in c or 'review' in c or 'tweet' in c:
            return cols_lower[c]
            
    # Fallback if df_peak provided: pick the first object column containing strings longer than 15 chars
    if df_peak is not None:
        for col in df_peak.columns:
            if df_peak[col].dtype == object:
                sample = str(df_peak.iloc[0][col])
                if len(sample) > 15:
                    return col
    return None

def process_csv(filepath: Path):
    filename = filepath.name
    output_path = PROCESSED_DIR / (filepath.stem + '.parquet')
    if output_path.exists():
        print(f"[SKIP] {output_path.name} already processed.")
        return

    print(f"\n[INFO] Processing CSV: {filename}...")
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='latin1')
    except Exception as e:
        # Maybe no header or bad lines? Let's try basic
        try:
            df = pd.read_csv(filepath, encoding='latin1', on_bad_lines='skip')
        except:
            print(f"[ERROR] Failed to read {filename} - {e}")
            return
            
    text_col = find_text_column(df.columns, df)
    if not text_col:
        print(f"[ERROR] Failed to identify text column in {filename}")
        return

    print(f"       Identified text column: '{text_col}'")
    df = df.dropna(subset=[text_col])
    
    # Cast all object columns to string to prevent PyArrow mixed-type errors
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)
        
    print(f"       Applying clean_text() to {len(df)} rows...")
    df['cleaned_text'] = df[text_col].apply(clean_text)
    
    df.to_parquet(output_path, engine='pyarrow', index=False)
    print(f"[OK]   Saved {output_path.name}")

def process_json(filepath: Path):
    filename = filepath.name
    output_path = PROCESSED_DIR / (filepath.stem + '.parquet')
    if output_path.exists():
        print(f"[SKIP] {output_path.name} already processed.")
        return

    print(f"\n[INFO] Processing JSON (chunked): {filename}...")
    try:
        peak_iter = pd.read_json(filepath, lines=True, chunksize=5)
        df_peak = next(peak_iter)
    except Exception as e:
        print(f"[ERROR] Failed to read JSON {filename}: {e}")
        return

    text_col = find_text_column(df_peak.columns, df_peak)
    if not text_col:
        print(f"[ERROR] Failed to identify text column in {filename}")
        return

    print(f"       Identified text column: '{text_col}'")
    
    chunks = pd.read_json(filepath, lines=True, chunksize=150000)
    writer = None
    chunk_idx = 1
    
    for df in chunks:
        print(f"       - Chunk {chunk_idx}")
        df = df.dropna(subset=[text_col])
        
        # Cast all object columns to string to prevent PyArrow mixed-type errors
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str)
            
        df['cleaned_text'] = df[text_col].apply(clean_text)
        
        table = pa.Table.from_pandas(df)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)
        
        writer.write_table(table)
        chunk_idx += 1
        
    if writer:
        writer.close()
    print(f"[OK]   Saved {output_path.name}")

def main():
    print("==================================================")
    print(" Processing ALL Raw Datasets")
    print("==================================================")
    
    for child in RAW_DIR.iterdir():
        if not child.is_file():
            continue
        ext = child.suffix.lower()
        if ext == '.csv':
            process_csv(child)
        elif ext == '.json':
            process_json(child)
        else:
            print(f"[IGNORE] Skipping non-csv/json file: {child.name}")

if __name__ == "__main__":
    main()

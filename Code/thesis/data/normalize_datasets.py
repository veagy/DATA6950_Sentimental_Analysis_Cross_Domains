import argparse
import sys
import shutil
from pathlib import Path
import pandas as pd
import importlib.util
import pyarrow as pa
import pyarrow.parquet as pq

_REPO = Path(__file__).resolve().parents[3]

# Load clean_text
code_path = _REPO / "Code" / "data" / "clean_text.py"
spec = importlib.util.spec_from_file_location("clean_text", str(code_path))
clean_text_module = importlib.util.module_from_spec(spec)
sys.modules["clean_text"] = clean_text_module
spec.loader.exec_module(clean_text_module)
clean_text = clean_text_module.clean_text

RAW_DIR = _REPO / "data" / "raw"
PROCESSED_DIR = _REPO / "data" / "processed"

_ALL_DATA_PARQUET = "all-data.parquet"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Normalize raw CSV/parquet into data/processed/*.parquet. "
            "Does NOT remove data/processed by default (preserves merged all-data.parquet)."
        )
    )
    ap.add_argument(
        "--flush-processed",
        action="store_true",
        help=(
            "DANGEROUS: delete the entire data/processed directory first, then recreate it. "
            "This removes all-data.parquet and every per-dataset parquet. "
            "Re-run merge_all_data_parquet.py afterward to rebuild all-data.parquet."
        ),
    )
    return ap.parse_args()


def _maybe_flush_processed(flush: bool) -> None:
    if not flush:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        return
    merged = PROCESSED_DIR / _ALL_DATA_PARQUET
    if merged.is_file():
        print(
            f"[WARN] --flush-processed will DELETE {merged} (and all other processed parquets).",
            file=sys.stderr,
        )
    if PROCESSED_DIR.exists():
        print(f"[flush] Removing {PROCESSED_DIR} ...")
        shutil.rmtree(PROCESSED_DIR)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# text: dict index (0-based) based on User's 1-based "column_1", "column_2"
MAPPING = {
    'all-data': {'text': 1, 'sentiment': 0, 'header': None},
    'amazon_reviews': {'text': 0, 'sentiment': 1, 'header': 'infer'},
    'tweet_eval': {'text': 0, 'sentiment': 1, 'header': 'infer'},
    'tweets_eval': {'text': 0, 'sentiment': 1, 'header': 'infer'}, # catching either name
    'HRAST': {'text': 0, 'sentiment': 1, 'header': 'infer'},
    'IMDB_Dataset': {'text': 0, 'sentiment': 1, 'header': 'infer'},
    'MedicalSentiment': {'text': 0, 'sentiment': 1, 'header': 'infer'},
    'PatientStatements': {'text': 0, 'sentiment': 1, 'header': 'infer'},
    'sentiment_140': {'text': 0, 'sentiment': 1, 'header': 'infer'},
    'yelp_business': {'text': 0, 'sentiment': 1, 'header': 'infer'},
    'yelp_review': {'text': 0, 'sentiment': 1, 'header': 'infer'}
}

def get_mapping(filename):
    for k, v in MAPPING.items():
        if k.lower() in filename.lower():
            return v
    return None

def clean_text_wrapper(text):
    if pd.isnull(text):
        return ""
    # Ensure utf-8
    text_str = str(text).encode('utf-8', 'ignore').decode('utf-8')
    res = clean_text(text_str)
    return res if res else ""

def process_chunk(df, mapping):
    cols = df.columns
    text_col = cols[mapping['text']]
    sentiment_col = cols[mapping['sentiment']]
    
    df_out = pd.DataFrame()
    df_out['text'] = df[text_col].apply(clean_text_wrapper)
    df_out['sentiment_value'] = df[sentiment_col].astype(str).str.strip()
    
    # If the ENTIRE row becomes empty (both text and sentiment), then delete that row ONLY
    is_text_empty = df_out['text'] == ""
    is_sent_empty = df_out['sentiment_value'].isin(["", "nan", "None", "NaN"])
    df_out = df_out[~(is_text_empty & is_sent_empty)]
    
    return df_out

def process_file(filepath):
    filename = filepath.name
    if filepath.stem.lower() == "all-data":
        print(f"[SKIP] {filename} (use per-dataset sources; merged all-data comes from merge_all_data_parquet.py)")
        return
    mapping = get_mapping(filename)
    if not mapping:
        print(f"[SKIP] No mapping defined for {filename}")
        return
        
    print(f"\n[INFO] Normalizing {filename}...")
    # Standardize output filenames slightly
    base = filepath.stem
    if base.endswith('_merged'):
        base = base.replace('_merged', '')
    output_path = PROCESSED_DIR / (base + '.parquet')
    
    ext = filepath.suffix.lower()
    
    if ext == '.csv':
        try:
            chunks = pd.read_csv(filepath, encoding='latin1', chunksize=250000, header=mapping['header'], low_memory=False)
            writer = None
            for i, df in enumerate(chunks):
                print(f"       Chunk {i+1}...")
                df_out = process_chunk(df, mapping)
                table = pa.Table.from_pandas(df_out)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema)
                writer.write_table(table)
            if writer: writer.close()
            print(f"[OK]   Saved {output_path.name}")
        except Exception as e:
            print(f"[ERROR] CSV read failed for {filename}: {e}")
            
    elif ext == '.parquet':
        try:
            parquet_file = pq.ParquetFile(filepath)
            writer = None
            for i, batch in enumerate(parquet_file.iter_batches(batch_size=250000)):
                print(f"       Chunk {i+1}...")
                df = batch.to_pandas()
                df_out = process_chunk(df, mapping)
                table = pa.Table.from_pandas(df_out)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema)
                writer.write_table(table)
            if writer: writer.close()
            print(f"[OK]   Saved {output_path.name}")
        except Exception as e:
            print(f"[ERROR] Parquet read failed for {filename}: {e}")

if __name__ == '__main__':
    args = _parse_args()
    _maybe_flush_processed(args.flush_processed)
    print("===============================================")
    print(" Final Normalization Pipeline ")
    print("===============================================")
    if not RAW_DIR.is_dir():
        print(f"[WARN] Missing raw dir {RAW_DIR}; nothing to normalize.")
        sys.exit(0)
    for child in RAW_DIR.iterdir():
        if child.is_file():
            process_file(child)

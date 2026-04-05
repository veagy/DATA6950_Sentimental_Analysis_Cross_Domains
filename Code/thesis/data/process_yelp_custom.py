import pandas as pd
import json
from pathlib import Path
import numpy as np

business_file = r"d:\CAPSTONE\capstone-2\data\raw\yelp_academic_dataset_business.json"
review_file = r"d:\CAPSTONE\capstone-2\data\raw\yelp_academic_dataset_review.json"

out_business = r"d:\CAPSTONE\capstone-2\data\raw\yelp_business_merged.parquet"
out_review = r"d:\CAPSTONE\capstone-2\data\raw\yelp_review_merged.parquet"

def process_business():
    print("Processing business dataset...")
    df = pd.read_json(business_file, lines=True)
    
    # Remove _id columns
    id_cols = [c for c in df.columns if str(c).endswith('_id')]
    df = df.drop(columns=id_cols)
    
    # Calculate sentiment score
    df['sentiment_score'] = df['stars'] * df['review_count']
    
    # Binning strategy for business (qcut using rank to avoid identical bucket edges)
    ranked_scores = df['sentiment_score'].rank(method='first')
    df['sentiment_values'] = pd.qcut(ranked_scores, q=3, labels=["negative", "neutral", "positive"])
    
    # The columns to merge into text are everything EXCEPT stars, review_count, sentiment_score, sentiment_values
    text_cols = [c for c in df.columns if c not in ['stars', 'review_count', 'sentiment_score', 'sentiment_values']]
    
    # Vectorized text construction
    text_series = None
    for col in text_cols:
        col_str = str(col) + ": " + df[col].astype(str)
        if text_series is None:
            text_series = col_str
        else:
            text_series = text_series + ", " + col_str
            
    df['text'] = text_series
    
    df_out = df[['text', 'sentiment_values']].copy()
    df_out['text'] = df_out['text'].astype(str)
    df_out['sentiment_values'] = df_out['sentiment_values'].astype(str)
    
    df_out.to_parquet(out_business, index=False)
    print(f"Business processed. Saved to {out_business}")

def process_review():
    print("\nProcessing review dataset in chunks...")
    chunks = pd.read_json(review_file, lines=True, chunksize=250000)
    
    import pyarrow as pa
    import pyarrow.parquet as pq
    writer = None
    
    for i, df in enumerate(chunks):
        print(f" Chunk {i+1}...")
        
        # Remove _id columns
        id_cols = [c for c in df.columns if str(c).endswith('_id')]
        df = df.drop(columns=id_cols)
        
        # Define sentiment_values based on stars (1,2=negative, 3=neutral, 4,5=positive)
        def map_stars(s):
            if s <= 2.5: return "negative"
            elif s < 4.0: return "neutral"
            else: return "positive"
            
        df['sentiment_values'] = df['stars'].apply(map_stars)
        
        # Merge all OTHER columns into text
        text_cols = [c for c in df.columns if c not in ['stars', 'sentiment_values']]
        
        text_series = None
        for col in text_cols:
            # We must escape newlines inside the text if we want a clean single line, but parquet handles multi-line fine.
            col_str = str(col) + ": " + df[col].astype(str)
            if text_series is None:
                text_series = col_str
            else:
                text_series = text_series + ", " + col_str
                
        df['text'] = text_series
        
        df_out = df[['text', 'sentiment_values']].copy()
        df_out['text'] = df_out['text'].astype(str)
        df_out['sentiment_values'] = df_out['sentiment_values'].astype(str)
        
        table = pa.Table.from_pandas(df_out)
        if writer is None:
            writer = pq.ParquetWriter(out_review, table.schema)
        writer.write_table(table)
        
    if writer: writer.close()
    print(f"Review processed. Saved to {out_review}")

if __name__ == '__main__':
    process_business()
    process_review()

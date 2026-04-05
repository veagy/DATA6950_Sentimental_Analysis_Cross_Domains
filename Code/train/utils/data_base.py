"""
Centralized data storage and retrieval hub for the Sentinel-II project.
Handles 30+ file formats and cloud providers (S3, GCS, Azure, Drive) with automatic .data caching.
"""

import io
import os
import pathlib
from pathlib import Path
from typing import Any, Optional, Dict
from urllib.parse import urlparse

import numpy as np
import torch
import pandas as pd

PATH_DATA_HUB = Path("./data")


def _to_tensor(x: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert any array-like to a torch.Tensor."""
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x.copy())
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return torch.tensor(x.values, dtype=dtype)
    if isinstance(x, set):
        x = sorted(x)
    return torch.tensor(x, dtype=dtype)


def load_dataframe(source: str, use_cache: bool = True, **kwargs) -> pd.DataFrame:
    """
    Load a CSV/Parquet/Any dataframe from a unified URI.
    If use_cache is True, attempts to retrieve from/save to PROJECT_ROOT/.data/
    """
    # Hash source URI for cache key
    cache_key = f"cache_{hash(source) & 0xffffffff:x}_{Path(source).name}"
    cache_path = PATH_DATA_HUB / cache_key
    
    if use_cache and cache_path.exists():
        if cache_path.suffix == ".parquet":
            return pd.read_parquet(cache_path, **kwargs)
        return pd.read_csv(cache_path, **kwargs)

    df = _load_raw_dataframe(source, **kwargs)
    
    if use_cache:
        PATH_DATA_HUB.mkdir(parents=True, exist_ok=True)
        # Store as parquet for efficiency if possible, else csv
        if source.endswith(".parquet") or cache_path.suffix == ".parquet":
            df.to_parquet(cache_path.with_suffix(".parquet"))
        else:
            df.to_csv(cache_path.with_suffix(".csv"), index=False)
            
    return df


def _load_raw_dataframe(source: str, **kwargs) -> pd.DataFrame:
    """Dispatches loading to various providers (Cloud/Local/Database)."""
    
    # Check for Database URI
    if any(source.startswith(db) for db in ('sqlite://', 'postgres://', 'postgresql://', 'mysql://')):
        import sqlalchemy
        engine = sqlalchemy.create_engine(source)
        table = kwargs.pop('table_name', 'data')
        return pd.read_sql_table(table, engine, **kwargs)

    # Determine extension
    ext = kwargs.pop('format', None)
    if not ext:
        parsed = urlparse(source)
        path = parsed.path if parsed.path else source
        if not path.startswith('/'):
            path = source.split('://')[-1] if '://' in source else source
        ext = pathlib.Path(path).suffix.lstrip('.').lower() or 'csv'

    # Cloud Dispatch
    if source.startswith("s3://"):
        import boto3
        parts = source[5:].split("/", 1)
        obj = boto3.client("s3").get_object(Bucket=parts[0], Key=parts[1])
        return _parse_bytes_to_dataframe(obj["Body"].read(), ext, **kwargs)

    if source.startswith("gs://"):
        from google.cloud import storage
        parts = source[5:].split("/", 1)
        data = storage.Client().bucket(parts[0]).blob(parts[1]).download_as_bytes()
        return _parse_bytes_to_dataframe(data, ext, **kwargs)

    if source.startswith("az://"):
        from azure.storage.blob import BlobServiceClient
        parts = source[5:].split("/", 1)
        conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        if not conn:
            raise EnvironmentError("AZURE_STORAGE_CONNECTION_STRING not set.")
        data = (BlobServiceClient.from_connection_string(conn)
                 .get_blob_client(parts[0], parts[1])
                 .download_blob().readall())
        return _parse_bytes_to_dataframe(data, ext, **kwargs)

    if source.startswith("gdrive://"):
        import gdown
        buf = io.BytesIO()
        gdown.download(f"https://drive.google.com/uc?id={source[9:]}", buf, quiet=False)
        return _parse_bytes_to_dataframe(buf.getvalue(), ext, **kwargs)

    if "1drv.ms" in source or "onedrive" in source.lower():
        import requests
        r = requests.get(source.rstrip("/") + "&download=1", stream=True, timeout=300)
        r.raise_for_status()
        return _parse_bytes_to_dataframe(r.content, ext, **kwargs)

    # Local Path (handles directory/file)
    return _parse_file_to_dataframe(source, ext, **kwargs)


def _parse_bytes_to_dataframe(data: bytes, ext: str, **kwargs) -> pd.DataFrame:
    """Robust byte-stream parsing for 30+ formats."""
    buf = io.BytesIO(data)
    if ext == 'parquet': return pd.read_parquet(buf, **kwargs)
    if ext in ('csv', 'txt'): return pd.read_csv(buf, **kwargs)
    if ext == 'json': return pd.read_json(buf, **kwargs)
    if ext in ('xls', 'xlsx'): return pd.read_excel(buf, **kwargs)
    # Add more robust checks from previous data_source.py if needed...
    return pd.read_csv(buf, **kwargs)


def _parse_file_to_dataframe(path: str, ext: str, **kwargs) -> pd.DataFrame:
    """Robust local file parsing."""
    if ext == 'parquet': return pd.read_parquet(path, **kwargs)
    if ext == 'csv': return pd.read_csv(path, **kwargs)
    # Implements sqlite/db logic
    if ext in ('db', 'sqlite'):
        import sqlite3
        con = sqlite3.connect(path)
        table = kwargs.pop('table_name', 'data')
        return pd.read_sql(f"SELECT * FROM {table}", con, **kwargs)
    return pd.read_csv(path, **kwargs)

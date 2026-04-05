"""
Storage-agnostic data source loader.
Reads from: local filesystem, AWS S3, Azure Blob, GCS, Google Drive, OneDrive, LAN/NAS.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_dataframe(source: str, path: str) -> pd.DataFrame:
    """
    Load a CSV / Parquet dataframe from any supported storage backend.

    Parameters
    ----------
    source : str
        One of: 'local', 'nas', 's3', 'azure', 'gcs', 'gdrive', 'onedrive'.
    path : str
        URI or local path to the data file.
    """
    source = source.lower().strip()

    if source in ("local", "nas"):
        p = Path(path)
        if p.suffix == ".parquet":
            return pd.read_parquet(p)
        if p.suffix in (".tsv",):
            return pd.read_csv(p, sep="\t")
        if p.suffix in (".xlsx", ".xls"):
            return pd.read_excel(p)
        return pd.read_csv(p)

    if source == "s3":
        import boto3  # type: ignore

        s3 = boto3.client("s3")
        bucket, key = path.replace("s3://", "").split("/", 1)
        obj = s3.get_object(Bucket=bucket, Key=key)
        raw = obj["Body"].read()
        return pd.read_parquet(io.BytesIO(raw)) if path.endswith(".parquet") else pd.read_csv(io.BytesIO(raw))

    if source == "azure":
        from azure.storage.blob import BlobServiceClient  # type: ignore

        conn_str = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
        client = BlobServiceClient.from_connection_string(conn_str)
        container, blob = path.split("/", 1)
        raw = client.get_blob_client(container, blob).download_blob().readall()
        return pd.read_parquet(io.BytesIO(raw)) if blob.endswith(".parquet") else pd.read_csv(io.BytesIO(raw))

    if source == "gcs":
        from google.cloud import storage as gcs  # type: ignore

        client = gcs.Client()
        bucket_name, blob_name = path.replace("gs://", "").split("/", 1)
        blob = client.bucket(bucket_name).blob(blob_name)
        raw = blob.download_as_bytes()
        return pd.read_parquet(io.BytesIO(raw)) if blob_name.endswith(".parquet") else pd.read_csv(io.BytesIO(raw))

    if source == "gdrive":
        import gdown  # type: ignore

        local_tmp = Path("/tmp/sentinel_gdrive_tmp.csv")
        gdown.download(path, str(local_tmp), quiet=False)
        return pd.read_csv(local_tmp)

    if source == "onedrive":
        import requests  # type: ignore

        r = requests.get(path + "&download=1", timeout=60)
        r.raise_for_status()
        return pd.read_csv(io.BytesIO(r.content))

    if source == "huggingface":
        from datasets import load_dataset  # type: ignore

        ds = load_dataset(path, split="train")
        return ds.to_pandas()

    raise ValueError(
        f"Unknown data source: {source!r}. "
        "Choose from: local | nas | s3 | azure | gcs | gdrive | onedrive | huggingface."
    )


def make_loader_from_source(
    source: str,
    path: str,
    split: str = "train",
    batch_size: int = 32,
    target_col: str = "label",
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader from any supported source.

    Feature columns are all columns except *target_col*.
    """
    df = load_dataframe(source, path)

    features = torch.tensor(
        df.drop(columns=[target_col], errors="ignore").select_dtypes(include=["number"]).values,
        dtype=torch.float32,
    )

    if target_col in df.columns:
        labels = torch.tensor(df[target_col].values, dtype=torch.long)
        dataset = TensorDataset(features, labels)
    else:
        dataset = TensorDataset(features)

    shuffle = split == "train"
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

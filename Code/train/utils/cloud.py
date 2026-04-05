# src/train/utils/cloud.py
"""
Cloud storage helpers for all supported backends.
Import and call after training to push/pull checkpoints.
"""

import os
from pathlib import Path


def upload_to_s3(local_path, bucket, key, region="us-east-1"):
    import boto3
    s3 = boto3.client("s3", region_name=region)
    local = Path(local_path)
    if local.is_dir():
        for f in local.rglob("*"):
            if f.is_file():
                s3_key = f"{key}/{f.relative_to(local)}"
                s3.upload_file(str(f), bucket, s3_key)
                print(f"Uploaded {f} -> s3://{bucket}/{s3_key}")
    else:
        s3.upload_file(local_path, bucket, key)
        print(f"Uploaded {local_path} -> s3://{bucket}/{key}")


def download_from_s3(bucket, key, local_path):
    import boto3
    boto3.client("s3").download_file(bucket, key, local_path)
    print(f"Downloaded s3://{bucket}/{key} -> {local_path}")


def upload_to_gcs(local_path, bucket_name, blob_prefix):
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    local  = Path(local_path)
    if local.is_dir():
        for f in local.rglob("*"):
            if f.is_file():
                blob_name = f"{blob_prefix}/{f.relative_to(local)}"
                bucket.blob(blob_name).upload_from_filename(str(f))
                print(f"Uploaded {f} -> gs://{bucket_name}/{blob_name}")
    else:
        bucket.blob(blob_prefix).upload_from_filename(local_path)
        print(f"Uploaded {local_path} -> gs://{bucket_name}/{blob_prefix}")


def upload_to_azure(local_path, connection_str, container, blob_prefix=""):
    from azure.storage.blob import BlobServiceClient
    client = BlobServiceClient.from_connection_string(connection_str)
    local  = Path(local_path)
    if local.is_dir():
        for f in local.rglob("*"):
            if f.is_file():
                bn = f"{blob_prefix}/{f.relative_to(local)}" if blob_prefix else str(f.relative_to(local))
                client.get_blob_client(container, bn).upload_blob(f.read_bytes(), overwrite=True)
                print(f"Uploaded {f} -> az://{container}/{bn}")
    else:
        bn = blob_prefix if blob_prefix else local.name
        client.get_blob_client(container, bn).upload_blob(local.read_bytes(), overwrite=True)
        print(f"Uploaded {local_path} -> az://{container}/{bn}")


def download_from_azure(connection_str, container, blob_name, local_path):
    from azure.storage.blob import BlobServiceClient
    data = BlobServiceClient.from_connection_string(connection_str) \
               .get_blob_client(container, blob_name).download_blob().readall()
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    Path(local_path).write_bytes(data)
    print(f"Downloaded az://{container}/{blob_name} -> {local_path}")


def upload_to_huggingface_hub(folder_path, repo_id, token=None):
    from huggingface_hub import HfApi, login
    if token: login(token=token)
    HfApi().upload_folder(folder_path=folder_path, repo_id=repo_id, repo_type="model")


def download_from_gdrive(file_id, local_path):
    import gdown
    gdown.download(f"https://drive.google.com/uc?id={file_id}", local_path, quiet=False)
    print(f"Downloaded GDrive/{file_id} -> {local_path}")


def download_from_onedrive(share_url, local_path):
    import requests
    r = requests.get(share_url.rstrip("/") + "&download=1", stream=True, timeout=300)
    r.raise_for_status()
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded OneDrive -> {local_path}")

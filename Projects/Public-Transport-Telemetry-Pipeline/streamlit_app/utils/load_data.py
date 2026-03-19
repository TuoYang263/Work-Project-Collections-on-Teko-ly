import os
import tempfile
from pathlib import Path

import pandas as pd
from azure.storage.blob import BlobServiceClient
import streamlit as st

USE_BLOB = os.getenv("USE_BLOB_STORAGE", "true").lower() == "true"
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_BLOB_CONTAINER = os.getenv("AZURE_BLOB_CONTAINER", "telemetry-demo")

LOCAL_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "output"

def _get_blob_service_client() -> BlobServiceClient:
    if not AZURE_STORAGE_CONNECTION_STRING:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING is not set.")
    return BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)

def _list_parquet_part_blobs(prefix: str) -> list[str]:
    container_client = _get_blob_service_client().get_container_client(AZURE_BLOB_CONTAINER)

    blob_names = []
    for blob in container_client.list_blobs(name_starts_with=prefix):
        name = blob.name

        # keep only actual parquet data parts
        filename = name.split("/")[-1]
        if filename.startswith("part-") and filename.endswith(".parquet"):
            blob_names.append(name)
    
    return sorted(blob_names)

def _read_parquet_dataset_from_blob(prefix: str) -> pd.DataFrame:
    blob_names = _list_parquet_part_blobs(prefix)

    if not blob_names:
        raise FileNotFoundError(f"No parquet part files found under prefix: {prefix}")
    
    service = _get_blob_service_client()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        frames = []

        for blob_name in blob_names:
            blob_client = service.get_blob_client(
                container=AZURE_BLOB_CONTAINER,
                blob=blob_name,
            )

            local_file = tmp_path / Path(blob_name).name
            with open(local_file, "wb") as f:
                f.write(blob_client.download_blob().readall())

            frames.append(pd.read_parquet(local_file))

    if not frames:
        return pd.DataFrame()
    
    return pd.concat(frames, ignore_index=True)

def _read_local_parquet_dataset(dataset_name: str) -> pd.DataFrame:
    return pd.read_parquet(LOCAL_DATA_DIR / dataset_name)

# Cache the function result to avoid reloading data on every rerun.
# ttl=900 means the cache is valid for 900 seconds (15 minutes).
@st.cache_data(ttl=900)
def load_route_window():
    if USE_BLOB:
        return _read_parquet_dataset_from_blob("telemetry/gold_route_window.parquet/")
    return _read_local_parquet_dataset("gold_route_window.parquet")

@st.cache_data(ttl=900)
def load_route_daily():
    if USE_BLOB:
        return _read_parquet_dataset_from_blob("telemetry/gold_route_daily.parquet/")
    return _read_local_parquet_dataset("gold_route_daily.parquet")

@st.cache_data(ttl=900)
def load_pipeline_metrics():
    if USE_BLOB:
        return _read_parquet_dataset_from_blob("telemetry/pipeline_metrics.parquet/")
    return _read_local_parquet_dataset("pipeline_metrics.parquet")
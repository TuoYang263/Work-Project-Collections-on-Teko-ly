from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
from azure.storage.blob import BlobServiceClient


USE_BLOB = os.getenv("USE_BLOB_STORAGE", "true").lower() == "true"
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_BLOB_CONTAINER = os.getenv("AZURE_BLOB_CONTAINER", "telemetry-demo")


def _get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _get_hsl_gold_dir() -> Path:
    return _get_project_root() / "data" / "gold" / "hsl"

def _get_weather_gold_dir() -> Path:
    return _get_project_root() / "data" / "gold" / "weather"

def _get_blob_service_client() -> BlobServiceClient:
    if not AZURE_STORAGE_CONNECTION_STRING:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING is not set.")
    return BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)


def _list_parquet_part_blobs(prefix: str) -> list[str]:
    container_client = _get_blob_service_client().get_container_client(AZURE_BLOB_CONTAINER)

    blob_names = []
    for blob in container_client.list_blobs(name_starts_with=prefix):
        filename = blob.name.split("/")[-1]
        if filename.startswith("part-") and filename.endswith(".parquet"):
            blob_names.append(blob.name)

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


def _read_local_hsl_dataset(dataset_name: str) -> pd.DataFrame:
    return pd.read_parquet(_get_hsl_gold_dir() / dataset_name)

def _read_local_weather_dataset(dataset_name: str) -> pd.DataFrame:
    return pd.read_parquet(_get_weather_gold_dir() / dataset_name)

def load_hsl_df_map() -> pd.DataFrame:
    if USE_BLOB:
        return _read_parquet_file_from_blob("telemetry/hsl/hsl_df_map.parquet")
    return _read_local_hsl_dataset("hsl_df_map.parquet")


def load_hsl_map_points() -> pd.DataFrame:
    if USE_BLOB:
        return _read_parquet_file_from_blob("telemetry/hsl/hsl_map_points.parquet")
    return _read_local_hsl_dataset("hsl_map_points.parquet")


def load_hsl_route_paths() -> pd.DataFrame:
    if USE_BLOB:
        return _read_parquet_file_from_blob("telemetry/hsl/hsl_route_paths.parquet")
    return _read_local_hsl_dataset("hsl_route_paths.parquet")


def load_hsl_route_options() -> pd.DataFrame:
    if USE_BLOB:
        return _read_parquet_file_from_blob("telemetry/hsl/hsl_route_options.parquet")
    return _read_local_hsl_dataset("hsl_route_options.parquet")

def _read_parquet_file_from_blob(blob_name: str) -> pd.DataFrame:
    service = _get_blob_service_client()
    blob_client = service.get_blob_client(
        container=AZURE_BLOB_CONTAINER,
        blob=blob_name,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"[Blob] Reading parquet file: {blob_name}")
        local_file = Path(tmp_dir) / Path(blob_name).name
        with open(local_file, "wb") as f:
            f.write(blob_client.download_blob().readall())

        return pd.read_parquet(local_file)

def load_weather_stations() -> pd.DataFrame:
    if USE_BLOB:
        return _read_parquet_file_from_blob(
            "telemetry/weather/weather_stations_latest.parquet"
        )
    return _read_local_weather_dataset("weather_stations_latest.parquet")
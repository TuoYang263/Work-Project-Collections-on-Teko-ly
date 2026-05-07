from __future__ import annotations

import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st
from azure.storage.blob import BlobServiceClient

METADATA_BLOB_PATH = "dashboard/refresh_metadata.json"
HELSINKI_TZ = ZoneInfo("Europe/Helsinki")


def _get_container_name() -> str | None:
    return os.getenv("AZURE_CONTAINER_NAME") or os.getenv("AZURE_BLOB_CONTAINER")


@st.cache_data(ttl=300, show_spinner=False)
def load_refresh_metadata() -> dict | None:
    """
    Load lightweight refresh metadata from Azure Blob.

    This metadata is written by Azure Function and only describes dashboard
    heartbeat/check status. It does not represent a Gold-layer data refresh.
    """
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = _get_container_name()

    if not connection_string or not container_name:
        return None

    try:
        blob_service = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service.get_blob_client(
            container=container_name,
            blob=METADATA_BLOB_PATH,
        )

        raw = blob_client.download_blob().readall()
        return json.loads(raw.decode("utf-8"))

    except Exception:
        return None


def format_metadata_check_time(metadata: dict | None) -> str:
    if not metadata:
        return "N/A"

    raw_ts = metadata.get("latest_metadata_check_at_helsinki")
    if not raw_ts:
        return "N/A"

    try:
        ts = datetime.fromisoformat(raw_ts)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=HELSINKI_TZ)
        else:
            ts = ts.astimezone(HELSINKI_TZ)

        return ts.strftime("%Y-%m-%d %H:%M Helsinki time")

    except Exception:
        return "N/A"

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import azure.functions as func
from azure.storage.blob import BlobServiceClient, ContentSettings

app = func.FunctionApp()

HELSINKI_TZ = ZoneInfo("Europe/Helsinki")
METADATA_BLOB_PATH = "dashboard/refresh_metadata.json"


def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@app.timer_trigger(
    schedule="%LIGHTWEIGHT_METADATA_REFRESH_CRON%",
    arg_name="timer",
    run_on_startup=False,
    use_monitor=True,
)
def lightweight_refresh_metadata(timer: func.TimerRequest) -> None:
    """
    Update lightweight dashboard refresh metadata.

    This function does not run the Spark/Delta pipeline.
    It does not trigger Databricks.
    It does not fetch HSL/FMI data.
    It only writes a small JSON metadata file to Azure Blob Storage.

    Full Bronze -> Silver -> Gold refresh is handled separately by the
    scheduled Azure Databricks Job.
    """
    if timer.past_due:
        logging.warning("Timer trigger is past due.")

    connection_string = get_required_env("AZURE_STORAGE_CONNECTION_STRING")
    container_name = get_required_env("AZURE_CONTAINER_NAME")

    now_utc = datetime.now(timezone.utc)
    now_helsinki = now_utc.astimezone(HELSINKI_TZ)

    metadata = {
        "refresh_model": "daily_databricks_snapshot_plus_lightweight_metadata",
        "latest_metadata_check_at_utc": now_utc.isoformat(),
        "latest_metadata_check_at_helsinki": now_helsinki.isoformat(),
        "metadata_blob_path": METADATA_BLOB_PATH,
        "serving_layer": "azure_blob_parquet",
        "is_live_monitor": False,
        "full_refresh_owner": "azure_databricks_scheduled_job",
        "lightweight_check_owner": "azure_function_timer_trigger",
        "notes": (
            "This metadata check does not run the Spark pipeline, trigger Databricks, "
            "fetch HSL/FMI data, or refresh Gold-layer dashboard outputs. "
            "Full Bronze-Silver-Gold refresh is handled by the scheduled Azure Databricks Job."
        ),
    }

    blob_service = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service.get_blob_client(
        container=container_name,
        blob=METADATA_BLOB_PATH,
    )

    blob_client.upload_blob(
        json.dumps(metadata, indent=2),
        overwrite=True,
        content_settings=ContentSettings(content_type="application/json"),
    )

    logging.info("Updated %s in container %s", METADATA_BLOB_PATH, container_name)

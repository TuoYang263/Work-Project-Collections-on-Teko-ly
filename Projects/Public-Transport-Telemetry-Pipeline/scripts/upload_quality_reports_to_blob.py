"""
Upload generated data quality and source validation reports to Azure Blob Storage.

This script is intended to run inside the scheduled Azure Container Apps Job
after pipeline refresh, Gold export, and validation checks have completed.

It uploads read-only JSON validatioin artifacts from:

    data/quality/reports/

to:

    quality/reports/

inside the configured Azure Blob Container.

The Streamlit dashboard can then read these reports remotely with local fallback.
"""

from __future__ import annotations

import os
from pathlib import Path

from azure.storage.blob import BlobServiceClient, ContentSettings

LOCAL_REPORT_DIR = Path("data/quality/reports")
REMOTE_REPORT_PREFIX = "quality/reports"

REQUIRED_REPORTS = [
    "pipeline_quality_report.json",
    "latest_quality_summary.json",
    "hsl_source_validation_report.json",
    "latest_hsl_source_summary.json",
    "fmi_source_validation_report.json",
    "latest_fmi_source_summary.json",
]


def get_blob_service_client() -> BlobServiceClient:
    """
    Create BlobServiceClient from Azure Stroage connection string.

    Excepted environment variable:
        AZURE_STORAGE_CONNECTION_STRING
    """
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

    if not connection_string:
        raise RuntimeError(
            "Missing AZURE_STORAGE_CONNECTION_STRING environment variable. "
            "Set it in the local shell or Container Apps environment."
        )

    return BlobServiceClient.from_connection_string(connection_string)


def get_container_name() -> str:
    """
    Read Azure Blob container name from environment.

    Supported environment variables:
        AZURE_BLOB_CONTAINER
        AZURE_STORAGE_CONTAINER_NAME

    AZURE_BLOB_CONTAINER is used by the existing output upload workflow.
    AZURE_STORAGE_CONTAINER_NAME is kept as a compatible fallback name.
    """
    container_name = os.getenv("AZURE_BLOB_CONTAINER") or os.getenv(
        "AZURE_STORAGE_CONTAINER_NAME"
    )

    if not container_name:
        raise RuntimeError(
            "Missing Azure Blob container environment variable. "
            "Set AZURE_BLOB_CONTAINER or AZURE_STORAGE_CONTAINER_NAME."
        )

    return container_name


def validate_required_reports() -> list[Path]:
    """
    Ensure all required validation report files exist before upload.

    The scheduled refresh should fail clearly if a validation step did not
    generate its excepted artifact. This prevents the dashboard from showing
    incomplete or stale quality evidence as if it were complete.
    """
    missing_files: list[Path] = []
    report_paths: list[Path] = []

    for filename in REQUIRED_REPORTS:
        report_path = LOCAL_REPORT_DIR / filename

        if report_path.exists():
            report_paths.append(report_path)
        else:
            missing_files.append(report_path)

    if missing_files:
        missing_list = "\n".join(f" - {path}" for path in missing_files)
        raise FileNotFoundError(
            "Missing required validation report files. "
            "The upload step was stoppped to avoid publishing incomplete "
            "quality artifacts:\n"
            f"{missing_list}"
        )

    return report_paths


def upload_report_file(
    blob_service_client: BlobServiceClient,
    container_name: str,
    local_path: Path,
) -> None:
    """
    Upload one JSON validation report to Azure Blob Storage.

    Local path:
        data/quality/reports/<filename>.json

    Remote blob path:
        quality/reports/<filename>.json
    """
    blob_name = f"{REMOTE_REPORT_PREFIX}/{local_path.name}"

    blob_client = blob_service_client.get_blob_client(
        container=container_name,
        blob=blob_name,
    )

    with local_path.open("rb") as file_obj:
        blob_client.upload_blob(
            file_obj,
            overwrite=True,
            content_settings=ContentSettings(content_type="application/json"),
        )

    print(f"Uploaded {local_path} -> {blob_name}")


def main() -> None:
    print("Validating required quality report artifacts...")
    report_paths = validate_required_reports()

    print("Connecting to Azure Blob Storage...")
    blob_service_client = get_blob_service_client()
    container_name = get_container_name()

    print(
        f"Uploading {len(report_paths)} quality report files to container '{container_name}'..."
    )
    for report_path in report_paths:
        upload_report_file(
            blob_service_client=blob_service_client,
            container_name=container_name,
            local_path=report_path,
        )

    print("Quality report upload completed successfully.")


if __name__ == "__main__":
    main()

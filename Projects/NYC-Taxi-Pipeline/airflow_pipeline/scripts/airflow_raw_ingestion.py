"""Airflow adapter for NYC V2 S3 raw ingestion."""

import os
import re
from pathlib import Path


def extract_raw_months(execution_contract: dict) -> list[str]:
    """Extract and validate raw months for full refresh."""

    if execution_contract.get("mode") != "CONFIGURED_FULL_REFRESH":
        raise ValueError("Raw ingestion requires CONFIGURED_FULL_REFRESH")

    read_scope = execution_contract.get("read_scope", {})
    months = read_scope.get("raw_file_months")

    if not isinstance(months, list) or not months:
        raise ValueError("read_scope.raw_file_months must be a non-empty list")

    for month in months:
        if not isinstance(month, str) or not re.fullmatch(
            r"\d{4}-(0[1-9]|1[0-2])", month
        ):
            raise ValueError(f"Invalid raw file month: {month!r}")

        if int(month[:4]) < 2009:
            raise ValueError(f"Unsupported TLC data year: {month}")

    if len(months) != len(set(months)):
        raise ValueError("Duplicate raw file months")

    return months


def stage_raw_months(execution_contract: dict) -> list[dict]:
    """Stage validated local Parquet files for required raw months."""

    from scripts.load_nyc_yellow_taxi_data import download_to_local

    raw_months = extract_raw_months(execution_contract)
    staged_files = []

    for data_month in raw_months:
        year, month = map(int, data_month.split("-"))

        paths = download_to_local(year, [month])

        if not isinstance(paths, list) or len(paths) != 1:
            raise RuntimeError(f"Required raw file unavailable: {data_month}")

        local_path = Path(paths[0])
        expected_name = f"yellow_tripdata_{data_month}.parquet"

        if local_path.name != expected_name:
            raise ValueError(f"Raw file month mismatch: {data_month}")

        if not local_path.is_file():
            raise FileNotFoundError(f"Staged raw file missing: {local_path}")

        staged_files.append(
            {
                "data_month": data_month,
                "local_path": str(local_path),
            }
        )

    return staged_files


def ingest_staged_months(
    staged_files: list[dict],
    s3_client,
    bucket: str,
) -> list[dict]:
    """Ingest validated staged files into S3."""

    from scripts.s3_raw_ingestion import ingest_month

    results = []

    for staged in staged_files:
        data_month = staged["data_month"]
        year, month = map(int, data_month.split("-"))

        result = ingest_month(
            s3_client=s3_client,
            bucket=bucket,
            year=year,
            month=month,
            local_path=staged["local_path"],
        )

        results.append(result)

    return results


def run_raw_ingestion(execution_contract: dict) -> list[dict]:
    """Run raw ingestion and return Airflow XCom-safe references."""

    from scripts.s3_raw_ingestion import create_s3_client

    bucket = os.getenv("NYC_V2_RAW_BUCKET")

    if not bucket:
        raise ValueError("NYC_V2_RAW_BUCKET is not configured")

    # Step 1: Download and validate required raw months.
    staged_files = stage_raw_months(execution_contract)

    # Step 2: Create the AWS client only when executing the task.
    s3_client = create_s3_client(
        region=os.getenv("AWS_REGION", "eu-central-1"),
        profile=os.getenv("AWS_PROFILE") or None,
    )

    # Step 3: Upload/reuse S3 objects and publish manifests.
    results = ingest_staged_months(
        staged_files=staged_files,
        s3_client=s3_client,
        bucket=bucket,
    )

    if len(results) != len(staged_files):
        raise RuntimeError("Raw ingestion result count mismatch")

    # Step 4: Return references, not local file paths.
    references = []

    for staged, result in zip(staged_files, results):
        if result["data_month"] != staged["data_month"]:
            raise RuntimeError("Raw ingestion month mismatch")

        references.append(
            {
                "data_month": result["data_month"],
                "bucket": bucket,
                "raw_key": result["raw_key"],
                "raw_version_id": result["raw_version_id"],
                "manifest_key": result["manifest_key"],
                "raw_status": result["raw_status"],
                "manifest_status": result["manifest_status"],
            }
        )

    return references

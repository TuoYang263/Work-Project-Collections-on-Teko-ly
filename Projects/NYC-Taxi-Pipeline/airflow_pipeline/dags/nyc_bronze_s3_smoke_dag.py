"""Isolated NYC V2 S3-to-Bronze Airflow smoke test."""

import os
from datetime import datetime
from pathlib import Path

import boto3
from airflow import DAG
from airflow.operators.python import PythonOperator

from scripts.pipeline_runner import submit_stage
from scripts.run_lifecycle import (
    start_pipeline_run,
    mark_pipeline_success,
    mark_pipeline_failure,
)
from scripts.s3_raw_ingestion import (
    build_raw_object_key,
    build_raw_manifest_key,
)

BUCKET = "nyc-v2-raw-438336772967-eu-central-1-an"
BRONZE = Path("/opt/airflow/data/delta/bronze/trips")

CONTRACT = {
    "mode": "CONFIGURED_FULL_REFRESH",
    "logical_window": {
        "start": "2024-01-01",
        "end_exclusive": "2024-02-01",
        "anchor": "PICKUP_DATETIME",
    },
    "read_scope": {
        "raw_file_months": ["2024-01"],
        "boundary_policy": "SMOKE_SINGLE_MONTH",
    },
    "write_scope": "FULL_TABLE_OVERWRITE",
    "rerun_semantics": "ISOLATED_SMOKE_OVERWRITE",
}


def begin(**context):
    if context["dag_run"].conf:
        raise ValueError("Smoke DAG does not accept overrides")
    start_pipeline_run(context, CONTRACT)


def prepare_reference():
    if os.environ.get("NYC_V2_RAW_BUCKET") != BUCKET:
        raise RuntimeError("Unexpected S3 bucket")

    s3 = boto3.client("s3", region_name="eu-central-1")
    key = build_raw_object_key(2024, 1)
    metadata = s3.head_object(Bucket=BUCKET, Key=key)

    if metadata["ContentLength"] != 49961641:
        raise ValueError("Unexpected S3 object size")

    version = metadata.get("VersionId")
    if not version or version == "null":
        raise ValueError("Missing S3 VersionId")

    return [{
        "data_month": "2024-01",
        "bucket": BUCKET,
        "raw_key": key,
        "raw_version_id": version,
        "manifest_key": build_raw_manifest_key(2024, 1),
    }]


def execute_bronze(raw_references):
    if not BRONZE.is_mount():
        raise RuntimeError("Bronze isolation mount is missing")

    if (BRONZE / "_delta_log").exists():
        raise RuntimeError("Smoke target already contains Delta")

    if boto3.Session().get_credentials().method != "login":
        raise RuntimeError("Expected AWS LoginProvider")

    submit_stage(
        stage="bronze",
        execution_contract=CONTRACT,
        raw_references=raw_references,
    )


def verify_bronze():
    if not BRONZE.is_mount():
        raise RuntimeError("Bronze isolation mount is missing")

    if not (BRONZE / "_delta_log").is_dir():
        raise RuntimeError("Missing Delta transaction log")

    if not any(BRONZE.rglob("*.parquet")):
        raise RuntimeError("Missing Bronze Parquet output")

    print("ISOLATED BRONZE DELTA OUTPUT: PASS")


def on_failure(context):
    try:
        mark_pipeline_failure(context)
    except FileNotFoundError:
        print("Run Evidence not created before failure")


with DAG(
    dag_id="nyc_bronze_s3_smoke",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 0},
    on_success_callback=mark_pipeline_success,
    on_failure_callback=on_failure,
    tags=["nyc_v2", "s3", "bronze", "smoke"],
) as dag:
    start = PythonOperator(
        task_id="start_run_evidence",
        python_callable=begin,
    )

    reference = PythonOperator(
        task_id="prepare_s3_reference",
        python_callable=prepare_reference,
    )

    bronze = PythonOperator(
        task_id="bronze_ingest_delta",
        python_callable=execute_bronze,
        op_kwargs={"raw_references": reference.output},
    )

    verify = PythonOperator(
        task_id="verify_bronze_output",
        python_callable=verify_bronze,
    )

    start >> reference >> bronze >> verify

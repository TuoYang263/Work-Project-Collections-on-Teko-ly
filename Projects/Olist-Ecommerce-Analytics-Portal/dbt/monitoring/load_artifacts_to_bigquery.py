import os

from google.cloud import bigquery

from artifact_parser import (
    ARTIFACT_DIR,
    build_model_column_snapshot_records,
    build_model_metadata_snapshot_records,
    build_model_run_result_records,
    build_pipeline_run_record,
    build_test_run_result_records,
    load_json,
)

DEFAULT_MONITORING_DATASET_ID = "olist_monitoring"


def get_bigquery_client() -> bigquery.Client:
    project_id = os.getenv("GCP_PROJECT_ID")

    if project_id:
        return bigquery.Client(project=project_id)

    return bigquery.Client()


def insert_records(
    client: bigquery.Client,
    project_id: str,
    dataset_id: str,
    table_name: str,
    records: list[dict],
) -> None:
    if not records:
        print(f"No records to insert into {table_name}.")
        return

    table_id = f"{project_id}.{dataset_id}.{table_name}"
    errors = client.insert_rows_json(table_id, records)

    if errors:
        raise RuntimeError(f"Failed to insert records into {table_id}: {errors}")

    print(f"Inserted {len(records)} records into {table_id}.")


def main() -> None:
    client = get_bigquery_client()
    project_id = client.project
    dataset_id = os.getenv(
        "MONITORING_DATASET_ID",
        DEFAULT_MONITORING_DATASET_ID,
    )

    manifest = load_json(ARTIFACT_DIR / "manifest.json")
    run_results = load_json(ARTIFACT_DIR / "run_results.json")
    catalog = load_json(ARTIFACT_DIR / "catalog.json")

    pipeline_run_record = build_pipeline_run_record(
        manifest=manifest,
        run_results=run_results,
        catalog=catalog,
    )

    model_run_records = build_model_run_result_records(
        manifest=manifest,
        run_results=run_results,
        pipeline_run_record=pipeline_run_record,
    )

    test_run_records = build_test_run_result_records(
        manifest=manifest,
        run_results=run_results,
        pipeline_run_record=pipeline_run_record,
    )

    model_metadata_records = build_model_metadata_snapshot_records(
        manifest=manifest,
        catalog=catalog,
        pipeline_run_record=pipeline_run_record,
    )

    model_column_records = build_model_column_snapshot_records(
        manifest=manifest,
        catalog=catalog,
        pipeline_run_record=pipeline_run_record,
    )

    insert_records(
        client=client,
        project_id=project_id,
        dataset_id=dataset_id,
        table_name="pipeline_runs",
        records=[pipeline_run_record],
    )

    insert_records(
        client=client,
        project_id=project_id,
        dataset_id=dataset_id,
        table_name="model_run_results",
        records=model_run_records,
    )

    insert_records(
        client=client,
        project_id=project_id,
        dataset_id=dataset_id,
        table_name="test_run_results",
        records=test_run_records,
    )

    insert_records(
        client=client,
        project_id=project_id,
        dataset_id=dataset_id,
        table_name="model_metadata_snapshots",
        records=model_metadata_records,
    )

    insert_records(
        client=client,
        project_id=project_id,
        dataset_id=dataset_id,
        table_name="model_column_snapshots",
        records=model_column_records,
    )


if __name__ == "__main__":
    main()

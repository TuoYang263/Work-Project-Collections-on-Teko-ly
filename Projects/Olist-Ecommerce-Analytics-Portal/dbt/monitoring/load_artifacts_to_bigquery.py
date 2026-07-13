import os

from google.cloud import bigquery

from artifact_parser import (
    ARTIFACT_DIR,
    build_pipeline_run_record,
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

    insert_records(
        client=client,
        project_id=project_id,
        dataset_id=dataset_id,
        table_name="pipeline_runs",
        records=[pipeline_run_record],
    )


if __name__ == "__main__":
    main()

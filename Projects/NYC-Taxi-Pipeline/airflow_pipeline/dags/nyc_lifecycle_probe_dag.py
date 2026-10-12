"""Small CeleryExecutor test DAG. No Spark, network ingestion or BigQuery."""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from scripts.run_lifecycle import (
    start_pipeline_run, mark_pipeline_success, mark_pipeline_failure,
)


def start_probe(**context):
    contract = {
        "mode": "BACKFILL", "execution_scope": "GOLD_ONLY",
        "logical_window": {"start": "2023-02-01", "end_exclusive": "2023-03-01", "anchor": "PICKUP_DATETIME"},
        "read_scope": {"input_layer": "SILVER", "boundary_policy": "REUSE_EXISTING_SILVER_STATE"},
        "write_scope": "WINDOW_REPLACE", "rerun_semantics": "REPLACE_LOGICAL_WINDOW_IDEMPOTENT",
    }
    return start_pipeline_run(context, contract)


def harmless_task(**context):
    conf = context["dag_run"].conf or {}
    if conf.get("probe_fail") is True:
        raise RuntimeError("Intentional lifecycle probe failure")
    if conf.get("probe_retry") is True:
        ti = context["ti"]
        print(f"[PROBE] Attempt: {ti.try_number}")
        if ti.try_number == 1:
            raise RuntimeError("Intentional first-attempt failure")
    print("Lifecycle probe executed successfully")


def on_probe_success(context):
    if (context["dag_run"].conf or {}).get("probe_callback_fail") is True:
        raise RuntimeError("Intentional success callback failure")
    mark_pipeline_success(context)


with DAG(
    dag_id="nyc_lifecycle_probe", start_date=datetime(2024, 1, 1),
    schedule_interval=None, catchup=False, default_args={"retries": 0},
    on_success_callback=on_probe_success,
    on_failure_callback=mark_pipeline_failure,
    tags=["nyc_v2", "integration_test"],
) as dag:
    start = PythonOperator(task_id="start_run_evidence", python_callable=start_probe)
    execute = PythonOperator(
        task_id="harmless_task", python_callable=harmless_task,
        retries=1, retry_delay=timedelta(seconds=30),
    )
    start >> execute

# nyc_taxi_medallion_dag.py
# Airflow DAG: Medallion Pipeline with Delta Lake → export Gold summary to BigQuery

import os
import sys

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import (
    BranchPythonOperator,
    PythonOperator,
)

# Ensure project root & scripts/ are importable
DAG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(DAG_DIR, os.pardir))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

for p in [SCRIPTS_DIR, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.append(p)

# Project imports must come AFTER sys.path setup.
from scripts import config as project_config
from scripts.execution_contract import (
    build_current_execution_contract,
)
from scripts.pipeline_runner import submit_stage

from scripts import get_logger  # or scripts.logger, both ok
logger = get_logger("nyc_taxi_medallion_dag")

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}


def validate_execution_contract(**context):
    dag_run = context.get("dag_run")

    dag_run_conf = (
        dict(dag_run.conf or {})
        if dag_run is not None
        else {}
    )

    contract = build_current_execution_contract(
        settings=project_config.SETTINGS,
        dag_run_conf=dag_run_conf,
    )

    airflow_run_id = (
        dag_run.run_id
        if dag_run is not None
        else "<no-dag-run>"
    )

    logger.info(
        f"[CONTROL] airflow_run_id={airflow_run_id}"
    )

    logger.info(
        f"[CONTROL] execution_contract={contract}"
    )

    return contract


def route_execution(
    execution_contract: dict,
    **_,
):
    mode = execution_contract.get("mode")
    write_scope = execution_contract.get(
        "write_scope"
    )

    if mode == "CONFIGURED_FULL_REFRESH":
        if write_scope != "FULL_TABLE_OVERWRITE":
            raise ValueError(
                "CONFIGURED_FULL_REFRESH requires "
                "write_scope='FULL_TABLE_OVERWRITE'"
            )

        return "bronze_ingest_delta"

    if mode == "BACKFILL":
        execution_scope = execution_contract.get(
            "execution_scope"
        )

        if execution_scope != "GOLD_ONLY":
            raise ValueError(
                "BACKFILL currently supports only "
                "execution_scope='GOLD_ONLY'"
            )

        if write_scope != "WINDOW_REPLACE":
            raise ValueError(
                "GOLD_ONLY backfill requires "
                "write_scope='WINDOW_REPLACE'"
            )

        return "gold_backfill_delta"

    raise ValueError(
        f"Unsupported execution mode: {mode!r}"
    )


with DAG(
    dag_id="nyc_taxi_medallion_delta",
    default_args=default_args,
    description="NYC Yellow Taxi Medallion Pipeline with Delta Lake",
    schedule_interval=None,   # manual trigger for now
    catchup=False,
    max_active_runs=1,
    tags=["nyc_taxi", "delta", "medallion"],
) as dag:
    execution_contract = PythonOperator(
        task_id="validate_execution_contract",
        python_callable=validate_execution_contract,
        retries=0,
    )

    route = BranchPythonOperator(
        task_id="route_execution",
        python_callable=route_execution,
        op_kwargs={
            "execution_contract":
                execution_contract.output,
        },
        retries=0,
    )

    gold_backfill = PythonOperator(
        task_id="gold_backfill_delta",
        python_callable=submit_stage,
        op_kwargs={
            "stage": "gold",
            "execution_contract":
                execution_contract.output,
        },
    )

    bronze = PythonOperator(
        task_id="bronze_ingest_delta",
        python_callable=submit_stage,
        op_kwargs={
            "stage": "bronze",
            "execution_contract": execution_contract.output,
        },
    )

    silver = PythonOperator(
        task_id="silver_clean_delta",
        python_callable=submit_stage,
        op_kwargs={
            "stage": "silver",
            "execution_contract": execution_contract.output,
        },
    )

    gold = PythonOperator(
        task_id="gold_aggregate_delta",
        python_callable=submit_stage,
        op_kwargs={
            "stage": "gold",
            "execution_contract": execution_contract.output,
        },
    )

    export_bq = PythonOperator(
        task_id="export_gold_to_bigquery",
        python_callable=submit_stage,
        op_kwargs={
            "stage": "export",
            "execution_contract": execution_contract.output,
        },
    )

    execution_contract >> route

    route >> bronze
    bronze >> silver >> gold >> export_bq

    route >> gold_backfill
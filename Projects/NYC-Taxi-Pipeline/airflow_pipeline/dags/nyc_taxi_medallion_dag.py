# nyc_taxi_medallion_dag.py
# Airflow DAG: Medallion Pipeline with Delta Lake → export Gold summary to BigQuery

import os
import sys

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

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

    bronze = PythonOperator(
        task_id="bronze_ingest_delta",
        python_callable=submit_stage,
        op_kwargs={"stage": "bronze"},
    )

    silver = PythonOperator(
        task_id="silver_clean_delta",
        python_callable=submit_stage,
        op_kwargs={"stage": "silver"},
    )

    gold = PythonOperator(
        task_id="gold_aggregate_delta",
        python_callable=submit_stage,
        op_kwargs={"stage": "gold"},
    )

    export_bq = PythonOperator(
        task_id="export_gold_to_bigquery",
        python_callable=submit_stage,
        op_kwargs={"stage": "export"},
    )

    execution_contract >> bronze >> silver >> gold >> export_bq
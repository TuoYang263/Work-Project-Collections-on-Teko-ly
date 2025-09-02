# nyc_taxi_medallion_dag.py
# Airflow DAG: Medallion Pipeline with Delta Lake → export Gold summary to BigQuery

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# Ensure project root & scripts/ are importable
DAG_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(DAG_DIR))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

for p in [PROJECT_ROOT, SCRIPTS_DIR]:
    if p not in sys.path:
        sys.path.append(p)

from logger import get_logger  # or scripts.logger, both ok
logger = get_logger("nyc_taxi_medallion_dag")

# Robust import of pipeline module
try:
    import scripts.delta_medallion_pipeline as pipeline  # type: ignore
except Exception:
    import delta_medallion_pipeline as pipeline  # type: ignore

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}

with DAG(
    dag_id="nyc_taxi_medallion_delta",
    default_args=default_args,
    description="NYC Yellow Taxi Medallion Pipeline with Delta Lake",
    schedule_interval=None,   # manual trigger for now
    catchup=False,
    tags=["nyc_taxi", "delta", "medallion"],
) as dag:

    bronze = PythonOperator(
        task_id="bronze_ingest_delta",
        python_callable=pipeline.bronze_task,
    )

    silver = PythonOperator(
        task_id="silver_clean_delta",
        python_callable=pipeline.silver_task,
    )

    gold = PythonOperator(
        task_id="gold_aggregate_delta",
        python_callable=pipeline.gold_task,
    )

    export_bq = PythonOperator(
        task_id="export_gold_to_bigquery",
        python_callable=pipeline.export_bq_task,
    )

    bronze >> silver >> gold >> export_bq
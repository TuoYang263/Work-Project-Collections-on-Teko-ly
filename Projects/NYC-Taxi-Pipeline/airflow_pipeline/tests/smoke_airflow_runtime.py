
"""
Unit 3A.3 — Airflow runtime smoke test.

Verifies:
1. DAG imports without loading the heavy PySpark pipeline.
2. Task dependencies and max_active_runs are configured.
3. Unsupported backfill parameters are rejected.
4. A real child-process failure propagates through
   the lightweight runner and Airflow PythonOperator.

No Spark jobs or Delta writes are performed.
"""

import subprocess
import sys
import json
from types import SimpleNamespace
from unittest.mock import patch

from airflow.models import DagBag

from scripts import pipeline_runner as runner


DAG_ID = "nyc_taxi_medallion_delta"

EXPECTED_TASKS = [
    "validate_execution_contract",
    "bronze_ingest_delta",
    "silver_clean_delta",
    "gold_aggregate_delta",
    "export_gold_to_bigquery",
]


def assert_no_heavy_pipeline_import():
    assert not any(
        name.endswith("delta_medallion_pipeline")
        for name in sys.modules
    ), "Heavy PySpark pipeline was imported unexpectedly"


def test_dag_loading():
    print("\n[1/3] Checking Airflow DAG loading...")

    bag = DagBag(
        dag_folder=(
            "/opt/airflow/dags/"
            "nyc_taxi_medallion_dag.py"
        ),
        include_examples=False,
        safe_mode=False,
    )

    assert not bag.import_errors, bag.import_errors

    dag = bag.dags.get(DAG_ID)

    assert dag is not None, f"DAG not found: {DAG_ID}"

    assert dag.max_active_runs == 1

    assert set(dag.task_ids) == set(EXPECTED_TASKS)

    for task_id in EXPECTED_TASKS[1:]:
        task = dag.get_task(task_id)

        assert "execution_contract" in task.op_kwargs

        contract_arg = task.op_kwargs[
            "execution_contract"
        ]

        assert (
            contract_arg.operator.task_id
            == "validate_execution_contract"
        )

    for upstream, downstream in zip(
        EXPECTED_TASKS,
        EXPECTED_TASKS[1:],
    ):
        task = dag.get_task(upstream)

        assert downstream in task.downstream_task_ids, (
            f"Missing dependency: {upstream} -> {downstream}"
        )

    assert_no_heavy_pipeline_import()

    print("PASS: DAG imports without the heavy pipeline")
    print("PASS: Task dependencies")
    print("PASS: max_active_runs=1")

    return dag


def test_execution_contract(dag):
    print("\n[2/3] Checking execution contract...")

    contract_task = dag.get_task(
        "validate_execution_contract"
    )

    assert contract_task.retries == 0

    # An unsupported February override must fail.
    invalid_run = SimpleNamespace(
        conf={"month": 2},
        run_id="smoke_invalid_window",
    )

    try:
        contract_task.execute(
            context={
                "dag_run": invalid_run,
            }
        )

    except ValueError as exc:
        assert "Window/backfill overrides" in str(exc)

        print(
            "PASS: Unsafe window rejected before Spark"
        )

    else:
        raise AssertionError(
            "Unsafe window request was not rejected"
        )

    assert_no_heavy_pipeline_import()


def test_failure_propagation(dag):
    print("\n[3/3] Checking child-process failure...")

    gold_task = dag.get_task(
        "gold_aggregate_delta"
    )

    valid_contract = {
        "mode": "CONFIGURED_FULL_REFRESH",
        "year": 2023,
        "months": [1, 2, 3],
        "window_start": "2023-01-01",
        "window_end_exclusive": "2023-04-01",
        "logical_window": {
            "start": "2023-01-01",
            "end_exclusive": "2023-04-01",
            "anchor": "PICKUP_DATETIME",
        },
        "read_scope": {
            "raw_file_months": [
                "2023-01",
                "2023-02",
                "2023-03",
            ],
            "boundary_policy":
                "CONFIGURED_MONTHS_ONLY",
        },
        "write_scope": "FULL_TABLE_OVERWRITE",
        "state_mode": "STATELESS",
        "rerun_semantics":
            "RECOMPUTE_CONFIGURED_WINDOW",
    }

    gold_task.op_kwargs = {
        "stage": "gold",
        "execution_contract": valid_contract,
    }

    real_run = subprocess.run

    def injected_failure(command, **kwargs):
        # Confirm that Airflow submitted the Gold stage
        # through the lightweight runner.
        assert command[:4] == [
            sys.executable,
            "-m",
            "scripts.pipeline_runner",
            "gold",
        ]

        assert command[4] == (
            "--execution-contract-json"
        )

        passed_contract = json.loads(
            command[5]
        )

        assert passed_contract == valid_contract

        assert kwargs["check"] is True

        # Run a harmless child process that deliberately
        # exits with a non-zero status.
        # Do NOT launch the actual Spark workload.
        return real_run(
            [
                sys.executable,
                "-c",
                "import sys; sys.exit(37)",
            ],
            cwd=kwargs["cwd"],
            check=True,
        )

    # Replace only the process-launching function.
    # The real Airflow Operator and submit_stage()
    # still execute.
    with patch.object(
        runner.subprocess,
        "run",
        side_effect=injected_failure,
    ):
        try:
            gold_task.execute(context={})

        except subprocess.CalledProcessError as exc:
            assert exc.returncode == 37

            print(
                "PASS: Child-process failure propagated"
            )

        else:
            raise AssertionError(
                "Child-process failure did not propagate"
            )

    assert_no_heavy_pipeline_import()


def main():
    print("=" * 55)
    print("NYC V2 — AIRFLOW RUNTIME SMOKE TEST")
    print("=" * 55)

    dag = test_dag_loading()

    test_execution_contract(dag)

    test_failure_propagation(dag)

    print("\n" + "=" * 55)
    print("AIRFLOW RUNTIME SMOKE PASS")
    print("=" * 55)


if __name__ == "__main__":
    main()
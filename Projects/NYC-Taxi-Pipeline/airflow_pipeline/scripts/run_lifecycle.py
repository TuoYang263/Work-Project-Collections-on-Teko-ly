
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from scripts.run_evidence import (
    DEFAULT_RUN_EVIDENCE_DIR,
    finalize_run_evidence,
    start_run_evidence,
)


def _require_dag_run(
    context: Mapping[str, Any],
):
    dag_run = context.get("dag_run")

    if dag_run is None or not getattr(
        dag_run, "run_id", None
    ):
        raise ValueError(
            "Airflow context requires dag_run.run_id"
        )

    return dag_run


def start_pipeline_run(
    context: Mapping[str, Any],
    execution_contract: Mapping[str, Any],
    *,
    directory: Path = DEFAULT_RUN_EVIDENCE_DIR,
):
    dag_run = _require_dag_run(context)

    return start_run_evidence(
        dag_run.run_id,
        execution_contract,
        directory=directory,
    )


def mark_pipeline_success(
    context: Mapping[str, Any],
    *,
    directory: Path = DEFAULT_RUN_EVIDENCE_DIR,
):
    dag_run = _require_dag_run(context)

    return finalize_run_evidence(
        dag_run.run_id,
        "SUCCEEDED",
        directory=directory,
    )


def mark_pipeline_failure(
    context: Mapping[str, Any],
    *,
    directory: Path = DEFAULT_RUN_EVIDENCE_DIR,
):
    dag_run = _require_dag_run(context)

    failed_tasks = sorted(
        task.task_id
        for task in dag_run.get_task_instances()
        if task.state == "failed"
    )

    if not failed_tasks:
        raise RuntimeError(
            "Cannot determine failed Airflow task"
        )

    return finalize_run_evidence(
        dag_run.run_id,
        "FAILED",
        failure_stage=failed_tasks[0],
        directory=directory,
    )

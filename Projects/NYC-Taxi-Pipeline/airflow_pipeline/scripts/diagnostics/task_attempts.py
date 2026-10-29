
"""Read-only Airflow task diagnostics (2.8.1)."""

from typing import Any

from airflow.models import TaskInstance
from airflow.models.taskfail import TaskFail
from airflow.utils.session import create_session


def get_task_attempts(
    dag_id: str,
    run_id: str,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Read current task snapshots, not historical attempts."""
    if not dag_id or not run_id:
        raise ValueError("dag_id and run_id are required")

    with create_session() as session:
        query = session.query(TaskInstance).filter(
            TaskInstance.dag_id == dag_id,
            TaskInstance.run_id == run_id,
        )

        if task_id is not None:
            query = query.filter(TaskInstance.task_id == task_id)

        rows = query.order_by(
            TaskInstance.task_id,
            TaskInstance.map_index,
        ).all()

        tasks = [
            {
                "task_id": ti.task_id,
                "map_index": ti.map_index,
                "state": ti.state,
                "prev_attempted_tries": ti.prev_attempted_tries,
                "start_date": (
                    ti.start_date.isoformat() if ti.start_date else None
                ),
                "end_date": (
                    ti.end_date.isoformat() if ti.end_date else None
                ),
            }
            for ti in rows
        ]

    return {
        "dag_id": dag_id,
        "run_id": run_id,
        "source": "TASK_INSTANCE_CURRENT",
        "attempt_history_complete": False,
        "tasks": tasks,
    }



def get_task_failures(
    dag_id: str,
    run_id: str,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Read recorded task failures without inferring attempt numbers."""
    if not dag_id or not run_id:
        raise ValueError("dag_id and run_id are required")

    with create_session() as session:
        query = session.query(TaskFail).filter(
            TaskFail.dag_id == dag_id,
            TaskFail.run_id == run_id,
        )

        if task_id is not None:
            query = query.filter(TaskFail.task_id == task_id)

        rows = query.order_by(
            TaskFail.start_date, TaskFail.id
        ).all()

        failures = [
            {
                "task_id": row.task_id,
                "map_index": row.map_index,
                "start_date": (
                    row.start_date.isoformat()
                    if row.start_date else None
                ),
                "end_date": (
                    row.end_date.isoformat()
                    if row.end_date else None
                ),
                "duration": row.duration,
            }
            for row in rows
        ]

    return {
        "dag_id": dag_id,
        "run_id": run_id,
        "source": "TASK_FAIL",
        "attempt_history_complete": False,
        "failures": failures,
    }


"""Read-only capture of observed lifecycle probe database evidence."""

import hashlib
import json
import subprocess
from collections import Counter
from typing import Any


DAG_ID = "nyc_lifecycle_probe"

RUN_IDS = (
    "probe_20261020T182304Z",
    "probe_retry_20261020T182846Z",
    "probe_failure_20261020T183606Z",
)

EXPECTED_FAILURES = dict(zip(RUN_IDS, (0, 1, 2)))

# Fixed SQL: no caller-supplied SQL or run identifiers.
SQL = """
WITH wanted(run_id) AS (
    VALUES
        ('probe_20261020T182304Z'),
        ('probe_retry_20261020T182846Z'),
        ('probe_failure_20261020T183606Z')
)
SELECT json_build_object(
    'source', 'ISOLATED_LIFECYCLE_DB_COPY',
    'dag_id', 'nyc_lifecycle_probe',
    'task_instances', (
        SELECT COALESCE(json_agg(row_to_json(t)), '[]'::json)
        FROM (
            SELECT dag_id, run_id, task_id, map_index,
            state, start_date, end_date
            FROM task_instance
            WHERE dag_id = 'nyc_lifecycle_probe'
              AND run_id IN (SELECT run_id FROM wanted)
            ORDER BY run_id, task_id, map_index
        ) t
    ),
    'task_failures', (
        SELECT COALESCE(json_agg(row_to_json(f)), '[]'::json)
        FROM (
            SELECT dag_id, run_id, task_id, map_index,
                   start_date, end_date, duration
            FROM task_fail
            WHERE dag_id = 'nyc_lifecycle_probe'
              AND run_id IN (SELECT run_id FROM wanted)
            ORDER BY run_id, start_date, task_id
        ) f
    )
)::text;
"""


def validate_db_evidence(payload: Any) -> dict[str, Any]:
    """Reject missing, unexpected, or cross-run database evidence."""

    if (
        not isinstance(payload, dict)
        or payload.get("source") != "ISOLATED_LIFECYCLE_DB_COPY"
        or payload.get("dag_id") != DAG_ID
    ):
        raise ValueError("Invalid database evidence source")

    tasks = payload.get("task_instances")
    failures = payload.get("task_failures")

    if not isinstance(tasks, list) or not isinstance(failures, list):
        raise ValueError("Invalid database evidence collections")

    for row in tasks + failures:
        if (
            not isinstance(row, dict)
            or row.get("dag_id") != DAG_ID
            or row.get("run_id") not in RUN_IDS
            or not isinstance(row.get("task_id"), str)
        ):
            raise ValueError("Database evidence identity mismatch")

    task_counts = Counter(row["run_id"] for row in tasks)
    failure_counts = Counter(row["run_id"] for row in failures)

    for run_id in RUN_IDS:
        task_ids = {
            row["task_id"]
            for row in tasks
            if row["run_id"] == run_id
        }

        if (
            task_counts[run_id] != 2
            or task_ids != {"start_run_evidence", "harmless_task"}
            or failure_counts[run_id] != EXPECTED_FAILURES[run_id]
        ):
            raise ValueError("Incomplete historical database evidence")

    if any(row["task_id"] != "harmless_task" for row in failures):
        raise ValueError("Unexpected failure task")

    return payload


def read_probe_db_evidence() -> dict[str, Any]:
    """Query the isolated PostgreSQL copy without starting Airflow."""

    command = [
        "docker", "exec",
        "-e",
        "PGOPTIONS=-c default_transaction_read_only=on "
        "-c statement_timeout=15000",
        "nyc-v2-probe-db",
        "psql",
        "-U", "lifecycle",
        "-d", "lifecycle",
        "-X", "-A", "-t",
        "-v", "ON_ERROR_STOP=1",
        "-c", SQL,
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("Read-only database capture failed") from exc

    if len(result.stdout.encode("utf-8")) > 256 * 1024:
        raise ValueError("Database evidence exceeds capture limit")

    try:
        payload = json.loads(result.stdout.strip())
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid database evidence JSON") from exc

    return validate_db_evidence(payload)


def prepare_db_artifact(payload: Any) -> dict[str, Any]:
    """Produce deterministic bytes; do not publish a snapshot."""

    validated = validate_db_evidence(payload)

    document = {
        "artifact_type": "AIRFLOW_TASK_DIAGNOSTICS",
        "capture_scope": "OBSERVED_TEST_RUN_DB_PROJECTION",
        "attempt_history_complete": False,
        "evidence": validated,
    }

    content = (
        json.dumps(document, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")

    return {
        "filename": "airflow_task_diagnostics.json",
        "content": content,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def prepare_run_db_artifact(
    payload: Any,
    run_id: str,
) -> dict[str, Any]:
    """Prepare evidence for exactly one approved incident run."""

    if run_id not in RUN_IDS:
        raise ValueError("Unsupported incident run")

    validated = validate_db_evidence(payload)

    evidence = {
        "source": validated["source"],
        "dag_id": DAG_ID,
        "run_id": run_id,
        "task_instances": [
            row for row in validated["task_instances"]
            if row["run_id"] == run_id
        ],
        "task_failures": [
            row for row in validated["task_failures"]
            if row["run_id"] == run_id
        ],
    }

    document = {
        "artifact_type": "AIRFLOW_TASK_DIAGNOSTICS",
        "source_run_id": run_id,
        "capture_scope": "OBSERVED_TEST_RUN_DB_PROJECTION",
        "attempt_history_complete": False,
        "evidence": evidence,
    }

    content = (
        json.dumps(document, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")

    return {
        "filename": "airflow_task_diagnostics.json",
        "content": content,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }

"""Build provenance-labeled incident pointers from observed run evidence."""

from typing import Any

from scripts.diagnostics.evidence_reader import get_run_evidence


def build_pointer_manifest(dag_id: str, run_id: str) -> dict[str, Any]:
    """Create an unarchived incident pointer, not a verified snapshot."""
    if dag_id != "nyc_lifecycle_probe":
        raise ValueError("Only lifecycle test DAG is supported")

    record = get_run_evidence(dag_id, run_id)

    if (
        record.get("schema_version") != 2
        or record.get("dag_id") != dag_id
        or record.get("run_id") != run_id
    ):
        raise ValueError("Invalid Run Evidence V2 identity")

    return {
        "schema_version": 1,
        "case_id": run_id,
        "revision": 1,
        "provenance": "OBSERVED_TEST_RUN",
        "source_dag_id": dag_id,
        "source_run_id": run_id,
        "source_status": record["status"],
        "source_started_at": record.get("started_at"),
        "source_finished_at": record.get("finished_at"),
        "source_failure_stage": record.get("failure_stage"),
        "capture_status": "POINTER_ONLY",
        "captured_at_utc": None,
        "source_commit": None,
        "snapshot_sha256": None,
        "attachments": [],
        "limitations": [
            "Original evidence has not been archived or independently verified."
        ],
    }
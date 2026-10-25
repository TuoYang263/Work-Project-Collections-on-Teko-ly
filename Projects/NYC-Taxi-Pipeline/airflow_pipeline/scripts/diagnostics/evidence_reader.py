"""Read-only adapter for NYC V2 Run Evidence."""

from typing import Any

from scripts.run_evidence import load_run_evidence


def get_run_evidence(dag_id: str, run_id: str) -> dict[str, Any]:
    """Load existing Run Evidence V2 for a specific DAG run."""
    return load_run_evidence(
        run_id,
        dag_id=dag_id,
    )

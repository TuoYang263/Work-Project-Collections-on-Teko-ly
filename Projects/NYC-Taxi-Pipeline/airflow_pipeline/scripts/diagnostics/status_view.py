from scripts.diagnostics.evidence_reader import (
    EvidenceConflictError,
    EvidenceInvalidError,
    EvidenceNotFoundError,
    get_run_evidence,
)



"""Read-only diagnostic status projection."""

from typing import Any

from scripts.diagnostics.evidence_reader import (
    EvidenceConflictError,
    EvidenceInvalidError,
    EvidenceNotFoundError,
    get_run_evidence,
)


def get_run_status_view(dag_id: str, run_id: str) -> dict[str, Any]:
    """Return a diagnostic status without inventing run outcomes."""
    try:
        record = get_run_evidence(dag_id=dag_id, run_id=run_id)
    except (
        EvidenceNotFoundError,
        EvidenceInvalidError,
        EvidenceConflictError,
    ) as exc:
        return {
            "dag_id": dag_id,
            "run_id": run_id,
            "run_status": "UNKNOWN",
            "reason_code": exc.code,
        }

    return {
        "dag_id": dag_id,
        "run_id": run_id,
        "run_status": record["status"],
        "reason_code": None,
    }

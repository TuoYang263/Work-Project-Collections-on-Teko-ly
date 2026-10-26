"""Read-only adapter for NYC V2 Run Evidence."""

from typing import Any
from json import JSONDecodeError

from scripts.run_evidence import load_run_evidence

class EvidenceNotFoundError(FileNotFoundError):
    """Requested Run Evidence does not exist"""

    code = "NOT_FOUND"


class EvidenceInvalidError(ValueError):
    """Run Evidence contains invalid data."""

    code = "INVALID_EVIDENCE"


class EvidenceConflictError(RuntimeError):
    """Run Evidence identity or version does not match."""

    code = "EVIDENCE_CONFLICT"

def get_run_evidence(dag_id: str, run_id: str) -> dict[str, Any]:
    """Load existing Run Evidence V2 for a specific DAG run."""
    try:
        record = load_run_evidence(
            run_id,
            dag_id=dag_id,
        )

        valid_statuses = {"RUNNING", "SUCCEEDED", "FAILED"}

        if record.get("status") not in valid_statuses:
            raise EvidenceInvalidError(
                "Run evidence has missing or invalid status"
            )

        return record
    except FileNotFoundError:
        raise EvidenceNotFoundError(
            "Run evidence not found"
        ) from None
    except JSONDecodeError as exc:
        raise EvidenceInvalidError(
            "Run evidence contains invalid JSON"
        ) from exc
    except UnicodeDecodeError as exc:
        raise EvidenceInvalidError(
            "Run evidence contains UTF-8 encoding"
        ) from exc
    except RuntimeError as exc:
        if str(exc) != "Run evidence identity mismatch":
            raise

        raise EvidenceConflictError(
            "Run evidence identity mismatch"
        ) from exc
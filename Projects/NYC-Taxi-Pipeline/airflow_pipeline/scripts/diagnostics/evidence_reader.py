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


def get_run_evidence(dag_id: str, run_id: str) -> dict[str, Any]:
    """Load existing Run Evidence V2 for a specific DAG run."""
    try:
        return load_run_evidence(
            run_id,
            dag_id=dag_id,
        )
    except FileNotFoundError:
        raise EvidenceNotFoundError(
            "Run evidence not found"
        ) from None
    except JSONDecodeError as exc:
        raise EvidenceInvalidError(
            "Run evidence contains invalid JSON"
        ) from exc

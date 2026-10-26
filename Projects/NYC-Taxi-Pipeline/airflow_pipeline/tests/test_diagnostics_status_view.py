
"""Tests for the read-only diagnostic status view."""

from unittest.mock import patch

import pytest

from scripts.diagnostics.status_view import get_run_status_view

from scripts.diagnostics.evidence_reader import (
    EvidenceNotFoundError,
    EvidenceInvalidError,
    EvidenceConflictError,
)


@pytest.mark.parametrize(
    "status",
    ["RUNNING", "SUCCEEDED", "FAILED"],
)
def test_valid_run_status_is_preserved(status):
    dag_id = "nyc_lifecycle_probe"
    run_id = "test_status_view_001"

    record = {
        "schema_version": 2,
        "dag_id": dag_id,
        "run_id": run_id,
        "status": status,
    }

    with patch(
        "scripts.diagnostics.status_view.get_run_evidence",
        return_value=record,
    ) as mock_reader:
        result = get_run_status_view(
            dag_id=dag_id,
            run_id=run_id,
        )

    mock_reader.assert_called_once_with(
        dag_id=dag_id,
        run_id=run_id,
    )

    assert result == {
        "dag_id": dag_id,
        "run_id": run_id,
        "run_status": status,
        "reason_code": None,
    }


@pytest.mark.parametrize(
    "error_type, reason_code",
    [
        (EvidenceNotFoundError, "NOT_FOUND"),
        (EvidenceInvalidError, "INVALID_EVIDENCE"),
        (EvidenceConflictError, "EVIDENCE_CONFLICT"),
    ],
)
def test_evidence_error_maps_to_unknown(error_type, reason_code):
    dag_id = "nyc_lifecycle_probe"
    run_id = "test_unknown_001"

    with patch(
        "scripts.diagnostics.status_view.get_run_evidence",
        side_effect=error_type("Test evidence error"),
    ) as mock_reader:
        result = get_run_status_view(
            dag_id=dag_id,
            run_id=run_id,
        )

    mock_reader.assert_called_once_with(
        dag_id=dag_id,
        run_id=run_id,
    )

    assert result == {
        "dag_id": dag_id,
        "run_id": run_id,
        "run_status": "UNKNOWN",
        "reason_code": reason_code,
    }

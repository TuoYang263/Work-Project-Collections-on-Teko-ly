"""Tests for the read-only Diagnostics Evidence Reader."""

from unittest.mock import patch

from scripts.diagnostics.evidence_reader import get_run_evidence


def test_get_run_evidence_delegates_to_v2_loader():
    expected = {
        "schema_version": 2,
        "dag_id": "nyc_lifecycle_probe",
        "run_id": "test_failure_001",
        "status": "FAILED",
        "failure_stage": "harmless_task",
    }

    with patch(
        "scripts.diagnostics.evidence_reader.load_run_evidence",
        return_value=expected,
    ) as mock_loader:
        result = get_run_evidence(
            dag_id="nyc_lifecycle_probe",
            run_id="test_failure_001",
        )

    mock_loader.assert_called_once_with(
        "test_failure_001",
        dag_id="nyc_lifecycle_probe",
    )

    assert result == expected

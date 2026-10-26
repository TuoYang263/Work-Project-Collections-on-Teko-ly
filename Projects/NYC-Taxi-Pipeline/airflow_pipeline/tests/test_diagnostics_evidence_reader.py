"""Tests for the read-only Diagnostics Evidence Reader."""

import pytest
from unittest.mock import patch

from scripts.diagnostics.evidence_reader import (
    EvidenceNotFoundError,
    get_run_evidence,
)


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


def test_wrong_dag_id_cannot_read_another_dag(tmp_path):
    """A run must not be accessible through another DAG identity."""
    from functools import partial

    import pytest

    from scripts.run_evidence import (
        load_run_evidence,
        start_run_evidence,
    )

    contract = {
        "mode": "BACKFILL",
        "logical_window": {
            "start": "2024-01-01",
            "end_exclusive": "2024-02-01",
        },
        "read_scope": {"raw_file_months": ["2024-01"]},
        "write_scope": "WINDOW_REPLACE",
        "rerun_semantics": "IDEMPOTENT",
    }

    start_run_evidence(
        "test_run_001",
        contract,
        dag_id="nyc_lifecycle_probe",
        directory=tmp_path,
    )

    with patch(
        "scripts.diagnostics.evidence_reader.load_run_evidence",
        new=partial(load_run_evidence, directory=tmp_path),
    ):
        correct = get_run_evidence(
            "nyc_lifecycle_probe", "test_run_001"
        )
        assert correct["status"] == "RUNNING"

        with pytest.raises(FileNotFoundError):
            get_run_evidence(
                "another_dag", "test_run_001"
            )

        with pytest.raises(FileNotFoundError):
            get_run_evidence(
                "nyc_lifecycle_probe", "nonexistent_run_002"
            )


def test_missing_evidence_is_classified_as_not_found():
    with patch(
        "scripts.diagnostics.evidence_reader.load_run_evidence",
        side_effect=FileNotFoundError,
    ):
        with pytest.raises(EvidenceNotFoundError) as exc_info:
            get_run_evidence(
                dag_id="nyc_lifecycle_probe",
                run_id="nonexistent_run_003",
            )

    assert exc_info.value.code == "NOT_FOUND"
    assert isinstance(exc_info.value, FileNotFoundError)
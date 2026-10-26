"""Tests for the read-only Diagnostics Evidence Reader."""

import pytest

from functools import partial
from unittest.mock import patch
from json import JSONDecodeError

from scripts.run_evidence import load_run_evidence, start_run_evidence
from scripts.diagnostics.evidence_reader import (
    EvidenceInvalidError,
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


def test_invalid_json_is_classified_as_invalid_evidence():
    invalid_json_error = JSONDecodeError(
        "Expecting property name",
        "{broken",
        1,
    )

    with patch(
        "scripts.diagnostics.evidence_reader.load_run_evidence",
        side_effect=invalid_json_error,
    ):
        with pytest.raises(EvidenceInvalidError) as exc_info:
            get_run_evidence(
                dag_id="nyc_lifecycle_probe",
                run_id="test_invalid_json_001",
            )

    assert exc_info.value.code == "INVALID_EVIDENCE"
    assert isinstance(exc_info.value, ValueError)
    assert exc_info.value.__cause__ is invalid_json_error



def test_corrupted_json_file_is_classified_as_invalid_evidence(tmp_path):
    dag_id = "nyc_lifecycle_probe"
    run_id = "test_corrupted_json_001"

    contract = {
        "mode": "incremental",
        "logical_window": {},
        "read_scope": {},
        "write_scope": "window_replace",
        "rerun_semantics": "idempotent",
    }

    start_run_evidence(
        run_id,
        contract,
        dag_id=dag_id,
        directory=tmp_path,
    )

    evidence_files = list((tmp_path / "v2").glob("*.json"))
    assert len(evidence_files) == 1

    evidence_files[0].write_text("{broken", encoding="utf-8")

    with patch(
        "scripts.diagnostics.evidence_reader.load_run_evidence",
        new=partial(load_run_evidence, directory=tmp_path),
    ):
        with pytest.raises(EvidenceInvalidError) as exc_info:
            get_run_evidence(dag_id=dag_id, run_id=run_id)

    assert exc_info.value.code == "INVALID_EVIDENCE"
    assert isinstance(exc_info.value.__cause__, JSONDecodeError)


def test_invalid_utf8_file_is_classified_as_invalid_evidence(tmp_path):
    dag_id = "nyc_lifecycle_probe"
    run_id = "test_invalid_utf8_001"

    contract = {
        "mode": "incremental",
        "logical_window": {},
        "read_scope": {},
        "write_scope": "window_replace",
        "rerun_semantics": "idempotent",
    }

    start_run_evidence(
        run_id,
        contract,
        dag_id=dag_id,
        directory=tmp_path,
    )

    evidence_files = list((tmp_path / "v2").glob("*.json"))
    assert len(evidence_files) == 1

    evidence_files[0].write_bytes(b"\xff\xfe\xfa")

    with patch(
        "scripts.diagnostics.evidence_reader.load_run_evidence",
        new=partial(load_run_evidence, directory=tmp_path),
    ):
        with pytest.raises(EvidenceInvalidError) as exc_info:
            get_run_evidence(dag_id=dag_id, run_id=run_id)

    assert exc_info.value.code == "INVALID_EVIDENCE"
    assert isinstance(exc_info.value.__cause__, UnicodeDecodeError)
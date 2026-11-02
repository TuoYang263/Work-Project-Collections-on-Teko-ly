
"""Tests for provenance-labeled incident pointers."""

import pytest

from scripts.diagnostics import incident_manifest


@pytest.mark.parametrize(
    "status, failure_stage",
    [
        ("SUCCEEDED", None),
        ("FAILED", "harmless_task"),
    ],
)
def test_pointer_preserves_source_truth(
    monkeypatch, status, failure_stage
):
    record = {
        "schema_version": 2,
        "dag_id": "nyc_lifecycle_probe",
        "run_id": "probe_test",
        "status": status,
        "started_at": "2026-10-20T18:00:00+00:00",
        "finished_at": "2026-10-20T18:01:00+00:00",
        "failure_stage": failure_stage,
    }

    monkeypatch.setattr(
        incident_manifest,
        "get_run_evidence",
        lambda dag_id, run_id: record,
    )

    manifest = incident_manifest.build_pointer_manifest(
        "nyc_lifecycle_probe", "probe_test"
    )

    assert manifest["source_status"] == status
    assert manifest["source_failure_stage"] == failure_stage
    assert manifest["provenance"] == "OBSERVED_TEST_RUN"
    assert manifest["capture_status"] == "POINTER_ONLY"
    assert manifest["snapshot_sha256"] is None
    assert manifest["attachments"] == []


@pytest.mark.parametrize(
    "field, invalid_value",
    [
        ("schema_version", 1),
        ("dag_id", "another_dag"),
        ("run_id", "another_run"),
    ],
)
def test_pointer_rejects_identity_mismatch(
    monkeypatch, field, invalid_value
):
    record = {
        "schema_version": 2,
        "dag_id": "nyc_lifecycle_probe",
        "run_id": "probe_test",
        "status": "SUCCEEDED",
    }
    record[field] = invalid_value

    monkeypatch.setattr(
        incident_manifest,
        "get_run_evidence",
        lambda dag_id, run_id: record,
    )

    with pytest.raises(
        ValueError, match="Invalid Run Evidence V2 identity"
    ):
        incident_manifest.build_pointer_manifest(
            "nyc_lifecycle_probe", "probe_test"
        )


def test_pointer_rejects_non_test_dag(monkeypatch):
    def unexpected_read(*args):
        raise AssertionError("Evidence reader must not be called")

    monkeypatch.setattr(
        incident_manifest,
        "get_run_evidence",
        unexpected_read,
    )

    with pytest.raises(
        ValueError, match="Only lifecycle test DAG is supported"
    ):
        incident_manifest.build_pointer_manifest(
            "nyc_taxi_medallion_delta",
            "production_run",
        )

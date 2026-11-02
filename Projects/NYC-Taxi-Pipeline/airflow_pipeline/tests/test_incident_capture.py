
"""Tests for incident evidence artifact preparation."""

import hashlib
import json

import pytest

from scripts.diagnostics.incident_capture import prepare_run_metadata


BASE_RECORD = {
    "schema_version": 2,
    "dag_id": "nyc_lifecycle_probe",
    "run_id": "probe_20261020T182304Z",
    "status": "SUCCEEDED",
    "started_at": "2026-10-20T18:23:11+00:00",
    "finished_at": "2026-10-20T18:23:13+00:00",
    "failure_stage": None,
}


def test_deterministic_content_and_sha256():
    first = prepare_run_metadata(BASE_RECORD)
    second = prepare_run_metadata(dict(reversed(list(BASE_RECORD.items()))))

    assert first["content"] == second["content"]
    assert first["sha256"] == second["sha256"]
    assert first["size_bytes"] == len(first["content"])
    assert first["sha256"] == hashlib.sha256(
        first["content"]
    ).hexdigest()


def test_changed_evidence_changes_sha256():
    original = prepare_run_metadata(BASE_RECORD)

    changed = {
        **BASE_RECORD,
        "status": "FAILED",
        "failure_stage": "harmless_task",
    }
    modified = prepare_run_metadata(changed)

    assert original["sha256"] != modified["sha256"]


def test_allowlist_excludes_unapproved_fields():
    record = {
        **BASE_RECORD,
        "secret_token": "DO_NOT_ARCHIVE",
        "aws_account": "PRIVATE_ACCOUNT",
    }

    artifact = prepare_run_metadata(record)
    payload = json.loads(artifact["content"])

    assert payload["capture_scope"] == "METADATA_ONLY"
    assert payload["redaction_policy"] == "ALLOWLIST_V1"
    assert "secret_token" not in payload["source"]
    assert "aws_account" not in payload["source"]
    assert b"DO_NOT_ARCHIVE" not in artifact["content"]
    assert b"PRIVATE_ACCOUNT" not in artifact["content"]


@pytest.mark.parametrize(
    "field, invalid_value",
    [
        ("schema_version", 1),
        ("dag_id", "another_dag"),
        ("run_id", ""),
        ("status", "RUNNING"),
    ],
)
def test_invalid_run_evidence_rejected(field, invalid_value):
    record = {**BASE_RECORD, field: invalid_value}

    with pytest.raises(
        ValueError, match="Invalid historical run evidence"
    ):
        prepare_run_metadata(record)
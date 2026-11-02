
"""Tests for private incident evidence staging."""

import hashlib

import pytest

from scripts.diagnostics import incident_capture
from scripts.diagnostics.incident_capture import (
    prepare_run_metadata,
    stage_metadata_artifact,
)


@pytest.fixture
def artifact():
    record = {
        "schema_version": 2,
        "dag_id": "nyc_lifecycle_probe",
        "run_id": "probe_test",
        "status": "FAILED",
        "started_at": "2026-10-20T18:00:00+00:00",
        "finished_at": "2026-10-20T18:01:00+00:00",
        "failure_stage": "harmless_task",
    }
    return prepare_run_metadata(record)


@pytest.fixture
def private_root(tmp_path):
    root = tmp_path / "private_staging"
    root.mkdir(mode=0o700)
    return root


def test_staging_writes_verified_bytes(artifact, private_root):
    stage_dir = stage_metadata_artifact(
        artifact, private_root
    )

    path = stage_dir / artifact["filename"]

    assert path.is_file()
    assert path.read_bytes() == artifact["content"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        artifact["sha256"]
    )


def test_staging_does_not_overwrite(artifact, private_root):
    first = stage_metadata_artifact(artifact, private_root)
    second = stage_metadata_artifact(artifact, private_root)

    assert first != second
    assert (first / artifact["filename"]).exists()
    assert (second / artifact["filename"]).exists()


def test_staging_rejects_invalid_checksum(
    artifact, private_root
):
    corrupted = {**artifact, "sha256": "0" * 64}

    with pytest.raises(
        ValueError, match="Invalid prepared metadata artifact"
    ):
        stage_metadata_artifact(corrupted, private_root)

    assert list(private_root.iterdir()) == []


def test_staging_rejects_insecure_directory(
    artifact, private_root
):
    private_root.chmod(0o755)

    with pytest.raises(
        PermissionError, match="must be private"
    ):
        stage_metadata_artifact(artifact, private_root)


def test_staging_cleans_up_on_failure(
    artifact, private_root, monkeypatch
):
    def fail_fsync(_fd):
        raise OSError("Simulated storage failure")

    monkeypatch.setattr(
        incident_capture.os, "fsync", fail_fsync
    )

    with pytest.raises(
        OSError, match="Simulated storage failure"
    ):
        stage_metadata_artifact(artifact, private_root)

    assert list(private_root.iterdir()) == []

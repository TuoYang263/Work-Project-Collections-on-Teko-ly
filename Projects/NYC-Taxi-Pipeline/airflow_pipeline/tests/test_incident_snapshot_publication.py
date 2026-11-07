"""Unit 1C: local publication of bounded, observed test-run evidence.

These tests do not start Airflow, AWS, PostgreSQL, or a data pipeline.
"""
import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from scripts.diagnostics.incident_capture import prepare_run_metadata
from scripts.diagnostics import incident_publication as publication

DAG_ID = "nyc_lifecycle_probe"
RUN_ID = "probe_failure_20261020T183606Z"
PREPARED_AT = "2026-10-20T20:00:00Z"


def _artifact(filename, document):
    content = (json.dumps(document, sort_keys=True) + "\n").encode("utf-8")
    return {
        "filename": filename,
        "content": content,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _case_artifacts():
    metadata = prepare_run_metadata({
        "schema_version": 2,
        "dag_id": DAG_ID,
        "run_id": RUN_ID,
        "status": "FAILED",
        "started_at": "2026-10-20T18:36:06+00:00",
        "finished_at": "2026-10-20T18:39:00+00:00",
        "failure_stage": "harmless_task",
    })
    database = _artifact("airflow_task_diagnostics.json", {
        "artifact_type": "AIRFLOW_TASK_DIAGNOSTICS",
        "source_run_id": RUN_ID,
        "evidence": {
            "dag_id": DAG_ID,
            "run_id": RUN_ID,
            "task_instances": [{
                "dag_id": DAG_ID,
                "run_id": RUN_ID,
                "task_id": "harmless_task",
            }],
            "task_failures": [],
        },
    })
    logs = _artifact("log_excerpts.json", {
        "artifact_type": "AIRFLOW_LOG_EXCERPTS",
        "dag_id": DAG_ID,
        "run_id": RUN_ID,
        "task_id": "harmless_task",
        "full_log_archived": False,
        "logs": [{
            "attempt_number": 1,
            "source_path": (
                f"dag_id={DAG_ID}/run_id={RUN_ID}/"
                "task_id=harmless_task/attempt=1.log"
            ),
        }],
    })
    return [metadata, database, logs]


@pytest.fixture
def private_root(tmp_path):
    root = tmp_path / "private_archive"
    root.mkdir(mode=0o700)
    return root


def _publish(root, artifacts=None, revision=1, run_id=RUN_ID):
    if artifacts is None:
        artifacts = _case_artifacts()
    return publication.publish_snapshot(
        artifacts,
        dag_id=DAG_ID,
        run_id=run_id,
        revision=revision,
        prepared_at_utc=PREPARED_AT,
        archive_root=root,
    )


def test_publish_writes_all_evidence_and_manifest(private_root):
    artifacts = _case_artifacts()
    result = _publish(private_root, artifacts)

    path = Path(result["archive_path"])
    assert path == private_root / RUN_ID / "revision-0001.zip"
    assert path.is_file()
    assert result["capture_status"] == "PUBLISHED_PENDING_BACKUP"
    assert result["snapshot_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

    with zipfile.ZipFile(path) as bundle:
        assert set(bundle.namelist()) == {
            "manifest.json",
            "run_evidence_metadata.json",
            "airflow_task_diagnostics.json",
            "log_excerpts.json",
        }
        for artifact in artifacts:
            assert bundle.read(artifact["filename"]) == artifact["content"]

        manifest_bytes = bundle.read("manifest.json")
        manifest = json.loads(manifest_bytes)
        assert manifest["source_run_id"] == RUN_ID
        assert manifest["capture_status"] == "PUBLISHED_PENDING_BACKUP"
        assert manifest["snapshot_sha256"] is None
        assert result["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()

    assert not list((private_root / RUN_ID).glob(".staging-*"))


def test_second_publish_cannot_replace_revision(private_root):
    original = _publish(private_root)
    original_bytes = Path(original["archive_path"]).read_bytes()

    with pytest.raises(FileExistsError):
        _publish(private_root)

    assert Path(original["archive_path"]).read_bytes() == original_bytes
    assert not list((private_root / RUN_ID).glob(".staging-*"))


def test_invalid_evidence_never_published(private_root):
    artifacts = _case_artifacts()
    artifacts[0] = {**artifacts[0], "content": b"changed"}

    with pytest.raises(ValueError, match="checksum|size mismatch"):
        _publish(private_root, artifacts)

    assert not (private_root / RUN_ID / "revision-0001.zip").exists()


def test_insecure_archive_root_rejected(tmp_path):
    root = tmp_path / "public"
    root.mkdir(mode=0o755)

    with pytest.raises(PermissionError, match="private"):
        _publish(root)
    assert list(root.iterdir()) == []


def test_symlink_archive_root_rejected(tmp_path):
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)

    with pytest.raises(PermissionError, match="private"):
        _publish(alias)


def test_invalid_run_name_rejected_before_write(private_root):
    with pytest.raises(ValueError, match="Invalid snapshot case identity"):
        _publish(private_root, run_id="../escape")
    assert not list(private_root.iterdir())


def test_atomic_link_failure_leaves_no_revision(private_root, monkeypatch):
    def simulated_failure(*args, **kwargs):
        raise OSError("simulated publication failure")

    monkeypatch.setattr(publication.os, "link", simulated_failure)
    with pytest.raises(OSError, match="simulated publication failure"):
        _publish(private_root)

    case_dir = private_root / RUN_ID
    assert case_dir.is_dir()
    assert list(case_dir.iterdir()) == []


def test_archive_is_deterministic_for_same_inputs(tmp_path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir(mode=0o700)
    root_b.mkdir(mode=0o700)

    artifacts = _case_artifacts()
    first = _publish(root_a, artifacts)
    second = _publish(root_b, list(reversed(artifacts)))

    assert first["snapshot_sha256"] == second["snapshot_sha256"]
    assert Path(first["archive_path"]).read_bytes() == Path(second["archive_path"]).read_bytes()

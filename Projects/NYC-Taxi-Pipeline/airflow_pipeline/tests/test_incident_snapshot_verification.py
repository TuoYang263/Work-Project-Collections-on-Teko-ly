"""Unit 1C.9: offline verification of locally published snapshots.

Does not start Airflow, PostgreSQL, Docker, or AWS.
Place in airflow_pipeline/tests/ and run after adding verify_published_snapshot().
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


def _evidence():
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
def published(tmp_path):
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    result = publication.publish_snapshot(
        _evidence(),
        dag_id=DAG_ID,
        run_id=RUN_ID,
        revision=1,
        prepared_at_utc=PREPARED_AT,
        archive_root=root,
    )
    return result


def _verify(result, **overrides):
    args = {
        "archive_path": Path(result["archive_path"]),
        "expected_snapshot_sha256": result["snapshot_sha256"],
        "expected_manifest_sha256": result["manifest_sha256"],
        "dag_id": DAG_ID,
        "run_id": RUN_ID,
        "revision": 1,
    }
    args.update(overrides)
    return publication.verify_published_snapshot(**args)


def _repack(path, *, updates=None, removed=(), extra=None):
    updates = updates or {}
    extra = extra or {}
    with zipfile.ZipFile(path) as bundle:
        contents = {
            name: bundle.read(name)
            for name in bundle.namelist()
            if name not in removed
        }
    contents.update(updates)
    contents.update(extra)
    with zipfile.ZipFile(path, "w") as bundle:
        for name, data in sorted(contents.items()):
            bundle.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_verification_succeeds_with_trusted_digests(published):
    verification = _verify(published)
    assert verification["verification_status"] == "LOCAL_INTEGRITY_VERIFIED"
    assert verification["capture_status"] == "PUBLISHED_PENDING_BACKUP"
    assert verification["case_id"] == RUN_ID
    assert verification["revision"] == 1
    assert verification["snapshot_sha256"] == published["snapshot_sha256"]


def test_rejects_wrong_external_archive_digest(published):
    with pytest.raises(ValueError, match="Snapshot checksum mismatch"):
        _verify(published, expected_snapshot_sha256="0" * 64)


def test_rejects_wrong_external_manifest_digest(published):
    with pytest.raises(ValueError, match="Manifest checksum mismatch"):
        _verify(published, expected_manifest_sha256="0" * 64)


def test_rejects_changed_archive_bytes(published):
    path = Path(published["archive_path"])
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="Snapshot checksum mismatch"):
        _verify(published)


def test_rejects_missing_zip_entry_even_when_archive_digest_updated(published):
    path = Path(published["archive_path"])
    new_digest = _repack(path, removed={"log_excerpts.json"})
    with pytest.raises(ValueError, match="Snapshot ZIP entries mismatch"):
        _verify(published, expected_snapshot_sha256=new_digest)


def test_rejects_unexpected_zip_entry(published):
    path = Path(published["archive_path"])
    new_digest = _repack(path, extra={"unexpected.txt": b"hello"})
    with pytest.raises(ValueError, match="Snapshot ZIP entries mismatch"):
        _verify(published, expected_snapshot_sha256=new_digest)


def test_rejects_changed_artifact_with_updated_archive_digest(published):
    path = Path(published["archive_path"])
    new_digest = _repack(path, updates={"log_excerpts.json": b"{}"})
    with pytest.raises(ValueError, match="Manifest attachments mismatch"):
        _verify(published, expected_snapshot_sha256=new_digest)


def test_rejects_changed_manifest_identity_even_with_updated_digests(published):
    path = Path(published["archive_path"])
    with zipfile.ZipFile(path) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
    manifest["source_run_id"] = "another_run"
    new_manifest = publication.manifest_bytes_and_digest(manifest)[0]
    new_digest = _repack(path, updates={"manifest.json": new_manifest})
    with pytest.raises(ValueError, match="Manifest identity or status mismatch"):
        _verify(
            published,
            expected_snapshot_sha256=new_digest,
            expected_manifest_sha256=hashlib.sha256(new_manifest).hexdigest(),
        )


def test_rejects_wrong_case_revision(published):
    with pytest.raises(ValueError, match="Invalid or missing snapshot path"):
        _verify(published, revision=2)


def test_rejects_symlinked_archive(published, tmp_path):
    symlink = tmp_path / "revision-0001.zip"
    symlink.symlink_to(published["archive_path"])
    with pytest.raises(ValueError, match="Invalid or missing snapshot path"):
        _verify(published, archive_path=symlink)


def test_rejects_zip_bomb_before_decompression(published):
    path = Path(published["archive_path"])
    new_digest = _repack(path, extra={"huge.txt": b"0" * (3 * 1024 * 1024)})
    with pytest.raises(ValueError, match="Snapshot ZIP entries mismatch|size limit"):
        _verify(published, expected_snapshot_sha256=new_digest)


def test_rejects_invalid_expected_digest_format(published):
    with pytest.raises(ValueError, match="Invalid expected SHA256"):
        _verify(published, expected_snapshot_sha256="not-a-sha256")

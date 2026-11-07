"""Unit 1C.10: local backup and restore drills with trusted digests.

No Airflow, Postgres, AWS or original operational logs are touched.
"""
import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts.diagnostics.incident_capture import prepare_run_metadata
from scripts.diagnostics.incident_publication import (
    publish_snapshot,
    verify_published_snapshot,
)
from scripts.diagnostics import incident_backup as backup

DAG = "nyc_lifecycle_probe"
RUN = "probe_failure_20261020T183606Z"
PREPARED = "2026-10-20T20:00:00Z"


def _artifact(filename, payload):
    content = (json.dumps(payload, sort_keys=True) + "\n").encode()
    return {
        "filename": filename,
        "content": content,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _artifacts():
    metadata = prepare_run_metadata({
        "schema_version": 2,
        "dag_id": DAG,
        "run_id": RUN,
        "status": "FAILED",
        "started_at": "2026-10-20T18:36:06+00:00",
        "finished_at": "2026-10-20T18:39:00+00:00",
        "failure_stage": "harmless_task",
    })
    database = _artifact("airflow_task_diagnostics.json", {
        "artifact_type": "AIRFLOW_TASK_DIAGNOSTICS",
        "source_run_id": RUN,
        "evidence": {
            "dag_id": DAG,
            "run_id": RUN,
            "task_instances": [{
                "dag_id": DAG,
                "run_id": RUN,
                "task_id": "harmless_task",
            }],
            "task_failures": [],
        },
    })
    logs = _artifact("log_excerpts.json", {
        "artifact_type": "AIRFLOW_LOG_EXCERPTS",
        "dag_id": DAG,
        "run_id": RUN,
        "task_id": "harmless_task",
        "logs": [{
            "attempt_number": 1,
            "source_path": (
                f"dag_id={DAG}/run_id={RUN}/"
                "task_id=harmless_task/attempt=1.log"
            ),
        }],
    })
    return [metadata, database, logs]


def _root(path):
    path.mkdir(mode=0o700)
    return path


@pytest.fixture
def published(tmp_path):
    root = _root(tmp_path / "source")
    result = publish_snapshot(
        _artifacts(),
        dag_id=DAG,
        run_id=RUN,
        revision=1,
        prepared_at_utc=PREPARED,
        archive_root=root,
    )
    return result, tmp_path


def _args(result):
    return {
        "expected_snapshot_sha256": result["snapshot_sha256"],
        "expected_manifest_sha256": result["manifest_sha256"],
        "dag_id": DAG,
        "run_id": RUN,
        "revision": 1,
    }


def _backup(result, tmp_path, **kwargs):
    root = _root(tmp_path / "backup")
    return backup.backup_verified_snapshot(
        Path(result["archive_path"]),
        backup_root=root,
        **_args(result),
        **kwargs,
    )


def test_local_backup_round_trip_and_own_inode(published):
    result, tmp_path = published
    receipt = _backup(result, tmp_path)
    origin = Path(result["archive_path"])
    copy = Path(receipt["backup_path"])

    assert copy.read_bytes() == origin.read_bytes()
    assert copy.stat().st_ino != origin.stat().st_ino
    assert receipt["backup_status"] == "LOCAL_COPY_VERIFIED"
    assert receipt["snapshot_sha256"] == result["snapshot_sha256"]
    assert receipt["manifest_sha256"] == result["manifest_sha256"]
    assert not list(copy.parent.glob(".backup-staging-*"))
    assert verify_published_snapshot(copy, **_args(result))[
        "verification_status"
    ] == "LOCAL_INTEGRITY_VERIFIED"


def test_restore_from_backup_after_disposable_source_deleted(published):
    result, tmp_path = published
    receipt = _backup(result, tmp_path)
    Path(result["archive_path"]).unlink()  # Disposable pytest source only.

    restore_root = _root(tmp_path / "restored")
    restored = backup.restore_verified_snapshot(
        Path(receipt["backup_path"]),
        restore_root=restore_root,
        **_args(result),
    )
    path = Path(restored["restored_path"])
    assert path.is_file()
    assert restored["restore_status"] == "LOCAL_RESTORE_VERIFIED"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == result["snapshot_sha256"]
    assert verify_published_snapshot(path, **_args(result))[
        "verification_status"
    ] == "LOCAL_INTEGRITY_VERIFIED"


def test_duplicate_backup_never_overwrites(published):
    result, tmp_path = published
    receipt = _backup(result, tmp_path)
    before = Path(receipt["backup_path"]).read_bytes()
    with pytest.raises(FileExistsError):
        backup.backup_verified_snapshot(
            Path(result["archive_path"]),
            backup_root=tmp_path / "backup",
            **_args(result),
        )
    assert Path(receipt["backup_path"]).read_bytes() == before
    assert not list((tmp_path / "backup" / RUN).glob(".backup-staging-*"))


def test_duplicate_restore_never_overwrites(published):
    result, tmp_path = published
    receipt = _backup(result, tmp_path)
    restore_root = _root(tmp_path / "restored")
    first = backup.restore_verified_snapshot(
        Path(receipt["backup_path"]),
        restore_root=restore_root,
        **_args(result),
    )
    before = Path(first["restored_path"]).read_bytes()
    with pytest.raises(FileExistsError):
        backup.restore_verified_snapshot(
            Path(receipt["backup_path"]),
            restore_root=restore_root,
            **_args(result),
        )
    assert Path(first["restored_path"]).read_bytes() == before


def test_corrupt_original_is_rejected_before_backup(published):
    result, tmp_path = published
    source = Path(result["archive_path"])
    source.write_bytes(source.read_bytes() + b"modified")
    root = _root(tmp_path / "backup")
    with pytest.raises(ValueError, match="Snapshot checksum mismatch"):
        backup.backup_verified_snapshot(
            source, backup_root=root, **_args(result)
        )
    assert list(root.iterdir()) == []


def test_corrupt_backup_is_rejected_before_restore(published):
    result, tmp_path = published
    receipt = _backup(result, tmp_path)
    copy = Path(receipt["backup_path"])
    copy.write_bytes(copy.read_bytes() + b"modified")
    restore_root = _root(tmp_path / "restored")
    with pytest.raises(ValueError, match="Snapshot checksum mismatch"):
        backup.restore_verified_snapshot(
            copy, restore_root=restore_root, **_args(result)
        )
    assert list(restore_root.iterdir()) == []


def test_wrong_trusted_manifest_digest_blocks_backup(published):
    result, tmp_path = published
    root = _root(tmp_path / "backup")
    args = _args(result)
    args["expected_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Manifest checksum mismatch"):
        backup.backup_verified_snapshot(
            Path(result["archive_path"]), backup_root=root, **args
        )
    assert list(root.iterdir()) == []


def test_public_destination_directory_rejected(published):
    result, tmp_path = published
    root = tmp_path / "public"
    root.mkdir(mode=0o755)
    with pytest.raises(PermissionError, match="private"):
        backup.backup_verified_snapshot(
            Path(result["archive_path"]),
            backup_root=root,
            **_args(result),
        )
    assert list(root.iterdir()) == []


def test_symlink_destination_root_rejected(published):
    result, tmp_path = published
    real = _root(tmp_path / "real")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(PermissionError, match="private"):
        backup.backup_verified_snapshot(
            Path(result["archive_path"]), backup_root=alias, **_args(result)
        )


def test_symlink_backup_source_rejected_before_restore(published):
    result, tmp_path = published
    receipt = _backup(result, tmp_path)
    root = _root(tmp_path / "alias")
    case_dir = _root(root / RUN)
    alias = case_dir / "revision-0001.zip"
    alias.symlink_to(receipt["backup_path"])
    restore_root = _root(tmp_path / "restored")
    with pytest.raises(ValueError, match="Invalid or missing snapshot path"):
        backup.restore_verified_snapshot(
            alias, restore_root=restore_root, **_args(result)
        )
    assert list(restore_root.iterdir()) == []


def test_atomic_publish_error_leaves_no_partial_copy(published, monkeypatch):
    result, tmp_path = published
    root = _root(tmp_path / "backup")

    def fail_link(*args, **kwargs):
        raise OSError("simulated no-link failure")

    monkeypatch.setattr(backup.os, "link", fail_link)
    with pytest.raises(OSError, match="simulated no-link failure"):
        backup.backup_verified_snapshot(
            Path(result["archive_path"]), backup_root=root, **_args(result)
        )
    case_dir = root / RUN
    assert case_dir.is_dir()
    assert list(case_dir.iterdir()) == []


def test_detect_source_mutation_between_validation_and_copy(published, monkeypatch):
    result, tmp_path = published
    root = _root(tmp_path / "backup")
    real_verify = backup.verify_published_snapshot
    calls = []

    def mutate_after_verify(path, **kwargs):
        response = real_verify(path, **kwargs)
        calls.append(path)
        if len(calls) == 1:
            source = Path(result["archive_path"])
            source.write_bytes(source.read_bytes() + b"late-change")
        return response

    monkeypatch.setattr(backup, "verify_published_snapshot", mutate_after_verify)
    with pytest.raises(ValueError, match="Copied snapshot checksum mismatch"):
        backup.backup_verified_snapshot(
            Path(result["archive_path"]), backup_root=root, **_args(result)
        )
    assert list((root / RUN).iterdir()) == []

"""Unit 1C.11: pinned, offline replay with two selectors and no live fallback.

This suite uses disposable published fixtures, not the three real Oct-20 runs.
"""

import hashlib
import json
from pathlib import Path

import pytest

from scripts.diagnostics.incident_publication import publish_snapshot
from scripts.diagnostics import incident_replay as replay

DAG_ID = "nyc_lifecycle_probe"
HEALTHY = "probe_20261020T182304Z"
FAILED = "probe_failure_20261020T183606Z"


def _artifact(filename, value):
    raw = (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")
    return {"filename": filename, "content": raw,
            "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _artifacts(run_id, *, failed=False):
    status = "FAILED" if failed else "SUCCEEDED"
    metadata = _artifact("run_evidence_metadata.json", {
        "artifact_type": "RUN_EVIDENCE_METADATA",
        "capture_scope": "METADATA_ONLY",
        "redaction_policy": "ALLOWLIST_V1",
        "source": {
            "schema_version": 2,
            "dag_id": DAG_ID,
            "run_id": run_id,
            "status": status,
            "started_at": "2026-10-20T18:00:00+00:00",
            "finished_at": "2026-10-20T18:02:00+00:00",
            "failure_stage": "harmless_task" if failed else None,
        },
    })
    database = _artifact("airflow_task_diagnostics.json", {
        "artifact_type": "AIRFLOW_TASK_DIAGNOSTICS",
        "source_run_id": run_id,
        "attempt_history_complete": False,
        "capture_scope": "OBSERVED_TEST_RUN_DB_PROJECTION",
        "evidence": {
            "dag_id": DAG_ID,
            "run_id": run_id,
            "task_instances": [{
                "dag_id": DAG_ID,
                "run_id": run_id,
                "task_id": "harmless_task",
                "state": "failed" if failed else "success",
            }],
            "task_failures": ([{
                "dag_id": DAG_ID,
                "run_id": run_id,
                "task_id": "harmless_task",
            }] if failed else []),
        },
    })
    logs = _artifact("log_excerpts.json", {
        "artifact_type": "AIRFLOW_LOG_EXCERPTS",
        "dag_id": DAG_ID,
        "run_id": run_id,
        "task_id": "harmless_task",
        "capture_scope": "BOUNDED_EXCERPTS_ONLY",
        "full_log_archived": False,
        "redaction_policy": "STRICT_MARKER_ALLOWLIST_V1",
        "logs": [{
            "attempt_number": 1,
            "source_path": (
                f"dag_id={DAG_ID}/run_id={run_id}/"
                "task_id=harmless_task/attempt=1.log"
            ),
            "excerpts": [{
                "event": "TERMINAL_FAILURE" if failed else "SUCCESS",
                "line_number": 12,
                "excerpt": ("Intentional lifecycle probe failure"
                            if failed else "Lifecycle probe executed successfully"),
            }],
        }],
    })
    return [metadata, database, logs]


def _private_dir(path):
    path.mkdir(mode=0o700)
    return path


def _index_file(tmp_path, entries):
    trust = _private_dir(tmp_path / "trust")
    path = trust / "trusted_cases.json"
    path.write_text(
        json.dumps({"schema_version": 1, "cases": entries}), encoding="utf-8"
    )
    path.chmod(0o600)
    return path


def _publish(root, run_id, revision, failed=False):
    receipt = publish_snapshot(
        _artifacts(run_id, failed=failed),
        dag_id=DAG_ID, run_id=run_id, revision=revision,
        prepared_at_utc="2026-10-20T20:00:00Z", archive_root=root,
    )
    return {
        "case_id": run_id,
        "label": "Terminal Failure" if failed else "Healthy Run",
        "revision": revision,
        "dag_id": DAG_ID,
        "snapshot_sha256": receipt["snapshot_sha256"],
        "manifest_sha256": receipt["manifest_sha256"],
    }


@pytest.fixture
def library(tmp_path):
    root = _private_dir(tmp_path / "archives")
    entries = [
        _publish(root, HEALTHY, 1),
        _publish(root, FAILED, 1, failed=True),
        _publish(root, FAILED, 2, failed=True),
    ]
    index = _index_file(tmp_path, entries)
    return root, index, entries


def _open(library, case_id=FAILED, revision=1):
    root, index, _ = library
    return replay.load_verified_replay(
        archive_root=root, trusted_index_path=index,
        case_id=case_id, revision=revision,
    )


def test_multiple_case_selectors_and_revision_order(library):
    _, index, _ = library
    cases = replay.available_cases(index)
    assert [c["case_id"] for c in cases] == [HEALTHY, FAILED]
    failed_case = next(c for c in cases if c["case_id"] == FAILED)
    assert failed_case["revisions"] == [2, 1]
    assert failed_case["label"] == "Terminal Failure"


def test_loads_only_verified_pinned_bytes(library):
    result = _open(library)
    assert result["verification_status"] == "LOCAL_INTEGRITY_VERIFIED"
    assert result["capture_status"] == "PUBLISHED_PENDING_BACKUP"
    assert result["source"]["run_id"] == FAILED
    assert result["source"]["status"] == "FAILED"
    assert result["task_failures"][0]["run_id"] == FAILED
    assert result["log_attempts"][0]["excerpts"][0]["event"] == "TERMINAL_FAILURE"
    assert result["capture_scope"]["attempt_history_complete"] is False


def test_case_switch_never_reuses_previous_content(library):
    failed = _open(library, FAILED, 2)
    healthy = _open(library, HEALTHY, 1)
    assert failed["source"]["status"] == "FAILED"
    assert healthy["source"]["status"] == "SUCCEEDED"
    assert healthy["task_failures"] == []
    assert healthy["log_attempts"][0]["excerpts"][0]["event"] == "SUCCESS"


def test_unregistered_case_or_revision_fails_closed(library):
    with pytest.raises(replay.ReplayError) as exc:
        _open(library, FAILED, 3)
    assert exc.value.code == "EVIDENCE_MISSING"
    with pytest.raises(replay.ReplayError) as exc:
        _open(library, "unregistered", 1)
    assert exc.value.code == "EVIDENCE_MISSING"


def test_corrupted_archive_is_never_shown(library):
    root, _, _ = library
    archive = root / FAILED / "revision-0001.zip"
    archive.write_bytes(archive.read_bytes() + b"tampered")
    with pytest.raises(replay.ReplayError) as exc:
        _open(library)
    assert exc.value.code == "INTEGRITY_FAILED"


def test_missing_archive_fails_closed(library):
    root, _, _ = library
    (root / FAILED / "revision-0001.zip").unlink()
    with pytest.raises(replay.ReplayError) as exc:
        _open(library)
    assert exc.value.code == "EVIDENCE_MISSING"


def test_wrong_external_digest_is_rejected(library):
    _, index, records = library
    modified = [dict(x) for x in records]
    modified[1]["snapshot_sha256"] = "0" * 64
    index.write_text(
        json.dumps({"schema_version": 1, "cases": modified}), encoding="utf-8"
    )
    with pytest.raises(replay.ReplayError) as exc:
        _open(library)
    assert exc.value.code == "INTEGRITY_FAILED"


def test_duplicate_revision_rejected(library):
    _, index, records = library
    index.write_text(
        json.dumps({"schema_version": 1, "cases": records + [records[1]]}),
        encoding="utf-8",
    )
    with pytest.raises(replay.ReplayError) as exc:
        replay.available_cases(index)
    assert exc.value.code == "INDEX_INVALID"


def test_inconsistent_labels_rejected(library):
    _, index, records = library
    modified = [dict(x) for x in records]
    modified[2]["label"] = "different-label"
    index.write_text(
        json.dumps({"schema_version": 1, "cases": modified}), encoding="utf-8"
    )
    with pytest.raises(replay.ReplayError) as exc:
        replay.available_cases(index)
    assert exc.value.code == "INDEX_INVALID"


def test_trusted_index_permission_enforced(library):
    _, index, _ = library
    index.chmod(0o644)
    with pytest.raises(replay.ReplayError) as exc:
        replay.available_cases(index)
    assert exc.value.code == "INDEX_INVALID"


def test_case_id_path_traversal_rejected(library):
    _, index, records = library
    changed = [dict(records[0], case_id="../escape")]
    index.write_text(
        json.dumps({"schema_version": 1, "cases": changed}), encoding="utf-8"
    )
    with pytest.raises(replay.ReplayError) as exc:
        replay.available_cases(index)
    assert exc.value.code == "INDEX_INVALID"


def test_symlinked_case_directory_rejected(library, tmp_path):
    root, _, _ = library
    # Redirect only the disposable fixture directory.
    target = root / FAILED
    alias = root / "renamed-real-case"
    target.rename(alias)
    target.symlink_to(alias, target_is_directory=True)
    with pytest.raises(replay.ReplayError) as exc:
        _open(library)
    assert exc.value.code == "UNAVAILABLE"


def test_index_must_be_separate_from_archive_root(library):
    root, index, entries = library
    protected = _private_dir(root / "index")
    bad = protected / "case_index.json"
    bad.write_text(
        json.dumps({"schema_version": 1, "cases": entries}), encoding="utf-8"
    )
    bad.chmod(0o600)
    with pytest.raises(replay.ReplayError) as exc:
        replay.load_verified_replay(
            archive_root=root, trusted_index_path=bad,
            case_id=HEALTHY, revision=1,
        )
    assert exc.value.code == "INDEX_INVALID"


def test_malformed_json_index_rejected(library):
    _, index, _ = library
    index.write_text("not json", encoding="utf-8")
    with pytest.raises(replay.ReplayError) as exc:
        replay.available_cases(index)
    assert exc.value.code == "INDEX_INVALID"


def test_verified_bytes_are_not_reopened_after_check(library, monkeypatch):
    root, _, _ = library
    archive = root / FAILED / "revision-0001.zip"
    original = replay.verify_published_snapshot

    def change_after_verify(path, **kwargs):
        checked = original(path, **kwargs)
        # Simulate a replacement AFTER the verifier finished its single read.
        archive.write_bytes(b"now-corrupted")
        return checked

    monkeypatch.setattr(replay, "verify_published_snapshot", change_after_verify)
    result = _open(library)
    assert result["source"]["run_id"] == FAILED
    assert result["log_attempts"][0]["excerpts"][0]["event"] == "TERMINAL_FAILURE"

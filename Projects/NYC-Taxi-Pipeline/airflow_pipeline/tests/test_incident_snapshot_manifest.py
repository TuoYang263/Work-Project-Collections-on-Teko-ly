"""Unit 1C — Snapshot manifest batch: identity, provenance and digest tests.

Place alongside test_incident_publication.py under airflow_pipeline/tests/.
"""
import hashlib
import json

import pytest

from scripts.diagnostics.incident_capture import prepare_run_metadata
from scripts.diagnostics.incident_publication import (
    build_snapshot_manifest,
    manifest_bytes_and_digest,
)

DAG_ID = "nyc_lifecycle_probe"
RUN_ID = "probe_failure_20261020T183606Z"
PREPARED_AT = "2026-11-04T18:00:00Z"


def _json_artifact(filename, payload):
    content = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
    return {
        "filename": filename,
        "content": content,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _artifacts(started_at="2026-10-20T18:36:00+00:00"):
    metadata = prepare_run_metadata({
        "schema_version": 2,
        "dag_id": DAG_ID,
        "run_id": RUN_ID,
        "status": "FAILED",
        "started_at": started_at,
        "finished_at": "2026-10-20T18:37:00+00:00",
        "failure_stage": "harmless_task",
    })
    database = _json_artifact("airflow_task_diagnostics.json", {
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
    logs = _json_artifact("log_excerpts.json", {
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
    return [logs, metadata, database]


def _manifest(artifacts=None, **overrides):
    arguments = {
        "artifacts": _artifacts() if artifacts is None else artifacts,
        "dag_id": DAG_ID,
        "run_id": RUN_ID,
        "revision": 1,
        "prepared_at_utc": PREPARED_AT,
    }
    arguments.update(overrides)
    return build_snapshot_manifest(**arguments)


def test_manifest_preserves_provenance_and_does_not_claim_publication():
    manifest = _manifest()
    assert manifest["schema_version"] == 1
    assert manifest["case_id"] == RUN_ID
    assert manifest["revision"] == 1
    assert manifest["provenance"] == "OBSERVED_TEST_RUN"
    assert manifest["source_dag_id"] == DAG_ID
    assert manifest["source_run_id"] == RUN_ID
    assert manifest["source_status"] == "FAILED"
    assert manifest["source_failure_stage"] == "harmless_task"
    assert manifest["prepared_at_utc"] == PREPARED_AT
    assert manifest["captured_at_utc"] is None
    assert manifest["capture_status"] == "PREPARED_NOT_PUBLISHED"
    assert manifest["snapshot_sha256"] is None
    assert len(manifest["attachments"]) == 3
    assert all("content" not in row for row in manifest["attachments"])


def test_manifest_and_digest_are_order_independent():
    artifacts = _artifacts()
    first = _manifest(artifacts)
    second = _manifest(list(reversed(artifacts)))
    assert first == second
    first_bytes, first_hash = manifest_bytes_and_digest(first)
    second_bytes, second_hash = manifest_bytes_and_digest(second)
    assert first_bytes == second_bytes
    assert first_hash == second_hash
    assert first_hash == hashlib.sha256(first_bytes).hexdigest()
    assert json.loads(first_bytes) == first


def test_digest_changes_when_evidence_changes():
    original = _manifest()
    modified = _manifest(_artifacts(started_at="2026-10-20T18:35:00+00:00"))
    assert manifest_bytes_and_digest(original)[1] != manifest_bytes_and_digest(modified)[1]


@pytest.mark.parametrize("revision", [0, -1, True, "1"])
def test_manifest_rejects_bad_revision(revision):
    with pytest.raises(ValueError, match="Invalid snapshot revision"):
        _manifest(revision=revision)


@pytest.mark.parametrize("timestamp", ["2026-11-04", "2026-11-04T18:00:00", "bad", "2026-11-04T20:00:00+02:00"])
def test_manifest_requires_explicit_utc_time(timestamp):
    with pytest.raises(ValueError, match="Invalid UTC preparation timestamp"):
        _manifest(prepared_at_utc=timestamp)


def test_manifest_rejects_path_like_case_id():
    with pytest.raises(ValueError, match="Invalid snapshot case identity"):
        _manifest(run_id="../escape")


def test_manifest_requires_all_three_artifacts():
    with pytest.raises(ValueError, match="Snapshot artifact set mismatch"):
        _manifest(_artifacts()[:2])


def test_manifest_preserves_source_identity_fail_closed():
    with pytest.raises(ValueError, match="Run evidence identity mismatch"):
        _manifest(run_id="another_run")

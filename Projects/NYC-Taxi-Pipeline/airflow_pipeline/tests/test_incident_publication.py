import json
import hashlib
import pytest

from scripts.diagnostics.incident_publication import (
    validate_prepared_artifact,
    build_attachment_record,
    build_attachment_records,
    validate_snapshot_identity,
)

from scripts.diagnostics.incident_capture import (
    prepare_run_metadata,
)


def test_valid_artifact():
    content = b'{"status":"FAILED"}'

    artifact = {
        "filename": "run_evidence_metadata.json",
        "content": content,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }

    validate_prepared_artifact(artifact)


def test_invalid_checksum():
    content = b'{"status":"FAILED"}'

    artifact = {
        "filename": "run_evidence_metadata.json",
        "content": content,
        "size_bytes": len(content),
        "sha256": "0" * 64,
    }

    with pytest.raises(
        ValueError,
        match="Artifact checksum mismatch",
    ):
        validate_prepared_artifact(artifact)


def test_artifact_size_mismatch():
    content = b'{"status":"FAILED"}'

    artifact = {
        "filename": "run_evidence_metadata.json",
        "content": content,
        "size_bytes": len(content) + 10,
        "sha256": hashlib.sha256(content).hexdigest(),
    }

    with pytest.raises(
        ValueError,
        match="Artifact size mismatch",
    ):
        validate_prepared_artifact(artifact)


def test_reject_path_traversal():
    content = b'{"status":"FAILED"}'

    artifact = {
        "filename": "../secret.json",
        "content": content,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }

    with pytest.raises(
        ValueError,
        match="Invalid artifact filename",
    ):
        validate_prepared_artifact(artifact)


def test_reject_empty_content():
    content = b""

    artifact = {
        "filename": "run_evidence_metadata.json",
        "content": content,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }

    with pytest.raises(
        ValueError,
        match="Invalid artifact content",
    ):
        validate_prepared_artifact(artifact)


def test_reject_string_content():
    content = '{"status":"FAILED"}'

    artifact = {
        "filename": "run_evidence_metadata.json",
        "content": content,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest(),
    }

    with pytest.raises(
        ValueError,
        match="Invalid artifact content",
    ):
        validate_prepared_artifact(artifact)


def test_build_attachment_record():
    content = b'{"status":"FAILED"}'
    checksum = hashlib.sha256(content).hexdigest()

    artifact = {
        "filename": "run_evidence_metadata.json",
        "content": content,
        "size_bytes": len(content),
        "sha256": checksum,
    }

    record = build_attachment_record(artifact)

    assert record == {
        "relative_path": "run_evidence_metadata.json",
        "size_bytes": len(content),
        "sha256": checksum,
    }

    assert "content" not in record


def test_reject_duplicate_artifacts():
    content = b'{"status":"FAILED"}'

    artifact = {
        "filename": "run_evidence_metadata.json",
        "content": content,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }

    with pytest.raises(
        ValueError,
        match="Duplicate artifact filename",
    ):
        build_attachment_records([
            artifact,
            artifact,
        ])


def test_multiple_artifacts_have_stable_order():
    filenames = [
        "log_excerpts.json",
        "run_evidence_metadata.json",
        "airflow_task_diagnostics.json",
    ]

    artifacts = []

    for filename in filenames:
        content = filename.encode("utf-8")

        artifacts.append({
            "filename": filename,
            "content": content,
            "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        })

    records = build_attachment_records(artifacts)

    assert [
        record["relative_path"]
        for record in records
    ] == sorted(filenames)

    reversed_records = build_attachment_records(
        list(reversed(artifacts))
    )

    assert records == reversed_records


def make_json_artifact(filename, payload):
    content = json.dumps(
        payload, sort_keys=True
    ).encode("utf-8")

    return {
        "filename": filename,
        "content": content,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def make_snapshot_artifacts(
    db_run_id=None,
    log_run_id=None,
):
    dag_id = "nyc_lifecycle_probe"
    run_id = "probe_failure_20261020T183606Z"

    metadata = prepare_run_metadata({
        "schema_version": 2,
        "dag_id": dag_id,
        "run_id": run_id,
        "status": "FAILED",
        "started_at": "2026-10-20T18:00:00+00:00",
        "finished_at": "2026-10-20T18:01:00+00:00",
        "failure_stage": "harmless_task",
    })

    db_id = db_run_id or run_id
    log_id = log_run_id or run_id

    database = make_json_artifact(
        "airflow_task_diagnostics.json",
        {
            "artifact_type": "AIRFLOW_TASK_DIAGNOSTICS",
            "source_run_id": db_id,
            "evidence": {
                "dag_id": dag_id,
                "run_id": db_id,
                "task_instances": [
                    {
                        "dag_id": dag_id,
                        "run_id": db_id,
                        "task_id": "harmless_task",
                    }
                ],
                "task_failures": [],
            },
        },
    )

    logs = make_json_artifact(
        "log_excerpts.json",
        {
            "artifact_type": "AIRFLOW_LOG_EXCERPTS",
            "dag_id": dag_id,
            "run_id": log_id,
            "task_id": "harmless_task",
            "logs": [
                {
                    "attempt_number": 1,
                    "source_path": (
                        f"dag_id={dag_id}/run_id={log_id}/"
                        "task_id=harmless_task/attempt=1.log"
                    ),
                }
            ],
        },
    )

    return [metadata, database, logs]


def test_snapshot_identity_accepts_matching_run():
    artifacts = make_snapshot_artifacts()

    validate_snapshot_identity(
        artifacts,
        "nyc_lifecycle_probe",
        "probe_failure_20261020T183606Z",
    )


def test_snapshot_rejects_cross_run_database():
    artifacts = make_snapshot_artifacts(
        db_run_id="another_run"
    )

    with pytest.raises(
        ValueError,
        match="Database evidence identity mismatch",
    ):
        validate_snapshot_identity(
            artifacts,
            "nyc_lifecycle_probe",
            "probe_failure_20261020T183606Z",
        )


def test_snapshot_rejects_cross_run_logs():
    artifacts = make_snapshot_artifacts(
        log_run_id="another_run"
    )

    with pytest.raises(
        ValueError,
        match="Log evidence identity mismatch",
    ):
        validate_snapshot_identity(
            artifacts,
            "nyc_lifecycle_probe",
            "probe_failure_20261020T183606Z",
        )
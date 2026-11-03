
"""Tests for the isolated PostgreSQL incident evidence collector."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts.diagnostics import incident_db_capture as capture


def sample_payload():
    tasks = []
    failures = []

    for run_id in capture.RUN_IDS:
        for task_id in ("start_run_evidence", "harmless_task"):
            tasks.append({
                "dag_id": capture.DAG_ID,
                "run_id": run_id,
                "task_id": task_id,
                "map_index": -1,
                "state": "success",
                "start_date": None,
                "end_date": None,
            })

        for _ in range(capture.EXPECTED_FAILURES[run_id]):
            failures.append({
                "dag_id": capture.DAG_ID,
                "run_id": run_id,
                "task_id": "harmless_task",
                "map_index": -1,
                "start_date": None,
                "end_date": None,
                "duration": 1.0,
            })

    return {
        "source": "ISOLATED_LIFECYCLE_DB_COPY",
        "dag_id": capture.DAG_ID,
        "task_instances": tasks,
        "task_failures": failures,
    }


def test_valid_database_evidence():
    data = sample_payload()
    assert capture.validate_db_evidence(data) == data


def test_missing_task_rejected():
    data = sample_payload()
    data["task_instances"].pop()

    with pytest.raises(ValueError, match="Incomplete"):
        capture.validate_db_evidence(data)


def test_missing_failure_rejected():
    data = sample_payload()
    data["task_failures"].pop()

    with pytest.raises(ValueError, match="Incomplete"):
        capture.validate_db_evidence(data)


def test_cross_run_evidence_rejected():
    data = sample_payload()
    data["task_instances"][0]["run_id"] = "another_run"

    with pytest.raises(ValueError, match="identity mismatch"):
        capture.validate_db_evidence(data)


def test_deterministic_artifact():
    data = sample_payload()

    first = capture.prepare_db_artifact(data)
    second = capture.prepare_db_artifact(copy.deepcopy(data))

    assert first["content"] == second["content"]
    assert first["size_bytes"] == len(first["content"])
    assert first["sha256"] == hashlib.sha256(
        first["content"]
    ).hexdigest()

    document = json.loads(first["content"])
    assert document["attempt_history_complete"] is False


def test_database_query_is_read_only(monkeypatch):
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return SimpleNamespace(stdout=json.dumps(sample_payload()))

    monkeypatch.setattr(capture.subprocess, "run", fake_run)

    result = capture.read_probe_db_evidence()

    assert len(result["task_instances"]) == 6
    assert len(result["task_failures"]) == 3
    assert any(
        "default_transaction_read_only=on" in argument
        for argument in observed["command"]
    )
    assert observed["kwargs"]["check"] is True
    assert observed["kwargs"]["timeout"] == 30


def test_per_run_artifacts_are_isolated():
    payload = sample_payload()
    hashes = set()

    for run_id in capture.RUN_IDS:
        artifact = capture.prepare_run_db_artifact(
            payload, run_id
        )

        document = json.loads(artifact["content"])
        evidence = document["evidence"]

        assert document["source_run_id"] == run_id
        assert len(evidence["task_instances"]) == 2
        assert len(evidence["task_failures"]) == (
            capture.EXPECTED_FAILURES[run_id]
        )

        for row in (
            evidence["task_instances"]
            + evidence["task_failures"]
        ):
            assert row["run_id"] == run_id

        hashes.add(artifact["sha256"])

    assert len(hashes) == 3

    with pytest.raises(
        ValueError, match="Unsupported incident run"
    ):
        capture.prepare_run_db_artifact(
            payload, "unknown_run"
        )

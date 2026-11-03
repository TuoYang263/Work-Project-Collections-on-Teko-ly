
"""Tests for bounded incident log excerpts."""

import hashlib
import json

import pytest

from scripts.diagnostics.incident_log_capture import (
    DAG_ID,
    TASK_ID,
    EXPECTED_ATTEMPTS,
    MAX_FILE_BYTES,
    prepare_log_excerpt_artifact,
)


@pytest.fixture
def log_root(tmp_path):
    for run_id, attempts in EXPECTED_ATTEMPTS.items():
        folder = (
            tmp_path
            / f"dag_id={DAG_ID}"
            / f"run_id={run_id}"
            / f"task_id={TASK_ID}"
        )
        folder.mkdir(parents=True)

        for attempt in attempts:
            message = (
                "Lifecycle probe executed successfully"
                if run_id == list(EXPECTED_ATTEMPTS)[0]
                else "Intentional first-attempt failure"
            )

            (folder / f"attempt={attempt}.log").write_text(
                "Bearer PRIVATE_TEST_TOKEN\n"
                + message + "\n",
                encoding="utf-8",
            )

    return tmp_path


@pytest.mark.parametrize("run_id", list(EXPECTED_ATTEMPTS))
def test_capture_is_scoped_and_redacted(log_root, run_id):
    artifact = prepare_log_excerpt_artifact(
        run_id, log_root=log_root
    )
    document = json.loads(artifact["content"])

    assert document["run_id"] == run_id
    assert document["full_log_archived"] is False
    assert len(document["logs"]) == len(
        EXPECTED_ATTEMPTS[run_id]
    )
    assert b"PRIVATE_TEST_TOKEN" not in artifact["content"]
    assert artifact["sha256"] == hashlib.sha256(
        artifact["content"]
    ).hexdigest()


def test_missing_attempt_is_rejected(log_root):
    run_id = list(EXPECTED_ATTEMPTS)[1]
    path = (
        log_root
        / f"dag_id={DAG_ID}"
        / f"run_id={run_id}"
        / f"task_id={TASK_ID}"
        / "attempt=2.log"
    )
    path.unlink()

    with pytest.raises(ValueError, match="Incomplete"):
        prepare_log_excerpt_artifact(
            run_id, log_root=log_root
        )


def test_oversized_log_is_rejected(log_root):
    run_id = list(EXPECTED_ATTEMPTS)[0]
    path = (
        log_root
        / f"dag_id={DAG_ID}"
        / f"run_id={run_id}"
        / f"task_id={TASK_ID}"
        / "attempt=1.log"
    )
    path.write_text("X" * (MAX_FILE_BYTES + 1))

    with pytest.raises(ValueError, match="size limit"):
        prepare_log_excerpt_artifact(
            run_id, log_root=log_root
        )


def test_unsupported_run_is_rejected(log_root):
    with pytest.raises(ValueError, match="Unsupported"):
        prepare_log_excerpt_artifact(
            "unknown_run", log_root=log_root
        )
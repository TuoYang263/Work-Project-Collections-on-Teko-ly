import pytest

from scripts.run_evidence import (
    finalize_run_evidence,
    load_run_evidence,
    start_run_evidence,
)


def _contract():
    return {
        "mode": "BACKFILL",
        "execution_scope": "GOLD_ONLY",
        "logical_window": {
            "start": "2023-02-01",
            "end_exclusive":
                "2023-03-01",
            "anchor": "PICKUP_DATETIME",
        },
        "read_scope": {
            "input_layer": "SILVER",
            "boundary_policy":
                "REUSE_EXISTING_SILVER_STATE",
        },
        "write_scope": "WINDOW_REPLACE",
        "rerun_semantics":
            "REPLACE_LOGICAL_WINDOW_IDEMPOTENT",
    }


def test_start_run_evidence_persists_record(
    tmp_path,
):
    record = start_run_evidence(
        "manual__2023-02",
        _contract(),
        directory=tmp_path,
        started_at="2026-10-10T10:00:00+00:00",
    )

    assert record["status"] == "RUNNING"
    assert record["mode"] == "BACKFILL"
    assert record["execution_scope"] == (
        "GOLD_ONLY"
    )
    assert record["write_scope"] == (
        "WINDOW_REPLACE"
    )

    loaded = load_run_evidence(
        "manual__2023-02",
        directory=tmp_path,
    )

    assert loaded == record


def test_finalize_run_success(
    tmp_path,
):
    start_run_evidence(
        "run-success",
        _contract(),
        directory=tmp_path,
        started_at="2026-10-10T10:00:00+00:00",
    )

    record = finalize_run_evidence(
        "run-success",
        "SUCCEEDED",
        directory=tmp_path,
        finished_at="2026-10-10T10:05:00+00:00",
    )

    assert record["status"] == "SUCCEEDED"
    assert record["failure_stage"] is None
    assert record["finished_at"] == (
        "2026-10-10T10:05:00+00:00"
    )


def test_finalize_failed_run_requires_stage(
    tmp_path,
):
    start_run_evidence(
        "run-failed",
        _contract(),
        directory=tmp_path,
    )

    with pytest.raises(
        ValueError,
        match="requires failure_stage",
    ):
        finalize_run_evidence(
            "run-failed",
            "FAILED",
            directory=tmp_path,
        )

    record = finalize_run_evidence(
        "run-failed",
        "FAILED",
        failure_stage="gold",
        directory=tmp_path,
        finished_at="2026-10-10T10:05:00+00:00",
    )

    assert record["status"] == "FAILED"
    assert record["failure_stage"] == "gold"


def test_terminal_run_cannot_be_overwritten(
    tmp_path,
):
    start_run_evidence(
        "run-terminal",
        _contract(),
        directory=tmp_path,
    )

    finalize_run_evidence(
        "run-terminal",
        "SUCCEEDED",
        directory=tmp_path,
    )

    with pytest.raises(
        RuntimeError,
        match="already terminal",
    ):
        finalize_run_evidence(
            "run-terminal",
            "FAILED",
            failure_stage="gold",
            directory=tmp_path,
        )
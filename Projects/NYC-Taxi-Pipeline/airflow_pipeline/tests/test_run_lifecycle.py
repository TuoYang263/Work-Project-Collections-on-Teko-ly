
from types import SimpleNamespace

import pytest

from scripts.run_evidence import (
    load_run_evidence,
)

from scripts.run_lifecycle import (
    mark_pipeline_failure,
    mark_pipeline_success,
    start_pipeline_run,
)


def _contract():
    return {
        "mode": "BACKFILL",
        "execution_scope": "GOLD_ONLY",
        "logical_window": {
            "start": "2023-02-01",
            "end_exclusive": "2023-03-01",
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


def _context(run_id, tasks=None):
    dag_run = SimpleNamespace(
        run_id=run_id,
        get_task_instances=lambda: tasks or [],
    )

    return {"dag_run": dag_run}


def test_start_pipeline_run(tmp_path):
    context = _context("run-start")

    record = start_pipeline_run(
        context,
        _contract(),
        directory=tmp_path,
    )

    assert record["run_id"] == "run-start"
    assert record["status"] == "RUNNING"
    assert record["mode"] == "BACKFILL"


def test_mark_pipeline_success(tmp_path):
    context = _context("run-success")

    start_pipeline_run(
        context,
        _contract(),
        directory=tmp_path,
    )

    mark_pipeline_success(
        context,
        directory=tmp_path,
    )

    record = load_run_evidence(
        "run-success",
        directory=tmp_path,
    )

    assert record["status"] == "SUCCEEDED"
    assert record["finished_at"] is not None
    assert record["failure_stage"] is None


def test_failure_ignores_skipped_branch(tmp_path):
    tasks = [
        SimpleNamespace(
            task_id="bronze_ingest_delta",
            state="skipped",
        ),
        SimpleNamespace(
            task_id="gold_backfill_delta",
            state="failed",
        ),
    ]

    context = _context(
        "run-failed",
        tasks,
    )

    start_pipeline_run(
        context,
        _contract(),
        directory=tmp_path,
    )

    mark_pipeline_failure(
        context,
        directory=tmp_path,
    )

    record = load_run_evidence(
        "run-failed",
        directory=tmp_path,
    )

    assert record["status"] == "FAILED"
    assert record["failure_stage"] == (
        "gold_backfill_delta"
    )


def test_failure_without_failed_task_is_rejected(
    tmp_path,
):
    context = _context(
        "run-no-failure",
        [
            SimpleNamespace(
                task_id="bronze_ingest_delta",
                state="skipped",
            ),
        ],
    )

    start_pipeline_run(
        context,
        _contract(),
        directory=tmp_path,
    )

    with pytest.raises(
        RuntimeError,
        match="Cannot determine failed",
    ):
        mark_pipeline_failure(
            context,
            directory=tmp_path,
        )

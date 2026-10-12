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

def test_v2_same_run_id_isolated_across_dags(tmp_path):
    start_run_evidence("shared", _contract(), dag_id="dag_a", directory=tmp_path)
    start_run_evidence("shared", _contract(), dag_id="dag_b", directory=tmp_path)
    finalize_run_evidence("shared", "SUCCEEDED", dag_id="dag_a", directory=tmp_path)
    assert load_run_evidence("shared", dag_id="dag_a", directory=tmp_path)["status"] == "SUCCEEDED"
    assert load_run_evidence("shared", dag_id="dag_b", directory=tmp_path)["status"] == "RUNNING"
    assert len(list((tmp_path / "v2").glob("*.json"))) == 2


def test_v2_strict_lookup_does_not_fallback_to_v1(tmp_path):
    start_run_evidence("v1-only", _contract(), directory=tmp_path)
    assert load_run_evidence("v1-only", directory=tmp_path)["schema_version"] == 1
    with pytest.raises(FileNotFoundError):
        load_run_evidence("v1-only", directory=tmp_path, dag_id="dag_a")
    with pytest.raises(RuntimeError, match="Legacy run evidence"):
        start_run_evidence("v1-only", _contract(), directory=tmp_path, dag_id="dag_a")


def test_v2_identity_validation_and_wrong_dag(tmp_path):
    start_run_evidence("x", _contract(), dag_id="dag_a", directory=tmp_path)
    with pytest.raises(FileNotFoundError):
        load_run_evidence("x", dag_id="dag_b", directory=tmp_path)
    with pytest.raises(ValueError, match="dag_id must"):
        start_run_evidence("y", _contract(), dag_id=" ", directory=tmp_path)


def test_concurrent_start_only_creates_once(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    def attempt(_):
        try:
            start_run_evidence("simultaneous", _contract(), dag_id="test_dag", directory=tmp_path)
            return "started"
        except RuntimeError as exc:
            assert "already exists" in str(exc)
            return "rejected"
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, range(2)))
    assert sorted(results) == ["rejected", "started"]


def test_concurrent_terminal_write_only_once(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    start_run_evidence("race", _contract(), dag_id="test_dag", directory=tmp_path)
    def attempt(status):
        try:
            finalize_run_evidence(
                "race", status, dag_id="test_dag", directory=tmp_path,
                failure_stage="failed_task" if status == "FAILED" else None,
            )
            return "finalized"
        except RuntimeError as exc:
            assert "already terminal" in str(exc)
            return "rejected"
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, ("FAILED", "SUCCEEDED")))
    assert sorted(results) == ["finalized", "rejected"]
    assert load_run_evidence("race", dag_id="test_dag", directory=tmp_path)["status"] in {"FAILED", "SUCCEEDED"}

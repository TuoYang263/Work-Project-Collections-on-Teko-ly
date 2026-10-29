
"""Tests for Airflow log reference isolation."""

import pytest

from scripts.diagnostics.log_references import (
    get_task_log_references,
)


def test_rejects_cross_run_symlink(tmp_path):
    root = tmp_path / "logs"

    dag_dir = root / "dag_id=nyc_lifecycle_probe"

    other_task = (
        dag_dir
        / "run_id=run-b"
        / "task_id=harmless_task"
    )

    other_task.mkdir(parents=True)
    (other_task / "attempt=1.log").write_text(
        "other run log",
        encoding="utf-8",
    )

    # run-a must not access run-b through a symlink.
    (dag_dir / "run_id=run-a").symlink_to(
        dag_dir / "run_id=run-b",
        target_is_directory=True,
    )

    with pytest.raises(ValueError):
        get_task_log_references(
            dag_id="nyc_lifecycle_probe",
            run_id="run-a",
            task_id="harmless_task",
            log_root=root,
        )


def test_ignores_cross_run_log_file_symlink(tmp_path):
    root = tmp_path / "logs"
    dag_dir = root / "dag_id=nyc_lifecycle_probe"

    other_task = (
        dag_dir / "run_id=run-b" / "task_id=harmless_task"
    )
    other_task.mkdir(parents=True)

    target = other_task / "attempt=1.log"
    target.write_text("other run log", encoding="utf-8")

    requested_task = (
        dag_dir / "run_id=run-a" / "task_id=harmless_task"
    )
    requested_task.mkdir(parents=True)

    (requested_task / "attempt=1.log").symlink_to(target)

    result = get_task_log_references(
        "nyc_lifecycle_probe",
        "run-a",
        "harmless_task",
        log_root=root,
    )

    assert result["available"] is False
    assert result["references"] == []


def test_discovers_logs_and_reports_missing(tmp_path):
    root = tmp_path / "logs"
    task_dir = (
        root
        / "dag_id=nyc_lifecycle_probe"
        / "run_id=run-a"
        / "task_id=harmless_task"
    )
    task_dir.mkdir(parents=True)

    # Create real log files in reverse order.
    (task_dir / "attempt=2.log").write_text("second")
    (task_dir / "attempt=1.log").write_text("first")

    result = get_task_log_references(
        "nyc_lifecycle_probe",
        "run-a",
        "harmless_task",
        log_root=root,
    )

    assert result["available"] is True
    assert result["truncated"] is False
    assert [
        ref["attempt_number"]
        for ref in result["references"]
    ] == [1, 2]

    # A missing run must not borrow another run's logs.
    missing = get_task_log_references(
        "nyc_lifecycle_probe",
        "missing-run",
        "harmless_task",
        log_root=root,
    )

    assert missing["available"] is False
    assert missing["references"] == []


def test_limits_log_references(tmp_path):
    root = tmp_path / "logs"
    task_dir = (
        root
        / "dag_id=test_dag"
        / "run_id=test_run"
        / "task_id=test_task"
    )
    task_dir.mkdir(parents=True)

    for n in range(1, 22):
        (task_dir / f"attempt={n}.log").write_text("test")

    result = get_task_log_references(
        "test_dag",
        "test_run",
        "test_task",
        log_root=root,
    )

    assert result["available"] is True
    assert result["truncated"] is True
    assert len(result["references"]) == 20

    attempts = [
        ref["attempt_number"]
        for ref in result["references"]
    ]
    assert attempts == list(range(1, 21))


@pytest.mark.parametrize(
    "dag_id, run_id, task_id",
    [
        ("", "run-a", "task-a"),
        ("test-dag", "", "task-a"),
        ("test-dag", "run-a", ""),
        ("test-dag", "../run-b", "task-a"),
        ("test-dag", "run-a", r"..\task-b"),
        ("test-dag", "run-a", "bad\x00id"),
    ],
)
def test_rejects_invalid_log_identity(
    tmp_path, dag_id, run_id, task_id
):
    with pytest.raises(ValueError):
        get_task_log_references(
            dag_id,
            run_id,
            task_id,
            log_root=tmp_path,
        )

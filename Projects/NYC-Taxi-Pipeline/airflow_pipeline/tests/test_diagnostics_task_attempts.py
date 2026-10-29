
"""Tests for read-only task diagnostics."""
import pytest

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

from scripts.diagnostics import task_attempts


def test_returns_current_task_snapshot(monkeypatch):
    ti = SimpleNamespace(
        task_id="harmless_task",
        map_index=-1,
        state="failed",
        prev_attempted_tries=1,
        start_date=None,
        end_date=None,
    )

    session = MagicMock()
    query = session.query.return_value
    query.filter.return_value.order_by.return_value.all.return_value = [ti]

    @contextmanager
    def fake_session():
        yield session

    monkeypatch.setattr(task_attempts, "create_session", fake_session)

    result = task_attempts.get_task_attempts(
        "nyc_lifecycle_probe", "test-run"
    )

    assert result["source"] == "TASK_INSTANCE_CURRENT"
    assert result["attempt_history_complete"] is False
    assert result["tasks"][0]["task_id"] == "harmless_task"
    assert result["tasks"][0]["state"] == "failed"


def test_filters_by_task_id(monkeypatch):
    session = MagicMock()
    query = session.query.return_value
    filtered = query.filter.return_value
    filtered.filter.return_value.order_by.return_value.all.return_value = []

    @contextmanager
    def fake_session():
        yield session

    monkeypatch.setattr(task_attempts, "create_session", fake_session)

    result = task_attempts.get_task_attempts(
        "nyc_lifecycle_probe",
        "test-run",
        task_id="harmless_task",
    )

    filtered.filter.assert_called_once()
    assert result["tasks"] == []


@pytest.mark.parametrize(
    "dag_id, run_id",
    [
        ("", "test-run"),
        ("nyc_lifecycle_probe", ""),
    ],
)
def test_rejects_missing_identity(monkeypatch, dag_id, run_id):
    fake_session = MagicMock()
    monkeypatch.setattr(
        task_attempts, "create_session", fake_session
    )

    with pytest.raises(ValueError):
        task_attempts.get_task_attempts(dag_id, run_id)

    fake_session.assert_not_called()


def test_returns_recorded_task_failures(monkeypatch):
    failures = [
        SimpleNamespace(
            task_id="harmless_task",
            map_index=-1,
            start_date=None,
            end_date=None,
            duration=0,
        )
        for _ in range(2)
    ]

    session = MagicMock()
    query = session.query.return_value
    query.filter.return_value.order_by.return_value.all.return_value = failures

    @contextmanager
    def fake_session():
        yield session

    monkeypatch.setattr(task_attempts, "create_session", fake_session)

    result = task_attempts.get_task_failures(
        "nyc_lifecycle_probe",
        "probe_failure_20261020T183606Z",
    )

    assert result["source"] == "TASK_FAIL"
    assert result["attempt_history_complete"] is False
    assert len(result["failures"]) == 2
    assert all(
        row["task_id"] == "harmless_task"
        for row in result["failures"]
    )
    assert all(
        "attempt_number" not in row
        for row in result["failures"]
    )

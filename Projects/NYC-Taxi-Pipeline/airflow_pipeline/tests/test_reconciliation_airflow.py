"""Airflow DB adapter unit tests; run in Docker where apache-airflow is installed."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
pytest.importorskip("airflow")

from scripts import reconciliation_airflow as service
from scripts.run_evidence import load_run_evidence, start_run_evidence


def contract():
    return {"mode": "BACKFILL", "logical_window": {"start": "2023-02-01", "end_exclusive": "2023-03-01"},
            "read_scope": {"input_layer": "SILVER"}, "write_scope": "WINDOW_REPLACE", "rerun_semantics": "IDEMPOTENT"}


class FakeQuery:
    def __init__(self, rows):
        self.rows = rows
    def filter(self, *args):
        return self
    def one_or_none(self):
        return self.rows[0] if self.rows else None
    def all(self):
        return self.rows


class FakeSession:
    def __init__(self, *, state="success", failed=(), dag_ids=("test_dag",), found=True):
        self.state = state
        self.failed = failed
        self.dag_ids = dag_ids
        self.found = found
    def query(self, column):
        if column is service.DagRun:
            return FakeQuery([SimpleNamespace(state=self.state)] if self.found else [])
        if column is service.TaskInstance.task_id:
            return FakeQuery([(task_id,) for task_id in self.failed])
        if column is service.DagRun.dag_id:
            return FakeQuery([(x,) for x in self.dag_ids])
        raise AssertionError(f"Unexpected query: {column!r}")


def set_db(monkeypatch, **kwargs):
    @contextmanager
    def fake_create_session():
        yield FakeSession(**kwargs)
    monkeypatch.setattr(service, "create_session", fake_create_session)


def test_v2_apply_and_repeat_is_safe(tmp_path, monkeypatch):
    set_db(monkeypatch, state="success")
    start_run_evidence("fix-me", contract(), dag_id="test_dag", directory=tmp_path)
    preview = service.inspect_reconciliation("test_dag", "fix-me", directory=tmp_path)
    assert preview["decision"] == "FINALIZE_SUCCESS"
    assert preview["evidence_format"] == "v2"
    assert preview["applied"] is False
    assert load_run_evidence("fix-me", dag_id="test_dag", directory=tmp_path)["status"] == "RUNNING"
    fixed = service.apply_reconciliation("test_dag", "fix-me", directory=tmp_path)
    assert fixed["applied"] is True
    assert fixed["status"] == "SUCCEEDED"
    again = service.apply_reconciliation("test_dag", "fix-me", directory=tmp_path)
    assert again["decision"] == "ALREADY_CONSISTENT"
    assert again["applied"] is False


def test_failure_stage_requires_real_failed_task(tmp_path, monkeypatch):
    set_db(monkeypatch, state="failed", failed=())
    start_run_evidence("no-task", contract(), dag_id="test_dag", directory=tmp_path)
    with pytest.raises(RuntimeError, match="without a failed task"):
        service.apply_reconciliation("test_dag", "no-task", directory=tmp_path)
    assert load_run_evidence("no-task", dag_id="test_dag", directory=tmp_path)["status"] == "RUNNING"

    set_db(monkeypatch, state="failed", failed=("z_task", "a_task"))
    output = service.apply_reconciliation("test_dag", "no-task", directory=tmp_path)
    assert output["applied"] is True
    assert output["failure_stage"] == "a_task"


def test_legacy_explicit_opt_in_and_no_recreation(tmp_path, monkeypatch):
    set_db(monkeypatch, state="success")
    start_run_evidence("legacy", contract(), directory=tmp_path)
    report = service.inspect_reconciliation("test_dag", "legacy", directory=tmp_path)
    assert report["evidence_format"] == "v1"
    with pytest.raises(PermissionError, match="--allow-legacy"):
        service.apply_reconciliation("test_dag", "legacy", directory=tmp_path)
    assert load_run_evidence("legacy", directory=tmp_path)["status"] == "RUNNING"
    fixed = service.apply_reconciliation("test_dag", "legacy", directory=tmp_path, allow_legacy=True)
    assert fixed["status"] == "SUCCEEDED"
    assert load_run_evidence("legacy", directory=tmp_path)["schema_version"] == 1
    missing = service.apply_reconciliation("test_dag", "absent", directory=tmp_path)
    assert missing["decision"] == "MISSING_EVIDENCE"
    assert missing["applied"] is False


def test_legacy_ambiguous_run_id_rejected(tmp_path, monkeypatch):
    set_db(monkeypatch, dag_ids=("test_dag", "other_dag"))
    start_run_evidence("legacy", contract(), directory=tmp_path)
    with pytest.raises(RuntimeError, match="Ambiguous legacy"):
        service.inspect_reconciliation("test_dag", "legacy", directory=tmp_path)


def test_nonterminal_wait_and_terminal_conflict(tmp_path, monkeypatch):
    start_run_evidence("ongoing", contract(), dag_id="test_dag", directory=tmp_path)
    set_db(monkeypatch, state="running")
    result = service.apply_reconciliation("test_dag", "ongoing", directory=tmp_path)
    assert result["decision"] == "WAIT"
    assert not result["applied"]
    set_db(monkeypatch, state="success")
    service.apply_reconciliation("test_dag", "ongoing", directory=tmp_path)
    set_db(monkeypatch, state="failed")
    with pytest.raises(RuntimeError, match="Conflicting terminal states"):
        service.apply_reconciliation("test_dag", "ongoing", directory=tmp_path)


def test_dag_run_not_found(tmp_path, monkeypatch):
    set_db(monkeypatch, found=False)
    with pytest.raises(LookupError):
        service.inspect_reconciliation("test_dag", "missing", directory=tmp_path)

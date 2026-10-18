import subprocess
import sys
import json
from types import SimpleNamespace

import pytest

from scripts import pipeline_runner as runner


def _contract():
    return {
        "mode": "CONFIGURED_FULL_REFRESH",
        "year": 2023,
        "months": [1, 2, 3],
        "window_start": "2023-01-01",
        "window_end_exclusive": "2023-04-01",
        "logical_window": {
            "start": "2023-01-01",
            "end_exclusive": "2023-04-01",
            "anchor": "PICKUP_DATETIME",
        },
        "read_scope": {
            "raw_file_months": [
                "2023-01",
                "2023-02",
                "2023-03",
            ],
            "boundary_policy": "CONFIGURED_MONTHS_ONLY",
        },
        "write_scope": "FULL_TABLE_OVERWRITE",
        "state_mode": "STATELESS",
        "rerun_semantics": "RECOMPUTE_CONFIGURED_WINDOW",
    }


def test_submit_builds_correct_command(monkeypatch):
    recorded = {}

    def fake_run(command, **kwargs):
        recorded["command"] = command
        recorded["kwargs"] = kwargs

    monkeypatch.setattr(
        runner.subprocess,
        "run",
        fake_run,
    )

    runner.submit_stage(
        "gold",
        _contract(),
    )

    assert recorded["command"][:4] == [
        sys.executable,
        "-m",
        "scripts.pipeline_runner",
        "gold",
    ]

    assert recorded["command"][4] == ("--execution-contract-json")

    passed_contract = json.loads(recorded["command"][5])

    assert passed_contract == _contract()

    assert recorded["kwargs"]["check"] is True
    assert recorded["kwargs"]["cwd"] == str(runner.PROJECT_ROOT)


def test_child_failure_propagates(monkeypatch):
    def failing_run(command, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=2,
            cmd=command,
        )

    monkeypatch.setattr(
        runner.subprocess,
        "run",
        failing_run,
    )

    with pytest.raises(subprocess.CalledProcessError):
        runner.submit_stage("gold", _contract())


def test_stage_dispatch(monkeypatch):
    called = []

    fake_pipeline = SimpleNamespace(
        gold_task=lambda **kwargs: called.append(kwargs["execution_contract"]),
    )

    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda name: fake_pipeline,
    )

    runner.execute_stage(
        "gold",
        _contract(),
    )

    assert called == [_contract()]


def test_rejects_unknown_stage():
    with pytest.raises(ValueError):
        runner.submit_stage(
            "february",
            _contract(),
        )


def test_quality_stage_dispatch(monkeypatch):
    called = []

    fake_pipeline = SimpleNamespace(
        quality_task=lambda **kwargs: called.append(kwargs["execution_contract"]),
    )

    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda name: fake_pipeline,
    )

    runner.execute_stage(
        "quality",
        _contract(),
    )

    assert called == [_contract()]


def test_quality_stage_failure_propagates(monkeypatch):
    def failing_quality_task(**kwargs):
        raise ValueError("Gold Quality Gate FAILED: " "rule=POSITIVE_COUNT")

    fake_pipeline = SimpleNamespace(
        quality_task=failing_quality_task,
    )

    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda name: fake_pipeline,
    )

    with pytest.raises(
        ValueError,
        match="Gold Quality Gate FAILED",
    ):
        runner.execute_stage(
            "quality",
            _contract(),
        )


def test_submit_bronze_forwards_raw_references(monkeypatch):
    recorded = {}

    def fake_run(command, **kwargs):
        recorded["command"] = command
        recorded["kwargs"] = kwargs

    monkeypatch.setattr(
        runner.subprocess,
        "run",
        fake_run,
    )

    references = [
        {
            "data_month": "2024-01",
            "bucket": "test-bucket",
            "raw_key": "raw/test.parquet",
            "raw_version_id": "raw-v1",
            "manifest_key": "raw/test.parquet.manifest.json",
        }
    ]

    runner.submit_stage(
        "bronze",
        _contract(),
        raw_references=references,
    )

    command = recorded["command"]

    assert command[6] == "--raw-references-json"
    assert json.loads(command[7]) == references
    assert recorded["kwargs"]["check"] is True


def test_cli_forwards_raw_references(monkeypatch):
    recorded = {}

    references = [
        {
            "data_month": "2024-01",
            "bucket": "test-bucket",
            "raw_key": "raw/test.parquet",
            "raw_version_id": "raw-v1",
        }
    ]

    def fake_execute(stage, execution_contract, raw_references=None):
        recorded["stage"] = stage
        recorded["contract"] = execution_contract
        recorded["references"] = raw_references

    monkeypatch.setattr(runner, "execute_stage", fake_execute)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pipeline_runner",
            "bronze",
            "--execution-contract-json",
            json.dumps(_contract()),
            "--raw-references-json",
            json.dumps(references),
        ],
    )

    runner.main()

    assert recorded["stage"] == "bronze"
    assert recorded["contract"] == _contract()
    assert recorded["references"] == references


def test_execute_bronze_receives_raw_references(monkeypatch):
    recorded = {}

    references = [
        {
            "data_month": "2024-01",
            "raw_version_id": "raw-v1",
        }
    ]

    def fake_bronze_task(**kwargs):
        recorded.update(kwargs)

    fake_pipeline = SimpleNamespace(
        bronze_task=fake_bronze_task,
    )

    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda name: fake_pipeline,
    )

    runner.execute_stage(
        "bronze",
        _contract(),
        raw_references=references,
    )

    assert recorded["execution_contract"] == _contract()
    assert recorded["raw_references"] == references


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

    assert recorded["command"][4] == (
        "--execution-contract-json"
    )

    passed_contract = json.loads(
        recorded["command"][5]
    )

    assert passed_contract == _contract()

    assert recorded["kwargs"]["check"] is True
    assert recorded["kwargs"]["cwd"] == str(
        runner.PROJECT_ROOT
    )


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
        gold_task=lambda **kwargs: called.append(
            kwargs["execution_contract"]
        ),
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
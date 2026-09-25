
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import pipeline_runner as runner


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

    runner.submit_stage("gold")

    assert recorded["command"] == [
        sys.executable,
        "-m",
        "scripts.pipeline_runner",
        "gold",
    ]

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
        runner.submit_stage("gold")


def test_stage_dispatch(monkeypatch):
    called = []

    fake_pipeline = SimpleNamespace(
        gold_task=lambda: called.append("gold"),
    )

    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda name: fake_pipeline,
    )

    runner.execute_stage("gold")

    assert called == ["gold"]


def test_rejects_unknown_stage():
    with pytest.raises(ValueError):
        runner.submit_stage("february")
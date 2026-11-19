import json
import sys

import pytest

from scripts import delta_medallion_pipeline as pipeline
from scripts import gold_publication as publication
from scripts import pipeline_runner as runner
from scripts.gold_release_plan import build_gold_release_spec

FULL = {
    "mode": "CONFIGURED_FULL_REFRESH",
    "write_scope": "FULL_TABLE_OVERWRITE",
    "logical_window": {
        "start": "2024-01-01",
        "end_exclusive": "2024-02-01",
    },
}


def make_spec(tmp_path, run_id="manual__2026-11-15"):
    return build_gold_release_spec(
        dag_id="nyc_taxi_medallion_delta",
        run_id=run_id,
        base_dir=str(tmp_path),
    )


def test_release_spec_is_stable_and_isolated(tmp_path):
    first = make_spec(tmp_path)
    second = make_spec(tmp_path)
    another = make_spec(tmp_path, "another-run")

    assert first == second
    assert first["release_id"] != another["release_id"]
    assert first["candidate_root"] != another["candidate_root"]

    assert first["candidate_root"].endswith(first["release_id"])
    assert first["active_manifest"].endswith("gold/active.json")


def test_release_spec_has_no_filesystem_side_effects(tmp_path):
    spec = make_spec(tmp_path)

    from pathlib import Path

    assert not Path(spec["candidate_root"]).exists()
    assert not Path(spec["active_manifest"]).exists()


def test_gold_task_forwards_candidate_root(tmp_path, monkeypatch):
    spec = make_spec(tmp_path)
    recorded = {}

    def fake_write_gold(**kwargs):
        recorded.update(kwargs)
        return {}

    monkeypatch.setattr(
        pipeline,
        "write_gold",
        fake_write_gold,
    )

    pipeline.gold_task(
        execution_contract=FULL,
        release_spec=spec,
    )

    assert recorded["execution_contract"] == FULL
    assert recorded["candidate_root"] == spec["candidate_root"]


def test_backfill_keeps_legacy_gold_path(monkeypatch):
    recorded = {}

    monkeypatch.setattr(
        pipeline,
        "write_gold",
        lambda **kwargs: recorded.update(kwargs),
    )

    backfill = {
        "mode": "BACKFILL",
        "write_scope": "WINDOW_REPLACE",
    }

    pipeline.gold_task(execution_contract=backfill)

    assert recorded["candidate_root"] is None


def test_publish_calls_existing_activation(tmp_path, monkeypatch):
    spec = make_spec(tmp_path)
    recorded = {}
    spark = object()

    monkeypatch.setattr(
        pipeline,
        "get_spark",
        lambda *_: spark,
    )

    def fake_activate(session, **kwargs):
        recorded["spark"] = session
        recorded.update(kwargs)
        return {"release_id": kwargs["release_id"]}

    monkeypatch.setattr(
        publication,
        "activate_gold_release",
        fake_activate,
    )

    result = pipeline.publish_gold_task(
        execution_contract=FULL,
        release_spec=spec,
    )

    assert recorded["spark"] is spark
    assert recorded["release_id"] == spec["release_id"]
    assert recorded["candidate_root"] == spec["candidate_root"]
    assert recorded["active_manifest"] == spec["active_manifest"]
    assert recorded["execution_contract"] == FULL
    assert result["release_id"] == spec["release_id"]


def test_runner_forwards_release_spec(tmp_path, monkeypatch):
    spec = make_spec(tmp_path)
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
        "publish",
        FULL,
        release_spec=spec,
    )

    command = recorded["command"]

    assert command[:4] == [
        sys.executable,
        "-m",
        "scripts.pipeline_runner",
        "publish",
    ]

    index = command.index("--release-spec-json")
    assert json.loads(command[index + 1]) == spec
    assert recorded["kwargs"]["check"] is True


def test_publish_without_spec_is_rejected():
    with pytest.raises(
        ValueError,
        match="requires release_spec",
    ):
        runner.submit_stage("publish", FULL)


def test_backfill_cannot_use_candidate(tmp_path):
    spec = make_spec(tmp_path)

    with pytest.raises(
        ValueError,
        match="configured full refresh only",
    ):
        pipeline.gold_task(
            execution_contract={
                "mode": "BACKFILL",
                "write_scope": "WINDOW_REPLACE",
            },
            release_spec=spec,
        )


def test_release_spec_rejected_for_bronze(tmp_path):
    spec = make_spec(tmp_path)

    with pytest.raises(
        ValueError,
        match="gold/publish only",
    ):
        runner.submit_stage(
            "bronze",
            FULL,
            release_spec=spec,
        )

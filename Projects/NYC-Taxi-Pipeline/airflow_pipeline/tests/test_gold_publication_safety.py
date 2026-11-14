
import json

import pytest

from scripts import gold_publication as pub
from scripts.gold_quality_gate import GOLD_TABLE_CONTRACTS


CONTRACT = {
    "write_scope": "FULL_TABLE_OVERWRITE",
    "logical_window": {
        "start": "2023-01-01",
        "end_exclusive": "2023-04-01",
    },
}


def make_candidate(tmp_path):
    root = tmp_path / "release-002"

    for name in GOLD_TABLE_CONTRACTS:
        (root / name / "_delta_log").mkdir(parents=True)

    return root


def test_rejected_candidate_preserves_active(tmp_path, monkeypatch):
    root = make_candidate(tmp_path)
    active = tmp_path / "active.json"

    previous = {"release_id": "release-001"}
    active.write_text(json.dumps(previous), encoding="utf-8")

    def reject(**kwargs):
        raise ValueError("Gold Quality Gate FAILED")

    monkeypatch.setattr(pub, "validate_all_gold_tables", reject)

    with pytest.raises(ValueError, match="Quality Gate FAILED"):
        pub.activate_gold_release(
            object(),
            release_id="release-002",
            candidate_root=str(root),
            active_manifest=str(active),
            execution_contract=CONTRACT,
        )

    assert json.loads(active.read_text()) == previous


def test_valid_candidate_activates_manifest(tmp_path, monkeypatch):
    root = make_candidate(tmp_path)
    active = tmp_path / "active.json"

    previous = {"release_id": "release-001"}
    active.write_text(json.dumps(previous), encoding="utf-8")

    calls = []

    def accept(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(pub, "validate_all_gold_tables", accept)

    result = pub.activate_gold_release(
        object(),
        release_id="release-002",
        candidate_root=str(root),
        active_manifest=str(active),
        execution_contract=CONTRACT,
    )

    current = json.loads(active.read_text())

    assert len(calls) == 1
    assert calls[0]["table_versions"] == {
        name: 0 for name in GOLD_TABLE_CONTRACTS
    }
    assert set(calls[0]["table_paths"]) == set(GOLD_TABLE_CONTRACTS)
    assert calls[0]["execution_contract"] == CONTRACT
    assert current == result
    assert current["release_id"] == "release-002"
    assert len(current["table_paths"]) == 6


def test_activation_failure_preserves_active(tmp_path, monkeypatch):
    root = make_candidate(tmp_path)
    active = tmp_path / "active.json"

    previous = {"release_id": "release-001"}
    active.write_text(json.dumps(previous), encoding="utf-8")

    monkeypatch.setattr(
        pub, "validate_all_gold_tables", lambda **kwargs: None
    )

    def fail_replace(*args):
        raise OSError("Simulated activation failure")

    monkeypatch.setattr(pub.os, "replace", fail_replace)

    with pytest.raises(OSError, match="activation failure"):
        pub.activate_gold_release(
            object(),
            release_id="release-002",
            candidate_root=str(root),
            active_manifest=str(active),
            execution_contract=CONTRACT,
        )

    assert json.loads(active.read_text()) == previous
    assert not list(tmp_path.glob(".gold-active-*.tmp"))


@pytest.fixture(autouse=True)
def mock_delta_versions(monkeypatch):
    def fake_versions(spark, table_paths):
        return {
            name: 0
            for name in table_paths
        }

    monkeypatch.setattr(
        pub,
        "_read_delta_versions",
        fake_versions,
    )


def test_candidate_version_change_preserves_active(
    tmp_path, monkeypatch
):
    root = make_candidate(tmp_path)
    active = tmp_path / "active.json"

    previous = {"release_id": "release-001"}
    active.write_text(
        json.dumps(previous),
        encoding="utf-8",
    )

    before = {
        name: 0 for name in GOLD_TABLE_CONTRACTS
    }
    after = before.copy()
    after["trip_summary_hourly"] = 1

    observed_versions = iter([before, after])

    monkeypatch.setattr(
        pub,
        "_read_delta_versions",
        lambda spark, paths: next(observed_versions),
    )
    monkeypatch.setattr(
        pub,
        "validate_all_gold_tables",
        lambda **kwargs: None,
    )

    with pytest.raises(
        RuntimeError,
        match="Gold candidate changed during validation",
    ):
        pub.activate_gold_release(
            object(),
            release_id="release-002",
            candidate_root=str(root),
            active_manifest=str(active),
            execution_contract=CONTRACT,
        )

    assert json.loads(
        active.read_text(encoding="utf-8")
    ) == previous

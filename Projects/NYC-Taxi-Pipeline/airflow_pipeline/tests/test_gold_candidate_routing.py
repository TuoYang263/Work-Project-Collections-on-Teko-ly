from pathlib import Path

import pytest

from scripts import delta_medallion_pipeline as pipeline
from scripts.gold_quality_gate import GOLD_TABLE_CONTRACTS

FULL = {"write_scope": "FULL_TABLE_OVERWRITE"}
BACKFILL = {"write_scope": "WINDOW_REPLACE"}


def test_candidate_resolves_all_six_isolated_paths(tmp_path):
    root = tmp_path / "release-003"

    paths = pipeline.resolve_gold_output_paths(
        FULL,
        candidate_root=str(root),
    )

    assert set(paths) == set(GOLD_TABLE_CONTRACTS)

    assert paths == {name: str(root / name) for name in GOLD_TABLE_CONTRACTS}

    assert not root.exists()

    assert set(paths.values()).isdisjoint(pipeline.DEFAULT_GOLD_PATHS.values())


def test_no_candidate_preserves_existing_paths():
    assert pipeline.resolve_gold_output_paths(FULL) == (pipeline.DEFAULT_GOLD_PATHS)

    assert pipeline.resolve_gold_output_paths(BACKFILL) == (pipeline.DEFAULT_GOLD_PATHS)


def test_candidate_forbids_window_replace(tmp_path):
    with pytest.raises(
        ValueError,
        match="FULL_TABLE_OVERWRITE",
    ):
        pipeline.resolve_gold_output_paths(
            BACKFILL,
            candidate_root=str(tmp_path / "release-003"),
        )


def test_candidate_forbids_empty_path():
    with pytest.raises(
        ValueError,
        match="candidate_root",
    ):
        pipeline.resolve_gold_output_paths(
            FULL,
            candidate_root=" ",
        )


def test_candidate_forbids_legacy_path_overlap():
    legacy_root = Path(pipeline.GOLD_SUMMARY_HOURLY).parent

    with pytest.raises(
        ValueError,
        match="overlaps legacy",
    ):
        pipeline.resolve_gold_output_paths(
            FULL,
            candidate_root=str(legacy_root),
        )


def test_candidate_forbids_reusing_existing_table(tmp_path):
    root = tmp_path / "release-003"

    (root / "trip_summary_hourly").mkdir(parents=True)

    with pytest.raises(
        ValueError,
        match="already exists",
    ):
        pipeline.resolve_gold_output_paths(
            FULL,
            candidate_root=str(root),
        )


def test_candidate_requires_all_six_tables(tmp_path):
    with pytest.raises(
        ValueError,
        match="all six Gold tables",
    ):
        pipeline.write_gold(
            FULL,
            hourly_only=True,
            candidate_root=str(tmp_path / "release-003"),
        )


def test_candidate_rejects_hourly_override(tmp_path):
    with pytest.raises(
        ValueError,
        match="hourly_output_path",
    ):
        pipeline.write_gold(
            FULL,
            hourly_output_path=str(tmp_path / "hourly"),
            candidate_root=str(tmp_path / "release-003"),
        )


def test_candidate_forbids_reusing_existing_root(tmp_path):
    root = tmp_path / "release-003"
    root.mkdir()

    with pytest.raises(
        ValueError,
        match="Candidate root already exists",
    ):
        pipeline.resolve_gold_output_paths(
            FULL,
            candidate_root=str(root),
        )

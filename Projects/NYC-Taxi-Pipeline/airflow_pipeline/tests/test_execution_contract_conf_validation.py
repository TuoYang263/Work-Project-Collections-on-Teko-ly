"""Regression tests for manual DAG-run parameters."""

import pytest

from scripts.execution_contract import build_current_execution_contract


SETTINGS = {"data_config": {"year": 2023, "months": [1, 2, 3]}}


def test_rejects_unknown_dag_run_key():
    with pytest.raises(ValueError, match="Unsupported DAG run configuration keys"):
        build_current_execution_contract(SETTINGS, {"write_scop": "WINDOW_REPLACE"})


def test_rejects_unknown_key_alongside_backfill():
    with pytest.raises(ValueError, match="Unsupported DAG run configuration keys"):
        build_current_execution_contract(
            SETTINGS, {"backfill": {"year": 2023, "month": 2}, "note": "example"}
        )


def test_rejects_null_backfill_instead_of_full_refresh():
    with pytest.raises(ValueError, match="backfill must be an object"):
        build_current_execution_contract(SETTINGS, {"backfill": None})


def test_empty_config_still_uses_full_refresh():
    contract = build_current_execution_contract(SETTINGS, {})
    assert contract["mode"] == "CONFIGURED_FULL_REFRESH"


def test_explicit_backfill_preserves_window():
    contract = build_current_execution_contract(
        SETTINGS, {"backfill": {"year": 2023, "month": 2}}
    )
    assert contract["mode"] == "BACKFILL"
    assert contract["write_scope"] == "WINDOW_REPLACE"
    assert contract["logical_window"]["end_exclusive"] == "2023-03-01"

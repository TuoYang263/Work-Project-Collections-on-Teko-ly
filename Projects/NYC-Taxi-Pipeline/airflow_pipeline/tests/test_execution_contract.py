import pytest

from scripts.execution_contract import (
    build_current_execution_contract,
)


def _settings(months=None):
    return {
        "data_config": {
            "year": 2023,
            "months": months or [1, 2, 3],
        }
    }


def test_current_execution_contract():
    contract = build_current_execution_contract(
        _settings(),
        dag_run_conf={},
    )

    assert contract == {
        "mode": "CONFIGURED_FULL_REFRESH",
        "year": 2023,
        "months": [1, 2, 3],
        "window_start": "2023-01-01",
        "window_end_exclusive": "2023-04-01",
        "write_scope": "FULL_TABLE_OVERWRITE",
        "state_mode": "STATELESS",
        "rerun_semantics": "RECOMPUTE_CONFIGURED_WINDOW",
    }


def test_rejects_unsafe_manual_window_override():
    with pytest.raises(
        ValueError,
        match="Window/backfill overrides are not supported yet",
    ):
        build_current_execution_contract(
            _settings(),
            dag_run_conf={
                "month": 2,
            },
        )


def test_rejects_non_contiguous_current_window():
    with pytest.raises(
        ValueError,
        match="contiguous configured months",
    ):
        build_current_execution_contract(
            _settings(months=[1, 3]),
            dag_run_conf={},
        )
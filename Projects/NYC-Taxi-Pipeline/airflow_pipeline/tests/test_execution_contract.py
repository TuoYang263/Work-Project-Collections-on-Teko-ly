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

        "logical_window": {
            "start": "2023-01-01",
            "end_exclusive": "2023-04-01",
            "anchor": "PICKUP_DATETIME",
        },

        "read_scope": {
            "raw_file_months": [
                "2022-12",
                "2023-01",
                "2023-02",
                "2023-03",
            ],
            "boundary_policy": "INCLUDE_PREVIOUS_MONTH",
        },

        "write_scope": "FULL_TABLE_OVERWRITE",
        "state_mode": "STATELESS",
        "rerun_semantics": "RECOMPUTE_CONFIGURED_WINDOW",
    }


def test_rejects_unsafe_manual_window_override():
    with pytest.raises(
        ValueError,
        match="Direct window overrides are not supported",
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


def test_execution_contract_crosses_year_boundary():
    settings = {
        "data_config": {
            "year": 2023,
            "months": [11, 12],
        }
    }

    contract = build_current_execution_contract(
        settings,
        dag_run_conf={},
    )

    assert contract["logical_window"] == {
        "start": "2023-11-01",
        "end_exclusive": "2024-01-01",
        "anchor": "PICKUP_DATETIME",
    }

    assert contract["read_scope"] == {
        "raw_file_months": [
            "2023-10",
            "2023-11",
            "2023-12",
        ],
        "boundary_policy": "INCLUDE_PREVIOUS_MONTH",
    }

    assert contract["write_scope"] == (
        "FULL_TABLE_OVERWRITE"
    )


def test_contract_read_scope_includes_previous_month():
    contract = build_current_execution_contract(
        _settings(),
        dag_run_conf={},
    )

    assert contract["year"] == 2023
    assert contract["months"] == [1, 2, 3]

    assert contract["read_scope"] == {
        "raw_file_months": [
            "2022-12",
            "2023-01",
            "2023-02",
            "2023-03",
        ],
        "boundary_policy": "INCLUDE_PREVIOUS_MONTH",
    }


def test_builds_gold_only_single_month_backfill():
    contract = build_current_execution_contract(
        _settings(),
        dag_run_conf={
            "backfill": {
                "year": 2023,
                "month": 2,
            },
        },
    )

    assert contract["mode"] == "BACKFILL"
    assert contract["execution_scope"] == (
        "GOLD_ONLY"
    )

    assert contract["logical_window"] == {
        "start": "2023-02-01",
        "end_exclusive": "2023-03-01",
        "anchor": "PICKUP_DATETIME",
    }

    assert contract["read_scope"] == {
        "input_layer": "SILVER",
        "boundary_policy":
            "REUSE_EXISTING_SILVER_STATE",
    }

    assert contract["write_scope"] == (
        "WINDOW_REPLACE"
    )


def test_backfill_january_window_crosses_year_boundary():
    contract = build_current_execution_contract(
        _settings(),
        dag_run_conf={
            "backfill": {
                "year": 2023,
                "month": 1,
            },
        },
    )

    assert contract[
        "window_start"
    ] == "2023-01-01"

    assert contract[
        "window_end_exclusive"
    ] == "2023-02-01"


def test_rejects_invalid_backfill_month():
    with pytest.raises(
        ValueError,
        match="Invalid backfill month",
    ):
        build_current_execution_contract(
            _settings(),
            dag_run_conf={
                "backfill": {
                    "year": 2023,
                    "month": 13,
                },
            },
        )


def test_rejects_backfill_mixed_with_direct_override():
    with pytest.raises(
        ValueError,
        match="Direct window overrides",
    ):
        build_current_execution_contract(
            _settings(),
            dag_run_conf={
                "month": 2,
                "backfill": {
                    "year": 2023,
                    "month": 2,
                },
            },
        )
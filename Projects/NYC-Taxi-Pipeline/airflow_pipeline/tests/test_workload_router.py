import pytest

from scripts.workload_router import (
    GOLD_PRODUCTS,
    plan_workload,
)
from scripts.gold_quality_gate import GOLD_TABLE_CONTRACTS

FULL = {
    "mode": "CONFIGURED_FULL_REFRESH",
    "write_scope": "FULL_TABLE_OVERWRITE",
    "logical_window": {
        "start": "2023-01-01",
        "end_exclusive": "2023-04-01",
        "anchor": "PICKUP_DATETIME",
    },
}


def manifest():
    return {
        "schema_version": 1,
        "release_id": "airflow-test-001",
        "logical_window": FULL["logical_window"],
        "table_paths": {name: f"/gold/releases/test/{name}" for name in GOLD_PRODUCTS},
        "table_versions": {name: 0 for name in GOLD_PRODUCTS},
    }


def read_request():
    return {
        "intent": "READ_PUBLISHED",
        "execution_contract": FULL,
        "active_manifest": manifest(),
        "publication_verified": True,
        "freshness_verified": True,
        "sources_unchanged_verified": True,
    }


def refresh_request():
    return {
        "intent": "FULL_REFRESH",
        "execution_contract": FULL,
        "raw_inputs_ready": True,
        "local_resources_ok": True,
        "spend_approved": True,
    }


def test_product_contract_has_not_drifted():
    assert GOLD_PRODUCTS == set(GOLD_TABLE_CONTRACTS)


def test_reuses_certified_release_without_compute():
    result = plan_workload(**read_request())

    assert result["selected_route"] == "NO_COMPUTE"
    assert result["release_id"] == "airflow-test-001"
    assert result["auto_launch"] is False


@pytest.mark.parametrize(
    "field,reason",
    [
        ("publication_verified", "PUBLICATION_UNVERIFIED"),
        ("freshness_verified", "FRESHNESS_UNVERIFIED"),
        (
            "sources_unchanged_verified",
            "SOURCE_STATE_UNVERIFIED",
        ),
    ],
)
def test_read_blocks_unverified_evidence(field, reason):
    request = read_request()
    request[field] = False

    result = plan_workload(**request)

    assert result["selected_route"] == "BLOCKED_REVIEW"
    assert result["reason_code"] == reason


def test_read_blocks_missing_manifest():
    request = read_request()
    request["active_manifest"] = None

    result = plan_workload(**request)

    assert result["selected_route"] == "BLOCKED_REVIEW"


def test_read_blocks_wrong_window():
    request = read_request()
    request["active_manifest"]["logical_window"] = {
        "start": "2024-01-01",
        "end_exclusive": "2024-02-01",
    }

    result = plan_workload(**request)

    assert result["reason_code"] == "NO_MATCHING_RELEASE"


def test_read_blocks_missing_gold_table():
    request = read_request()
    request["active_manifest"]["table_paths"].pop("od_corridor_daily")

    result = plan_workload(**request)

    assert result["selected_route"] == "BLOCKED_REVIEW"


def test_read_blocks_invalid_delta_version():
    request = read_request()
    request["active_manifest"]["table_versions"]["trip_summary_hourly"] = -1

    result = plan_workload(**request)

    assert result["selected_route"] == "BLOCKED_REVIEW"


def test_approved_refresh_uses_local_spark():
    result = plan_workload(**refresh_request())

    assert result["selected_route"] == "LOCAL_SPARK"
    assert result["auto_launch"] is False


@pytest.mark.parametrize(
    "field,reason",
    [
        ("raw_inputs_ready", "RAW_INPUTS_NOT_READY"),
        ("local_resources_ok", "LOCAL_CAPACITY_UNVERIFIED"),
        ("spend_approved", "SPEND_NOT_APPROVED"),
    ],
)
def test_refresh_blocks_missing_preflight(field, reason):
    request = refresh_request()
    request[field] = False

    result = plan_workload(**request)

    assert result["selected_route"] == "BLOCKED_REVIEW"
    assert result["reason_code"] == reason


def test_backfill_is_not_auto_routed():
    request = refresh_request()
    request["execution_contract"] = {
        "mode": "BACKFILL",
        "write_scope": "WINDOW_REPLACE",
    }

    result = plan_workload(**request)

    assert result["reason_code"] == "UNSUPPORTED_WRITE_SCOPE"


def test_unknown_intent_fails_closed():
    result = plan_workload(
        intent="DATABRICKS_JOB",
        execution_contract=FULL,
    )

    assert result["selected_route"] == "BLOCKED_REVIEW"
    assert result["auto_launch"] is False

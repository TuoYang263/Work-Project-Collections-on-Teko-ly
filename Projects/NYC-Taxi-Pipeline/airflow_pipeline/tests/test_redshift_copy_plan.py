"""Offline R4 planner tests; no Spark or AWS credentials required."""

import hashlib
import json

import pytest

from scripts.redshift_copy_plan import build_copy_plan
from scripts.serving_snapshot_integrity import TABLES

BUCKET = "nyc-test-bucket"
RELEASE = "airflow-a521894129ca813bd8977d28"


@pytest.fixture
def manifest():

    prefix = f"s3://{BUCKET}/serving/releases/{RELEASE}"

    tables = {}

    for name in TABLES:

        stage = f"{name}_r14d4f7f09d1f"

        tables[name] = {
            "row_count": 3,
            "columns": ["day", "amount"],
            "staging_table": stage,
            "create_table_sql": (
                f'CREATE TABLE nyc_stage."{stage}" (\n'
                '  "day" DATE,\n'
                '  "amount" DECIMAL(12,2)\n'
                ");"
            ),
            "copy_manifest": (f"{prefix}/copy_manifests/{name}.json"),
            "files": [
                {
                    "url": (f"{prefix}/{name}/" "part-00000-abcd.parquet"),
                    "size_bytes": 42,
                    "sha256": "a" * 64,
                }
            ],
        }

    return {
        "schema_version": 2,
        "gold_release_id": RELEASE,
        "r1_inventory_sha256": "b" * 64,
        "logical_window": {
            "start": "2023-01-01",
            "end_exclusive": "2023-04-01",
            "anchor": "PICKUP_DATETIME",
        },
        "gold_delta_versions": {name: 0 for name in TABLES},
        "tables": tables,
    }


def build(manifest, approved=None):

    raw = json.dumps(
        manifest,
        sort_keys=True,
    ).encode()

    return build_copy_plan(
        raw,
        approved_sha256=(approved or hashlib.sha256(raw).hexdigest()),
        bucket=BUCKET,
        release_id=RELEASE,
    )


def test_deterministic_six_table_plan(manifest):

    plan = build(manifest)

    assert plan == build(manifest)
    assert plan["auto_execute"] is False

    assert [t["table"] for t in plan["tables"]] == list(TABLES)

    for item in plan["tables"]:

        assert item["expected_rows"] == 3

        assert item["create_table_sql"] == (
            manifest["tables"][item["table"]]["create_table_sql"]
        )

        assert item["copy_sql"].endswith(
            "IAM_ROLE default\n" "FORMAT AS PARQUET\n" "MANIFEST;"
        )

        assert "REGION" not in item["copy_sql"]
        assert "DROP" not in item["create_table_sql"]
        assert "IF NOT EXISTS" not in item["create_table_sql"]

        assert (
            item["expected_copy_manifest"]["entries"][0]["meta"]["content_length"] == 42
        )

        assert item["count_sql"] == (
            "SELECT COUNT(*) AS row_count " f'FROM nyc_stage."{item["staging_table"]}";'
        )


def test_manifest_mutation_rejected(manifest):

    approved = build(manifest)["manifest_sha256"]

    manifest["tables"][TABLES[0]]["row_count"] += 1

    with pytest.raises(
        ValueError,
        match="SHA256 mismatch",
    ):
        build(manifest, approved)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("row_count", True, "row count"),
        ("row_count", 0, "row count"),
        ("columns", ["amount", "day"], "column order"),
        ("columns", ["day", "day"], "Duplicate columns"),
        ("columns", ["Day", "amount"], "Unsafe column"),
        ("staging_table", "active_table", "Staging table"),
        (
            "copy_manifest",
            "s3://other-bucket/file.json",
            "COPY manifest path",
        ),
        ("files", [], "Missing Parquet"),
    ],
)
def test_invalid_table_contract(
    manifest,
    field,
    value,
    error,
):

    manifest["tables"][TABLES[0]][field] = value

    with pytest.raises(ValueError, match=error):
        build(manifest)


@pytest.mark.parametrize(
    "replacement",
    [
        "DECIMAL(39,2)",
        "DECIMAL(2,3)",
        "BIGINT DEFAULT 1",
        "BIGINT); DROP TABLE nyc_stage.x; --",
    ],
)
def test_ddl_rejects_noncanonical_types_and_sql(
    manifest,
    replacement,
):

    info = manifest["tables"][TABLES[0]]

    info["create_table_sql"] = info["create_table_sql"].replace(
        "DECIMAL(12,2)",
        replacement,
    )

    with pytest.raises(
        ValueError,
        match="Unsupported Redshift type",
    ):
        build(manifest)


def test_missing_table(manifest):

    del manifest["tables"][TABLES[0]]

    with pytest.raises(
        ValueError,
        match="Table inventory",
    ):
        build(manifest)


def test_wrong_release(manifest):

    manifest["gold_release_id"] = "different-release"

    with pytest.raises(
        ValueError,
        match="Release ID mismatch",
    ):
        build(manifest)


def test_duplicate_file(manifest):

    files = manifest["tables"][TABLES[0]]["files"]

    files.append(files[0].copy())

    with pytest.raises(
        ValueError,
        match="Duplicate Parquet",
    ):
        build(manifest)


def test_wrong_file_prefix(manifest):

    manifest["tables"][TABLES[0]]["files"][0][
        "url"
    ] = "s3://other-bucket/part-0.parquet"

    with pytest.raises(
        ValueError,
        match="Parquet path",
    ):
        build(manifest)


def test_invalid_delta_version(manifest):

    manifest["gold_delta_versions"][TABLES[0]] = -1

    with pytest.raises(
        ValueError,
        match="Invalid Delta version",
    ):
        build(manifest)


def test_invalid_window(manifest):

    manifest["logical_window"]["end_exclusive"] = "2022-01-01"

    with pytest.raises(
        ValueError,
        match="logical window order",
    ):
        build(manifest)


def test_two_file_manifest_keeps_both_entries(manifest):

    info = manifest["tables"]["od_corridor_daily"]

    second = info["files"][0].copy()

    second["url"] = second["url"].replace(
        "00000",
        "00001",
    )

    second["size_bytes"] = 43

    info["files"].append(second)

    item = build(manifest)["tables"][-1]

    assert item["files"] == info["files"]

    assert [
        entry["meta"]["content_length"]
        for entry in item["expected_copy_manifest"]["entries"]
    ] == [42, 43]

    assert all(
        entry["mandatory"] is True
        for entry in item["expected_copy_manifest"]["entries"]
    )


def test_duplicate_json_key_rejected(manifest):

    raw = json.dumps(manifest).encode()

    raw = raw[:-1] + b', "schema_version": 2}'

    with pytest.raises(
        ValueError,
        match="Duplicate JSON key",
    ):
        build_copy_plan(
            raw,
            approved_sha256=(hashlib.sha256(raw).hexdigest()),
            bucket=BUCKET,
            release_id=RELEASE,
        )


def test_wrong_manifest_version(manifest):

    manifest["schema_version"] = 1

    with pytest.raises(
        ValueError,
        match="Unsupported manifest version",
    ):
        build(manifest)

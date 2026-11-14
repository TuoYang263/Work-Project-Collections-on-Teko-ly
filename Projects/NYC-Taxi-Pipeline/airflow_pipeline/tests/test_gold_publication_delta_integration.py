import json

import pytest
from delta.tables import DeltaTable

from scripts import gold_publication as pub
from scripts import delta_medallion_pipeline as pipeline
from scripts.gold_publication import activate_gold_release
from scripts.gold_quality_gate import (
    GOLD_TABLE_CONTRACTS,
    validate_all_gold_tables,
)


CONTRACT = {
    "write_scope": "FULL_TABLE_OVERWRITE",
    "logical_window": {
        "start": "2023-01-01",
        "end_exclusive": "2023-04-01",
    },
}


QUERIES = {
    "trip_summary_hourly": """
        SELECT
            TIMESTAMP '2023-02-10 10:00:00' AS pickup_hour,
            100L AS trip_count
    """,
    "zone_summary_pickup_daily": """
        SELECT
            DATE '2023-02-10' AS pickup_day,
            10L AS zone_id,
            100L AS trip_count
    """,
    "zone_summary_dropoff_daily": """
        SELECT
            DATE '2023-02-10' AS dropoff_day,
            20L AS zone_id,
            100L AS trip_count
    """,
    "flow_imbalance_daily": """
        SELECT
            DATE '2023-02-10' AS service_day,
            10L AS zone_id,
            100L AS total_flow,
            0.25D AS flow_imbalance
    """,
    "airport_economics_daily": """
        SELECT
            DATE '2023-02-10' AS service_day,
            'JFK' AS airport_code,
            'ARRIVAL' AS direction,
            100L AS trip_count
    """,
    "od_corridor_daily": """
        SELECT
            DATE '2023-02-10' AS service_day,
            10L AS pickup_zone_id,
            20L AS dropoff_zone_id,
            100L AS trip_count
    """,
}


@pytest.fixture(scope="module")
def spark():
    session = pipeline.get_spark(
        "GoldPublicationDeltaIntegration"
    )
    session.conf.set("spark.sql.session.timeZone", "UTC")
    yield session
    session.stop()


def test_real_delta_release_pins_versions(spark, tmp_path):
    candidate = tmp_path / "release-002"
    active = tmp_path / "active.json"

    assert set(QUERIES) == set(GOLD_TABLE_CONTRACTS)

    for name, sql in QUERIES.items():
        path = candidate / name
        (
            spark.sql(sql)
            .write.format("delta")
            .mode("overwrite")
            .save(str(path))
        )

    manifest = activate_gold_release(
        spark,
        release_id="release-002",
        candidate_root=str(candidate),
        active_manifest=str(active),
        execution_contract=CONTRACT,
    )

    assert json.loads(
        active.read_text(encoding="utf-8")
    ) == manifest

    # A certified release must pin all six Delta versions.
    versions = manifest.get("table_versions")

    assert isinstance(versions, dict), (
        "Active Manifest does not pin Delta versions"
    )
    assert set(versions) == set(GOLD_TABLE_CONTRACTS)

    for name in GOLD_TABLE_CONTRACTS:
        path = manifest["table_paths"][name]

        latest = (
            DeltaTable.forPath(spark, path)
            .history(1)
            .select("version")
            .first()["version"]
        )

        assert versions[name] == latest

    # Simulate an unvalidated write after release activation.
    target_name = "trip_summary_hourly"
    target_path = manifest["table_paths"][target_name]

    (
        spark.sql("""
            SELECT
                TIMESTAMP '2023-02-10 10:00:00' AS pickup_hour,
                0L AS trip_count
        """)
        .write.format("delta")
        .mode("overwrite")
        .save(target_path)
    )

    latest_version = int(
        DeltaTable.forPath(spark, target_path)
        .history(1)
        .select("version")
        .first()["version"]
    )

    assert latest_version > versions[target_name]

    # Latest-version validation must reject the bad data.
    with pytest.raises(ValueError, match="rule=POSITIVE_COUNT"):
        validate_all_gold_tables(
            spark=spark,
            table_paths=manifest["table_paths"],
            execution_contract=CONTRACT,
        )

    # Pinned validation must accept the certified versions.
    validate_all_gold_tables(
        spark=spark,
        table_paths=manifest["table_paths"],
        execution_contract=CONTRACT,
        table_versions=versions,
    )


def test_manifest_consumer_reads_pinned_delta_version(
    spark, tmp_path
):
    table_name = "trip_summary_hourly"
    table_path = tmp_path / "trip_summary_hourly"
    active_path = tmp_path / "active.json"

    # 1. Create the original Delta version.
    (
        spark.sql("SELECT 100L AS trip_count")
        .write.format("delta")
        .mode("overwrite")
        .save(str(table_path))
    )

    # 2. Record its actual Delta version.
    published_version = int(
        DeltaTable.forPath(spark, str(table_path))
        .history(1)
        .select("version")
        .first()["version"]
    )

    # 3. Simulate an already activated release.
    manifest = {
        "schema_version": 1,
        "release_id": "release-002",
        "table_paths": {
            table_name: str(table_path),
        },
        "table_versions": {
            table_name: published_version,
        },
    }

    active_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    # 4. Simulate a later, unvalidated write.
    (
        spark.sql("SELECT 999L AS trip_count")
        .write.format("delta")
        .mode("overwrite")
        .save(str(table_path))
    )

    latest_version = int(
        DeltaTable.forPath(spark, str(table_path))
        .history(1)
        .select("version")
        .first()["version"]
    )

    assert latest_version > published_version

    # 5. A normal Delta read sees the latest data.
    latest_df = (
        spark.read.format("delta")
        .load(str(table_path))
    )

    assert latest_df.first()["trip_count"] == 999

    # 6. Our consumer must read the published version.
    published_df = pub.read_active_gold_table(
        spark=spark,
        active_manifest=str(active_path),
        table_name=table_name,
    )

    assert published_df.first()["trip_count"] == 100

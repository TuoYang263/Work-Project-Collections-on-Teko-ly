import pytest

from pyspark.sql.types import (
    ArrayType,
    DateType,
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampNTZType,
    TimestampType,
)

from scripts.redshift_serving_export import (
    redshift_type,
    staging_ddl,
    staging_table_name,
    validate_manifest,
)
from scripts.gold_quality_gate import GOLD_TABLE_CONTRACTS


def test_ntz_maps_to_redshift_timestamp():
    assert redshift_type(TimestampNTZType()) == "TIMESTAMP"


def test_timezone_adjusted_timestamp_is_rejected():
    with pytest.raises(ValueError, match="not allowed"):
        redshift_type(TimestampType())


def test_numeric_and_date_mapping():
    assert redshift_type(LongType()) == "BIGINT"
    assert redshift_type(DoubleType()) == "DOUBLE PRECISION"
    assert redshift_type(DateType()) == "DATE"


def test_unsupported_complex_type_fails_closed():
    with pytest.raises(ValueError, match="Unsupported"):
        redshift_type(ArrayType(StringType()))


def test_ddl_preserves_column_order():
    schema = StructType(
        [
            StructField("pickup_hour", TimestampNTZType()),
            StructField("trip_count", LongType()),
            StructField("avg_fare", DoubleType()),
        ]
    )

    ddl = staging_ddl(
        "trip_summary_hourly",
        "airflow-release-001",
        schema,
    )

    assert ddl.index('"pickup_hour"') < ddl.index('"trip_count"')
    assert ddl.index('"trip_count"') < ddl.index('"avg_fare"')
    assert '"pickup_hour" TIMESTAMP' in ddl
    assert '"trip_count" BIGINT' in ddl


def test_staging_name_is_stable_and_release_specific():
    first = staging_table_name("trip_summary_hourly", "r1")
    again = staging_table_name("trip_summary_hourly", "r1")
    second = staging_table_name("trip_summary_hourly", "r2")

    assert first == again
    assert first != second


def test_invalid_manifest_rejected():
    with pytest.raises(ValueError, match="Unsupported manifest version"):
        validate_manifest({})


def test_manifest_requires_all_six_tables():
    manifest = {
        "schema_version": 1,
        "release_id": "release-001",
        "table_paths": {
            "trip_summary_hourly": "/tmp/gold/hourly",
        },
        "table_versions": {
            "trip_summary_hourly": 0,
        },
    }

    with pytest.raises(ValueError, match="mapping mismatch"):
        validate_manifest(manifest)


def test_rejects_invalid_delta_version():
    names = GOLD_TABLE_CONTRACTS

    manifest = {
        "schema_version": 1,
        "release_id": "release-001",
        "table_paths": {name: f"/tmp/gold/{name}" for name in names},
        "table_versions": {name: 0 for name in names},
    }

    manifest["table_versions"]["trip_summary_hourly"] = -1

    with pytest.raises(ValueError, match="Invalid pinned"):
        validate_manifest(manifest)

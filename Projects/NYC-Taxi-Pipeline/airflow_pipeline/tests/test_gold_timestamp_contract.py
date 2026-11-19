from types import SimpleNamespace

import pytest

from pyspark.sql.types import (
    StructField,
    StructType,
    TimestampNTZType,
    TimestampType,
)

from scripts.delta_medallion_pipeline import (
    require_nyc_local_timestamps,
)


def make_df(pickup_type, dropoff_type):
    schema = StructType(
        [
            StructField("tpep_pickup_datetime", pickup_type),
            StructField("tpep_dropoff_datetime", dropoff_type),
        ]
    )

    return SimpleNamespace(
        columns=schema.names,
        schema=schema,
    )


def test_accepts_both_ntz_columns():
    df = make_df(
        TimestampNTZType(),
        TimestampNTZType(),
    )

    require_nyc_local_timestamps(df, layer="Silver")


def test_rejects_legacy_timestamp():
    df = make_df(
        TimestampType(),
        TimestampNTZType(),
    )

    with pytest.raises(ValueError, match="requires TIMESTAMP_NTZ"):
        require_nyc_local_timestamps(df, layer="Silver")


def test_rejects_missing_column():
    schema = StructType(
        [
            StructField(
                "tpep_pickup_datetime",
                TimestampNTZType(),
            ),
        ]
    )

    df = SimpleNamespace(
        columns=schema.names,
        schema=schema,
    )

    with pytest.raises(ValueError, match="missing required"):
        require_nyc_local_timestamps(df, layer="Silver")

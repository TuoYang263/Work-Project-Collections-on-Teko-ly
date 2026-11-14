import pytest
from pyspark.sql import SparkSession

from scripts.transform_trip_data_spark import (
    cast_column_types,
    filter_invalid_rows,
)

@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder
        .master("local[2]")
        .appName("SilverTimestampValidityTest")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    yield session
    session.stop()


def test_silver_filters_invalid_trip_timestamps(spark):
    rows = [
        ("valid", "2023-01-01 10:00:00",
         "2023-01-01 10:15:00"),
        ("missing_pickup", None,
         "2023-01-01 10:15:00"),
        ("missing_dropoff", "2023-01-01 10:00:00",
         None),
        ("reversed", "2023-01-01 10:30:00",
         "2023-01-01 10:15:00"),
        ("equal", "2023-01-01 10:00:00",
         "2023-01-01 10:00:00"),
        ("cross_month", "2022-12-31 23:55:00",
         "2023-01-01 00:05:00"),
    ]

    df = spark.createDataFrame(
        rows,
        [
            "case_id",
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
        ],
    )

    df = cast_column_types(df)
    cleaned = filter_invalid_rows(df)

    actual = {
        row["case_id"]
        for row in cleaned.select("case_id").collect()
    }

    assert actual == {"valid", "cross_month"}


def test_silver_preserves_ntz_business_timestamps(spark):
    from pyspark.sql import functions as F
    from pyspark.sql.types import TimestampNTZType

    # Simulate the corrected Bronze timestamp schema.
    bronze = spark.sql("""
        SELECT
            CAST(
                '2023-02-16 00:26:51' AS TIMESTAMP_NTZ
            ) AS tpep_pickup_datetime,
            CAST(
                '2023-02-16 00:45:48' AS TIMESTAMP_NTZ
            ) AS tpep_dropoff_datetime
    """)

    silver = cast_column_types(bronze)

    for name in (
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
    ):
        assert isinstance(
            silver.schema[name].dataType,
            TimestampNTZType,
        ), f"Silver must preserve TIMESTAMP_NTZ: {name}"

    original_tz = spark.conf.get(
        "spark.sql.session.timeZone"
    )

    try:
        results = {}

        for timezone in ("UTC", "America/New_York"):
            spark.conf.set(
                "spark.sql.session.timeZone",
                timezone,
            )

            row = silver.select(
                F.date_format(
                    "tpep_pickup_datetime",
                    "yyyy-MM-dd HH:mm:ss",
                ).alias("pickup"),
                F.date_format(
                    "tpep_dropoff_datetime",
                    "yyyy-MM-dd HH:mm:ss",
                ).alias("dropoff"),
            ).first()

            results[timezone] = (
                row["pickup"],
                row["dropoff"],
            )

        expected = (
            "2023-02-16 00:26:51",
            "2023-02-16 00:45:48",
        )

        assert results["UTC"] == expected
        assert results["America/New_York"] == expected

    finally:
        spark.conf.set(
            "spark.sql.session.timeZone",
            original_tz,
        )

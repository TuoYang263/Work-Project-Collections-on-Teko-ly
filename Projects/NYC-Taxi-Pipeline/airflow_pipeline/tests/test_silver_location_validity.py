
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
        .appName("SilverLocationValidityTest")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.ansi.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    yield session
    session.stop()


def test_silver_rejects_invalid_location_ids(spark):
    cases = [
        ("valid", "1", "2"),
        ("missing_pickup", None, "2"),
        ("missing_dropoff", "1", None),
        ("zero_pickup", "0", "2"),
        ("negative_dropoff", "1", "-1"),
        ("invalid_cast", "not_a_zone", "2"),
    ]

    rows = [
        (
            case_id,
            pickup_zone,
            dropoff_zone,
            "2023-01-01 10:00:00",
            "2023-01-01 10:15:00",
            1.0,
            1,
            12.0,
            14.0,
        )
        for case_id, pickup_zone, dropoff_zone in cases
    ]

    df = spark.createDataFrame(
        rows,
        [
            "case_id",
            "pulocationid",
            "dolocationid",
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "trip_distance",
            "passenger_count",
            "fare_amount",
            "total_amount",
        ],
    )

    df = cast_column_types(df)
    cleaned = filter_invalid_rows(df)

    actual = {
        row["case_id"]
        for row in cleaned.select("case_id").collect()
    }

    assert actual == {"valid"}
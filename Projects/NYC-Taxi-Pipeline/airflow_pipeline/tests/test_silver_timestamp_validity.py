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
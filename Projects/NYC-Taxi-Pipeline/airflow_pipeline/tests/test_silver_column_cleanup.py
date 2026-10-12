import pytest

from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    StringType,
)

from scripts.transform_trip_data_spark import (
    drop_useless_columns,
)


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.master("local[2]")
        .appName("SilverColumnCleanupTest")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )

    yield session
    session.stop()


def test_silver_column_cleanup(spark):
    schema = StructType(
        [
            StructField("trip_id", IntegerType()),
            StructField("all_null", StringType()),
            StructField("all_empty", StringType()),
            StructField("mixed_values", StringType()),
            StructField("useful", StringType()),
        ]
    )

    rows = [
        (1, None, "", None, "A"),
        (2, None, "", "", "B"),
        (3, None, "", None, "C"),
    ]

    df = spark.createDataFrame(
        rows,
        schema=schema,
    )

    cleaned = drop_useless_columns(df)

    assert "all_null" not in cleaned.columns
    assert "all_empty" not in cleaned.columns

    assert "mixed_values" in cleaned.columns
    assert "useful" in cleaned.columns
    assert "trip_id" in cleaned.columns

    actual_rows = cleaned.orderBy("trip_id").collect()

    assert [row["mixed_values"] for row in actual_rows] == [None, "", None]

    assert [row["useful"] for row in actual_rows] == ["A", "B", "C"]


def test_cleanup_uses_single_aggregation(
    spark,
    monkeypatch,
):
    schema = StructType(
        [
            StructField("trip_id", IntegerType()),
            StructField("all_null", StringType()),
            StructField("useful", StringType()),
        ]
    )

    df = spark.createDataFrame(
        [
            (1, None, "A"),
            (2, None, "B"),
        ],
        schema=schema,
    )

    original_agg = DataFrame.agg
    agg_calls = 0

    def tracked_agg(self, *expressions):
        nonlocal agg_calls
        agg_calls += 1
        return original_agg(self, *expressions)

    def forbidden_operation(*args, **kwargs):
        raise AssertionError("Per-column count/distinct is forbidden")

    monkeypatch.setattr(DataFrame, "agg", tracked_agg)
    monkeypatch.setattr(DataFrame, "count", forbidden_operation)
    monkeypatch.setattr(DataFrame, "distinct", forbidden_operation)

    cleaned = drop_useless_columns(df)

    assert cleaned.columns == [
        "trip_id",
        "useful",
    ]
    assert agg_calls == 1

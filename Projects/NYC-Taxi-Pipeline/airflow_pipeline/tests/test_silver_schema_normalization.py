
import pytest
from pyspark.sql import SparkSession

from scripts.transform_trip_data_spark import (
    clean_column_names,
)


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder
        .master("local[2]")
        .appName("SilverSchemaNormalizationTest")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


def test_normal_column_names(spark):
    df = spark.createDataFrame(
        [(1, 2)],
        ["VendorID", "PULocationID"],
    )

    cleaned = clean_column_names(df)

    assert cleaned.columns == [
        "vendorid",
        "pulocationid",
    ]


@pytest.mark.parametrize(
    "first, second",
    [
        ("VendorID", "vendorid"),
        ("PULocationID", " PULocationID "),
    ],
)
def test_normalized_name_collisions_rejected(
    spark,
    first,
    second,
):
    df = spark.createDataFrame(
        [(1, 2)],
        [first, second],
    )

    with pytest.raises(
        ValueError,
        match="(?i)(collision|duplicate)",
    ):
        clean_column_names(df)
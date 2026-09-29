import pytest

from pyspark.sql import functions as F

from scripts import delta_medallion_pipeline as pipeline

from scripts.delta_medallion_pipeline import (
    build_zone_flow_imbalance,
    filter_timestamp_window,
)


WINDOW_START = "2023-01-01"
WINDOW_END_EXCLUSIVE = "2023-04-01"


@pytest.fixture(scope="module")
def spark():
    # Initialize Spark through the production Delta-aware
    # factory so later Delta tests share a compatible JVM.
    session = pipeline.get_spark(
        "NYC Gold Window Semantics Test"
    )

    session.conf.set(
        "spark.sql.session.timeZone",
        "UTC",
    )

    yield session

    session.stop()


def test_filter_timestamp_window_is_half_open(spark):
    raw = spark.createDataFrame(
        [
            ("before", "2022-12-31 23:59:59"),
            ("start", "2023-01-01 00:00:00"),
            ("inside", "2023-03-31 23:59:59"),
            ("end", "2023-04-01 00:00:00"),
        ],
        ["trip_id", "event_time_raw"],
    )

    df = raw.withColumn(
        "event_time",
        F.to_timestamp("event_time_raw"),
    )

    filtered = filter_timestamp_window(
        df,
        timestamp_column="event_time",
        window_start=WINDOW_START,
        window_end_exclusive=WINDOW_END_EXCLUSIVE,
    )

    actual_ids = {
        row["trip_id"]
        for row in filtered.select("trip_id").collect()
    }

    assert actual_ids == {
        "start",
        "inside",
    }


def test_pickup_and_dropoff_use_different_business_time(spark):
    raw = spark.createDataFrame(
        [
            (
                "cross_into_window",
                "2022-12-31 23:58:00",
                "2023-01-01 00:12:00",
            ),
            (
                "cross_out_of_window",
                "2023-03-31 23:58:00",
                "2023-04-01 00:12:00",
            ),
        ],
        [
            "trip_id",
            "pickup_raw",
            "dropoff_raw",
        ],
    )

    df = (
        raw
        .withColumn(
            "tpep_pickup_datetime",
            F.to_timestamp("pickup_raw"),
        )
        .withColumn(
            "tpep_dropoff_datetime",
            F.to_timestamp("dropoff_raw"),
        )
    )

    pickup_window = filter_timestamp_window(
        df,
        timestamp_column="tpep_pickup_datetime",
        window_start=WINDOW_START,
        window_end_exclusive=WINDOW_END_EXCLUSIVE,
    )

    dropoff_window = filter_timestamp_window(
        df,
        timestamp_column="tpep_dropoff_datetime",
        window_start=WINDOW_START,
        window_end_exclusive=WINDOW_END_EXCLUSIVE,
    )

    pickup_ids = {
        row["trip_id"]
        for row in pickup_window.select("trip_id").collect()
    }

    dropoff_ids = {
        row["trip_id"]
        for row in dropoff_window.select("trip_id").collect()
    }

    assert pickup_ids == {
        "cross_out_of_window",
    }

    assert dropoff_ids == {
        "cross_into_window",
    }


def test_flow_imbalance_uses_pickup_and_dropoff_windows(
    spark,
):
    raw = spark.createDataFrame(
        [
            (
                "cross_into_window",
                "2022-12-31 23:58:00",
                "2023-01-01 00:12:00",
                10,
                20,
            ),
            (
                "inside",
                "2023-02-10 10:00:00",
                "2023-02-10 10:20:00",
                50,
                60,
            ),
            (
                "cross_out_of_window",
                "2023-03-31 23:58:00",
                "2023-04-01 00:12:00",
                30,
                40,
            ),
        ],
        [
            "trip_id",
            "pickup_raw",
            "dropoff_raw",
            "pulocationid",
            "dolocationid",
        ],
    )

    df = (
        raw
        .withColumn(
            "tpep_pickup_datetime",
            F.to_timestamp("pickup_raw"),
        )
        .withColumn(
            "tpep_dropoff_datetime",
            F.to_timestamp("dropoff_raw"),
        )
    )

    pickup_window = filter_timestamp_window(
        df,
        timestamp_column="tpep_pickup_datetime",
        window_start=WINDOW_START,
        window_end_exclusive=WINDOW_END_EXCLUSIVE,
    )

    dropoff_window = filter_timestamp_window(
        df,
        timestamp_column="tpep_dropoff_datetime",
        window_start=WINDOW_START,
        window_end_exclusive=WINDOW_END_EXCLUSIVE,
    )

    flow = build_zone_flow_imbalance(
        pickup_window,
        dropoff_window,
    )

    actual = {
        (
            str(row["service_day"]),
            row["zone_id"],
        ): (
            row["pickup_count"],
            row["dropoff_count"],
        )
        for row in flow.collect()
    }

    assert actual == {
        ("2023-01-01", 20): (0, 1),
        ("2023-02-10", 50): (1, 0),
        ("2023-02-10", 60): (0, 1),
        ("2023-03-31", 30): (1, 0),
    }


def test_airport_economics_uses_direction_specific_windows(
    spark,
    tmp_path,
    monkeypatch,
):
    lookup_path = tmp_path / "taxi_zone_lookup.csv"

    lookup_path.write_text(
        "LocationID,Borough,Zone,service_zone\n"
        "132,Queens,JFK Airport,Airports\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        pipeline,
        "TAXI_ZONE_LOOKUP",
        str(lookup_path),
    )

    raw = spark.createDataFrame(
        [
            (
                "cross_into_window",
                "2022-12-31 23:50:00",
                "2023-01-01 00:10:00",
                10,
                132,
                25.0,
                30.0,
                8.0,
            ),
            (
                "cross_out_of_window",
                "2023-03-31 23:50:00",
                "2023-04-01 00:10:00",
                132,
                20,
                35.0,
                40.0,
                10.0,
            ),
        ],
        [
            "trip_id",
            "pickup_raw",
            "dropoff_raw",
            "pulocationid",
            "dolocationid",
            "fare_amount",
            "total_amount",
            "trip_distance",
        ],
    )

    df = (
        raw
        .withColumn(
            "tpep_pickup_datetime",
            F.to_timestamp("pickup_raw"),
        )
        .withColumn(
            "tpep_dropoff_datetime",
            F.to_timestamp("dropoff_raw"),
        )
    )

    pickup_window = filter_timestamp_window(
        df,
        timestamp_column="tpep_pickup_datetime",
        window_start=WINDOW_START,
        window_end_exclusive=WINDOW_END_EXCLUSIVE,
    )

    dropoff_window = filter_timestamp_window(
        df,
        timestamp_column="tpep_dropoff_datetime",
        window_start=WINDOW_START,
        window_end_exclusive=WINDOW_END_EXCLUSIVE,
    )

    airport = pipeline.build_airport_economics_daily(
        pickup_window,
        dropoff_window,
        spark,
    )

    actual = {
        (
            str(row["service_day"]),
            row["airport_code"],
            row["direction"],
        ): row["trip_count"]
        for row in airport.collect()
    }

    assert actual == {
        (
            "2023-01-01",
            "JFK",
            "ARRIVAL",
        ): 1,
        (
            "2023-03-31",
            "JFK",
            "DEPARTURE",
        ): 1,
    }
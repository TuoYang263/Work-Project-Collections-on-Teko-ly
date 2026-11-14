from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq

from pyspark.sql import functions as F
from pyspark.sql.types import TimestampNTZType

from scripts import delta_medallion_pipeline as pipeline
from scripts.gold_quality_gate import (
    GOLD_TABLE_CONTRACTS,
    validate_all_gold_tables,
)


def make_trip(pickup, dropoff, pickup_zone, dropoff_zone):
    return {
        "VendorID": 1,
        "tpep_pickup_datetime": datetime.fromisoformat(pickup),
        "tpep_dropoff_datetime": datetime.fromisoformat(dropoff),
        "PULocationID": pickup_zone,
        "DOLocationID": dropoff_zone,
        "passenger_count": 1,
        "trip_distance": 3.5,
        "fare_amount": 15.0,
        "tip_amount": 3.0,
        "total_amount": 20.0,
    }


def test_ntz_bronze_silver_six_gold_isolated(
    tmp_path, monkeypatch
):
    feb = tmp_path / "feb.parquet"
    mar = tmp_path / "mar.parquet"

    # Previous-month trip ending inside the March window.
    feb_rows = [
        make_trip(
            "2023-02-28 23:55:00",
            "2023-03-01 00:05:00",
            10, 132,
        ),
    ]

    mar_rows = [
        make_trip(
            "2023-03-12 01:30:00",
            "2023-03-12 01:45:00",
            132, 10,
        ),
        make_trip(
            "2023-03-12 03:30:00",
            "2023-03-12 03:50:00",
            10, 132,
        ),
        make_trip(
            "2023-03-31 23:50:00",
            "2023-04-01 00:10:00",
            132, 10,
        ),
    ]

    pq.write_table(pa.Table.from_pylist(feb_rows), feb)
    pq.write_table(pa.Table.from_pylist(mar_rows), mar)

    zone_lookup = tmp_path / "zones.csv"
    zone_lookup.write_text(
        "LocationID,Borough,Zone,service_zone\n"
        "132,Queens,JFK Airport,Airports\n"
        "10,Manhattan,Sample Zone,Yellow Zone\n",
        encoding="utf-8",
    )

    root = tmp_path / "delta"

    paths = {
        "BRONZE_TRIPS": root / "bronze",
        "SILVER_TRIPS": root / "silver",
        "GOLD_SUMMARY_HOURLY": root / "gold/trip_summary_hourly",
        "GOLD_ZONE_PICKUP_DAILY": root / "gold/zone_summary_pickup_daily",
        "GOLD_ZONE_DROPOFF_DAILY": root / "gold/zone_summary_dropoff_daily",
        "GOLD_FLOW_IMBALANCE_DAILY": root / "gold/flow_imbalance_daily",
        "GOLD_AIRPORT_ECONOMICS_DAILY": root / "gold/airport_economics_daily",
        "GOLD_OD_CORRIDOR_DAILY": root / "gold/od_corridor_daily",
    }

    # Redirect every production output to an isolated test path.
    for name, path in paths.items():
        monkeypatch.setattr(pipeline, name, str(path))

    monkeypatch.setattr(
        pipeline, "TAXI_ZONE_LOOKUP", str(zone_lookup)
    )

    def fake_raw_resolver(raw_file_months, raw_references):
        assert raw_file_months == ["2023-02", "2023-03"]
        return [str(feb), str(mar)]

    monkeypatch.setattr(
        pipeline,
        "resolve_bronze_s3_paths",
        fake_raw_resolver,
    )

    contract = {
        "year": 2023,
        "months": [3],
        "write_scope": "FULL_TABLE_OVERWRITE",
        "logical_window": {
            "start": "2023-03-01",
            "end_exclusive": "2023-04-01",
        },
    }

    spark = pipeline.get_spark("NYC NTZ Medallion Integration")
    old_tz = spark.conf.get("spark.sql.session.timeZone")

    try:
        spark.conf.set("spark.sql.session.timeZone", "UTC")
        spark.conf.set("spark.sql.shuffle.partitions", "2")

        # Bronze
        pipeline.write_bronze(
            raw_file_months=["2023-02", "2023-03"],
            raw_references=[
                {"data_month": "2023-02"},
                {"data_month": "2023-03"},
            ],
        )

        bronze = spark.read.format("delta").load(
            str(paths["BRONZE_TRIPS"])
        )

        assert bronze.count() == 4

        # Silver
        pipeline.write_silver()

        silver = spark.read.format("delta").load(
            str(paths["SILVER_TRIPS"])
        )

        assert silver.count() == 4

        for df in (bronze, silver):
            for name in (
                "tpep_pickup_datetime",
                "tpep_dropoff_datetime",
            ):
                assert isinstance(
                    df.schema[name].dataType,
                    TimestampNTZType,
                )

        # All six Gold tables
        pipeline.write_gold(execution_contract=contract)

        gold_paths = {
            name: str(root / "gold" / name)
            for name in GOLD_TABLE_CONTRACTS
        }

        validate_all_gold_tables(
            spark=spark,
            table_paths=gold_paths,
            execution_contract=contract,
        )

        hourly = spark.read.format("delta").load(
            gold_paths["trip_summary_hourly"]
        )

        assert isinstance(
            hourly.schema["pickup_hour"].dataType,
            TimestampNTZType,
        )

        actual_hours = {
            row["hour"]
            for row in hourly.select(
                F.date_format(
                    "pickup_hour",
                    "yyyy-MM-dd HH:mm:ss",
                ).alias("hour")
            ).collect()
        }

        assert actual_hours == {
            "2023-03-12 01:00:00",
            "2023-03-12 03:00:00",
            "2023-03-31 23:00:00",
        }

        dropoff = spark.read.format("delta").load(
            gold_paths["zone_summary_dropoff_daily"]
        )

        # The February pickup has a valid March dropoff.
        assert dropoff.filter(
            F.col("dropoff_day") == "2023-03-01"
        ).count() == 1

    finally:
        spark.conf.set("spark.sql.session.timeZone", old_tz)
        spark.stop()

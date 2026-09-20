from datetime import datetime

from scripts.delta_medallion_pipeline import (
    build_hourly_summary,
    get_spark,
)


def main():
    spark = get_spark("Gold Hourly Correctness Test")

    input_rows = [
        (
            datetime(2023, 1, 1, 10, 15),
            1.0,
            10.0,
            2.0,
            2.0,
        ),
        (
            datetime(2023, 1, 1, 10, 45),
            2.0,
            20.0,
            4.0,
            4.0,
        ),
        (
            datetime(2023, 1, 1, 11, 5),
            3.0,
            30.0,
            6.0,
            6.0,
        ),
    ]

    input_df = spark.createDataFrame(
        input_rows,
        [
            "tpep_pickup_datetime",
            "passenger_count",
            "fare_amount",
            "tip_amount",
            "trip_distance",
        ],
    )

    result = build_hourly_summary(input_df).collect()

    assert len(result) == 2

    first = result[0]
    second = result[1]

    # 10:00 hour: two trips
    assert first.pickup_hour == datetime(2023, 1, 1, 10, 0)
    assert first.trip_count == 2
    assert first.avg_fare == 15.0
    assert first.avg_tip == 3.0
    assert first.total_passengers == 3.0
    assert first.avg_distance == 3.0

    # 11:00 hour: one trip
    assert second.pickup_hour == datetime(2023, 1, 1, 11, 0)
    assert second.trip_count == 1
    assert second.avg_fare == 30.0
    assert second.avg_tip == 6.0
    assert second.total_passengers == 3.0
    assert second.avg_distance == 6.0

    print("PASS: Gold hourly correctness test")
    print(f"rows={len(result)}")

    for row in result:
        print(row)


if __name__ == "__main__":
    main()
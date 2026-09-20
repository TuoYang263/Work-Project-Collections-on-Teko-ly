from datetime import datetime

from scripts.delta_medallion_pipeline import (
    get_spark,
    build_airport_economics_daily,
)


def main():
    spark = get_spark("Test Airport Economics")

    rows = [
        # EWR ARRIVAL -- should remain
        (
            datetime(2023, 1, 1, 10, 0),
            datetime(2023, 1, 1, 10, 30),
            161,
            1,
            100.0,
            80.0,
            18.0,
        ),

        # EWR DEPARTURE -- should be excluded
        (
            datetime(2023, 1, 1, 11, 0),
            datetime(2023, 1, 1, 11, 30),
            1,
            161,
            120.0,
            95.0,
            18.0,
        ),

        # JFK DEPARTURE
        (
            datetime(2023, 1, 1, 12, 0),
            datetime(2023, 1, 1, 12, 30),
            132,
            161,
            80.0,
            65.0,
            16.0,
        ),

        # JFK ARRIVAL
        (
            datetime(2023, 1, 1, 13, 0),
            datetime(2023, 1, 1, 13, 30),
            161,
            132,
            82.0,
            66.0,
            16.0,
        ),

        # LGA DEPARTURE
        (
            datetime(2023, 1, 1, 14, 0),
            datetime(2023, 1, 1, 14, 30),
            138,
            161,
            60.0,
            42.0,
            10.0,
        ),

        # LGA ARRIVAL
        (
            datetime(2023, 1, 1, 15, 0),
            datetime(2023, 1, 1, 15, 30),
            161,
            138,
            62.0,
            43.0,
            10.0,
        ),
    ]

    schema = """
        tpep_pickup_datetime timestamp,
        tpep_dropoff_datetime timestamp,
        pulocationid long,
        dolocationid long,
        total_amount double,
        fare_amount double,
        trip_distance double
    """

    df = spark.createDataFrame(rows, schema=schema)

    result = build_airport_economics_daily(
        df,
        spark,
    )

    actual = {
        (row.airport_code, row.direction): row
        for row in result.collect()
    }

    expected_keys = {
        ("EWR", "ARRIVAL"),
        ("JFK", "ARRIVAL"),
        ("JFK", "DEPARTURE"),
        ("LGA", "ARRIVAL"),
        ("LGA", "DEPARTURE"),
    }

    assert set(actual.keys()) == expected_keys

    # Explicitly verify the business rule.
    assert ("EWR", "DEPARTURE") not in actual

    # One synthetic trip per valid airport-direction combination.
    for row in actual.values():
        assert row.trip_count == 1
        assert row.avg_duration_minutes == 30.0

    assert actual[("EWR", "ARRIVAL")].gross_amount == 100.0
    assert actual[("JFK", "DEPARTURE")].gross_amount == 80.0
    assert actual[("JFK", "ARRIVAL")].gross_amount == 82.0
    assert actual[("LGA", "DEPARTURE")].gross_amount == 60.0
    assert actual[("LGA", "ARRIVAL")].gross_amount == 62.0

    print("PASS: airport economics contract")
    print(f"rows={len(actual)}")


if __name__ == "__main__":
    main()
from datetime import datetime

from scripts.delta_medallion_pipeline import (
    get_spark,
    build_od_corridor_daily,
)


def main():
    spark = get_spark("Test OD Corridor")

    rows = [
        # JFK -> Times Square
        (
            datetime(2023, 1, 1, 10, 0),
            datetime(2023, 1, 1, 10, 30),
            132,
            230,
            100.0,
            80.0,
            10.0,
        ),

        # JFK -> Times Square, second trip
        (
            datetime(2023, 1, 1, 11, 0),
            datetime(2023, 1, 1, 11, 40),
            132,
            230,
            120.0,
            90.0,
            10.0,
        ),

        # Reverse direction:
        # Times Square -> JFK
        (
            datetime(2023, 1, 1, 12, 0),
            datetime(2023, 1, 1, 12, 30),
            230,
            132,
            90.0,
            70.0,
            15.0,
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

    df = spark.createDataFrame(
        rows,
        schema=schema,
    )

    result = build_od_corridor_daily(
        df,
        spark,
    )

    actual = {
        (
            row.pickup_zone_id,
            row.dropoff_zone_id,
        ): row
        for row in result.collect()
    }

    # Directional corridor contract:
    # A -> B and B -> A are different corridors.
    assert set(actual.keys()) == {
        (132, 230),
        (230, 132),
    }

    outbound = actual[(132, 230)]
    inbound = actual[(230, 132)]

    # JFK -> Times Square contains two trips.
    assert outbound.trip_count == 2
    assert outbound.gross_amount == 220.0
    assert outbound.avg_total_amount == 110.0
    assert outbound.avg_trip_distance == 10.0

    # Times Square -> JFK is independent.
    assert inbound.trip_count == 1
    assert inbound.gross_amount == 90.0
    assert inbound.avg_trip_distance == 15.0

    # Zone enrichment must resolve both roles.
    assert outbound.pickup_zone == "JFK Airport"
    assert outbound.dropoff_zone == "Times Sq/Theatre District"

    assert inbound.pickup_zone == "Times Sq/Theatre District"
    assert inbound.dropoff_zone == "JFK Airport"

    # Single reverse trip:
    # 30 minutes / 15 miles = 2 min/mile.
    assert inbound.median_duration_per_mile == 2.0
    assert inbound.p90_duration_per_mile == 2.0

    print("PASS: OD corridor contract")
    print(f"rows={len(actual)}")


if __name__ == "__main__":
    main()
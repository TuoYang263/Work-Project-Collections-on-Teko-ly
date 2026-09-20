from datetime import datetime

from scripts.delta_medallion_pipeline import (
    build_zone_flow_imbalance,
    get_spark,
)


def main():
    spark = get_spark("Zone Flow Imbalance Correctness Test")

    input_rows = [
        # Zone 1: origin-heavy
        # pickups=3, dropoffs=1 -> (3-1)/(3+1) = 0.5
        (datetime(2023, 1, 1, 10, 0), datetime(2023, 1, 1, 10, 20), 1, 101),
        (datetime(2023, 1, 1, 11, 0), datetime(2023, 1, 1, 11, 20), 1, 102),
        (datetime(2023, 1, 1, 12, 0), datetime(2023, 1, 1, 12, 20), 1, 103),
        (datetime(2023, 1, 1, 13, 0), datetime(2023, 1, 1, 13, 20), 104, 1),

        # Zone 2: balanced
        # pickups=2, dropoffs=2 -> 0.0
        (datetime(2023, 1, 1, 14, 0), datetime(2023, 1, 1, 14, 20), 2, 201),
        (datetime(2023, 1, 1, 15, 0), datetime(2023, 1, 1, 15, 20), 2, 202),
        (datetime(2023, 1, 1, 16, 0), datetime(2023, 1, 1, 16, 20), 203, 2),
        (datetime(2023, 1, 1, 17, 0), datetime(2023, 1, 1, 17, 20), 204, 2),

        # Zone 3: destination-heavy
        # pickups=1, dropoffs=3 -> (1-3)/(1+3) = -0.5
        (datetime(2023, 1, 1, 18, 0), datetime(2023, 1, 1, 18, 20), 3, 301),
        (datetime(2023, 1, 1, 19, 0), datetime(2023, 1, 1, 19, 20), 302, 3),
        (datetime(2023, 1, 1, 20, 0), datetime(2023, 1, 1, 20, 20), 303, 3),
        (datetime(2023, 1, 1, 21, 0), datetime(2023, 1, 1, 21, 20), 304, 3),
    ]

    input_df = spark.createDataFrame(
        input_rows,
        [
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "pulocationid",
            "dolocationid",
        ],
    )

    result = (
        build_zone_flow_imbalance(input_df)
        .filter("zone_id IN (1, 2, 3)")
        .collect()
    )

    by_zone = {row.zone_id: row for row in result}

    # Origin-heavy
    assert by_zone[1].pickup_count == 3
    assert by_zone[1].dropoff_count == 1
    assert by_zone[1].total_flow == 4
    assert by_zone[1].flow_imbalance == 0.5

    # Balanced
    assert by_zone[2].pickup_count == 2
    assert by_zone[2].dropoff_count == 2
    assert by_zone[2].total_flow == 4
    assert by_zone[2].flow_imbalance == 0.0

    # Destination-heavy
    assert by_zone[3].pickup_count == 1
    assert by_zone[3].dropoff_count == 3
    assert by_zone[3].total_flow == 4
    assert by_zone[3].flow_imbalance == -0.5

    print("PASS: Zone flow imbalance correctness test")

    for row in result:
        print(row)


if __name__ == "__main__":
    main()
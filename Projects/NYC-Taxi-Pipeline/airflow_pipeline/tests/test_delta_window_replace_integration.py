from pyspark.sql import functions as F

from scripts import delta_medallion_pipeline as pipeline


def _make_daily_df(
    spark,
    rows,
):
    return (
        spark.createDataFrame(
            rows,
            [
                "service_day_raw",
                "trip_count",
            ],
        )
        .withColumn(
            "service_day",
            F.to_date("service_day_raw"),
        )
        .drop("service_day_raw")
    )


def _read_rows(
    spark,
    target_path,
):
    return [
        (
            row["service_day"].isoformat(),
            row["trip_count"],
        )
        for row in (
            spark.read
            .format("delta")
            .load(target_path)
            .orderBy("service_day")
            .collect()
        )
    ]


def test_window_replace_preserves_outside_window_and_is_idempotent(
    tmp_path,
):
    spark = pipeline.get_spark(
        "NYC Delta Window Replace Test"
    )

    target_path = str(
        tmp_path / "gold_window_replace"
    )

    initial_df = _make_daily_df(
        spark,
        [
            ("2023-01-10", 10),
            ("2023-02-10", 20),
            ("2023-02-20", 30),
            ("2023-03-10", 40),
        ],
    )

    (
        initial_df.write
        .mode("overwrite")
        .format("delta")
        .save(target_path)
    )

    backfill_df = _make_daily_df(
        spark,
        [
            ("2023-02-10", 200),
            ("2023-02-25", 250),
        ],
    )

    execution_contract = {
        "write_scope": "WINDOW_REPLACE",
        "logical_window": {
            "start": "2023-02-01",
            "end_exclusive": "2023-03-01",
        },
    }

    pipeline.write_delta_table(
        backfill_df,
        target_path=target_path,
        execution_contract=execution_contract,
        label="synthetic_daily",
        window_column="service_day",
    )

    expected_rows = [
        ("2023-01-10", 10),
        ("2023-02-10", 200),
        ("2023-02-25", 250),
        ("2023-03-10", 40),
    ]

    after_first_backfill = _read_rows(
        spark,
        target_path,
    )

    assert after_first_backfill == expected_rows

    # Retry the exact same backfill.
    pipeline.write_delta_table(
        backfill_df,
        target_path=target_path,
        execution_contract=execution_contract,
        label="synthetic_daily",
        window_column="service_day",
    )

    after_second_backfill = _read_rows(
        spark,
        target_path,
    )

    assert after_second_backfill == expected_rows
    assert (
        after_second_backfill
        == after_first_backfill
    )
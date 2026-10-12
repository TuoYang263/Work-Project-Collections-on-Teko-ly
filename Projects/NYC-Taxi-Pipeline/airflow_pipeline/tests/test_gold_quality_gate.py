
from datetime import datetime
from types import SimpleNamespace

import pytest

from scripts.gold_quality_gate import (
    GOLD_TABLE_CONTRACTS,
    validate_gold_schema,
    validate_all_gold_tables,
    run_gold_quality_gate,
    validate_gold_business_keys,
    validate_gold_logical_window,
    validate_gold_positive_counts,
    validate_gold_imbalance_range,
    validate_gold_airport_direction,
)
from datetime import date

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DateType,
    IntegerType,
    StructField,
    StructType,
    DoubleType,
    StringType,
    TimestampType,
)


def test_gold_schema_passes_with_required_columns():
    df = SimpleNamespace(
        columns=["pickup_hour", "trip_count"]
    )

    validate_gold_schema(
        df,
        table_name="trip_summary_hourly",
    )


def test_gold_schema_fails_when_column_missing():
    df = SimpleNamespace(
        columns=["pickup_hour"]
    )

    with pytest.raises(
        ValueError,
        match="rule=REQUIRED_COLUMNS",
    ) as exc_info:
        validate_gold_schema(
            df,
            table_name="trip_summary_hourly",
        )

    assert "trip_count" in str(exc_info.value)


@pytest.mark.parametrize(
    "table_name, columns, missing_key",
    [
        (
            "zone_summary_pickup_daily",
            ["pickup_day", "zone_id", "trip_count"],
            "zone_id",
        ),
        (
            "zone_summary_dropoff_daily",
            ["dropoff_day", "zone_id", "trip_count"],
            "zone_id",
        ),
        (
            "flow_imbalance_daily",
            [
                "service_day",
                "zone_id",
                "total_flow",
                "flow_imbalance",
            ],
            "zone_id",
        ),
        (
            "airport_economics_daily",
            [
                "service_day",
                "airport_code",
                "direction",
                "trip_count",
            ],
            "airport_code",
        ),
        (
            "od_corridor_daily",
            [
                "service_day",
                "pickup_zone_id",
                "dropoff_zone_id",
                "trip_count",
            ],
            "pickup_zone_id",
        ),
    ],
)
def test_remaining_gold_table_schemas(
    table_name,
    columns,
    missing_key,
):
    # Case A: complete schema should pass.
    valid_df = SimpleNamespace(columns=columns)

    validate_gold_schema(
        valid_df,
        table_name,
    )

    # Case B: removing a required key should fail.
    invalid_columns = [
        name
        for name in columns
        if name != missing_key
    ]

    invalid_df = SimpleNamespace(
        columns=invalid_columns
    )

    with pytest.raises(
        ValueError,
        match="rule=REQUIRED_COLUMNS",
    ) as exc_info:
        validate_gold_schema(
            invalid_df,
            table_name,
        )

    assert missing_key in str(exc_info.value)



@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder
        .master("local[2]")
        .appName("GoldBusinessKeyTest")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )

    yield session
    session.stop()


def make_zone_df(spark, rows):
    schema = StructType([
        StructField("pickup_day", DateType()),
        StructField("zone_id", IntegerType()),
        StructField("trip_count", IntegerType()),
    ])

    return spark.createDataFrame(
        rows,
        schema=schema,
    )


def test_business_keys_pass(spark):
    df = make_zone_df(spark, [
        (date(2023, 1, 1), 132, 150),
        (date(2023, 1, 1), 237, 210),
    ])

    validate_gold_business_keys(
        df,
        "zone_summary_pickup_daily",
    )


@pytest.mark.parametrize(
    "rows, expected_rule",
    [
        (
            [
                (date(2023, 1, 1), None, 80),
            ],
            "NULL_BUSINESS_KEY",
        ),
        (
            [
                (date(2023, 1, 1), 132, 150),
                (date(2023, 1, 1), 132, 160),
            ],
            "DUPLICATE_BUSINESS_KEY",
        ),
    ],
)
def test_business_keys_reject_invalid(
    spark,
    rows,
    expected_rule,
):
    df = make_zone_df(spark, rows)

    with pytest.raises(
        ValueError,
        match=expected_rule,
    ):
        validate_gold_business_keys(
            df,
            "zone_summary_pickup_daily",
        )



def test_positive_trip_count_passes(spark):
    df = make_zone_df(
        spark,
        [(date(2023, 1, 1), 132, 150)],
    )

    validate_gold_positive_counts(
        df,
        "zone_summary_pickup_daily",
    )


@pytest.mark.parametrize(
    "bad_count",
    [0, -5, None],
)
def test_positive_trip_count_rejects_invalid(
    spark,
    bad_count,
):
    df = make_zone_df(
        spark,
        [(date(2023, 1, 1), 132, bad_count)],
    )

    with pytest.raises(
        ValueError,
        match="rule=POSITIVE_COUNT",
    ):
        validate_gold_positive_counts(
            df,
            "zone_summary_pickup_daily",
        )


def test_flow_uses_total_flow_instead_of_trip_count(
    spark,
):
    df = spark.createDataFrame(
        [
            (date(2023, 1, 1), 132, 0, 0.0),
        ],
        [
            "service_day",
            "zone_id",
            "total_flow",
            "flow_imbalance",
        ],
    )

    with pytest.raises(
        ValueError,
        match="column=total_flow",
    ):
        validate_gold_positive_counts(
            df,
            "flow_imbalance_daily",
        )


def make_flow_df(spark, imbalance_value):
    schema = StructType([
        StructField("service_day", DateType()),
        StructField("zone_id", IntegerType()),
        StructField("total_flow", IntegerType()),
        StructField("flow_imbalance", DoubleType()),
    ])

    return spark.createDataFrame(
        [
            (
                date(2023, 1, 1),
                132,
                100,
                imbalance_value,
            ),
        ],
        schema=schema,
    )


@pytest.mark.parametrize(
    "valid_value",
    [-1.0, 0.0, 1.0],
)
def test_imbalance_range_accepts_valid(
    spark,
    valid_value,
):
    df = make_flow_df(spark, valid_value)

    validate_gold_imbalance_range(df)


@pytest.mark.parametrize(
    "invalid_value",
    [-1.01, 1.01, None, float("nan")],
)
def test_imbalance_range_rejects_invalid(
    spark,
    invalid_value,
):
    df = make_flow_df(spark, invalid_value)

    with pytest.raises(
        ValueError,
        match="rule=IMBALANCE_RANGE",
    ):
        validate_gold_imbalance_range(df)



def make_airport_df(spark, direction_value):
    schema = StructType([
        StructField("service_day", DateType()),
        StructField("airport_code", StringType()),
        StructField("direction", StringType()),
        StructField("trip_count", IntegerType()),
    ])

    return spark.createDataFrame(
        [
            (
                date(2023, 1, 1),
                "JFK",
                direction_value,
                100,
            ),
        ],
        schema=schema,
    )


@pytest.mark.parametrize(
    "valid_direction",
    ["ARRIVAL", "DEPARTURE"],
)
def test_airport_direction_accepts_valid(
    spark,
    valid_direction,
):
    df = make_airport_df(
        spark,
        valid_direction,
    )

    validate_gold_airport_direction(df)


@pytest.mark.parametrize(
    "invalid_direction",
    ["UNKNOWN", None],
)
def test_airport_direction_rejects_invalid(
    spark,
    invalid_direction,
):
    df = make_airport_df(
        spark,
        invalid_direction,
    )

    with pytest.raises(
        ValueError,
        match="rule=AIRPORT_DIRECTION",
    ):
        validate_gold_airport_direction(df)



WINDOW = {
    "start": "2023-02-01",
    "end_exclusive": "2023-03-01",
}


def test_daily_window_accepts_valid_dates(spark):
    df = make_zone_df(
        spark,
        [
            (date(2023, 2, 1), 132, 100),
            (date(2023, 2, 28), 237, 200),
        ],
    )

    validate_gold_logical_window(
        df,
        "zone_summary_pickup_daily",
        WINDOW,
    )


@pytest.mark.parametrize(
    "bad_day",
    [
        date(2023, 1, 31),
        date(2023, 3, 1),
        None,
    ],
)
def test_daily_window_rejects_invalid_dates(
    spark,
    bad_day,
):
    df = make_zone_df(
        spark,
        [(bad_day, 132, 100)],
    )

    with pytest.raises(
        ValueError,
        match="rule=LOGICAL_WINDOW",
    ):
        validate_gold_logical_window(
            df,
            "zone_summary_pickup_daily",
            WINDOW,
        )


def make_hourly_window_df(spark, hour):
    schema = StructType([
        StructField("pickup_hour", TimestampType()),
        StructField("trip_count", IntegerType()),
    ])

    return spark.createDataFrame(
        [(hour, 100)],
        schema=schema,
    )


def test_hourly_window_accepts_valid_timestamp(spark):
    df = make_hourly_window_df(
        spark,
        datetime(2023, 2, 28, 23, 0),
    )

    validate_gold_logical_window(
        df,
        "trip_summary_hourly",
        WINDOW,
    )


@pytest.mark.parametrize(
    "bad_hour",
    [
        datetime(2023, 1, 31, 23, 0),
        datetime(2023, 3, 1, 0, 0),
        None,
    ],
)
def test_hourly_window_rejects_invalid_timestamp(
    spark,
    bad_hour,
):
    df = make_hourly_window_df(spark, bad_hour)

    with pytest.raises(
        ValueError,
        match="rule=LOGICAL_WINDOW",
    ):
        validate_gold_logical_window(
            df,
            "trip_summary_hourly",
            WINDOW,
        )


def test_logical_window_rejects_invalid_contract(spark):
    df = make_zone_df(
        spark,
        [(date(2023, 2, 1), 132, 100)],
    )

    with pytest.raises(
        ValueError,
        match="Invalid logical window",
    ):
        validate_gold_logical_window(
            df,
            "zone_summary_pickup_daily",
            {
                "start": "2023-03-01",
                "end_exclusive": "2023-02-01",
            },
        )



JAN_WINDOW = {
    "start": "2023-01-01",
    "end_exclusive": "2023-02-01",
}


def test_quality_gate_accepts_valid_zone(spark):
    df = make_zone_df(
        spark,
        [(date(2023, 1, 1), 132, 150)],
    )

    run_gold_quality_gate(
        df,
        "zone_summary_pickup_daily",
        JAN_WINDOW,
    )


def test_quality_gate_rejects_invalid_imbalance(spark):
    df = make_flow_df(spark, 1.25)

    with pytest.raises(
        ValueError,
        match="rule=IMBALANCE_RANGE",
    ):
        run_gold_quality_gate(
            df,
            "flow_imbalance_daily",
            JAN_WINDOW,
        )


def test_quality_gate_rejects_invalid_airport(spark):
    df = make_airport_df(spark, "UNKNOWN")

    with pytest.raises(
        ValueError,
        match="rule=AIRPORT_DIRECTION",
    ):
        run_gold_quality_gate(
            df,
            "airport_economics_daily",
            JAN_WINDOW,
        )



def test_backfill_rejects_empty_validation_scope(spark):
    # Gold contains January data only.
    jan_df = make_hourly_window_df(
        spark,
        datetime(2023, 1, 15, 8, 0),
    )

    class FakeDeltaReader:
        def format(self, file_format):
            assert file_format == "delta"
            return self

        def load(self, path):
            assert path == "mock://trip_summary_hourly"
            return jan_df

    fake_spark = SimpleNamespace(
        read=FakeDeltaReader()
    )

    table_paths = {
        name: f"mock://{name}"
        for name in GOLD_TABLE_CONTRACTS
    }

    # WINDOW was defined earlier for February 2023.
    contract = {
        "write_scope": "WINDOW_REPLACE",
        "logical_window": WINDOW,
    }

    with pytest.raises(
        ValueError,
        match="rule=EMPTY_VALIDATION_SCOPE",
    ):
        validate_all_gold_tables(
            fake_spark,
            table_paths,
            contract,
        )



def test_backfill_accepts_nonempty_target_window(spark):
    base = spark.createDataFrame(
        [
            ("2023-01-15",),
            ("2023-02-15",),
        ],
        ["day_text"],
    )

    # January deliberately has invalid counts.
    # The February backfill should validate February only.
    trip_count = (
        "CASE WHEN day_text = '2023-01-15' "
        "THEN -5 ELSE 10 END AS trip_count"
    )

    specs = {
        "trip_summary_hourly": [
            "CAST(day_text AS TIMESTAMP) AS pickup_hour",
            trip_count,
        ],
        "zone_summary_pickup_daily": [
            "CAST(day_text AS DATE) AS pickup_day",
            "132 AS zone_id",
            trip_count,
        ],
        "zone_summary_dropoff_daily": [
            "CAST(day_text AS DATE) AS dropoff_day",
            "237 AS zone_id",
            trip_count,
        ],
        "flow_imbalance_daily": [
            "CAST(day_text AS DATE) AS service_day",
            "132 AS zone_id",
            "10 AS total_flow",
            "0.2 AS flow_imbalance",
        ],
        "airport_economics_daily": [
            "CAST(day_text AS DATE) AS service_day",
            "'JFK' AS airport_code",
            "'ARRIVAL' AS direction",
            trip_count,
        ],
        "od_corridor_daily": [
            "CAST(day_text AS DATE) AS service_day",
            "132 AS pickup_zone_id",
            "237 AS dropoff_zone_id",
            trip_count,
        ],
    }

    tables = {
        name: base.selectExpr(*columns)
        for name, columns in specs.items()
    }

    assert set(tables) == set(GOLD_TABLE_CONTRACTS)

    class FakeDeltaReader:
        def __init__(self):
            self.loaded = []

        def format(self, file_format):
            assert file_format == "delta"
            return self

        def load(self, path):
            self.loaded.append(path)
            return tables[path.removeprefix("mock://")]

    reader = FakeDeltaReader()
    fake_spark = SimpleNamespace(read=reader)

    table_paths = {
        name: f"mock://{name}"
        for name in GOLD_TABLE_CONTRACTS
    }

    validate_all_gold_tables(
        fake_spark,
        table_paths,
        {
            "write_scope": "WINDOW_REPLACE",
            "logical_window": WINDOW,
        },
    )

    assert set(reader.loaded) == set(table_paths.values())
    assert len(reader.loaded) == 6



def test_full_refresh_rejects_out_of_window_data(spark):
    jan_df = make_hourly_window_df(
        spark,
        datetime(2023, 1, 15, 8, 0),
    )

    feb_df = make_hourly_window_df(
        spark,
        datetime(2023, 2, 15, 8, 0),
    )

    # February is valid, January is outside the window.
    mixed_df = jan_df.unionByName(feb_df)

    class FakeDeltaReader:
        def format(self, file_format):
            assert file_format == "delta"
            return self

        def load(self, path):
            assert path == "mock://trip_summary_hourly"
            return mixed_df

    fake_spark = SimpleNamespace(
        read=FakeDeltaReader()
    )

    table_paths = {
        name: f"mock://{name}"
        for name in GOLD_TABLE_CONTRACTS
    }

    contract = {
        "write_scope": "FULL_TABLE_OVERWRITE",
        "logical_window": WINDOW,
    }

    with pytest.raises(
        ValueError,
        match="rule=LOGICAL_WINDOW",
    ):
        validate_all_gold_tables(
            fake_spark,
            table_paths,
            contract,
        )



def test_full_refresh_accepts_all_six_tables(spark):
    base = spark.createDataFrame(
        [("2023-02-15",)],
        ["day_text"],
    )

    specs = {
        "trip_summary_hourly": [
            "CAST(day_text AS TIMESTAMP) AS pickup_hour",
            "10 AS trip_count",
        ],
        "zone_summary_pickup_daily": [
            "CAST(day_text AS DATE) AS pickup_day",
            "132 AS zone_id",
            "10 AS trip_count",
        ],
        "zone_summary_dropoff_daily": [
            "CAST(day_text AS DATE) AS dropoff_day",
            "237 AS zone_id",
            "10 AS trip_count",
        ],
        "flow_imbalance_daily": [
            "CAST(day_text AS DATE) AS service_day",
            "132 AS zone_id",
            "10 AS total_flow",
            "0.2 AS flow_imbalance",
        ],
        "airport_economics_daily": [
            "CAST(day_text AS DATE) AS service_day",
            "'JFK' AS airport_code",
            "'ARRIVAL' AS direction",
            "10 AS trip_count",
        ],
        "od_corridor_daily": [
            "CAST(day_text AS DATE) AS service_day",
            "132 AS pickup_zone_id",
            "237 AS dropoff_zone_id",
            "10 AS trip_count",
        ],
    }

    tables = {
        name: base.selectExpr(*columns)
        for name, columns in specs.items()
    }

    assert set(tables) == set(GOLD_TABLE_CONTRACTS)

    class FakeDeltaReader:
        def __init__(self):
            self.loaded = []

        def format(self, file_format):
            assert file_format == "delta"
            return self

        def load(self, path):
            self.loaded.append(path)
            return tables[path.removeprefix("mock://")]

    reader = FakeDeltaReader()

    paths = {
        name: f"mock://{name}"
        for name in GOLD_TABLE_CONTRACTS
    }

    validate_all_gold_tables(
        SimpleNamespace(read=reader),
        paths,
        {
            "write_scope": "FULL_TABLE_OVERWRITE",
            "logical_window": WINDOW,
        },
    )

    assert set(reader.loaded) == set(paths.values())
    assert len(reader.loaded) == 6

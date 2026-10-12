
"""
NYC V2 — Gold Data Quality Gate.

Step 2: Schema validation for six Gold tables.
"""
from datetime import date
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


GOLD_TABLE_CONTRACTS = {
    "trip_summary_hourly": {
        "keys": ("pickup_hour",),
        "metrics": ("trip_count",),
    },
    "zone_summary_pickup_daily": {
        "keys": ("pickup_day", "zone_id"),
        "metrics": ("trip_count",),
    },
    "zone_summary_dropoff_daily": {
        "keys": ("dropoff_day", "zone_id"),
        "metrics": ("trip_count",),
    },
    "flow_imbalance_daily": {
        "keys": ("service_day", "zone_id"),
        "metrics": ("total_flow", "flow_imbalance"),
    },
    "airport_economics_daily": {
        "keys": ("service_day", "airport_code", "direction"),
        "metrics": ("trip_count",),
    },
    "od_corridor_daily": {
        "keys": (
            "service_day",
            "pickup_zone_id",
            "dropoff_zone_id",
        ),
        "metrics": ("trip_count",),
    },
}


def validate_gold_schema(
    df: DataFrame,
    table_name: str,
) -> None:
    """
    Ensure all required Gold columns exist.

    Schema-only validation: no Spark Action.
    """
    if table_name not in GOLD_TABLE_CONTRACTS:
        raise ValueError(
            f"Unknown Gold table: {table_name}"
        )

    contract = GOLD_TABLE_CONTRACTS[table_name]

    required_columns = (
        set(contract["keys"])
        | set(contract["metrics"])
    )

    actual_columns = set(df.columns)

    missing_columns = sorted(
        required_columns - actual_columns
    )

    if missing_columns:
        raise ValueError(
            f"Gold Quality Gate FAILED: "
            f"table={table_name}, "
            f"rule=REQUIRED_COLUMNS, "
            f"missing={missing_columns}"
        )



def validate_gold_business_keys(
    df: DataFrame,
    table_name: str,
) -> None:
    """
    Check Gold business keys for NULLs and duplicates.
    """
    validate_gold_schema(df, table_name)

    keys = GOLD_TABLE_CONTRACTS[table_name]["keys"]

    grouped = df.groupBy(*keys).count()

    invalid_condition = F.col("count") > 1

    for key in keys:
        invalid_condition = (
            invalid_condition | F.col(key).isNull()
        )

    violations = (
        grouped
        .filter(invalid_condition)
        .limit(1)
        .collect()
    )

    if not violations:
        return

    row = violations[0]

    has_null = any(
        row[key] is None
        for key in keys
    )

    rule = (
        "NULL_BUSINESS_KEY"
        if has_null
        else "DUPLICATE_BUSINESS_KEY"
    )

    key_values = {
        key: row[key]
        for key in keys
    }

    raise ValueError(
        f"Gold Quality Gate FAILED: "
        f"table={table_name}, "
        f"rule={rule}, "
        f"key_values={key_values}"
    )


def validate_gold_positive_counts(
    df: DataFrame,
    table_name: str,
) -> None:
    """
    Ensure Gold trip counts or total flow are positive.
    """
    validate_gold_schema(df, table_name)

    metric_column = (
        "total_flow"
        if table_name == "flow_imbalance_daily"
        else "trip_count"
    )

    metric = F.col(metric_column)

    invalid_rows = (
        df.filter(
            metric.isNull() | (metric <= 0)
        )
        .select(metric_column)
        .limit(1)
        .collect()
    )

    if invalid_rows:
        raise ValueError(
            f"Gold Quality Gate FAILED: "
            f"table={table_name}, "
            f"rule=POSITIVE_COUNT, "
            f"column={metric_column}, "
            f"value={invalid_rows[0][metric_column]}"
        )


def validate_gold_imbalance_range(
    df: DataFrame,
) -> None:
    """
    Ensure flow imbalance is within [-1, 1].
    """
    table_name = "flow_imbalance_daily"

    validate_gold_schema(df, table_name)

    imbalance = F.col("flow_imbalance")

    invalid_rows = (
        df.filter(
            imbalance.isNull()
            | F.isnan(imbalance)
            | (imbalance < -1.0)
            | (imbalance > 1.0)
        )
        .select("flow_imbalance")
        .limit(1)
        .collect()
    )

    if invalid_rows:
        raise ValueError(
            f"Gold Quality Gate FAILED: "
            f"table={table_name}, "
            f"rule=IMBALANCE_RANGE, "
            f"value={invalid_rows[0]['flow_imbalance']}"
        )



def validate_gold_airport_direction(
    df: DataFrame,
) -> None:
    """
    Ensure airport direction is ARRIVAL or DEPARTURE.
    """
    table_name = "airport_economics_daily"

    validate_gold_schema(df, table_name)

    direction = F.col("direction")

    invalid_rows = (
        df.filter(
            direction.isNull()
            | ~direction.isin(
                "ARRIVAL",
                "DEPARTURE",
            )
        )
        .select("direction")
        .limit(1)
        .collect()
    )

    if invalid_rows:
        raise ValueError(
            f"Gold Quality Gate FAILED: "
            f"table={table_name}, "
            f"rule=AIRPORT_DIRECTION, "
            f"value={invalid_rows[0]['direction']}"
        )


def validate_gold_logical_window(
    df: DataFrame,
    table_name: str,
    logical_window: dict,
) -> None:
    """
    Ensure Gold business dates stay inside
    the half-open logical window [start, end).
    """
    validate_gold_schema(df, table_name)

    start = date.fromisoformat(
        logical_window["start"]
    )
    end = date.fromisoformat(
        logical_window["end_exclusive"]
    )

    if start >= end:
        raise ValueError(
            "Invalid logical window: start must precede end"
        )

    # All six Gold contracts use the business-time
    # column as their first key.
    time_column = (
        GOLD_TABLE_CONTRACTS[table_name]["keys"][0]
    )

    # Hourly summaries use timestamps.
    # Daily summaries use dates.
    column_type = (
        "timestamp"
        if table_name == "trip_summary_hourly"
        else "date"
    )

    business_time = F.col(time_column)

    invalid_rows = (
        df.filter(
            business_time.isNull()
            | (
                business_time
                < F.lit(start.isoformat()).cast(column_type)
            )
            | (
                business_time
                >= F.lit(end.isoformat()).cast(column_type)
            )
        )
        .select(time_column)
        .limit(1)
        .collect()
    )

    if invalid_rows:
        raise ValueError(
            f"Gold Quality Gate FAILED: "
            f"table={table_name}, "
            f"rule=LOGICAL_WINDOW, "
            f"column={time_column}, "
            f"value={invalid_rows[0][time_column]}"
        )



def run_gold_quality_gate(
    df: DataFrame,
    table_name: str,
    logical_window: dict,
) -> None:
    """
    Run all applicable Gold quality checks.

    Raise ValueError when a blocking rule fails.
    Return normally when all checks pass.
    """
    validate_gold_schema(df, table_name)

    validate_gold_logical_window(
        df,
        table_name,
        logical_window,
    )

    validate_gold_business_keys(
        df,
        table_name,
    )

    validate_gold_positive_counts(
        df,
        table_name,
    )

    if table_name == "flow_imbalance_daily":
        validate_gold_imbalance_range(df)

    if table_name == "airport_economics_daily":
        validate_gold_airport_direction(df)



def validate_all_gold_tables(
    spark: SparkSession,
    table_paths: dict[str, str],
    execution_contract: dict,
) -> None:
    """
    Validate all six Gold Delta tables.

    Full refresh: validate whole tables.
    Backfill: validate the target logical window.
    """
    expected = set(GOLD_TABLE_CONTRACTS)
    actual = set(table_paths)

    if expected != actual:
        raise ValueError(
            "Gold table path mismatch: "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )

    logical_window = execution_contract["logical_window"]
    write_scope = execution_contract["write_scope"]

    if write_scope not in {
        "FULL_TABLE_OVERWRITE",
        "WINDOW_REPLACE",
    }:
        raise ValueError(
            f"Unsupported write_scope: {write_scope}"
        )

    for table_name in GOLD_TABLE_CONTRACTS:
        df = (
            spark.read
            .format("delta")
            .load(table_paths[table_name])
        )

        validate_gold_schema(df, table_name)

        if write_scope == "WINDOW_REPLACE":
            time_column = (
                GOLD_TABLE_CONTRACTS[table_name]["keys"][0]
            )

            column_type = (
                "timestamp"
                if table_name == "trip_summary_hourly"
                else "date"
            )

            start = F.lit(
                logical_window["start"]
            ).cast(column_type)

            end = F.lit(
                logical_window["end_exclusive"]
            ).cast(column_type)

            business_time = F.col(time_column)

            df = df.filter(
                business_time.isNull()
                | (
                    (business_time >= start)
                    & (business_time < end)
                )
            )

        # Fail closed if the validation scope is empty.
        if not df.limit(1).collect():
            raise ValueError(
                "Gold Quality Gate FAILED: "
                f"table={table_name}, "
                "rule=EMPTY_VALIDATION_SCOPE"
            )

        run_gold_quality_gate(
            df,
            table_name,
            logical_window,
        )

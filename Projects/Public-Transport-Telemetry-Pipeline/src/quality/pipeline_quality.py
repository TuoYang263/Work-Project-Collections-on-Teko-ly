from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.quality.check_utils import (
    check_allowed_values,
    check_coordinate_bounds,
    check_datetime_order,
    check_file_exists,
    check_no_null_like_strings,
    check_not_empty,
    check_numeric_range,
    check_parseable_datetime,
    check_required_columns,
    check_unique_key,
    read_parquet_safely,
)
from src.quality.contracts import (
    ALLOWED_DAILY_DQ_FLAGS,
    ALLOWED_TRANSPORT_MODES,
    ALLOWED_WINDOW_DQ_FLAGS,
    ALL_DATASET_CONTRACTS,
    DASHBOARD_REGION_BOUNDS,
    RAIN_MM_MIN,
    TEMPERATURE_C_MAX,
    TEMPERATURE_C_MIN,
    DatasetContract,
)
from src.quality.validation_report import QualityReport

DEFAULT_REPORT_PATH = Path("data/quality/reports/pipeline_quality_report.json")
DEFAULT_SUMMARY_PATH = Path("data/quality/reports/latest_quality_summary.json")


def validate_dataset_contract(
    report: QualityReport,
    contract: DatasetContract,
) -> None:
    """
    Validate basic file, readability, emptiness, and required schema checks
    for one configured output dataset.
    """
    if not check_file_exists(report, contract.path, contract.name):
        return

    df = read_parquet_safely(report, contract.path, contract.name)
    if df is None:
        return

    has_required_columns = check_required_columns(
        report=report,
        df=df,
        required_columns=contract.required_columns,
        dataset_name=contract.name,
    )

    has_rows = check_not_empty(
        report=report,
        df=df,
        dataset_name=contract.name,
        allow_empty=contract.allow_empty,
    )

    if not has_required_columns or not has_rows:
        return

    validate_dataset_specific_rules(report, contract.name, df)


def validate_dataset_specific_rules(
    report: QualityReport,
    dataset_name: str,
    df: pd.DataFrame,
) -> None:
    """
    Run dataset-specific technical and lightweight semantic checks.

    These checks are intentionally limited to safe data quality rules:
    schema consistency, timestamp parsing, numeric ranges, allowed values,
    coordinate bounds, and basic dashboard-output sanity checks.
    """
    if dataset_name == "gold_route_window":
        check_parseable_datetime(report, df, "window_start", dataset_name)
        check_parseable_datetime(report, df, "window_end", dataset_name)
        check_datetime_order(
            report,
            df,
            start_column="window_start",
            end_column="window_end",
            dataset_name=dataset_name,
        )
        check_unique_key(
            report,
            df,
            key_columns=("route_id", "window_start", "window_end"),
            dataset_name=dataset_name,
        )
        check_numeric_range(report, df, "avg_occupancy_pct", dataset_name, 0, 100)
        check_numeric_range(report, df, "late_rate_delay", dataset_name, 0, 1)
        check_numeric_range(report, df, "avg_ingest_delay_sec", dataset_name, 0)
        check_numeric_range(report, df, "n_events_delay", dataset_name, 0)
        check_numeric_range(report, df, "n_events_occupancy", dataset_name, 0)
        check_allowed_values(
            report,
            df,
            "dq_flag",
            dataset_name,
            ALLOWED_WINDOW_DQ_FLAGS,
        )

    elif dataset_name == "gold_route_daily":
        check_parseable_datetime(report, df, "date", dataset_name)
        check_unique_key(
            report,
            df,
            key_columns=("route_id", "date"),
            dataset_name=dataset_name,
        )
        check_numeric_range(report, df, "avg_occupancy_pct", dataset_name, 0, 100)
        check_numeric_range(report, df, "avg_late_rate_delay", dataset_name, 0, 1)
        check_numeric_range(report, df, "avg_ingest_delay_sec", dataset_name, 0)
        check_numeric_range(report, df, "total_events_delay", dataset_name, 0)
        check_numeric_range(report, df, "total_events_occupancy", dataset_name, 0)
        check_allowed_values(
            report,
            df,
            "dq_flag",
            dataset_name,
            ALLOWED_DAILY_DQ_FLAGS,
        )

    elif dataset_name == "pipeline_metrics":
        check_parseable_datetime(report, df, "window_start", dataset_name)
        check_parseable_datetime(report, df, "window_end", dataset_name)
        check_datetime_order(
            report,
            df,
            start_column="window_start",
            end_column="window_end",
            dataset_name=dataset_name,
        )
        check_unique_key(
            report,
            df,
            key_columns=("window_start", "window_end"),
            dataset_name=dataset_name,
            required=False,
        )
        check_numeric_range(
            report,
            df,
            "transit_total_events",
            dataset_name,
            0,
            allow_null=True,
        )
        check_numeric_range(
            report,
            df,
            "weather_total_events",
            dataset_name,
            0,
            allow_null=True,
        )
        check_numeric_range(
            report,
            df,
            "transit_avg_ingest_delay_sec",
            dataset_name,
            0,
            allow_null=True,
        )
        check_numeric_range(
            report,
            df,
            "weather_avg_ingest_delay_sec",
            dataset_name,
            0,
            allow_null=True,
        )

    elif dataset_name == "hsl_map_points":
        check_coordinate_bounds(
            report,
            df,
            lat_column="lat",
            lon_column="lon",
            dataset_name=dataset_name,
            bounds=DASHBOARD_REGION_BOUNDS,
            required=False,
        )

        if "transport_mode" in df.columns:
            check_allowed_values(
                report,
                df,
                "transport_mode",
                dataset_name,
                ALLOWED_TRANSPORT_MODES,
                required=False,
            )

    elif dataset_name in {
        "hsl_route_options",
        "hsl_route_paths",
        "hsl_route_paths_overview",
    }:
        if "transport_mode" in df.columns:
            check_allowed_values(
                report,
                df,
                "transport_mode",
                dataset_name,
                ALLOWED_TRANSPORT_MODES,
                required=False,
            )

        if "route_label" in df.columns:
            check_no_null_like_strings(report, df, "route_label", dataset_name)

        if "route_short_name" in df.columns:
            check_no_null_like_strings(report, df, "route_short_name", dataset_name)

    elif dataset_name == "weather_stations_latest":
        check_coordinate_bounds(
            report,
            df,
            lat_column="lat",
            lon_column="lon",
            dataset_name=dataset_name,
            bounds=DASHBOARD_REGION_BOUNDS,
            required=True,
        )
        check_parseable_datetime(report, df, "observation_time", dataset_name)
        check_numeric_range(
            report,
            df,
            "temperature",
            dataset_name,
            TEMPERATURE_C_MIN,
            TEMPERATURE_C_MAX,
        )
        check_numeric_range(
            report,
            df,
            "precipitation",
            dataset_name,
            RAIN_MM_MIN,
        )


def run_pipeline_quality_checks(
    report_path: Path = DEFAULT_REPORT_PATH,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
) -> QualityReport:
    """
    Run portfolio-scale quality checks against existing pipeline outputs.

    This function does not run the pipeline, fetch live data, upload files,
    or modify dashboard outputs. It only validates already generated datasets.
    """

    report = QualityReport(source="pipeline_outputs")

    # Store high-level context about this quality check run.
    # These metadata fields help readers understand the report scope without
    # inspecting the implementation details.
    report.add_metadata("validation_scope", "pipeline_output_quality")
    report.add_metadata("connected_to_dashboard", False)
    report.add_metadata("runs_pipeline", False)
    report.add_metadata("checks_real_apis", False)

    for contract in ALL_DATASET_CONTRACTS:
        validate_dataset_contract(report, contract)

    report.save(report_path)
    report.save_summary(summary_path)

    return report

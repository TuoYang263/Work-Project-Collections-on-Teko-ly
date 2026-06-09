from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.quality.check_utils import (
    check_allowed_values,
    check_coordinate_bounds,
    check_file_exists,
    check_not_empty,
    check_parseable_datetime,
    check_required_columns,
)
from src.quality.contracts import (
    ALLOWED_TRANSPORT_MODES,
    REAL_SOURCE_COMPATIBILITY_BOUNDS,
)
from src.quality.validation_report import QualityReport

DEFAULT_INPUT_PATH = Path("data/source_samples/hsl_vehicle_snapshot.parquet")
DEFAULT_REPORT_PATH = Path("data/quality/reports/hsl_source_validation_report.json")
DEFAULT_SUMMARY_PATH = Path("data/quality/reports/latest_hsl_source_summary.json")

HSL_REQUIRED_COLUMNS = (
    "vehicle_id",
    "route_id",
    "transport_mode",
    "lat",
    "lon",
    "timestamp",
)


def read_snapshot(path: Path) -> pd.DataFrame:
    """
    Read a local HSL source snapshot from parquet, JSON or JSON Lines.

    This script validates a local compatibility snapshot instead of calling
    the live HSL API directly. This keeps source validation reproducible and
    separate from the dashboard serving path.
    """
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        if isinstance(payload, list):
            return pd.DataFrame(payload)

        if isinstance(payload, dict):
            if "records" in payload and isinstance(payload["records"], list):
                return pd.DataFrame(payload["records"])
            if "data" in payload and isinstance(payload["data"], list):
                return pd.DataFrame(payload["data"])
            return pd.DataFrame([payload])

    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)

    raise ValueError(
        f"Unsupported snapshot format: {suffix}. "
        "Supported formats: .parquet, .json, .jsonl"
    )


def run_hsl_source_validation(
    input_path: Path = DEFAULT_INPUT_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
) -> QualityReport:
    """
    Run optional compatibility checks for HSL source snapshots.

    This script does not feed the dashboard or modify pipeline outputs.
    It validates whether a source snapshot can be interpreted by the
    project's telemetry event model.
    """
    report = QualityReport(source="hsl_source_snapshot")

    report.add_metadata("validation_scope", "real_source_compatibility")
    report.add_metadata("source_system", "HSL")
    report.add_metadata("input_path", str(input_path))
    report.add_metadata("connected_to_dashboard", False)
    report.add_metadata("runs_pipeline", False)
    report.add_metadata("modifies_pipeline_outputs", False)
    report.add_metadata("calls_live_api", False)

    dataset_name = "hsl_source_snapshot"

    if not check_file_exists(report, input_path, dataset_name):
        report.save(report_path)
        report.save_summary(summary_path)
        return report

    try:
        df = read_snapshot(input_path)
    except Exception as exc:
        report.add_check(
            name=f"{dataset_name}_readable",
            status="failed",
            severity="critical",
            details=f"Could not read source snapshot: {type(exc).__name__}: {exc}",
        )
        report.save(report_path)
        report.save_summary(summary_path)
        return report

    report.add_check(
        name=f"{dataset_name}_readable",
        status="passed",
        severity="critical",
        details=(
            f"Source snapshot is readable with {len(df)} rows "
            f"and {len(df.columns)} columns."
        ),
    )
    report.set_record_count(dataset_name, len(df))

    has_required_columns = check_required_columns(
        report=report,
        df=df,
        required_columns=HSL_REQUIRED_COLUMNS,
        dataset_name=dataset_name,
    )

    has_rows = check_not_empty(
        report=report,
        df=df,
        dataset_name=dataset_name,
    )

    if has_required_columns and has_rows:
        check_parseable_datetime(report, df, "timestamp", dataset_name)

        check_coordinate_bounds(
            report,
            df,
            lat_column="lat",
            lon_column="lon",
            dataset_name=dataset_name,
            bounds=REAL_SOURCE_COMPATIBILITY_BOUNDS,
            required=True,
        )

        check_allowed_values(
            report,
            df,
            "transport_mode",
            dataset_name,
            ALLOWED_TRANSPORT_MODES,
            required=False,
        )

    report.save(report_path)
    report.save_summary(summary_path)

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate optional HSL source snapshot compatibility."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=(
            "Path to a local HSL source snapshot file. "
            "Supported formats: parquet, json, jsonl."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path for the full HSL source validation report JSON.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Path for the compact HSL source validation summary JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_hsl_source_validation(
        input_path=args.input,
        report_path=args.output,
        summary_path=args.summary_output,
    )

    summary = report.summary()

    print("[data-quality] HSL source validation completed")
    print(f"[data-quality] Input snapshot: {args.input}")
    print(f"[data-quality] Status: {summary['status']}")
    print(f"[data-quality] Total checks: {summary['total_checks']}")
    print(f"[data-quality] Passed: {summary['passed']}")
    print(f"[data-quality] Warnings: {summary['warnings']}")
    print(f"[data-quality] Failed: {summary['failed']}")
    print(f"[data-quality] Full report written to: {args.output}")
    print(f"[data-quality] Summary written to: {args.summary_output}")

    return report.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())

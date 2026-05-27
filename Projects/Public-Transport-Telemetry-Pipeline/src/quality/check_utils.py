from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.quality.contracts import CoordinateBounds
from src.quality.validation_report import QualityReport

NULL_LIKE_STRINGS = {"", "nan", "none", "null"}


def check_file_exists(report: QualityReport, path: Path, dataset_name: str) -> bool:
    if path.exists():
        report.add_check(
            name=f"{dataset_name}_file_exists",
            status="passed",
            severity="critical",
            details=f"Found {path}",
        )
        return True

    report.add_check(
        name=f"{dataset_name}_file_exists",
        status="failed",
        severity="critical",
        details=f"File not found: {path}",
    )
    return False


def read_parquet_safely(
    report: QualityReport,
    path: Path,
    dataset_name: str,
) -> pd.DataFrame | None:
    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        report.add_check(
            name=f"{dataset_name}_readable",
            status="failed",
            severity="critical",
            details=f"Could not read parquet file: {type(exc).__name__}: {exc}",
        )
        return None

    report.add_check(
        name=f"{dataset_name}_readable",
        status="passed",
        severity="critical",
        details=f"Parquet file is readable with {len(df)} rows and {len(df.columns)} columns.",
    )
    report.set_record_count(dataset_name, len(df))
    return df


def check_required_columns(
    report: QualityReport,
    df: pd.DataFrame,
    required_columns: Iterable[str],
    dataset_name: str,
) -> bool:
    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        report.add_check(
            name=f"{dataset_name}_required_columns",
            status="failed",
            severity="critical",
            details=f"Missing required columns: {missing}",
        )
        return False

    report.add_check(
        name=f"{dataset_name}_required_columns",
        status="passed",
        severity="critical",
        details="All required columns are present.",
    )
    return True


def check_not_empty(
    report: QualityReport,
    df: pd.DataFrame,
    dataset_name: str,
    allow_empty: bool = False,
) -> bool:
    if len(df) > 0:
        report.add_check(
            name=f"{dataset_name}_not_empty",
            status="passed",
            severity="critical",
            details=f"Dataset contains {len(df)} rows.",
        )
        return True

    if allow_empty:
        report.add_check(
            name=f"{dataset_name}_not_empty",
            status="warning",
            severity="warning",
            details="Dataset is empty. This can be acceptable for optional map/context outputs.",
        )
        return False

    report.add_check(
        name=f"{dataset_name}_not_empty",
        status="failed",
        severity="critical",
        details="Dataset is empty.",
    )
    return False


def check_parseable_datetime(
    report: QualityReport,
    df: pd.DataFrame,
    column: str,
    dataset_name: str,
    required: bool = True,
) -> bool:
    if column not in df.columns:
        status = "failed" if required else "warning"
        severity = "critical" if required else "warning"
        report.add_check(
            name=f"{dataset_name}_{column}_parseable",
            status=status,
            severity=severity,
            details=f"Column {column} is missing.",
        )
        return False

    parsed = pd.to_datetime(df[column], errors="coerce")
    invalid_count = int(parsed.isna().sum())

    if invalid_count == 0:
        report.add_check(
            name=f"{dataset_name}_{column}_parseable",
            status="passed",
            severity="critical" if required else "warning",
            details=f"All values in {column} are parseable as datetime.",
        )
        return True

    status = "failed" if required else "warning"
    severity = "critical" if required else "warning"
    report.add_check(
        name=f"{dataset_name}_{column}_parseable",
        status=status,
        severity=severity,
        details=f"Found {invalid_count} unparseable values in {column}.",
    )
    return False


def check_numeric_range(
    report: QualityReport,
    df: pd.DataFrame,
    column: str,
    dataset_name: str,
    min_value: float | None = None,
    max_value: float | None = None,
    required: bool = True,
) -> bool:
    if column not in df.columns:
        status = "failed" if required else "warning"
        severity = "critical" if required else "warning"
        report.add_check(
            name=f"{dataset_name}_{column}_numeric_range",
            status=status,
            severity=severity,
            details=f"Column {column} is missing.",
        )
        return False

    values = pd.to_numeric(df[column], errors="coerce")
    invalid_numeric = int(values.isna().sum())
    out_of_range = pd.Series(False, index=df.index)

    if min_value is not None:
        out_of_range = out_of_range | (values < min_value)
    if max_value is not None:
        out_of_range = out_of_range | (values > max_value)

    out_of_range_count = int(out_of_range.fillna(False).sum())

    if invalid_numeric == 0 and out_of_range_count == 0:
        report.add_check(
            name=f"{dataset_name}_{column}_numeric_range",
            status="passed",
            severity="critical" if required else "warning",
            details=f"Column {column} is numeric and within expected range.",
        )
        return True

    status = "failed" if required else "warning"
    severity = "critical" if required else "warning"
    report.add_check(
        name=f"{dataset_name}_{column}_numeric_range",
        status=status,
        severity=severity,
        details=(
            f"Column {column} has {invalid_numeric} non-numeric values and "
            f"{out_of_range_count} values outside expected range."
        ),
    )
    return False


def check_allowed_values(
    report: QualityReport,
    df: pd.DataFrame,
    column: str,
    dataset_name: str,
    allowed_values: Iterable[str],
    required: bool = True,
) -> bool:
    if column not in df.columns:
        status = "failed" if required else "warning"
        severity = "critical" if required else "warning"
        report.add_check(
            name=f"{dataset_name}_{column}_allowed_values",
            status=status,
            severity=severity,
            details=f"Column {column} is missing.",
        )
        return False

    allowed = set(allowed_values)
    observed = set(df[column].dropna().astype(str).unique())
    unexpected = sorted(observed - allowed)

    if not unexpected:
        report.add_check(
            name=f"{dataset_name}_{column}_allowed_values",
            status="passed",
            severity="critical" if required else "warning",
            details=f"Column {column} only contains expected values.",
        )
        return True

    status = "failed" if required else "warning"
    severity = "critical" if required else "warning"
    report.add_check(
        name=f"{dataset_name}_{column}_allowed_values",
        status=status,
        severity=severity,
        details=f"Unexpected values in {column}: {unexpected}",
    )
    return False


def check_no_null_like_strings(
    report: QualityReport,
    df: pd.DataFrame,
    column: str,
    dataset_name: str,
) -> bool:
    if column not in df.columns:
        report.add_check(
            name=f"{dataset_name}_{column}_null_like_strings",
            status="warning",
            severity="warning",
            details=f"Column {column} is missing; null-like string check skipped.",
        )
        return False

    normalized = df[column].fillna("").astype(str).str.strip().str.lower()
    bad_count = int(normalized.isin(NULL_LIKE_STRINGS).sum())

    if bad_count == 0:
        report.add_check(
            name=f"{dataset_name}_{column}_null_like_strings",
            status="passed",
            severity="warning",
            details=f"No null-like strings found in {column}.",
        )
        return True

    report.add_check(
        name=f"{dataset_name}_{column}_null_like_strings",
        status="warning",
        severity="warning",
        details=f"Found {bad_count} null-like string values in {column}.",
    )
    return False


def check_coordinate_bounds(
    report: QualityReport,
    df: pd.DataFrame,
    lat_column: str,
    lon_column: str,
    dataset_name: str,
    bounds: CoordinateBounds,
    required: bool = True,
) -> bool:
    if lat_column not in df.columns or lon_column not in df.columns:
        status = "failed" if required else "warning"
        severity = "critical" if required else "warning"
        report.add_check(
            name=f"{dataset_name}_coordinate_bounds",
            status=status,
            severity=severity,
            details=f"Missing coordinate columns: {lat_column}, {lon_column}",
        )
        return False

    lat = pd.to_numeric(df[lat_column], errors="coerce")
    lon = pd.to_numeric(df[lon_column], errors="coerce")

    invalid = lat.isna() | lon.isna()

    inside_bounds = (
        (lat >= bounds.min_lat)
        & (lat <= bounds.max_lat)
        & (lon >= bounds.min_lon)
        & (lon <= bounds.max_lon)
    )

    outside = inside_bounds.eq(False) & invalid.eq(False)

    invalid_count = int(invalid.sum())
    outside_count = int(outside.sum())

    if invalid_count == 0 and outside_count == 0:
        report.add_check(
            name=f"{dataset_name}_coordinate_bounds",
            status="passed",
            severity="critical" if required else "warning",
            details="Coordinates are numeric and within configured coordinate bounds.",
        )
        return True

    status = "failed" if required else "warning"
    severity = "critical" if required else "warning"
    report.add_check(
        name=f"{dataset_name}_coordinate_bounds",
        status=status,
        severity=severity,
        details=(
            f"Found {invalid_count} invalid coordinate pairs and "
            f"{outside_count} pairs outside configured coordinate bounds."
        ),
    )
    return False

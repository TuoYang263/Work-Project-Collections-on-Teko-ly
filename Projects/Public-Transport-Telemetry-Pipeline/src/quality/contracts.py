from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.pipeline.config import DATA_DIR, EXPORT_DIR


# Frozen dataclass is used because coordinate bounds are static configuration.
# Once created, the bounds should not be modified at runtime.
@dataclass(frozen=True)
class CoordinateBounds:
    """
    Broad coordinate bounds for defensive validation.

    These bounds are not used for routing, geofencing, or operational decisions.
    They only help catch clearly invalid coordinates in quality checks.
    """

    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    def contains(self, lat: float, lon: float) -> bool:
        return (
            self.min_lat <= lat <= self.max_lat and self.min_lon <= lon <= self.max_lon
        )


# Broad Helsinki-region bounds for dashboard-level quality checks.
# The project focuses on scheduled snapshot visualization around the Helsinki
# region, not full VR or long-distance commuter rail coverage.
# These bounds are only used to catch clearly broken coordinates in dashboard
# outputs and portfolio-scale quality checks.
DASHBOARD_REGION_BOUNDS = CoordinateBounds(
    min_lat=59.8,
    max_lat=60.6,
    min_lon=24.0,
    max_lon=25.5,
)


# Broader bounds for optional real-source compatibility checks.
# Real HSL feeds or reference data may include records outside the dashboard
# scope. In that case, records can still be valid source data, but they should
# not automatically expand the main dashboard scope.
REAL_SOURCE_COMPATIBILITY_BOUNDS = CoordinateBounds(
    min_lat=59.6,
    max_lat=61.2,
    min_lon=23.0,
    max_lon=26.2,
)


@dataclass(frozen=True)
class DatasetContract:
    """
    Expected contract for one output dataset.

    required_columns should be treated as critical schema requirements.
    optional_columns are useful for documentation and future checks, but should
    not fail the dataset if they are missing.
    """

    name: str
    path: Path
    required_columns: tuple[str, ...]
    optional_columns: tuple[str, ...] = ()
    allow_empty: bool = False


ALLOWED_WINDOW_DQ_FLAGS = (
    "OK",
    "CLOCK_SKEW",
    "LOW_VOLUME",
    "HIGH_LATE_RATE",
)

ALLOWED_DAILY_DQ_FLAGS = (
    "OK",
    "CHECK",
)

ALLOWED_TRANSPORT_MODES = ("bus", "tram", "metro", "rail", "ferry")

TEMPERATURE_C_MIN = -50.0
TEMPERATURE_C_MAX = 50.0
RAIN_MM_MIN = 0.0


PIPELINE_OUTPUT_CONTRACTS: tuple[DatasetContract, ...] = (
    DatasetContract(
        name="gold_route_window",
        path=EXPORT_DIR / "gold_route_window.parquet",
        required_columns=(
            "window_start",
            "window_end",
            "route_id",
            "avg_delay_sec",
            "avg_occupancy_pct",
            "n_events_delay",
            "n_events_occupancy",
            "late_rate_delay",
            "avg_ingest_delay_sec",
            "dq_flag",
        ),
    ),
    DatasetContract(
        name="gold_route_daily",
        path=EXPORT_DIR / "gold_route_daily.parquet",
        required_columns=(
            "date",
            "route_id",
            "avg_delay_sec",
            "avg_occupancy_pct",
            "total_events_delay",
            "total_events_occupancy",
            "avg_late_rate_delay",
            "avg_ingest_delay_sec",
            "dq_flag",
        ),
    ),
    DatasetContract(
        name="pipeline_metrics",
        path=EXPORT_DIR / "pipeline_metrics.parquet",
        required_columns=(
            "window_start",
            "window_end",
            "transit_total_events",
            "weather_total_events",
            "transit_avg_ingest_delay_sec",
            "weather_avg_ingest_delay_sec",
        ),
    ),
)


MAP_OUTPUT_CONTRACTS: tuple[DatasetContract, ...] = (
    DatasetContract(
        name="hsl_map_points",
        path=DATA_DIR / "gold" / "hsl" / "hsl_map_points.parquet",
        required_columns=("lat", "lon"),
        optional_columns=(
            "vehicle_id",
            "route_short_name",
            "route_label",
            "transport_mode",
        ),
        allow_empty=True,
    ),
    DatasetContract(
        name="hsl_route_options",
        path=DATA_DIR / "gold" / "hsl" / "hsl_route_options.parquet",
        required_columns=("route_label", "route_short_name", "transport_mode"),
    ),
    DatasetContract(
        name="hsl_route_paths",
        path=DATA_DIR / "gold" / "hsl" / "hsl_route_paths.parquet",
        required_columns=("route_short_name", "transport_mode", "path"),
        allow_empty=True,
    ),
    DatasetContract(
        name="hsl_route_paths_overview",
        path=DATA_DIR / "gold" / "hsl" / "hsl_route_paths_overview.parquet",
        required_columns=("route_short_name", "transport_mode", "path"),
        allow_empty=True,
    ),
    DatasetContract(
        name="weather_stations_latest",
        path=DATA_DIR / "gold" / "weather" / "weather_stations_latest.parquet",
        required_columns=(
            "station_id",
            "station_name",
            "lat",
            "lon",
            "observation_time",
            "temperature",
            "precipitation",
        ),
        allow_empty=True,
    ),
)


@dataclass(frozen=True)
class RealSourceContract:
    """
    Logical event contract for optional real-source validation.

    These contracts are used to check whether real external snapshots can be
    mapped to the project model. They do not change the main controlled-data
    dashboard path.
    """

    source_name: str
    required_event_fields: tuple[str, ...]
    recommended_event_fields: tuple[str, ...]
    target_model_fields: tuple[str, ...]


HSL_REAL_SOURCE_CONTRACT = RealSourceContract(
    source_name="hsl_vehicle_snapshot",
    required_event_fields=("vehicle_id", "event_time", "ingest_time", "lat", "lon"),
    recommended_event_fields=("route_id", "line_id", "direction_id", "speed"),
    target_model_fields=(
        "source",
        "vehicle_id",
        "route_id",
        "line_id",
        "event_time",
        "ingest_time",
        "lat",
        "lon",
        "speed",
        "direction_id",
    ),
)


FMI_REAL_SOURCE_CONTRACT = RealSourceContract(
    source_name="fmi_weather_observation",
    required_event_fields=(
        "station_name",
        "observation_time",
        "ingest_time",
        "lat",
        "lon",
    ),
    recommended_event_fields=("temperature", "precipitation"),
    target_model_fields=(
        "source",
        "station_name",
        "observation_time",
        "ingest_time",
        "lat",
        "lon",
        "temperature",
        "precipitation",
    ),
)


ALL_DATASET_CONTRACTS: tuple[DatasetContract, ...] = (
    *PIPELINE_OUTPUT_CONTRACTS,
    *MAP_OUTPUT_CONTRACTS,
)

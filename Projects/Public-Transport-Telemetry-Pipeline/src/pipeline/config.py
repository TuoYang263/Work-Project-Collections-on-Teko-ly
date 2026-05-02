"""
Central configuration for the Public Transport Telemetry Pipeline.

This module keeps pipeline-wide settings in one place so that
notebooks, scripts, and future deployment targets can share the same
configuration.
"""

from pathlib import Path
import os

# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
EXPORT_DIR = DATA_DIR / "output"

LOGS_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOGS_DIR / "pipeline.log"

REQUIRED_DIRS = [
    DATA_DIR,
    RAW_DIR,
    BRONZE_DIR,
    SILVER_DIR,
    GOLD_DIR,
    EXPORT_DIR,
    LOGS_DIR,
]

# -----------------------------------------------------------------------------
# Runtime / storage settings
# -----------------------------------------------------------------------------

STORAGE_MODE = os.getenv("STORAGE_MODE", "local")  # allowed: "local", "azure_blob"
DATABASE_NAME = os.getenv("DATABASE_NAME", "azure_streaming_mvp")

# Future Azure placeholder
AZURE_GOLD_BASE_PATH = os.getenv("AZURE_GOLD_BASE_PATH")

# -----------------------------------------------------------------------------
# Logical dataset / table names
# -----------------------------------------------------------------------------

BRONZE_EVENTS_TABLE = "bronze_events"

SILVER_TRANSIT_TABLE = "silver_transit_metrics"
SILVER_WEATHER_TABLE = "silver_weather_metrics"

GOLD_ROUTE_WINDOW_TABLE = "gold_route_kpi_window"
GOLD_ROUTE_DAILY_TABLE = "gold_route_kpi_daily"
GOLD_PIPELINE_METRICS_TABLE = "gold_pipeline_metrics_window"

# -----------------------------------------------------------------------------
# Export outputs (for downstream apps / dashboards)
# These files are produced by export_gold.py and consumed by
# Streamlit dashboards or Github Release assets.
# -----------------------------------------------------------------------------
GOLD_ROUTE_WINDOW_EXPORT_PATH = EXPORT_DIR / "gold_route_window.parquet"
GOLD_ROUTE_DAILY_EXPORT_PATH = EXPORT_DIR / "gold_route_daily.parquet"
GOLD_PIPELINE_METRICS_EXPORT_PATH = EXPORT_DIR / "pipeline_metrics.parquet"

# -----------------------------------------------------------------------------
# Local file outputs
# -----------------------------------------------------------------------------

BRONZE_EVENTS_PATH = BRONZE_DIR / "bronze_events.csv"

SILVER_TRANSIT_PATH = SILVER_DIR / "silver_transit_metrics.csv"
SILVER_WEATHER_PATH = SILVER_DIR / "silver_weather_metrics.csv"

GOLD_ROUTE_WINDOW_PATH = GOLD_DIR / "gold_route_kpi_window.csv"
GOLD_ROUTE_DAILY_PATH = GOLD_DIR / "gold_route_kpi_daily.csv"
GOLD_PIPELINE_METRICS_PATH = GOLD_DIR / "gold_pipeline_metrics_window.csv"

# -----------------------------------------------------------------------------
# Transit processing parameters
# -----------------------------------------------------------------------------

TRANSIT_WINDOW = "10 minutes"
TRANSIT_LOOKBACK_MINUTES = 180
TRANSIT_LATE_THRESHOLD_SEC = 120

# -----------------------------------------------------------------------------
# Weather processing parameters
# -----------------------------------------------------------------------------

WEATHER_WINDOW = "15 minutes"
WEATHER_LOOKBACK_MINUTES = 360

# -----------------------------------------------------------------------------
# FMI source defaults
# -----------------------------------------------------------------------------

FMI_DEFAULT_PLACE = "helsinki"

FMI_DEFAULT_PLACES = os.getenv(
    "FMI_DEFAULT_PLACES",
    "helsinki,espoo,vantaa,kauniainen",
)

FMI_PLACES = [
    place.strip()
    for place in FMI_DEFAULT_PLACES.split(",")
    if place.strip()
]

FMI_DEFAULT_PARAMS = "t2m,r_1h"
FMI_DEFAULT_LOOKBACK_MINUTES = 360
FMI_WFS_URL = "https://opendata.fmi.fi/wfs"

FMI_REQUEST_TIMEOUT_CONNECT = 15
FMI_REQUEST_TIMEOUT_READ = 60

FMI_MAX_RETRIES = 3
FMI_BACKOFF_FACTOR = 2

FMI_ALLOW_FAILURE = True

# -----------------------------------------------------------------------------
# Simulation parameters
# -----------------------------------------------------------------------------

SIM_ROUTE_IDS = ["M1", "M2", "T1", "R10", "B1", "B2", "X3", "X7"]

# Demo history settings
SIM_HISTORY_WINDOWS = int(os.getenv("SIM_HISTORY_WINDOWS", "12"))
SIM_WINDOW_MINUTES = int(os.getenv("SIM_WINDOW_MINUTES", "10"))
SIM_EVENTS_PER_ROUTE_WINDOW = int(os.getenv("SIM_EVENTS_PER_ROUTE_WINDOW", "10"))

# Batch size is kept moderate so local and GitHub Actions runs stay lightweight
SIM_DEFAULT_BATCH_SIZE = int(os.getenv("SIM_DEFAULT_BATCH_SIZE", "1000"))

# Simulated event-to-ingest delay range.
SIM_INGEST_DELAY_MIN_SEC = int(os.getenv("SIM_INGEST_DELAY_MIN_SEC", "20"))
SIM_INGEST_DELAY_MAX_SEC = int(os.getenv("SIM_INGEST_DELAY_MAX_SEC", "240"))

# -----------------------------------------------------------------------------
# Spark local runtime settings (WSL / local development)
# -----------------------------------------------------------------------------
SPARK_WAREHOUSE_DIR = os.getenv(
    "SPARK_WAREHOUSE_DIR", "file:/tmp/spark-warehouse/telemetry"
)

SPARK_LOCAL_DIR = os.getenv("SPARK_LOCAL_DIR", "/tmp/spark-tmp")

SPARK_DATABASE_LOCATION = os.getenv(
    "SPARK_DATABASE_LOCATION", f"{SPARK_WAREHOUSE_DIR}/{DATABASE_NAME}.db"
)

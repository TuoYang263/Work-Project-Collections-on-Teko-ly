"""
Environment and schema setup for the telemetry pipeline.
"""

import logging
import os
import shutil
from pathlib import Path
from pyspark.sql import SparkSession

from .config import (
    REQUIRED_DIRS,
    LOG_FILE,
    DATABASE_NAME,
    BRONZE_EVENTS_TABLE,
    SILVER_TRANSIT_TABLE,
    SILVER_WEATHER_TABLE,
    GOLD_ROUTE_WINDOW_TABLE,
    GOLD_ROUTE_DAILY_TABLE,
    GOLD_PIPELINE_METRICS_TABLE,
    SPARK_DATABASE_LOCATION,
    SPARK_WAREHOUSE_DIR,
    SPARK_LOCAL_DIR,
)


def build_spark(app_name: str = "telemetry_pipeline") -> SparkSession:
    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sql.warehouse.dir", SPARK_WAREHOUSE_DIR)
        .config("spark.local.dir", SPARK_LOCAL_DIR)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.default.parallelism", "8")
        .enableHiveSupport()
    )

    # Databricks Runtime already includes Spark and Delta.
    # Local / Github Actions execution still uses delta-spark from pip.
    if os.getenv("DATABRICKS_RUNTIME_VERSION"):
        return builder.getOrCreate()

    from delta import configure_spark_with_delta_pip

    return configure_spark_with_delta_pip(builder).getOrCreate()


def _file_uri_to_path(file_uri: str) -> Path:
    """Convert a file: URI to a local Path."""
    if file_uri.startswith("file:"):
        return Path(file_uri.replace("file:", "", 1))
    return Path(file_uri)


def cleanup_database_storage() -> None:
    """
    Remove the local warehouse directory for the configured Spark database.

    This is useful in local environment when managed table metadata is dropped
    but non-Delta files remain in the table directory.
    """
    db_path = _file_uri_to_path(SPARK_DATABASE_LOCATION)

    if db_path.exists():
        shutil.rmtree(db_path)


def ensure_directories() -> None:
    """Create required project directories if they do not exist."""
    for directory in REQUIRED_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def setup_logging(name: str = "telemetry_pipeline") -> logging.Logger:
    """Configure and return a logger for the pipeline."""

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    # Ensure log directory exists
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    # File handler
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False

    logger.info("Logger initialized")

    return logger


def use_database(spark: SparkSession) -> None:
    """Select the configured database."""
    spark.sql(f"""
              CREATE DATABASE IF NOT EXISTS {DATABASE_NAME}
              LOCATION '{SPARK_DATABASE_LOCATION}'
              """)
    spark.sql(f"USE {DATABASE_NAME}")


def reset_tables(spark: SparkSession) -> None:
    """
    Drop pipeline tables so the MVP can be rebuilt from a clean slate.
    """
    tables = [
        BRONZE_EVENTS_TABLE,
        SILVER_TRANSIT_TABLE,
        SILVER_WEATHER_TABLE,
        GOLD_ROUTE_WINDOW_TABLE,
        GOLD_ROUTE_DAILY_TABLE,
        GOLD_PIPELINE_METRICS_TABLE,
    ]

    for table_name in tables:
        spark.sql(f"DROP TABLE IF EXISTS {table_name}")


def create_bronze_events_table(spark: SparkSession) -> None:
    """
    Create the Bronze table used for raw transit and weather events.
    """
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {BRONZE_EVENTS_TABLE} (
            event_id STRING,
            event_time_raw STRING,
            source STRING,
            entity_type STRING,
            entity_id STRING,
            metric STRING,
            value DOUBLE,
            unit STRING,
            attrs MAP<STRING, STRING>,
            event_time_ts TIMESTAMP,
            ingest_time_ts TIMESTAMP
        )
        USING DELTA
        """)


def initialize_environment(spark: SparkSession, reset: bool = False) -> None:
    """
    Prepare the runtime environment and create the base Bronze schema.

    Parameters
    ----------
    spark:
        Active Spark session.
    reset:
        If True, drop existing pipeline tables and remove local warehouse
        storage before recreating the base Bronze table.
    """
    ensure_directories()
    # Keep timestamp handling stable across stages
    spark.conf.set("spark.sql.session.timeZone", "UTC")

    print("DEBUG: before use_database", flush=True)
    use_database(spark)
    print("DEBUG: after use_database", flush=True)

    if reset:
        print("DEBUG: before reset_tables", flush=True)
        reset_tables(spark)
        print("DEBUG: after reset_tables", flush=True)

        print("DEBUG: before cleanup_database_storage", flush=True)
        cleanup_database_storage()
        print("DEBUG: after cleanup_database_storage", flush=True)

        print("DEBUG: before use_database after cleanup", flush=True)
        use_database(spark)
        print("DEBUG: after use_database after cleanup", flush=True)

    print("DEBUG: before create_bronze_events_table", flush=True)
    create_bronze_events_table(spark)
    print("DEBUG: after create_bronze_events_table", flush=True)

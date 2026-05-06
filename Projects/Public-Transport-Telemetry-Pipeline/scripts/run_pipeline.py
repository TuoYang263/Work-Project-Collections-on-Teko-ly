"""
Pipeline runner for the Public Transport Telemetry Pipeline

This script orchestrates bronze -> silver -> gold execution
and allows partial runs for development or debugging.
"""

import os
import time
import argparse
import logging

from src.pipeline.setup import ensure_directories, setup_logging, build_spark
from src.pipeline.bronze import run_bronze_layer
from src.pipeline.silver import run_silver_layer
from src.pipeline.gold import run_gold_layer
from src.pipeline.config import (
    SILVER_TRANSIT_TABLE,
    SILVER_WEATHER_TABLE,
    GOLD_ROUTE_WINDOW_TABLE,
    GOLD_ROUTE_DAILY_TABLE,
    GOLD_PIPELINE_METRICS_TABLE,
)


def drop_derived_tables(spark) -> None:
    """
    Drop Silver / Gold tables before a clean full rebuild.

    This avoids Hive metastore conflicts when overwritting existing
    Delta-backed tables with evolved schemas.
    """
    tables = [
        GOLD_PIPELINE_METRICS_TABLE,
        GOLD_ROUTE_DAILY_TABLE,
        GOLD_ROUTE_WINDOW_TABLE,
        SILVER_WEATHER_TABLE,
        SILVER_TRANSIT_TABLE,
    ]

    for table_name in tables:
        spark.sql(f"DROP TABLE IF EXISTS {table_name}")


def run_pipeline(layer: str, logger: logging.Logger) -> None:
    spark = build_spark("telemetry_pipeline")

    try:
        if layer == "bronze":
            logger.info("Running BRONZE layer")
            run_bronze_layer(spark, logger, reset=True)

        if layer == "silver":
            logger.info("Running SILVER layer")
            run_silver_layer(spark, logger)

        if layer == "gold":
            logger.info("Running GOLD layer")
            run_gold_layer(spark, logger)

        elif layer == "full":
            logger.info("Running BRONZE layer")
            run_bronze_layer(spark, logger, reset=True)

            logger.info("Dropping existing Silver / Gold tables for clean rebuild")
            drop_derived_tables(spark)

            logger.info("Running SILVER layer")
            run_silver_layer(spark, logger)

            logger.info("Running GOLD layer")
            run_gold_layer(spark, logger)

    finally:
        logger.info("Starting Spark session cleanup")

        if os.getenv("DATABRICKS_RUNTIME_VERSION"):
            logger.info(
                "Databricks runtime detected; skipping explicit spark.stop() "
                "and leaving Spark lifecycle to the job cluster."
            )
        else:
            logger.info("Stopping local Spark session")
            spark.stop()
            logger.info("Local Spark session stopped")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the telemetry data pipeline.")

    parser.add_argument(
        "--layer",
        type=str,
        default="full",
        choices=["bronze", "silver", "gold", "full"],
        help="Which layer of the pipeline to run",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    ensure_directories()
    logger = setup_logging()

    logger.info("Telemetry pipeline starting")
    logger.info(f"Selected layer: {args.layer}")

    try:
        start = time.time()
        run_pipeline(args.layer, logger)
        logger.info("Pipeline completed successfully")
        elapsed = time.time() - start
        logger.info(f"Pipeline completed successfully in {elapsed:.2f}s")
    except Exception:
        logger.exception("Pipeline failed")
        raise


if __name__ == "__main__":
    main()

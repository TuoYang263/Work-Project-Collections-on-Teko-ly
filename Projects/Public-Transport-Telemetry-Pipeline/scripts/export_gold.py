"""
Export Gold-layer Delta tables to lightweight parquet outputs.

These exported files are intended for downstream consumers such as:
- Streamlit dashboards
- Github Release assets
- lightweight local inspection
"""

from __future__ import annotations

import logging
from pathlib import Path

from pyspark.sql import SparkSession

from src.pipeline.setup import build_spark, setup_logging, ensure_directories, use_database
from src.pipeline.config import (
    DATABASE_NAME,
    GOLD_ROUTE_WINDOW_TABLE,
    GOLD_ROUTE_DAILY_TABLE,
    GOLD_PIPELINE_METRICS_TABLE,
    GOLD_ROUTE_WINDOW_EXPORT_PATH,
    GOLD_ROUTE_DAILY_EXPORT_PATH,
    GOLD_PIPELINE_METRICS_EXPORT_PATH,
)

def export_table_to_parquet(
    spark: SparkSession,
    table_name: str,
    output_path: Path,
    logger: logging.Logger
):
    """
    Read a managed Gold table and export it as parquet.

    Parameters
    ----------
    spark:
        Active Spark Session
    table_name:
        Spark table name in the configured database
    output_path:
        Local parquet output path.
    logger:
        Pipeline logger.
    """
    logger.info(f"Exporting table: {table_name} -> {output_path}")

    df = spark.table(table_name)

    row_count = df.count()
    logger.info(f"Row count for {table_name}: {row_count}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Spark writes parquet outputs as directories
    df.write.mode("overwrite").parquet(str(output_path))

    logger.info(f"Export completed: {table_name}")

def main() -> None:
    ensure_directories()
    logger = setup_logging("telemetry_export")

    logger.info("Gold export started")

    spark = build_spark("telemetry_export")

    try:
        use_database(spark)
        logger.info(f"Using database: {DATABASE_NAME}")

        export_table_to_parquet(
            spark=spark,
            table_name=GOLD_ROUTE_WINDOW_TABLE,
            output_path=GOLD_ROUTE_WINDOW_EXPORT_PATH,
            logger=logger,
        )
        
        export_table_to_parquet(
            spark=spark,
            table_name=GOLD_ROUTE_DAILY_TABLE,
            output_path=GOLD_ROUTE_DAILY_EXPORT_PATH,
            logger=logger,
        )

        export_table_to_parquet(
            spark=spark,
            table_name=GOLD_PIPELINE_METRICS_TABLE,
            output_path=GOLD_PIPELINE_METRICS_EXPORT_PATH,
            logger=logger,
        )

        logger.info("Gold export finished successfully")

    finally:
        spark.stop()

if __name__ == "__main__":
    main()
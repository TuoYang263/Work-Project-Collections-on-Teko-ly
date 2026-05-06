"""
Export Gold-layer Delta tables to lightweight parquet outputs.

These exported files are intended for downstream consumers such as:
- Streamlit dashboards
- Github Release assets
- lightweight local inspection
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

from pyspark.sql import SparkSession

from src.pipeline.setup import (
    build_spark,
    setup_logging,
    ensure_directories,
    use_database,
)
from src.pipeline.config import (
    DATABASE_NAME,
    GOLD_ROUTE_WINDOW_TABLE,
    GOLD_ROUTE_DAILY_TABLE,
    GOLD_PIPELINE_METRICS_TABLE,
    GOLD_ROUTE_WINDOW_PATH,
    GOLD_ROUTE_DAILY_PATH,
    GOLD_PIPELINE_METRICS_PATH,
    GOLD_ROUTE_WINDOW_EXPORT_PATH,
    GOLD_ROUTE_DAILY_EXPORT_PATH,
    GOLD_PIPELINE_METRICS_EXPORT_PATH,
)


def is_databricks() -> bool:
    return bool(os.getenv("DATABRICKS_RUNTIME_VERSION"))


def delta_path(path: Path) -> str:
    if is_databricks():
        return f"file:{path}"
    return str(path)


def spark_path(path: Path) -> str:
    if is_databricks():
        return f"file:{path}"
    return str(path)


def export_table_to_parquet(
    spark: SparkSession,
    table_name: str,
    delta_source_path: Path,
    output_path: Path,
    logger: logging.Logger,
):
    """
    Export a Gold-layer dataset to lightweight parquet output.

    Runtime behavior:
    - Local / GitHub Actions: read from the managed Spark table.
    - Azure Databricks Jobs: read from a path-based Delta source to avoid
      workspace metastore / Unity Catalog dependencies.

    Parameters
    ----------
    spark:
        Active Spark session.
    table_name:
        Managed Spark table name used in local execution.
    delta_source_path:
        Path-based Delta source used in Databricks execution.
    output_path:
        Parquet output directory for downstream consumers.
    logger:
        Pipeline logger.
    """
    logger.info(f"Exporting table: {table_name} -> {output_path}")

    if is_databricks():
        logger.info(f"Reading path-based Gold Delta source: {delta_source_path}")
        df = spark.read.format("delta").load(spark_path(delta_source_path))
    else:
        df = spark.table(table_name)

    row_count = df.count()
    logger.info(f"Row count for {table_name}: {row_count}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.write.mode("overwrite").parquet(spark_path(output_path))

    logger.info(f"Export completed: {table_name}")


def main() -> None:
    ensure_directories()
    logger = setup_logging("telemetry_export")

    logger.info("Gold export started")

    spark = build_spark("telemetry_export")

    try:
        if is_databricks():
            logger.info(
                "Databricks runtime detected; using path-based Gold Delta sources"
            )
        else:
            use_database(spark)
            logger.info(f"Using database: {DATABASE_NAME}")

        export_table_to_parquet(
            spark=spark,
            table_name=GOLD_ROUTE_WINDOW_TABLE,
            delta_source_path=GOLD_ROUTE_WINDOW_PATH,
            output_path=GOLD_ROUTE_WINDOW_EXPORT_PATH,
            logger=logger,
        )

        export_table_to_parquet(
            spark=spark,
            table_name=GOLD_ROUTE_DAILY_TABLE,
            delta_source_path=GOLD_ROUTE_DAILY_PATH,
            output_path=GOLD_ROUTE_DAILY_EXPORT_PATH,
            logger=logger,
        )

        export_table_to_parquet(
            spark=spark,
            table_name=GOLD_PIPELINE_METRICS_TABLE,
            delta_source_path=GOLD_PIPELINE_METRICS_PATH,
            output_path=GOLD_PIPELINE_METRICS_EXPORT_PATH,
            logger=logger,
        )

        logger.info("Gold export finished successfully")

    finally:
        if is_databricks():
            logger.info(
                "Databricks runtime detected; skipping explicit spark.stop() "
                "and leaving Spark lifecycle to the job cluster."
            )
        else:
            logger.info("Stopping local Spark session")
            spark.stop()
            logger.info("Local Spark session stopped")


if __name__ == "__main__":
    main()

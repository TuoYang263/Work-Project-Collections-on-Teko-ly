"""
Silver-layer transformations for transit and weather metrics.
"""
import logging
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from src.pipeline.setup import use_database

from .config import (
    BRONZE_EVENTS_TABLE,
    TRANSIT_LOOKBACK_MINUTES,
    TRANSIT_LATE_THRESHOLD_SEC,
    SILVER_TRANSIT_TABLE,
    SILVER_WEATHER_TABLE,
    TRANSIT_WINDOW,
    WEATHER_LOOKBACK_MINUTES,
    WEATHER_WINDOW,
)

def run_silver_layer(spark: SparkSession, logger: logging.Logger) -> None:
    logger.info("Silver layer started.")

    use_database(spark)

    build_silver_transit_metrics(spark)
    logger.info(f"Silver transit table updated: {SILVER_TRANSIT_TABLE}")

    build_silver_weather_metrics(spark)
    logger.info(f"Silver weather table updated: {SILVER_WEATHER_TABLE}")

    transit_count = spark.table(SILVER_TRANSIT_TABLE).count()
    weather_count = spark.table(SILVER_WEATHER_TABLE).count()

    logger.info(f"Silver transit row count: {transit_count}")
    logger.info(f"Silver weather row count: {weather_count}")

def build_silver_transit_metrics(spark: SparkSession) -> None:
    """
    Aggregate simulated transit events into windowed Silver metrics.
    """
    bronze_all = spark.table(BRONZE_EVENTS_TABLE)

    bronze_recent = (
        bronze_all
        .filter(F.col("source") == F.lit("sim_transit"))
        .filter(
            F.col("event_time_ts")
            >= F.expr(f"current_timestamp() - INTERVAL {TRANSIT_LOOKBACK_MINUTES} MINUTES")
        )
    )

    bronze_enriched = (
        bronze_recent
        .withColumn("route_id", F.col("attrs").getItem("route_id"))
        .withColumn(
            "ingest_delay_sec_raw",
            F.unix_timestamp("ingest_time_ts") - F.unix_timestamp("event_time_ts"),
        )
        .withColumn("is_clock_skew", F.col("ingest_delay_sec_raw") < F.lit(0))
        .withColumn("ingest_delay_sec", F.greatest(F.col("ingest_delay_sec_raw"), F.lit(0)))
        .withColumn("is_late_event",
                    F.when(
                        F.col("metric") == F.lit("delay_sec"),
                        F.col("value") > F.lit(TRANSIT_LATE_THRESHOLD_SEC),
                    ).otherwise(F.lit(False)),
        )
    )

    silver_transit = (
        bronze_enriched
        .groupBy(
            F.window("event_time_ts", TRANSIT_WINDOW).alias("window"),
            F.col("metric"),
            F.col("route_id"),
        )
        .agg(
            F.avg("value").alias("avg_value"),
            F.count(F.lit(1)).alias("n_events"),
            F.avg(F.col("ingest_delay_sec")).alias("avg_ingest_delay_sec"),
            F.sum(F.col("is_late_event").cast("int")).alias("n_late_events"),
            F.sum(F.col("is_clock_skew").cast("int")).alias("n_clock_skew"),
        )
        .withColumn(
            "late_event_rate",
            F.when(F.col("n_events") == 0, F.lit(0.0))
            .otherwise(F.col("n_late_events") / F.col("n_events")),
        )
        .select(
            F.col("window.start").alias("window_start"),
            F.col("window.end").alias("window_end"),
            F.col("metric"),
            F.col("route_id"),
            F.col("avg_value"),
            F.col("n_events"),
            F.col("avg_ingest_delay_sec"),
            F.col("n_late_events"),
            F.col("late_event_rate"),
            F.col("n_clock_skew"),
        )
    )

    silver_transit.write.format("delta").mode("overwrite").saveAsTable(SILVER_TRANSIT_TABLE)

def build_silver_weather_metrics(spark: SparkSession) -> None:
    """
    Aggregate FMI weather events into windowed Silver metrics.
    """
    bronze_all = spark.table(BRONZE_EVENTS_TABLE)

    weather_recent = (
        bronze_all
        .filter(F.col("source") == F.lit("fmi_weather"))
        .filter(
            F.col("event_time_ts")
            >= F.expr(f"current_timestamp() - INTERVAL {WEATHER_LOOKBACK_MINUTES} MINUTES")
        )
        .withColumn("station_id", F.col("entity_id"))
    )

    weather_clean = (
        weather_recent
        .filter(F.col("value").isNotNull())
        .filter(
            ~(
                (F.col("metric") == F.lit("t2m"))
                & ((F.col("value") < -60) | (F.col("value") > 60))
            )
        )
    )

    silver_weather = (
        weather_clean
        .groupBy(
            F.window("event_time_ts", WEATHER_WINDOW).alias("window"),
            F.col("metric"),
            F.col("station_id"),
        )
        .agg(
            F.avg("value").alias("avg_value"),
            F.count(F.lit(1)).alias("n_events"),
            F.avg(
                F.greatest(
                    F.unix_timestamp("ingest_time_ts") - F.unix_timestamp("event_time_ts"),
                    F.lit(0),
                )
            ).alias("avg_ingest_delay_sec"),
        )
        .select(
            F.col("window.start").alias("window_start"),
            F.col("window.end").alias("window_end"),
            F.col("metric"),
            F.col("station_id"),
            F.col("avg_value"),
            F.col("n_events"),
            F.col("avg_ingest_delay_sec"),
        )
    )

    silver_weather.write.format("delta").mode("overwrite").saveAsTable(SILVER_WEATHER_TABLE)
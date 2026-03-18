"""
Gold-layer KPI and operational metrics builders.
"""
import logging
from pyspark.sql import SparkSession
from src.pipeline.setup import use_database

from .config import (
    GOLD_PIPELINE_METRICS_TABLE,
    GOLD_ROUTE_DAILY_TABLE,
    GOLD_ROUTE_WINDOW_TABLE,
    SILVER_TRANSIT_TABLE,
    SILVER_WEATHER_TABLE,
)

def run_gold_layer(spark: SparkSession, logger: logging.Logger) -> None:
    logger.info("Gold layer started")
    use_database(spark)

    build_gold_route_kpi_window(spark)
    logger.info(f"Gold route window table updated: {GOLD_ROUTE_WINDOW_TABLE}")

    build_gold_route_kpi_daily(spark)
    logger.info(f"Gold route daily table updated: {GOLD_ROUTE_DAILY_TABLE}")

    build_gold_pipeline_metrics_window(spark)
    logger.info(f"Gold pipeline metrics table updated: {GOLD_PIPELINE_METRICS_TABLE}")

    logger.info(f"Gold route window row count: {spark.table(GOLD_ROUTE_WINDOW_TABLE).count()}")
    logger.info(f"Gold route daily row count: {spark.table(GOLD_ROUTE_DAILY_TABLE).count()}")
    logger.info(f"Gold pipeline metrics row count: {spark.table(GOLD_PIPELINE_METRICS_TABLE).count()}")

def build_gold_route_kpi_window(spark: SparkSession) -> None:
    """
    Create route-level KPI metrics at the window level.
    """
    spark.sql(
        f"""
        CREATE OR REPLACE TABLE {GOLD_ROUTE_WINDOW_TABLE}
        USING DELTA
        AS
        WITH s AS (
            SELECT *
            FROM {SILVER_TRANSIT_TABLE}
        ),
        wide AS (
            SELECT
                window_start,
                window_end,
                route_id,

                MAX(CASE WHEN metric = 'delay_sec' THEN avg_value END) AS avg_delay_sec,
                MAX(CASE WHEN metric = 'occupancy' THEN avg_value END) AS avg_occupancy_pct,

                MAX(CASE WHEN metric = 'delay_sec' THEN n_events END) AS n_events_delay,
                MAX(CASE WHEN metric = 'occupancy' THEN n_events END) AS n_events_occupancy,

                MAX(CASE WHEN metric = 'delay_sec' THEN late_event_rate END) AS late_rate_delay,
                MAX(CASE WHEN metric = 'delay_sec' THEN avg_ingest_delay_sec END) AS avg_ingest_delay_sec,
                MAX(CASE WHEN metric = 'delay_sec' THEN n_clock_skew END) AS n_clock_skew

            FROM s
            GROUP BY window_start, window_end, route_id
        )
        SELECT
            *,
            CASE
                WHEN COALESCE(n_clock_skew, 0) > 0 THEN 'CLOCK_SKEW'
                WHEN COALESCE(n_events_delay, 0) < 5 THEN 'LOW_VOLUME'
                WHEN COALESCE(late_rate_delay, 0.0) > 0.30 THEN 'HIGH_LATE_RATE'
                ELSE 'OK'
            END AS dq_flag
        FROM wide
        """
    )


def build_gold_route_kpi_daily(spark: SparkSession) -> None:
    """
    Aggregate route-level window KPIs into daily summaries.
    """
    spark.sql(
        f"""
        CREATE OR REPLACE TABLE {GOLD_ROUTE_DAILY_TABLE}
        USING DELTA
        AS
        SELECT
            DATE(window_start) AS date,
            route_id,
            AVG(avg_delay_sec) AS avg_delay_sec,
            AVG(avg_occupancy_pct) AS avg_occupancy_pct,
            SUM(COALESCE(n_events_delay, 0)) AS total_events_delay,
            SUM(COALESCE(n_events_occupancy, 0)) AS total_events_occupancy,
            AVG(COALESCE(late_rate_delay, 0.0)) AS avg_late_rate_delay,
            AVG(COALESCE(avg_ingest_delay_sec, 0.0)) AS avg_ingest_delay_sec,
            CASE
                WHEN SUM(CASE WHEN dq_flag <> 'OK' THEN 1 ELSE 0 END) > 0 THEN 'CHECK'
                ELSE 'OK'
            END AS dq_flag
        FROM {GOLD_ROUTE_WINDOW_TABLE}
        GROUP BY DATE(window_start), route_id
        """
    )


def build_gold_pipeline_metrics_window(spark: SparkSession) -> None:
    """
    Build a lightweight operational summary combining transit and weather
    Silver metrics at the window level.
    """
    spark.sql(
        f"""
        CREATE OR REPLACE TABLE {GOLD_PIPELINE_METRICS_TABLE}
        USING DELTA
        AS
        WITH transit AS (
            SELECT
                window_start,
                window_end,
                SUM(n_events) AS transit_total_events,
                AVG(avg_ingest_delay_sec) AS transit_avg_ingest_delay_sec
            FROM {SILVER_TRANSIT_TABLE}
            GROUP BY window_start, window_end
        ),
        weather AS (
            SELECT
                window_start,
                window_end,
                SUM(n_events) AS weather_total_events,
                AVG(avg_ingest_delay_sec) AS weather_avg_ingest_delay_sec
            FROM {SILVER_WEATHER_TABLE}
            GROUP BY window_start, window_end
        )
        SELECT
            COALESCE(t.window_start, w.window_start) AS window_start,
            COALESCE(t.window_end, w.window_end) AS window_end,
            t.transit_total_events,
            w.weather_total_events,
            t.transit_avg_ingest_delay_sec,
            w.weather_avg_ingest_delay_sec
        FROM transit t
        FULL OUTER JOIN weather w
            ON t.window_start = w.window_start
           AND t.window_end = w.window_end
        """
    )
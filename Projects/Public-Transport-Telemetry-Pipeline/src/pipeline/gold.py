"""
Gold-layer KPI and operational metrics builders.
"""

import logging
import pandas as pd
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from src.pipeline.setup import use_database
from src.pipeline.hsl import build_hsl_map_outputs

from .config import (
    GOLD_PIPELINE_METRICS_TABLE,
    GOLD_ROUTE_DAILY_TABLE,
    GOLD_ROUTE_WINDOW_TABLE,
    SILVER_TRANSIT_TABLE,
    SILVER_WEATHER_TABLE,
)


def build_gold_weather_station_outputs(
    spark: SparkSession, logger: logging.Logger
) -> None:
    """
    Build lightweight weather-station map output for Streamlit.

    Output:
        data/gold/weather/weather_stations_latest.parquet

    Strategy:
    - read FMI weather events from Bronze
    - keep the latest observation per station and metric
    - pivot metrics into one row per station
    """
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"
    output_dir = data_dir / "gold" / "weather"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = spark.table("bronze_events").filter(F.col("source") == "fmi_weather")

    if df.rdd.isEmpty():
        empty_df = pd.DataFrame(
            columns=[
                "station_id",
                "station_name",
                "lat",
                "lon",
                "observation_time",
                "temperature",
                "precipitation",
            ]
        )
        empty_df.to_parquet(output_dir / "weather_stations_latest.parquet", index=False)
        logger.info("No FMI weather rows found; wrote empty weather station output.")
        return

    station_df = (
        df.select(
            F.col("entity_id").alias("station_id"),
            F.col("attrs.station_name").alias("station_name"),
            F.col("attrs.lat").cast("double").alias("lat"),
            F.col("attrs.lon").cast("double").alias("lon"),
            F.col("metric"),
            F.col("value").cast("double").alias("value"),
            F.col("event_time_ts"),
        )
        .filter(F.col("lat").isNotNull() & F.col("lon").isNotNull())
        .filter(F.col("value").isNotNull())
        .filter(~F.isnan(F.col("value")))
    )

    # Keep latest observation per station + metric
    latest_ts = station_df.groupBy("station_id", "metric").agg(
        F.max("event_time_ts").alias("event_time_ts")
    )

    latest_df = (
        station_df.alias("s")
        .join(
            latest_ts.alias("m"),
            on=[
                F.col("s.station_id") == F.col("m.station_id"),
                F.col("s.metric") == F.col("m.metric"),
                F.col("s.event_time_ts") == F.col("m.event_time_ts"),
            ],
            how="inner",
        )
        .select(
            F.col("s.station_id"),
            F.col("s.station_name"),
            F.col("s.lat"),
            F.col("s.lon"),
            F.col("s.metric"),
            F.col("s.value"),
            F.col("s.event_time_ts"),
        )
    )

    # Pivot metrics into map-ready columns
    weather_wide = (
        latest_df.groupBy("station_id", "station_name", "lat", "lon")
        .pivot("metric", ["t2m", "r_1h"])
        .agg(F.first("value"))
    )

    obs_time = latest_df.groupBy("station_id").agg(
        F.max("event_time_ts").alias("observation_time")
    )

    result_df = (
        weather_wide.join(obs_time, on="station_id", how="left")
        .withColumnRenamed("t2m", "temperature")
        .withColumnRenamed("r_1h", "precipitation")
        .select(
            "station_id",
            "station_name",
            "lat",
            "lon",
            "observation_time",
            "temperature",
            "precipitation",
        )
        .orderBy("station_name")
    )

    pdf = result_df.toPandas()
    pdf.to_parquet(output_dir / "weather_stations_latest.parquet", index=False)

    logger.info("Weather station gold output written to data/gold/weather/")
    logger.info(f"Weather station output directory: {output_dir}")
    logger.info(f"Weather station rows: {len(pdf)}")


def run_gold_layer(spark: SparkSession, logger: logging.Logger) -> None:
    logger.info("Gold layer started")
    use_database(spark)

    build_gold_route_kpi_window(spark)
    logger.info(f"Gold route window table updated: {GOLD_ROUTE_WINDOW_TABLE}")

    build_gold_route_kpi_daily(spark)
    logger.info(f"Gold route daily table updated: {GOLD_ROUTE_DAILY_TABLE}")

    build_gold_pipeline_metrics_window(spark)
    logger.info(f"Gold pipeline metrics table updated: {GOLD_PIPELINE_METRICS_TABLE}")

    build_gold_hsl_map_outputs(logger)
    logger.info("HSL map gold outputs updated")

    build_gold_weather_station_outputs(spark, logger)
    logger.info("Weather station gold outputs updated")

    logger.info(
        f"Gold route window row count: {spark.table(GOLD_ROUTE_WINDOW_TABLE).count()}"
    )
    logger.info(
        f"Gold route daily row count: {spark.table(GOLD_ROUTE_DAILY_TABLE).count()}"
    )
    logger.info(
        f"Gold pipeline metrics row count: {spark.table(GOLD_PIPELINE_METRICS_TABLE).count()}"
    )


def build_gold_route_kpi_window(spark: SparkSession) -> None:
    """
    Create route-level KPI metrics at the window level.
    """
    spark.sql(f"""
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
        """)


def build_gold_route_kpi_daily(spark: SparkSession) -> None:
    """
    Aggregate route-level window KPIs into daily summaries.
    """
    spark.sql(f"""
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
        """)


def build_gold_pipeline_metrics_window(spark: SparkSession) -> None:
    """
    Build a lightweight operational summary combining transit and weather
    Silver metrics at the window level.
    """
    spark.sql(f"""
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
        """)


def build_gold_hsl_map_outputs(logger: logging.Logger) -> None:
    """
    Build lightweight HSL map-ready outputs for Streamlit map visualization
    These outputs are stored as parquet files rather than delta tables.
    """
    project_root = Path(__file__).resolve().parents[2]
    DATA_DIR = project_root / "data"

    gtfs_dir = DATA_DIR / "external" / "gtfs_hsl"
    output_dir = DATA_DIR / "gold" / "hsl"

    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = build_hsl_map_outputs(
        gtfs_dir=gtfs_dir,
        mode="all",
        route="all",
        lookback_minutes=60,
    )

    outputs["df_map"].to_parquet(output_dir / "hsl_df_map.parquet", index=False)
    outputs["route_options"].to_parquet(
        output_dir / "hsl_route_options.parquet", index=False
    )
    outputs["map_points"].to_parquet(output_dir / "hsl_map_points.parquet", index=False)
    outputs["paths"].to_parquet(output_dir / "hsl_route_paths.parquet", index=False)

    # lightweight overview paths for All routes
    overview_paths = outputs["paths"].copy()

    if not overview_paths.empty:
        group_col = None

        if "route_label" in overview_paths.columns:
            group_col = "route_label"
        elif "route_short_name" in overview_paths.columns:
            group_col = "route_short_name"

        if group_col is not None:
            overview_paths = (
                overview_paths.sort_values(group_col)
                .groupby(group_col, as_index=False)
                .head(1)
                .copy()
            )

    overview_paths.to_parquet(
        output_dir / "hsl_route_paths_overview.parquet",
        index=False,
    )

    logger.info("HSL gold map outputs written to data/gold/hsl/")
    logger.info(f"HSL gold output directory: {output_dir}")
    logger.info(f"HSL df_map rows: {len(outputs['df_map'])}")
    logger.info(f"HSL route_options rows: {len(outputs['route_options'])}")
    logger.info(f"HSL map_points rows: {len(outputs['map_points'])}")
    logger.info(f"HSL paths rows: {len(outputs['paths'])}")
    logger.info(f"HSL overview paths rows: {len(overview_paths)}")

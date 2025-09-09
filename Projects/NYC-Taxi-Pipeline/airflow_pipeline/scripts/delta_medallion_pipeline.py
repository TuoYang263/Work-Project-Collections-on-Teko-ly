# scripts/delta_medallion_pipeline.py
# Medallion (Bronze -> Silver -> Gold) pipeline using Delta Lake + Spark
# Bronze: ingest raw monthly parquet -> Delta
# Silver: clean/standardize
# Gold: aggregate hourly summary & zone daily summaries
# Optional: export Gold hourly summary -> BigQuery

import os
import sys
from functools import reduce
from typing import List

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import TimestampType

# --- Make parent (project root) importable ---
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- Robust imports (works whether files are in the root path or in scripts/) ---
try:
    from scripts import config
except Exception:
    import config

try:
    from scripts.load_nyc_yellow_taxi_data import download_to_local
except Exception:
    from load_nyc_yellow_taxi_data import download_to_local

try:
    from scripts.transform_trip_data_spark import transform_trip_data
except Exception:
    from transform_trip_data_spark import transform_trip_data

try:
    from scripts.logger import get_logger
except Exception:
    from logger import get_logger

try:
    from scripts.get_bigquery_client import get_bigquery_client
except Exception:
    from get_bigquery_client import get_bigquery_client

# Optional: delta-spark helper (if available in env)
try:
    from delta import configure_spark_with_delta_pip
except Exception:
    configure_spark_with_delta_pip = None  # fall back to plain builder

logger = get_logger()

# ---------- Paths (default under <BASE_DIR>/data/delta/*) ----------
BASE_DIR = getattr(config, "BASE_DIR", PROJECT_ROOT)
DELTA_BASE = os.path.join(BASE_DIR, "data", "delta")

BRONZE_TRIPS = os.path.join(DELTA_BASE, "bronze", "trips")          # raw delta
SILVER_TRIPS = os.path.join(DELTA_BASE, "silver", "trips_clean")    # cleaned
GOLD_SUMMARY_HOURLY = os.path.join(DELTA_BASE, "gold", "trip_summary_hourly")
GOLD_ZONE_PICKUP_DAILY = os.path.join(DELTA_BASE, "gold", "zone_summary_pickup_daily")
GOLD_ZONE_DROPOFF_DAILY = os.path.join(DELTA_BASE, "gold", "zone_summary_dropoff_daily")

for p in [BRONZE_TRIPS, SILVER_TRIPS, GOLD_SUMMARY_HOURLY,
           GOLD_ZONE_PICKUP_DAILY, GOLD_ZONE_DROPOFF_DAILY]:
    os.makedirs(p, exist_ok=True)

# ---------- Spark ----------
from pyspark.sql import SparkSession

def get_spark(app_name: str = "NYC Medallion (Delta)") -> SparkSession:
    """
    Create or get a SparkSession with Delta Lake support enabled.
    This function makes sure we always use a session configured with Delta,
    even if another plain Spark session was created before.
    """

    # 1) If an active Spark session exists but Delta is not enabled,
    #    stop it to avoid reusing a misconfigured session.
    active = SparkSession.getActiveSession()
    if active:
        exts = active.conf.get("spark.sql.extensions", "")
        if "DeltaSparkSessionExtension" not in exts:
            try:
                active.stop()
            except Exception:
                pass

    # 2) Try to import Delta's helper inside the function (not at module level).
    #    This avoids the case where module import failed before Delta was installed.
    try:
        from delta import configure_spark_with_delta_pip  # type: ignore
    except Exception:
        configure_spark_with_delta_pip = None  # noqa: F811

    # 3) Build SparkSession with Delta configurations.
    builder = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.databricks.delta.schema.autoMerge.enabled", "true")
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.2.0")
    )

    # 4) If Delta helper is available, wrap the builder with it; otherwise fallback.
    spark = (
        configure_spark_with_delta_pip(builder).getOrCreate()
        if configure_spark_with_delta_pip is not None
        else builder.getOrCreate()
    )

    # 5) Final validation: make sure Delta extension is actually enabled.
    exts = spark.conf.get("spark.sql.extensions", "")
    if "DeltaSparkSessionExtension" not in exts:
        raise RuntimeError(f"Delta not enabled. current spark.sql.extensions={exts}")

    return spark

# ---------- Bronze ----------
def write_bronze(year: int, months: List[int]) -> str:
    """
    Ingest raw monthly parquet files and persist as a single Delta table (Bronze).
    Adds _ingest_time and partitions by pickup_day (if pickup timestsamp exists).
    """
    spark = get_spark("Bronze Ingest (Delta)")
    logger.info(f"[DEBUG] spark.sql.extensions = {spark.conf.get('spark.sql.extensions')}")
    local_paths = download_to_local(year, months)   # returns local file paths
    if not local_paths:
        raise RuntimeError("No input files found for Bronze ingest.")
    
    dfs = []
    for p in local_paths:
        uri = f"file:///{p.replace(os.sep, '/')}"
        logger.info(f"[Bronze] Reading raw parquet: {uri}")
        dfs.append(spark.read.parquet(uri))

    bronze_df = dfs[0]
    for df in dfs[1:]:
        bronze_df = bronze_df.unionByName(df, allowMissingColumns=True)

    bronze_df = bronze_df.withColumn("_ingest_time", F.current_timestamp())

    # If pickup column exists, cast & add pickup_day for partitioning
    has_pickup = "tpep_pickup_datetime" in [c.lower() for c in bronze_df.columns]
    writer = (
        bronze_df.write.mode("overwrite")
        .format("delta")
        .option("overwriteSchema", "true")
        .option("mergeSchema", "true")
    )

    if has_pickup and "tpep_pickup_datetime" in bronze_df.columns:
        bronze_df = bronze_df.withColumn(
            "tpep_pickup_datetime",
            F.col("tpep_pickup_datetime").cast(TimestampType())
        )
        bronze_df = bronze_df.withColumn("pickup_day", F.to_date(F.col("tpep_pickup_datetime")))
        writer = (
            bronze_df.write.mode("overwrite")
            .format("delta")
            .option("overwriteSchema", "true")
            .option("mergeSchema", "true")
            .partitionBy("pickup_day")
        )

    logger.info(f"[Bronze] Writing Delta table at: {BRONZE_TRIPS}")
    writer.save(BRONZE_TRIPS)
    return BRONZE_TRIPS

# ---------- Silver ----------
def write_silver() -> str:
    """
    Read Bronze Delta, apply cleaning/standardization (reuse transform_trip_data),
    and persist Silver Delta.
    """
    spark = get_spark("Silver Clean (Delta)")
    logger.info(f"[DEBUG] spark.sql.extensions = {spark.conf.get('spark.sql.extensions')}")

    logger.info(f"[Silver] Loading Bronze from: {BRONZE_TRIPS}")
    bronze = spark.read.format("delta").load(BRONZE_TRIPS)

    logger.info("[Silver] Applying transformations...")
    silver = transform_trip_data(bronze)  # your existing cleaning logic

    logger.info(f"[Silver] Writing Delta to: {SILVER_TRIPS}")
    (
        silver.write.mode("overwrite")
        .format("delta")
        .option("overwriteSchema", "true")
        .save(SILVER_TRIPS)
    )
    return SILVER_TRIPS


# ---------- Gold ----------
def write_gold() -> dict:
    """
    Build Gold Delta tables:
      - trip_summary_hourly
      - zone_summary_pickup_daily
      - zone_summary_dropoff_daily
    """
    spark = get_spark("Gold Aggregate (Delta)")
    logger.info(f"[DEBUG] spark.sql.extensions = {spark.conf.get('spark.sql.extensions')}")
    
    df = spark.read.format("delta").load(SILVER_TRIPS)

    # Minimal additional DQ filters (already mostly done in transform)
    filters = []
    for c in ["fare_amount", "trip_distance", "passenger_count", "total_amount"]:
        if c in df.columns:
            filters.append(F.col(c) > 0)
    if filters:
        df = df.filter(reduce(lambda a, b: a & b, filters))

    # Time Range Filtering
    year = config.SETTINGS["data_config"]["year"]
    months = config.SETTINGS["data_config"]["months"]

    print(f"Year: {year}")
    print(f"Month: {months}")

    df = df.filter(
        (F.year("tpep_pickup_datetime") == year) &
        (F.month("tpep_pickup_datetime").isin(months))
    )

    # Hourly summary
    summary_hourly = (
        df.withColumn("pickup_hour", F.date_trunc("hour", F.col("tpep_pickup_datetime")))
          .groupBy("pickup_hour")
          .agg(
              F.count("*").alias("trip_count"),
              F.round(F.avg("fare_amount"), 2).alias("avg_fare"),
              F.round(F.avg("tip_amount"), 2).alias("avg_tip"),
              F.sum("passenger_count").alias("total_passengers"),
              F.round(F.avg("trip_distance"), 2).alias("avg_distance"),
          )
          .orderBy("pickup_hour")
    )
    logger.info(f"[Gold] Writing hourly summary to: {GOLD_SUMMARY_HOURLY}")
    summary_hourly.write.mode("overwrite").format("delta").save(GOLD_SUMMARY_HOURLY)

    # Zone summaries (daily) for pickup & dropoff
    def write_zone_summary(loc: str, out_path: str):
        # loc = 'pickup' | 'dropoff'
        time_col = F.to_date(F.col(f"tpep_{loc}_datetime")).alias(f"{loc}_day")
        zone_col = F.col("pulocationid" if loc == "pickup" else "dolocationid").alias("zone_id")

        zdf = (
            df.withColumn(f"{loc}_day", time_col)
              .withColumn("zone_id", zone_col)
              .groupBy(f"{loc}_day", "zone_id")
              .agg(
                  F.count("*").alias("trip_count"),
                  F.round(F.avg("fare_amount"), 2).alias("avg_fare"),
                  F.round(F.avg("tip_amount"), 2).alias("avg_tip"),
                  F.sum("passenger_count").alias("total_passengers"),
                  F.round(F.avg("trip_distance"), 2).alias("avg_distance"),
              )
              .orderBy(f"{loc}_day", "zone_id")
        )
        logger.info(f"[Gold] Writing zone summary ({loc}) to: {out_path}")
        zdf.write.mode("overwrite").format("delta").save(out_path)

    write_zone_summary("pickup", GOLD_ZONE_PICKUP_DAILY)
    write_zone_summary("dropoff", GOLD_ZONE_DROPOFF_DAILY)

    return {
        "hourly_summary": GOLD_SUMMARY_HOURLY,
        "zone_pickup_daily": GOLD_ZONE_PICKUP_DAILY,
        "zone_dropoff_daily": GOLD_ZONE_DROPOFF_DAILY,
    }

# ---------- Export Gold → BigQuery (3 tables) ----------
def _bq_table_ref(project_id: str, dataset_id: str, table_name: str) -> str:
    return f"{project_id}.{dataset_id}.{table_name}"

def _export_delta_to_bq(delta_path: str, table_ref: str):
    from google.cloud import bigquery
    spark = get_spark("Export Delta -> BigQuery")
    df = spark.read.format("delta").load(delta_path)
    pdf = df.toPandas()

    client = get_bigquery_client()
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        autodetect=True,
    )
    logger.info(f"[Export] Uploading {delta_path} → {table_ref} ...")
    job = client.load_table_from_dataframe(pdf, table_ref, job_config=job_config)
    job.result()
    logger.info(f"[Export] Upload complete: {table_ref}")

def export_gold_to_bigquery():
    """
    Backward-compatible: export ONLY hourly summary to BigQuery.
    Table name comes from config.SUMMARY_TABLE_NAME (legacy).
    """
    project_id = getattr(config, "PROJECT_ID", None)
    dataset_id = getattr(config, "DATASET_ID", None)
    table_name = getattr(config, "SUMMARY_TABLE_NAME", None)  # legacy key

    if not (project_id and dataset_id and table_name):
        raise RuntimeError("PROJECT_ID/DATASET_ID/SUMMARY_TABLE_NAME must be set in config.py")

    table_ref = _bq_table_ref(project_id, dataset_id, table_name)
    _export_delta_to_bq(GOLD_SUMMARY_HOURLY, table_ref)
    return table_ref

def export_all_gold_to_bigquery():
    """
    Export 3 Gold tables to BigQuery:
      - hourly summary                → SUMMARY_TABLE_NAME (legacy) or 'trip_summary_hourly'
      - zone_summary_pickup_daily     → ZONE_PICKUP_SUMMARY_TABLE_NAME or default name
      - zone_summary_dropoff_daily    → ZONE_DROPOFF_SUMMARY_TABLE_NAME or default name
    """
    project_id = getattr(config, "PROJECT_ID", None)
    dataset_id = getattr(config, "DATASET_ID", None)

    if not (project_id and dataset_id):
        raise RuntimeError("PROJECT_ID and DATASET_ID must be set in config.py")

    # Get the table name (use names from config if it does have, otherwise by default)
    hourly_tbl   = getattr(config, "SUMMARY_TABLE_NAME", "trip_summary_hourly")
    pickup_tbl   = getattr(config, "ZONE_PICKUP_SUMMARY_TABLE_NAME", "zone_summary_pickup_daily")
    dropoff_tbl  = getattr(config, "ZONE_DROPOFF_SUMMARY_TABLE_NAME", "zone_summary_dropoff_daily")

    # Delta → BQ
    _export_delta_to_bq(GOLD_SUMMARY_HOURLY,      _bq_table_ref(project_id, dataset_id, hourly_tbl))
    _export_delta_to_bq(GOLD_ZONE_PICKUP_DAILY,   _bq_table_ref(project_id, dataset_id, pickup_tbl))
    _export_delta_to_bq(GOLD_ZONE_DROPOFF_DAILY,  _bq_table_ref(project_id, dataset_id, dropoff_tbl))

    return {
        "hourly":  _bq_table_ref(project_id, dataset_id, hourly_tbl),
        "pickup":  _bq_table_ref(project_id, dataset_id, pickup_tbl),
        "dropoff": _bq_table_ref(project_id, dataset_id, dropoff_tbl),
    }

# ---------- Airflow-friendly callables ----------
def bronze_task(**_):
    y = config.SETTINGS["data_config"]["year"]
    ms = config.SETTINGS["data_config"]["months"]
    return write_bronze(y, ms)

def silver_task(**_):
    return write_silver()

def gold_task(**_):
    return write_gold()

def export_bq_all_task(**_):
    return export_all_gold_to_bigquery()

if __name__ == "__main__":
    # Local sequential run (optional)
    y = config.SETTINGS["data_config"]["year"]
    ms = config.SETTINGS["data_config"]["months"]
    write_bronze(y, ms)
    write_silver()
    write_gold()
    export_gold_to_bigquery()
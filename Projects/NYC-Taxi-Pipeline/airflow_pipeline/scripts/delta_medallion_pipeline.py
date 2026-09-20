# scripts/delta_medallion_pipeline.py
# Medallion (Bronze -> Silver -> Gold) pipeline using Delta Lake + Spark
# Bronze: ingest raw monthly parquet -> Delta
# Silver: clean/standardize
# Gold: aggregate hourly summary & zone daily summaries
# Optional: export Gold hourly summary -> BigQuery

import os
import sys
import time
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

REFERENCE_DIR = os.path.join(BASE_DIR, "data", "reference")
TAXI_ZONE_LOOKUP = os.path.join(
    REFERENCE_DIR,
    "taxi_zone_lookup.csv",
)

BRONZE_TRIPS = os.path.join(DELTA_BASE, "bronze", "trips")          # raw delta
SILVER_TRIPS = os.path.join(DELTA_BASE, "silver", "trips_clean")    # cleaned
GOLD_SUMMARY_HOURLY = os.path.join(DELTA_BASE, "gold", "trip_summary_hourly")
GOLD_ZONE_PICKUP_DAILY = os.path.join(DELTA_BASE, "gold", "zone_summary_pickup_daily")
GOLD_ZONE_DROPOFF_DAILY = os.path.join(DELTA_BASE, "gold", "zone_summary_dropoff_daily")
GOLD_FLOW_IMBALANCE_DAILY = os.path.join(
    DELTA_BASE,
    "gold",
    "flow_imbalance_daily",
)
GOLD_AIRPORT_ECONOMICS_DAILY = os.path.join(
    DELTA_BASE,
    "gold",
    "airport_economics_daily",
)

AIRPORT_ZONE_CODES = {
    "JFK Airport": "JFK",
    "LaGuardia Airport": "LGA",
    "Newark Airport": "EWR",
}

for p in [
    BRONZE_TRIPS,
    SILVER_TRIPS,
    GOLD_SUMMARY_HOURLY,
    GOLD_ZONE_PICKUP_DAILY,
    GOLD_ZONE_DROPOFF_DAILY,
    GOLD_FLOW_IMBALANCE_DAILY,
    GOLD_AIRPORT_ECONOMICS_DAILY,
]:
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

# ---------- Spark baseline instrumentation ----------

BASELINE_SPARK_CONF_KEYS = [
    "spark.sql.shuffle.partitions",
    "spark.sql.adaptive.enabled",
    "spark.sql.autoBroadcastJoinThreshold",
]


def log_spark_baseline(
    spark: SparkSession,
    label: str,
) -> None:
    """
    Log Spark runtime and configuration metadata without touching
    DataFrame execution.
    """
    logger.info(f"[BASELINE][{label}] spark_version={spark.version}")
    logger.info(f"[BASELINE][{label}] master={spark.sparkContext.master}")
    logger.info(
        f"[BASELINE][{label}] "
        f"default_parallelism={spark.sparkContext.defaultParallelism}"
    )

    for key in BASELINE_SPARK_CONF_KEYS:
        try:
            value = spark.conf.get(key)
        except Exception:
            value = "<unset>"

        logger.info(f"[BASELINE][{label}] {key}={value}")


def log_physical_plan(
    df: DataFrame,
    label: str,
) -> None:
    """
    Print the formatted Spark physical plan without triggering
    an explicit DataFrame action.
    """
    logger.info(
        f"[BASELINE][{label}] formatted_physical_plan_start"
    )

    df.explain(mode="formatted")

    logger.info(
        f"[BASELINE][{label}] formatted_physical_plan_end"
    )


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


def build_hourly_summary(df: DataFrame) -> DataFrame:
    """
    Build the Gold hourly trip summary from an already-filtered
    Silver trip DataFrame.

    Grain:
        One row per pickup_hour.
    """
    return (
        df.withColumn(
            "pickup_hour",
            F.date_trunc(
                "hour",
                F.col("tpep_pickup_datetime"),
            ),
        )
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


# ---------- Gold ----------
def write_gold(
    hourly_only: bool = False,
    hourly_output_path: str | None = None,
) -> dict:
    """
    Build Gold Delta tables:
    - trip_summary_hourly
    - zone_summary_pickup_daily
    - zone_summary_dropoff_daily
    - flow_imbalance_daily
    - airport_economics_daily
    """
    spark = get_spark("Gold Aggregate (Delta)")
    logger.info(f"[DEBUG] spark.sql.extensions = {spark.conf.get('spark.sql.extensions')}")

    aqe_override = os.getenv("NYC_SPARK_AQE")

    if aqe_override is not None:
        spark.conf.set(
            "spark.sql.adaptive.enabled",
            aqe_override.lower(),
        )

    shuffle_override = os.getenv("NYC_SPARK_SHUFFLE_PARTITIONS")

    if shuffle_override is not None:
        shuffle_partitions = int(shuffle_override)

        if shuffle_partitions <= 0:
            raise ValueError(
                "NYC_SPARK_SHUFFLE_PARTITIONS must be greater than 0"
            )

        spark.conf.set(
            "spark.sql.shuffle.partitions",
            str(shuffle_partitions),
        )
    
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

    log_spark_baseline(
        spark=spark,
        label="gold_input_after_filters",
    )

    # Hourly summary
    summary_hourly = build_hourly_summary(df)

    log_spark_baseline(
        spark=spark,
        label="gold_hourly_summary",
    )

    log_physical_plan(
        df=summary_hourly,
        label="gold_hourly_summary",
    )

    hourly_target = hourly_output_path or GOLD_SUMMARY_HOURLY

    logger.info(f"[Gold] Writing hourly summary to: {hourly_target}")

    started_at = time.perf_counter()

    summary_hourly.write \
        .mode("overwrite") \
        .format("delta") \
        .save(hourly_target)

    elapsed_seconds = time.perf_counter() - started_at

    logger.info(
        f"[BASELINE][gold_hourly_summary] "
        f"write_wall_seconds={elapsed_seconds:.3f}"
    )

    if hourly_only:
        return {
            "hourly_summary": hourly_target,
        }

    # Zone passenger-flow imbalance
    flow_imbalance = build_zone_flow_imbalance(df)

    flow_imbalance = flow_imbalance.filter(
        (F.year("service_day") == year)
        & (F.month("service_day").isin(months))
    )

    flow_imbalance = enrich_with_taxi_zone(
        flow_imbalance,
        spark,
    )

    logger.info(
        f"[Gold] Writing flow imbalance daily to: "
        f"{GOLD_FLOW_IMBALANCE_DAILY}"
    )

    (
        flow_imbalance.write
        .mode("overwrite")
        .format("delta")
        .save(GOLD_FLOW_IMBALANCE_DAILY)
    )

    # Airport economics
    airport_economics = build_airport_economics_daily(
        df,
        spark,
    )

    airport_economics = airport_economics.filter(
        (F.year("service_day") == year)
        & (F.month("service_day").isin(months))
    )

    logger.info(
        f"[Gold] Writing airport economics daily to: "
        f"{GOLD_AIRPORT_ECONOMICS_DAILY}"
    )

    (
        airport_economics.write
        .mode("overwrite")
        .format("delta")
        .save(GOLD_AIRPORT_ECONOMICS_DAILY)
    )

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
        "hourly_summary": hourly_target,
        "zone_pickup_daily": GOLD_ZONE_PICKUP_DAILY,
        "zone_dropoff_daily": GOLD_ZONE_DROPOFF_DAILY,
        "flow_imbalance_daily": GOLD_FLOW_IMBALANCE_DAILY,
        "airport_economics_daily": GOLD_AIRPORT_ECONOMICS_DAILY,
    }


def build_zone_flow_imbalance(df: DataFrame) -> DataFrame:
    pickups = (
        df.withColumn(
            "service_day",
            F.to_date(F.col("tpep_pickup_datetime")),
        )
        .groupBy(
            "service_day",
            F.col("pulocationid").alias("zone_id"),
        )
        .agg(
            F.count("*").alias("pickup_count"),
        )
    )

    dropoffs = (
        df.withColumn(
            "service_day",
            F.to_date(F.col("tpep_dropoff_datetime")),
        )
        .groupBy(
            "service_day",
            F.col("dolocationid").alias("zone_id"),
        )
        .agg(
            F.count("*").alias("dropoff_count"),
        )
    )

    return (
        pickups
        .join(
            dropoffs,
            on=["service_day", "zone_id"],
            how="full",
        )
        .fillna(
            0,
            subset=["pickup_count", "dropoff_count"],
        )
        .withColumn(
            "total_flow",
            F.col("pickup_count") + F.col("dropoff_count"),
        )
        .withColumn(
            "flow_imbalance",
            F.round(
                (
                    F.col("pickup_count") -
                    F.col("dropoff_count")
                )
                /
                F.col("total_flow"),
                4,
            ),
        )
        .orderBy("service_day", "zone_id")
    )


def build_airport_flow_daily(
    zone_flow_df: DataFrame,
) -> DataFrame:
    """
    Build daily airport passenger-flow metrics from the enriched
    zone flow imbalance dataset.

    Grain:
        One row per service_day x airport_code.
    """

    airport_code = (
        F.when(F.col("zone") == "JFK Airport", F.lit("JFK"))
        .when(F.col("zone") == "LaGuardia Airport", F.lit("LGA"))
        .when(F.col("zone") == "Newark Airport", F.lit("EWR"))
    )

    return (
        zone_flow_df
        .filter(F.col("zone").isin(list(AIRPORT_ZONE_CODES.keys())))
        .withColumn("airport_code", airport_code)
        .withColumnRenamed("zone", "airport_name")
        .select(
            "service_day",
            "airport_code",
            "airport_name",
            "borough",
            "pickup_count",
            "dropoff_count",
            "total_flow",
            "flow_imbalance",
        )
        .orderBy("service_day", "airport_code")
    )


def build_airport_economics_daily(
    df: DataFrame,
    spark: SparkSession,
) -> DataFrame:
    """
    Build daily airport economics by airport and trip direction.

    Grain:
        One row per service_day x airport_code x direction.
    """

    airport_lookup = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(TAXI_ZONE_LOOKUP)
        .filter(
            F.col("Zone").isin(
                list(AIRPORT_ZONE_CODES.keys())
            )
        )
        .select(
            F.col("LocationID").cast("long").alias("zone_id"),
            F.col("Zone").alias("airport_name"),
        )
        .withColumn(
            "airport_code",
            F.when(
                F.col("airport_name") == "JFK Airport",
                F.lit("JFK"),
            )
            .when(
                F.col("airport_name") == "LaGuardia Airport",
                F.lit("LGA"),
            )
            .when(
                F.col("airport_name") == "Newark Airport",
                F.lit("EWR"),
            )
        )
    )

    base = (
        df.withColumn(
            "trip_duration_minutes",
            (
                F.col("tpep_dropoff_datetime").cast("long")
                - F.col("tpep_pickup_datetime").cast("long")
            ) / 60.0,
        )
        .filter(F.col("trip_duration_minutes") > 0)
    )

    departures = (
        base.join(
            F.broadcast(
                airport_lookup.select(
                    F.col("zone_id").alias("pulocationid"),
                    "airport_code",
                    "airport_name",
                )
            ),
            on="pulocationid",
            how="inner",
        )
        .withColumn(
            "service_day",
            F.to_date("tpep_pickup_datetime"),
        )
        .withColumn("direction", F.lit("DEPARTURE"))
    )

    arrivals = (
        base.join(
            F.broadcast(
                airport_lookup.select(
                    F.col("zone_id").alias("dolocationid"),
                    "airport_code",
                    "airport_name",
                )
            ),
            on="dolocationid",
            how="inner",
        )
        .withColumn(
            "service_day",
            F.to_date("tpep_dropoff_datetime"),
        )
        .withColumn("direction", F.lit("ARRIVAL"))
    )

    airport_trips = departures.unionByName(
        arrivals,
        allowMissingColumns=True,
    )

    # EWR departures are excluded from airport economics.
    # NYC yellow taxis are not permitted to pick up passengers
    # at Newark Airport, and observed pickup records contain
    # substantial anomalous / non-representative trip patterns.
    airport_trips = airport_trips.filter(
        ~(
            (F.col("airport_code") == "EWR")
            & (F.col("direction") == "DEPARTURE")
        )
    )

    return (
        airport_trips
        .groupBy(
            "service_day",
            "airport_code",
            "airport_name",
            "direction",
        )
        .agg(
            F.count("*").alias("trip_count"),
            F.round(F.sum("total_amount"), 2).alias("gross_amount"),
            F.round(F.avg("total_amount"), 2).alias("avg_total_amount"),
            F.round(F.avg("fare_amount"), 2).alias("avg_fare_amount"),
            F.round(F.avg("trip_distance"), 2).alias("avg_trip_distance"),
            F.round(
                F.avg("trip_duration_minutes"),
                2,
            ).alias("avg_duration_minutes"),
            F.round(
                F.sum("total_amount")
                /
                (F.sum("trip_duration_minutes") / 60.0),
                2,
            ).alias("gross_amount_per_occupied_hour"),
        )
        .orderBy(
            "service_day",
            "airport_code",
            "direction",
        )
    )


def enrich_with_taxi_zone(
    df: DataFrame,
    spark: SparkSession,
) -> DataFrame:
    """
    Enrich a zone-level DataFrame with the official NYC TLC
    taxi-zone dimension.

    The lookup is intentionally broadcast because it is a very
    small reference dataset compared with the trip fact data.
    """

    zone_lookup = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(TAXI_ZONE_LOOKUP)
        .select(
            F.col("LocationID").cast("long").alias("zone_id"),
            F.col("Borough").alias("borough"),
            F.col("Zone").alias("zone"),
            F.col("service_zone"),
        )
    )

    return (
        df.join(
            F.broadcast(zone_lookup),
            on="zone_id",
            how="left",
        )
        .select(
            "service_day",
            "zone_id",
            "borough",
            "zone",
            "service_zone",
            "pickup_count",
            "dropoff_count",
            "total_flow",
            "flow_imbalance",
        )
        .orderBy("service_day", "zone_id")
    )

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
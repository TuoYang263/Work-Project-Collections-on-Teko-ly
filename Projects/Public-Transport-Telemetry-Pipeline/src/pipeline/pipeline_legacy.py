# Databricks notebook source
# MAGIC %md
# MAGIC # Public Transport Telemetry - MVP
# MAGIC
# MAGIC Minimal simulation of transit and weather telemetry.
# MAGIC Structured in Bronze -> Silver -> Gold layers

# COMMAND ----------

# Imports
import uuid
import time, random
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import *

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bronze layer
# MAGIC
# MAGIC Append-only storage for raw transit and weather events.
# MAGIC Only minimal normalization is applied.

# COMMAND ----------

# Use UTC for consistent event-time handling
print("session timeZone =", spark.conf.get("spark.sql.session.timeZone"))
spark.conf.set("spark.sql.session.timeZone", "UTC")

# Use a database for the MVP (managed storage, no explicit DBFS paths)
spark.sql("CREATE DATABASE IF NOT EXISTS azure_streaming_mvp")
spark.sql("USE azure_streaming_mvp")

RESET = True # drop and recreate tables

def reset_tables():
  spark.sql("DROP TABLE IF EXISTS bronze_events")
  spark.sql("DROP TABLE IF EXISTS silver_transit_metrics")
  spark.sql("DROP TABLE IF EXISTS silver_weather_metrics")
  # For reference
  # spark.sql("DROP TABLE IF EXISTS silver_metrics")
  print("Tables dropped: bronze_events, silver_transit_metrics, silver_weather_metrics")

# Clean slate
if RESET:
  reset_tables()

# Create an empty Bronze table with a fixed schema (append-only)
spark.sql("""
CREATE TABLE bronze_events (
  event_id STRING,
  -- event_time_raw: Raw event time from source (UTC, ISO 8601)
  event_time_raw STRING,
  source STRING,
  entity_type STRING,
  entity_id STRING,
  metric STRING,
  value DOUBLE,
  unit STRING,
  attrs MAP<STRING, STRING>,
  -- event_time_ts: Parsed event time used for windowing
  event_time_ts TIMESTAMP,
  -- ingest_time_ts: Ingestion timestamp (system time)
  ingest_time_ts TIMESTAMP
)
USING DELTA
-- Use Delta Lake (append-only Bronze storage)
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Simulated transit telemetry ingestion
# MAGIC
# MAGIC Generate synthetic vehicle events per route and append to Bronze

# COMMAND ----------

ROUTES = ["M1", "M2", "T1", "R10", "B1", "B2", "X3", "X7"]
TIME_SPAN_MINUTES = 60 # simulation window (minutes)

def make_event(i):
    metric = random.choice(["delay_sec", "occupancy"])
    value = random.randint(-30, 600) if metric == "delay_sec" else random.randint(0, 80)

    # Randomize event time within last TIME_SPAN_MINUTES
    offset_sec = random.randint(0, TIME_SPAN_MINUTES * 60)
    event_epoch = int(time.time()) - offset_sec # UTC epoch seconds
    event_time_raw = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(event_epoch))

    return {
        "event_id": str(uuid.uuid4()),
        "event_time_raw": event_time_raw,
        "source": "sim_telemetry",
        "entity_type": "vehicle",
        "entity_id": f"veh_{random.randint(100, 130)}",
        "metric": metric,
        "value": float(value),
        "unit": "sec" if metric == "delay_sec" else "pct",
        "attrs": {"route_id": random.choice(ROUTES),
                  "stop_id": str(random.randint(1, 60))}
    }

def ingest_batch(batch_id, n=200):
    # Generate simulated events
    rows = [make_event(i) for i in range(n)]
    # Create Spark DataFrame
    df = spark.createDataFrame(rows)    

    # Parse event time and add ingest timestamp
    df2 = (df
           .withColumn("event_time_ts", F.to_timestamp("event_time_raw", "yyyy-MM-dd'T'HH:mm:ss'Z'"))
           .withColumn("ingest_time_ts", F.current_timestamp())
    )

    # Append to Bronze table
    (df2.write.mode("append").saveAsTable("bronze_events"))
    print(f"Batch {batch_id} appended to bronze_events: {n} rows")

ingest_batch(0, 200)
display(spark.table("bronze_events").orderBy(F.col("ingest_time_ts").desc()).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silver layer: transit metrics
# MAGIC
# MAGIC Aggregate route events into 5-minute windows and compute basic volume and latency signals.

# COMMAND ----------

TRANSIT_WINDOW = "5 minutes"    # window size
LOOKBACK_MINUTES = 60           # filter recent events
LATE_THRESHOLD_SEC = 120        # late-event threshold

# COMMAND ----------

bronze_all = spark.table("bronze_events")

# Filter to simulated transit events
bronze_recent = (
    bronze_all
    .filter(F.col("source") == F.lit("sim_telemetry"))
    .filter(F.col("event_time_ts") >= (F.current_timestamp() - F.expr(f"INTERVAL 1 MINUTE * {LOOKBACK_MINUTES}")))
)

bronze_enriched = (
    bronze_recent
    .withColumn("route_id", F.col("attrs").getItem("route_id"))
    .withColumn(
        "ingest_delay_sec_raw",
        F.unix_timestamp("ingest_time_ts") - F.unix_timestamp("event_time_ts")
    )
    # Negative delay => clock skew (event_time > ingest_time). Keep raw, clamp for metrics.
    .withColumn("is_clock_skew", F.col("ingest_delay_sec_raw") < F.lit(0))
    .withColumn("ingest_delay_sec", F.greatest(F.col("ingest_delay_sec_raw"), F.lit(0)))
    .withColumn("is_late_event", F.col("ingest_delay_sec") > F.lit(LATE_THRESHOLD_SEC))
)

# Windowed aggeration
silver_transit = (
    bronze_enriched
    .groupBy(
        F.window("event_time_ts", TRANSIT_WINDOW).alias("window"),
        F.col("metric"),
        F.col("route_id")
    )
    .agg(
        F.avg("value").alias("avg_value"),
        F.count(F.lit(1)).alias("n_events"),
        F.avg(F.col("ingest_delay_sec")).alias("avg_ingest_delay_sec"),
        F.sum(F.col("is_late_event").cast("int")).alias("n_late_events"),
        F.sum(F.col("is_clock_skew").cast("int")).alias("n_clock_skew")
    )
    .withColumn("late_event_rate", 
                F.when(F.col("n_events") == 0, F.lit(0.0))
                .otherwise(F.col("n_late_events") / F.col("n_events"))
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
        F.col("n_clock_skew")
    )
)

# Persist Silver output
(silver_transit.write.mode("overwrite").saveAsTable("silver_transit_metrics"))
display(spark.table("silver_transit_metrics").orderBy(F.col("window_start").desc()).limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation checks
# MAGIC
# MAGIC Basic sanity checks on Silver output.

# COMMAND ----------

# Row count
spark.sql("SELECT COUNT(*) AS c FROM silver_transit_metrics").show()

# Latest window
spark.sql("""
SELECT MAX(window_end) AS latest_window_end
FROM silver_transit_metrics
""").show()

# Check duplicate keys (window_start, metric, route_id)
spark.sql("""
SELECT window_start, metric, route_id, COUNT(*) AS row_count
FROM silver_transit_metrics
GROUP BY window_start, metric, route_id
HAVING row_count > 1
""").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Weather ingestion (FMI WFS)
# MAGIC
# MAGIC Fetch recent weather observations and append to Bronze.

# COMMAND ----------

import requests
import xml.etree.ElementTree as ET
import uuid
from datetime import datetime, timedelta, timezone

FMI_WFS = "https://opendata.fmi.fi/wfs"

def fetch_fmi_timevaluepair(place="helsinki", params="t2m", minutes=60):
    """
    Fetch recent FMI observations (WFS timevaluepair).

    Args:
        place: Location name used by FMI (default: "helsinki").
        params: Comma-separated FMI parameter names (e.g. "t2m", "ws_10min").
        minutes: Lookback window in minutes.

    Returns:
        Raw XML response text from FMI WFS.
    """
    now = datetime.now(timezone.utc)
    start = now - timedelta(minutes=minutes)

    q = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "getFeature",
        "storedquery_id": "fmi::observations::weather::timevaluepair",
        "place": place,
        "parameters": params,
        "starttime": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "endtime": now.strftime("%Y-%m-%dT%H:%M:%SZ")
    }

    r = requests.get(FMI_WFS, params=q, timeout=60)
    r.raise_for_status()
    return r.text

def parse_fmi_events(xml_text, place="helsinki"):
    """
    Parse FMI WFS timevaluepair XML into event rows.

    Extracts (time, value) points and attaches minimal station metadata
    (id, name, lat/lon) when available.

    Args:
        xml_text: Raw XML text returned by FMI WFS.
        place: Place name used for the request context (fallback id).

    Returns:
        List of dict rows compatible with the Bronze schema.
    """
    ns = {
        "wfs": "http://www.opengis.net/wfs/2.0",
        "gml": "http://www.opengis.net/gml/3.2",
        "om": "http://www.opengis.net/om/2.0",
        "wml2": "http://www.opengis.net/waterml/2.0",
        "target": "http://xml.fmi.fi/namespace/om/atmosphericfeatures/1.1",
        "xlink": "http://www.w3.org/1999/xlink",
    }

    def safe_text(node):
        if node is None or node.text is None:
            return None
        return node.text.strip()
    
    def xlink_href(node):
        if node is None:
            return None
        return node.attrib.get("{" + ns["xlink"] + "}href")
    
    def infer_metric(href):
        if not href:
            return None
        if "param=" in href:
            tail = href.split("param=", 1)[1] # split once, in href &amp <=> &
            return tail.split("&", 1)[0] 
        return None
    
    root = ET.fromstring(xml_text)

    events = []

    for member in root.findall(".//wfs:member", ns):
        # Extract metric
        observed = member.find(".//om:observedProperty", ns)
        metric = infer_metric(xlink_href(observed)) or "t2m" # fallback

        # Station metadata
        fmisid = safe_text(member.find(".//gml:identifier", ns))
        station_name = safe_text(member.find(".//gml:Point/gml:name", ns)) or safe_text(member.find(".//gml:name", ns))
        region = safe_text(member.find(".//target:region", ns))

        pos_text = safe_text(member.find(".//gml:Point/gml:pos", ns))   # "lat lon"
        lat, lon = None, None
        if pos_text:
            parts = pos_text.split()
            if len(parts) >= 2:
                lat, lon = parts[0], parts[1]

        # Time-value pairs
        for tvp in member.findall(".//wml2:MeasurementTVP", ns):
            t = safe_text(tvp.find("./wml2:time", ns))
            v = safe_text(tvp.find("./wml2:value", ns))
            if t is None or v is None:
                continue
            try:
                v_float = float(v)
            except ValueError:
                continue

            events.append({
                "event_id": str(uuid.uuid4()),
                "event_time_raw": t,     # ISO UTC string
                "source": "fmi_weather",
                "entity_type": "weather_station",
                "entity_id": fmisid or place,
                "metric": metric,
                "value": v_float,
                "unit": "C",     # MVP: t2m is Celsius
                "attrs": {
                    "place": place,
                    "station_name": station_name,
                    "region": region,
                    "lat": lat,
                    "lon": lon,
                    "fmisid": fmisid,
                    "observed_property_href": xlink_href(observed),
                }
            })
    return events
    
def ingest_fmi(place="helsinki", params="t2m", minutes=360):
    xml_text = fetch_fmi_timevaluepair(place=place, params=params, minutes=minutes)
    rows = parse_fmi_events(xml_text, place=place)

    member_count = xml_text.count("<wfs:member")
    if not rows:
        msg = (
        f"No FMI rows parsed (place={place}, params={params}, minutes={minutes}). "
        f"WFS members found: {member_count}. "
        "Next: verify storedquery/parameters and XML namespaces; "
        "if members=0, consider widening the time window."
        )
        print(msg)
        return
    
    df = spark.createDataFrame(rows)
    df2 = (df
           .withColumn("event_time_ts", F.to_timestamp("event_time_raw", "yyyy-MM-dd'T'HH:mm:ssX"))
           .withColumn("ingest_time_ts", F.current_timestamp())
    )

    df2.write.mode("append").saveAsTable("bronze_events")
    print(f"Appended FMI events: {len(rows)}")

# Example run
ingest_fmi(place="helsinki", params="t2m", minutes=360)

# COMMAND ----------

# Check duplicate keys
spark.sql("""
SELECT metric, entity_id, event_time_ts, COUNT(*) AS row_count
FROM bronze_events
WHERE source='fmi_weather'
GROUP BY metric, entity_id, event_time_ts
HAVING row_count > 1
""").show(truncate=False)

# COMMAND ----------

# Preview recent FMI rows
spark.sql("""
SELECT event_time_ts,
       value,
       attrs.station_name,
       attrs.lat,
       attrs.lon
FROM bronze_events
WHERE source='fmi_weather'
ORDER BY ingest_time_ts DESC
LIMIT 10
""").show(truncate=False)

# COMMAND ----------

# Row count
spark.sql("""
SELECT COUNT(*) FROM bronze_events WHERE source='fmi_weather'
""").show()

# COMMAND ----------

# Latest event time
spark.sql("""
SELECT MAX(event_time_ts) FROM bronze_events WHERE source='fmi_weather'
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silver: weather metrics (windowed)

# COMMAND ----------

WEATHER_WINDOW = "15 minutes"
WEATHER_LOOKBACK_MINUTES = 360  # 6 hours

# COMMAND ----------

bronze_all = spark.table("bronze_events")

weather_recent = (
    bronze_all
    .filter(F.col("source") == F.lit("fmi_weather"))
    .filter(F.col("event_time_ts") >= (F.current_timestamp() - F.expr(f"INTERVAL 1 MINUTE * {WEATHER_LOOKBACK_MINUTES}")))
    .withColumn("station_id", F.col("entity_id"))   # entity_id already is fmisid or place
)

# Basic filtering for MVP
weather_clean = (
    weather_recent
    .filter(F.col("value").isNotNull())   # keep True, discard False
    # Filter extreme temperature values
    .filter(~((F.col("metric") == F.lit("t2m")) & ((F.col("value") < -60) | (F.col("value") > 60))))
)

silver_weather = (
    weather_clean
    .groupBy(
        F.window("event_time_ts", WEATHER_WINDOW).alias("window"),
        F.col("metric"),
        F.col("station_id")
    )
    .agg(
        F.avg("value").alias("avg_value"),
        F.count(F.lit(1)).alias("n_events"),
        F.avg(
            F.greatest(
                F.unix_timestamp("ingest_time_ts") - F.unix_timestamp("event_time_ts"),
                F.lit(0)
            )
        ).alias("avg_ingest_delay_sec")
    )
    .select(
        F.col("window.start").alias("window_start"),
        F.col("window.end").alias("window_end"),
        F.col("metric"),
        F.col("station_id"),
        F.col("avg_value"),
        F.col("n_events"),
        F.col("avg_ingest_delay_sec")
    )
)

# Persist Silver output
(silver_weather.write.mode("overwrite").saveAsTable("silver_weather_metrics"))
display(spark.table("silver_weather_metrics").orderBy(F.col("window_start").desc()).limit(20))

# COMMAND ----------

# Row Count
spark.sql("SELECT COUNT(*) AS row_count FROM silver_weather_metrics").show()

# Latest window
spark.sql("""
SELECT MAX(window_end) AS latest_window_end
FROM silver_weather_metrics
""").show()

# Check duplicate keys
spark.sql("""
SELECT window_start, metric, station_id, COUNT(*) AS row_count
FROM silver_weather_metrics
GROUP BY window_start, metric, station_id
HAVING row_count > 1
""").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gold: route KPI window

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Gold: route KPIs per window
# MAGIC CREATE OR REPLACE TABLE gold_route_kpi_window
# MAGIC USING DELTA
# MAGIC AS
# MAGIC WITH s AS (
# MAGIC   SELECT *
# MAGIC   FROM silver_transit_metrics
# MAGIC ),
# MAGIC wide AS (
# MAGIC   SELECT 
# MAGIC     window_start,
# MAGIC     window_end,
# MAGIC     route_id,
# MAGIC     
# MAGIC     -- KPI metrics
# MAGIC     MAX(CASE WHEN metric = 'delay_sec' THEN avg_value END) AS avg_delay_sec,
# MAGIC     MAX(CASE WHEN metric = 'occupancy' THEN avg_value END) AS avg_occupancy_pct,
# MAGIC
# MAGIC     -- Event counts
# MAGIC     MAX(CASE WHEN metric = 'delay_sec' THEN n_events END) AS n_events_delay,
# MAGIC     MAX(CASE WHEN metric = 'occupancy' THEN n_events END) AS n_events_occupancy,
# MAGIC
# MAGIC     -- Latency signals (based on delay metric)
# MAGIC     MAX(CASE WHEN metric = 'delay_sec' THEN late_event_rate END) AS late_rate_delay,
# MAGIC     MAX(CASE WHEN metric = 'delay_sec' THEN avg_ingest_delay_sec END) AS avg_ingest_delay_sec,
# MAGIC     MAX(CASE WHEN metric = 'delay_sec' THEN n_clock_skew END) AS n_clock_skew
# MAGIC         
# MAGIC   FROM s
# MAGIC   GROUP BY window_start, window_end, route_id
# MAGIC )
# MAGIC SELECT
# MAGIC   *,
# MAGIC   CASE
# MAGIC     WHEN COALESCE(n_clock_skew, 0.0) > 0 THEN 'CLOCK_SKEW'
# MAGIC     WHEN COALESCE(n_events_delay, 0) < 5 THEN 'LOW_VOLUME'
# MAGIC     WHEN COALESCE(late_rate_delay, 0.0) > 0.30 THEN 'HIGH_LATE_RATE'
# MAGIC     ELSE 'OK'
# MAGIC   END AS dq_flag
# MAGIC FROM wide;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 'row_count' AS metric, CAST(COUNT(*) AS STRING) AS value
# MAGIC FROM gold_route_kpi_window
# MAGIC
# MAGIC UNION ALL
# MAGIC SELECT 'latest_window_end' AS metric, CAST(MAX(window_end) AS STRING) AS value
# MAGIC FROM gold_route_kpi_window
# MAGIC
# MAGIC UNION ALL
# MAGIC SELECT CONCAT('dq_flag=', dq_flag) AS metric, CAST(COUNT(*) AS STRING) AS value
# MAGIC FROM gold_route_kpi_window
# MAGIC GROUP BY dq_flag;

# COMMAND ----------

# MAGIC %sql
# MAGIC --- Check duplicates
# MAGIC SELECT window_start, route_id, COUNT(*) AS row_count
# MAGIC FROM gold_route_kpi_window
# MAGIC GROUP BY window_start, route_id
# MAGIC HAVING row_count > 1
# MAGIC ORDER BY row_count DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gold: daily route KPIs

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE gold_route_kpi_daily
# MAGIC USING DELTA
# MAGIC AS
# MAGIC SELECT
# MAGIC   DATE(window_start) AS date,
# MAGIC   route_id,
# MAGIC
# MAGIC   AVG(avg_delay_sec) AS avg_delay_sec,
# MAGIC   AVG(avg_occupancy_pct) AS avg_occupancy_pct,
# MAGIC
# MAGIC   SUM(COALESCE(n_events_delay, 0)) AS total_events_delay,
# MAGIC   AVG(COALESCE(late_rate_delay, 0.0)) AS avg_late_rate_delay,
# MAGIC   AVG(COALESCE(avg_ingest_delay_sec, 0.0)) AS avg_ingest_delay_sec
# MAGIC
# MAGIC FROM gold_route_kpi_window
# MAGIC GROUP BY DATE(window_start), route_id;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Preview daily KPIs
# MAGIC SELECT * FROM gold_route_kpi_daily ORDER BY date DESC, route_id LIMIT 50;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gold: pipeline window summary

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE gold_pipeline_health_window
# MAGIC USING DELTA
# MAGIC AS
# MAGIC
# MAGIC WITH transit AS (
# MAGIC   SELECT
# MAGIC     window_start,
# MAGIC     window_end,
# MAGIC     SUM(n_events) AS transit_total_events,
# MAGIC     AVG(avg_ingest_delay_sec) AS transit_avg_ingest_delay_sec
# MAGIC   FROM silver_transit_metrics
# MAGIC   GROUP BY window_start, window_end
# MAGIC ),
# MAGIC weather AS (
# MAGIC   SELECT
# MAGIC     window_start,
# MAGIC     window_end,
# MAGIC     SUM(n_events) AS weather_total_events,
# MAGIC     AVG(avg_ingest_delay_sec) AS weather_avg_ingest_delay_sec
# MAGIC   FROM silver_weather_metrics
# MAGIC   GROUP BY window_start, window_end
# MAGIC )
# MAGIC SELECT
# MAGIC   COALESCE(t.window_start, w.window_start) AS window_start,
# MAGIC   COALESCE(t.window_end , w.window_end) AS window_end,
# MAGIC   t.transit_total_events,
# MAGIC   w.weather_total_events,
# MAGIC   t.transit_avg_ingest_delay_sec,
# MAGIC   w.weather_avg_ingest_delay_sec
# MAGIC FROM transit t
# MAGIC FULL OUTER JOIN weather w
# MAGIC ON t.window_start = w.window_start;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM gold_pipeline_health_window
# MAGIC ORDER BY window_start DESC
# MAGIC LIMIT 20;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic visual checks

# COMMAND ----------

def plot_transit_total_events(table_name="gold_route_kpi_window"):
    """
    Plot total transit events per window (throughput signal).

    Args:
        table_name: Gold table name with route KPIs.
    """

    gold_df = spark.table(table_name)

    volume_df = (
        gold_df
        .groupBy("window_start")
        .agg(
            F.sum(
                F.coalesce(F.col("n_events_delay"), F.lit(0))
            ).alias("transit_total_events")
        )
        .orderBy("window_start")
    )

    pdf = volume_df.toPandas()

    plt.figure()
    plt.plot(pdf["window_start"], pdf["transit_total_events"])
    plt.xticks(rotation=45)
    plt.title("Transit total events per window")
    plt.xlabel("Window Start")
    plt.ylabel("Total Events")
    plt.tight_layout()
    plt.show()

plot_transit_total_events()

# COMMAND ----------

def plot_transit_avg_ingest_delay(table_name="gold_route_kpi_window"):
    """
    Plot average ingest delay per window (latency signal).

    Args:
        table_name: Gold table name with route KPIs.
    """

    gold_df = spark.table(table_name)

    delay_df = (
        gold_df
        .groupBy("window_start")
        .agg(
            F.avg("avg_ingest_delay_sec").alias("transit_avg_ingest_delay_sec")
        )
        .orderBy("window_start")
    )

    pdf = delay_df.toPandas()

    plt.figure()
    plt.plot(pdf["window_start"], pdf["transit_avg_ingest_delay_sec"])
    plt.xticks(rotation=45)
    plt.title("Transit avg ingest delay per window")
    plt.xlabel("Window Start")
    plt.ylabel("Avg Ingest Delay (sec)")
    plt.tight_layout()
    plt.show()

plot_transit_avg_ingest_delay()

# COMMAND ----------

def plot_route_delay_window(route_id: str = "M1", freq: str = "5min"):
    """
    Plot windowed delay KPI for a single route.

    Args:
        route_id: Route identifier (e.g. "M1").
        freq: Pandas offset alias for resampling (e.g. "5min", "15min").

    Notes:
        Missing windows stay as NaN to keep gaps visible.
    """
    gold_df = spark.table("gold_route_kpi_window")

    route_pd = (
        gold_df
        .filter(F.col("route_id") == route_id)
        .select("window_start", "avg_delay_sec")
        .orderBy("window_start")
        .toPandas()
    )

    route_pd["window_start"] = pd.to_datetime(route_pd["window_start"])
    route_pd = route_pd.set_index("window_start").sort_index().asfreq(freq)

    # Ensure numeric type
    route_pd["avg_delay_sec"] = pd.to_numeric(route_pd["avg_delay_sec"], errors="coerce")

    plt.figure()
    plt.step(route_pd.index, route_pd["avg_delay_sec"], where="post")
    plt.xticks(rotation=45)
    plt.title("Route M1 avg delay per window")
    plt.xlabel("Window Start")
    plt.ylabel("Avg Delay (sec)")
    plt.tight_layout()
    plt.show()

# Example
plot_route_delay_window("M1")
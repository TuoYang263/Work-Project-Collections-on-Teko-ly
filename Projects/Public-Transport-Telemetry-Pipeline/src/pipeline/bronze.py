"""
Bronze-layer ingestion functions.

This module contains:
- simulated public transport telemetry ingestion
- FMI weather ingestion
"""

from __future__ import annotations

import os
import time
import uuid
import logging
import random
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from itertools import product
from typing import Dict, List
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from src.pipeline.setup import initialize_environment
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


from .config import (
    BRONZE_EVENTS_TABLE,
    BRONZE_EVENTS_PATH,
    FMI_ALLOW_FAILURE,
    FMI_BACKOFF_FACTOR,
    FMI_DEFAULT_LOOKBACK_MINUTES,
    FMI_DEFAULT_PARAMS,
    FMI_DEFAULT_PLACE,
    FMI_PLACES,
    FMI_MAX_RETRIES,
    FMI_REQUEST_TIMEOUT_CONNECT,
    FMI_REQUEST_TIMEOUT_READ,
    FMI_WFS_URL,
    SIM_DEFAULT_BATCH_SIZE,
    SIM_ROUTE_IDS,
    SIM_HISTORY_WINDOWS,
    SIM_WINDOW_MINUTES,
    SIM_EVENTS_PER_ROUTE_WINDOW,
    SIM_INGEST_DELAY_MIN_SEC,
    SIM_INGEST_DELAY_MAX_SEC,
)


def infer_unit(metric: str) -> str:
    if metric == "t2m":
        return "C"
    if metric in {"r_1h", "precipitation", "r1_10min"}:
        return "mm"
    return "unknown"


def run_bronze_layer(
    spark: SparkSession,
    logger: logging.Logger,
    reset: bool = True,
) -> None:
    logger.info("Bronze layer started.")

    logger.info("DEBUG: before initialize_environment")
    initialize_environment(spark, reset=reset)
    logger.info("DEBUG: after initialize_environment")

    logger.info("DEBUG: before ingest_simulated_transit_batch")
    ingest_simulated_transit_batch(
        spark=spark,
        batch_id=0,
        n=SIM_DEFAULT_BATCH_SIZE,
    )
    logger.info("DEBUG: after ingest_simulated_transit_batch")
    logger.info("Simulated transit batch appended to Bronze.")

    try:
        logger.info("DEBUG: before ingest_fmi_weather_for_places")
        ingest_fmi_weather_for_places(
            spark=spark,
            places=FMI_PLACES,
            params=FMI_DEFAULT_PARAMS,
            minutes=FMI_DEFAULT_LOOKBACK_MINUTES,
        )
        logger.info("DEBUG: after ingest_fmi_weather_for_places")
        logger.info(
            "FMI weather ingest completed "
            f"(places={FMI_PLACES}, "
            f"params={FMI_DEFAULT_PARAMS}, "
            f"minutes={FMI_DEFAULT_LOOKBACK_MINUTES})."
        )
    except Exception as exc:
        if FMI_ALLOW_FAILURE:
            logger.warning(
                "FMI weather ingest failed, continuing without weather data.",
                exc_info=True,
            )
        else:
            logger.exception("FMI weather ingest failed.")
            raise

    if os.getenv("DATABRICKS_RUNTIME_VERSION"):
        bronze_count = spark.read.format("delta").load(str(BRONZE_EVENTS_PATH)).count()
    else:
        bronze_count = spark.table(BRONZE_EVENTS_TABLE).count()
    logger.info(f"Bronze table updated: {BRONZE_EVENTS_TABLE}")
    logger.info(f"Bronze row count after transit + weather ingest: {bronze_count}")


# -----------------------------------------------------------------------------
# Transit simulation
# -----------------------------------------------------------------------------


# Optional parameters allow tests or callers to override specific fields.
# When not provided, the function generates random/default values internally.
def make_simulated_event(
    route_ids: List[str] | None = None,
    route_id: str | None = None,
    metric: str | None = None,
    event_epoch: int | None = None,
) -> Dict:
    """
    Build one simulated telemetry event row compatible with the Bronze schema.

    The event time can be injected so demo data can cover multiple historical
    windows instead of only the latest few minutes.
    """
    route_ids = route_ids or SIM_ROUTE_IDS
    route_id = route_id or random.choice(route_ids)
    metric = metric or random.choice(["delay_sec", "occupancy"])

    if metric == "delay_sec":
        # Simulate vehicle delay in seconds.
        # Most values are close to schedule, with occasional operational delays.
        r = random.random()

        if r < 0.65:
            value = random.randint(-30, 90)
        elif r < 0.90:
            value = random.randint(90, 180)
        else:
            value = random.randint(180, 420)

        unit = "sec"
    else:
        value = random.randint(10, 85)
        unit = "pct"

    if event_epoch is None:
        event_epoch = int(time.time())

    ingest_delay_sec = random.randint(
        SIM_INGEST_DELAY_MIN_SEC,
        SIM_INGEST_DELAY_MAX_SEC,
    )
    ingest_epoch = event_epoch + ingest_delay_sec

    event_time_raw = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(event_epoch))
    ingest_time_raw = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ingest_epoch))

    return {
        "event_id": str(uuid.uuid4()),
        "event_time_raw": event_time_raw,
        "sim_ingest_time_raw": ingest_time_raw,
        "source": "sim_transit",
        "entity_type": "vehicle",
        "entity_id": f"veh_{random.randint(100, 130)}",
        "metric": metric,
        "value": float(value),
        "unit": unit,
        "attrs": {
            "route_id": route_id,
            "stop_id": str(random.randint(1, 60)),
        },
    }


def generate_route_window_events(
    route_ids: List[str],
    metrics: List[str],
    window_starts: List[int],
    window_sec: int,
    events_per_metric: int,
) -> List[Dict]:
    """
    Generate stable route-window telemetry coverage.

    Each route receives delay and occupancy events in each historical window.
    This keeps dashboard trend charts meaningful without creating frontend-only mock data.
    """
    rows: List[Dict] = []

    for window_start_epoch, route_id, metric in product(
        window_starts, route_ids, metrics
    ):
        for _ in range(events_per_metric):
            event_epoch = random.randint(
                window_start_epoch,
                window_start_epoch + window_sec - 1,
            )

            rows.append(
                make_simulated_event(
                    route_ids=route_ids,
                    route_id=route_id,
                    metric=metric,
                    event_epoch=event_epoch,
                )
            )

    return rows


def generate_random_fill_events(
    route_ids: List[str],
    earliest_window_start: int,
    latest_completed_window_start: int,
    window_sec: int,
    n: int,
) -> List[Dict]:
    """
    Generate additional random events across the same historical time range.

    These events add variability while the stable route-window coverage ensures
    every route has enough records for dashboard trends.
    """
    rows: List[Dict] = []

    for _ in range(max(0, n)):
        event_epoch = random.randint(
            earliest_window_start,
            latest_completed_window_start + window_sec - 1,
        )

        rows.append(
            make_simulated_event(
                route_ids=route_ids,
                event_epoch=event_epoch,
            )
        )

    return rows


def ingest_simulated_transit_batch(
    spark: SparkSession,
    batch_id: int = 0,
    n: int = SIM_DEFAULT_BATCH_SIZE,
) -> None:
    """
    Generate simulated transit telemetry and append it to Bronze.

    Strategy:
    - generate multiple historical 10-minute windows
    - guarantee delay and occupancy events for every route in every window
    - add random fill events to preserve variability
    - keep the batch lightweight enough for local and GitHub Actions runs
    """
    route_ids = SIM_ROUTE_IDS
    metrics = ["delay_sec", "occupancy"]

    window_sec = SIM_WINDOW_MINUTES * 60
    now_epoch = int(time.time())

    # Use the latest completed window to avoid future-looking window_end values.
    latest_completed_window_start = (now_epoch // window_sec) * window_sec - window_sec

    window_starts = [
        latest_completed_window_start
        - (SIM_HISTORY_WINDOWS - 1 - window_idx) * window_sec
        for window_idx in range(SIM_HISTORY_WINDOWS)
    ]

    events_per_metric = max(
        1,
        SIM_EVENTS_PER_ROUTE_WINDOW // len(metrics),
    )

    stable_rows = generate_route_window_events(
        route_ids=route_ids,
        metrics=metrics,
        window_starts=window_starts,
        window_sec=window_sec,
        events_per_metric=events_per_metric,
    )

    remaining = max(0, n - len(stable_rows))

    fill_rows = generate_random_fill_events(
        route_ids=route_ids,
        earliest_window_start=window_starts[0],
        latest_completed_window_start=latest_completed_window_start,
        window_sec=window_sec,
        n=remaining,
    )

    print("DEBUG: before generate simulated rows", flush=True)

    rows = stable_rows + fill_rows

    print(f"DEBUG: generated rows: {len(rows)}", flush=True)

    print("DEBUG: before createDataFrame", flush=True)
    df = spark.createDataFrame(rows)
    print("DEBUG: after createDataFrame", flush=True)

    print("DEBUG: before transform bronze df", flush=True)
    df2 = (
        df.withColumn(
            "event_time_ts",
            F.to_timestamp("event_time_raw", "yyyy-MM-dd'T'HH:mm:ss'Z'"),
        )
        .withColumn(
            "ingest_time_ts",
            F.to_timestamp("sim_ingest_time_raw", "yyyy-MM-dd'T'HH:mm:ss'Z'"),
        )
        .select(
            "event_id",
            "event_time_raw",
            "source",
            "entity_type",
            "entity_id",
            "metric",
            "value",
            "unit",
            "attrs",
            "event_time_ts",
            "ingest_time_ts",
        )
    )
    print("DEBUG: after transform bronze df", flush=True)

    if os.getenv("DATABRICKS_RUNTIME_VERSION"):
        print(f"DEBUG: before delta write to {BRONZE_EVENTS_PATH}", flush=True)
        df2.write.format("delta").mode("append").save(str(BRONZE_EVENTS_PATH))
        print("DEBUG: after delta write", flush=True)
    else:
        print("DEBUG: before saveAsTable", flush=True)
        df2.write.format("delta").mode("append").saveAsTable(BRONZE_EVENTS_TABLE)
        print("DEBUG: after saveAsTable", flush=True)

    print(
        f"Batch {batch_id} appended to {BRONZE_EVENTS_TABLE}: "
        f"{len(rows)} rows "
        f"({len(stable_rows)} stable coverage rows, {len(fill_rows)} random fill rows) "
        f"across {SIM_HISTORY_WINDOWS} windows"
    )


# -----------------------------------------------------------------------------
# FMI weather ingestion
# -----------------------------------------------------------------------------


def build_fmi_retry_session() -> requests.Session:
    """
    Build a requests session with retry handling for transient FMI failures.
    """
    retry = Retry(
        total=FMI_MAX_RETRIES,
        connect=FMI_MAX_RETRIES,
        read=FMI_MAX_RETRIES,
        backoff_factor=FMI_BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_fmi_timevaluepair(
    place: str = FMI_DEFAULT_PLACE,
    params: str = FMI_DEFAULT_PARAMS,
    minutes: int = FMI_DEFAULT_LOOKBACK_MINUTES,
) -> str:
    """
    Fetch recent FMI observations (WFS timevaluepair format).
    """
    now = datetime.now(timezone.utc)
    start = now - timedelta(minutes=minutes)

    query = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "getFeature",
        "storedquery_id": "fmi::observations::weather::timevaluepair",
        "place": place,
        "parameters": params,
        "starttime": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "endtime": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    session = build_fmi_retry_session()
    response = session.get(
        FMI_WFS_URL,
        params=query,
        timeout=(FMI_REQUEST_TIMEOUT_CONNECT, FMI_REQUEST_TIMEOUT_READ),
    )
    response.raise_for_status()
    return response.text


def parse_fmi_events(xml_text: str, place: str = FMI_DEFAULT_PLACE) -> List[Dict]:
    """
    Parse FMI WFS XML into Bronze-compatible event rows.
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
            tail = href.split("param=", 1)[1]
            return tail.split("&", 1)[0]
        return None

    root = ET.fromstring(xml_text)
    events = []

    for member in root.findall(".//wfs:member", ns):
        observed = member.find(".//om:observedProperty", ns)
        metric = infer_metric(xlink_href(observed)) or "t2m"

        fmisid = safe_text(member.find(".//gml:identifier", ns))
        station_name = safe_text(member.find(".//gml:Point/gml:name", ns)) or safe_text(
            member.find(".//gml:name", ns)
        )
        region = safe_text(member.find(".//target:region", ns))

        pos_text = safe_text(member.find(".//gml:Point/gml:pos", ns))
        lat, lon = None, None
        if pos_text:
            parts = pos_text.split()
            if len(parts) >= 2:
                lat, lon = parts[0], parts[1]

        for tvp in member.findall(".//wml2:MeasurementTVP", ns):
            t = safe_text(tvp.find("./wml2:time", ns))
            v = safe_text(tvp.find("./wml2:value", ns))

            if t is None or v is None:
                continue

            try:
                v_float = float(v)
            except ValueError:
                continue

            events.append(
                {
                    "event_id": str(uuid.uuid4()),
                    "event_time_raw": t,
                    "source": "fmi_weather",
                    "entity_type": "weather_station",
                    "entity_id": fmisid or place,
                    "metric": metric,
                    "value": v_float,
                    "unit": infer_unit(metric),
                    "attrs": {
                        "place": place,
                        "station_name": station_name,
                        "region": region,
                        "lat": lat,
                        "lon": lon,
                        "fmisid": fmisid,
                        "observed_property_href": xlink_href(observed),
                    },
                }
            )

    return events


def ingest_fmi_weather(
    spark: SparkSession,
    place: str = FMI_DEFAULT_PLACE,
    params: str = FMI_DEFAULT_PARAMS,
    minutes: int = FMI_DEFAULT_LOOKBACK_MINUTES,
) -> None:
    """
    Fetch, parse, and append FMI weather events to Bronze.
    """
    xml_text = fetch_fmi_timevaluepair(
        place=place,
        params=params,
        minutes=minutes,
    )
    rows = parse_fmi_events(xml_text, place=place)

    if not rows:
        print(
            f"No FMI rows parsed "
            f"(place={place}, params={params}, minutes={minutes})."
        )
        return

    df = spark.createDataFrame(rows)
    df2 = df.withColumn(
        "event_time_ts", F.to_timestamp("event_time_raw", "yyyy-MM-dd'T'HH:mm:ssX")
    ).withColumn("ingest_time_ts", F.current_timestamp())

    if os.getenv("DATABRICKS_RUNTIME_VERSION"):
        df2.write.format("delta").mode("append").save(str(BRONZE_EVENTS_PATH))
    else:
        df2.write.format("delta").mode("append").saveAsTable(BRONZE_EVENTS_TABLE)
    print(f"Appended FMI events: {len(rows)}")


def ingest_fmi_weather_for_places(
    spark: SparkSession,
    places: List[str],
    params: str = FMI_DEFAULT_PARAMS,
    minutes: int = FMI_DEFAULT_LOOKBACK_MINUTES,
) -> None:
    """
    Fetch FMI weather observations for multiple nearby place queries and append
    deduplicated station observations to Bronze.

    Different place queries can resolve to the same FMI station, so rows are
    deduplicated by station, metric, and observation timestamp.
    """
    all_rows: List[Dict] = []

    for place in places:
        xml_text = fetch_fmi_timevaluepair(
            place=place,
            params=params,
            minutes=minutes,
        )
        rows = parse_fmi_events(xml_text, place=place)
        all_rows.extend(rows)

    if not all_rows:
        print(
            f"No FMI rows parsed "
            f"(places={places}, params={params}, minutes={minutes})."
        )
        return

    df = spark.createDataFrame(all_rows)

    df2 = (
        df.withColumn(
            "event_time_ts",
            F.to_timestamp("event_time_raw", "yyyy-MM-dd'T'HH:mm:ssX"),
        )
        .withColumn("ingest_time_ts", F.current_timestamp())
        .dropDuplicates(["entity_id", "metric", "event_time_raw"])
    )

    row_count = df2.count()

    if os.getenv("DATABRICKS_RUNTIME_VERSION"):
        df2.write.format("delta").mode("append").save(str(BRONZE_EVENTS_PATH))
    else:
        df2.write.format("delta").mode("append").saveAsTable(BRONZE_EVENTS_TABLE)

    print(
        f"Appended FMI events: {row_count} deduplicated rows " f"from places={places}"
    )

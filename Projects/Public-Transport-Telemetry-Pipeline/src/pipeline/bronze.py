"""
Bronze-layer ingestion functions.

This module contains:
- simulated public transport telemetry ingestion
- FMI weather ingestion
"""

from __future__ import annotations

import time
import uuid
import logging
import random
import requests
import xml.etree.ElementTree as ET
from typing import Dict, List
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from datetime import datetime, timedelta, timezone
from src.pipeline.setup import initialize_environment

from .config import (
    BRONZE_EVENTS_TABLE,
    FMI_DEFAULT_LOOKBACK_MINUTES,
    FMI_DEFAULT_PARAMS,
    FMI_DEFAULT_PLACE,
    FMI_WFS_URL,
    SIM_DEFAULT_BATCH_SIZE,
    SIM_ROUTE_IDS,
    SIM_TIME_SPAN_MINUTES,
)

def run_bronze_layer(
    spark: SparkSession,
    logger: logging.Logger,
    reset: bool = True,
) -> None:
    logger.info("Bronze layer started.")

    initialize_environment(spark, reset=reset)

    ingest_simulated_transit_batch(
        spark=spark,
        batch_id=0,
        n=SIM_DEFAULT_BATCH_SIZE,
    )

    bronze_count = spark.table(BRONZE_EVENTS_TABLE).count()
    logger.info(f"Bronze table updated: {BRONZE_EVENTS_TABLE}")
    logger.info(f"Bronze row count after transit ingest: {bronze_count}")

# -----------------------------------------------------------------------------
# Transit simulation
# -----------------------------------------------------------------------------

def make_simulated_event(route_ids: List[str] | None = None) -> Dict:
    """
    Build one simulated telemetry event row compatible with the Bronze schema.
    """
    route_ids = route_ids or SIM_ROUTE_IDS

    metric = random.choice(["delay_sec", "occupancy"])
    value = random.randint(-30, 600) if metric == "delay_sec" else random.randint(0, 80)

    offset_sec = random.randint(0, SIM_TIME_SPAN_MINUTES * 60)
    event_epoch = int(time.time()) - offset_sec
    event_time_raw = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(event_epoch))

    return {
        "event_id": str(uuid.uuid4()),
        "event_time_raw": event_time_raw,
        "source": "sim_transit",
        "entity_type": "vehicle",
        "entity_id": f"veh_{random.randint(100, 130)}",
        "metric": metric,
        "value": float(value),
        "unit": "sec" if metric == "delay_sec" else "pct",
        "attrs": {
            "route_id": random.choice(route_ids),
            "stop_id": str(random.randint(1, 60)),
        },
    }


def ingest_simulated_transit_batch(
    spark: SparkSession,
    batch_id: int = 0,
    n: int = SIM_DEFAULT_BATCH_SIZE,
) -> None:
    """
    Generate simulated transit telemetry and append it to Bronze.
    """
    rows = [make_simulated_event() for _ in range(n)]
    df = spark.createDataFrame(rows)

    df2 = (
        df.withColumn("event_time_ts", F.to_timestamp("event_time_raw", "yyyy-MM-dd'T'HH:mm:ss'Z'"))
        .withColumn("ingest_time_ts", F.current_timestamp())
    )

    df2.write.format("delta").mode("append").saveAsTable(BRONZE_EVENTS_TABLE)
    print(f"Batch {batch_id} appended to {BRONZE_EVENTS_TABLE}: {n} rows")


# -----------------------------------------------------------------------------
# FMI weather ingestion
# -----------------------------------------------------------------------------

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

    response = requests.get(FMI_WFS_URL, params=query, timeout=60)
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
        station_name = (
            safe_text(member.find(".//gml:Point/gml:name", ns))
            or safe_text(member.find(".//gml:name", ns))
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
                    "unit": "C",
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
    xml_text = fetch_fmi_timevaluepair(place=place, params=params, minutes=minutes)
    rows = parse_fmi_events(xml_text, place=place)

    member_count = xml_text.count("<wfs:member")
    if not rows:
        print(
            f"No FMI rows parsed (place={place}, params={params}, minutes={minutes}). "
            f"WFS members found: {member_count}."
        )
        return

    df = spark.createDataFrame(rows)
    df2 = (
        df.withColumn("event_time_ts", F.to_timestamp("event_time_raw", "yyyy-MM-dd'T'HH:mm:ssX"))
        .withColumn("ingest_time_ts", F.current_timestamp())
    )

    df2.write.format("delta").mode("append").saveAsTable(BRONZE_EVENTS_TABLE)
    print(f"Appended FMI events: {len(rows)}")
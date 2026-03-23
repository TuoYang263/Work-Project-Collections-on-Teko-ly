from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from typing import Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google.transit import gtfs_realtime_pb2


load_dotenv()

HSL_VEHICLE_POSITIONS_URL = "https://realtime.hsl.fi/realtime/vehicle-positions/v2/hsl"
DIGITRANSIT_BASE_URL = "https://api.digitransit.fi/routing-data/v3/hsl/"

MODE_COLORS = {
    "bus": [255, 140, 0],      # orange
    "tram": [0, 200, 0],       # green
    "metro": [255, 80, 80],    # red
    "rail": [80, 160, 255],    # blue
    "ferry": [180, 120, 255],  # purple
}


def _request_with_retry(
    url: str,
    *,
    headers: dict | None = None,
    timeout: int = 30,
    attempts: int = 3,
) -> requests.Response:
    """
    Small helper for resilient HTTP requests.
    """
    last_error = None

    for _ in range(attempts):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        except Exception as e:
            last_error = e

    raise RuntimeError(f"Request failed after {attempts} attempts: {url} | error={last_error}")


def fetch_hsl_vehicle_positions(
    url: str = HSL_VEHICLE_POSITIONS_URL,
) -> Tuple[gtfs_realtime_pb2.FeedMessage | None, list[dict]]:
    """
    Fetch a snapshot of GTFS-Realtime vehicle positions from HSL.

    Returns:
        feed: Parsed FeedMessage
        records: Flattened list of vehicle position records
    """
    try:
        response = _request_with_retry(url, timeout=30, attempts=3)
    except Exception as e:
        print(f"[WARNING] Failed to fetch HSL vehicle positions: {e}")
        return None, []

    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(response.content)

    records: list[dict] = []

    for entity in feed.entity:
        if not entity.HasField("vehicle"):
            continue

        vehicle = entity.vehicle
        trip = vehicle.trip
        position = vehicle.position
        descriptor = vehicle.vehicle

        records.append(
            {
                "entity_id": entity.id,
                "trip_route_id": trip.route_id if trip.HasField("route_id") else None,
                "trip_direction_id": trip.direction_id if trip.HasField("direction_id") else None,
                "trip_start_date": trip.start_date if trip.HasField("start_date") else None,
                "trip_start_time": trip.start_time if trip.HasField("start_time") else None,
                "vehicle_id": descriptor.id if descriptor.HasField("id") else None,
                "vehicle_label": descriptor.label if descriptor.HasField("label") else None,
                "latitude": position.latitude if position.HasField("latitude") else None,
                "longitude": position.longitude if position.HasField("longitude") else None,
                "bearing": position.bearing if position.HasField("bearing") else None,
                "speed": position.speed if position.HasField("speed") else None,
                "current_status": vehicle.current_status,
                "stop_id": vehicle.stop_id if vehicle.HasField("stop_id") else None,
                "timestamp": (
                    pd.to_datetime(vehicle.timestamp, unit="s", utc=True)
                    if vehicle.HasField("timestamp")
                    else None
                ),
            }
        )

    return feed, records


def download_gtfs_if_needed(gtfs_dir: Path) -> Path:
    """
    Download and extract HSL GTFS static files if they do not already exist locally.
    """
    gtfs_dir.mkdir(parents=True, exist_ok=True)

    required_files = [
        gtfs_dir / "routes.txt",
        gtfs_dir / "trips.txt",
        gtfs_dir / "shapes.txt",
    ]

    if all(path.exists() for path in required_files):
        return gtfs_dir

    digitransit_key = os.getenv("DIGITRANSIT_KEY")
    if not digitransit_key:
        raise ValueError("DIGITRANSIT_KEY is not set")

    headers = {"digitransit-subscription-key": digitransit_key}

    index_response = _request_with_retry(
        DIGITRANSIT_BASE_URL,
        headers=headers,
        timeout=30,
        attempts=3,
    )

    soup = BeautifulSoup(index_response.text, "html.parser")
    links = [a.get("href") for a in soup.find_all("a")]
    zip_files = [l for l in links if l and l.endswith(".zip")]

    target_zip = None
    for z in zip_files:
        if "gtfs" in z.lower():
            target_zip = z
            break

    if target_zip is None:
        raise RuntimeError("No GTFS zip found from Digitransit index page.")

    zip_url = DIGITRANSIT_BASE_URL + target_zip

    zip_response = _request_with_retry(
        zip_url,
        headers=headers,
        timeout=60,
        attempts=3,
    )

    with zipfile.ZipFile(io.BytesIO(zip_response.content)) as zf:
        zf.extractall(gtfs_dir)

    return gtfs_dir


def load_gtfs_tables(gtfs_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load static GTFS tables required for HSL map construction.
    """
    routes = pd.read_csv(gtfs_dir / "routes.txt")
    trips = pd.read_csv(gtfs_dir / "trips.txt")
    shapes = pd.read_csv(gtfs_dir / "shapes.txt")
    return routes, trips, shapes


def map_route_type(rt) -> str | None:
    """
    Map GTFS route_type to normalized transport mode labels.
    Includes HSL-specific GTFS extensions.
    """
    if pd.isna(rt):
        return None

    rt = int(rt)

    # Standard GTFS
    if rt == 0:
        return "tram"
    if rt == 1:
        return "metro"
    if rt == 2:
        return "rail"
    if rt == 3:
        return "bus"
    if rt == 4:
        return "ferry"

    # HSL extension
    if 700 <= rt < 800:
        return "bus"
    if rt == 109:
        return "rail"
    if rt == 900:
        return "ferry"

    return "other"


def build_hsl_map_dataframe(
    raw_records: list[dict],
    routes: pd.DataFrame,
    *,
    lookback_minutes: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build enriched telemetry dataframe and route lookup dataframe.

    Returns:
        df_map: map-ready enriched telemetry base dataframe
        df_route_lookup: lookup dataframe used later for shape/path construction
    """
    df_raw = pd.DataFrame(raw_records)

    if df_raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    df_route_lookup = df_raw.merge(
        routes[["route_id", "route_short_name", "route_type"]],
        left_on="trip_route_id",
        right_on="route_id",
        how="left",
    )

    df_map = df_route_lookup[
        [
            "trip_route_id",
            "route_short_name",
            "route_type",
            "vehicle_id",
            "latitude",
            "longitude",
            "timestamp",
        ]
    ].dropna(subset=["latitude", "longitude"]).copy()

    df_map["route_type"] = df_map["route_type"].astype("Int64")
    df_map["transport_mode"] = df_map["route_type"].apply(map_route_type)

    if lookback_minutes is not None and "timestamp" in df_map.columns:
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=lookback_minutes)
        df_map = df_map[df_map["timestamp"] > cutoff].copy()

    return df_map, df_route_lookup


def filter_data(df: pd.DataFrame, mode: str = "all", route: str = "all") -> pd.DataFrame:
    """
    Filter telemetry dataframe by transport mode and/or route_short_name.
    """
    filtered = df.copy()

    if mode != "all":
        filtered = filtered[filtered["transport_mode"] == mode]

    if route != "all":
        filtered = filtered[filtered["route_short_name"].astype(str) == route]

    return filtered


def build_route_options(df_map: pd.DataFrame) -> pd.DataFrame:
    """
    Build route metadata for UI selection.
    """
    if df_map.empty:
        return pd.DataFrame(
            columns=["trip_route_id", "route_short_name", "transport_mode", "route_label"]
        )

    route_options_df = (
        df_map[["trip_route_id", "route_short_name", "transport_mode"]]
        .dropna(subset=["trip_route_id"])
        .drop_duplicates()
        .sort_values(["transport_mode", "route_short_name"])
        .copy()
    )

    route_options_df["route_label"] = route_options_df["route_short_name"].astype(str)
    return route_options_df


def build_map_points(
    filtered_df: pd.DataFrame,
    *,
    mode_colors: dict | None = None,
) -> pd.DataFrame:
    """
    Build map-ready vehicle points for visualization.
    """
    if mode_colors is None:
        mode_colors = MODE_COLORS

    if filtered_df.empty:
        return pd.DataFrame(
            columns=["lat", "lon", "transport_mode", "route_short_name", "vehicle_id", "color"]
        )

    df_map_view = filtered_df[
        ["latitude", "longitude", "transport_mode", "route_short_name", "vehicle_id"]
    ].dropna().copy()

    df_map_view = df_map_view.rename(columns={"latitude": "lat", "longitude": "lon"})

    df_map_view["color"] = df_map_view["transport_mode"].map(mode_colors)
    df_map_view["color"] = df_map_view["color"].apply(
        lambda x: x if isinstance(x, list) else [255, 255, 255]
    )

    return df_map_view


def build_hsl_paths(
    df_route_lookup: pd.DataFrame,
    trips: pd.DataFrame,
    shapes: pd.DataFrame,
    filtered_df: pd.DataFrame,
    *,
    mode_colors: dict | None = None,
) -> pd.DataFrame:
    """
    Build filtered GTFS route geometry paths consistent with the current telemetry subset.
    """
    if mode_colors is None:
        mode_colors = MODE_COLORS

    if df_route_lookup.empty or trips.empty or shapes.empty or filtered_df.empty:
        return pd.DataFrame(
            columns=["shape_id", "route_short_name", "transport_mode", "path", "color"]
        )

    df_route_shape = df_route_lookup.merge(
        trips[["route_id", "shape_id"]],
        left_on="trip_route_id",
        right_on="route_id",
        how="left",
    )

    target_shapes = (
        df_route_shape[
            df_route_shape["route_short_name"].isin(filtered_df["route_short_name"].unique())
        ][["route_short_name", "shape_id"]]
        .dropna()
        .drop_duplicates()
    )

    if target_shapes.empty:
        return pd.DataFrame(
            columns=["shape_id", "route_short_name", "transport_mode", "path", "color"]
        )

    df_shapes_selected = shapes.merge(
        target_shapes,
        on="shape_id",
        how="inner",
    )

    df_shapes_selected = df_shapes_selected.merge(
        filtered_df[["route_short_name", "transport_mode"]].drop_duplicates(),
        on="route_short_name",
        how="left",
    )

    paths = (
        df_shapes_selected.sort_values(["shape_id", "shape_pt_sequence"])
        .groupby(["shape_id", "route_short_name", "transport_mode"])[
            ["shape_pt_lon", "shape_pt_lat"]
        ]
        .apply(lambda x: x.values.tolist())
        .reset_index(name="path")
    )

    paths["color"] = paths["transport_mode"].map(mode_colors)
    paths["color"] = paths["color"].apply(
        lambda x: x if isinstance(x, list) else [160, 160, 160]
    )

    return paths


def build_hsl_map_outputs(
    *,
    gtfs_dir: Path,
    mode: str = "all",
    route: str = "all",
    lookback_minutes: int | None = None,
) -> dict[str, pd.DataFrame]:
    """
    End-to-end helper:
    fetch HSL realtime, load GTFS static, build all map outputs.

    Returns:
        {
            "df_map": ...,
            "route_options": ...,
            "map_points": ...,
            "paths": ...
        }
    """
    _, raw_records = fetch_hsl_vehicle_positions()
    if not raw_records:
        return {
            "df_map": pd.DataFrame(),
            "route_options": pd.DataFrame(),
            "map_points": pd.DataFrame(),
            "paths": pd.DataFrame(),
        }

    gtfs_dir = download_gtfs_if_needed(gtfs_dir)
    routes, trips, shapes = load_gtfs_tables(gtfs_dir)

    df_map, df_route_lookup = build_hsl_map_dataframe(
        raw_records,
        routes,
        lookback_minutes=lookback_minutes,
    )

    route_options = build_route_options(df_map)
    filtered_df = filter_data(df_map, mode=mode, route=route)
    map_points = build_map_points(filtered_df)
    paths = build_hsl_paths(
        df_route_lookup=df_route_lookup,
        trips=trips,
        shapes=shapes,
        filtered_df=filtered_df,
    )

    return {
        "df_map": df_map,
        "route_options": route_options,
        "map_points": map_points,
        "paths": paths,
    }
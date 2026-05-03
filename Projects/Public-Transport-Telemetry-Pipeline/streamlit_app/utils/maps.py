"""
Map utilities for the Streamlit visualization layer.

Responsibilities:
- Load exported HSL map datasets through data_access.py
- Normalize column names across heterogeneous map sources
- Prepare lightweight point/path data for pydeck layers
- Provide route filtering and map center calculation

This module is intentionally storage-agnostic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from utils.data_access import (
    load_hsl_map_points,
    load_hsl_route_paths,
    load_hsl_route_options,
)

COL_ALIASES = {
    "route_id": ["route_id", "route", "line_id", "line", "short_name", "trip_route_id"],
    "route_label": [
        "route_label",
        "route_name",
        "label",
        "display_name",
        "route_short_name",
    ],
    "lat": ["lat", "latitude", "stop_lat", "point_lat"],
    "lon": ["lon", "lng", "longitude", "stop_lon", "point_lon"],
    "seq": ["seq", "stop_sequence", "point_sequence", "shape_pt_sequence", "order"],
    "geometry": ["geometry", "path", "coordinates"],
}

MODE_LABELS = {
    "bus": "Bus",
    "tram": "Tram",
    "metro": "Metro",
    "rail": "Rail",
    "ferry": "Ferry",
    "other": "Other",
}

MODE_ORDER = {
    "Metro": 0,
    "Tram": 1,
    "Bus": 2,
    "Rail": 3,
    "Ferry": 4,
    "Other": 5,
}

MODE_COLORS = {
    "Bus": [0, 122, 201],
    "Tram": [0, 129, 81],
    "Metro": [202, 64, 0],
    "Rail": [140, 71, 153],
    "Ferry": [0, 150, 170],
    "Other": [130, 130, 130],
}


def _normalize_mode_label(value) -> str:
    if pd.isna(value):
        return "Other"

    text = str(value).strip().lower()
    return MODE_LABELS.get(text, "Other")


def _mode_color(model_label: str) -> List[int]:
    return MODE_COLORS.get(model_label, MODE_COLORS["Other"])


def _first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _rename_to_standard(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}

    for standard_col, candidates in COL_ALIASES.items():
        actual_col = _first_existing_col(df, candidates)
        if actual_col and actual_col != standard_col:
            rename_map[actual_col] = (
                actual_col if standard_col in df.columns else standard_col
            )

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _normalize_route_label(df: pd.DataFrame) -> pd.DataFrame:
    if "route_label" not in df.columns:
        if "route_id" in df.columns:
            df["route_label"] = df["route_id"].astype(str)
        else:
            df["route_label"] = "unknown"
    return df


@st.cache_data(show_spinner=False, ttl=900, max_entries=2)
def load_map_parquets() -> Dict[str, pd.DataFrame]:
    loaded: Dict[str, pd.DataFrame] = {
        "map_points": load_hsl_map_points(),
        "paths": load_hsl_route_paths(),
    }

    for key in ["map_points", "paths"]:
        if key in loaded and loaded[key] is not None and not loaded[key].empty:
            loaded[key] = _rename_to_standard(loaded[key])
            loaded[key] = _coerce_numeric(loaded[key], ["lat", "lon", "seq"])

    return loaded


@st.cache_data(show_spinner=False, ttl=900, max_entries=2)
def load_route_options_only() -> List[str]:
    """
    Load only route options for the sidebar selector.

    This avoids building the full map bundle just to populate the route dropdown.
    Falls back to map points if the dedicated route options file is missing or empty.
    """
    try:
        route_options_df = load_hsl_route_options()

        if route_options_df is not None and not route_options_df.empty:
            route_options_df = _rename_to_standard(route_options_df)
            values = get_route_options(route_options_df)

            if values:
                return values
    except Exception:
        pass

    # Fallback only when the dedicated route options file is unavailable.
    try:
        fallback_df = load_hsl_map_points()

        if fallback_df is not None and not fallback_df.empty:
            fallback_df = _rename_to_standard(fallback_df)
            fallback_df = _coerce_numeric(fallback_df, ["lat", "lon", "seq"])
            return get_route_options(pd.DataFrame(), fallback_df=fallback_df)
    except Exception:
        pass

    return []


@st.cache_data(show_spinner=False, ttl=900, max_entries=2)
def load_route_options_with_modes() -> pd.DataFrame:
    """
    Load route options with transport mode metadata for sidebar filtering.

    This uses the existing HSL exported route options and does not add stop-level
    or journey-planning scope.
    """
    try:
        route_options_df = load_hsl_route_options()

        if route_options_df is None or route_options_df.empty:
            return pd.DataFrame(columns=["route_label", "mode_label", "route_display"])

        route_options_df = _rename_to_standard(route_options_df)
        route_options_df = _normalize_route_label(route_options_df)

        if "transport_mode" in route_options_df.columns:
            route_options_df["mode_label"] = route_options_df["transport_mode"].apply(
                _normalize_mode_label
            )
        else:
            route_options_df["mode_label"] = "Other"

        route_options_df["route_label"] = route_options_df["route_label"].astype(str)
        route_options_df["route_display"] = (
            route_options_df["mode_label"] + " · " + route_options_df["route_label"]
        )
        route_options_df["mode_order"] = (
            route_options_df["mode_label"].map(MODE_ORDER).fillna(99)
        )

        return (
            route_options_df[
                ["route_label", "mode_label", "route_display", "mode_order"]
            ]
            .drop_duplicates()
            .sort_values(["mode_order", "route_label"])
            .reset_index(drop=True)
        )

    except Exception:
        return pd.DataFrame(columns=["route_label", "mode_label", "route_display"])


def get_route_options(
    route_options_df: pd.DataFrame,
    fallback_df: Optional[pd.DataFrame] = None,
) -> List[str]:
    if route_options_df is not None and not route_options_df.empty:
        df = _normalize_route_label(route_options_df.copy())

        values = df["route_label"].dropna().astype(str).sort_values().unique().tolist()
        if values:
            return values

    if fallback_df is not None and not fallback_df.empty:
        df = _normalize_route_label(fallback_df.copy())

        values = df["route_label"].dropna().astype(str).sort_values().unique().tolist()
        return values

    return []


def filter_by_route(df: pd.DataFrame, selected_route: Optional[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    if not selected_route or str(selected_route).lower() == "all":
        return df

    candidate_cols = [col for col in ["route_label", "route_id"] if col in df.columns]
    if not candidate_cols:
        return df

    mask = pd.Series(False, index=df.index)
    for col in candidate_cols:
        mask = mask | (df[col].astype(str) == str(selected_route))

    return df.loc[mask]


def filter_by_mode(df: pd.DataFrame, selected_mode: Optional[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    if not selected_mode or selected_mode == "All":
        return df

    if "transport_mode" not in df.columns:
        return df

    mode_label = df["transport_mode"].apply(_normalize_mode_label)

    return df.loc[mode_label == selected_mode].copy()


def get_map_center(*dfs: pd.DataFrame) -> Tuple[float, float]:
    lats: List[float] = []
    lons: List[float] = []

    for df in dfs:
        if df is None or df.empty:
            continue

        if "lat" in df.columns and "lon" in df.columns:
            tmp = df[["lat", "lon"]].dropna()
            if not tmp.empty:
                # Lightweight sampling to reduce memory usage
                if len(tmp) > 500:
                    tmp = tmp.sample(n=500, random_state=42)
                lats.extend(tmp["lat"].astype(float).tolist())
                lons.extend(tmp["lon"].astype(float).tolist())

    if lats and lons:
        return float(sum(lats) / len(lats)), float(sum(lons) / len(lons))

    return 60.1699, 24.9384


def _safe_color(value, fallback: str = "Other") -> List[int]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return _mode_color(fallback)


def prepare_paths_for_pydeck(
    paths_df: pd.DataFrame, max_paths: int = 2
) -> pd.DataFrame:
    if paths_df is None or paths_df.empty:
        return pd.DataFrame(columns=["path", "route_label"])

    df = _rename_to_standard(paths_df)
    df = _normalize_route_label(df)

    geometry_col = _first_existing_col(df, ["path", "geometry", "coordinates"])
    if geometry_col:
        keep_cols = [
            c
            for c in [
                geometry_col,
                "route_label",
                "route_id",
                "transport_mode",
                "color",
            ]
            if c in df.columns
        ]

        out = df[keep_cols].copy()

        if geometry_col != "path":
            out = out.rename(columns={geometry_col: "path"})

        if "path" in out.columns:

            def _safe_path(p):
                if not isinstance(p, (list, tuple, np.ndarray)):
                    return []
                cleaned = []
                for point in p:
                    if isinstance(point, (list, tuple, np.ndarray)) and len(point) >= 2:
                        try:
                            cleaned.append([float(point[0]), float(point[1])])
                        except Exception:
                            continue
                return cleaned

            out["path"] = out["path"].apply(_safe_path)
            out = out[out["path"].map(len) > 1]

        if "path" in out.columns:
            out["path_len"] = out["path"].map(len)
            out = out.sort_values("path_len", ascending=False)

        out = out.head(max_paths)

        if "transport_mode" in out.columns:
            out["mode_label"] = out["transport_mode"].apply(_normalize_mode_label)
        else:
            out["mode_label"] = "Other"

        if "color" not in out.columns:
            out["color"] = out["mode_label"].apply(_mode_color)
        else:
            out["color"] = out["color"].apply(lambda x: _safe_color(x))

        keep_cols = [
            col
            for col in [
                "path",
                "route_label",
                "route_id",
                "transport_mode",
                "mode_label",
                "color",
            ]
            if col in out.columns
        ]
        return out[keep_cols].copy()

    required_cols = {"lat", "lon"}
    if not required_cols.issubset(df.columns):
        return pd.DataFrame(columns=["path", "route_label"])

    df = _coerce_numeric(df, ["lat", "lon", "seq"])
    df = df.dropna(subset=["lat", "lon"])

    sort_cols = [col for col in ["route_label", "seq"] if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    group_col = "route_label" if "route_label" in df.columns else "route_id"

    grouped = (
        df.groupby(group_col, dropna=False, sort=False)
        .apply(
            lambda x: [[float(lon), float(lat)] for lon, lat in zip(x["lon"], x["lat"])]
        )
        .reset_index(name="path")
    )

    if group_col != "route_label":
        grouped["route_label"] = grouped[group_col].astype(str)

    grouped = grouped[grouped["path"].map(len) > 1].head(max_paths)

    return grouped[["path", "route_label"]].copy()


def _add_vehicle_observed_at(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for candidate in ["timestamp", "event_time", "observation_time", "event_time_raw"]:
        if candidate in out.columns:
            out["vehicle_observed_at"] = pd.to_datetime(
                out[candidate],
                errors="coerce",
                utc=True,
            )
            return out

    out["vehicle_observed_at"] = pd.NaT
    return out


def prepare_points_for_pydeck(
    points_df: pd.DataFrame,
    max_points: int = 400,
    lookback_minutes: Optional[int] = None,
) -> pd.DataFrame:
    if points_df is None or points_df.empty:
        return pd.DataFrame(columns=["lat", "lon", "route_label"])

    df = _rename_to_standard(points_df)
    df = _normalize_route_label(df)
    df = _coerce_numeric(df, ["lat", "lon", "seq"])
    df = df.dropna(subset=["lat", "lon"])

    keep_cols = [
        col
        for col in df.columns
        if col
        in [
            "lat",
            "lon",
            "route_label",
            "route_id",
            "route_short_name",
            "seq",
            "color",
            "transport_mode",
            "vehicle_id",
            "timestamp",
            "event_time",
            "event_time_raw",
            "observation_time",
        ]
    ]

    if not keep_cols:
        keep_cols = ["lat", "lon", "route_label"]

    out = df[keep_cols].copy()
    out = _add_vehicle_observed_at(out)

    if "transport_mode" in out.columns:
        out["mode_label"] = out["transport_mode"].apply(_normalize_mode_label)
    else:
        out["mode_label"] = "Other"

    if "color" not in out.columns:
        out["color"] = out["mode_label"].apply(_mode_color)
    else:
        out["color"] = out["color"].apply(lambda x: _safe_color(x))

    if lookback_minutes is not None and "vehicle_observed_at" in out.columns:
        valid_times = out["vehicle_observed_at"].dropna()

        if not valid_times.empty:
            latest_vehicle_time = valid_times.max()
            cutoff_time = latest_vehicle_time - pd.Timedelta(minutes=lookback_minutes)
            out = out[out["vehicle_observed_at"] >= cutoff_time].copy()

    # Limit number of points to avoid excessive memory usage in map rendering
    sort_col = None
    for candidate in [
        "vehicle_observed_at",
        "timestamp",
        "event_time",
        "observation_time",
        "event_time_raw",
        "seq",
    ]:
        if candidate in out.columns:
            sort_col = candidate
            break

    if sort_col is not None:
        try:
            out = out.sort_values(sort_col)
        except Exception:
            pass

    if len(out) > max_points:
        out = out.tail(max_points).copy()

    return out


def build_map_bundle(
    selected_route: Optional[str],
    selected_mode: Optional[str] = None,
    vehicle_lookback_minutes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build the map data required for pydeck rendering.

    Route options are loaded separately through load_route_options_only()
    so this function only loads data needed to render the selected map view.
    """
    data = load_map_parquets()

    map_points = data.get("map_points", pd.DataFrame())
    paths = data.get("paths", pd.DataFrame())

    filtered_points = filter_by_mode(map_points, selected_mode)
    filtered_paths = filter_by_mode(paths, selected_mode)

    filtered_points = filter_by_route(filtered_points, selected_route)
    filtered_paths = filter_by_route(filtered_paths, selected_route)

    points_layer_df = prepare_points_for_pydeck(
        filtered_points,
        max_points=400,
        lookback_minutes=vehicle_lookback_minutes,
    )
    path_layer_df = prepare_paths_for_pydeck(filtered_paths, max_paths=4)

    center_lat, center_lon = get_map_center(points_layer_df, path_layer_df)

    return {
        "routes": [],
        "points": points_layer_df,
        "paths": path_layer_df,
        "center": {
            "lat": center_lat,
            "lon": center_lon,
        },
    }

"""
Map utilities for the Streamlit visualization layer.

Responsibilities:
- Load exported HSL map datasets through data_access.py
- Normalize column names across heterogeneous map sources
- Prepare point/path data for pydeck layers
- Provide route filtering and map center calculation

This module is intentionally storage-agnostic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from utils.data_access import (
    load_hsl_df_map,
    load_hsl_map_points,
    load_hsl_route_paths,
    load_hsl_route_paths_overview,
    load_hsl_route_options,
)


COL_ALIASES = {
    "route_id": ["route_id", "route", "line_id", "line", "short_name", "trip_route_id"],
    "route_label": ["route_label", "route_name", "label", "display_name", "route_short_name"],
    "lat": ["lat", "latitude", "stop_lat", "point_lat"],
    "lon": ["lon", "lng", "longitude", "stop_lon", "point_lon"],
    "seq": ["seq", "stop_sequence", "point_sequence", "shape_pt_sequence", "order"],
    "geometry": ["geometry", "path", "coordinates"],
}


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
            rename_map[actual_col] = actual_col if standard_col in df.columns else standard_col

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


@st.cache_data(show_spinner=False, ttl=300, max_entries=4)
def load_map_parquets() -> Dict[str, pd.DataFrame]:
    loaded: Dict[str, pd.DataFrame] = {
        "df_map": load_hsl_df_map(),
        "map_points": load_hsl_map_points(),
        "paths": load_hsl_route_paths(),
        "route_options": load_hsl_route_options(),
    }

    for key in ["df_map", "map_points", "paths", "route_options"]:
        if key in loaded and loaded[key] is not None and not loaded[key].empty:
            loaded[key] = _rename_to_standard(loaded[key])
            loaded[key] = _coerce_numeric(loaded[key], ["lat", "lon", "seq"])

    return loaded


@st.cache_data(show_spinner=False, ttl=300, max_entries=4)
def load_map_overview_parquets() -> Dict[str, pd.DataFrame]:
    loaded: Dict[str, pd.DataFrame] = {
        "paths": load_hsl_route_paths_overview(),
        "route_options": load_hsl_route_options(),
    }

    for key in ["paths", "route_options"]:
        if key in loaded and loaded[key] is not None and not loaded[key].empty:
            loaded[key] = _rename_to_standard(loaded[key])
            loaded[key] = _coerce_numeric(loaded[key], ["lat", "lon", "seq"])

    return loaded


def get_route_options(
    route_options_df: pd.DataFrame,
    fallback_df: Optional[pd.DataFrame] = None,
) -> List[str]:
    if route_options_df is not None and not route_options_df.empty:
        df = _normalize_route_label(route_options_df.copy())

        values = (
            df["route_label"]
            .dropna()
            .astype(str)
            .sort_values()
            .unique()
            .tolist()
        )
        if values:
            return values

    if fallback_df is not None and not fallback_df.empty:
        df = _normalize_route_label(fallback_df.copy())

        values = (
            df["route_label"]
            .dropna()
            .astype(str)
            .sort_values()
            .unique()
            .tolist()
        )
        return values

    return []


def filter_by_route(df: pd.DataFrame, selected_route: Optional[str]) -> pd.DataFrame:
    if df is None or df.empty or not selected_route:
        return df

    candidate_cols = [col for col in ["route_label", "route_id"] if col in df.columns]
    if not candidate_cols:
        return df

    mask = pd.Series(False, index=df.index)
    for col in candidate_cols:
        mask = mask | (df[col].astype(str) == str(selected_route))

    return df.loc[mask].copy()


def get_map_center(*dfs: pd.DataFrame) -> Tuple[float, float]:
    lats: List[float] = []
    lons: List[float] = []

    for df in dfs:
        if df is None or df.empty:
            continue

        if "lat" in df.columns and "lon" in df.columns:
            tmp = df[["lat", "lon"]].dropna()
            if not tmp.empty:
                lats.extend(tmp["lat"].astype(float).tolist())
                lons.extend(tmp["lon"].astype(float).tolist())

    if lats and lons:
        return float(sum(lats) / len(lats)), float(sum(lons) / len(lons))

    return 60.1699, 24.9384


def prepare_paths_for_pydeck(paths_df: pd.DataFrame) -> pd.DataFrame:
    if paths_df is None or paths_df.empty:
        return pd.DataFrame(columns=["path", "route_label"])

    df = paths_df.copy()
    df = _rename_to_standard(df)
    df = _normalize_route_label(df)

    geometry_col = _first_existing_col(df, ["path", "geometry", "coordinates"])
    if geometry_col:
        out = df.copy()
        if geometry_col != "path":
            out = out.rename(columns={geometry_col: "path"})

        if "path" in out.columns:
            out["path"] = out["path"].apply(
                lambda p: [[float(coord) for coord in point] for point in p]
                if isinstance(p, (list, tuple, np.ndarray))
                else []
            )

        keep_cols = [col for col in ["path", "route_label", "route_id"] if col in out.columns]
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
        df.groupby(group_col, dropna=False)
        .apply(
            lambda x: [
                [float(lon), float(lat)]
                for lon, lat in zip(x["lon"], x["lat"])
            ]
        )
        .reset_index(name="path")
    )

    if group_col != "route_label":
        grouped["route_label"] = grouped[group_col].astype(str)

    return grouped[["path", "route_label"]].copy()


def prepare_points_for_pydeck(points_df: pd.DataFrame) -> pd.DataFrame:
    if points_df is None or points_df.empty:
        return pd.DataFrame(columns=["lat", "lon", "route_label"])

    df = points_df.copy()
    df = _rename_to_standard(df)
    df = _normalize_route_label(df)
    df = _coerce_numeric(df, ["lat", "lon", "seq"])
    df = df.dropna(subset=["lat", "lon"])

    keep_cols = [
        col
        for col in df.columns
        if col in [
            "lat",
            "lon",
            "route_label",
            "route_id",
            "stop_name",
            "seq",
            "color",
            "transport_mode",
            "vehicle_id",
            "event_time",
            "event_time_raw",
            "observation_time",
        ]
    ]

    if not keep_cols:
        keep_cols = ["lat", "lon", "route_label"]

    return df[keep_cols].copy()


@st.cache_data(show_spinner=False, ttl=300, max_entries=32)
def build_map_bundle(selected_route: Optional[str] = None) -> Dict[str, Any]:
    data = load_map_parquets()

    df_map = data.get("df_map", pd.DataFrame())
    map_points = data.get("map_points", pd.DataFrame())
    paths = data.get("paths", pd.DataFrame())
    route_options = data.get("route_options", pd.DataFrame())

    fallback_df = df_map if not df_map.empty else map_points
    routes = get_route_options(route_options, fallback_df=fallback_df)

    filtered_points = filter_by_route(map_points, selected_route)
    filtered_paths = filter_by_route(paths, selected_route)

    points_layer_df = prepare_points_for_pydeck(filtered_points)
    path_layer_df = prepare_paths_for_pydeck(filtered_paths)

    center_lat, center_lon = get_map_center(points_layer_df, filtered_points, df_map)

    return {
        "routes": routes,
        "points": points_layer_df,
        "paths": path_layer_df,
        "center": {
            "lat": center_lat,
            "lon": center_lon,
        },
        "raw": {
            "df_map": df_map,
            "map_points": map_points,
            "paths": paths,
            "route_options": route_options,
        },
    }
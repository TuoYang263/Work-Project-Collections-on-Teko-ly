from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pydeck as pdk
import streamlit as st

from utils.data_access import load_weather_stations
from utils.maps import build_map_bundle


def compute_dynamic_view_state(points_df: pd.DataFrame, paths_df: pd.DataFrame):
    """
    Lightweight dynamic view state.
    Prefer points first. Only fall back to a small number of path coordinates.
    """
    if points_df is not None and not points_df.empty and {"lat", "lon"}.issubset(points_df.columns):
        tmp = points_df[["lat", "lon"]].dropna()
        if not tmp.empty:
            lat_min = float(tmp["lat"].min())
            lat_max = float(tmp["lat"].max())
            lon_min = float(tmp["lon"].min())
            lon_max = float(tmp["lon"].max())

            lat_center = (lat_min + lat_max) / 2
            lon_center = (lon_min + lon_max) / 2

            lat_span = lat_max - lat_min
            lon_span = lon_max - lon_min
            max_span = max(lat_span, lon_span)

            if max_span > 0.35:
                zoom = 10.8
            elif max_span > 0.18:
                zoom = 11.4
            elif max_span > 0.08:
                zoom = 12.0
            elif max_span > 0.03:
                zoom = 12.8
            else:
                zoom = 13.5

            return {"lat": lat_center, "lon": lon_center, "zoom": zoom}

    if paths_df is not None and not paths_df.empty and "path" in paths_df.columns:
        lats = []
        lons = []

        for path in paths_df["path"].head(2):
            if isinstance(path, list):
                for coord in path[:300]:
                    if isinstance(coord, list) and len(coord) >= 2:
                        try:
                            lons.append(float(coord[0]))
                            lats.append(float(coord[1]))
                        except Exception:
                            continue

        if lats and lons:
            lat_min, lat_max = min(lats), max(lats)
            lon_min, lon_max = min(lons), max(lons)

            lat_center = (lat_min + lat_max) / 2
            lon_center = (lon_min + lon_max) / 2

            lat_span = lat_max - lat_min
            lon_span = lon_max - lon_min
            max_span = max(lat_span, lon_span)

            if max_span > 0.35:
                zoom = 10.8
            elif max_span > 0.18:
                zoom = 11.4
            elif max_span > 0.08:
                zoom = 12.0
            elif max_span > 0.03:
                zoom = 12.8
            else:
                zoom = 13.5

            return {"lat": lat_center, "lon": lon_center, "zoom": zoom}

    return {"lat": 60.1699, "lon": 24.9384, "zoom": 11.5}


st.markdown(
    """
    <style>
    .block-container {
        max-width: 110rem;
        padding-top: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Map View")
st.caption("Route-level map view with sampled vehicle points, route shape, and optional weather context.")

latest_time_text = "Latest map context time: N/A"
latest_time_placeholder = st.empty()

@st.cache_data(show_spinner="Loading FMI weather data...", ttl=300, max_entries=2)
def load_weather():
    return load_weather_stations()

# -----------------------------
# Route selection
# -----------------------------
with st.spinner("Loading route options..."):
    initial_bundle = build_map_bundle(None)
    route_options = initial_bundle.get("routes", [])
    route_options = sorted({
        str(r).strip()
        for r in route_options
        if pd.notna(r)
        and str(r).strip() != ""
        and str(r).strip().lower() != "nan"
        and str(r).strip().lower() != "none"
    })

if not route_options:
    st.warning("No route options available.")
    st.stop()

selected_route = st.sidebar.selectbox(
    "Select route",
    options=route_options,
    index=0,
)

show_weather = st.sidebar.checkbox("Show weather context", value=False)

# -----------------------------
# Main bundle
# -----------------------------
with st.spinner("Loading route map data..."):
    bundle = build_map_bundle(selected_route)

points_df = bundle["points"]
paths_df = bundle["paths"]
weather_df = load_weather()

# -----------------------------
# Final pydeck-safe dataframes
# -----------------------------
if points_df is not None and not points_df.empty:
    keep_cols = [c for c in ["lon", "lat", "route_label"] if c in points_df.columns]
    safe_points_df = points_df[keep_cols].copy()
    safe_points_df["lon"] = pd.to_numeric(safe_points_df["lon"], errors="coerce")
    safe_points_df["lat"] = pd.to_numeric(safe_points_df["lat"], errors="coerce")
    safe_points_df = safe_points_df.dropna(subset=["lon", "lat"])

    if "route_label" not in safe_points_df.columns:
        safe_points_df["route_label"] = str(selected_route)
    else:
        safe_points_df["route_label"] = safe_points_df["route_label"].astype(str)

    safe_points_df["lat_display"] = safe_points_df["lat"].round(4)
    safe_points_df["lon_display"] = safe_points_df["lon"].round(4)
    safe_points_df["tooltip_title"] = "Route: " + safe_points_df["route_label"]
    safe_points_df["tooltip_line_1"] = "Lat: " + safe_points_df["lat_display"].astype(str)
    safe_points_df["tooltip_line_2"] = "Lon: " + safe_points_df["lon_display"].astype(str)
else:
    safe_points_df = pd.DataFrame(columns=["lon", "lat", "route_label"])


if paths_df is not None and not paths_df.empty:
    keep_cols = [c for c in ["path", "route_label"] if c in paths_df.columns]
    safe_paths_df = paths_df[keep_cols].copy()

    if "route_label" in safe_paths_df.columns:
        safe_paths_df["route_label"] = safe_paths_df["route_label"].astype(str)
else:
    safe_paths_df = pd.DataFrame(columns=["path", "route_label"])


if weather_df is not None and not weather_df.empty:
    keep_cols = [
        "station_id",
        "station_name",
        "lat",
        "lon",
        "observation_time",
        "temperature",
        "precipitation",
    ]
    existing_cols = [c for c in keep_cols if c in weather_df.columns]
    safe_weather_df = weather_df[existing_cols].copy()

    safe_weather_df["lat"] = pd.to_numeric(safe_weather_df["lat"], errors="coerce")
    safe_weather_df["lon"] = pd.to_numeric(safe_weather_df["lon"], errors="coerce")
    safe_weather_df = safe_weather_df.dropna(subset=["lat", "lon"])

    if "station_name" in safe_weather_df.columns:
        safe_weather_df["station_name"] = safe_weather_df["station_name"].astype(str)
    else:
        safe_weather_df["station_name"] = "Helsinki weather"

    if "temperature" in safe_weather_df.columns:
        safe_weather_df["temperature"] = pd.to_numeric(
            safe_weather_df["temperature"], errors="coerce"
        )
        safe_weather_df["temp_display"] = safe_weather_df["temperature"].round(1)
    else:
        safe_weather_df["temp_display"] = None

    if "precipitation" in safe_weather_df.columns:
        safe_weather_df["precipitation"] = pd.to_numeric(
            safe_weather_df["precipitation"], errors="coerce"
        )
        safe_weather_df["precip_display"] = safe_weather_df["precipitation"].round(2)
        safe_weather_df["precip_display"] = safe_weather_df["precip_display"].fillna(0.0)
    else:
        safe_weather_df["precip_display"] = 0.0

    if "observation_time" in safe_weather_df.columns:
        safe_weather_df["observation_time"] = pd.to_datetime(
            safe_weather_df["observation_time"], errors="coerce"
        )
        safe_weather_df["observation_display"] = safe_weather_df["observation_time"].dt.strftime(
            "%Y-%m-%d %H:%M"
        )
    else:
        safe_weather_df["observation_display"] = ""

    safe_weather_df["tooltip_title"] = safe_weather_df["station_name"]
    safe_weather_df["tooltip_line_1"] = (
        "Temp: " + safe_weather_df["temp_display"].astype(str) + " °C"
    )
    safe_weather_df["tooltip_line_2"] = (
        "Rain: " + safe_weather_df["precip_display"].astype(str) + " mm"
    )
    safe_weather_df["tooltip_line_3"] = (
        "Observed: " + safe_weather_df["observation_display"].astype(str)
    )
    safe_weather_df["tooltip_line_4"] = ""

    # -----------------------------
    # Latest data time (weather anchor)
    # -----------------------------
    now_local = datetime.now(ZoneInfo("Europe/Helsinki"))

    latest_obs_time = None

    if not safe_weather_df.empty and "observation_time" in safe_weather_df.columns:
        latest_obs = safe_weather_df["observation_time"].dropna()
        if not latest_obs.empty:
            latest_obs_time = latest_obs.max()

    if latest_obs_time is not None and pd.notna(latest_obs_time):
        if latest_obs_time.tzinfo is None:
            latest_obs_time = latest_obs_time.tz_localize("UTC").tz_convert("Europe/Helsinki")
        else:
            latest_obs_time = latest_obs_time.tz_convert("Europe/Helsinki")

        freshness_min = max(
            0,
            int((now_local - latest_obs_time.to_pydatetime()).total_seconds() / 60)
        )

        latest_time_text = (
            f"Latest map context time: {latest_obs_time.strftime('%Y-%m-%d %H:%M')} "
            f"(Helsinki time) · freshness lag: ~{freshness_min} min"
        )
else:
    safe_weather_df = pd.DataFrame()
    latest_time_text = "Latest map context time: N/A"

latest_time_placeholder.caption(latest_time_text)

# -----------------------------
# Enrich vehicle tooltip with city weather summary
# -----------------------------
if not safe_points_df.empty:
    city_temp_text = "City temp: N/A"
    city_rain_text = "City rain: N/A"

    if not safe_weather_df.empty:
        if "temp_display" in safe_weather_df.columns and safe_weather_df["temp_display"].notna().any():
            city_temp = safe_weather_df["temp_display"].dropna().iloc[0]
            city_temp_text = f"City temp: {city_temp} °C"

        if "precip_display" in safe_weather_df.columns and safe_weather_df["precip_display"].notna().any():
            city_rain = safe_weather_df["precip_display"].dropna().iloc[0]
            city_rain_text = f"City rain: {city_rain} mm"

    safe_points_df["tooltip_line_3"] = city_temp_text
    safe_points_df["tooltip_line_4"] = city_rain_text


# -----------------------------
# Metrics
# -----------------------------
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Selected route", selected_route)
with m2:
    st.metric("Vehicles / points", len(safe_points_df))
with m3:
    st.metric("Route paths", len(safe_paths_df))
with m4:
    st.metric("Weather stations", len(safe_weather_df) if show_weather else 0)

# -----------------------------
# Legend
# -----------------------------
st.markdown(
    """
    <div style="display:flex; gap:24px; align-items:center; font-size:14px; margin-bottom:8px;">
        <div style="display:flex; align-items:center; gap:8px;">
            <span style="width:12px; height:12px; border-radius:50%; background:#1e78ff; display:inline-block;"></span>
            <span>HSL vehicle points</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
            <span style="width:12px; height:12px; border-radius:50%; background:#ffa500; border:2px solid #b46e00; display:inline-block;"></span>
            <span>FMI weather station</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
            <span style="width:22px; height:2px; background:#7a7a7a; display:inline-block;"></span>
            <span>Route path</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Layers
# -----------------------------
st.caption(
    "Vehicle points represent sampled recent positions from telemetry batches, shown as a near-real-time operational snapshot."
)

layers = []

if not safe_paths_df.empty and "path" in safe_paths_df.columns:
    layers.append(
        pdk.Layer(
            "PathLayer",
            data=safe_paths_df,
            get_path="path",
            get_width=12,
            width_min_pixels=2,
            get_color=[90, 170, 255],
            pickable=False,
            opacity=0.95,
        )
    )

if not safe_points_df.empty:
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=safe_points_df,
            get_position="[lon, lat]",
            get_radius=60,
            get_fill_color=[30, 120, 255],
            get_line_color=[255, 255, 255],
            line_width_min_pixels=1,
            stroked=True,
            filled=True,
            pickable=True,
            opacity=0.82,
        )
    )

if show_weather and not safe_weather_df.empty:
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=safe_weather_df,
            get_position="[lon, lat]",
            get_radius=70,
            get_fill_color=[255, 165, 0],
            get_line_color=[180, 110, 0],
            line_width_min_pixels=2,
            stroked=True,
            filled=True,
            pickable=True,
            opacity=0.78,
        )
    )

# -----------------------------
# View state
# -----------------------------
view_cfg = compute_dynamic_view_state(safe_points_df, safe_paths_df)

tooltip = {
    "html": """
    <div style="font-size: 13px;">
        <div><b>{tooltip_title}</b></div>
        <div>{tooltip_line_1}</div>
        <div>{tooltip_line_2}</div>
        <div>{tooltip_line_3}</div>
        <div>{tooltip_line_4}</div>
    </div>
    """,
    "style": {"backgroundColor": "white", "color": "black"},
}

deck = pdk.Deck(
    layers=layers,
    initial_view_state=pdk.ViewState(
        latitude=view_cfg["lat"],
        longitude=view_cfg["lon"],
        zoom=view_cfg["zoom"],
        pitch=0,
    ),
    tooltip=tooltip,
    map_style=None,
)

st.pydeck_chart(deck, use_container_width=True)
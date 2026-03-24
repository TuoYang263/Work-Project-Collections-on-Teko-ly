from __future__ import annotations

import pandas as pd
import pydeck as pdk
import streamlit as st

from datetime import datetime
from utils.maps import build_map_bundle
from utils.data_access import load_weather_stations

def compute_dynamic_view_state(points_df, paths_df, is_all_routes: bool):
    """
    Compute a lightweight dynamic map center and zoom.

    - All routes: use a fixed Helsinki-centered overview
    - Selected route: prefer route path geometry for centering
    """
    if is_all_routes:
        return {
            "lat": 60.1699,
            "lon": 24.9384,
            "zoom": 11.5,
        }

    lats = []
    lons = []

    if not paths_df.empty and "path" in paths_df.columns:
        for path in paths_df["path"]:
            if isinstance(path, list):
                for coord in path:
                    if isinstance(coord, list) and len(coord) == 2:
                        lons.append(float(coord[0]))
                        lats.append(float(coord[1]))

    if not lats or not lons:
        if not points_df.empty:
            lats.extend(points_df["lat"].astype(float).tolist())
            lons.extend(points_df["lon"].astype(float).tolist())

    if not lats or not lons:
        return {"lat": 60.1699, "lon": 24.9384, "zoom": 11.5}

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
st.caption("HSL realtime vehicles, GTFS route shapes, and FMI weather context")

now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
st.caption(f"Last updated: {now_str}")


@st.cache_data(show_spinner="Loading HSL map data...")
def load_bundle(selected_route: str | None):
    return build_map_bundle(selected_route=selected_route)


@st.cache_data(show_spinner="Loading FMI weather data...")
def load_weather():
    return load_weather_stations()


initial_bundle = build_map_bundle(selected_route=None)
route_options = initial_bundle["routes"]

selected_route = st.sidebar.selectbox(
    "Select route",
    options=["All"] + route_options if route_options else ["All"],
    index=0,
)

show_weather = st.sidebar.checkbox("Show weather context", value=True)

max_points = st.sidebar.slider(
    "Max points in overview",
    min_value=100,
    max_value=1000,
    value=400,
    step=100,
)

selected_route_value = None if selected_route == "All" else selected_route
is_all_routes = selected_route_value is None

bundle = load_bundle(selected_route_value)

points_df = bundle["points"]
paths_df = bundle["paths"]
weather_df = load_weather() if show_weather else pd.DataFrame()

# -----------------------------
# Final pydeck-safe dataframes
# -----------------------------
safe_points_df = points_df.copy()
safe_paths_df = paths_df.copy()
safe_weather_df = weather_df.copy()

if not safe_points_df.empty:
    keep_cols = ["lon", "lat", "route_label"]
    existing_cols = [c for c in keep_cols if c in safe_points_df.columns]
    safe_points_df = safe_points_df[existing_cols].copy()

    safe_points_df["lon"] = safe_points_df["lon"].astype(float)
    safe_points_df["lat"] = safe_points_df["lat"].astype(float)
    safe_points_df["route_label"] = safe_points_df["route_label"].astype(str)

    safe_points_df["lat_display"] = safe_points_df["lat"].round(4)
    safe_points_df["lon_display"] = safe_points_df["lon"].round(4)

    # Unified tooltip fields for HSL points
    safe_points_df["tooltip_title"] = "Route: " + safe_points_df["route_label"]
    safe_points_df["tooltip_line_1"] = "Lat: " + safe_points_df["lat_display"].astype(str)
    safe_points_df["tooltip_line_2"] = "Lon: " + safe_points_df["lon_display"].astype(str)
    safe_points_df["tooltip_line_3"] = "HSL vehicle point"

if not safe_paths_df.empty:
    keep_cols = ["path", "route_label"]
    existing_cols = [c for c in keep_cols if c in safe_paths_df.columns]
    safe_paths_df = safe_paths_df[existing_cols].copy()

    safe_paths_df["route_label"] = safe_paths_df["route_label"].astype(str)
    safe_paths_df["path"] = safe_paths_df["path"].apply(
        lambda p: [[float(x), float(y)] for x, y in p]
        if isinstance(p, list)
        else []
    )

if not safe_weather_df.empty:
    keep_cols = [
        "station_id",
        "station_name",
        "lat",
        "lon",
        "observation_time",
        "temperature",
        "precipitation",
    ]
    existing_cols = [c for c in keep_cols if c in safe_weather_df.columns]
    safe_weather_df = safe_weather_df[existing_cols].copy()

    safe_weather_df["lat"] = safe_weather_df["lat"].astype(float)
    safe_weather_df["lon"] = safe_weather_df["lon"].astype(float)

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

    # Unified tooltip fields for weather points
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

# -----------------------------
# Metrics
# -----------------------------
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Selected route", selected_route if selected_route_value else "All")
with m2:
    st.metric("Vehicles / points", len(safe_points_df))
with m3:
    if is_all_routes:
        skeleton_count = (
            safe_paths_df["route_label"].nunique()
            if not safe_paths_df.empty
            else 0
        )
        st.metric("Overview routes", skeleton_count)
    else:
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
layers = []

render_points_df = safe_points_df.copy()
if is_all_routes and len(render_points_df) > max_points:
    render_points_df = render_points_df.sample(n=max_points, random_state=42).copy()

render_weather_df = safe_weather_df.copy()

if is_all_routes:
    skeleton_paths_df = (
        safe_paths_df.groupby("route_label", as_index=False).head(2).copy()
        if not safe_paths_df.empty
        else safe_paths_df
    )

    if not skeleton_paths_df.empty:
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=skeleton_paths_df,
                get_path="path",
                get_width=3,
                get_color=[120, 120, 120],
                opacity=0.35,
                pickable=False,
            )
        )

    if not render_points_df.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=render_points_df,
                get_position="[lon, lat]",
                get_radius=20,
                get_fill_color=[40, 120, 255],
                opacity=0.85,
                pickable=True,
            )
        )
else:
    if not safe_paths_df.empty and "path" in safe_paths_df.columns:
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=safe_paths_df,
                get_path="path",
                get_width=5,
                get_color=[60, 150, 255],
                pickable=False,
                opacity=0.9,
            )
        )

    if not render_points_df.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=render_points_df,
                get_position="[lon, lat]",
                get_radius=45,
                get_fill_color=[30, 120, 255],
                pickable=True,
                opacity=0.95,
            )
        )

if show_weather and not render_weather_df.empty:
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=render_weather_df,
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
view_cfg = compute_dynamic_view_state(render_points_df, safe_paths_df, is_all_routes)

view_state = pdk.ViewState(
    latitude=view_cfg["lat"],
    longitude=view_cfg["lon"],
    zoom=view_cfg["zoom"],
    pitch=0,
)

tooltip = {
    "html": """
    <b>{tooltip_title}</b><br/>
    {tooltip_line_1}<br/>
    {tooltip_line_2}<br/>
    {tooltip_line_3}
    """,
    "style": {
        "backgroundColor": "#1f2c3a",
        "color": "white",
        "fontSize": "12px",
    },
}

st.pydeck_chart(
    pdk.Deck(
        map_style="light",
        initial_view_state=view_state,
        layers=layers,
        tooltip=tooltip,
    ),
    use_container_width=True,
    height=650,
)

with st.expander("Debug preview"):
    st.write("Selected route:", selected_route_value)
    st.write("Original points shape:", safe_points_df.shape)
    st.write("Rendered points shape:", render_points_df.shape)
    st.write("Paths shape:", safe_paths_df.shape)
    st.write("Weather shape:", safe_weather_df.shape)
    st.dataframe(render_points_df.head(20), use_container_width=True)
    st.dataframe(safe_paths_df.head(10), use_container_width=True)
    st.dataframe(safe_weather_df.head(10), use_container_width=True)
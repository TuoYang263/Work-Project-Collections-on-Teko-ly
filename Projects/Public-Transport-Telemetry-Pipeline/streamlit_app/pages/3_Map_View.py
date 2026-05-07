from __future__ import annotations

import time
from datetime import datetime
from zoneinfo import ZoneInfo

import math
import pandas as pd
import pydeck as pdk
import streamlit as st

from utils.data_access import load_weather_stations
from utils.maps import build_map_bundle, load_route_options_with_modes


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """
    Calculate approximate distance between two lat/lon points in kilometers.
    """
    radius_km = 6371.0

    lat1_rad = math.radians(float(lat1))
    lon1_rad = math.radians(float(lon1))
    lat2_rad = math.radians(float(lat2))
    lon2_rad = math.radians(float(lon2))

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return radius_km * c


def attach_nearest_weather_station_context(
    points_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add nearest FMI weather station context to vehicle points.

    This is distance-based context only. It does not assign weather impact
    or causal meaning to vehicle observations.
    """
    if points_df.empty or weather_df.empty:
        return points_df

    required_point_cols = {"lat", "lon"}
    required_weather_cols = {"lat", "lon", "station_name"}

    if not required_point_cols.issubset(points_df.columns):
        return points_df

    if not required_weather_cols.issubset(weather_df.columns):
        return points_df

    out = points_df.copy()

    nearest_names = []
    nearest_distances = []
    nearest_temps = []
    nearest_rains = []
    nearest_observed = []

    station_rows = weather_df.dropna(subset=["lat", "lon"]).to_dict("records")

    if not station_rows:
        return out

    for _, point in out.iterrows():
        best_station = None
        best_distance = None

        for station in station_rows:
            distance_km = haversine_km(
                point["lat"],
                point["lon"],
                station["lat"],
                station["lon"],
            )

            if best_distance is None or distance_km < best_distance:
                best_distance = distance_km
                best_station = station

        if best_station is None:
            nearest_names.append("N/A")
            nearest_distances.append(None)
            nearest_temps.append("N/A")
            nearest_rains.append("N/A")
            nearest_observed.append("N/A")
            continue

        nearest_names.append(str(best_station.get("station_name", "N/A")))
        nearest_distances.append(
            round(best_distance, 1) if best_distance is not None else None
        )
        nearest_temps.append(best_station.get("temp_display", "N/A"))
        nearest_rains.append(best_station.get("precip_display", "N/A"))
        nearest_observed.append(best_station.get("observation_display", "N/A"))

    out["nearest_weather_station"] = nearest_names
    out["nearest_weather_distance_km"] = nearest_distances
    out["nearest_weather_distance_display"] = (
        pd.Series(nearest_distances)
        .apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
        .values
    )
    out["nearest_weather_temp"] = nearest_temps
    out["nearest_weather_rain"] = nearest_rains
    out["nearest_weather_observed"] = nearest_observed

    return out


def compute_dynamic_view_state(points_df: pd.DataFrame, paths_df: pd.DataFrame):
    """
    Lightweight dynamic view state.
    Prefer points first. Only fall back to a small number of path coordinates.
    """
    if (
        points_df is not None
        and not points_df.empty
        and {"lat", "lon"}.issubset(points_df.columns)
    ):
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

start_time = time.time()

st.title("Map View")
st.caption(
    "Scheduled snapshot map view with sampled HSL vehicle points, route shapes, "
    "and optional FMI weather station context from the latest pipeline refresh."
)

latest_time_text = "Latest map context time: N/A"


@st.cache_data(show_spinner="Loading FMI weather data...", ttl=900, max_entries=2)
def load_weather():
    return load_weather_stations()


# -----------------------------
# Route and context controls
# -----------------------------
with st.spinner("Loading route options..."):
    route_meta_df = load_route_options_with_modes()

if route_meta_df is None or route_meta_df.empty:
    st.warning("No route options available.")
    st.stop()

available_modes = ["All"] + [
    mode
    for mode in ["Metro", "Tram", "Bus", "Rail", "Ferry", "Other"]
    if mode in set(route_meta_df["mode_label"].dropna())
]

selected_mode = st.sidebar.selectbox(
    "Transport mode",
    options=available_modes,
    index=0,
)

filtered_route_meta = route_meta_df.copy()

if selected_mode != "All":
    filtered_route_meta = filtered_route_meta[
        filtered_route_meta["mode_label"] == selected_mode
    ].copy()

route_display_to_label = dict(
    zip(filtered_route_meta["route_display"], filtered_route_meta["route_label"])
)

route_display_options = list(route_display_to_label.keys())

if not route_display_options:
    st.warning("No routes available for the selected transport mode.")
    st.stop()

selected_route_display = st.sidebar.selectbox(
    "Select route",
    options=route_display_options,
    index=0,
)

route_display_to_mode = dict(
    zip(filtered_route_meta["route_display"], filtered_route_meta["mode_label"])
)

selected_route = route_display_to_label[selected_route_display]

effective_selected_mode = (
    selected_mode
    if selected_mode != "All"
    else route_display_to_mode.get(selected_route_display, "All")
)

show_weather = st.sidebar.checkbox("Show weather context", value=False)

# -----------------------------
# Main bundle
# -----------------------------
with st.spinner("Loading route map data..."):
    bundle = build_map_bundle(
        selected_route=selected_route,
        selected_mode=effective_selected_mode,
    )

points_df = bundle["points"]
paths_df = bundle["paths"]
weather_df = load_weather() if show_weather else pd.DataFrame()

# -----------------------------
# Final pydeck-safe dataframes
# -----------------------------
if points_df is not None and not points_df.empty:
    keep_cols = [
        c
        for c in [
            "lon",
            "lat",
            "route_label",
            "vehicle_id",
            "vehicle_observed_at",
            "transport_mode",
            "mode_label",
            "color",
        ]
        if c in points_df.columns
    ]

    safe_points_df = points_df[keep_cols].copy()
    safe_points_df["lon"] = pd.to_numeric(safe_points_df["lon"], errors="coerce")
    safe_points_df["lat"] = pd.to_numeric(safe_points_df["lat"], errors="coerce")
    safe_points_df = safe_points_df.dropna(subset=["lon", "lat"])

    safe_points_df["lat_display"] = safe_points_df["lat"].round(5).astype(str)
    safe_points_df["lon_display"] = safe_points_df["lon"].round(5).astype(str)

    if "route_label" not in safe_points_df.columns:
        safe_points_df["route_label"] = str(selected_route)
    else:
        safe_points_df["route_label"] = safe_points_df["route_label"].astype(str)

    if "vehicle_id" in safe_points_df.columns:
        safe_points_df["vehicle_id"] = (
            safe_points_df["vehicle_id"].fillna("N/A").astype(str)
        )
    else:
        safe_points_df["vehicle_id"] = "N/A"

    if "mode_label" in safe_points_df.columns:
        safe_points_df["mode_label"] = (
            safe_points_df["mode_label"].fillna("Other").astype(str)
        )
    else:
        safe_points_df["mode_label"] = "Other"

    if "color" not in safe_points_df.columns:
        safe_points_df["color"] = [[130, 130, 130]] * len(safe_points_df)

    if "vehicle_observed_at" in safe_points_df.columns:
        safe_points_df["vehicle_observed_at"] = pd.to_datetime(
            safe_points_df["vehicle_observed_at"],
            errors="coerce",
            utc=True,
        )
        safe_points_df["observed_display"] = (
            safe_points_df["vehicle_observed_at"]
            .dt.tz_convert("Europe/Helsinki")
            .dt.strftime("%Y-%m-%d %H:%M")
        )
    else:
        safe_points_df["observed_display"] = "N/A"

    safe_points_df["observed_display"] = safe_points_df["observed_display"].fillna(
        "N/A"
    )

    safe_points_df["tooltip_title"] = (
        safe_points_df["mode_label"] + " · " + safe_points_df["route_label"]
    )
    safe_points_df["tooltip_line_1"] = "Vehicle: " + safe_points_df["vehicle_id"]
    safe_points_df["tooltip_line_2"] = "Observed: " + safe_points_df["observed_display"]
    safe_points_df["tooltip_line_location"] = (
        "Location: "
        + safe_points_df["lat_display"]
        + ", "
        + safe_points_df["lon_display"]
    )
else:
    safe_points_df = pd.DataFrame(columns=["lon", "lat", "route_label"])


if paths_df is not None and not paths_df.empty:
    keep_cols = [
        c
        for c in ["path", "route_label", "transport_mode", "mode_label", "color"]
        if c in paths_df.columns
    ]
    safe_paths_df = paths_df[keep_cols].copy()

    if "route_label" in safe_paths_df.columns:
        safe_paths_df["route_label"] = safe_paths_df["route_label"].astype(str)

    if "mode_label" in safe_paths_df.columns:
        safe_paths_df["mode_label"] = (
            safe_paths_df["mode_label"].fillna("Other").astype(str)
        )

    if "color" not in safe_paths_df.columns:
        safe_paths_df["color"] = [[130, 130, 130]] * len(safe_paths_df)
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

    safe_weather_df["lat_display"] = safe_weather_df["lat"].round(5).astype(str)
    safe_weather_df["lon_display"] = safe_weather_df["lon"].round(5).astype(str)

    if "observation_time" in safe_weather_df.columns:
        safe_weather_df["observation_time"] = pd.to_datetime(
            safe_weather_df["observation_time"],
            errors="coerce",
            utc=True,
        )

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
        safe_weather_df["precip_display"] = safe_weather_df["precip_display"].fillna(
            0.0
        )
    else:
        safe_weather_df["precip_display"] = 0.0

    if "observation_time" in safe_weather_df.columns:
        safe_weather_df["observation_display"] = (
            safe_weather_df["observation_time"]
            .dt.tz_convert("Europe/Helsinki")
            .dt.strftime("%Y-%m-%d %H:%M")
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
    safe_weather_df["tooltip_line_location"] = (
        "Location: "
        + safe_weather_df["lat_display"]
        + ", "
        + safe_weather_df["lon_display"]
    )
    safe_weather_df["tooltip_line_3"] = "Observed: " + safe_weather_df[
        "observation_display"
    ].astype(str)
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
            latest_obs_time = latest_obs_time.tz_localize("UTC").tz_convert(
                "Europe/Helsinki"
            )
        else:
            latest_obs_time = latest_obs_time.tz_convert("Europe/Helsinki")

        freshness_min = max(
            0, int((now_local - latest_obs_time.to_pydatetime()).total_seconds() / 60)
        )

        latest_time_text = (
            f"Latest map context time: {latest_obs_time.strftime('%Y-%m-%d %H:%M')} "
            f"Helsinki time · data age: ~{freshness_min} min"
        )
else:
    safe_weather_df = pd.DataFrame()
    if show_weather:
        latest_time_text = (
            "Latest map context time: weather context enabled, but no weather data is available. "
            "Map is based on the latest exported route snapshot."
        )
    else:
        latest_time_text = (
            "Latest map context time: weather context disabled. "
            "Map is based on the latest exported route snapshot."
        )

latest_vehicle_text = "Latest vehicle observation: N/A"

if not safe_points_df.empty and "vehicle_observed_at" in safe_points_df.columns:
    latest_vehicle_time = safe_points_df["vehicle_observed_at"].dropna().max()

    if pd.notna(latest_vehicle_time):
        latest_vehicle_time = latest_vehicle_time.tz_convert("Europe/Helsinki")
        latest_vehicle_text = (
            f"Latest vehicle observation: "
            f"{latest_vehicle_time.strftime('%Y-%m-%d %H:%M')} Helsinki time"
        )

st.caption(latest_vehicle_text)
st.caption(latest_time_text)

if show_weather and not safe_weather_df.empty:
    station_summary_cols = [
        c
        for c in [
            "station_name",
            "temp_display",
            "precip_display",
            "observation_display",
        ]
        if c in safe_weather_df.columns
    ]

    if station_summary_cols:
        weather_summary_df = safe_weather_df[station_summary_cols].copy()

        weather_summary_df = weather_summary_df.rename(
            columns={
                "station_name": "Station",
                "temp_display": "Temp °C",
                "precip_display": "Rain mm",
                "observation_display": "Observed",
            }
        )

        st.markdown("**Weather station context**")
        st.dataframe(
            weather_summary_df,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Weather stations are shown as contextual external signals. "
            "They are not assigned to individual vehicles or used for causal analysis."
        )

if show_weather and not safe_weather_df.empty and not safe_points_df.empty:
    safe_points_df = attach_nearest_weather_station_context(
        safe_points_df,
        safe_weather_df,
    )

# -----------------------------
# Enrich vehicle tooltip with weather context status
# -----------------------------
if not safe_points_df.empty:
    if show_weather and not safe_weather_df.empty:
        if "nearest_weather_station" in safe_points_df.columns:
            safe_points_df["tooltip_line_3"] = (
                "Nearest FMI station: "
                + safe_points_df["nearest_weather_station"].astype(str)
                + " ("
                + safe_points_df["nearest_weather_distance_display"].astype(str)
                + " km)"
            )
            safe_points_df["tooltip_line_4"] = (
                "Observed weather: "
                + safe_points_df["nearest_weather_temp"].astype(str)
                + " °C, rain "
                + safe_points_df["nearest_weather_rain"].astype(str)
                + " mm"
            )
        else:
            safe_points_df["tooltip_line_3"] = "Weather context: see station summary"
            safe_points_df["tooltip_line_4"] = ""
    elif show_weather:
        safe_points_df["tooltip_line_3"] = "Weather context: no station data"
        safe_points_df["tooltip_line_4"] = ""
    else:
        safe_points_df["tooltip_line_3"] = "Weather context: off"
        safe_points_df["tooltip_line_4"] = ""

# -----------------------------
# Metrics
# -----------------------------
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Selected route", selected_route_display)
with m2:
    st.metric("Vehicles / points", len(safe_points_df))
with m3:
    st.metric("Route paths", len(safe_paths_df))
with m4:
    st.metric("Weather stations", len(safe_weather_df) if show_weather else 0)

# -----------------------------
# Legend
# -----------------------------
legend_items = [
    '<div class="legend-item"><span class="legend-dot bus-dot"></span><span>Bus</span></div>',
    '<div class="legend-item"><span class="legend-dot tram-dot"></span><span>Tram</span></div>',
    '<div class="legend-item"><span class="legend-dot metro-dot"></span><span>Metro</span></div>',
    '<div class="legend-item"><span class="legend-dot rail-dot"></span><span>Rail</span></div>',
    '<div class="legend-item"><span class="legend-dot ferry-dot"></span><span>Ferry</span></div>',
]

if show_weather:
    legend_items.append(
        '<div class="legend-item">'
        '<span class="legend-dot weather-dot"></span>'
        "<span>FMI weather station</span>"
        "</div>"
    )

legend_html = "".join(legend_items)

st.markdown(
    f"""
    <style>
    .legend-container {{
        display: flex;
        gap: 24px;
        align-items: center;
        flex-wrap: wrap;
        font-size: 14px;
        margin-bottom: 8px;
    }}
    .legend-item {{
        display: flex;
        align-items: center;
        gap: 8px;
        white-space: nowrap;
    }}
    .legend-dot {{
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
    }}
    .bus-dot {{
        background: #007AC9;
    }}
    .tram-dot {{
        background: #008151;
    }}
    .metro-dot {{
        background: #CA4000;
    }}
    .rail-dot {{
        background: #8C4799;
    }}
    .ferry-dot {{
        background: #0096AA;
    }}
    .weather-dot {{
        background: #ffa500;
        border: 2px solid #b46e00;
    }}
    </style>

    <div class="legend-container">
        {legend_html}
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Layers
# -----------------------------
st.caption(
    "Vehicle points represent the latest exported HSL realtime snapshot. "
    "Nearest weather station context is selected by geographic distance and shown for context only. "
    "It is not used for causal weather impact analysis."
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
            get_color="color",
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
            get_fill_color="color",
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
        <div>{tooltip_line_location}</div>
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

st.caption(f"Page rendered in {time.time() - start_time:.2f}s")

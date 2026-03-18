import streamlit as st
import pandas as pd
from utils.load_data import load_route_window, load_route_daily

st.set_page_config(
    page_title="Public Transport Telemetry Pipeline",
    layout="wide"
)

st.title("Route Performance")

def safe_mean(series: pd.Series, decimals: int = 2):
    series = series.dropna()
    if series.empty:
        return "N/A"
    return round(series.mean(), decimals)

# ===== Load data =====
df_window = load_route_window()
df_daily = load_route_daily()

# ===== Basic guard =====
if df_window is None or df_window.empty:
    st.warning("No route window data available.")
    st.stop()

if df_daily is None:
    df_daily = pd.DataFrame()

# ===== Filter =====
routes = ["All"] + sorted(df_window["route_id"].dropna().unique().tolist())
selected_route = st.selectbox("Select Route", routes)

if selected_route != "All":
    df_window_filtered = df_window[df_window["route_id"] == selected_route].copy()
    df_daily_filtered = df_daily[df_daily["route_id"] == selected_route].copy() if not df_daily.empty else pd.DataFrame()
else:
    df_window_filtered = df_window.copy()
    df_daily_filtered = df_daily.copy()

# ===== KPI =====
col1, col2, col3 = st.columns(3)

col1.metric("Avg Delay (s)", safe_mean(df_window_filtered["avg_delay_sec"]))
col2.metric("Avg Occupancy (%)", safe_mean(df_window_filtered["avg_occupancy_pct"]))
col3.metric("Late Rate", safe_mean(df_window_filtered["late_rate_delay"]))

# ===== Main chart =====
if df_window_filtered.empty:
    st.warning("No data available for the selected route.")
else:
    if selected_route == "All":
        st.subheader("Route Ranking by Average Delay")

        if df_daily_filtered.empty:
            st.info("No daily summary data available for route comparison.")
        else:
            route_rank = (
                df_daily_filtered.groupby("route_id", as_index=False)["avg_delay_sec"]
                .mean()
                .sort_values("avg_delay_sec", ascending=False)
            )

            if route_rank.empty:
                st.info("No route comparison data available.")
            else:
                st.bar_chart(route_rank.set_index("route_id")["avg_delay_sec"])
    else:
        st.subheader("Delay Trend")

        trend_df = (
            df_window_filtered[["window_start", "avg_delay_sec"]]
            .dropna()
            .copy()
        )

        if trend_df.empty:
            st.info("No delay trend data available for the selected route.")
        else:
            trend_df["window_start"] = pd.to_datetime(trend_df["window_start"], errors="coerce")
            trend_df = trend_df.dropna(subset=["window_start"]).sort_values("window_start")

            if trend_df.empty:
                st.info("No delay trend data available for the selected route.")
            else:
                st.line_chart(trend_df.set_index("window_start")["avg_delay_sec"])
                st.caption("Current trend is based on simulated telemetry windows, so coverage may be sparse or discontinuous.")

# ===== Daily summary =====
st.subheader("Daily Summary")

if df_daily_filtered.empty:
    st.info("No daily summary data available.")
else:
    display_cols = [
        "date",
        "route_id",
        "avg_delay_sec",
        "avg_occupancy_pct",
        "total_events_delay",
        "total_events_occupancy",
        "avg_late_rate_delay",
        "avg_ingest_delay_sec",
        "dq_flag",
    ]

    available_cols = [col for col in display_cols if col in df_daily_filtered.columns]
    st.caption("Daily aggregated metrics per route")
    st.dataframe(df_daily_filtered[available_cols],
                 use_container_width=True,
                 height=300
    )
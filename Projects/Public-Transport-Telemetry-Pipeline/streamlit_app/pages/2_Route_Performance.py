import time
import streamlit as st
import pandas as pd
from datetime import datetime
from utils.load_data import load_route_window, load_route_daily

def format_latest_timestamp(df: pd.DataFrame, column: str) -> str:
    if df is None or df.empty or column not in df.columns:
        return "N/A"

    ts = pd.to_datetime(df[column], errors="coerce").max()
    if pd.isna(ts):
        return "N/A"
    return ts.strftime("%Y-%m-%d %H:%M")


def safe_mean(series: pd.Series, decimals: int = 2):
    series = series.dropna()
    if series.empty:
        return "N/A"
    return round(series.mean(), decimals)


start_time = time.time()
st.title("Route Performance")
st.caption("Recent route-level KPIs derived from exported Gold-layer aggregates.")

# ===== Load data =====
with st.spinner("Loading route performance data..."):
    df_window = load_route_window()
    df_daily = load_route_daily()

# ===== Basic guard =====
if df_window is None or df_window.empty:
    st.warning("No route window data available.")
    st.stop()

df_window = df_window.copy()
for col in ["window_start", "window_end"]:
    if col in df_window.columns:
        df_window[col] = pd.to_datetime(df_window[col], errors="coerce")

if df_daily is None:
    df_daily = pd.DataFrame()
else:
    df_daily = df_daily.copy()
    if "date" in df_daily.columns:
        df_daily["date"] = pd.to_datetime(df_daily["date"], errors="coerce")

# ===== Sidebar filter =====
routes = ["All"] + sorted(df_window["route_id"].dropna().unique().tolist())
selected_route = st.sidebar.selectbox("Select Route", routes)

if selected_route != "All":
    df_window_filtered = df_window[df_window["route_id"] == selected_route].copy()
    df_daily_filtered = (
        df_daily[df_daily["route_id"] == selected_route].copy()
        if not df_daily.empty else pd.DataFrame()
    )
else:
    df_window_filtered = df_window.copy()
    df_daily_filtered = df_daily.copy()

now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
st.caption(f"Last updated: {now_str}")

if "window_end" in df_window_filtered.columns and df_window_filtered["window_end"].notna().any():
    latest_window_end = df_window_filtered["window_end"].max()
    latest_window_df = df_window_filtered[df_window_filtered["window_end"] == latest_window_end].copy()
else:
    latest_window_df = df_window_filtered.copy()

# ===== KPI =====
col1, col2, col3 = st.columns(3)
col1.metric("Avg Delay (s)", safe_mean(latest_window_df["avg_delay_sec"]))
col2.metric("Avg Occupancy (%)", safe_mean(latest_window_df["avg_occupancy_pct"]))
col3.metric("Late Rate", safe_mean(latest_window_df["late_rate_delay"]))

st.caption(
    "KPIs above reflect the latest available exported route window, while the charts below show recent historical context."
)

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
                st.caption(
                    "Route ranking is based on exported Gold-layer daily summaries and should be read as a recent analytical snapshot."
                )
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
            trend_df = trend_df.sort_values("window_start").tail(12)

            if trend_df.empty:
                st.info("No delay trend data available for the selected route.")
            else:
                st.line_chart(trend_df.set_index("window_start")["avg_delay_sec"])
                st.caption(
                    "Window-level route KPIs are exported from Gold-layer aggregates over recent telemetry batches."
                )

# ===== Daily summary =====
st.subheader("Daily Summary")
st.caption("Daily aggregated metrics per route")

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
    with st.expander("Inspect underlying daily metrics"):
        st.dataframe(
            df_daily_filtered[available_cols],
            use_container_width=True,
            height=300,
        )

st.caption(f"Page rendered in {time.time() - start_time:.2f}s")
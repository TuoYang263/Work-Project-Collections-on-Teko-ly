import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from utils.load_data import load_route_window, load_route_daily


def safe_mean(series: pd.Series, decimals: int = 2):
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return "N/A"
    return round(series.mean(), decimals)


def safe_rate_pct(series: pd.Series, decimals: int = 1):
    # Normalize input to numeric values. Non-numeric values are treated as missing
    # to keep dashboard metrics stable when source data contains dirty records.
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return "N/A"
    return f"{series.mean() * 100:.{decimals}f}%"


def safe_sum(series: pd.Series):
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return 0
    return int(series.sum())


start_time = time.time()

st.title("Route Performance")
st.caption("Route-level KPIs derived from recent exported Gold-layer aggregates.")

# ===== Load data =====
with st.spinner("Loading route performance data..."):
    df_window = load_route_window()
    df_daily = load_route_daily()

# ===== Basic guard =====
if df_window is None or df_window.empty:
    st.warning("No route window data available.")
    st.stop()

df_window = df_window.copy()

now_local = datetime.now(ZoneInfo("Europe/Helsinki"))


def to_helsinki(ts: pd.Series) -> pd.Series:
    s = pd.to_datetime(ts, errors="coerce", utc=True)
    return s.dt.tz_convert("Europe/Helsinki")


for col in ["window_start", "window_end"]:
    if col in df_window.columns:
        df_window[col] = to_helsinki(df_window[col])

latest_window_end = (
    df_window["window_end"].max()
    if "window_end" in df_window.columns and df_window["window_end"].notna().any()
    else pd.NaT
)

if pd.notna(latest_window_end):
    freshness_min = max(
        0, int((now_local - latest_window_end.to_pydatetime()).total_seconds() / 60)
    )
    st.caption(
        f"Latest exported data window: {latest_window_end.strftime('%Y-%m-%d %H:%M')} "
        f"Helsinki time · data age: ~{freshness_min} min"
    )
else:
    st.caption("Latest exported data window: N/A")

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
        if not df_daily.empty
        else pd.DataFrame()
    )
else:
    df_window_filtered = df_window.copy()
    df_daily_filtered = df_daily.copy()

if (
    "window_end" in df_window_filtered.columns
    and df_window_filtered["window_end"].notna().any()
):
    latest_window_end = df_window_filtered["window_end"].max()
    latest_window_df = df_window_filtered[
        df_window_filtered["window_end"] == latest_window_end
    ].copy()
else:
    latest_window_df = df_window_filtered.copy()

# ===== KPI =====
col1, col2, col3, col4 = st.columns(4)

col1.metric("Avg Delay (s)", safe_mean(latest_window_df["avg_delay_sec"]))
col2.metric("Avg Occupancy (%)", safe_mean(latest_window_df["avg_occupancy_pct"]))
col3.metric("Late Rate (%)", safe_rate_pct(latest_window_df["late_rate_delay"]))

if "n_events_delay" in latest_window_df.columns:
    observed_events = safe_sum(latest_window_df["n_events_delay"])
elif "total_events_delay" in latest_window_df.columns:
    observed_events = safe_sum(latest_window_df["total_events_delay"])
else:
    observed_events = len(latest_window_df)

col4.metric("Observed Events", observed_events)

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
                st.caption("X-axis: route ID.")
                st.caption("Y-axis: average delay in seconds.")
                st.caption(
                    "Route ranking is based on exported Gold-layer daily summaries. "
                    "Higher average delay may indicate routes that need closer operational attention."
                )
    else:
        st.subheader("Delay Trend")

        trend_df = df_window_filtered[["window_start", "avg_delay_sec"]].dropna().copy()

        if trend_df.empty:
            st.info("No delay trend data available for the selected route.")
        else:
            trend_df = trend_df.sort_values("window_start").tail(24)

            if trend_df.empty:
                st.info("No delay trend data available for the selected route.")
            else:
                chart_df = (
                    trend_df.sort_values("window_start")
                    .set_index("window_start")[["avg_delay_sec"]]
                    .rename(columns={"avg_delay_sec": "Avg delay (s)"})
                )
                st.line_chart(chart_df)
                st.caption("X-axis: Helsinki local time.")
                st.caption("Y-axis: average delay in seconds.")
                st.caption(
                    "Window-level route KPIs are exported from Gold-layer aggregates over recent telemetry batches."
                )

        # ===== Window-level reference table =====
        if selected_route != "All":
            st.subheader("Recent Window Metrics")
            st.caption("Recent window-level route metrics for reference.")

            window_display_cols = [
                "window_start",
                "window_end",
                "route_id",
                "avg_delay_sec",
                "avg_occupancy_pct",
                "n_events_delay",
                "n_events_occupancy",
                "late_rate_pct",
                "avg_ingest_delay_sec",
                "dq_flag",
            ]

            if df_window_filtered.empty:
                st.info("No recent window-level metrics available.")
            else:
                window_table_df = df_window_filtered.copy()

                if "late_rate_delay" in window_table_df.columns:
                    window_table_df["late_rate_pct"] = (
                        pd.to_numeric(
                            window_table_df["late_rate_delay"], errors="coerce"
                        )
                        * 100
                    ).round(1)

                if "window_start" in window_table_df.columns:
                    window_table_df = window_table_df.sort_values(
                        "window_start", ascending=False
                    ).head(12)

                available_window_cols = [
                    c for c in window_display_cols if c in window_table_df.columns
                ]

                with st.expander("Underlying recent window metrics"):
                    st.dataframe(
                        window_table_df[available_window_cols],
                        use_container_width=True,
                        height=300,
                    )

# ===== Daily summary =====
st.subheader("Daily Summary")
st.caption("Daily aggregated metrics per route")

if df_daily_filtered.empty:
    st.info("No daily summary data available.")
else:
    daily_table_df = df_daily_filtered.copy()

    if "avg_late_rate_delay" in daily_table_df.columns:
        daily_table_df["avg_late_rate_pct"] = (
            pd.to_numeric(daily_table_df["avg_late_rate_delay"], errors="coerce") * 100
        ).round(1)

    display_cols = [
        "date",
        "route_id",
        "avg_delay_sec",
        "avg_occupancy_pct",
        "total_events_delay",
        "total_events_occupancy",
        "avg_late_rate_pct",
        "avg_ingest_delay_sec",
        "dq_flag",
    ]

    available_cols = [col for col in display_cols if col in daily_table_df.columns]
    with st.expander("Underlying daily metrics (for reference)"):
        st.dataframe(
            daily_table_df[available_cols],
            use_container_width=True,
            height=300,
        )
    st.caption(
        "This table reflects the exported Gold-layer daily aggregates per route."
    )

st.caption(
    "Data is refreshed by the scheduled pipeline. Route metrics represent "
    "the latest exported Gold-layer snapshots."
)

st.caption(f"Page rendered in {time.time() - start_time:.2f}s")

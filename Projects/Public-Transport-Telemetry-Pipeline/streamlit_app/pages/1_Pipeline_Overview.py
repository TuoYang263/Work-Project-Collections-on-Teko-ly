import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from utils.load_data import load_pipeline_metrics


start_time = time.time()
now_str = datetime.now(ZoneInfo("Europe/Helsinki")).strftime("%Y-%m-%d %H:%M")

st.title("Pipeline Overview")
st.caption("Recent 60-minute operational snapshot based on exported Gold-layer pipeline metrics.")
st.caption(f"Last updated: {now_str} (Helsinki time)")

with st.spinner("Loading latest pipeline metrics..."):
    df = load_pipeline_metrics()

if df is None or df.empty:
    st.warning("No pipeline metrics available.")
    st.stop()

df = df.copy()
for col in ["window_start", "window_end"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

# --------------------------------------------------
# Use the most recent valid transit window for KPI
# --------------------------------------------------
transit_valid_df = df.copy()

if "transit_total_events" in transit_valid_df.columns:
    transit_valid_df = transit_valid_df[
        transit_valid_df["transit_total_events"].notna()
        & (transit_valid_df["transit_total_events"].fillna(0) > 0)
    ].copy()

if "window_end" in transit_valid_df.columns and not transit_valid_df.empty:
    latest_transit_window_end = transit_valid_df["window_end"].max()
    latest_df = transit_valid_df[
        transit_valid_df["window_end"] == latest_transit_window_end
    ].copy()
else:
    latest_df = pd.DataFrame(columns=df.columns)

# ===== KPI =====
col1, col2, col3 = st.columns(3)

if not latest_df.empty:
    avg_ingest_gap = latest_df["transit_avg_ingest_delay_sec"].dropna().mean()
    total_transit_events = int(latest_df["transit_total_events"].fillna(0).sum())
    metric_rows = len(latest_df)
else:
    avg_ingest_gap = None
    total_transit_events = 0
    metric_rows = 0

col1.metric(
    "Avg Event-to-Ingest Gap (s)",
    round(avg_ingest_gap, 2) if pd.notna(avg_ingest_gap) else "N/A",
)
col2.metric("Transit Events (recent batch)", total_transit_events)
col3.metric("Pipeline Metric Rows", metric_rows)

st.caption(
    "KPI values are computed from the most recent transit window with valid metrics, to ensure stability when different data streams are not perfectly aligned."
    "This avoids empty latest-window cases when weather and transit windows do not align perfectly."
)

# ===== Trend =====
st.subheader("Pipeline Delay Trend")

df_plot = df.copy()

if "transit_avg_ingest_delay_sec" in df_plot.columns:
    df_plot = df_plot.dropna(subset=["window_start", "transit_avg_ingest_delay_sec"]).copy()
else:
    df_plot = pd.DataFrame()

if not df_plot.empty:
    df_plot = df_plot.sort_values("window_start").tail(12)

if df_plot.empty:
    st.info("No transit pipeline trend data available.")
else:
    st.line_chart(
        df_plot.set_index("window_start")["transit_avg_ingest_delay_sec"],
        height=300,
    )

st.subheader("Delivered Scope")
st.markdown(
    """
- Gold-layer pipeline metrics exported for dashboard consumption
- Recent batch-oriented telemetry summary over the latest available transit windows
- Lightweight, decoupled serving layer based on exported outputs
    """
)

st.caption(
    "This page is intended for stable inspection of recent exported pipeline outputs, "
    "not for live pipeline monitoring."
)

st.caption(f"Page rendered in {time.time() - start_time:.2f}s")

# --------------------------------------------------
# Temporary debug panels
# --------------------------------------------------
with st.expander("Pipeline metrics debug"):
    debug_df = load_pipeline_metrics()

    st.write("shape:", debug_df.shape)
    st.write("columns:", list(debug_df.columns))

    if not debug_df.empty:
        st.dataframe(debug_df.tail(10), use_container_width=True)
    else:
        st.warning("pipeline_metrics.parquet is empty.")

with st.expander("Latest pipeline metric row"):
    debug_df = load_pipeline_metrics()

    if not debug_df.empty:
        latest_row = debug_df.tail(1)
        st.dataframe(latest_row, use_container_width=True)
    else:
        st.warning("No pipeline metric rows available.")

with st.expander("Latest valid transit row used by KPI"):
    if not latest_df.empty:
        st.dataframe(latest_df, use_container_width=True)
    else:
        st.warning("No valid transit row available for KPI rendering.")

with st.expander("Null check summary"):
    debug_df = load_pipeline_metrics()

    if not debug_df.empty:
        latest = debug_df.tail(1)

        cols_to_check = [
            "transit_total_events",
            "transit_avg_ingest_delay_sec",
            "weather_total_events",
            "weather_avg_ingest_delay_sec",
        ]
        existing_cols = [c for c in cols_to_check if c in latest.columns]

        if existing_cols:
            st.dataframe(latest[existing_cols], use_container_width=True)
        else:
            st.write("Expected metric columns not found.")
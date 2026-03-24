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

if "window_end" in df.columns and df["window_end"].notna().any():
    latest_window_end = df["window_end"].max()
    latest_df = df[df["window_end"] == latest_window_end].copy()
else:
    latest_df = df.copy()

# ===== KPI =====
col1, col2, col3 = st.columns(3)

avg_ingest_gap = latest_df["transit_avg_ingest_delay_sec"].dropna().mean()
total_transit_events = int(latest_df["transit_total_events"].fillna(0).sum())

col1.metric(
    "Avg Event-to-Ingest Gap (s)",
    round(avg_ingest_gap, 2) if pd.notna(avg_ingest_gap) else "N/A",
)
col2.metric("Transit Events (latest batch)", total_transit_events)
col3.metric("Pipeline Metric Rows", len(latest_df))

st.caption(
    "The gap metric reflects recent simulated telemetry batches rather than a live streaming SLA."
)

# ===== Trend =====
st.subheader("Pipeline Delay Trend")

df_plot = df.dropna(subset=["window_start"]).sort_values("window_start").copy()
if not df_plot.empty:
    df_plot = df_plot.tail(12)

if df_plot.empty:
    st.info("No pipeline trend data available.")
else:
    st.line_chart(
        df_plot.set_index("window_start")["transit_avg_ingest_delay_sec"],
        height=300,
    )

st.subheader("Delivered Scope")
st.markdown(
    """
- Gold-layer pipeline metrics exported for dashboard consumption
- Recent batch-oriented telemetry summary over the latest available 60-minute lookback
- Lightweight, decoupled serving layer based on exported outputs
    """
)

st.caption(
    "This page is intended for stable inspection of recent exported pipeline outputs, "
    "not for live pipeline monitoring."
)

st.caption(f"Page rendered in {time.time() - start_time:.2f}s")

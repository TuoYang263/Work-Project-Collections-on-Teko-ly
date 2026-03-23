import time
import streamlit as st
import pandas as pd
from utils.load_data import load_pipeline_metrics


def format_latest_timestamp(df: pd.DataFrame, column: str) -> str:
    if df is None or df.empty or column not in df.columns:
        return "N/A"

    ts = pd.to_datetime(df[column], errors="coerce").max()
    if pd.isna(ts):
        return "N/A"
    return ts.strftime("%Y-%m-%d %H:%M")


start_time = time.time()
st.title("Pipeline Overview")
st.caption("Operational summary based on exported Gold-layer pipeline metrics.")

with st.spinner("Loading latest pipeline metrics..."):
    df = load_pipeline_metrics()

if df is None or df.empty:
    st.warning("No pipeline metrics available.")
    st.stop()

latest_pipeline_ts = format_latest_timestamp(df, "window_end")
st.caption(f"Latest pipeline window: {latest_pipeline_ts}")

# ===== KPI =====
col1, col2, col3 = st.columns(3)

avg_ingest_delay = df["transit_avg_ingest_delay_sec"].dropna().mean()
total_transit_events = int(df["transit_total_events"].fillna(0).sum())

col1.metric(
    "Avg Ingest Delay (s)",
    round(avg_ingest_delay, 2) if pd.notna(avg_ingest_delay) else "N/A",
)
col2.metric("Transit Events", total_transit_events)
col3.metric("Pipeline Metric Rows", len(df))

# ===== Trend =====
st.subheader("Pipeline Delay Trend")

df_plot = df.copy()
df_plot["window_start"] = pd.to_datetime(df_plot["window_start"], errors="coerce")
df_plot = df_plot.dropna(subset=["window_start"]).sort_values("window_start")

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
- Delay-oriented operational summary for recent telemetry windows
- Lightweight, decoupled serving layer based on exported outputs
    """
)

st.caption(
    "This page presents exported Gold-layer summaries intended for stable inspection "
    "rather than live pipeline execution."
)

st.caption(f"Page rendered in {time.time() - start_time:.2f}s")
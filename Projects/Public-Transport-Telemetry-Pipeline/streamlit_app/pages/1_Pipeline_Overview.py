import time
import streamlit as st
import pandas as pd
from utils.load_data import load_pipeline_metrics

start_time = time.time()
st.title("Pipeline Overview")

with st.spinner("Loading latest pipeline metrics..."):
    df = load_pipeline_metrics()

if df is None or df.empty:
    st.warning("No pipeline metrics available.")
    st.stop()

# ===== KPI =====
col1, col2, col3 = st.columns(3)

col1.metric("Avg Ingest Delay (s)", round(df["transit_avg_ingest_delay_sec"].mean(), 2))
col2.metric("Transit Events", int(df["transit_total_events"].fillna(0).sum()))
col3.metric("Pipeline Metric Rows", len(df))

# ===== Trend =====
st.subheader("Pipeline Delay Trend")

df_plot = df.copy()
df_plot["window_start"] = pd.to_datetime(df_plot["window_start"])
df_plot = df_plot.sort_values("window_start")

st.line_chart(
    df_plot.set_index("window_start")["transit_avg_ingest_delay_sec"],
    height=300)

st.caption("Simulated pipeline latency over fixed time windows (demo data).")

st.caption(f"Page rendered in {time.time() - start_time:.2f}s")
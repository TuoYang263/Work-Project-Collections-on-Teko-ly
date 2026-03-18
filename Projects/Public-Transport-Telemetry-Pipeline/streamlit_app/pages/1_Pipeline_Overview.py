import streamlit as st
import pandas as pd
from utils.load_data import load_pipeline_metrics

st.title("Pipeline Overview")

df = load_pipeline_metrics()

# ===== KPI =====
col1, col2, col3 = st.columns(3)

col1.metric("Avg Ingest Delay (s)", round(df["transit_avg_ingest_delay_sec"].mean(), 2))
col2.metric("Transit Events", int(df["transit_total_events"].fillna(0).sum()))
col3.metric("Pipeline Metric Rows", len(df))

# ===== Trend =====
st.subheader("Pipeline Delay Trend")

df_sorted = df.sort_values("window_start")
st.line_chart(df_sorted.set_index("window_start")["transit_avg_ingest_delay_sec"],
              height=300)
st.caption("Simulated pipeline latency over fixed time windows (demo data).")
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from utils.load_data import load_pipeline_metrics
from utils.openai_explainer import generate_ai_explanation, has_openai_config

from utils.insights import (
    explain_pipeline_metrics,
    explain_snapshot_status,
    render_insight_box,
)

start_time = time.time()
now_local = datetime.now(ZoneInfo("Europe/Helsinki"))

st.title("Pipeline Overview")
st.caption("Recent operational snapshot based on exported Gold-layer pipeline metrics.")

with st.spinner("Loading latest pipeline metrics..."):
    df = load_pipeline_metrics()

if df is None or df.empty:
    st.warning("No pipeline metrics available.")
    st.stop()

df = df.copy()


def to_helsinki(ts: pd.Series) -> pd.Series:
    s = pd.to_datetime(ts, errors="coerce", utc=True)
    return s.dt.tz_convert("Europe/Helsinki")


for col in ["window_start", "window_end"]:
    if col in df.columns:
        df[col] = to_helsinki(df[col])

latest_window_end = df["window_end"].max() if "window_end" in df.columns else pd.NaT
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
    if "transit_avg_ingest_delay_sec" in latest_df.columns:
        avg_ingest_gap = latest_df["transit_avg_ingest_delay_sec"].dropna().mean()
    else:
        avg_ingest_gap = None

    if "transit_total_events" in latest_df.columns:
        total_transit_events = int(latest_df["transit_total_events"].fillna(0).sum())
    else:
        total_transit_events = 0

    # Use a business-facing data quality indicator instead of exposing row counts.
    # If dq_flag is not available in pipeline_metrics, infer a lightweight status
    # from the latest valid transit window.
    if "dq_flag" in latest_df.columns:
        dq_values = latest_df["dq_flag"].dropna().astype(str).str.upper()

        if dq_values.empty:
            dq_status = (
                "OK"
                if total_transit_events > 0 and pd.notna(avg_ingest_gap)
                else "Check"
            )
        elif dq_values.isin(["OK", "PASS", "VALID", "CHECK"]).all():
            dq_status = "OK"
        else:
            dq_status = "Check"
    else:
        dq_status = (
            "OK" if total_transit_events > 0 and pd.notna(avg_ingest_gap) else "Check"
        )
else:
    avg_ingest_gap = None
    total_transit_events = 0
    dq_status = "N/A"

col1.metric(
    "Avg Ingest Delay (s)",
    round(avg_ingest_gap, 2) if pd.notna(avg_ingest_gap) else "N/A",
)
col2.metric("Events Processed", total_transit_events)
col3.metric("Data Quality Status", dq_status)

st.caption(
    "KPIs are calculated from the most recent valid transit window. "
    "This keeps the dashboard stable when transit and weather data are refreshed at different times."
)

snapshot_age_min = freshness_min if pd.notna(latest_window_end) else None

pipeline_insights = [
    explain_snapshot_status(snapshot_age_min),
    *explain_pipeline_metrics(
        total_events=total_transit_events,
        avg_ingest_delay_sec=avg_ingest_gap,
        dq_status=dq_status,
    ),
]

render_insight_box(
    st,
    "How to read this snapshot",
    pipeline_insights,
)

st.caption(
    "Optional AI explanation rewrites the rule-based facts above into a short plain-English summary. "
    "It does not calculate metrics or infer live operational causes."
)

if "pipeline_ai_explanation" not in st.session_state:
    st.session_state["pipeline_ai_explanation"] = None

generate_pipeline_ai = st.button(
    "Generate AI explanation",
    key="pipeline_ai_explanation_button",
)

if generate_pipeline_ai:
    if not has_openai_config():
        st.info(
            "OpenAI API key is not configured. Rule-based insights remain available."
        )
    else:
        with st.spinner("Generating AI explanation..."):
            st.session_state["pipeline_ai_explanation"] = generate_ai_explanation(
                facts=pipeline_insights,
                page_context=(
                    "Pipeline Overview page showing the latest exported Gold-layer "
                    "pipeline metrics for a scheduled snapshot dashboard."
                ),
            )

        if st.session_state["pipeline_ai_explanation"] is None:
            st.info(
                "AI explanation is unavailable. Rule-based insights remain available."
            )

if st.session_state["pipeline_ai_explanation"]:
    st.markdown("**AI-generated explanation**")
    st.info(st.session_state["pipeline_ai_explanation"])

    if st.button(
        "Clear AI explanation",
        key="clear_pipeline_ai_explanation_button",
    ):
        st.session_state["pipeline_ai_explanation"] = None
        st.rerun()

# ===== Trend =====
st.subheader("Pipeline Delay Trend")

df_plot = df.copy()

if "transit_avg_ingest_delay_sec" in df_plot.columns:
    df_plot = df_plot.dropna(
        subset=["window_start", "transit_avg_ingest_delay_sec"]
    ).copy()
else:
    df_plot = pd.DataFrame()

if not df_plot.empty:
    df_plot = df_plot.sort_values("window_start").tail(24)

if df_plot.empty:
    st.info("No transit pipeline trend data available.")
else:
    # Horizontal axis
    chart_df = (
        df_plot.sort_values("window_start")
        .set_index("window_start")[["transit_avg_ingest_delay_sec"]]
        .rename(columns={"transit_avg_ingest_delay_sec": "Avg ingest delay (s)"})
    )

    st.line_chart(chart_df, height=300)

    # Vertical axis instructions
    st.caption("Average ingest delay (seconds). Time axis in Helsinki local time.")

st.subheader("Delivered Scope")
st.markdown("""
- Gold-layer pipeline metrics exported for dashboard consumption
- Recent batch-oriented telemetry summary over the latest available transit windows
- Lightweight, decoupled serving layer based on exported outputs
    """)

st.caption(
    "This page is intended for stable inspection of recent exported pipeline outputs, "
    "not for live pipeline monitoring."
)

st.caption(
    "Data is refreshed by the Azure Container Apps scheduled pipeline. Metrics represent the latest exported Gold-layer snapshot."
)

st.caption(f"Page rendered in {time.time() - start_time:.2f}s")

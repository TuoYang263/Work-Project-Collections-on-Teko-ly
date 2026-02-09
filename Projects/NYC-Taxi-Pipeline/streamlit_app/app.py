# app.py
import sys
import streamlit as st
from datetime import datetime

st.set_page_config(
    page_title="NYC Yellow Taxi Dashboard",
    layout="wide"
)

# kill switch via query param
if st.query_params.get("sleep") == "1":
    st.write("Forcing app to exit for cold-start test")
    sys.exit(1)

if "health" in st.query_params:
    st.write("ok")
    st.stop()

st.title("NYC Yellow Taxi Analytics Dashboard")

st.markdown("""
Welcome to a **Production-Style Data Engineering Pipeline** built by Tuo Yang.

This dashboard demonstrates a complete data engineering pipeline for analyzing NYC Yellow Taxi data, including:

---

### Pipeline Overview
- **Ingestion**: Parquet data loaded from public TLC source
- **Transformation**: Data cleaning and standardization with PySpark
- **Storage**: Processed data uploaded to BigQuery
- **Orchestration**: Airflow DAG for automation (demo)
- **Visualization**: Interactive UI built with Streamlit & Plotly

---

### Available Visualizations
- **Trend Viewer**: Analyze hourly/daily/weekly changes in fare, tip, trip count and more
- **Zone Heatmap**: Explore NYC zones by pickup/dropoff metrics

Navigate via the **sidebar** to start exploring the data!
""")

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
import streamlit as st

st.set_page_config(page_title="Telemetry Pipeline", layout="wide")

st.title("Public Transport Telemetry Pipeline")

st.markdown("""
This project demonstrates a production-style data pipeline for public transport telemetry.
            
**Current Scope**
- Simulated telemetry events
- Bronze -> Silver -> Gold pipeline
- Exported parquet datasets for downstream consumption
            
**Planned extensions:**
- HSL real transport data ingestion
- FMI weather data integration
- Map-based impact analysis
""")
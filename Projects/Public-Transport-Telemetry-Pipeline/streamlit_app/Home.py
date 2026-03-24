import streamlit as st

st.set_page_config(
    page_title="Public Transport Telemetry Pipeline",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 110rem;
        padding-top: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Public Transport Telemetry Pipeline")

st.markdown("""
A lightweight, production-oriented telemetry pipeline simulating public transport operations in the Helsinki region.

The project focuses on data flow clarity, engineering trade-offs, and delivery of stable, query-ready outputs rather than feature complexity.
""")

st.divider()

st.subheader("Delivered Scope")

st.markdown("""
- End-to-end Bronze → Silver → Gold pipeline using Spark and Delta
- Route-level performance metrics for window and daily views
- Gold parquet export for downstream consumption
- Azure Blob upload for lightweight serving and portability
- Streamlit dashboard with pipeline overview, route performance, and map view
- HSL route and vehicle map with FMI weather station context
""")

st.divider()

st.subheader("System Highlights")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
- **Table-driven pipeline design**
- **Clear time semantics** (`event_time` vs `ingest_time`)
- **Precomputed Gold outputs** for lightweight serving
    """)

with col2:
    st.markdown("""
- **Decoupled dashboard layer** reading exported parquet
- **Azure-compatible storage pattern**
- **Delivery-focused scope control**
    """)

st.divider()

st.caption(
    "This project is intentionally scoped as a compact, production-friendly telemetry system designed for clarity, portability, and explainable engineering trade-offs."
)
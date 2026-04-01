import os
import json
import streamlit as st
from utils import config
from google.cloud import bigquery
from google.oauth2 import service_account

@st.cache_resource
def get_bigquery_client():
    """
    Priority:
    1) Streamlit secrets -> use st.secrets["gcp_service_account"]
    2) Render dev -> GCP_SA_JSON
    3) Local dev -> use SERVICE_ACCOUNT_PATH if exists
    4) Fallback -> Application Default Credentials (ADC)
    """

    streamlit_error_1 = None

    # 1) Streamlit secrets
    try:
        sa_info = dict(st.secrets["gcp_service_account"])
        credentials = service_account.Credentials.from_service_account_info(sa_info)
        return bigquery.Client(project=sa_info.get("project_id"), credentials=credentials)
    except Exception as e:
        streamlit_error_1 = f"st.secrets auth failed: {type(e).__name__}: {e}"

    # 2) Render env
    try:
        sa_json = os.getenv("GCP_SA_JSON")
        if not sa_json:
            raise RuntimeError("GCP_SA_JSON is missing or empty")

        sa_info = json.loads(sa_json)
        if isinstance(sa_info, str):
            sa_info = json.loads(sa_info)

        credentials = service_account.Credentials.from_service_account_info(sa_info)
        return bigquery.Client(project=sa_info.get("project_id"), credentials=credentials)

    except Exception as e:
        raise RuntimeError(
            "BigQuery auth debug:\n"
            f"1) {streamlit_error_1}\n"
            f"2) GCP_SA_JSON failed: {type(e).__name__}: {e}"
        )
    
    """
    # 3) Local: service account file path (optional)
    sa_path = getattr(config, "SERVICE_ACCOUNT_PATH", None)
    if sa_path and os.path.exists(sa_path):
        credentials = service_account.Credentials.from_service_account_file(sa_path)
        return bigquery.Client(project=config.PROJECT_ID, credentials=credentials)
    
    # 4) Fallback: ADC (gcloud auth application-default login, etc.)
    return bigquery.Client(project=getattr(config, "PROJECT_ID", None))
    """
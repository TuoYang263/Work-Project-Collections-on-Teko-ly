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
    1) Streamlit Community Cloud -> use st.secrets["gcp_service_account"]
    2) Local dev -> use SERVICE_ACCOUNT_PATH if exists
    3) Fallback -> Application Default Credentials (ADC)
    """

    # 1) Cloud: Streamlit secrets (recommended)
    try:
        sa_info = dict(st.secrets["gcp_service_account"])
        credentials = service_account.Credentials.from_service_account_info(sa_info)
        return bigquery.Client(project=sa_info.get("project_id"), credentials=credentials)
    except Exception:
        pass
    
    # 2) Local: service account file path (optional)
    sa_path = getattr(config, "SERVICE_ACCOUNT_PATH", None)
    if sa_path and os.path.exists(sa_path):
        credentials = service_account.Credentials.from_service_account_file(sa_path)
        return bigquery.Client(project=config.PROJECT_ID, credentials=credentials)
    
    # 3) Fallback: ADC (gcloud auth application-default login, etc.)
    return bigquery.Client(project=getattr(config, "PROJECT_ID", None))
import os
import sys
from google.cloud import bigquery
from google.oauth2 import service_account
from dotenv import load_dotenv
load_dotenv()

# --- Robust imports (works whether files are in the root path or in scripts/) ---
try:
    from scripts import config
except Exception:
    import config

sys.path.append(config.BASE_DIR)
sys.path.append(config.PROJECT_BASE_DIR)

from scripts.logger import get_logger
logger = get_logger()

def get_bigquery_client():
    logger.info("Initializing BigQuery client using ADC")
    return bigquery.Client()
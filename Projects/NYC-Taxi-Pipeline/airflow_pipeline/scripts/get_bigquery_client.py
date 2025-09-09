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
    # Define credential paths for both local and container environments
    default_cred_rel = os.path.join("creds", "gcp_service_account.json")
    env_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_PATH", default_cred_rel)
    cred_path_local = env_path if os.path.isabs(env_path) else os.path.join(config.PROJECT_BASE_DIR, env_path)

    cred_path_container = "/opt/airflow/creds/gcp_service_account.json"

    # Determine whether running inside a Docker container
    is_container = os.path.exists("/.dockerenv") or os.getenv("DOCKER_CONTAINER") == "true"

    # Choose appropriate credential path
    cred_path = cred_path_container if is_container else cred_path_local
    logger.info(config.PROJECT_BASE_DIR)
    logger.info(f"Using credential path: {cred_path}")

    # Raise error if the credential file does not exist
    if not os.path.exists(cred_path):
        raise FileNotFoundError(f"Credential file not found: {cred_path}")

    # Load credentials and initialize BigQuery client
    credentials = service_account.Credentials.from_service_account_file(cred_path)
    return bigquery.Client(credentials=credentials, location="US")
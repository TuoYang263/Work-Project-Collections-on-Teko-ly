#!/usr/bin/env bash
set -euo pipefail

# Write service account JSON to file
echo "$GCP_SA_JSON" > /tmp/gcp_sa.json

# Export credential path
export GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp_sa.json

# Start Streamlit on Render port
exec streamlit run app.py \
  --server.address 0.0.0.0 \
  --server.port "$PORT"
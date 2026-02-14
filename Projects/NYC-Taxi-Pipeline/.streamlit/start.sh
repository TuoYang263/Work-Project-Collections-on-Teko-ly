#!/usr/bin/env bash
set -euo pipefail

# Use Render Secret File if provided; fallback to default secret path.
export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-/etc/secrets/gcp-sa.json}"

# Start Streamlit on Render port
exec streamlit run app.py \
  --server.address 0.0.0.0 \
  --server.port "$PORT"
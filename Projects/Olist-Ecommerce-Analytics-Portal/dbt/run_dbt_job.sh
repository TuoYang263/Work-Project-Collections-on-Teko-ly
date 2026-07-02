#!/usr/bin/env bash
set -euo pipefail

DBT_TARGET="${DBT_TARGET:-prod}"
DBT_DATASET="${DBT_DATASET:-olist}"
DBT_LOCATION="${DBT_LOCATION:-EU}"
DBT_THREADS="${DBT_THREADS:-4}"

if [ -z "${DBT_PROJECT_ID:-}" ]; then
  echo "ERROR: DBT_PROJECT_ID environment variable is required."
  exit 1
fi

if [ ! -f "profiles.yml.template" ]; then
  echo "ERROR: profiles.yml.template was not found in the current working directory."
  exit 1
fi

echo "Starting Olist dbt pipeline..."
echo "Current working directory: $(pwd)"
echo "Using dbt target: ${DBT_TARGET}"
echo "Using BigQuery project: ${DBT_PROJECT_ID}"
echo "Using default dbt dataset: ${DBT_DATASET}"
echo "Using BigQuery location: ${DBT_LOCATION}"

echo "Generating dbt profiles.yml from runtime environment..."
envsubst < profiles.yml.template > profiles.yml

echo "dbt version:"
dbt --version

if [ -f "packages.yml" ]; then
  echo "packages.yml found. Installing dbt packages..."
  dbt deps
else
  echo "No packages.yml found. Skipping dbt deps."
fi

echo "Validating dbt connection..."
dbt debug --target "${DBT_TARGET}"

echo "Running dbt build..."
dbt build --target "${DBT_TARGET}"

echo "Olist dbt pipeline completed successfully."
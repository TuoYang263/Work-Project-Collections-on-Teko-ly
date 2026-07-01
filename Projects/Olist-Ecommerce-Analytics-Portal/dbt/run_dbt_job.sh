#!/usr/bin/env bash
set -euo pipefail

DBT_TARGET="${DBT_TARGET:-prod}"

echo "Starting Olist dbt pipeline..."
echo "Current working directory: $(pwd)"
echo "Using dbt target: ${DBT_TARGET}"

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
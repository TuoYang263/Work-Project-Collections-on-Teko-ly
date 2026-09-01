# M7-M8 GCP Orchestration and Monitoring Commands

## Current note

This is a historical deployment and validation runbook for M7 and M8. The commands and image tag `m8` are kept because they record the cloud setup that was actually validated at that milestone.

The repository now also includes M9 and M10 U1. The existing Cloud Scheduler / Cloud Run Job still uses `run_dbt_job.sh` as the container entry point, so these commands should not be read as evidence that the scheduled job already starts `run_window_controller.py`.

For the current window-control runtime, see `docs/m10_window_control.md`.

---

## Purpose

This document records the commands used to deploy, update, execute, and validate the cloud orchestration and dbt artifact monitoring layers for the Olist E-Commerce Analytics & Pipeline Monitoring Portal.

Completed flow:

```text
Cloud Scheduler
    ↓
Cloud Run Job
    ↓
Containerized dbt build
    ↓
dbt artifact preservation and catalog generation
    ↓
BigQuery monitoring ingestion
    ↓
Cross-table validation
```

The commands are written with environment variables so the Google Cloud project ID is not hard-coded into the repository.

---

## Working directory

The Olist project is one project inside a larger portfolio repository.

From the portfolio repository root:

```bash
cd Projects/Olist-Ecommerce-Analytics-Portal
```

Most commands in this document assume the current directory is the Olist project root.

Confirm the location:

```bash
pwd
git status
```

---

## Common variables

```bash
export GCP_PROJECT_ID="$(gcloud config get-value project)"

export CLOUD_RUN_REGION="europe-north1"
export SCHEDULER_LOCATION="europe-west1"
export BQ_LOCATION="EU"

export AR_REPOSITORY="olist-dbt-jobs"
export IMAGE_NAME="olist-dbt-job"
export IMAGE_TAG="m8"
export REMOTE_IMAGE="${CLOUD_RUN_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${AR_REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}"

export CLOUD_RUN_JOB="olist-dbt-build-job"
export CLOUD_RUN_SA="olist-dbt-runner@${GCP_PROJECT_ID}.iam.gserviceaccount.com"

export SCHEDULER_JOB="olist-dbt-daily-trigger"
export SCHEDULER_SA="olist-scheduler-invoker@${GCP_PROJECT_ID}.iam.gserviceaccount.com"
export SCHEDULER_TIME_ZONE="Europe/Helsinki"
export SCHEDULER_CRON="0 6 * * *"
```

Verify important values:

```bash
printf 'GCP_PROJECT_ID=<%s>\n' "${GCP_PROJECT_ID}"
printf 'REMOTE_IMAGE=<%s>\n' "${REMOTE_IMAGE}"

declare -p \
  GCP_PROJECT_ID \
  CLOUD_RUN_REGION \
  SCHEDULER_LOCATION \
  AR_REPOSITORY \
  IMAGE_NAME \
  IMAGE_TAG \
  REMOTE_IMAGE
```

Set the active project when needed:

```bash
gcloud config set project "${GCP_PROJECT_ID}"
```

---

## Enable required Google Cloud APIs

```bash
gcloud services enable \
  run.googleapis.com \
  cloudscheduler.googleapis.com \
  artifactregistry.googleapis.com \
  iam.googleapis.com \
  bigquery.googleapis.com
```

---

## Service accounts

### Create Cloud Run runtime account

```bash
gcloud iam service-accounts create olist-dbt-runner \
  --display-name="Olist dbt Cloud Run Job Runner"
```

### Create Scheduler invoker account

```bash
gcloud iam service-accounts create olist-scheduler-invoker \
  --display-name="Olist Cloud Scheduler Invoker"
```

Skip create commands when the accounts already exist.

---

## BigQuery IAM

The Cloud Run runtime service account needs to run jobs and write data.

```bash
gcloud projects add-iam-policy-binding "${GCP_PROJECT_ID}" \
  --member="serviceAccount:${CLOUD_RUN_SA}" \
  --role="roles/bigquery.jobUser"

gcloud projects add-iam-policy-binding "${GCP_PROJECT_ID}" \
  --member="serviceAccount:${CLOUD_RUN_SA}" \
  --role="roles/bigquery.dataEditor"
```

For a stricter production deployment, limit data-editor access to the required datasets.

---

## Artifact Registry repository

Create the Docker repository if it does not exist:

```bash
gcloud artifacts repositories create "${AR_REPOSITORY}" \
  --repository-format=docker \
  --location="${CLOUD_RUN_REGION}" \
  --description="Docker images for Olist dbt Cloud Run Jobs"
```

Configure Docker authentication:

```bash
gcloud auth configure-docker \
  "${CLOUD_RUN_REGION}-docker.pkg.dev" \
  --quiet
```

---

# M8 BigQuery monitoring setup

## Create the monitoring dataset

```bash
bq \
  --project_id="${GCP_PROJECT_ID}" \
  --location="${BQ_LOCATION}" \
  query \
  --use_legacy_sql=false \
  < dbt/sql/monitoring/create_olist_monitoring_dataset.sql
```

## Create the six monitoring tables

```bash
bq \
  --project_id="${GCP_PROJECT_ID}" \
  --location="${BQ_LOCATION}" \
  query \
  --use_legacy_sql=false \
  < dbt/sql/monitoring/create_monitoring_tables.sql
```

## List the dataset and tables

```bash
bq --project_id="${GCP_PROJECT_ID}" ls
bq --project_id="${GCP_PROJECT_ID}" ls olist_monitoring
```

## Inspect table schemas

```bash
bq --project_id="${GCP_PROJECT_ID}" show olist_monitoring.pipeline_runs
bq --project_id="${GCP_PROJECT_ID}" show olist_monitoring.model_run_results
bq --project_id="${GCP_PROJECT_ID}" show olist_monitoring.test_run_results
```

## Validate table existence

Before the first load, all row counts are expected to be zero.

```bash
bq \
  --project_id="${GCP_PROJECT_ID}" \
  --location="${BQ_LOCATION}" \
  query \
  --use_legacy_sql=false \
  < dbt/sql/monitoring/validate_monitoring_tables.sql
```

---

# Local dbt artifact inspection and parsing

## Inspect artifact files

The artifact directory is ignored by Git.

```bash
python dbt/monitoring/inspect_artifacts.py
```

Validated artifact summary:

```text
manifest nodes: 115
manifest sources: 9
run_results results: 115
catalog nodes: 21
catalog sources: 9
```

## Format and run the parser

```bash
black dbt/monitoring/artifact_parser.py
python dbt/monitoring/artifact_parser.py
```

Validated parser output:

```text
pipeline run records:            1
model run records:              21
test run records:               94
model metadata records:         21
model column records:          259
model lineage records:         146
```

## Test runtime identity without loading BigQuery

```bash
MONITORING_JOB_NAME="olist-dbt-build-job" \
MONITORING_ENVIRONMENT="prod" \
python dbt/monitoring/artifact_parser.py
```

The pipeline record should show:

```text
job_name=olist-dbt-build-job
environment=prod
```

---

# Local BigQuery monitoring load

## Verify the BigQuery Python client

```bash
python -c \
  'from google.cloud import bigquery; print("BigQuery client OK")'
```

## Set the project and run the loader

```bash
export GCP_PROJECT_ID="$(gcloud config get-value project)"

python dbt/monitoring/load_artifacts_to_bigquery.py
```

Expected output:

```text
Inserted 1 records into ...olist_monitoring.pipeline_runs.
Inserted 21 records into ...olist_monitoring.model_run_results.
Inserted 94 records into ...olist_monitoring.test_run_results.
Inserted 21 records into ...olist_monitoring.model_metadata_snapshots.
Inserted 259 records into ...olist_monitoring.model_column_snapshots.
Inserted 146 records into ...olist_monitoring.model_lineage_edges.
```

The loader is append-only. Re-running it creates another monitoring run rather than overwriting prior history.

---

## Validate pipeline run records

```bash
bq \
  --project_id="${GCP_PROJECT_ID}" \
  --location="${BQ_LOCATION}" \
  query \
  --use_legacy_sql=false '
SELECT
  monitoring_run_id,
  dbt_invocation_id,
  job_name,
  environment,
  status,
  models_total,
  tests_total,
  ingested_at
FROM `olist_monitoring.pipeline_runs`
ORDER BY ingested_at DESC
LIMIT 5
'
```

## Validate model execution records

```bash
bq \
  --project_id="${GCP_PROJECT_ID}" \
  --location="${BQ_LOCATION}" \
  query \
  --use_legacy_sql=false '
SELECT
  monitoring_run_id,
  COUNT(*) AS model_record_count,
  COUNTIF(status = "success") AS successful_models
FROM `olist_monitoring.model_run_results`
GROUP BY monitoring_run_id
ORDER BY monitoring_run_id DESC
LIMIT 5
'
```

Expected latest-run values:

```text
model_record_count=21
successful_models=21
```

## Validate test records

```bash
bq \
  --project_id="${GCP_PROJECT_ID}" \
  --location="${BQ_LOCATION}" \
  query \
  --use_legacy_sql=false '
SELECT
  monitoring_run_id,
  COUNT(*) AS test_record_count,
  COUNTIF(status = "pass") AS passed_tests,
  COUNTIF(status = "fail") AS failed_tests,
  COUNTIF(status = "warn") AS warned_tests,
  COUNTIF(status = "error") AS error_tests
FROM `olist_monitoring.test_run_results`
GROUP BY monitoring_run_id
ORDER BY monitoring_run_id DESC
LIMIT 5
'
```

Expected latest-run values:

```text
test_record_count=94
passed_tests=94
failed_tests=0
warned_tests=0
error_tests=0
```

## Validate model metadata snapshots

```bash
bq \
  --project_id="${GCP_PROJECT_ID}" \
  --location="${BQ_LOCATION}" \
  query \
  --use_legacy_sql=false '
SELECT
  monitoring_run_id,
  COUNT(*) AS metadata_record_count,
  COUNTIF(materialized = "table") AS table_models,
  COUNTIF(materialized = "view") AS view_models,
  COUNTIF(row_count IS NOT NULL) AS models_with_row_count
FROM `olist_monitoring.model_metadata_snapshots`
GROUP BY monitoring_run_id
ORDER BY monitoring_run_id DESC
LIMIT 5
'
```

Validated output:

```text
metadata_record_count=21
table_models=9
view_models=12
models_with_row_count=9
```

Views normally do not have physical row-count statistics in the catalog.

## Validate model/source column snapshots

```bash
bq \
  --project_id="${GCP_PROJECT_ID}" \
  --location="${BQ_LOCATION}" \
  query \
  --use_legacy_sql=false '
SELECT
  monitoring_run_id,
  COUNT(*) AS column_record_count,
  COUNTIF(resource_type = "model") AS model_columns,
  COUNTIF(resource_type = "source") AS source_columns,
  COUNTIF(description IS NOT NULL AND description != "") AS documented_columns,
  COUNTIF(tests_json != "[]") AS columns_with_tests
FROM `olist_monitoring.model_column_snapshots`
GROUP BY monitoring_run_id
ORDER BY monitoring_run_id DESC
LIMIT 5
'
```

Validated output:

```text
column_record_count=259
model_columns=207
source_columns=52
documented_columns=176
columns_with_tests=58
```

## Validate lineage edges

```bash
bq \
  --project_id="${GCP_PROJECT_ID}" \
  --location="${BQ_LOCATION}" \
  query \
  --use_legacy_sql=false '
SELECT
  monitoring_run_id,
  COUNT(*) AS lineage_record_count,
  COUNTIF(
    parent_resource_type = "source"
    AND child_resource_type = "model"
  ) AS source_to_model_edges,
  COUNTIF(
    parent_resource_type = "model"
    AND child_resource_type = "model"
  ) AS model_to_model_edges,
  COUNTIF(child_resource_type = "test") AS model_to_test_edges
FROM `olist_monitoring.model_lineage_edges`
GROUP BY monitoring_run_id
ORDER BY monitoring_run_id DESC
LIMIT 5
'
```

Validated output:

```text
lineage_record_count=146
source_to_model_edges=9
model_to_model_edges=21
model_to_test_edges=116
```

A relationships test may depend on more than one model, so test lineage edges can exceed the number of test result rows.

---

## Validate the latest monitoring run across all six tables

```bash
bq \
  --project_id="${GCP_PROJECT_ID}" \
  --location="${BQ_LOCATION}" \
  query \
  --use_legacy_sql=false \
  < dbt/sql/monitoring/validate_latest_monitoring_run.sql
```

Validated result:

```text
pipeline_status                 success
models_total                    21
model_run_result_count          21
tests_total                     94
test_run_result_count           94
model_metadata_snapshot_count   21
model_column_snapshot_count     259
model_lineage_edge_count        146
successful_models               21
passed_tests                    94
non_passing_tests                0
```

---

# Repository and shell validation

## Force Linux line endings for shell scripts

Repository-root `.gitattributes`:

```gitattributes
*.sh text eol=lf
```

Renormalize the Cloud Run entrypoint:

```bash
git add .gitattributes

git add --renormalize \
  Projects/Olist-Ecommerce-Analytics-Portal/dbt/run_dbt_job.sh
```

## Ignore Python caches in Docker context

Project `.dockerignore`:

```dockerignore
**/__pycache__/
**/*.py[cod]
```

## Validate shell syntax and whitespace

From the Olist project root:

```bash
bash -n dbt/run_dbt_job.sh
git diff --check
```

No output indicates success.

---

# Docker image

## Build the local M8 image

Run from the Olist project root:

```bash
docker build \
  -f dbt/Dockerfile \
  -t olist-dbt-monitoring-test \
  .
```

## Check monitoring files inside the image

```bash
docker run --rm \
  --entrypoint bash \
  olist-dbt-monitoring-test \
  -c 'ls -l /app/dbt/monitoring'
```

Expected files include:

```text
artifact_parser.py
inspect_artifacts.py
load_artifacts_to_bigquery.py
```

## Check the BigQuery Python dependency

```bash
docker run --rm \
  --entrypoint python \
  olist-dbt-monitoring-test \
  -c 'from google.cloud import bigquery; print("BigQuery client OK")'
```

Expected:

```text
BigQuery client OK
```

## Tag the image for Artifact Registry

```bash
docker tag \
  olist-dbt-monitoring-test \
  "${REMOTE_IMAGE}"
```

## Push the M8 image

```bash
docker push "${REMOTE_IMAGE}"
```

Validated M8 push:

```text
image tag: m8
```

## List Artifact Registry images and tags

```bash
gcloud artifacts docker images list \
  "${CLOUD_RUN_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${AR_REPOSITORY}/${IMAGE_NAME}" \
  --include-tags
```

The list should include:

```text
m7
m8
```

---

# Cloud Run Job deployment

## Create the job

Use this only for a new environment.

```bash
gcloud run jobs create "${CLOUD_RUN_JOB}" \
  --project="${GCP_PROJECT_ID}" \
  --image="${REMOTE_IMAGE}" \
  --region="${CLOUD_RUN_REGION}" \
  --service-account="${CLOUD_RUN_SA}" \
  --set-env-vars="DBT_PROJECT_ID=${GCP_PROJECT_ID},GOOGLE_CLOUD_PROJECT=${GCP_PROJECT_ID},DBT_DATASET=olist,DBT_LOCATION=${BQ_LOCATION},DBT_THREADS=4,DBT_TARGET=prod,MONITORING_DATASET_ID=olist_monitoring,MONITORING_JOB_NAME=${CLOUD_RUN_JOB},MONITORING_ENVIRONMENT=prod" \
  --tasks=1 \
  --max-retries=1 \
  --task-timeout=3600 \
  --memory=1Gi \
  --cpu=1
```

## Inspect the existing job before updating

```bash
gcloud run jobs describe "${CLOUD_RUN_JOB}" \
  --project="${GCP_PROJECT_ID}" \
  --region="${CLOUD_RUN_REGION}" \
  --format=export \
  > /tmp/olist-dbt-build-job.yaml
```

```bash
grep -nE \
  'image:|serviceAccount:|DBT_PROJECT_ID|DBT_TARGET|DBT_DATASET|DBT_LOCATION|MONITORING_DATASET_ID|MONITORING_JOB_NAME|MONITORING_ENVIRONMENT' \
  /tmp/olist-dbt-build-job.yaml
```

## Update the existing job to image `m8`

`--update-env-vars` preserves unrelated existing environment variables.

```bash
gcloud run jobs update "${CLOUD_RUN_JOB}" \
  --project="${GCP_PROJECT_ID}" \
  --region="${CLOUD_RUN_REGION}" \
  --image="${REMOTE_IMAGE}" \
  --update-env-vars="MONITORING_DATASET_ID=olist_monitoring,MONITORING_JOB_NAME=${CLOUD_RUN_JOB},MONITORING_ENVIRONMENT=prod"
```

## Verify the updated job configuration

```bash
gcloud run jobs describe "${CLOUD_RUN_JOB}" \
  --project="${GCP_PROJECT_ID}" \
  --region="${CLOUD_RUN_REGION}" \
  --format=export \
  > /tmp/olist-dbt-build-job-m8.yaml
```

```bash
grep -nE \
  'image:|DBT_PROJECT_ID|DBT_TARGET|DBT_DATASET|DBT_LOCATION|MONITORING_DATASET_ID|MONITORING_JOB_NAME|MONITORING_ENVIRONMENT' \
  /tmp/olist-dbt-build-job-m8.yaml
```

Expected image:

```text
.../olist-dbt-job:m8
```

Expected monitoring variables:

```text
MONITORING_DATASET_ID=olist_monitoring
MONITORING_JOB_NAME=olist-dbt-build-job
MONITORING_ENVIRONMENT=prod
```

---

## Grant Scheduler permission to execute the job

```bash
gcloud run jobs add-iam-policy-binding "${CLOUD_RUN_JOB}" \
  --project="${GCP_PROJECT_ID}" \
  --region="${CLOUD_RUN_REGION}" \
  --member="serviceAccount:${SCHEDULER_SA}" \
  --role="roles/run.invoker"
```

Fallback project-level binding:

```bash
gcloud projects add-iam-policy-binding "${GCP_PROJECT_ID}" \
  --member="serviceAccount:${SCHEDULER_SA}" \
  --role="roles/run.invoker"
```

---

# Execute and inspect Cloud Run

## Manual Cloud Run smoke test

```bash
gcloud run jobs execute "${CLOUD_RUN_JOB}" \
  --project="${GCP_PROJECT_ID}" \
  --region="${CLOUD_RUN_REGION}" \
  --wait
```

Validated M8 manual execution:

```text
1 / 1 task completed
job_name=olist-dbt-build-job
environment=prod
status=success
models_total=21
tests_total=94
```

## List recent executions

```bash
gcloud run jobs executions list \
  --job="${CLOUD_RUN_JOB}" \
  --project="${GCP_PROJECT_ID}" \
  --region="${CLOUD_RUN_REGION}" \
  --sort-by="~metadata.creationTimestamp" \
  --limit=5
```

## Describe the latest execution

Some gcloud installations do not support `executions describe-latest`.

Compatible approach:

```bash
LATEST_EXECUTION="$(
  gcloud run jobs executions list \
    --job="${CLOUD_RUN_JOB}" \
    --project="${GCP_PROJECT_ID}" \
    --region="${CLOUD_RUN_REGION}" \
    --sort-by="~metadata.creationTimestamp" \
    --limit=1 \
    --format="value(metadata.name)"
)"

echo "${LATEST_EXECUTION}"

gcloud run jobs executions describe "${LATEST_EXECUTION}" \
  --project="${GCP_PROJECT_ID}" \
  --region="${CLOUD_RUN_REGION}"
```

## View Cloud Run logs

```bash
gcloud logging read \
  "resource.type=cloud_run_job AND resource.labels.job_name=\"${CLOUD_RUN_JOB}\"" \
  --project="${GCP_PROJECT_ID}" \
  --limit=150 \
  --format="table(timestamp,textPayload)"
```

---

# Cloud Scheduler

## Create the Scheduler trigger

Use this only in a new environment.

```bash
gcloud scheduler jobs create http "${SCHEDULER_JOB}" \
  --project="${GCP_PROJECT_ID}" \
  --location="${SCHEDULER_LOCATION}" \
  --schedule="${SCHEDULER_CRON}" \
  --time-zone="${SCHEDULER_TIME_ZONE}" \
  --uri="https://run.googleapis.com/v2/projects/${GCP_PROJECT_ID}/locations/${CLOUD_RUN_REGION}/jobs/${CLOUD_RUN_JOB}:run" \
  --http-method=POST \
  --oauth-service-account-email="${SCHEDULER_SA}" \
  --oauth-token-scope="https://www.googleapis.com/auth/cloud-platform"
```

## Inspect the Scheduler job

```bash
gcloud scheduler jobs describe "${SCHEDULER_JOB}" \
  --project="${GCP_PROJECT_ID}" \
  --location="${SCHEDULER_LOCATION}" \
  --format="yaml(name,schedule,timeZone,state,httpTarget.uri)"
```

## Force-run the Scheduler job

```bash
gcloud scheduler jobs run "${SCHEDULER_JOB}" \
  --project="${GCP_PROJECT_ID}" \
  --location="${SCHEDULER_LOCATION}"
```

Then list and describe the latest Cloud Run execution with the commands above.

Validated Scheduler-triggered M8 execution:

```text
execution: olist-dbt-build-job-f59xf
RUN BY: olist-scheduler-invoker service account
tasks: 1 / 1 completed successfully
elapsed time: approximately 1 minute 19 seconds
```

## Final Scheduler-triggered BigQuery validation

```bash
bq \
  --project_id="${GCP_PROJECT_ID}" \
  --location="${BQ_LOCATION}" \
  query \
  --use_legacy_sql=false \
  < dbt/sql/monitoring/validate_latest_monitoring_run.sql
```

Validated output:

```text
models: 21 / 21 successful
tests: 94 / 94 passed
metadata snapshots: 21
column snapshots: 259
lineage edges: 146
non-passing tests: 0
```

---

## Pause or resume the Scheduler

Pause after portfolio validation to avoid unnecessary runs:

```bash
gcloud scheduler jobs pause "${SCHEDULER_JOB}" \
  --project="${GCP_PROJECT_ID}" \
  --location="${SCHEDULER_LOCATION}"
```

Resume later:

```bash
gcloud scheduler jobs resume "${SCHEDULER_JOB}" \
  --project="${GCP_PROJECT_ID}" \
  --location="${SCHEDULER_LOCATION}"
```

---

# Troubleshooting

## Empty `GCP_PROJECT_ID`

Symptom:

```text
Cannot start a job without a project id
```

Fix:

```bash
export GCP_PROJECT_ID="$(gcloud config get-value project)"
echo "${GCP_PROJECT_ID}"
```

## `executions describe-latest` is unavailable

Use `executions list` plus `executions describe` as documented above.

## Shell script passes `bash -n` but fails at runtime

Bash variable assignments cannot contain spaces around `=`:

```bash
ARTIFACT_BACKUP_DIR="$(mktemp -d)"
```

File-test brackets require spaces:

```bash
if [ ! -f "monitoring/load_artifacts_to_bigquery.py" ]; then
```

## CRLF warning for shell scripts

Keep this in repository-root `.gitattributes`:

```gitattributes
*.sh text eol=lf
```

## BigQuery Preview does not show new streamed rows

Use a `SELECT` query. The BigQuery Preview pane may lag behind streaming inserts.

## Repeated local loads create several monitoring runs

This is expected. The M8 tables are append-only and generate a new `monitoring_run_id` for each loader execution.

## `dbt docs generate` replaces build artifacts

The entrypoint script temporarily backs up the build `manifest.json` and `run_results.json`, generates `catalog.json`, and restores the build files before loading monitoring data.

---

# Final validation checklist

## M7 orchestration

```text
[x] Dockerized dbt runtime
[x] Artifact Registry image
[x] Cloud Run Job
[x] Cloud Scheduler OAuth trigger
[x] Manual Cloud Run execution
[x] Scheduler-triggered execution
[x] dbt build PASS=115
```

## M8 monitoring

```text
[x] olist_monitoring dataset
[x] Six monitoring tables
[x] Artifact inspection
[x] Six parser record types
[x] Local BigQuery append loader
[x] Latest-run cross-table validation
[x] google-cloud-bigquery included in image
[x] M8 image pushed to Artifact Registry
[x] Cloud Run Job updated to m8
[x] Manual cloud monitoring load
[x] Scheduler-triggered monitoring load
[x] Production job/environment identity
[x] 21 models successful
[x] 94 tests passed
[x] 259 column snapshots
[x] 146 lineage edges
[x] 0 non-passing tests
```

---

## Validated deployment summary

```text
Cloud Run Job: olist-dbt-build-job
Cloud Run region: europe-north1
Cloud Scheduler: olist-dbt-daily-trigger
Scheduler location: europe-west1
Schedule: 0 6 * * *
Time zone: Europe/Helsinki
Artifact Registry image: olist-dbt-job:m8
Monitoring dataset: olist_monitoring
Validation date: 2026-07-15
```

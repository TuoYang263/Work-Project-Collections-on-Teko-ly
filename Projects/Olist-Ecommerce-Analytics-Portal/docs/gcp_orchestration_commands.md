# M7 GCP Orchestration Commands

## Purpose

This document records the Google Cloud commands used to deploy and validate the M7 orchestration layer for the Olist E-Commerce Analytics & Pipeline Monitoring Portal project.

M7 uses:

```text
Artifact Registry
    ↓
Cloud Run Job
    ↓
Containerized dbt project
    ↓
Cloud Scheduler trigger
    ↓
BigQuery dbt models and tests
```

The goal is to make the existing dbt pipeline cloud-executable and schedulable without adding M8 metadata refresh or M9 AI-assisted pipeline intelligence.

---

## Variables

Set the following variables before running deployment commands.

```bash
export PROJECT_ID="your-gcp-project-id"
export REGION="europe-north1"
export BQ_LOCATION="EU"

export ARTIFACT_REPO="olist-dbt-jobs"
export IMAGE_NAME="olist-dbt-job"
export IMAGE_TAG="m7"
export IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPO}/${IMAGE_NAME}:${IMAGE_TAG}"

export CLOUD_RUN_JOB_NAME="olist-dbt-build-job"
export CLOUD_RUN_SA="olist-dbt-runner@${PROJECT_ID}.iam.gserviceaccount.com"

export SCHEDULER_JOB_NAME="olist-dbt-daily-trigger"
export SCHEDULER_SA="olist-scheduler-invoker@${PROJECT_ID}.iam.gserviceaccount.com"
export SCHEDULER_TIME_ZONE="Europe/Helsinki"
export SCHEDULER_CRON="0 6 * * *"
```

Set the active Google Cloud project:

```bash
gcloud config set project "${PROJECT_ID}"
```

Notes:

- `REGION` is the Cloud Run and Artifact Registry region.
- `BQ_LOCATION` should match the BigQuery dataset location.
- `DBT_DATASET` should be set to `olist`, because dbt schema suffixes create `olist_staging`, `olist_intermediate`, and `olist_marts`.

---

## Enable required APIs

```bash
gcloud services enable \
  run.googleapis.com \
  cloudscheduler.googleapis.com \
  artifactregistry.googleapis.com \
  iam.googleapis.com \
  bigquery.googleapis.com
```

---

## Create service accounts

Create a runtime service account for the Cloud Run Job.

```bash
gcloud iam service-accounts create olist-dbt-runner \
  --display-name="Olist dbt Cloud Run Job Runner"
```

Create a service account for Cloud Scheduler to trigger the Cloud Run Job.

```bash
gcloud iam service-accounts create olist-scheduler-invoker \
  --display-name="Olist Cloud Scheduler Invoker"
```

---

## Grant BigQuery permissions to the Cloud Run Job service account

The Cloud Run Job service account needs permission to run BigQuery jobs and write dbt-managed models.

Simple project-level setup for the portfolio project:

```bash
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${CLOUD_RUN_SA}" \
  --role="roles/bigquery.jobUser"

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${CLOUD_RUN_SA}" \
  --role="roles/bigquery.dataEditor"
```

In a stricter production setup, `roles/bigquery.dataEditor` should be limited to the required datasets instead of the whole project.

---

## Create Artifact Registry repository

```bash
gcloud artifacts repositories create "${ARTIFACT_REPO}" \
  --repository-format=docker \
  --location="${REGION}" \
  --description="Docker images for Olist dbt Cloud Run Jobs"
```

Configure Docker authentication for Artifact Registry.

```bash
gcloud auth configure-docker "${REGION}-docker.pkg.dev"
```

---

## Build Docker image locally

Run this command from the repository root.

```bash
docker build -f dbt/Dockerfile -t "${IMAGE_URI}" .
```

The final `.` is important because the Docker build context must be the repository root.

The Dockerfile copies the dbt project with:

```dockerfile
COPY dbt/ /app/dbt/
```

Do not run the build command from inside the `dbt/` directory.

---

## Push Docker image

```bash
docker push "${IMAGE_URI}"
```

---

## Create Cloud Run Job

```bash
gcloud run jobs create "${CLOUD_RUN_JOB_NAME}" \
  --image="${IMAGE_URI}" \
  --region="${REGION}" \
  --service-account="${CLOUD_RUN_SA}" \
  --set-env-vars="DBT_PROJECT_ID=${PROJECT_ID},DBT_DATASET=olist,DBT_LOCATION=${BQ_LOCATION},DBT_THREADS=4,DBT_TARGET=prod" \
  --tasks=1 \
  --max-retries=1 \
  --task-timeout=3600 \
  --memory=1Gi \
  --cpu=1
```

If the job already exists, update it instead:

```bash
gcloud run jobs update "${CLOUD_RUN_JOB_NAME}" \
  --image="${IMAGE_URI}" \
  --region="${REGION}" \
  --service-account="${CLOUD_RUN_SA}" \
  --set-env-vars="DBT_PROJECT_ID=${PROJECT_ID},DBT_DATASET=olist,DBT_LOCATION=${BQ_LOCATION},DBT_THREADS=4,DBT_TARGET=prod" \
  --tasks=1 \
  --max-retries=1 \
  --task-timeout=3600 \
  --memory=1Gi \
  --cpu=1
```

---

## Grant Scheduler permission to run the Cloud Run Job

For this project, Cloud Scheduler triggers the Cloud Run Job through the Cloud Run Admin API.

```bash
gcloud run jobs add-iam-policy-binding "${CLOUD_RUN_JOB_NAME}" \
  --region="${REGION}" \
  --member="serviceAccount:${SCHEDULER_SA}" \
  --role="roles/run.invoker"
```

If job-level IAM binding is not available in the local gcloud version, use a project-level binding instead:

```bash
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SCHEDULER_SA}" \
  --role="roles/run.invoker"
```

---

## Manually execute Cloud Run Job

```bash
gcloud run jobs execute "${CLOUD_RUN_JOB_NAME}" \
  --region="${REGION}" \
  --wait
```

Expected result:

```text
Cloud Run Job execution succeeds
dbt debug succeeds
dbt build succeeds
BigQuery staging, intermediate, and marts models are refreshed
```

---

## View Cloud Run Job executions

```bash
gcloud run jobs executions list \
  --job="${CLOUD_RUN_JOB_NAME}" \
  --region="${REGION}"
```

---

## View logs

```bash
gcloud logging read \
  "resource.type=cloud_run_job AND resource.labels.job_name=${CLOUD_RUN_JOB_NAME}" \
  --limit=50 \
  --format="value(textPayload)"
```

---

## Create Cloud Scheduler trigger

Cloud Scheduler triggers the Cloud Run Job by sending an authenticated HTTP POST request to the Cloud Run Admin API.

Because the target is a Google API endpoint under `googleapis.com`, the Scheduler job should use an OAuth token.

```bash
gcloud scheduler jobs create http "${SCHEDULER_JOB_NAME}" \
  --location="${REGION}" \
  --schedule="${SCHEDULER_CRON}" \
  --time-zone="${SCHEDULER_TIME_ZONE}" \
  --uri="https://run.googleapis.com/v2/projects/${PROJECT_ID}/locations/${REGION}/jobs/${CLOUD_RUN_JOB_NAME}:run" \
  --http-method=POST \
  --oauth-service-account-email="${SCHEDULER_SA}" \
  --oauth-token-scope="https://www.googleapis.com/auth/cloud-platform"
```

---

## Force-run Cloud Scheduler job

```bash
gcloud scheduler jobs run "${SCHEDULER_JOB_NAME}" \
  --location="${REGION}"
```

Expected result:

```text
Cloud Scheduler successfully triggers the Cloud Run Job
Cloud Run Job starts a new execution
dbt debug and dbt build run inside the container
BigQuery models are refreshed
```

---

## Validation checklist

Manual Cloud Run Job validation:

```text
[ ] Docker image builds successfully
[ ] Docker image is pushed to Artifact Registry
[ ] Cloud Run Job is created or updated
[ ] Manual Cloud Run Job execution succeeds
[ ] dbt debug succeeds
[ ] dbt build succeeds
[ ] BigQuery models refresh successfully
[ ] Cloud Logging shows dbt output
```

Cloud Scheduler validation:

```text
[ ] Cloud Scheduler job is created
[ ] Scheduler job uses authenticated OAuth trigger
[ ] Scheduler force-run succeeds
[ ] Cloud Run Job execution is created by Scheduler
[ ] Scheduled execution logs are visible
```

---

## Screenshot evidence

Recommended screenshot directory:

```text
assets/screenshots/m7_orchestration/
```

Recommended screenshots:

```text
cloud_run_job_overview.png
cloud_run_job_execution_success.png
cloud_scheduler_trigger.png
cloud_scheduler_success.png
cloud_logging_dbt_success.png
artifact_registry_image.png
```

---

## M7 boundary

This document only covers orchestration deployment commands.

M7 does not implement:

- dbt artifact parsing
- metadata refresh tables
- `olist_monitoring` dataset
- historical run comparison
- AI-assisted pipeline intelligence
- React or custom portal integration

Those capabilities are reserved for later milestones.


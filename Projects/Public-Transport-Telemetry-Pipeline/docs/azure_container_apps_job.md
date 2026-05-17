# Azure Container Apps Job Refresh Path

This document records the Azure Container Apps Job deployment path for the Public Transport Telemetry Pipeline.

The goal is to run the existing Spark-style Bronze → Silver → Gold refresh workflow as a short-lived containerized batch job, then upload dashboard-ready parquet outputs to Azure Blob Storage.

The Streamlit dashboard remains decoupled from pipeline execution. It only reads the latest exported Gold-layer outputs from Azure Blob Storage.

---

## Current Execution Model

The current routine refresh path is:

```text
Azure Container Registry
  → Azure Container Apps Scheduled Job
  → Containerized Spark pipeline
  → Gold parquet export
  → Azure Blob Storage
  → Streamlit dashboard on Render
```

Current Azure Container Apps Jobs:

```text
telemetry-refresh-job-scheduled
  Primary scheduled refresh job
  Trigger: Schedule
  Cron: 0 */3 * * *

telemetry-refresh-job
  Manual validation / fallback job
```

Azure Databricks was used earlier as an optional managed Spark validation path. It is not used as the routine scheduler.

---

## Why Azure Container Apps Jobs

The refresh workload is batch-oriented:

1. Start container
2. Run the full pipeline
3. Export Gold outputs
4. Upload outputs to Blob Storage
5. Exit

This does not require always-on compute.

Azure Container Apps Jobs provide a lighter execution path for this portfolio-scale deployment than keeping managed Spark infrastructure active for routine refreshes.

---

## Local Docker Validation

Build the Docker image:

```bash
docker build -t telemetry-pipeline-job .
```

Key points:

- `telemetry-pipeline-job` is the local image name.
- The `Dockerfile` defines the runtime environment and the container entrypoint.
- `.dockerignore` keeps the Docker build context small and avoids copying unnecessary local files into the image build process.

Run the container locally:

```bash
docker run --rm \
  -e AZURE_STORAGE_CONNECTION_STRING="$AZURE_STORAGE_CONNECTION_STRING" \
  -e AZURE_BLOB_CONTAINER="$AZURE_BLOB_CONTAINER" \
  telemetry-pipeline-job
```

Key parameters:

- `--rm`  
  Removes the container after execution. This keeps local testing clean.

- `-e AZURE_STORAGE_CONNECTION_STRING=...`  
  Passes the Azure Storage connection string into the container at runtime.

- `-e AZURE_BLOB_CONTAINER=...`  
  Tells the upload script which Blob container to use.

- `telemetry-pipeline-job`  
  Runs the locally built image.

Expected success signal:

```text
[container-refresh] Refresh job completed successfully
```

The container entrypoint runs:

```text
python scripts/run_pipeline.py --layer full
python scripts/export_gold.py
python scripts/upload_outputs_to_blob.py
```

---

## Required Environment Variables

For local Docker and Azure Container Apps Jobs:

```text
AZURE_STORAGE_CONNECTION_STRING
AZURE_BLOB_CONTAINER
```

Current Blob container:

```text
telemetry-demo
```

Do not commit connection strings, ACR passwords, `.env` files, or local settings files.

---

## Azure CLI Setup

Login:

```bash
az login
az account show -o table
```

Set variables:

```bash
export RG="azure_resource_group_1_tuo_yang"
export LOCATION="eastus"
export ACR_NAME="telemetryacr263"
export ACA_ENV="telemetry-aca-env"
export JOB_NAME="telemetry-refresh-job"
export SCHEDULED_JOB_NAME="telemetry-refresh-job-scheduled"
export IMAGE_NAME="telemetry-pipeline-job"
export IMAGE_TAG="local-v1"
export CONTAINER_NAME="telemetry-demo"
export ACR_LOGIN_SERVER="telemetryacr263.azurecr.io"
export SUB_ID=$(az account show --query id -o tsv)
```

Key variables:

- `RG`  
  Azure resource group where the project resources are created.

- `LOCATION`  
  Azure region used for the resources.

- `ACR_NAME`  
  Azure Container Registry name. ACR names must be globally unique.

- `ACA_ENV`  
  Azure Container Apps Environment. Container Apps Jobs run inside this environment.

- `JOB_NAME`  
  Manual validation / fallback job.

- `SCHEDULED_JOB_NAME`  
  Primary scheduled refresh job.

- `IMAGE_NAME` and `IMAGE_TAG`  
  Container image name and version tag.

- `CONTAINER_NAME`  
  Azure Blob container used by the dashboard serving layer.

Check variables:

```bash
echo $RG
echo $ACA_ENV
echo $SCHEDULED_JOB_NAME
echo $ACR_LOGIN_SERVER
echo $CONTAINER_NAME
```

---

## Azure Container Registry

Azure Container Registry stores the pipeline refresh image used by Azure Container Apps Jobs.

Check ACR name availability:

```bash
az acr check-name --name "$ACR_NAME"
```

Create ACR:

```bash
az acr create \
  --resource-group "$RG" \
  --name "$ACR_NAME" \
  --sku Basic \
  --location "$LOCATION"
```

Key parameters:

- `--resource-group`  
  Places the registry in the project resource group.

- `--name`  
  Sets the globally unique registry name.

- `--sku Basic`  
  Uses the lowest-cost ACR tier suitable for this portfolio deployment.

- `--location`  
  Creates the registry in the selected Azure region.

Login to ACR:

```bash
az acr login --name "$ACR_NAME"
```

Get login server:

```bash
export ACR_LOGIN_SERVER=$(az acr show \
  --name "$ACR_NAME" \
  --resource-group "$RG" \
  --query loginServer \
  -o tsv)

echo $ACR_LOGIN_SERVER
```

Tag and push image:

```bash
docker tag telemetry-pipeline-job:latest "$ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG"

docker push "$ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG"
```

Key points:

- `docker tag` gives the local image a registry-qualified name.
- `docker push` uploads the image to Azure Container Registry.
- The image is later pulled by Azure Container Apps Jobs.

Verify repository:

```bash
az acr repository list \
  --name "$ACR_NAME" \
  -o table
```

---

## Azure Container Apps Environment

Create the Container Apps Environment:

```bash
az containerapp env create \
  --name "$ACA_ENV" \
  --resource-group "$RG" \
  --location "$LOCATION"
```

Key parameters:

- `--name`  
  Name of the Container Apps Environment.

- `--resource-group`  
  Resource group where the environment is created.

- `--location`  
  Azure region for the environment.

Purpose:

```text
Azure Container Apps Environment
  = managed runtime boundary for Container Apps and Container Apps Jobs
```

---

## ACR Pull Credentials

The Container Apps Job needs credentials to pull the private image from Azure Container Registry.

For this portfolio deployment, ACR admin credentials were enabled:

```bash
az acr update \
  --name "$ACR_NAME" \
  --resource-group "$RG" \
  --admin-enabled true
```

Retrieve credentials into local shell variables:

```bash
export ACR_USERNAME=$(az acr credential show \
  --name "$ACR_NAME" \
  --query username \
  -o tsv)

export ACR_PASSWORD=$(az acr credential show \
  --name "$ACR_NAME" \
  --query "passwords[0].value" \
  -o tsv)
```

Key points:

- `ACR_USERNAME` and `ACR_PASSWORD` allow the Container Apps Job to pull the image from private ACR.
- Do not print or commit the password.
- In a production environment, Managed Identity would be preferred over ACR admin credentials.

---

## Manual Validation Job

The manual job is kept as a validation and fallback path.

Create the manual job:

```bash
az containerapp job create \
  --name "$JOB_NAME" \
  --resource-group "$RG" \
  --environment "$ACA_ENV" \
  --trigger-type Manual \
  --replica-timeout 1800 \
  --replica-retry-limit 1 \
  --replica-completion-count 1 \
  --parallelism 1 \
  --image "$ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG" \
  --cpu 2 \
  --memory 4Gi \
  --registry-server "$ACR_LOGIN_SERVER" \
  --registry-username "$ACR_USERNAME" \
  --registry-password "$ACR_PASSWORD" \
  --secrets azure-storage-connection-string="$AZURE_STORAGE_CONNECTION_STRING" \
  --env-vars \
      AZURE_STORAGE_CONNECTION_STRING=secretref:azure-storage-connection-string \
      AZURE_BLOB_CONTAINER="$CONTAINER_NAME"
```

Key parameters:

- `--trigger-type Manual`  
  Creates a job that only runs when explicitly started.

- `--replica-timeout 1800`  
  Allows each execution to run for up to 30 minutes before timing out.

- `--replica-retry-limit 1`  
  Allows one retry if the job execution fails.

- `--replica-completion-count 1`  
  Marks the job as successful after one replica completes successfully.

- `--parallelism 1`  
  Ensures only one replica runs at a time. This avoids concurrent writes to the same Blob output paths.

- `--image`  
  Points to the ACR image used by the job.

- `--cpu 2 --memory 4Gi`  
  Allocates enough resources for the Spark-based batch refresh in the container.

- `--registry-server`, `--registry-username`, `--registry-password`  
  Allow the job to pull the private image from ACR.

- `--secrets azure-storage-connection-string=...`  
  Stores the Blob connection string as a Container Apps secret.

- `AZURE_STORAGE_CONNECTION_STRING=secretref:azure-storage-connection-string`  
  Injects the secret into the container environment without hardcoding it in the image or repository.

- `AZURE_BLOB_CONTAINER="$CONTAINER_NAME"`  
  Provides the target Blob container name.

Start the manual job:

```bash
az containerapp job start \
  --name "$JOB_NAME" \
  --resource-group "$RG"
```

---

## Scheduled Refresh Job

The scheduled job is the primary routine refresh path.

Create the scheduled job:

```bash
az containerapp job create \
  --name "$SCHEDULED_JOB_NAME" \
  --resource-group "$RG" \
  --environment "$ACA_ENV" \
  --trigger-type Schedule \
  --cron-expression "0 */3 * * *" \
  --replica-timeout 1800 \
  --replica-retry-limit 1 \
  --replica-completion-count 1 \
  --parallelism 1 \
  --image "$ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG" \
  --cpu 2 \
  --memory 4Gi \
  --registry-server "$ACR_LOGIN_SERVER" \
  --registry-username "$ACR_USERNAME" \
  --registry-password "$ACR_PASSWORD" \
  --secrets azure-storage-connection-string="$AZURE_STORAGE_CONNECTION_STRING" \
  --env-vars \
      AZURE_STORAGE_CONNECTION_STRING=secretref:azure-storage-connection-string \
      AZURE_BLOB_CONTAINER="$CONTAINER_NAME"
```

Key parameters:

- `--trigger-type Schedule`  
  Creates a scheduled job instead of a manually triggered job.

- `--cron-expression "0 */3 * * *"`  
  Runs the job every three hours in UTC.

- `--replica-timeout 1800`  
  Allows each execution to run for up to 30 minutes.

- `--replica-retry-limit 1`  
  Allows one retry for failed executions.

- `--replica-completion-count 1`  
  The job succeeds when one replica completes successfully.

- `--parallelism 1`  
  Prevents multiple replicas from writing to the same Blob output paths at the same time.

- `--image "$ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG"`  
  Uses the pipeline refresh image pushed to ACR.

- `--cpu 2 --memory 4Gi`  
  Resource allocation for the Spark-based container refresh.

- `--secrets` and `secretref:`  
  Keep the Blob connection string outside the image, code, and plain environment variable configuration.

Current schedule:

```text
0 */3 * * *
```

This runs every three hours in UTC:

```text
00:00, 03:00, 06:00, 09:00, 12:00, 15:00, 18:00, 21:00 UTC
```

This frequency is used during validation. It can be reduced later for stricter cost control.

---

## Validate Job Configuration

Because the `az containerapp` extension may call an unsupported API version, REST validation is used with a working API version:

```bash
az rest \
  --method get \
  --url "https://management.azure.com/subscriptions/$SUB_ID/resourceGroups/$RG/providers/Microsoft.App/jobs/$SCHEDULED_JOB_NAME?api-version=2025-02-02-preview" \
  --query "{name:name, trigger:properties.configuration.triggerType, cron:properties.configuration.scheduleTriggerConfig.cronExpression, image:properties.template.containers[0].image, provisioningState:properties.provisioningState}"
```

Key parameters:

- `az rest`  
  Calls the Azure Resource Manager API directly.

- `api-version=2025-02-02-preview`  
  Uses a supported API version for `Microsoft.App/jobs` in this environment.

- `--query`  
  Extracts the most important job configuration fields from the full JSON response.

Expected result:

```json
{
  "cron": "0 */3 * * *",
  "image": "telemetryacr263.azurecr.io/telemetry-pipeline-job:local-v1",
  "name": "telemetry-refresh-job-scheduled",
  "provisioningState": "Succeeded",
  "trigger": "Schedule"
}
```

---

## Start Scheduled Job Manually

A scheduled job can also be triggered manually for validation:

```bash
az rest \
  --method post \
  --url "https://management.azure.com/subscriptions/$SUB_ID/resourceGroups/$RG/providers/Microsoft.App/jobs/$SCHEDULED_JOB_NAME/start?api-version=2025-02-02-preview"
```

Key point:

- This does not change the schedule.
- It only starts an extra execution for validation.

Check executions:

```bash
az rest \
  --method get \
  --url "https://management.azure.com/subscriptions/$SUB_ID/resourceGroups/$RG/providers/Microsoft.App/jobs/$SCHEDULED_JOB_NAME/executions?api-version=2025-02-02-preview" \
  --query "value[].{name:name,status:properties.status,startTime:properties.startTime}"
```

Expected status:

```text
Succeeded
```

---

## View Logs

Get logs for the latest execution:

```bash
az containerapp job logs show \
  --name "$SCHEDULED_JOB_NAME" \
  --resource-group "$RG" \
  --container "$SCHEDULED_JOB_NAME" \
  --follow
```

Key parameters:

- `--name`  
  Job name.

- `--resource-group`  
  Resource group containing the job.

- `--container`  
  Container name inside the job template.

- `--follow`  
  Streams logs while the execution is running or while recent logs are available.

If the latest execution has no available replica logs, specify a known execution name:

```bash
az containerapp job logs show \
  --name "$SCHEDULED_JOB_NAME" \
  --resource-group "$RG" \
  --execution "<execution-name>" \
  --container "$SCHEDULED_JOB_NAME" \
  --follow
```

Expected success signal:

```text
[container-refresh] Refresh job completed successfully
```

---

## Validate Blob Outputs

Check that dashboard-ready files were updated in Azure Blob Storage.

Portal path:

```text
Storage Account
  → container: telemetry-demo
  → telemetry/
```

Important output paths include:

```text
telemetry/hsl/hsl_route_paths_overview.parquet
telemetry/hsl/hsl_route_options.parquet
telemetry/weather/weather_stations_latest.parquet
telemetry/output/
```

CLI example:

```bash
az storage blob list \
  --container-name "$CONTAINER_NAME" \
  --connection-string "$AZURE_STORAGE_CONNECTION_STRING" \
  --prefix "telemetry/" \
  --query "sort_by([].{name:name,lastModified:properties.lastModified,size:properties.contentLength}, &lastModified)[-10:]" \
  -o table
```

Key parameters:

- `--container-name`  
  Target Blob container.

- `--connection-string`  
  Authenticates the CLI call to Azure Storage.

- `--prefix "telemetry/"`  
  Limits results to dashboard output files.

- `sort_by(..., &lastModified)[-10:]`  
  Shows the latest updated files.

---

## Azure CLI / API Version Issue

The `az containerapp job show` and `az containerapp job execution list` commands may fail if the local `containerapp` extension calls an unsupported API version.

Example error:

```text
InvalidApiVersionParameter
api-version is invalid
```

Workaround:

Use `az rest` with an API version supported by the current region and resource provider.

Working version used in this project:

```text
2025-02-02-preview
```

Example:

```bash
az rest \
  --method get \
  --url "https://management.azure.com/subscriptions/$SUB_ID/resourceGroups/$RG/providers/Microsoft.App/jobs/$SCHEDULED_JOB_NAME?api-version=2025-02-02-preview"
```

If this fails in the future, check the supported API versions returned by Azure and update the version accordingly.

---

## Cost-Control Notes

The first cloud execution path used Azure Databricks Jobs as an optional managed Spark validation path.

After validation, routine refresh was moved to Azure Container Apps Jobs because the workload is batch-oriented and does not require persistent Spark infrastructure.

Observed cost issue:

```text
Databricks managed resource group
  → NAT Gateway
  → fixed networking cost even after compute was paused
```

Cost-control actions:

```text
1. Pause Databricks job schedule
2. Confirm no active all-purpose compute
3. Confirm no active SQL warehouse
4. Move routine refresh to Azure Container Apps Jobs
5. Delete Databricks workspace after validation
6. Confirm managed resource group and NAT Gateway are removed
```

The Azure Container Apps Job remains the routine refresh path.

---

## Security Notes

Do not commit:

```text
AZURE_STORAGE_CONNECTION_STRING
ACR_PASSWORD
.env
local.settings.json
```

Current implementation uses:

```text
Container Apps secret reference for Blob connection string
ACR credentials for private image pull
```

Recommended production upgrades:

```text
Managed Identity
Azure Key Vault
ACR pull via identity
stricter secret rotation
centralized monitoring and alerting
```

---

## Operational Summary

The final refresh setup is:

```text
telemetry-refresh-job-scheduled
  Primary scheduled refresh job
  Runs every 3 hours
  Pulls image from Azure Container Registry
  Runs containerized Spark-style refresh pipeline
  Uploads Gold outputs to Azure Blob Storage

telemetry-refresh-job
  Manual validation / fallback job

Azure Databricks
  Used for managed Spark validation
  Not used as routine scheduler
  Removed after validation for cost control
```

This keeps the portfolio deployment reproducible, cloud-based, and cost-aware without relying on always-on infrastructure.
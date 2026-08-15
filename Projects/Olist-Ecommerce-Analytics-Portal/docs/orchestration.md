# M7 Orchestration Design

## Current note

This document records the M7 design and validation as it was implemented.

The current repository also contains M8 monitoring, M9 pipeline review, and M10 window control. The existing Cloud Scheduler / Cloud Run Job still starts `run_dbt_job.sh`; it has not yet been changed to start the M10 controller.

When `run_dbt_job.sh` receives no control window, it runs in full-history compatibility mode. M10 governed runs call the same script through `run_window_controller.py` and supply the window and attempt ID.

---

## Purpose

M7 adds a lightweight cloud orchestration layer to the Olist E-Commerce Analytics & Pipeline Monitoring Portal project.

The goal is to make the existing dbt transformation pipeline executable in Google Cloud, instead of relying only on local manual execution. The orchestration layer should be simple, reproducible, and aligned with the current project scope.

The selected design uses:

```text
Cloud Scheduler
    ↓
Cloud Run Job
    ↓
Containerized dbt project
    ↓
BigQuery staging and marts datasets
```

This approach keeps the project cloud-native while avoiding unnecessary orchestration complexity.

---

## M7 Scope

M7 focuses only on Google Cloud Scheduler and Cloud Run Job orchestration.

The milestone includes:

- designing the orchestration architecture;
- packaging the dbt pipeline as a containerized Cloud Run Job;
- defining how Cloud Scheduler triggers the Cloud Run Job;
- documenting service account and IAM requirements;
- documenting manual and scheduled execution validation;
- updating project documentation after validation.

The expected pipeline execution command for M7 is:

```bash
dbt build --target prod
```

This allows dbt models and tests to run together as one scheduled transformation job.

---

## Out of Scope

The following items are intentionally excluded from M7:

- dbt artifact parsing;
- `manifest.json` processing;
- `run_results.json` processing;
- `catalog.json` processing;
- metadata refresh tables;
- `olist_monitoring` dataset creation;
- pipeline health dashboards;
- pipeline quality review;
- React portal implementation;
- Power BI report expansion;
- Airflow, Cloud Composer, Dagster, or Prefect orchestration.

These items belong to later milestones, especially M8 and M9.

---

## Current Pipeline Context

Before M7, the project already contains:

- raw Olist source tables in BigQuery;
- dbt staging models;
- dbt marts models;
- dimensional modeling layer;
- dbt tests;
- dbt documentation;
- README and architecture documentation;
- GitHub Project milestone tracking.

The current pipeline can be executed manually from the local dbt project.

M7 turns this into a cloud-executable job.

---

## Target Architecture

The M7 target architecture is:

```text
+-------------------+
| Cloud Scheduler   |
| cron trigger      |
+---------+---------+
          |
          v
+-------------------+
| Cloud Run Job     |
| dbt batch runner  |
+---------+---------+
          |
          v
+-------------------+
| Container Image   |
| dbt project       |
| dbt-bigquery      |
| run script        |
+---------+---------+
          |
          v
+-------------------+
| BigQuery          |
| olist_raw         |
| olist_staging     |
| olist_marts       |
+-------------------+
```

Cloud Scheduler is responsible for triggering the job on a schedule.

Cloud Run Job is responsible for running the dbt pipeline once and exiting.

---

## Execution Flow

The expected M7 execution flow is:

1. Cloud Scheduler reaches the configured schedule.
2. Cloud Scheduler triggers the Cloud Run Job.
3. Cloud Run Job starts a container from the dbt job image.
4. The container runs the dbt entrypoint script.
5. The script validates the dbt connection.
6. The script executes the dbt pipeline.
7. dbt builds staging and marts models in BigQuery.
8. dbt tests are executed as part of `dbt build`.
9. Cloud Run Job exits with success or failure.
10. Cloud Logging stores the execution logs.

The manual execution flow should also be supported for validation and debugging.

---

## Cloud Run Job Responsibilities

Cloud Run Job is used as the batch execution environment for the dbt pipeline.

Its responsibilities are:

- run the dbt project from a container image;
- use a dedicated service account;
- connect to BigQuery through Google Cloud authentication;
- execute dbt commands;
- write execution logs to Cloud Logging;
- exit successfully only when the dbt pipeline succeeds.

The Cloud Run Job should not contain business logic outside the dbt pipeline.

The job should be treated as an execution wrapper around the existing dbt project.

---

## Cloud Scheduler Responsibilities

Cloud Scheduler is used as the time-based trigger for the dbt pipeline.

Its responsibilities are:

- define the pipeline schedule;
- trigger the Cloud Run Job;
- use an authenticated service account;
- provide a simple operational entry point for scheduled execution.

The initial schedule can be conservative, such as a daily run.

The schedule can be adjusted later if the project needs more frequent refreshes.

---

## Containerization Strategy

The dbt project will be packaged into a container image.

The image should include:

- Python runtime;
- dbt Core;
- dbt BigQuery adapter;
- the dbt project files;
- a job entrypoint script.

The container should run a command similar to:

```bash
dbt debug --target prod
dbt build --target prod
```

This keeps the runtime environment consistent between local development and cloud execution.

No credentials should be committed into the repository.

---

## Service Account and IAM Design

M7 should use a dedicated service account for the Cloud Run Job.

The Cloud Run Job service account needs enough permission to:

- read required BigQuery source tables;
- create or replace dbt-managed models in the target datasets;
- write job execution metadata required by BigQuery;
- write logs through the managed Cloud Run execution environment.

Cloud Scheduler should use an authenticated identity to trigger the Cloud Run Job.

The exact IAM bindings will be documented in the deployment command documentation.

The principle is to keep permissions specific to the orchestration use case and avoid using personal user credentials in runtime execution.

---

## Runtime Configuration

Runtime configuration should be handled through environment variables or Google Cloud-managed identity, not through committed credential files.

The dbt profile should be configured through non-secret runtime configuration, so that cloud execution can authenticate through the Cloud Run Job service account.

Local development can continue using the existing local `profiles.yml`, which remains outside the repository.

The repository should not contain:

- service account key files;
- personal credentials;
- local dbt profiles containing secrets;
- raw source data files.

---

## Validation Plan

M7 validation should cover both manual and scheduled execution.

Manual validation:

- build and push the dbt container image;
- create or update the Cloud Run Job;
- manually execute the Cloud Run Job;
- confirm that the job completes successfully;
- confirm that dbt models and tests succeed;
- check Cloud Logging for dbt output.

Scheduled validation:

- create the Cloud Scheduler trigger;
- manually force-run the Scheduler job if needed;
- confirm that Scheduler triggers the Cloud Run Job;
- confirm successful Cloud Run Job execution;
- confirm BigQuery models are refreshed;
- save screenshots as project evidence.

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
```

---

## Documentation Deliverables

M7 should update or add the following documentation:

```text
docs/orchestration.md
docs/architecture.md
docs/gcp_orchestration_commands.md
README.md
```

The README should only be updated after the orchestration flow has been validated.

---

## Later milestones

At M7 completion, monitoring and review were still future work. They were added later as separate milestones.

M8 added parsing for:

- `manifest.json`
- `run_results.json`
- `catalog.json`

and loaded the results into BigQuery monitoring tables.

M9 later added the pipeline quality reviewer.

These features remain outside the M7 milestone itself.







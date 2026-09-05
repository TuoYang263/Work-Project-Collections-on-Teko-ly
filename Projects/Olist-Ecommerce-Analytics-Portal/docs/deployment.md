# Deployment and CI/CD

## Purpose

This document describes the **current deployment boundaries** for the Olist data product after the M10 production-cycle cutover.

Historical M7/M8 setup commands remain in `gcp_orchestration_commands.md`.

Those historical runbooks are retained as engineering evidence but should not be treated as the current production topology.

\---

## Current deployed topology

The system has two independently deployed runtime surfaces:

```text
Public Portal
    → Render
    → Next.js
    → read-only BigQuery access

Scheduled data pipeline
    → Cloud Scheduler
    → Cloud Run Job
    → M10 Window Controller
    → dbt \+ monitoring \+ reliability review
    → BigQuery
```

Portal and pipeline delivery are intentionally separated.

\---

## Public Portal

Live URL:

```text
https://olist-analytics-portal.onrender.com
```

Health endpoint:

```text
https://olist-analytics-portal.onrender.com/health
```

The Portal is deployed as a Render Web Service.

Application runtime:

```text
Node.js
Next.js
server-rendered BigQuery access
```

The public demo uses Render's free tier. A cold first request after inactivity can therefore be slower while the service wakes.

### Portal build boundary

Portal project:

```text
Projects/Olist-Ecommerce-Analytics-Portal/portal
```

Build:

```bash
npm ci && npm run build
```

Start:

```bash
npm start
```

The Portal uses server-side credentials. BigQuery credentials are not exposed to browser JavaScript.

The public deployment is read-only from the Portal perspective. Application-level organization authentication is outside the M10 boundary.

\---

## Scheduled GCP runtime

The deployed scheduled pipeline path is:

```text
Cloud Scheduler
        ↓
Cloud Run Job
        ↓
Docker image
        ↓
python /app/dbt/control/run_window_controller.py
        ↓
calendar-month control window
        ↓
run_dbt_job.sh
        ↓
dbt build
        ↓
M8 monitoring persistence
        ↓
exact monitoring-run resolution
        ↓
M9 deterministic review
        ↓
controller success / failure transition
        ↓
BigQuery
```

Current cloud boundaries:

```text
Cloud Run Job region:       europe-north1
Cloud Scheduler location:   europe-west1
Scheduler timezone:         Europe/Helsinki
Scheduler cadence:          0 * * * *
BigQuery location:          EU
Cloud Run max retries:      0
```

The production Scheduler invokes the Cloud Run Job once per hour.

One successful invocation processes approximately one historical calendar month.

\---

## Production controller command

The deployed Cloud Run Job overrides the container default command.

Command:

```text
python
```

Controller path:

```text
/app/dbt/control/run_window_controller.py
```

Production arguments configure:

```text
project      \= balmy-nuance-468118-g4
control data \= olist_control
pipeline     \= olist-dbt-build-job
environment  \= prod
dbt dataset  \= olist
location     \= EU
source start \= 2016-09-01T00:00:00+00:00
source end   \= 2018-11-01T00:00:00+00:00
```

The source range contains 26 calendar-month windows. At the current hourly cadence, one complete successful production cycle takes approximately 26 scheduled executions.

\---

## Why Cloud Run platform retries are disabled

The Cloud Run Job is deployed with:

```text
maxRetries \= 0
```

This is intentional.

M10 owns retry state inside BigQuery so that retries preserve:

```text
same failed window
new attempt_id
attempt_number \+ 1
retry_of_attempt_id
explicit audit events
```

The controller therefore remains the source of truth for retry behavior.

\---

## `run_dbt_job.sh` compatibility boundary

`run_dbt_job.sh` still supports execution without control variables.

In that mode it prints:

```text
No control window supplied.
Running in full-history compatibility mode.
```

That behavior is retained for compatibility and manual use.

It is **not** the normal scheduled production entry point after the M10 cutover.

\---

## Runtime identities

### Scheduler

The Scheduler uses:

```text
olist-scheduler-invoker
```

Its responsibility is to invoke the Cloud Run Job.

### Cloud Run runtime

The Cloud Run Job uses:

```text
olist-dbt-runner
```

This is the workload identity used by the running container when it accesses BigQuery and related GCP resources.

### GitHub deployment

Pipeline deployment uses:

```text
olist-github-deployer
```

The deployment identity is deliberately separate from the runtime data-processing identity.

\---

## GitHub Actions boundaries

The monorepo separates Portal CI and pipeline CI/CD.

### Portal CI

Validation includes:

```text
npm ci
npm test
npm run lint
npm run build
```

Current Portal regression inventory:

```text
5 Vitest files
21 tests
```

### Pipeline CI/CD

Validation includes:

```text
Python 3.11
Window Controller unit tests
Monitoring-run resolver unit tests
Pipeline reviewer unit tests
run_dbt_job.sh shell syntax
Docker image build
Docker runtime smoke checks
```

Current Python inventory:

```text
Window Controller:       52 tests
Monitoring resolver:      5 tests
Pipeline reviewer:       59 tests
                         \--------
Total:                  116 tests
```

\---

## GCP deployment guard

The deploy job runs only when:

```text
event \= push
ref   \= refs/heads/main
```

Feature branches can run validation.

The Workload Identity Provider also restricts accepted GitHub claims to the expected repository and `refs/heads/main`.

\---

## OIDC and Workload Identity Federation

GitHub does not store a long-lived GCP service-account JSON key for pipeline deployment.

Authentication path:

```text
GitHub Actions
        ↓
GitHub OIDC token
        ↓
Google Workload Identity Provider
        ↓
Google STS
        ↓
short-lived Google credentials
        ↓
impersonate olist-github-deployer
```

\---

## Least-privilege deployment permissions

```text
Artifact Registry repository
    → writer

Cloud Run Job
    → developer

Cloud Run runtime service account
    → serviceAccountUser
```

This allows the deployer to push a new image, update the existing Cloud Run Job, and retain the runtime service account without becoming the data-processing identity.

\---

## Pipeline deployment sequence

```text
Validate Pipeline
        ↓
Authenticate with OIDC/WIF
        ↓
configure Artifact Registry Docker auth
        ↓
build image
        ↓
tag with Git commit SHA
        ↓
push image
        ↓
update Cloud Run Job image
        ↓
set production controller command / arguments
        ↓
read deployed image
        ↓
verify deployed image \== expected image
```

CI must succeed before deployment.

\---

## Current Scheduler configuration

Scheduler resource:

```text
olist-dbt-daily-trigger
```

The historical resource name is retained even though the cadence is no longer daily.

Current schedule:

```text
0 * * * *
```

Timezone:

```text
Europe/Helsinki
```

State:

```text
ENABLED
```

Target:

```text
Cloud Run Jobs API
→ europe-north1
→ olist-dbt-build-job:run
```

\---

## Production validation

The production Scheduler-to-controller path has been validated end to end:

```text
Cloud Scheduler
        ↓
scheduler invoker service account
        ↓
Cloud Run Job
        ↓
M10 Window Controller
        ↓
WINDOW_STARTED
        ↓
windowed dbt workload
        ↓
M8 monitoring
        ↓
M9 deterministic review
        ↓
WINDOW_SUCCEEDED
        ↓
successful watermark advance
```

A real production execution completed:

```text
1 / 1 Cloud Run tasks successful
```

Validated first production monthly window:

```text
cycle_id \= 1
2016-09-01T00:00:00+00:00
→
2016-10-01T00:00:00+00:00
```

Final control state:

```text
state \= IDLE
cycle_id \= 1
control_version \= 2
last_successful_window_start \= 2016-09-01T00:00:00+00:00
last_successful_window_end   \= 2016-10-01T00:00:00+00:00
active attempt \= NULL
```

The corresponding audit history contained `WINDOW_STARTED` and `WINDOW_SUCCEEDED` for the same attempt ID.

\---

## Exact monitoring correlation validation

The same controller attempt was found in:

```text
olist_monitoring.pipeline_runs.control_attempt_id
```

The matching monitoring run completed with:

```text
22 / 22 models successful
96 / 96 tests passed
```

This validates the intended correlation:

```text
controller attempt
        ↓
control_attempt_id
        ↓
exact monitoring_run_id
        ↓
M9 deterministic review
```

\---

## Analytics deployment consequence

The current analytics-serving views are scoped to:

```text
last_successful_window_end
```

Serving behavior is therefore:

```text
RUNNING
→ keep showing previous successful analytical scope

FAILED
→ keep showing previous successful analytical scope

WINDOW_SUCCEEDED
→ analytical scope advances
```

The views read control state dynamically, so future successful watermark movement does not require rebuilding them for every monthly execution.

\---

## Portal and pipeline deployment are intentionally separate

```text
portal/**
   ↓
Portal CI
   ↓
Render

dbt/**
   ↓
Pipeline CI/CD
   ↓
GCP Artifact Registry
   ↓
Cloud Run Job
```

A Portal-only change should not require rebuilding the dbt runtime image. A pipeline-only change should not require Portal deployment.

\---

## Security notes

\- credentials and local profiles are ignored by Git
\- Portal queries execute server-side
\- the public Portal uses a dedicated read-only BigQuery identity
\- GitHub-to-GCP deployment uses short-lived federated credentials
\- deployment and runtime identities are separate
\- deployment permissions are scoped to specific resources
\- branch eligibility is enforced in both GitHub Actions and Workload Identity Federation
\- Cloud Run platform retries are disabled so retry ownership remains explicit in the controller

\---

## Historical runbooks

For original M7/M8 orchestration setup and command history, see:

\- [`orchestration.md`](orchestration.md)
\- [`gcp_orchestration_commands.md`](gcp_orchestration_commands.md)

Those files preserve the milestone state that was actually validated at the time. They should not be interpreted as the current production deployment reference when they describe historical image tags, schedules, or pre-controller entry points.

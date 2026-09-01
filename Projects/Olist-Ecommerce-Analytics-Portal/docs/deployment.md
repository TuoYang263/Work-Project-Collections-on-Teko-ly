# Deployment and CI/CD

## Purpose

This document describes the **current deployment boundaries** for the Olist data product.

Historical M7/M8 setup commands remain in `gcp_orchestration_commands.md`. This document describes the current topology after M10.

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

Current application runtime:

```text
Node.js
Next.js
server-rendered BigQuery access
```

The public demo uses Render's free tier. A cold first request after inactivity can be slower while the service wakes.

### Portal build boundary

The Portal project lives under:

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

The public deployment is read-only from the Portal perspective. Application-level organization authentication is not part of M10.

## Scheduled GCP runtime

The current scheduled data-pipeline path is:

```text
Cloud Scheduler
      ↓
Cloud Run Job
      ↓
Docker image
      ↓
run_dbt_job.sh
      ↓
dbt build + monitoring persistence
      ↓
BigQuery
```

Current cloud boundaries:

```text
Cloud Run Job region: europe-north1
Cloud Scheduler location: europe-west1
Scheduler timezone: Europe/Helsinki
BigQuery location: EU
```

The current scheduled entry point remains:

```text
/app/dbt/run_dbt_job.sh
```

Without controller window variables, it runs in full-history compatibility mode.

The M10 Window Controller is implemented and validated, but the Scheduler has not been switched to `run_window_controller.py`.

## Runtime identities

### Scheduler

The Scheduler uses a dedicated invoker service account.

Its responsibility is to invoke the Cloud Run Job, not to execute dbt directly.

### Cloud Run

The Cloud Run Job uses:

```text
olist-dbt-runner
```

This is the workload/runtime identity used by the container when it accesses BigQuery and related GCP resources.

### GitHub deployment

Pipeline deployment uses a separate account:

```text
olist-github-deployer
```

The deployer is not the dbt runtime identity.

## GitHub Actions boundaries

The monorepo separates Portal and pipeline workflows.

### Portal CI

Portal CI is scoped to Portal changes.

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

Pipeline CI/CD is scoped to `dbt/**` and the pipeline workflow itself.

Validation includes:

```text
Python 3.11
Window Controller unit tests
Monitoring run resolver unit tests
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
                         --------
Total:                  116 tests
```

## GCP deployment guard

Pipeline deployment has two independent branch guards.

### GitHub workflow guard

The deploy job only runs when:

```text
event = push
ref = refs/heads/main
```

Feature branches can run pipeline validation, but deployment is skipped.

### Workload Identity Provider guard

The GCP Workload Identity Provider also restricts accepted GitHub identity claims to:

```text
the expected repository
AND
refs/heads/main
```

This means a workflow condition mistake alone is not enough to authenticate a feature branch to the GCP deployment identity.

## OIDC and Workload Identity Federation

GitHub does not store a long-lived GCP service-account JSON key for pipeline deployment.

The authentication path is:

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

The identity provider validates the GitHub repository and branch claims before accepting the workload identity.

## Least-privilege deployment permissions

The deployer receives only the resource-level capabilities required for the pipeline deployment path:

```text
Artifact Registry repository
  → writer

Cloud Run Job
  → developer

Cloud Run runtime service account
  → serviceAccountUser
```

This allows the deployer to:

1. push a new container image
2. update the existing Cloud Run Job
3. keep the existing runtime service account attached

It does not make the deployer the runtime data-processing identity.

## Pipeline deployment sequence

On an eligible main-branch pipeline change:

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
read deployed image
      ↓
verify deployed image == expected image
```

CI must pass before deployment.

## Current Scheduler validation

The Scheduler-to-Cloud-Run path has been manually triggered and validated.

The observed invocation path was:

```text
Cloud Scheduler
      ↓
scheduler invoker service account
      ↓
Cloud Run Job execution
      ↓
1 / 1 task completed successfully
```

This validates the current scheduled runtime path, not a scheduled M10 Window Controller path.

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
GCP Artifact Registry / Cloud Run
```

A Portal-only change should not rebuild/deploy the dbt runtime image.

A pipeline-only change should not require Portal validation/deployment.

This split keeps deployment cost, failure domain, and feedback time smaller.

## Security notes

- credentials and local profiles are ignored by Git
- Portal queries execute server-side
- GitHub-to-GCP deployment uses short-lived federated credentials
- deployment and runtime identities are separated
- deployment roles are scoped to specific resources
- the public Portal currently has no application-level organization login
- the Render runtime uses a dedicated read-only BigQuery identity

## Historical runbooks

For the original M7/M8 setup and command history, see:

- [`orchestration.md`](orchestration.md)
- [`gcp_orchestration_commands.md`](gcp_orchestration_commands.md)

Those files preserve the milestone state that was actually validated at the time. They should not be treated as the current CI/CD reference when they mention historical image tags.

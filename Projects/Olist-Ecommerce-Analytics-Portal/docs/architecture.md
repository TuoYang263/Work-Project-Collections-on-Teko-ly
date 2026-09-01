# Architecture

## Purpose

This document describes the **current system architecture** of the Olist E-Commerce Analytics & Pipeline Monitoring Portal.

Detailed milestone reasoning remains in the milestone documents. This file is the current-state reference.

![Olist system architecture](../assets/architecture/olist-system-architecture.png)

Editable source: [`../assets/architecture/olist-system-architecture.drawio`](../assets/architecture/olist-system-architecture.drawio)

## System boundaries

The system has four primary boundaries:

1. data transformation and analytics serving
2. monitoring and deterministic reliability review
3. pipeline execution and window control
4. Portal serving and deployment

The current production Scheduler still invokes `run_dbt_job.sh` directly. The M10 Window Controller is implemented and validated but is not yet the scheduled entry point.

## 1. Data and analytics path

```text
Olist CSV files
      ↓
BigQuery raw
      ↓
dbt staging
      ↓
dbt intermediate
      ↓
dbt marts
      ↓
analytics serving
      ↓
Next.js Portal
```

### Warehouse responsibilities

| Layer | Responsibility |
|---|---|
| Raw | Keep source-aligned records |
| Staging | Cast, rename, normalize, expose stable source interfaces |
| Intermediate | Centralize reusable order-level logic and current-window order IDs |
| Marts | Produce analytics-ready dimensions and facts |
| Analytics serving | Produce state summaries and persisted diagnostics for Portal consumption |

The project uses nine Olist source files. Local raw CSV files are ignored by Git.

### Transaction grain

Dimensions remain full-history reference data.

The transactional fact path uses:

```text
stg_orders
    ↓
int_orders_windowed
    ↓
current order IDs
   ↙    ↓    ↘
items payments reviews
   ↘    ↓    ↙
incremental facts
```

The window anchor uses a half-open interval:

```text
window_start <= order_purchase_timestamp < window_end
```

This prevents overlap between adjacent windows.

The four fact models use incremental `MERGE` semantics with stable unique keys:

| Fact | Unique key |
|---|---|
| `fct_orders` | `order_id` |
| `fct_order_items` | `order_item_key` |
| `fct_order_payments` | `order_payment_key` |
| `fct_order_reviews` | `review_key` |

This supports same-window retry without appending duplicate rows for the same keys.

## 2. Monitoring and reliability path

The monitoring path is separate from the business marts.

```text
dbt build
   ↓
manifest.json + run_results.json
   ↓
dbt docs generate
   ↓
catalog.json
   ↓
artifact parser / loader
   ↓
olist_monitoring
   ↓
deterministic reviewer
   ↓
persisted evaluations and findings
```

Monitoring is append-only. Historical evidence is not overwritten by a single "latest" snapshot.

Core monitoring tables:

```text
pipeline_runs
model_run_results
test_run_results
model_metadata_snapshots
model_column_snapshots
model_lineage_edges
```

M10 adds `control_attempt_id` to `pipeline_runs`.

The governed path resolves:

```text
control_attempt_id
       ↓
exact monitoring_run_id
       ↓
M9 review of that exact run
```

This avoids selecting a generic latest monitoring run during a controlled attempt.

### Deterministic review

M9 evaluates:

```text
R001 Pipeline Run Unsuccessful
R002 Model Execution Non-Success
R003 Test Result Non-Passing
R004 Model Missing from Current Run
R005 Row-Count Anomaly
R006 Runtime Regression
```

Every rule produces exactly one of:

```text
PASS
TRIGGERED
NOT_EVALUATED
```

Missing evidence remains visible as `NOT_EVALUATED`.

Optional AI explanation is downstream. It cannot create findings, change severity, or change deterministic results.

## 3. Window-control architecture

The control layer uses two BigQuery tables:

```text
olist_control.pipeline_control_state
olist_control.pipeline_window_events
```

The current-state table stores one row per pipeline/environment pair.

The event table is append-only and records state transitions, attempts, windows, retry lineage, errors, versions, and timestamps.

### State model

```text
IDLE
  ↓
RUNNING
  ├─ success → IDLE
  └─ failure → FAILED
                  ↓
             WAITING_RETRY
                  ↓
               RUNNING
```

`QUARANTINED` exists in the state model. Automatic quarantine/release policy is outside M10.

### Invariants

The implemented controller enforces:

```text
normal watermark moves only forward
failure does not advance watermark
retry reuses the failed window
retry creates a new attempt_id
retry links to the prior attempt
state and audit event commit atomically
stale control_version writers are rejected
one attempt resolves one monitoring run
```

### Concurrency

Each control transition is written inside a BigQuery transaction.

The state update includes compare-and-set logic:

```text
WHERE pipeline_name = ...
  AND environment = ...
  AND control_version = expected_version
```

If the expected version is stale, zero state rows are updated and the transaction rolls back, including the matching audit event.

## 4. Current scheduled runtime

The current production scheduling path is:

```text
Cloud Scheduler
      ↓
Cloud Run Job
      ↓
Docker image
      ↓
run_dbt_job.sh
      ↓
dbt execution
      ↓
BigQuery / monitoring
```

The container contains:

- dbt and dbt-bigquery
- BigQuery client libraries
- monitoring code
- deterministic reviewer code
- M10 Window Controller code

The current scheduled entry point remains:

```text
/app/dbt/run_dbt_job.sh
```

When no control window is supplied, it runs in full-history compatibility mode.

The M10 controller can invoke the same script with:

```text
CONTROL_ATTEMPT_ID
CONTROL_WINDOW_START
CONTROL_WINDOW_END
MONITORING_ENVIRONMENT
```

That controller path has been validated independently but has not replaced the Scheduler entry point.

## 5. Portal architecture

The Portal is a Next.js server-rendered application.

The server boundary is:

```text
Server Component
      ↓
Service
      ↓
Repository
      ↓
BigQuery
```

The project deliberately avoids adding an internal HTTP API layer where server components can call the service boundary directly.

Main routes:

```text
/overview
/analytics
/reliability
/findings/[findingId]
/health
```

### Analytics integrity boundary

Persisted statistical diagnostic rows are not trusted blindly.

The service layer verifies diagnostic fields and mathematical relationships before data is returned to the UI, including model/version and snapshot consistency.

### Reliability integrity boundary

Finding routes use server-side repository/service access and validate persisted finding identity/evidence consistency before rendering.

## 6. Delivery architecture

### Portal

```text
portal/** change
      ↓
GitHub Portal CI
      ↓
Vitest + lint + production build
      ↓
Render deployment boundary
      ↓
public Portal
```

### Data pipeline

```text
dbt/** change
      ↓
GitHub Pipeline CI/CD
      ↓
Python tests
shell validation
Docker build/smoke
      ↓
main branch only
      ↓
GitHub OIDC
      ↓
Google Workload Identity Federation
      ↓
dedicated deployer service account
      ↓
Artifact Registry
      ↓
Cloud Run Job update
      ↓
image verification
```

The deployer identity is separate from the runtime identity.

The GitHub-to-GCP path uses short-lived credentials and does not store a long-lived GCP service-account JSON key in GitHub.

## 7. Identity boundaries

The main service identities are deliberately separated:

| Identity | Responsibility |
|---|---|
| Cloud Scheduler invoker | Trigger the Cloud Run Job |
| Cloud Run runtime account | Execute dbt/monitoring workload |
| GitHub deployer account | Push pipeline images and update the Cloud Run Job |
| Render read-only BigQuery account | Serve Portal queries from BigQuery |

This prevents the deployment identity from automatically inheriting runtime data privileges.

## 8. Current implementation boundary

Implemented:

- BigQuery layered warehouse
- dbt transformations, tests, docs, lineage
- incremental transactional facts
- Cloud Scheduler + Cloud Run scheduled execution
- append-only monitoring evidence
- deterministic M9 reviewer
- M10 forward window/watermark controller
- retry lineage and audit events
- BigQuery transaction + CAS protection
- exact control-attempt to monitoring-run correlation
- Next.js operational/reliability/analytics Portal
- Brazil state-level decision analytics
- persisted statistical diagnostics and service validation
- Render public deployment
- split GitHub Portal/Pipeline CI
- main-only GCP pipeline deployment path using OIDC/WIF

Not implemented in M10:

- Scheduler entry through `run_window_controller.py`
- automatic retry limit/quarantine/release policy
- M11 replay/backfill/resume
- alert delivery
- application-level organization authentication

## Detailed documents

- [`deployment.md`](deployment.md)
- [`m10_window_control.md`](m10_window_control.md)
- [`m10_portal_analytics.md`](m10_portal_analytics.md)
- [`m9_expert_system_closing.md`](m9_expert_system_closing.md)
- [`orchestration.md`](orchestration.md)
- [`README.md`](README.md)

# Architecture

## Purpose

This document describes the current architecture of the Olist E-Commerce Analytics & Pipeline Monitoring Portal.

The project includes:

- BigQuery raw, staging, intermediate, marts, monitoring, control, and analytics datasets
- dbt transformations, tests, documentation, and lineage
- Dockerized dbt execution
- Google Cloud Run Job and Cloud Scheduler orchestration
- append-only dbt artifact monitoring
- deterministic pipeline review rules R001-R006
- window and watermark control with retry history
- exact correlation from a control attempt to its monitoring run
- BigQuery transaction and version checks for state updates
- a Next.js operational and analytics portal
- state-level business actions and statistical review diagnostics
- service-layer integrity checks for persisted analytics results
- basic portal security headers

M10 is complete. M11 will focus on replay, backfill, resume, and recovery.

---

## Architecture status

```text
M1  Project Setup & Source Understanding        completed
M2  BigQuery Raw Layer                          completed
M3  Staging Layer Planning                      completed
M4  dbt Staging Layer                           completed
M5  Dimensional Modeling / Analytics Marts      completed
M6  Project Documentation Cleanup               completed
M7  Cloud Run + Cloud Scheduler                 completed
M8  dbt Monitoring                              completed
M9  Pipeline Quality Reviewer                   completed
M10 Window Control, Portal & Analytics           completed
M11 Replay / Backfill / Recovery                planned
```

Key validation dates:

```text
M8 cloud monitoring:        2026-07-15
M9 reviewer:                2026-08-10
M10 U1 window control:      2026-08-15
M10 portal hardening:       2026-08-30
```

---

## High-level architecture

The project has four connected paths.

```text
                         Olist source CSVs
                                ↓
                            olist_raw
                                ↓
                           dbt staging
                                ↓
                  ┌─────────────┴─────────────┐
                  │                           │
          business data path          pipeline evidence path
                  │                           │
          dbt intermediate             dbt build artifacts
                  ↓                           ↓
             olist_marts               M8 artifact loader
                  ↓                           ↓
          analytics serving             olist_monitoring
                  │                           ↓
                  │                   M9 rule-based review
                  │                           │
                  └──────────────┬────────────┘
                                 ↓
                           Next.js portal
```

M10 adds a control path around the transactional workload:

```text
olist_control
     ↓
claim next window
     ↓
run dbt for that window
     ↓
M8 monitoring
     ↓
resolve exact monitoring run
     ↓
M9 review
     ↓
success / failure
     ↓
update control state
```

Portal pages use a direct server-side path:

```text
Next.js Server Component
        ↓
     service
        ↓
   repository
        ↓
     BigQuery
```

The portal does not use an internal HTTP API layer unless one is needed. The two early API routes for overview and reliability were removed because no client code used them.

---

## Source data

The project uses 9 files from the Olist Brazilian E-Commerce dataset:

- customers
- geolocation
- orders
- order items
- order payments
- order reviews
- products
- sellers
- product category translation

The local raw files are ignored by Git.

---

## BigQuery datasets

| Dataset | Purpose |
|---|---|
| `olist_raw` | Source-aligned raw tables |
| `olist_staging` | Clean dbt staging views |
| `olist_intermediate` | Reusable order-level logic |
| `olist_marts` | Analytics-ready facts and dimensions |
| `olist_monitoring` | Append-only pipeline history |
| `olist_control` | Current pipeline state and append-only state events |

Validation runs use isolated dbt datasets such as:

```text
olist_validation_staging
olist_validation_intermediate
olist_validation_marts
```

The raw source remains shared.

---

## dbt transformation path

### Staging

The staging layer keeps transformations light:

- rename fields
- cast types
- normalize simple values
- keep stable source-facing models
- run source-level quality tests

There are 9 staging views.

### Intermediate

Current intermediate models:

```text
int_orders_windowed
int_order_items_agg
int_order_payments_agg
int_order_reviews_agg
```

`int_orders_windowed` is the M10 transaction anchor.

Without control variables, it reads full order history. With a control window, it applies:

```text
order_purchase_timestamp >= window_start
and order_purchase_timestamp < window_end
```

This half-open interval prevents overlap between adjacent windows.

The other order-level intermediate models use the current window's order IDs to limit items, payments, and reviews.

### Marts

Dimensions:

```text
dim_customers
dim_sellers
dim_products
dim_geolocation_zip_prefix
dim_dates
```

Facts:

```text
fct_orders
fct_order_items
fct_order_payments
fct_order_reviews
```

The dimensions remain full-history tables.

The four fact models are incremental `MERGE` models:

| Fact | Unique key |
|---|---|
| `fct_orders` | `order_id` |
| `fct_order_items` | `order_item_key` |
| `fct_order_payments` | `order_payment_key` |
| `fct_order_reviews` | `review_key` |

This lets a retry process the same window again without appending duplicate fact rows for the same key.

---

## Windowed data flow

The M10 data path keeps reference data and transaction processing separate.

```text
                    full-history source context
                              ↓
                         dimensions
                              │
                              │
stg_orders → int_orders_windowed
                    ↓
             current order IDs
             ↙       ↓       ↘
       order_items payments reviews
             ↘       ↓       ↙
         order-level intermediate models
                    ↓
             incremental facts
                    ↓
                  marts
                    ↓
            analytics serving
```

The window is based on `order_purchase_timestamp` from orders. Related item, payment, and review records enter the window through their `order_id` relationship to those orders.

---

## Data model decisions

### Review fact grain

`review_id` is not unique by itself in the source.

The review fact uses:

```text
review_id + order_id
```

as the source grain and creates `review_key` for the dbt model key.

### Geolocation

The raw geolocation source contains several coordinate rows for the same zip-code prefix.

`dim_geolocation_zip_prefix` stores one representative row per prefix.

### Shared date dimension

`dim_dates` provides a common reporting date table for order, shipping, and review dates.

---

## Data quality

The project uses dbt tests for repeatable model checks.

Common tests include:

- `not_null`
- `unique`
- `relationships`
- `accepted_values`

Historical milestone results:

```text
M4 staging:                    39 tests passed
M5 intermediate + marts:      67 nodes passed
M8 cloud build:              115 nodes passed
```

Current M10 validation build:

```text
22 models
96 dbt tests
118 / 118 PASS
```

The current backend unit-test inventory is:

```text
M10 controller tests:         52
M8 run resolver tests:         5
M9 reviewer tests:            53
Total:                       110
```

Portal validation includes:

```text
Vitest:                       21 / 21 PASS
npm audit:                    0 vulnerabilities
```

---

## M7 cloud execution

M7 introduced the existing scheduled cloud path:

```text
Cloud Scheduler
      ↓
Cloud Run Job
      ↓
Docker image
      ↓
run_dbt_job.sh
      ↓
dbt debug + dbt build
      ↓
BigQuery
```

The Cloud Scheduler job and Cloud Run Job were validated in M7/M8.

The container entry point is still:

```text
/app/dbt/run_dbt_job.sh
```

This matters for the current M10 boundary: the scheduled Cloud Run job has not yet been changed to start `run_window_controller.py`.

---

## M8 monitoring

M8 adds an append-only monitoring path after dbt execution.

```text
dbt build
   ↓
manifest.json + run_results.json
   ↓ preserve build artifacts
dbt docs generate
   ↓
catalog.json
   ↓ restore build manifest/run_results
artifact parser
   ↓
BigQuery loader
   ↓
olist_monitoring
```

The loader reads:

- `manifest.json` from `dbt build`
- `run_results.json` from `dbt build`
- `catalog.json` from `dbt docs generate`

Monitoring tables:

| Table | Grain |
|---|---|
| `pipeline_runs` | One row per monitoring run |
| `model_run_results` | One row per model execution per run |
| `test_run_results` | One row per test execution per run |
| `model_metadata_snapshots` | One row per model per run |
| `model_column_snapshots` | One row per model/source column per run |
| `model_lineage_edges` | One row per dependency edge per run |

The tables keep historical runs instead of overwriting the latest state.

M10 adds `control_attempt_id` to `pipeline_runs`. This field links M8 evidence to one M10 attempt.

The original M8 cloud validation on 2026-07-15 recorded:

```text
21 successful models
94 passed tests
259 model/source column snapshots
146 lineage edges
```

Those values are historical M8 evidence. The current dbt project contains one additional intermediate model and two additional tests.

---

## M9 pipeline review

M9 reads M8 monitoring history and evaluates six rules.

```text
R001  Pipeline Run Unsuccessful
R002  Model Execution Non-Success
R003  Test Result Non-Passing
R004  Model Missing from Current Run
R005  Row-Count Anomaly
R006  Runtime Regression
```

Every evaluation is one of:

```text
PASS
TRIGGERED
NOT_EVALUATED
```

Missing evidence stays visible as `NOT_EVALUATED`.

Historical comparisons use successful runs with the same `job_name` and `environment`. R005 and R006 use the median from up to five prior comparable runs where the required evidence exists.

Vertex AI is optional and comes after deterministic evaluation. It can explain existing findings but cannot create findings, change severity, or change the rule result.

M9 final validation on 2026-08-10:

```text
179 evaluations
166 PASS
1 TRIGGERED
12 NOT_EVALUATED
53 unit tests passed
```

The real triggered finding was an R006 runtime regression for `fct_order_payments`.

---

## M10 U1 control architecture

### Control tables

M10 uses two BigQuery tables.

#### `pipeline_control_state`

One current row per pipeline/environment pair.

Main fields:

```text
pipeline_name
environment
state
last_successful_window_start
last_successful_window_end
active_window_start
active_window_end
active_attempt_id
active_attempt_number
active_retry_of_attempt_id
control_version
last_error_code
last_error_message
updated_at
```

#### `pipeline_window_events`

Append-only audit history for state changes.

It records:

- event ID
- attempt ID
- window
- attempt number
- old and new state
- old and new control version
- retry link
- error details
- event time
- optional metadata

The event table is partitioned by event date and clustered for common operational filters.

### State model

```text
IDLE
  ↓
RUNNING
  ├── success → IDLE
  └── failure → FAILED
                  ↓
             WAITING_RETRY
                  ↓
               RUNNING
```

The model also contains `QUARANTINED`.

U1 does not implement an automatic quarantine limit, automatic release, or a quarantine CLI command.

### State initialization

Control state is created explicitly with `bootstrap_window_control.py`.

The normal controller does not silently create missing state.

Initial state:

```text
state = IDLE
control_version = 0
last_successful_window = NULL
active_attempt = NULL
```

### New window

The next window starts at:

```text
initial_start
```

when no successful window exists.

After that, it starts at:

```text
last_successful_window.end
```

The normal flow only moves forward.

### Success

A successful workload:

- changes `RUNNING → IDLE`
- stores the completed window as `last_successful_window`
- clears the active attempt
- clears the last error
- increases `control_version`

### Failure

A failed workload:

- changes `RUNNING → FAILED`
- keeps the same active window and attempt
- records an error code/message
- does not change `last_successful_window`

### Retry

Retry supports state `FAILED` or `WAITING_RETRY`.

Normal retry flow:

```text
FAILED
  ↓ WINDOW_RETRY_SCHEDULED
WAITING_RETRY
  ↓ WINDOW_RETRY_STARTED
RUNNING
```

The retry gets:

- the same window
- a new `attempt_id`
- `attempt_number + 1`
- `retry_of_attempt_id = previous attempt_id`

If the runtime stops after writing `WAITING_RETRY`, a later retry call can resume from that state.

---

## State persistence and concurrency

The control repository uses a BigQuery transaction for each transition.

Inside one transaction it:

1. updates the current control-state row
2. checks that exactly one row matched the expected `control_version`
3. inserts the matching audit event
4. commits both changes together

The update uses compare-and-set logic:

```text
WHERE pipeline_name = ...
  AND environment = ...
  AND control_version = expected_version
```

If another writer has already moved the version, the update changes zero rows. The transaction raises a stale-version error and rolls back.

A real BigQuery validation confirmed that a stale writer was rejected and no test audit event was inserted.

---

## M10 workload integration

`run_window_controller.py` sets these values for the dbt workload:

```text
CONTROL_ATTEMPT_ID
CONTROL_WINDOW_START
CONTROL_WINDOW_END
MONITORING_ENVIRONMENT
```

`run_dbt_job.sh` converts the window into dbt variables and runs:

```text
dbt debug
dbt build --vars ...
dbt docs generate --vars ...
M8 artifact loader
M9 exact-run reviewer
```

When no control window is supplied, the same script keeps a full-history compatibility mode.

In window-controlled mode, `CONTROL_ATTEMPT_ID` is required.

---

## Exact M8/M9 run correlation

The M8 `pipeline_runs` table stores `control_attempt_id`.

After M8 writes the monitoring data, the resolver runs:

```text
control_attempt_id
        ↓
olist_monitoring.pipeline_runs
        ↓
exact monitoring_run_id
        ↓
M9 --monitoring-run-id <exact id>
```

The resolver requires exactly one matching monitoring run.

It fails if:

- no run exists for the attempt
- more than one run exists for the attempt
- the matching row has an empty run ID

This avoids using a generic "latest run" in the window-controlled path.

---

## M10 real validation

### Successful windows

The validation environment processed historical daily windows in isolated dbt datasets.

The first controlled window created real fact rows and completed the full path:

```text
controller
→ windowed dbt build
→ incremental facts
→ M8 monitoring
→ exact monitoring run resolution
→ M9 review
→ watermark advance
```

A second window confirmed that the next start came from the previous successful window end.

### Failure and retry

For the window:

```text
2016-09-06 00:00:00
→ 2016-09-07 00:00:00
```

real audit history showed:

```text
attempt 1  WINDOW_STARTED
attempt 1  WINDOW_FAILED
attempt 1  WINDOW_RETRY_SCHEDULED
attempt 2  WINDOW_RETRY_STARTED
attempt 2  WINDOW_FAILED
attempt 2  WINDOW_RETRY_SCHEDULED
attempt 3  WINDOW_RETRY_STARTED
attempt 3  WINDOW_SUCCEEDED
```

Control versions moved continuously from 4 to 12.

The watermark stayed at the prior successful window through both failures. It moved to `2016-09-07` only after attempt 3 succeeded.

The final state was:

```text
state = IDLE
control_version = 12
last_successful_window = 2016-09-06 → 2016-09-07
active attempt = NULL
last error = NULL
```

The CAS probe then tried to write with stale version 11. BigQuery rejected the transition, state stayed at version 12, and the test event count remained zero.

---

## M10 portal and analytics

### Routes

The portal has four main routes:

```text
/overview
/analytics
/reliability
/findings/[findingId]
```

`/overview` shows operational control state. `/reliability` shows the latest deterministic review and triggered findings. `/findings/[findingId]` shows persisted evidence for one finding. `/analytics` shows business actions and state-level review diagnostics.

### Analytics serving layer

The analytics page reads governed BigQuery data through server-side repositories and services.

```text
BigQuery
  ↓
repository
  ↓
service
  ↓
Server Component
  ↓
client-side interaction
```

The current state summary covers all 27 Brazilian states. The map links state selection to KPI and diagnostic cards.

### Business Decision Model v1

The first decision model is deterministic. It uses peer-relative thresholds for market value and service health.

Actions are:

```text
RECOVER_SERVICE
PROTECT_VALUE
EXPAND
INVESTIGATE
MONITOR
```

The current full-history snapshot does not provide a governed previous-window growth measure. `EXPAND` therefore remains reserved until M11 adds monthly playback and previous-window comparison.

### Review Diagnostic v2

The statistical diagnostic estimates negative-review risk after accounting for order and delivery mix. The stored result contains observed risk, expected risk, residual in percentage points, confidence interval, evidence count, model version, and diagnostic state.

The diagnostic contract is:

```text
evidence_count < 100
→ INSUFFICIENT_EVIDENCE

residual_pp >= 1 and ci_lower_pp > 0
→ WORSE_THAN_EXPECTED

residual_pp <= -1 and ci_upper_pp < 0
→ BETTER_THAN_EXPECTED

otherwise
→ AS_EXPECTED
```

### Service-layer verification

Persisted data is checked before it reaches the UI.

```text
BigQuery row
   ↓
field validation
   ↓
row-level consistency checks
   ↓
snapshot-level checks
   ↓
UI
```

The service checks:

- all 27 state rows are present and unique
- probabilities and counts are in valid ranges
- confidence interval bounds are ordered
- `residual_pp` matches `(actual - expected) × 100` within a small tolerance
- `diagnostic_state` matches the fixed classification rule
- all rows use the same `model_version`
- all rows use the same `generated_at`

This keeps the repository responsible for reading data and the service responsible for deciding whether the data is safe to use.

### Finding identifier boundary

Finding detail uses a URL identifier. The page decodes the path segment once. The service then checks the decoded identifier before the repository can query BigQuery.

Allowed finding IDs are limited to 512 characters and a small character set used by the persisted IDs. BigQuery queries remain parameterized.

### Portal security baseline

The portal sets these response headers:

- `Content-Security-Policy`
- `X-Content-Type-Options: nosniff`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Permissions-Policy`
- `X-Frame-Options: DENY`

The CSP allows the CARTO basemap resources used by the map. Development allows `unsafe-eval` for the local Next.js toolchain; the production policy does not.

The portal currently has no application-level login. A public or shared deployment should sit behind organization authentication or an equivalent platform control, and the runtime service account should only have the BigQuery permissions it needs.

See [`m10_portal_analytics.md`](m10_portal_analytics.md) for the M10 portal close-out record.

---

## Security and runtime configuration

Credentials are not stored in the repository.

Runtime values are supplied through environment variables or Google Cloud identity.

Important dbt runtime variables include:

```text
DBT_PROJECT_ID
DBT_TARGET
DBT_DATASET
DBT_LOCATION
DBT_THREADS
```

Monitoring values include:

```text
GCP_PROJECT_ID
MONITORING_DATASET_ID
MONITORING_JOB_NAME
MONITORING_ENVIRONMENT
```

Governed runs add:

```text
CONTROL_ATTEMPT_ID
CONTROL_WINDOW_START
CONTROL_WINDOW_END
```

Non-production controller runs must use an isolated dbt dataset instead of the default `olist` dataset.

---

## Container boundary

The Docker image contains dbt, BigQuery libraries, monitoring code, the reviewer, and the M10 controller code.

Current entry point:

```text
/app/dbt/run_dbt_job.sh
```

This is still the M7/M8 scheduled-job entry point.

The M10 controller can run the same workload, but Cloud Scheduler / Cloud Run has not yet been changed to start the controller by default.

This distinction is intentional in the documentation so the project does not claim a scheduled M10 control path that has not been validated.

---

## Current boundary

Implemented:

- raw → staging → intermediate → marts
- facts and dimensions
- dbt tests and docs
- scheduled Cloud Run dbt execution
- append-only monitoring history
- deterministic M9 review rules
- optional Vertex AI explanation
- M10 control state and audit tables
- explicit bootstrap and forward windows
- failure handling and same-window retries
- retry lineage and incremental fact writes
- exact M8/M9 run correlation
- BigQuery transaction and CAS protection
- operational portal and reliability views
- finding detail view
- state-level analytics map
- deterministic Business Decision Model v1
- statistical Review Diagnostic v2
- service-layer integrity checks
- portal security headers
- portal regression tests

Not implemented yet:

- scheduled Cloud Run entry through `run_window_controller.py`
- automatic retry limit
- automatic quarantine policy and release workflow
- replay and backfill
- historical resume workflow
- application-level login for a public portal deployment
- alert delivery

---

## Next architecture work

### M11 replay and recovery

M11 will add controlled historical processing. The default playback window will be one month.

Planned work:

- monthly window planning
- one-window replay
- multi-window backfill
- resume after failure
- idempotency checks
- incremental-versus-replay comparison
- separate replay audit state

Replay state must stay separate from the normal forward watermark.

The project will first produce and validate the monthly history. Trend analysis, seasonality, and forecasting are not part of the current M11 scope.

---

## Design rules

- Keep each layer's job clear.
- Keep raw data close to the source.
- Make model grain explicit.
- Use tests for repeatable checks.
- Keep monitoring history append-only.
- Keep current control state separate from event history.
- Advance the normal watermark only after success.
- Retry the same failed window before moving forward.
- Reject stale state updates.
- Link one control attempt to one monitoring run.
- Keep deterministic review results separate from optional explanations.
- Do not mix M11 replay logic into the normal M10 forward path.

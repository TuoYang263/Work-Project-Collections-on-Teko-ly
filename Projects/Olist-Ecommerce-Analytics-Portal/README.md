# Olist E-Commerce Analytics & Pipeline Monitoring Portal

A **production-style e-commerce data platform and decision portal** built on the Olist Brazilian E-Commerce dataset.

Instead of exposing the historical dataset as one static full-history snapshot, the deployed runtime processes it through governed calendar-month production cycles, persists monitoring and reliability evidence for every controlled run, and exposes only successfully processed analytics through a public Next.js portal.

**Live portal:** https://olist-analytics-portal.onrender.com
**Health check:** https://olist-analytics-portal.onrender.com/health

\> The public demo runs on Render's free tier, so the first request after inactivity can take longer while the service wakes up.

## What this project does

The current production path is:

```text
Cloud Scheduler
        ↓
Cloud Run Job
        ↓
M10 Window Controller
        ↓
calendar-month processing window
        ↓
run_dbt_job.sh
        ↓
dbt build
        ↓
M8 monitoring persistence
        ↓
exact monitoring-run resolution
        ↓
M9 deterministic reliability review
        ↓
successful watermark advance
        ↓
BigQuery analytics serving
        ↓
Next.js Portal
```

The production Scheduler runs **hourly**.

```text
Cron:      0 * * * *
Timezone:  Europe/Helsinki
```

One successful scheduled execution processes approximately one historical calendar month.

The bounded production source range is:

```text
2016-09-01T00:00:00+00:00
→
2018-11-01T00:00:00+00:00
```

This gives 26 calendar-month windows per complete cycle.

When the source end is reached, the controller starts a new `cycle_id` from the configured source beginning instead of rewinding the previous production cycle.

## Engineering highlights

\- **Governed monthly processing:** forward-only calendar-month `[window_start, window_end)` execution with explicit `cycle_id`.
\- **Failure-safe watermark semantics:** a failed workload never advances the successful watermark.
\- **Exact-window retry:** retries reuse the same failed window with a new attempt ID, incremented attempt number, and explicit retry lineage.
\- **Transactional control state:** BigQuery transaction \+ `control_version` compare-and-set protection keeps current state and append-only audit history consistent.
\- **Exact observability correlation:** each controlled attempt resolves the exact M8 monitoring run through `control_attempt_id`.
\- **Deterministic reliability review:** R001-R006 preserve `PASS / TRIGGERED / NOT_EVALUATED`; optional AI explanation cannot modify deterministic findings or severity.
\- **Successful-watermark analytics:** KPI and state-level serving expose data only through `last_successful_window_end`.
\- **Complete 27-state analytical universe:** states remain visible even when the current analytical scope contains no orders for them.
\- **Decision-oriented analytics:** deterministic business actions combine commercial value, delivery quality, and review evidence.
\- **Historical statistical diagnostics:** review-risk diagnostics remain a separately persisted historical statistical layer rather than being confused with current watermark-scoped KPIs.
\- **Production-oriented delivery:** Portal and pipeline CI/CD are separate, with GitHub OIDC, Workload Identity Federation, and least-privilege GCP identities.

## Current production behavior

```text
Scheduler cadence:      hourly
Cron:                   0 * * * *
Scheduler timezone:     Europe/Helsinki
Cloud Scheduler region: europe-west1
Cloud Run Job region:   europe-north1
BigQuery location:      EU
Cloud Run max retries:  0
```

Cloud Run platform retries are disabled for the controlled job.

Retry semantics are owned by the M10 controller so that failures preserve the exact processing window and leave an explicit audit trail.

The public Portal therefore changes over time as successful scheduled runs advance through the historical production cycle.

\---

## Product preview

\![Olist analytics workspace](assets/screenshots/portal/portal-analytics-hero.png)

The analytics workspace connects commercial value and service-health evidence to state-level actions such as **Recover Service**, **Protect Value**, **Investigate**, and **Monitor**.

\---

## System architecture

\![Olist system architecture](assets/architecture/olist-system-architecture.png)

Editable source:

[`assets/architecture/olist-system-architecture.drawio`](assets/architecture/olist-system-architecture.drawio)

The deployed Cloud Run Job starts:

```text
python /app/dbt/control/run_window_controller.py
```

The controller claims the next governed calendar-month window and injects:

```text
CONTROL_ATTEMPT_ID
CONTROL_WINDOW_START
CONTROL_WINDOW_END
```

into the dbt runtime.

`run_dbt_job.sh` then executes the windowed dbt workload, monitoring persistence, exact monitoring-run resolution, and deterministic M9 review.

Only after the complete controlled workload succeeds does the controller persist `WINDOW_SUCCEEDED` and advance the successful watermark.

\---

## Product capabilities

### Analytics

The `/analytics` workspace provides:

\- successful-watermark-scoped order count
\- GMV
\- average order value
\- delivery observation count
\- late-delivery rate
\- reviewed-order count
\- average review score
\- Brazil state-level geospatial selection
\- deterministic Business Decision Model v1
\- P1 / P2 / P3 priorities
\- Historical Statistical Review Diagnostic v2
\- deterministic validation before persisted diagnostic data reaches the UI

The current business KPI and state-serving layer reads data only through:

```text
last_successful_window_end
```

This means an active or failed processing window cannot leak partial business data into the Portal.

While a new window is `RUNNING`, Analytics remains on the previous successful scope.

After `WINDOW_SUCCEEDED` commits, the analytical serving views advance automatically with the successful watermark.

### Complete 27-state market universe

The state-serving layer intentionally preserves all 27 Brazilian state codes.

Eligible orders are watermark-filtered and then left-joined to the complete state universe.

For zero-order states:

```text
order_count \= 0
gmv         \= 0
aov         \= 0
```

Metrics that require observed evidence remain unavailable rather than being fabricated:

```text
late_delivery_rate   \= NULL when no delivery observations exist
average_review_score \= NULL when no reviewed orders exist
```

\---

## Business Decision Model v1

Business Decision Model v1 is deterministic.

Inputs include:

```text
stateCode
gmv
gmvGrowthRate
lateDeliveryRate
averageReviewScore
```

Peer-relative thresholds use:

```text
GMV                 P75
GMV growth          P75
late-delivery rate  P75
review score        P25
```

Actions:

```text
RECOVER_SERVICE
PROTECT_VALUE
EXPAND
INVESTIGATE
MONITOR
```

Priority levels:

```text
P1
P2
P3
```

The current successful-watermark analytical layer does not yet expose a governed previous-period growth metric, so `gmvGrowthRate` remains unavailable.

`EXPAND` is reserved until a governed comparison series exists.

\---

## Historical Review Diagnostic v2

The statistical analytics layer estimates negative-review risk after accounting for order and delivery mix.

Its scope is intentionally different from the current business-action scope.

```text
Business actions
→ current successful-watermark data

Historical Review Diagnostic v2
→ persisted historical statistical evidence
```

The UI therefore labels this section:

```text
Historical review risk vs expected
```

and its sample size:

```text
Historical orders evaluated
```

This prevents a historical model estimate from being mistaken for current watermark-scoped KPI evidence.

\---

## Reliability

The `/reliability` workspace reads persisted deterministic M9 review results.

Three outcomes are preserved:

```text
PASS
TRIGGERED
NOT_EVALUATED
```

The deterministic rule set is:

| Rule | Check |
|---|---|
| R001 | Pipeline Run Unsuccessful |
| R002 | Model Execution Non-Success |
| R003 | Test Result Non-Passing |
| R004 | Model Missing from Current Run |
| R005 | Row-Count Anomaly |
| R006 | Runtime Regression |

\---

## Operational state

The `/overview` workspace exposes:

\- current controller state
\- environment
\- cycle ID
\- active attempt
\- active processing window
\- last successful processing window
\- control version
\- controller freshness
\- retry/failure information
\- latest error evidence

\---

## Monitoring evidence

M10 adds `control_attempt_id` to `pipeline_runs`.

Correlation path:

```text
controller attempt_id
        ↓
pipeline_runs.control_attempt_id
        ↓
monitoring_run_id
        ↓
M9 deterministic review
```

The controller does not guess the latest monitoring run.

\---

## Window and watermark control

Core M10 invariants:

```text
1\. Normal production processing uses forward calendar-month windows.
2\. A successful window advances the successful watermark.
3\. A failed workload does not advance the successful watermark.
4\. Retry reuses the exact failed window.
5\. Every retry receives a new attempt_id.
6\. State and audit-event writes commit in one BigQuery transaction.
7\. Stale concurrent writers are rejected by control_version CAS.
8\. One controller attempt resolves one exact monitoring run.
9\. Reaching SOURCE_END creates a new cycle_id instead of rewinding a cycle.
10\. Arbitrary replay/backfill does not silently move the normal production watermark.
```

Detailed documentation:

[`docs/m10_window_control.md`](docs/m10_window_control.md)

\---

## Production validation

The first real deployed monthly production window was:

```text
cycle_id \= 1
2016-09-01T00:00:00+00:00
→
2016-10-01T00:00:00+00:00
```

The control audit recorded `WINDOW_STARTED` and `WINDOW_SUCCEEDED` for the same controller attempt.

The matching monitoring run completed with:

```text
22 / 22 models successful
96 / 96 tests passed
```

The first watermark-scoped analytical serving result contained:

```text
4 orders
GMV \= R$354.75
AOV \= R$88.69
observed order dates \= 2016-09-04 → 2016-09-15
```

State-serving integrity check:

```text
27 total states
4 total orders
24 zero-order states
```

Portal regression checks:

```text
Vitest:                   21 / 21 PASS
ESLint:                   PASS
Next.js production build: PASS
```

\---

## Tech stack

| Area | Technology |
|---|---|
| Warehouse | Google BigQuery |
| Transformation | dbt Core, dbt-bigquery |
| Modeling | Dimensional modeling, incremental MERGE |
| Monitoring | dbt artifacts, Python, BigQuery |
| Reliability | Deterministic Python rules R001-R006 |
| Window control | Python, BigQuery transactions, CAS |
| Cloud runtime | Cloud Run Jobs, Cloud Scheduler |
| Containers | Docker, Artifact Registry |
| Portal | Next.js, React, TypeScript |
| Geospatial | MapLibre, deck.gl, CARTO basemap |
| Statistics | Persisted logistic-regression diagnostics |
| CI/CD | GitHub Actions, OIDC, Workload Identity Federation |
| Portal hosting | Render |

\---

## Current boundary and M11

M10 is complete and deployed.

The production Scheduler now invokes the M10 Window Controller.

Normal production execution uses bounded calendar-month windows and a successful watermark.

The cycle mechanism is a normal forward production simulation, not an arbitrary historical replay interface.

M11 is reserved for controlled historical playback and recovery:

\- one-window replay
\- multi-window backfill
\- resume after failure
\- replay idempotency
\- replay versus incremental consistency checks
\- separate replay state from the normal forward watermark
\- replay audit history

Replay must never silently move the normal incremental production watermark backward.

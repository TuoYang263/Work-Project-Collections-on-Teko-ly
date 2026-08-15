# Project Management

## Purpose

This document tracks the current milestone plan for the Olist E-Commerce Analytics & Pipeline Monitoring Portal.

The project is built in small milestones. Each milestone should have a clear result, a validation step, and a clean commit before the next area grows.

M10 is currently in progress. Its first unit, window and watermark control, is complete. The portal and analytics work comes next.

---

## Git workflow

Development branch:

```text
feature/olist-analytics-portal
```

Work stays on the feature branch until the current milestone has been reviewed, tested, documented, committed, and pushed.

Raw data, local credentials, generated dbt targets, and local profiles should not be committed.

---

## Commit style

Use short commit messages that describe the change.

Examples:

```text
feat: add window control state
feat: add retry runtime
fix: export dbt runtime variables
docs: document M10 window control
```

Common prefixes:

- `feat:` new implementation
- `fix:` correction
- `docs:` documentation
- `chore:` setup or maintenance
- `data:` source metadata or sample-data notes

---

## Milestone plan

| Milestone | Focus | Status | Main output |
|---|---|---:|---|
| M1 | Project setup and source understanding | Completed | Repository structure and source inventory |
| M2 | BigQuery raw layer | Completed | `olist_raw` and 9 raw tables |
| M3 | Staging layer planning | Completed | Staging rules and source mapping |
| M4 | dbt staging layer | Completed | 9 staging views and 39 tests |
| M5 | Dimensional modeling / marts | Completed | Facts, dimensions, intermediate models, dbt docs |
| M6 | Documentation cleanup | Completed | README, architecture, and project showcase |
| M7 | Cloud orchestration | Completed | Docker, Cloud Run Job, Cloud Scheduler |
| M8 | Pipeline monitoring | Completed | Six append-only monitoring tables |
| M9 | Pipeline quality reviewer | Completed | Rules R001-R006 and optional explanations |
| M10 U1 | Window / watermark control | Completed | Window state, retries, audit history, CAS protection, exact M8/M9 correlation |
| M10 portal / analytics | In progress | Operational UI, analytics UI, Brazil map slice |
| M11 | Replay / backfill / recovery | Planned | Historical processing and recovery controls |

---

## Current scope

### Completed through M10 U1

The project currently has:

- BigQuery raw, staging, intermediate, marts, monitoring, and control datasets
- dbt models, tests, docs, and lineage
- Dockerized dbt execution
- Cloud Run Job and Cloud Scheduler for the existing scheduled dbt path
- append-only M8 monitoring history
- deterministic M9 rules R001-R006
- optional Vertex AI explanation for triggered M9 findings
- explicit M10 control-state bootstrap
- forward window derivation
- windowed transactional dbt processing
- incremental `MERGE` facts
- success and failure state updates
- same-window retry with attempt history
- exact `control_attempt_id → monitoring_run_id` resolution
- BigQuery transaction and stale-version protection
- real validation for success, failure, retry, and stale concurrent writes

### Not part of U1

U1 does not include:

- replay or backfill
- moving the normal watermark backward
- automatic retry limits
- automatic quarantine or release policy
- portal screens
- alert delivery

`QUARANTINED` exists in the state model, but U1 does not provide a full runtime workflow for it.

---

## Development rules

The project should continue to follow these rules:

- solve one clear problem at a time
- keep completed and planned work separate
- keep data grain explicit
- keep control state explicit
- keep historical monitoring records append-only
- use deterministic checks for pass/fail decisions
- use simple interfaces between milestones
- validate with real data where practical
- prefer small, testable changes over large framework additions
- document boundaries so the repository does not claim work that is not running yet

---

## Milestone summaries

### M1 - Project setup and source understanding

Main outputs:

- repository structure
- source data review
- source table inventory
- BigQuery naming plan
- GitHub Projects plan

### M2 - BigQuery raw layer

Created `olist_raw` in the EU location and loaded all 9 source CSV files into source-aligned tables.

### M3 - Staging planning

Documented:

- staging model purpose
- source-to-staging mapping
- column cleanup rules
- timestamp, numeric, null, and duplicate handling

### M4 - dbt staging

Implemented 9 staging views and 39 dbt tests.

```text
dbt run --select staging:   9 PASS
dbt test --select staging: 39 PASS
```

### M5 - Dimensional modeling and marts

Implemented:

- `int_order_items_agg`
- `int_order_payments_agg`
- `int_order_reviews_agg`
- 5 dimensions
- 4 facts

Important corrections included the review fact grain and representative geolocation logic.

Historical M5 validation:

```text
PASS=67 WARN=0 ERROR=0 SKIP=0 NO-OP=0 TOTAL=67
```

### M6 - Documentation cleanup

Refreshed the README, architecture, dbt docs evidence, and project roadmap after the core analytics layer was complete.

### M7 - Cloud orchestration

Implemented:

```text
Cloud Scheduler
→ Cloud Run Job
→ Dockerized dbt build
→ BigQuery
```

Both manual and Scheduler-triggered runs were validated.

### M8 - Pipeline monitoring

Added append-only monitoring tables:

```text
pipeline_runs
model_run_results
test_run_results
model_metadata_snapshots
model_column_snapshots
model_lineage_edges
```

Original M8 cloud validation recorded:

```text
21 successful models
94 passed tests
259 model/source column snapshots
146 lineage edges
```

M10 later added `control_attempt_id` to `pipeline_runs` for exact run correlation.

### M9 - Pipeline quality reviewer

Implemented rules:

```text
R001 pipeline run unsuccessful
R002 model execution non-success
R003 test result non-passing
R004 model missing from current run
R005 row-count anomaly
R006 runtime regression
```

The rule result is deterministic. Vertex AI is optional and only explains triggered findings.

Final M9 validation on 2026-08-10:

```text
179 evaluations
166 PASS
1 TRIGGERED
12 NOT_EVALUATED
53 unit tests passed
```

### M10 U1 - Window and watermark control

Implemented:

- `olist_control.pipeline_control_state`
- `olist_control.pipeline_window_events`
- explicit state bootstrap
- states `IDLE`, `RUNNING`, `FAILED`, `WAITING_RETRY`, `QUARANTINED`
- forward `[start, end)` windows
- `int_orders_windowed`
- incremental fact `MERGE`
- retry with new attempt IDs and `retry_of_attempt_id`
- control version checks
- state update + event insert in one BigQuery transaction
- `control_attempt_id` in M8 monitoring
- exact monitoring-run resolver
- M9 exact-run review in window-controlled mode

Current dbt validation:

```text
22 models
96 tests
118 / 118 PASS
```

Current Python unit tests:

```text
52 M10 controller tests
5 monitoring-run resolver tests
53 M9 reviewer tests
110 total
```

Real validation also covered:

- two successful forward windows
- workload failure without watermark advance
- two failed attempts for the same window
- successful third retry for that window
- continuous audit versions
- final clean `IDLE` state
- stale BigQuery write rejection
- rollback without a false audit event

---

## Current M10 work

### Portal

The portal should start with a small set of operational views:

```text
/overview
/reliability
/findings/[findingId]
/analytics
```

The implementation foundation is Next.js + React + TypeScript.

The UI should keep a stable layout, clear status meaning, and a simple summary-to-detail flow.

### Analytics

BigQuery remains the analytical source.

The first geospatial slice should be small:

- one state-level BigQuery aggregate
- one Brazil state map
- order count, GMV, average order value, delivery time, late-delivery rate, and review score
- current window versus previous window
- map selection linked to KPIs, trends, and detail data

CARTO and deck.gl are planned for the map layer.

---

## M11 roadmap

M11 will add historical processing and recovery.

Planned work:

- replay one window
- backfill several windows
- resume after failure
- keep writes idempotent
- compare replay and incremental results
- preserve replay audit history
- keep replay/backfill state separate from the normal forward watermark

M11 should deepen recovery behavior instead of adding unrelated platform features.

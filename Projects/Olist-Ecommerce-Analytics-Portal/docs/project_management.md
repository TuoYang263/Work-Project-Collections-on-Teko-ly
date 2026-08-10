# Project Management

## Purpose

This document describes how the Olist E-Commerce Analytics & Pipeline Monitoring Portal is planned and managed.

The goal is to keep the project structured, realistic, reviewable, and easy to continue across milestones.

This document has been refreshed after **M9 - Evidence-Grounded Pipeline Quality Reviewer** to reflect the completed implementation and the next M10/M11 roadmap.

---

## Project Approach

This project is developed milestone by milestone.

Each milestone should produce a clear and reviewable output, such as documentation, BigQuery datasets, dbt models, data quality tests, dimensional models, dbt docs, lineage screenshots, orchestration, monitoring tables, or future AI-assisted explanations.

The project should avoid doing too many things at once. Each stage should be completed, validated, documented, and committed before moving to the next stage.

The project is currently completed through M9. M10 is the next implementation milestone.

---

## Git Workflow

The current development branch is:

```text
feature/olist-analytics-portal
```

The branch is used for milestone-based development of the Olist analytics project.

The branch should not be merged into `main` until the milestone has a clean structure, reviewable commits, and no committed raw data or credentials.

---

## Commit Style

Commit messages should be short and clear.

Examples:

```text
chore: initialize Olist analytics portal structure
docs: add source data overview
docs: document BigQuery raw layer validation
feat: add dbt staging models
feat: add dimensional mart models
fix: correct fct_order_reviews grain
docs: refresh Olist project showcase for M6
```

General commit prefixes:

- `chore:` for setup and maintenance
- `docs:` for documentation
- `data:` for source metadata or sample data notes
- `feat:` for implemented project features
- `fix:` for corrections

---

## Milestone Plan

| Milestone | Focus | Status | Output |
|---|---|---:|---|
| M1 | Project setup and source understanding | Completed | Folder structure, source inventory, naming plan, GitHub Project setup |
| M2 | BigQuery raw layer | Completed | BigQuery `olist_raw` dataset and 9 source-aligned raw tables |
| M3 | Staging layer planning | Completed | Staging design, source-to-staging mapping, cleanup rules |
| M4 | dbt staging layer | Completed | dbt project setup, sources, 9 staging views, schema docs, 39 dbt tests |
| M5 | Dimensional modeling / analytics marts | Completed | Intermediate models, dimensions, facts, mart tests, dbt docs validation |
| M6 | README / dbt docs / project showcase cleanup | Completed | Portfolio-ready README, as-built architecture docs, dbt docs screenshots, roadmap cleanup |
| M7 | Google Cloud Scheduler + Cloud Run Job orchestration | Completed | Scheduled containerized dbt execution |
| M8 | dbt metadata refresh / monitoring tables | Completed | dbt artifact parsing and append-only `olist_monitoring` history |
| M9 | Evidence-grounded pipeline quality reviewer | Completed | R001-R006, historical baselines, finding package, Vertex AI explanation |
| M10 | Window control and operational / analytics portal | Planned | Watermark/state control, monitoring UI, findings, analytics, geospatial slice |
| M11 | Replay, backfill, and recovery | Planned | Controlled replay, backfill, resume, idempotency, consistency validation |

---

## Current Project Scope

The implemented project boundary is M9.

Completed capabilities now include:

- BigQuery raw, staging, intermediate, marts, and monitoring datasets
- dbt modeling, tests, documentation, and lineage
- Cloud Run Job and Cloud Scheduler orchestration
- append-only dbt artifact monitoring history
- deterministic pipeline quality rules R001-R006
- historical model inventory, row-count, and runtime comparison
- finding package generation
- optional Vertex AI explanation with strict finding-ID validation
- AI skip and failure fallback behavior

M10 and M11 are intentionally separate. M9 should not continue growing with portal, watermark, replay, or backfill logic.

---

## Development Principles

The project should follow these principles:

- keep the scope controlled
- document important design decisions
- use clear folder structure
- avoid committing raw data or credentials
- keep raw, staging, intermediate, marts, monitoring, and future AI layers conceptually separated
- prefer simple and explainable design over unnecessary complexity
- make fact and dimension grain explicit
- validate assumptions with dbt tests
- use dbt docs and lineage for transparency
- keep outputs useful for BI and analytics consumption

---

## Current Status

### Completed

- M1 - Project Setup & Source Understanding
- M2 - BigQuery Raw Layer
- M3 - Staging Layer Planning
- M4 - dbt Staging Layer
- M5 - Dimensional Modeling / Analytics Marts
- M6 - README / dbt docs / Project Showcase Cleanup
- M7 - Google Cloud Scheduler + Cloud Run Job Orchestration
- M8 - dbt Metadata Refresh & Pipeline Monitoring
- M9 - Evidence-Grounded Pipeline Quality Reviewer

### Next

- M10 - Window Control and Operational / Analytics Portal
- M11 - Replay, Backfill, and Recovery

---

## Completed Milestone Summaries

### M1 Summary

M1 established the project foundation:

- project folder structure
- README skeleton
- initial architecture documentation
- source CSV inspection
- source data overview
- source table inventory
- BigQuery dataset naming plan
- GitHub Projects board planning

### M2 Summary

The BigQuery raw dataset `olist_raw` was created in the EU location.

All 9 Olist source CSV files were loaded into source-aligned raw tables.

Raw layer validation was completed and documented in:

```text
metadata/bigquery/raw_layer_validation.md
```

### M3 Summary

Staging layer planning was completed.

Completed M3 work:

- staging layer purpose documented
- staging dataset naming documented
- source-to-staging mapping documented
- column cleanup rules documented
- timestamp, numeric, null, and duplicate handling rules included in staging cleanup rules

M3 documents:

```text
docs/staging_layer_plan.md
metadata/staging/source_to_staging_mapping.md
metadata/staging/column_cleanup_rules.md
```

### M4 Summary

The dbt staging layer was implemented.

Completed M4 work:

- dbt project initialized under `dbt/`
- BigQuery connection validated through local `profiles.yml`
- 9 raw BigQuery tables registered as dbt sources
- 9 staging views created in `olist_staging`
- staging SQL uses `source()` references, light cleaning, type casting, and standardization
- staging model and column documentation added
- 39 dbt data tests added and passed

M4 validation:

```text
dbt run --select staging
PASS=9 WARN=0 ERROR=0 SKIP=0 NO-OP=0 TOTAL=9

dbt test --select staging
PASS=39 WARN=0 ERROR=0 SKIP=0 NO-OP=0 TOTAL=39
```

M4 document:

```text
docs/m4_dbt_staging_validation.md
```

### M5 Summary

The dimensional modeling and analytics marts layer was implemented.

Completed M5 work:

- dimensional modeling design documented
- intermediate layer completed:
  - `int_order_items_agg`
  - `int_order_payments_agg`
  - `int_order_reviews_agg`
- dimension models completed:
  - `dim_customers`
  - `dim_sellers`
  - `dim_products`
  - `dim_geolocation_zip_prefix`
  - `dim_dates`
- fact models completed:
  - `fct_orders`
  - `fct_order_items`
  - `fct_order_payments`
  - `fct_order_reviews`
- mart model and column documentation added in `dbt/models/marts/core/schema.yml`
- review fact grain corrected from `review_id` to `review_id + order_id`
- generated `review_key` used as the primary key for `fct_order_reviews`
- `dim_geolocation_zip_prefix` uses median coordinates as representative coordinates and retains average coordinates
- `dim_dates` covers order, shipping, and review-related dates
- dbt docs generated and reviewed

M5 validation:

```text
dbt build --select intermediate marts
PASS=67 WARN=0 ERROR=0 SKIP=0 NO-OP=0 TOTAL=67
```

M5 documents:

```text
docs/m5_dimensional_modeling_design.md
docs/m5_dbt_marts_validation.md
```

### M6 Summary

M6 cleaned up the project showcase and documentation after the core dbt analytics layer was complete.

Main outputs:

- README refresh
- as-built architecture documentation
- dbt docs and lineage review
- roadmap cleanup

### M7 Summary

M7 moved the dbt pipeline into a scheduled Google Cloud execution path.

Implemented flow:

```text
Cloud Scheduler
→ Cloud Run Job
→ containerized dbt build
→ BigQuery
```

Manual and Scheduler-triggered executions were validated.

### M8 Summary

M8 added append-only pipeline monitoring history from dbt artifacts.

Implemented monitoring tables:

```text
pipeline_runs
model_run_results
test_run_results
model_metadata_snapshots
model_column_snapshots
model_lineage_edges
```

The validated cloud run recorded 21 successful models, 94 passed tests, 259 column snapshots, and 146 lineage edges.

### M9 Summary

M9 added the evidence-grounded pipeline quality reviewer.

Implemented rules:

```text
R001 pipeline run unsuccessful
R002 model execution non-success
R003 test result non-passing
R004 model missing from current run
R005 row-count anomaly
R006 runtime regression
```

The reviewer keeps deterministic rules as the source of truth. Vertex AI only explains triggered findings.

Final validation on 2026-08-10:

```text
179 evaluations
166 PASS
1 TRIGGERED
12 NOT_EVALUATED
53 unit tests passed
Vertex AI status: SUCCESS
```

The real triggered finding was an R006 runtime regression for `fct_order_payments`.

Detailed M9 implementation notes:

```text
docs/m9_expert_system_closing.md
```

---

## Future Roadmap

### M10 - Window Control and Operational / Analytics Portal

Main goal:

- add explicit window and watermark state
- advance the watermark only after successful processing
- preserve retry and attempt history
- expose pipeline status, run history, findings, and evidence in a usable portal
- add an analytics area backed by governed BigQuery data
- start geospatial analytics with a small Brazil state-level slice using CARTO and deck.gl

The first portal version should stay small and verifiable rather than trying to build every dashboard at once.

### M11 - Replay, Backfill, and Recovery

Main goal:

- replay one historical window
- backfill multiple windows
- resume after partial failure
- keep business writes idempotent
- keep monitoring history append-only
- validate incremental and replay consistency
- keep normal production watermark behavior separate from backfill control

M11 should focus on recovery depth rather than adding another large feature surface.

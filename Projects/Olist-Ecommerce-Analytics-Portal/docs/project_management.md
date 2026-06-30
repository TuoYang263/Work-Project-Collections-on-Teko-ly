# Project Management

## Purpose

This document describes how the Olist E-Commerce Analytics & Pipeline Monitoring Portal is planned and managed.

The goal is to keep the project structured, realistic, reviewable, and easy to continue across milestones.

This document has been refreshed during **M6 - README / dbt docs / Project Showcase Cleanup** to reflect the current completed project scope and updated roadmap.

---

## Project Approach

This project is developed milestone by milestone.

Each milestone should produce a clear and reviewable output, such as documentation, BigQuery datasets, dbt models, data quality tests, dimensional models, dbt docs, lineage screenshots, orchestration, monitoring tables, or future AI-assisted explanations.

The project should avoid doing too many things at once. Each stage should be completed, validated, documented, and committed before moving to the next stage.

Current M6 scope is intentionally documentation and project showcase cleanup only. It does not add new pipeline features.

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
| M1 | Project setup and source understanding | Completed | Folder structure, README skeleton, source inventory, naming plan, GitHub Project setup |
| M2 | BigQuery raw layer | Completed | BigQuery `olist_raw` dataset and 9 source-aligned raw tables |
| M3 | Staging layer planning | Completed | Staging design, source-to-staging mapping, cleanup rules |
| M4 | dbt staging layer | Completed | dbt project setup, sources, 9 staging views, schema docs, 39 dbt tests |
| M5 | Dimensional modeling / analytics marts | Completed | Intermediate models, dimensions, facts, mart tests, dbt docs validation |
| M6 | README / dbt docs / project showcase cleanup | In progress | Portfolio-ready README, as-built architecture docs, dbt docs screenshots, roadmap cleanup |
| M7 | Google Cloud Scheduler + Cloud Run Job orchestration | Future | Scheduled dbt execution using Google Cloud services |
| M8 | ADE-inspired metadata refresh / monitoring tables | Future | dbt artifact parsing and BigQuery `olist_monitoring` tables |
| M9 | AI-assisted pipeline intelligence layer | Future | Explanation layer on top of tests, artifacts, metadata tables, and docs |

---

## Current M6 Scope

M6 includes:

- refreshing `README.md` as a portfolio-ready project showcase
- updating `docs/architecture.md` from planned architecture to as-built architecture
- adding dbt docs and lineage screenshots under `assets/screenshots/dbt_docs/`
- highlighting the M5 dimensional model and validation summary
- documenting dbt docs / lineage review
- updating project management and roadmap documentation
- keeping future orchestration, monitoring, and AI work clearly separated from the current implemented scope

M6 does not include:

- building a Power BI dashboard
- building a React or Node portal
- adding Google Cloud Scheduler or Cloud Run orchestration
- parsing dbt artifacts into monitoring tables
- creating BigQuery `olist_monitoring` tables
- adding an AI module
- adding new dbt models or expanding the mart layer

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

### In progress

- M6 - README / dbt docs / Project Showcase Cleanup

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

---

## Future Roadmap

### M7 - Google Cloud Scheduler + Cloud Run Job orchestration

Future goal:

- schedule dbt execution using Google Cloud Scheduler and Cloud Run Jobs
- keep orchestration separate from modeling logic
- make the pipeline easier to run repeatedly

### M8 - ADE-inspired metadata refresh and monitoring tables

Future goal:

- borrow metadata-driven DataOps ideas from Agile Data Engine
- parse dbt artifacts:
  - `manifest.json`
  - `run_results.json`
  - `catalog.json`
- load dbt artifact metadata into BigQuery `olist_monitoring` tables
- track model status, test results, row counts, execution metadata, runtime metadata, and lineage metadata

This will not be a direct Agile Data Engine integration.

### M9 - AI-assisted pipeline intelligence layer

Future goal:

- build an explanation layer on top of dbt docs, dbt artifacts, monitoring tables, and validation outputs
- answer questions about pipeline health, data quality, failed tests, validation, lineage, runtime performance, and downstream impact
- help interpret metadata rather than replace structured tests

The AI layer will not replace dbt tests, dbt validation, or monitoring tables.
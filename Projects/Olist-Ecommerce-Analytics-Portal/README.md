# Olist E-Commerce Analytics & Pipeline Monitoring Portal

## Overview

This portfolio project demonstrates how raw e-commerce data can be transformed into a documented, tested, cloud-orchestrated, and monitorable analytics platform using BigQuery and dbt.

The project is built on the Olist Brazilian E-Commerce public dataset and currently includes:

- BigQuery raw, staging, intermediate, and marts layers
- dbt dimensional modeling with facts and dimensions
- dbt data tests, documentation, and lineage
- Dockerized dbt execution
- Google Cloud Run Job and Cloud Scheduler orchestration
- Append-only dbt artifact monitoring in BigQuery
- Historical model, test, metadata, column, and lineage records
- GitHub Projects milestone-based delivery

The completed implementation boundary is M8. Power BI monitoring, a custom portal, and the M9 AI-assisted pipeline reviewer remain future work.

---

## What this project demonstrates

### Analytics engineering

- Layered warehouse design
- dbt staging, intermediate, and mart models
- Dimensional modeling and star-schema principles
- Explicit fact and dimension grain
- Reusable aggregation models
- dbt tests for keys, relationships, accepted values, and business fields
- dbt docs and lineage

### Data engineering and operations

- BigQuery dataset and table design
- Dockerized dbt runtime
- Runtime-generated dbt profiles
- Artifact Registry deployment
- Cloud Run batch execution
- Cloud Scheduler OAuth triggering
- Service accounts and IAM
- Runtime environment configuration

### Pipeline monitoring

- Parsing `manifest.json`, `run_results.json`, and `catalog.json`
- Converting nested dbt artifact JSON into relational monitoring records
- Append-only run history
- Model and test execution history
- Model and column metadata snapshots
- Column documentation and test coverage
- Queryable lineage edges
- Local and cloud cross-table validation

---

## Current status

| Milestone | Status | Summary |
|---|---:|---|
| M1 - Project Setup & Source Understanding | Completed | Repository structure, source review, documentation foundation, GitHub Project board |
| M2 - BigQuery Raw Layer | Completed | 9 Olist source CSV files loaded into `olist_raw` |
| M3 - Staging Layer Planning | Completed | Staging design, naming rules, source-to-staging mapping |
| M4 - dbt Staging Layer | Completed | 9 staging views, dbt sources, documentation, 39 tests |
| M5 - Dimensional Modeling / Analytics Marts | Completed | Intermediate models, facts, dimensions, mart tests, dbt docs validation |
| M6 - README / dbt docs / Project Showcase Cleanup | Completed | Portfolio documentation and dbt docs evidence |
| M7 - Cloud Scheduler + Cloud Run Job Orchestration | Completed | Dockerized dbt build, Artifact Registry, Cloud Run Job, authenticated Scheduler trigger |
| M8 - dbt Metadata Refresh & Pipeline Monitoring | Completed | Six append-only monitoring tables, artifact parser/loader, Cloud Run and Scheduler validation |

M8 cloud end-to-end validation completed on **2026-07-15**.

---

## Tech stack

| Area | Tools |
|---|---|
| Cloud data warehouse | Google BigQuery |
| Transformation | dbt Core, dbt-bigquery |
| Modeling | Dimensional modeling, star schema, layered warehouse |
| Data quality | dbt `not_null`, `unique`, `relationships`, `accepted_values` tests |
| Metadata monitoring | dbt artifacts, Python, google-cloud-bigquery |
| Cloud orchestration | Google Cloud Run Jobs, Google Cloud Scheduler |
| Containerization | Docker, Artifact Registry |
| Security and runtime configuration | Google Cloud service accounts, IAM, environment variables, runtime-generated dbt profile |
| Documentation | Markdown, dbt docs, dbt lineage |
| Workflow | Git, GitHub, GitHub Projects, milestone-based delivery |

---

## Architecture

### Business-data path

```text
Olist CSV source data
        ↓
BigQuery raw dataset: olist_raw
        ↓
dbt staging views: olist_staging
        ↓
dbt intermediate views: olist_intermediate
        ↓
dbt marts tables: olist_marts
        ↓
dbt tests + dbt docs + lineage
        ↓
BI-ready analytics layer
```

### Scheduled execution and monitoring path

```text
Cloud Scheduler
        ↓
Cloud Run Job
        ↓
dbt debug
        ↓
dbt build --target prod
        ↓
BigQuery models and tests refreshed
        ↓
preserve build manifest.json + run_results.json
        ↓
dbt docs generate --target prod
        ↓
keep catalog.json and restore build artifacts
        ↓
Python artifact parser and BigQuery loader
        ↓
BigQuery dataset: olist_monitoring
```

The project therefore contains two analytical outputs:

```text
olist_marts
→ e-commerce business analytics
```

```text
olist_monitoring
→ pipeline health, tests, metadata, schema, documentation, and lineage analytics
```

Detailed architecture:

- [`docs/architecture.md`](docs/architecture.md)
- [`docs/metadata_refresh.md`](docs/metadata_refresh.md)
- [`docs/orchestration.md`](docs/orchestration.md)
- [`docs/gcp_orchestration_commands.md`](docs/gcp_orchestration_commands.md)

---

## Warehouse layers

| Layer | Purpose | Example objects |
|---|---|---|
| Raw | Preserve source-aligned data | `raw_orders`, `raw_order_items`, `raw_products` |
| Staging | Clean, rename, cast, and standardize | `stg_orders`, `stg_order_items`, `stg_products` |
| Intermediate | Reusable business aggregations | `int_order_items_agg`, `int_order_payments_agg`, `int_order_reviews_agg` |
| Marts | BI-ready dimensional models | `fct_orders`, `dim_customers`, `dim_products` |
| Monitoring | Append-only pipeline metadata history | `pipeline_runs`, `test_run_results`, `model_lineage_edges` |

---

## Source and raw layer

The project uses 9 Olist source files:

- customers
- geolocation
- orders
- order items
- order payments
- order reviews
- products
- sellers
- product category translation

BigQuery raw tables:

- `raw_customers`
- `raw_geolocation`
- `raw_orders`
- `raw_order_items`
- `raw_order_payments`
- `raw_order_reviews`
- `raw_products`
- `raw_sellers`
- `raw_product_category_translation`

The local source files are ignored by Git.

---

## dbt model layers

### Staging models

The staging layer contains 9 source-aligned BigQuery views:

- `stg_customers`
- `stg_geolocation`
- `stg_orders`
- `stg_order_items`
- `stg_order_payments`
- `stg_order_reviews`
- `stg_products`
- `stg_sellers`
- `stg_product_category_translation`

Responsibilities:

- light cleaning
- type casting
- naming standardization
- stable source-aligned interfaces
- source-assumption tests

### Intermediate models

- `int_order_items_agg`
- `int_order_payments_agg`
- `int_order_reviews_agg`

These models centralize reusable order-level aggregations and keep mart SQL smaller and easier to validate.

### Mart models

Dimensions:

- `dim_customers`
- `dim_sellers`
- `dim_products`
- `dim_geolocation_zip_prefix`
- `dim_dates`

Facts:

- `fct_orders`
- `fct_order_items`
- `fct_order_payments`
- `fct_order_reviews`

---

## Dimensional model

### Dimensions

| Model | Grain | Primary key | Purpose |
|---|---|---|---|
| `dim_customers` | One row per customer | `customer_id` | Customer attributes and location |
| `dim_sellers` | One row per seller | `seller_id` | Seller attributes and location |
| `dim_products` | One row per product | `product_id` | Product attributes and translated category |
| `dim_geolocation_zip_prefix` | One row per zip prefix | `geolocation_zip_code_prefix` | Representative geographic coordinates |
| `dim_dates` | One row per calendar date | `date_day` | Shared reporting date dimension |

### Facts

| Model | Grain | Primary key | Purpose |
|---|---|---|---|
| `fct_orders` | One row per order | `order_id` | Order lifecycle and summary metrics |
| `fct_order_items` | One row per order item | `order_item_key` | Item sales, product, seller, price, and freight |
| `fct_order_payments` | One row per payment sequence | `order_payment_key` | Payment method, installments, and value |
| `fct_order_reviews` | One row per review and order | `review_key` | Review score and timing |

---

## Key modeling decisions

### Review grain correction

`review_id` is not unique in the source.

`fct_order_reviews` uses the true source grain:

```text
one row per review_id + order_id
```

A generated `review_key` is used as the primary key.

### Representative geolocation

The raw geolocation data contains multiple coordinate records per zip-code prefix.

`dim_geolocation_zip_prefix` uses median latitude and longitude to reduce outlier impact and retains average values for transparency.

### Shared date dimension

`dim_dates` is generated from order, shipping, and review dates and provides a consistent reporting date layer for facts.

---

## Data quality and validation

The project uses dbt tests as deterministic quality controls.

Coverage includes:

- primary-key `not_null` and `unique`
- fact-to-dimension `relationships`
- accepted business values
- important key and measure checks

### Staging validation

```text
dbt run --select staging
PASS=9 WARN=0 ERROR=0 SKIP=0 NO-OP=0 TOTAL=9

dbt test --select staging
PASS=39 WARN=0 ERROR=0 SKIP=0 NO-OP=0 TOTAL=39
```

### Intermediate and marts validation

```text
dbt build --select intermediate marts
PASS=67 WARN=0 ERROR=0 SKIP=0 NO-OP=0 TOTAL=67
```

### Full cloud build validation

```text
PASS=115 WARN=0 ERROR=0 SKIP=0 NO-OP=0 TOTAL=115
```

The full build contains:

```text
21 model executions
94 test executions
```

---

## M7 cloud orchestration

M7 implemented:

```text
Artifact Registry repository: olist-dbt-jobs
Cloud Run Job: olist-dbt-build-job
Cloud Run region: europe-north1
Cloud Scheduler job: olist-dbt-daily-trigger
Scheduler location: europe-west1
Schedule: 0 6 * * *
Time zone: Europe/Helsinki
```

The Scheduler invokes the Cloud Run Admin API with an OAuth token issued for the scheduler service account.

Both manual and Scheduler-triggered executions were validated.

---

## M8 artifact monitoring

M8 implements an append-only dbt artifact monitoring layer.

### Artifact sources

| Artifact | Role |
|---|---|
| `manifest.json` | Models, tests, sources, metadata, columns, dependencies |
| `run_results.json` | Build execution status, runtime, failures, invocation details |
| `catalog.json` | Warehouse relation and column metadata, row count and bytes where available |

`manifest.json` and `run_results.json` are preserved from `dbt build`.

`catalog.json` is generated by `dbt docs generate`.

### Monitoring tables

| Table | Grain |
|---|---|
| `pipeline_runs` | One row per monitoring run |
| `model_run_results` | One row per model execution per run |
| `test_run_results` | One row per test execution per run |
| `model_metadata_snapshots` | One row per model per run |
| `model_column_snapshots` | One row per model/source column per run |
| `model_lineage_edges` | One row per dependency edge per run |

### Validated production monitoring run

```text
job_name                      olist-dbt-build-job
environment                   prod
pipeline_status               success

model_run_results             21
successful_models             21

test_run_results              94
passed_tests                  94
non_passing_tests              0

model_metadata_snapshots      21
model_column_snapshots       259
model_lineage_edges          146
```

### Validated Scheduler execution

```text
execution: olist-dbt-build-job-f59xf
tasks: 1 / 1 completed successfully
triggered by: olist-scheduler-invoker
```

The latest-run validation query confirms that all six tables share the same `monitoring_run_id` and contain a complete record set.

---

## dbt docs and visual lineage

dbt docs provide:

- model descriptions
- column descriptions
- test visibility
- upstream and downstream relationships
- lineage graphs

Useful commands:

```bash
cd dbt
dbt docs generate
dbt docs serve --port 8081
```

### dbt docs overview

![dbt docs project overview](assets/screenshots/dbt_docs/dbt_docs_project_overview.jpg)

### `fct_orders` lineage

![fct_orders lineage](assets/screenshots/dbt_docs/fct_orders_lineage.jpg)

### `fct_order_reviews` lineage

![fct_order_reviews lineage](assets/screenshots/dbt_docs/fct_order_reviews_lineage.jpg)

### Model documentation and tests

![marts tests overview](assets/screenshots/dbt_docs/marts_tests_overview.png)

M8 complements the visual dbt docs graph with historical, queryable lineage edges in BigQuery.

---

## Business analysis enabled by the marts

The mart layer supports:

- order volume and status trends
- revenue and freight analysis
- product category performance
- seller performance
- customer geography
- delivery timing
- payment method and installment behavior
- review score and customer satisfaction

---

## Pipeline analysis enabled by M8

The monitoring layer supports:

- run status and duration history
- model runtime and status trends
- failed-test history
- row-count and table-size comparisons where statistics exist
- column documentation coverage
- column test coverage
- source-to-model and model-to-model lineage
- downstream impact analysis foundation
- evidence for the future M9 reviewer

---

## Repository structure

```text
.
├── assets/
├── bi/
├── data/
├── dbt/
│   ├── Dockerfile
│   ├── dbt_project.yml
│   ├── profiles.yml.template
│   ├── run_dbt_job.sh
│   ├── models/
│   ├── monitoring/
│   │   ├── artifact_parser.py
│   │   ├── inspect_artifacts.py
│   │   └── load_artifacts_to_bigquery.py
│   └── sql/monitoring/
│       ├── create_olist_monitoring_dataset.sql
│       ├── create_monitoring_tables.sql
│       ├── validate_monitoring_tables.sql
│       └── validate_latest_monitoring_run.sql
├── docs/
├── metadata/
├── portal/
└── sql/
```

Important documentation:

```text
docs/architecture.md
docs/metadata_refresh.md
docs/gcp_orchestration_commands.md
docs/orchestration.md
docs/source_data_overview.md
docs/m4_dbt_staging_validation.md
docs/m5_dimensional_modeling_design.md
docs/m5_dbt_marts_validation.md
```

---

## Local dbt usage

The local `profiles.yml` is not committed.

From the project root:

```bash
cd dbt
```

Connection validation:

```bash
dbt debug
```

Staging:

```bash
dbt run --select staging
dbt test --select staging
```

Intermediate and marts:

```bash
dbt build --select intermediate marts
```

Docs:

```bash
dbt docs generate
dbt docs serve --port 8081
```

Monitoring commands and complete deployment instructions are documented in:

- [`docs/gcp_orchestration_commands.md`](docs/gcp_orchestration_commands.md)

---

## Project workflow

Development branch:

```text
feature/olist-analytics-portal
```

The workflow emphasizes:

- small milestones
- clear acceptance criteria
- controlled scope
- small reviewable commits
- validation before progression
- documentation of commands and design decisions
- separation between deterministic engineering and future AI features

---

## Future work

### Monitoring analytics presentation

Use `olist_monitoring` as a source for a lightweight pipeline-health analytics presentation.

This remains separate from the completed M8 ingestion layer.

### M9 - AI-assisted pipeline quality reviewer

M9 will use M8 monitoring tables, dbt artifacts, and documentation as evidence.

Planned capabilities:

- production-readiness review
- failed-test and anomaly explanation
- missing-test and documentation-gap detection
- lineage and downstream-impact reasoning
- evidence-only structured output
- deterministic fallback behavior

The AI layer will not replace dbt tests, SQL validation, monitoring tables, or engineering judgment.

---

## Project positioning

The main value of the project is the engineering process, not only the final dashboard or tables:

```text
understand the source
→ design layered warehouse models
→ define correct grain
→ validate with dbt tests
→ document lineage
→ automate cloud execution
→ preserve pipeline metadata history
→ build future intelligence on validated evidence
```

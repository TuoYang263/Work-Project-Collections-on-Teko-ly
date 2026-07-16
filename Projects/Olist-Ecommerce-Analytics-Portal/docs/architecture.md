# Architecture

## Purpose

This document describes the implemented architecture of the Olist E-Commerce Analytics & Pipeline Monitoring Portal.

The completed architecture now includes:

- BigQuery raw, staging, intermediate, marts, and monitoring datasets
- dbt transformations, tests, documentation, and lineage
- Dockerized dbt execution
- Google Cloud Run Job and Cloud Scheduler orchestration
- dbt artifact parsing and append-only pipeline monitoring history

The current completed implementation boundary is M8. M9 AI-assisted pipeline review remains future work.

---

## Architecture status

```text
M1 - Project Setup & Source Understanding: completed
M2 - BigQuery Raw Layer: completed
M3 - Staging Layer Planning: completed
M4 - dbt Staging Layer: completed
M5 - Dimensional Modeling / Analytics Marts: completed
M6 - README / dbt docs / Project Showcase Cleanup: completed
M7 - Google Cloud Scheduler + Cloud Run Job Orchestration: completed
M8 - dbt Metadata Refresh & Pipeline Monitoring: completed
```

M8 cloud validation completed on 2026-07-15.

---

## End-to-end architecture

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

The scheduled operational flow is:

```text
Cloud Scheduler
        ↓ authenticated OAuth request
Cloud Run Job
        ↓
Containerized dbt project
        ↓
dbt debug --target prod
        ↓
dbt build --target prod
        ↓
BigQuery staging, intermediate, and marts refreshed
        ↓
preserve build manifest.json and run_results.json
        ↓
dbt docs generate --target prod
        ↓
keep catalog.json and restore build artifacts
        ↓
Python artifact parser and BigQuery loader
        ↓
BigQuery dataset: olist_monitoring
        ↓
Monitoring analytics and future M9 evidence layer
```

---

## Source data

The project uses the Olist Brazilian E-Commerce public dataset.

Nine source CSV files are used:

- customers
- geolocation
- orders
- order items
- order payments
- order reviews
- products
- sellers
- product category translation

The source files are stored locally during development and are not committed to Git.

Source documentation:

```text
docs/source_data_overview.md
metadata/source/source_tables_inventory.md
```

---

## BigQuery datasets

| Dataset | Purpose |
|---|---|
| `olist_raw` | Source-aligned raw tables |
| `olist_staging` | Cleaned and standardized dbt views |
| `olist_intermediate` | Reusable transformation views |
| `olist_marts` | BI-ready fact and dimension tables |
| `olist_monitoring` | Append-only dbt pipeline monitoring history |

All datasets use the EU BigQuery location.

---

## Raw layer

Dataset:

```text
olist_raw
```

Tables:

- `raw_customers`
- `raw_geolocation`
- `raw_orders`
- `raw_order_items`
- `raw_order_payments`
- `raw_order_reviews`
- `raw_products`
- `raw_sellers`
- `raw_product_category_translation`

The raw layer stays close to the source structure and provides a stable input for dbt.

---

## dbt transformation architecture

The dbt project is located under:

```text
dbt/
```

Main model directories:

```text
dbt/models/staging/
dbt/models/intermediate/
dbt/models/marts/core/
```

Transformation flow:

```text
source()
    ↓
staging models
    ↓
intermediate models
    ↓
mart models
```

Each layer has a distinct responsibility.

---

## Staging layer

Dataset:

```text
olist_staging
```

Materialization:

```text
views
```

Models:

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

- reference raw tables through dbt `source()`
- standardize names and data types
- apply light cleaning
- expose stable source-aligned views
- document important fields
- validate source assumptions with dbt tests

Validated result:

```text
dbt run --select staging
PASS=9 WARN=0 ERROR=0 SKIP=0 NO-OP=0 TOTAL=9

dbt test --select staging
PASS=39 WARN=0 ERROR=0 SKIP=0 NO-OP=0 TOTAL=39
```

---

## Intermediate layer

Dataset:

```text
olist_intermediate
```

Models:

- `int_order_items_agg`
- `int_order_payments_agg`
- `int_order_reviews_agg`

Responsibilities:

- centralize reusable aggregation logic
- prepare order-level measures
- keep mart SQL readable
- reduce repeated joins and aggregations

---

## Marts layer

Dataset:

```text
olist_marts
```

The marts layer follows a star-schema style dimensional design.

### Dimensions

| Model | Grain | Primary key | Purpose |
|---|---|---|---|
| `dim_customers` | One row per customer | `customer_id` | Customer attributes and location |
| `dim_sellers` | One row per seller | `seller_id` | Seller attributes and location |
| `dim_products` | One row per product | `product_id` | Product attributes and translated category |
| `dim_geolocation_zip_prefix` | One row per zip prefix | `geolocation_zip_code_prefix` | Representative coordinates |
| `dim_dates` | One row per calendar date | `date_day` | Shared reporting date dimension |

### Facts

| Model | Grain | Primary key | Purpose |
|---|---|---|---|
| `fct_orders` | One row per order | `order_id` | Order lifecycle and summary measures |
| `fct_order_items` | One row per order item | `order_item_key` | Item-level sales, product, seller, and freight |
| `fct_order_payments` | One row per payment sequence | `order_payment_key` | Payment method, installments, and value |
| `fct_order_reviews` | One row per review and order | `review_key` | Review score and timing |

Validated M5 result:

```text
dbt build --select intermediate marts
PASS=67 WARN=0 ERROR=0 SKIP=0 NO-OP=0 TOTAL=67
```

---

## Key modeling decisions

### Review fact grain

`review_id` is not unique in the source data.

`fct_order_reviews` therefore uses the true source grain:

```text
one row per review_id + order_id
```

A generated `review_key` is used as the primary key.

### Geolocation coordinates

The raw geolocation table contains multiple coordinates per zip-code prefix.

`dim_geolocation_zip_prefix` uses median latitude and longitude as representative coordinates and retains averages for transparency.

The coordinates are intended for approximate geographic analysis, not routing.

### Shared date dimension

`dim_dates` is generated from order, shipping, and review dates and supports consistent date relationships across facts.

---

## Data quality architecture

Data quality is implemented with dbt tests.

Test categories include:

- `not_null`
- `unique`
- `relationships`
- `accepted_values`
- important business-field checks

The full scheduled dbt build validated in M7 and M8 contains:

```text
21 model executions
94 test executions
PASS=115 WARN=0 ERROR=0 SKIP=0 NO-OP=0 TOTAL=115
```

Tests remain the deterministic data-quality control layer. M8 stores their historical outcomes; M9 will not replace them.

---

## Documentation and lineage

dbt docs provide:

- model and column descriptions
- data-test visibility
- upstream and downstream dependencies
- lineage graphs
- `Depends On` and `Referenced By` relationships

Useful local commands:

```bash
cd dbt
dbt docs generate
dbt docs serve --port 8081
```

M8 additionally converts dbt lineage metadata into queryable BigQuery edges.

---

## M7 orchestration architecture

M7 introduced lightweight cloud orchestration.

```text
Cloud Scheduler
    ↓
Cloud Run Job
    ↓
Containerized dbt project
    ↓
dbt build --target prod
    ↓
BigQuery transformation datasets
```

Implemented components:

```text
Artifact Registry repository: olist-dbt-jobs
Cloud Run Job: olist-dbt-build-job
Cloud Run region: europe-north1
Cloud Scheduler job: olist-dbt-daily-trigger
Cloud Scheduler location: europe-west1
Schedule: 0 6 * * *
Time zone: Europe/Helsinki
```

Cloud Scheduler calls the Cloud Run Admin API through an authenticated OAuth request using the scheduler service account.

M7 validation confirmed:

- manual Cloud Run execution
- Scheduler-triggered Cloud Run execution
- containerized dbt build
- BigQuery model refresh
- `PASS=115`

---

## M8 monitoring architecture

M8 extends the same Cloud Run Job after the dbt build.

### Artifact preservation flow

```text
dbt build
→ manifest.json + run_results.json
→ temporary backup
→ dbt docs generate
→ catalog.json
→ restore build manifest.json + run_results.json
→ monitoring ingestion
```

This is replacement and restoration, not JSON merging.

### Parser and loader

```text
dbt/monitoring/artifact_parser.py
dbt/monitoring/load_artifacts_to_bigquery.py
```

The parser:

- reads the three dbt artifacts
- joins objects through `unique_id`
- normalizes status and metadata
- flattens nested JSON into six record collections

The loader:

- uses the Google Cloud BigQuery client
- appends one complete monitoring run to `olist_monitoring`
- keeps parser and database-loading responsibilities separate

### Monitoring dataset

```text
olist_monitoring
```

Tables:

| Table | Grain |
|---|---|
| `pipeline_runs` | One row per monitoring run |
| `model_run_results` | One row per model execution per run |
| `test_run_results` | One row per test execution per run |
| `model_metadata_snapshots` | One row per model per run |
| `model_column_snapshots` | One row per model/source column per run |
| `model_lineage_edges` | One row per direct dependency edge per run |

The tables are append-only, partitioned by ingestion date, and clustered for common monitoring queries.

### Runtime identity

Cloud Run writes:

```text
job_name=olist-dbt-build-job
environment=prod
monitoring dataset=olist_monitoring
```

Local development defaults remain:

```text
job_name=local-dbt-artifact-inspection
environment=dev
```

### Validated M8 output

Cloud Run and Cloud Scheduler validation completed on 2026-07-15.

```text
pipeline_runs                  1
model_run_results             21
test_run_results              94
model_metadata_snapshots      21
model_column_snapshots       259
model_lineage_edges          146

successful_models             21
passed_tests                  94
non_passing_tests              0
```

Validated Scheduler execution:

```text
olist-dbt-build-job-f59xf
1 / 1 task completed successfully
triggered by olist-scheduler-invoker
```

---

## Monitoring data flow

The monitoring architecture supports a second analytics path alongside the business marts.

```text
Business analytics path
olist_raw → olist_staging → olist_intermediate → olist_marts
```

```text
Pipeline analytics path
dbt artifacts → artifact parser → olist_monitoring
```

`olist_marts` supports e-commerce business analysis.

`olist_monitoring` supports pipeline health, data quality, runtime, schema, documentation, and lineage analysis.

---

## Security and runtime configuration

The Cloud Run Job uses:

```text
Service account: olist-dbt-runner
```

The Scheduler uses:

```text
Service account: olist-scheduler-invoker
```

Credentials are not committed to Git.

The dbt profile is generated at runtime from environment variables.

Important runtime variables include:

```text
DBT_PROJECT_ID
DBT_DATASET
DBT_LOCATION
DBT_THREADS
DBT_TARGET
GCP_PROJECT_ID
DBT_ARTIFACT_DIR
MONITORING_DATASET_ID
MONITORING_JOB_NAME
MONITORING_ENVIRONMENT
```

---

## Container architecture

The Docker image uses:

```text
python:3.11-slim
dbt-bigquery
google-cloud-bigquery
```

It copies the dbt project and monitoring scripts into:

```text
/app/dbt
```

The entrypoint is:

```text
/app/dbt/run_dbt_job.sh
```

The M8 production image tag is:

```text
olist-dbt-job:m8
```

Shell scripts are normalized to LF through `.gitattributes`, and Python cache files are excluded through `.dockerignore`.

---

## Project workflow

Development branch:

```text
feature/olist-analytics-portal
```

The project follows milestone-based delivery with:

- controlled scope
- explicit acceptance criteria
- validation before progression
- small commits
- architecture and command documentation
- separation between implemented and future capabilities

---

## Current implementation boundary

Completed through M8:

- source inventory and raw loading
- staging, intermediate, and mart dbt models
- dimensional modeling
- dbt tests and documentation
- Dockerized dbt runtime
- Artifact Registry deployment
- Cloud Run Job orchestration
- Cloud Scheduler triggering
- append-only BigQuery monitoring tables
- dbt artifact ingestion
- model/test execution history
- metadata and column snapshots
- lineage edges
- local and cloud end-to-end validation

Not yet implemented:

- Power BI monitoring dashboard
- React portal
- alerting and notification delivery
- automated root-cause analysis
- AI-assisted pipeline reviewer

---

## Future architecture direction

### M9 - AI-assisted pipeline quality reviewer

M9 will build on the deterministic M8 evidence layer.

Planned responsibilities:

- evaluate pipeline correctness and production readiness
- explain failed tests and anomalies
- identify documentation and test gaps
- reason about lineage and downstream impact
- compare specifications with implementation evidence
- return strict evidence-based output

M9 will not replace:

- dbt tests
- BigQuery monitoring tables
- deterministic validation queries
- human engineering judgment

---

## Design principles

- Keep source, staging, intermediate, mart, and monitoring layers separated.
- Preserve raw data as source-aligned as possible.
- Make transformation grain explicit.
- Use dbt tests for repeatable quality validation.
- Preserve monitoring history instead of overwriting the latest state.
- Keep orchestration separate from modeling logic.
- Keep parser logic separate from database loading.
- Use runtime configuration instead of project-specific credentials in code.
- Keep M8 deterministic and explainable.
- Build AI capabilities only on top of validated evidence.

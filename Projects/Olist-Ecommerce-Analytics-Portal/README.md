# Olist E-Commerce Analytics & Pipeline Monitoring Portal

## Overview

This project is a portfolio analytics engineering project built on the Olist Brazilian E-Commerce public dataset.

The goal is to demonstrate how raw e-commerce data can be transformed into a documented, tested, and BI-ready dimensional model using BigQuery and dbt.

The current implementation focuses on the warehouse and transformation foundation:

- BigQuery raw layer for source-aligned data
- dbt staging layer for cleaning and standardization
- dbt intermediate layer for reusable aggregations
- dbt marts layer with fact and dimension tables
- dbt tests for primary keys, relationships, accepted values, and important business fields
- dbt docs and lineage for model documentation and impact understanding
- GitHub Projects milestone workflow for enterprise-style project tracking
- Dockerized dbt execution for cloud deployment
- Google Cloud Run Job and Cloud Scheduler orchestration

The project title includes pipeline monitoring and portal direction. The current completed scope covers the analytics engineering foundation and a lightweight cloud orchestration layer. Monitoring tables, Power BI reporting, React portal integration, and AI-assisted pipeline intelligence are planned as later milestones.

---

## What this project demonstrates

This project is designed to show practical data engineering and analytics engineering skills rather than only dashboard building.

It demonstrates:

- Designing a layered data warehouse structure
- Loading and organizing raw source data in BigQuery
- Building dbt models with clear separation between staging, intermediate, and marts layers
- Applying dimensional modeling and star schema principles
- Defining fact and dimension table grain explicitly
- Writing dbt data tests for data quality and relationship validation
- Generating dbt documentation and lineage
- Documenting decisions, validation results, and milestone progress
- Managing project work through GitHub Projects and milestone-based delivery
- Containerizing dbt workflows for cloud execution
- Orchestrating scheduled dbt builds with Cloud Run Jobs and Cloud Scheduler

---

## Current status

| Milestone | Status | Summary |
|---|---:|---|
| M1 - Project Setup & Source Understanding | Completed | Repository structure, source data review, documentation foundation, GitHub Project board |
| M2 - BigQuery Raw Layer | Completed | 9 Olist source CSV files loaded into BigQuery `olist_raw` |
| M3 - Staging Layer Planning | Completed | Staging design, naming rules, source-to-staging mapping |
| M4 - dbt Staging Layer | Completed | 9 staging views, dbt sources, documentation, 39 dbt tests |
| M5 - Dimensional Modeling / Analytics Marts | Completed | Intermediate models, dimensions, facts, mart tests, dbt docs validation |
| M6 - README / dbt docs / Project Showcase Cleanup | Completed | Portfolio-ready README, architecture docs, dbt docs screenshots, project presentation cleanup |
| M7 - Google Cloud Scheduler + Cloud Run Job Orchestration | Completed | Dockerized dbt execution, Artifact Registry image, Cloud Run Job, Cloud Scheduler trigger, orchestration validation |

---

## Tech stack

| Area | Tools |
|---|---|
| Cloud data warehouse | Google BigQuery |
| Transformation | dbt Core, dbt-bigquery |
| Modeling approach | Dimensional modeling, star schema, layered warehouse design |
| Data quality | dbt tests: `not_null`, `unique`, `relationships`, `accepted_values` |
| Cloud orchestration | Google Cloud Run Jobs, Google Cloud Scheduler |
| Containerization / deployment | Docker, Artifact Registry |
| Security / runtime configuration | Google Cloud service accounts, IAM, runtime-generated dbt profile |
| Documentation | Markdown docs, dbt docs, dbt lineage graph |
| Project workflow | Git, GitHub, GitHub Projects, milestone-based delivery |

---

## Architecture

The implemented data flow is:

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

Layer responsibilities:

| Layer | Purpose | Example objects |
|---|---|---|
| Raw | Preserve source-aligned data loaded from CSV files | `raw_orders`, `raw_order_items`, `raw_products` |
| Staging | Clean, rename, cast, and standardize source data | `stg_orders`, `stg_order_items`, `stg_products` |
| Intermediate | Create reusable business aggregations | `int_order_items_agg`, `int_order_payments_agg`, `int_order_reviews_agg` |
| Marts | Provide BI-ready dimensional models | `fct_orders`, `dim_customers`, `dim_products` |
| Documentation / quality | Validate and explain the data model | dbt tests, dbt docs, lineage graph |

### M7 cloud orchestration flow

M7 adds a lightweight cloud orchestration layer for scheduled dbt execution.

```text
Cloud Scheduler
        ↓
Cloud Run Job
        ↓
Containerized dbt project
        ↓
dbt build --target prod
        ↓
BigQuery staging, intermediate, and marts datasets
```

The dbt project is packaged into a Docker image and stored in Artifact Registry. Cloud Run Job runs the containerized dbt pipeline as a batch job, while Cloud Scheduler triggers the job through an authenticated HTTP request.

The orchestration layer is intentionally separated from modeling logic. dbt continues to own transformations, tests, documentation, and lineage. Cloud Scheduler and Cloud Run Job only provide scheduled cloud execution.

More details are documented in:

- [`docs/architecture.md`](docs/architecture.md)
- [`docs/source_data_overview.md`](docs/source_data_overview.md)
- [`docs/staging_layer_plan.md`](docs/staging_layer_plan.md)
- [`docs/m5_dimensional_modeling_design.md`](docs/m5_dimensional_modeling_design.md)
- [`docs/m5_dbt_marts_validation.md`](docs/m5_dbt_marts_validation.md)
- [`docs/orchestration.md`](docs/orchestration.md)
- [`docs/gcp_orchestration_commands.md`](docs/gcp_orchestration_commands.md)

---

## dbt model layers

### Staging models

The staging layer contains 9 source-aligned dbt models built as BigQuery views:

- `stg_customers`
- `stg_geolocation`
- `stg_orders`
- `stg_order_items`
- `stg_order_payments`
- `stg_order_reviews`
- `stg_products`
- `stg_sellers`
- `stg_product_category_translation`

The staging layer is responsible for light cleaning, type casting, column standardization, and preparing source data for downstream modeling.

### Intermediate models

The intermediate layer contains reusable order-level aggregation models:

- `int_order_items_agg`
- `int_order_payments_agg`
- `int_order_reviews_agg`

These models keep repeated aggregation logic out of the final fact tables and make the mart layer easier to read and validate.

### Mart models

The marts layer contains BI-ready fact and dimension tables.

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

The mart layer follows a star schema style design. Each model has an explicit grain and primary key.

### Dimensions

| Model | Grain | Primary key | Purpose |
|---|---|---|---|
| `dim_customers` | One row per customer | `customer_id` | Customer location and customer-level attributes |
| `dim_sellers` | One row per seller | `seller_id` | Seller location and seller-level attributes |
| `dim_products` | One row per product | `product_id` | Product attributes and translated product category |
| `dim_geolocation_zip_prefix` | One row per zip code prefix | `geolocation_zip_code_prefix` | Representative geographic coordinates by zip prefix |
| `dim_dates` | One row per calendar date | `date_day` | Shared date dimension for order, shipping, and review dates |

### Facts

| Model | Grain | Primary key | Purpose |
|---|---|---|---|
| `fct_orders` | One row per order | `order_id` | Order lifecycle, delivery, payment, and review summary metrics |
| `fct_order_items` | One row per order item | `order_item_key` | Item-level sales, product, seller, price, and freight analysis |
| `fct_order_payments` | One row per order payment sequence | `order_payment_key` | Payment type, installments, and payment value analysis |
| `fct_order_reviews` | One row per review and order | `review_key` | Review score and review timing analysis |

---

## Key modeling decisions

### Review fact grain correction

During M5 validation, `review_id` was found not to be unique in the source dataset.

To model the true source grain correctly, `fct_order_reviews` was changed to one row per `review_id + order_id`. A generated `review_key` is used as the primary key.

This is an important modeling decision because it avoids forcing the source data into an incorrect grain just to satisfy a test.

### Geolocation representative coordinates

The raw geolocation table contains multiple latitude and longitude records per zip code prefix.

`dim_geolocation_zip_prefix` uses median latitude and longitude as representative coordinates to reduce the impact of outliers. Average latitude and longitude are also retained for transparency.

The coordinates are intended for approximate geographic analysis, not precise routing.

### Date dimension coverage

`dim_dates` is generated from order, shipping, and review-related dates used across the mart layer.

This supports relationship tests from fact tables to the shared date dimension and makes date-based reporting more consistent.

---

## Data quality and validation

The project uses dbt tests to validate model assumptions and analytical relationships.

Test coverage includes:

- Primary key `not_null` and `unique` tests
- Foreign key relationship tests between facts and dimensions
- Accepted values tests for fields such as order status, payment type, and review score
- Not null tests for important business keys and measures

### M4 staging validation

```text
dbt run --select staging
PASS=9 WARN=0 ERROR=0 SKIP=0 NO-OP=0 TOTAL=9

dbt test --select staging
PASS=39 WARN=0 ERROR=0 SKIP=0 NO-OP=0 TOTAL=39
```

### M5 marts validation

```text
dbt build --select intermediate marts
PASS=67 WARN=0 ERROR=0 SKIP=0 NO-OP=0 TOTAL=67
```

The M5 build included:

- 3 intermediate view models
- 9 mart table models
- 55 data tests

Validation details are documented in:

- [`docs/m4_dbt_staging_validation.md`](docs/m4_dbt_staging_validation.md)
- [`docs/m5_dbt_marts_validation.md`](docs/m5_dbt_marts_validation.md)

### M7 cloud orchestration validation

M7 validated that the dbt pipeline can run as a containerized Cloud Run Job and be triggered by Cloud Scheduler.

Validated orchestration flow:

```text
Cloud Scheduler force-run
    ↓
Cloud Run Job execution
    ↓
Containerized dbt build
    ↓
BigQuery staging, intermediate, and marts models refreshed
```

Validation result:

```text
Cloud Run Job: olist-dbt-build-job
Cloud Run Job region: europe-north1
Cloud Scheduler job: olist-dbt-daily-trigger
Cloud Scheduler location: europe-west1
Schedule: 0 6 * * *
Time zone: Europe/Helsinki
Manual Cloud Run Job execution: succeeded
Scheduler-triggered Cloud Run Job execution: succeeded
dbt result: PASS=115 WARN=0 ERROR=0 SKIP=0 NO-OP=0 TOTAL=115
```

Deployment commands and validation details are documented in:

- [`docs/gcp_orchestration_commands.md`](docs/gcp_orchestration_commands.md)

---

## dbt docs and lineage

dbt docs are used to review:

- Model descriptions
- Column descriptions
- Data tests
- Model dependencies
- Upstream and downstream lineage
- `Depends On` and `Referenced By` relationships

Useful commands:

```bash
cd dbt
dbt docs generate
dbt docs serve --port 8081
```

### dbt docs project overview

The dbt docs overview shows the implemented project structure, including staging, intermediate, and marts models.

![dbt docs project overview](assets/screenshots/dbt_docs/dbt_docs_project_overview.jpg)

### `fct_orders` lineage

The `fct_orders` lineage graph shows how raw order items, payments, reviews, and orders flow through staging and intermediate models before being joined into the order-level fact table.

![fct_orders lineage](assets/screenshots/dbt_docs/fct_orders_lineage.jpg)

### `fct_order_reviews` lineage

The `fct_order_reviews` lineage graph supports the M5 review grain correction. The model uses both order review data and order context, and is modeled at one row per `review_id + order_id`.

![fct_order_reviews lineage](assets/screenshots/dbt_docs/fct_order_reviews_lineage.jpg)

### Model documentation and tests

The dbt model detail view shows model descriptions, column documentation, data tests, upstream dependencies, and downstream references.

![marts tests overview](assets/screenshots/dbt_docs/marts_tests_overview.png)

---

## Business analysis enabled by the mart layer

The dimensional model supports analysis such as:

- Order volume and order status trends
- Revenue and freight analysis
- Product category performance
- Seller performance
- Customer geography analysis
- Delivery and shipping timing analysis
- Payment method and installment analysis
- Review score and customer satisfaction analysis

The mart layer is intentionally designed to be consumed by BI tools or downstream analytical applications.

---

## Repository structure

```text
.
├── assets/                     # Images, diagrams, and dbt docs screenshots
├── bi/                         # Optional future BI files and notes
├── data/                       # Local data folder; raw data is ignored by Git
├── dbt/                        # dbt project and Cloud Run Job runtime files
│   ├── dbt_project.yml
│   ├── Dockerfile              # Container image definition for Cloud Run Job
│   ├── profiles.yml.template   # Runtime-generated dbt profile template
│   ├── run_dbt_job.sh          # Cloud Run Job entrypoint script
│   └── models/
├── docs/                       # Architecture, validation, and modeling docs
├── metadata/                   # Source, BigQuery, staging, and project planning metadata
├── portal/                     # Placeholder for possible future portal work
└── sql/                        # SQL exploration and helper scripts
```

Important documentation files:

```text
docs/architecture.md
docs/source_data_overview.md
docs/staging_layer_plan.md
docs/m4_dbt_staging_validation.md
docs/m5_dimensional_modeling_design.md
docs/m5_dbt_marts_validation.md
docs/project_management.md
docs/orchestration.md
docs/gcp_orchestration_commands.md
```

---

## How to run the dbt project

This project uses a local dbt profile for BigQuery authentication.

The local `profiles.yml` file is not committed to the repository because it contains environment-specific connection settings.

From the repository root:

```bash
cd dbt
```

Validate dbt connection:

```bash
dbt debug
```

Build staging models:

```bash
dbt run --select staging
dbt test --select staging
```

Build intermediate and mart models:

```bash
dbt build --select intermediate marts
```

Generate and serve dbt docs:

```bash
dbt docs generate
dbt docs serve --port 8081
```

---

## Project workflow

The project is developed through controlled milestones on the branch:

```text
feature/olist-analytics-portal
```

GitHub Projects is used to organize work into cards, track milestone progress, and keep the workflow close to a practical enterprise delivery process.

The current workflow emphasizes:

- Small milestone-based delivery
- Clear acceptance criteria
- Documentation for each important design decision
- Validation before moving to the next milestone
- Separation between implemented scope and future scope

---

## Future work

The following items are planned as future milestones after the completed BigQuery, dbt, and M7 cloud orchestration foundation.

### M8 - ADE-inspired metadata refresh and monitoring tables

Borrow metadata-driven DataOps ideas from Agile Data Engine without directly integrating ADE.

Planned direction:

- Parse `manifest.json`
- Parse `run_results.json`
- Parse `catalog.json`
- Load dbt artifact metadata into BigQuery `olist_monitoring` tables
- Track model status, test results, row counts, execution metadata, and lineage metadata

### M9 - AI-assisted pipeline intelligence layer

Add a controlled AI-assisted explanation layer on top of dbt docs, dbt artifacts, and monitoring tables.

The AI layer should help explain:

- Pipeline health
- Data quality issues
- Failed tests
- Validation results
- Model lineage
- Runtime performance
- Downstream impact

The AI layer will not replace dbt tests, dbt validation, or structured monitoring tables. It will only provide an explanation and analysis layer on top of validated metadata.

---

## Project positioning

This project is a portfolio project, but it is structured like a practical analytics engineering workflow.

The main value is not only the final tables, but also the engineering process behind them:

- Understand the source system
- Design the warehouse layers
- Model facts and dimensions with explicit grain
- Validate assumptions through dbt tests
- Document lineage and decisions
- Keep project scope controlled through milestones
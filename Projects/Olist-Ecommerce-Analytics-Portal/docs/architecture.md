# Architecture

## Purpose

This document describes the implemented architecture of the Olist E-Commerce Analytics & Pipeline Monitoring Portal project.

The current completed scope focuses on the analytics engineering foundation:

- BigQuery raw layer
- dbt staging layer
- dbt intermediate layer
- dbt marts layer
- dbt tests
- dbt docs and lineage
- milestone-based project workflow

The project title includes pipeline monitoring and portal direction, but the current implemented architecture does not yet include orchestration, monitoring tables, BI dashboards, or a custom portal. These are planned as future milestones.

---

## Architecture status

Current architecture status:

```text
M1 - Project Setup & Source Understanding: completed
M2 - BigQuery Raw Layer: completed
M3 - Staging Layer Planning: completed
M4 - dbt Staging Layer: completed
M5 - Dimensional Modeling / Analytics Marts: completed
M6 - README / dbt docs / Project Showcase Cleanup: in progress
```

Implemented warehouse and dbt layers:

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

---

## High-level flow

The project follows a layered analytics engineering architecture.

```text
Source files
    Olist public CSV dataset
        |
        v
Raw warehouse layer
    BigQuery dataset: olist_raw
    Source-aligned raw tables
        |
        v
Staging transformation layer
    dbt models: staging
    BigQuery dataset: olist_staging
    Cleaned, renamed, typed source views
        |
        v
Intermediate transformation layer
    dbt models: intermediate
    BigQuery dataset: olist_intermediate
    Reusable order-level aggregations
        |
        v
Mart transformation layer
    dbt models: marts/core
    BigQuery dataset: olist_marts
    Fact and dimension tables
        |
        v
Quality and documentation layer
    dbt tests
    dbt docs
    dbt lineage graph
        |
        v
Consumption-ready analytics layer
    BI-ready dimensional model
```

The architecture intentionally separates source ingestion, cleaning, reusable transformations, dimensional modeling, validation, and documentation.

---

## Source data

The source data comes from the Olist Brazilian E-Commerce public dataset.

The project uses 9 source CSV files:

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

## BigQuery raw layer

The raw layer is implemented in BigQuery.

Dataset:

```text
olist_raw
```

Raw tables:

- `raw_customers`
- `raw_geolocation`
- `raw_orders`
- `raw_order_items`
- `raw_order_payments`
- `raw_order_reviews`
- `raw_products`
- `raw_sellers`
- `raw_product_category_translation`

The raw layer preserves source-aligned data. It is intended to stay close to the original CSV structure and provide a stable source for dbt transformations.

Raw layer documentation:

```text
metadata/bigquery/dataset_naming_plan.md
metadata/bigquery/raw_layer_validation.md
```

---

## dbt transformation architecture

The dbt project is located under:

```text
dbt/
```

Main dbt layers:

```text
dbt/models/staging/
dbt/models/intermediate/
dbt/models/marts/core/
```

The transformation flow is:

```text
source()
    ↓
staging models
    ↓
intermediate models
    ↓
mart models
```

This structure keeps each layer responsible for a specific type of transformation.

---

## Staging layer

The staging layer is implemented as dbt models built from BigQuery raw tables.

Dataset:

```text
olist_staging
```

Materialization:

```text
views
```

Staging models:

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
- standardize column names
- cast timestamps and numeric fields
- apply light cleaning
- expose stable source-aligned views for downstream models
- document important columns
- validate source assumptions through dbt tests

Validation result:

```text
dbt run --select staging
PASS=9 WARN=0 ERROR=0 SKIP=0 NO-OP=0 TOTAL=9

dbt test --select staging
PASS=39 WARN=0 ERROR=0 SKIP=0 NO-OP=0 TOTAL=39
```

Staging documentation:

```text
docs/staging_layer_plan.md
docs/m4_dbt_staging_validation.md
metadata/staging/source_to_staging_mapping.md
metadata/staging/column_cleanup_rules.md
```

---

## Intermediate layer

The intermediate layer contains reusable aggregation models.

Dataset:

```text
olist_intermediate
```

Intermediate models:

- `int_order_items_agg`
- `int_order_payments_agg`
- `int_order_reviews_agg`

Responsibilities:

- centralize repeated aggregation logic
- prepare order-level measures for fact tables
- reduce complexity inside the mart models
- make the mart layer easier to validate and document

Examples:

- aggregate item count, product count, seller count, price, and freight by order
- aggregate payment value, payment count, payment types, and installment metrics by order
- aggregate review counts, review score metrics, and review timing fields by order

---

## Marts layer

The marts layer contains BI-ready fact and dimension tables.

Dataset:

```text
olist_marts
```

The marts layer follows a star schema style design.

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

Responsibilities:

- provide business-friendly fact and dimension tables
- make model grain explicit
- define stable primary keys
- support relationship tests between facts and dimensions
- prepare data for BI tools and analytical applications

Marts documentation:

```text
docs/m5_dimensional_modeling_design.md
docs/m5_dbt_marts_validation.md
dbt/models/marts/core/schema.yml
```

---

## Dimensional model

The dimensional model is designed around e-commerce business entities such as orders, customers, sellers, products, payments, reviews, geolocation, and dates.

### Dimension tables

| Model | Grain | Primary key | Purpose |
|---|---|---|---|
| `dim_customers` | One row per customer | `customer_id` | Customer attributes and location fields |
| `dim_sellers` | One row per seller | `seller_id` | Seller attributes and location fields |
| `dim_products` | One row per product | `product_id` | Product attributes and translated category |
| `dim_geolocation_zip_prefix` | One row per zip code prefix | `geolocation_zip_code_prefix` | Representative geographic coordinates |
| `dim_dates` | One row per calendar date | `date_day` | Shared date dimension for order, shipping, and review dates |

### Fact tables

| Model | Grain | Primary key | Purpose |
|---|---|---|---|
| `fct_orders` | One row per order | `order_id` | Order lifecycle, delivery, payment, and review summary metrics |
| `fct_order_items` | One row per order item | `order_item_key` | Item-level price, freight, product, seller, and order analysis |
| `fct_order_payments` | One row per order payment sequence | `order_payment_key` | Payment method, installment, and payment value analysis |
| `fct_order_reviews` | One row per review and order | `review_key` | Review score and review timing analysis |

---

## Key modeling decisions

### Review fact grain

During M5 validation, `review_id` was found not to be unique in the source dataset.

Instead of forcing `review_id` to behave as a unique primary key, `fct_order_reviews` was modeled at the correct source grain:

```text
one row per review_id + order_id
```

A generated `review_key` is used as the primary key.

This keeps the dimensional model aligned with the real data and avoids hiding a source data quality issue.

### Geolocation representative coordinates

The raw geolocation table contains multiple coordinates per zip code prefix.

`dim_geolocation_zip_prefix` uses median latitude and longitude as representative coordinates to reduce the impact of outliers.

Average latitude and longitude are also retained for transparency.

These coordinates are intended for approximate geographic analysis, not precise route planning.

### Shared date dimension

`dim_dates` is generated from order, shipping, and review-related dates.

This allows fact tables to reference a shared date dimension and supports consistent time-based analysis.

---

## Data quality architecture

Data quality is implemented through dbt tests.

Test categories include:

- primary key `not_null` tests
- primary key `unique` tests
- foreign key relationship tests
- accepted values tests
- important business field `not_null` tests

The project uses dbt tests as structured validation checks, not as informal manual checks.

M5 validation result:

```text
dbt build --select intermediate marts
PASS=67 WARN=0 ERROR=0 SKIP=0 NO-OP=0 TOTAL=67
```

The M5 build included:

- 3 intermediate view models
- 9 mart table models
- 55 data tests

---

## Documentation and lineage architecture

dbt docs are used as the main technical documentation and lineage review tool for dbt models.

dbt docs provide:

- model descriptions
- column descriptions
- data test visibility
- upstream dependencies
- downstream dependencies
- lineage graph
- `Depends On` and `Referenced By` sections

Useful commands:

```bash
cd dbt
dbt docs generate
dbt docs serve
```

Recommended M6 screenshot folder:

```text
assets/screenshots/dbt_docs/
```

Recommended screenshots:

```text
assets/screenshots/dbt_docs/dbt_docs_project_overview.png
assets/screenshots/dbt_docs/fct_orders_lineage.png
assets/screenshots/dbt_docs/fct_order_reviews_lineage.png
assets/screenshots/dbt_docs/marts_tests_overview.png
```

---

## Project workflow architecture

The project is developed on the branch:

```text
feature/olist-analytics-portal
```

The workflow is milestone-based.

Current completed milestones:

- M1 - Project Setup & Source Understanding
- M2 - BigQuery Raw Layer
- M3 - Staging Layer Planning
- M4 - dbt Staging Layer
- M5 - Dimensional Modeling / Analytics Marts

Current milestone:

- M6 - README / dbt docs / Project Showcase Cleanup

GitHub Projects is used to organize cards, track milestone progress, and keep development work reviewable.

The workflow emphasizes:

- small controlled milestones
- clear acceptance criteria
- documentation before and after implementation
- validation before moving to the next milestone
- separation between completed scope and future work

---

## Current implementation boundary

The current architecture includes:

- BigQuery raw tables
- dbt staging views
- dbt intermediate views
- dbt marts tables
- dbt tests
- dbt docs
- dbt lineage review
- Markdown documentation
- GitHub Project workflow

The current architecture does not yet include:

- Power BI dashboard
- React or Node portal
- Google Cloud Scheduler
- Cloud Run Job orchestration
- BigQuery monitoring tables
- automated dbt artifact ingestion
- AI-assisted pipeline intelligence

These items are intentionally kept outside the current M6 scope.

---

## Future architecture direction

Future milestones are planned as follows.

### M7 - Google Cloud Scheduler + Cloud Run Job orchestration

Planned goal:

- run dbt workflows through scheduled Google Cloud execution
- keep orchestration separate from modeling logic
- make the pipeline easier to operate repeatedly

### M8 - ADE-inspired metadata refresh and monitoring tables

Planned goal:

- borrow metadata-driven DataOps ideas from Agile Data Engine
- parse dbt artifacts such as `manifest.json`, `run_results.json`, and `catalog.json`
- load metadata into BigQuery `olist_monitoring` tables
- track model status, tests, run results, row counts, and lineage metadata

This will not be a direct Agile Data Engine integration.

### M9 - AI-assisted pipeline intelligence layer

Planned goal:

- build an explanation layer on top of dbt docs, dbt artifacts, and monitoring tables
- answer questions about pipeline health, data quality, failed tests, validation, lineage, and runtime performance
- help interpret metadata rather than replace structured tests

The AI layer will not replace dbt tests, dbt validation, or monitoring tables.

---

## Design principles

The architecture follows these principles:

- keep source, staging, intermediate, and mart layers separated
- preserve raw data as source-aligned as possible
- make transformations explicit and testable
- define fact and dimension grain clearly
- avoid forcing source data into incorrect assumptions
- use dbt tests for repeatable validation
- use dbt docs and lineage for transparency
- avoid expanding scope before the foundation is stable
- keep the project understandable for both engineering and BI audiences
# BigQuery Dataset Naming Plan

## Purpose

This document defines the planned BigQuery dataset and table naming approach for the Olist E-Commerce Analytics & Pipeline Monitoring Portal.

This file is part of **M1 - Project Setup & Source Understanding**.

No BigQuery datasets or tables are created in M1.

## BigQuery Project

Planned BigQuery project:

```text
<your-gcp-project-id>
```

The actual GCP project ID will be decided before BigQuery implementation starts.

## Planned BigQuery Location

Planned location:

```text
EU
```

This is selected as the default planned location for the portfolio project. It can be changed before implementation if needed.

## Dataset Naming Convention

Dataset names should use:

```text
lowercase_snake_case
```

Planned datasets:

| Dataset name | Purpose |
|---|---|
| `olist_raw` | Source-aligned tables loaded from Olist CSV files |
| `olist_staging` | Cleaned and standardized staging tables |
| `olist_marts` | Dimensional and reporting-ready tables |
| `olist_monitoring` | Pipeline status, data quality, and freshness monitoring tables |

## Raw Layer

The raw layer keeps source-aligned data.

Planned dataset:

```text
olist_raw
```

Planned raw tables:

| Raw table | Source file |
|---|---|
| `raw_customers` | `olist_customers_dataset.csv` |
| `raw_geolocation` | `olist_geolocation_dataset.csv` |
| `raw_order_items` | `olist_order_items_dataset.csv` |
| `raw_order_payments` | `olist_order_payments_dataset.csv` |
| `raw_order_reviews` | `olist_order_reviews_dataset.csv` |
| `raw_orders` | `olist_orders_dataset.csv` |
| `raw_products` | `olist_products_dataset.csv` |
| `raw_sellers` | `olist_sellers_dataset.csv` |
| `raw_product_category_translation` | `product_category_name_translation.csv` |

Raw tables should preserve original source column names.

## Staging Layer

The staging layer prepares source data for modeling.

Planned dataset:

```text
olist_staging
```

Expected table naming pattern:

```text
stg_<source_entity>
```

Examples:

| Staging table | Purpose |
|---|---|
| `stg_customers` | Standardized customer fields |
| `stg_orders` | Standardized order lifecycle fields |
| `stg_order_items` | Standardized order item fields |
| `stg_order_payments` | Standardized payment fields |
| `stg_order_reviews` | Standardized review fields |
| `stg_products` | Standardized product fields |
| `stg_sellers` | Standardized seller fields |
| `stg_geolocation` | Standardized geolocation fields |
| `stg_product_category_translation` | Standardized category translation fields |

Staging models may rename source columns into clearer business-friendly names.

For example, source columns such as `product_name_lenght` and `product_description_lenght` may be corrected in staging while remaining unchanged in the raw layer.

## Marts Layer

The marts layer contains dimensional and reporting-ready tables.

Planned dataset:

```text
olist_marts
```

Expected table naming patterns:

```text
fact_<business_process>
dim_<business_entity>
mart_<reporting_area>
```

Initial fact table candidates:

| Table | Purpose |
|---|---|
| `fact_orders` | Order-level lifecycle and status metrics |
| `fact_order_items` | Item-level sales and freight metrics |
| `fact_payments` | Payment value and payment method metrics |
| `fact_reviews` | Review score and customer feedback metrics |

Initial dimension table candidates:

| Table | Purpose |
|---|---|
| `dim_customers` | Customer identifiers and location context |
| `dim_sellers` | Seller identifiers and location context |
| `dim_products` | Product attributes and category context |
| `dim_product_categories` | Product category translation and labels |
| `dim_dates` | Date attributes for reporting |
| `dim_geolocation` | Cleaned geolocation reference data |

Initial reporting mart candidates:

| Table | Purpose |
|---|---|
| `mart_sales_overview` | Sales and order overview metrics |
| `mart_delivery_performance` | Delivery delay and fulfillment metrics |
| `mart_product_performance` | Product and category performance metrics |
| `mart_seller_performance` | Seller-level performance metrics |
| `mart_customer_geography` | Customer geography and regional reporting |
| `mart_review_satisfaction` | Review score and satisfaction metrics |

These tables are only planning candidates in M1.

## Monitoring Layer

The monitoring layer stores pipeline and data quality visibility outputs.

Planned dataset:

```text
olist_monitoring
```

Expected table naming pattern:

```text
<monitoring_area>_<check_or_event>
```

Initial monitoring table candidates:

| Table | Purpose |
|---|---|
| `pipeline_runs` | Pipeline run history and status |
| `source_freshness_checks` | Source file or raw table freshness checks |
| `source_row_count_checks` | Source row count validation results |
| `data_quality_results` | Data quality check results |
| `dbt_test_results` | dbt test execution results |
| `mart_availability_checks` | Reporting table availability checks |

These tables are not implemented in M1.

## Naming Rules

General naming rules:

- use lowercase names
- use underscores instead of spaces
- avoid abbreviations unless common and clear
- keep raw table names close to source entities
- use `stg_` for staging models
- use `fact_` for measurable business events or outcomes
- use `dim_` for business context entities
- use `mart_` for reporting-ready business views
- keep monitoring table names clear and operational

## M1 Scope

M1 only defines the naming plan.

M1 does not include:

- creating BigQuery datasets
- loading CSV files into BigQuery
- creating raw tables
- creating dbt models
- creating marts
- creating monitoring tables
- configuring scheduled pipeline runs

## Current Status

```text
M1 - Project Setup & Source Understanding: in progress
```

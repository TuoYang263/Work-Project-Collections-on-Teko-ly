# Source-to-Staging Mapping

## Purpose

This document defines the planned mapping from BigQuery raw source tables to source-aligned staging tables.

The staging layer keeps one staging table per source table.

Each staging table should make the source data cleaner, more consistent, and easier to validate, without changing the business grain of the source table.

## Mapping Overview

| Raw table | Staging table | Source grain | Key columns | Main staging focus |
|---|---|---|---|---|
| raw_customers | stg_customers | One row per customer record | customer_id, customer_unique_id | Standardize customer location fields and preserve customer identifiers |
| raw_geolocation | stg_geolocation | One row per geolocation record | geolocation_zip_code_prefix | Standardize location fields and document duplicate zip prefix behavior; zip prefix is a reference column, not a unique key |
| raw_order_items | stg_order_items | One row per order item | order_id, order_item_id | Standardize item-level numeric fields and shipping limit timestamp |
| raw_order_payments | stg_order_payments | One row per order payment record | order_id, payment_sequential | Standardize payment fields and numeric payment value |
| raw_order_reviews | stg_order_reviews | One row per review record | review_id, order_id | Standardize review timestamps and handle optional review comments |
| raw_orders | stg_orders | One row per order | order_id | Standardize order status and order lifecycle timestamps |
| raw_product_category_translation | stg_product_category_translation | One row per product category translation | product_category_name | Standardize category name fields |
| raw_products | stg_products | One row per product | product_id | Standardize product category and product attribute numeric fields |
| raw_sellers | stg_sellers | One row per seller | seller_id | Standardize seller location fields |

## Staging Rules at Mapping Level

The staging layer should:

- keep the same business grain as the raw source table
- keep source identifiers available for traceability
- rename columns only for consistency and readability
- apply safe type conversions only when the expected type is clear
- document known nulls instead of hiding them
- document duplicate patterns instead of removing records too early
- avoid business-level joins between different source entities

## Out of Scope

The following are out of scope for source-to-staging mapping:

- fact tables
- dimension tables
- star schema design
- business KPI definitions
- Power BI models
- React portal data models
- Azure deployment design
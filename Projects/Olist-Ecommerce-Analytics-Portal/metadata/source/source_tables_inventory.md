# Source Tables Inventory

## Purpose

This file provides a compact inventory of the source CSV files used in the Olist E-Commerce Analytics & Pipeline Monitoring Portal.

It records source file names, row counts, main entities, key fields, expected joins, and initial modeling roles.

This file is part of **M1 - Project Setup & Source Understanding**.

## Source Table Inventory

| Source file | Rows | Main entity | Key field / grain | Expected joins | Initial modeling role |
|---|---:|---|---|---|---|
| `olist_orders_dataset.csv` | 99,441 | Orders | `order_id` / one row per order | `customer_id` → customers | Order-level fact |
| `olist_order_items_dataset.csv` | 112,650 | Order items | `order_id`, `order_item_id` / one row per order item | `order_id` → orders, `product_id` → products, `seller_id` → sellers | Item-level sales fact |
| `olist_order_payments_dataset.csv` | 103,886 | Payments | `order_id`, `payment_sequential` / one row per payment record | `order_id` → orders | Payment fact |
| `olist_order_reviews_dataset.csv` | 104,719 | Reviews | `review_id` / one row per review record | `order_id` → orders | Review fact |
| `olist_customers_dataset.csv` | 99,441 | Customers | `customer_id` / one row per order customer record | `customer_id` → orders, `customer_zip_code_prefix` → geolocation | Customer dimension |
| `olist_sellers_dataset.csv` | 3,095 | Sellers | `seller_id` / one row per seller | `seller_id` → order items, `seller_zip_code_prefix` → geolocation | Seller dimension |
| `olist_products_dataset.csv` | 32,951 | Products | `product_id` / one row per product | `product_id` → order items, `product_category_name` → category translation | Product dimension |
| `olist_geolocation_dataset.csv` | 1,000,163 | Geolocation | `geolocation_zip_code_prefix` / multiple rows may exist per zip prefix | zip prefix joins from customers and sellers | Geolocation reference / dimension candidate |
| `product_category_name_translation.csv` | 71 | Product category translation | `product_category_name` / one row per product category | `product_category_name` → products | Product category dimension |

## Notes

- Raw tables should preserve original source column names.
- Staging models may rename columns into more business-friendly names.
- The original product source contains `product_name_lenght` and `product_description_lenght`; these should be preserved in the raw layer.
- Geolocation may require deduplication or aggregation before being used as a clean dimension.
- The initial modeling roles are not final. They will be refined after profiling, data quality checks, and reporting requirements are defined.

## M1 Scope

This inventory is based on file names, row counts, columns, and expected source relationships.

M1 does not include:

- full data profiling
- data quality test implementation
- dbt source configuration
- BigQuery table creation
- final dimensional model design

## Current Status

```text
M1 - Project Setup & Source Understanding: in progress
```
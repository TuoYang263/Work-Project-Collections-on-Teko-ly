# BigQuery Raw Table Loading Plan

## Purpose

This document defines how the original Olist CSV files are loaded into BigQuery raw tables.

The raw layer keeps the source-aligned structure. Column names and source granularity are preserved as much as possible.

## Dataset

| Layer | BigQuery dataset | Location |
|---|---|---|
| Raw | `olist_raw` | EU |

## Loading approach

| Setting | Decision |
|---|---|
| Source format | CSV |
| Header row | Skip 1 header row |
| Schema handling | BigQuery auto-detect for initial raw load |
| Table naming | Prefix source-aligned tables with `raw_` |
| Transformation | No business transformation in raw layer |
| Load mode | Create or replace table during initial development |

## Raw table mapping

| Source CSV file | BigQuery raw table |
|---|---|
| `olist_customers_dataset.csv` | `raw_customers` |
| `olist_geolocation_dataset.csv` | `raw_geolocation` |
| `olist_orders_dataset.csv` | `raw_orders` |
| `olist_order_items_dataset.csv` | `raw_order_items` |
| `olist_order_payments_dataset.csv` | `raw_order_payments` |
| `olist_order_reviews_dataset.csv` | `raw_order_reviews` |
| `olist_products_dataset.csv` | `raw_products` |
| `olist_sellers_dataset.csv` | `raw_sellers` |
| `product_category_name_translation.csv` | `raw_product_category_translation` |

## Validation after loading

After each table is loaded, check:

- the table exists in `olist_raw`
- row count matches the source inventory
- columns are visible and readable
- sample rows can be queried

## Notes

Raw tables are not designed for direct BI reporting. They are the source-aligned foundation for later staging and analytics layers.
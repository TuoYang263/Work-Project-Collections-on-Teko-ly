# BigQuery Raw Layer Validation

## Purpose

This document records the row count validation for the BigQuery raw layer.

The raw layer stores source-aligned Olist CSV data in BigQuery with minimal transformation.

## Dataset

| Dataset | Location |
|---|---|
| `olist_raw` | EU |

## Row count validation

| Raw table | Expected rows | BigQuery rows | Status |
|---|---:|---:|---|
| `raw_customers` | 99,441 | 99,441 | Passed |
| `raw_geolocation` | 1,000,163 | 1,000,163 | Passed |
| `raw_order_items` | 112,650 | 112,650 | Passed |
| `raw_order_payments` | 103,886 | 103,886 | Passed |
| `raw_order_reviews` | 99,224 | 99,224 | Passed |
| `raw_orders` | 99,441 | 99,441 | Passed |
| `raw_product_category_translation` | 71 | 71 | Passed |
| `raw_products` | 32,951 | 32,951 | Passed |
| `raw_sellers` | 3,095 | 3,095 | Passed |

## Notes

Raw tables preserve the source-aligned structure and original source column names.

Some source-level data quality issues, such as null values or duplicated geolocation records, are not fixed in the raw layer. They will be handled in later staging and modelling steps.
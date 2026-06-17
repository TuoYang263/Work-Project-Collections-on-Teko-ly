# Source Data Overview

## Purpose

This document provides an initial overview of the source data used in the Olist E-Commerce Analytics & Pipeline Monitoring Portal.

The goal is to understand the source files, business entities, row counts, and key fields before designing BigQuery datasets, dbt models, dimensional models, data quality checks, or dashboards.

This document is part of **M1 - Project Setup & Source Understanding**.

## Dataset

The project uses the Olist Brazilian E-Commerce public dataset.

The dataset contains historical e-commerce order data from Brazil. It includes information about orders, customers, sellers, products, payments, reviews, order items, and geolocation.

The dataset is historical and is used as a stable public dataset for modeling, reporting, and pipeline monitoring practice. The project focuses on analytics engineering workflow design rather than current market trend analysis.

## Source Files

The downloaded source data contains 9 CSV files.

| Source file | Rows | Main entity | Planned raw table |
|---|---:|---|---|
| `olist_customers_dataset.csv` | 99,441 | Customers | `raw_customers` |
| `olist_geolocation_dataset.csv` | 1,000,163 | Geolocation | `raw_geolocation` |
| `olist_order_items_dataset.csv` | 112,650 | Order items | `raw_order_items` |
| `olist_order_payments_dataset.csv` | 103,886 | Payments | `raw_order_payments` |
| `olist_order_reviews_dataset.csv` | 104,719 | Reviews | `raw_order_reviews` |
| `olist_orders_dataset.csv` | 99,441 | Orders | `raw_orders` |
| `olist_products_dataset.csv` | 32,951 | Products | `raw_products` |
| `olist_sellers_dataset.csv` | 3,095 | Sellers | `raw_sellers` |
| `product_category_name_translation.csv` | 71 | Product category translation | `raw_product_category_translation` |

## Business Entities

The main business entities are:

- orders
- order items
- customers
- sellers
- products
- payments
- reviews
- geolocation
- product categories
- dates and delivery events

These entities will later support dimensional modeling and reporting marts.

## Source Table Notes

### Customers

Source file:

```text
olist_customers_dataset.csv
```

Rows:

```text
99,441
```

Columns:

```text
customer_id
customer_unique_id
customer_zip_code_prefix
customer_city
customer_state
```

Initial notes:

- `customer_id` can be used to join customers to orders.
- `customer_unique_id` can support customer-level analysis across orders.
- Customer city, state, and zip code prefix can support regional analysis.

### Geolocation

Source file:

```text
olist_geolocation_dataset.csv
```

Rows:

```text
1,000,163
```

Columns:

```text
geolocation_zip_code_prefix
geolocation_lat
geolocation_lng
geolocation_city
geolocation_state
```

Initial notes:

- This table provides location context for zip code prefixes.
- It can support customer and seller geography analysis.
- The table may require deduplication or aggregation because zip code prefixes can appear more than once.

### Order Items

Source file:

```text
olist_order_items_dataset.csv
```

Rows:

```text
112,650
```

Columns:

```text
order_id
order_item_id
product_id
seller_id
shipping_limit_date
price
freight_value
```

Initial notes:

- This table provides item-level order details.
- It connects orders, products, and sellers.
- It can support product sales, seller performance, price, and freight analysis.

### Order Payments

Source file:

```text
olist_order_payments_dataset.csv
```

Rows:

```text
103,886
```

Columns:

```text
order_id
payment_sequential
payment_type
payment_installments
payment_value
```

Initial notes:

- This table provides payment-level information.
- One order may have one or more payment records.
- It can support payment method, installment, and payment value analysis.

### Order Reviews

Source file:

```text
olist_order_reviews_dataset.csv
```

Rows:

```text
104,719
```

Columns:

```text
review_id
order_id
review_score
review_comment_title
review_comment_message
review_creation_date
review_answer_timestamp
```

Initial notes:

- This table provides customer feedback after purchase.
- It can support review score, customer satisfaction, and review timing analysis.
- Review text fields may be incomplete and should be treated carefully in later data quality checks.

### Orders

Source file:

```text
olist_orders_dataset.csv
```

Rows:

```text
99,441
```

Columns:

```text
order_id
customer_id
order_status
order_purchase_timestamp
order_approved_at
order_delivered_carrier_date
order_delivered_customer_date
order_estimated_delivery_date
```

Initial notes:

- This is the central order-level table.
- It links orders to customers.
- It contains key order lifecycle timestamps.
- It can support order trend, status, and delivery performance analysis.

### Products

Source file:

```text
olist_products_dataset.csv
```

Rows:

```text
32,951
```

Columns:

```text
product_id
product_category_name
product_name_lenght
product_description_lenght
product_photos_qty
product_weight_g
product_length_cm
product_height_cm
product_width_cm
```

Initial notes:

- This table provides product attributes.
- `product_category_name` can be joined to the translation table.
- The original source uses `lenght` in two column names. The raw layer should preserve source column names, while staging models may rename them to corrected business-friendly names.

### Sellers

Source file:

```text
olist_sellers_dataset.csv
```

Rows:

```text
3,095
```

Columns:

```text
seller_id
seller_zip_code_prefix
seller_city
seller_state
```

Initial notes:

- This table provides seller location information.
- It can support seller performance and regional fulfillment analysis.

### Product Category Translation

Source file:

```text
product_category_name_translation.csv
```

Rows:

```text
71
```

Columns:

```text
product_category_name
product_category_name_english
```

Initial notes:

- This table provides English translations for product category names.
- It can support more readable reporting labels in BI dashboards and the analytics portal.

## Initial Relationship Understanding

The initial source relationships are expected to be:

```text
customers.customer_id
        -> orders.customer_id

orders.order_id
        -> order_items.order_id
        -> order_payments.order_id
        -> order_reviews.order_id

order_items.product_id
        -> products.product_id

order_items.seller_id
        -> sellers.seller_id

products.product_category_name
        -> product_category_name_translation.product_category_name

customers.customer_zip_code_prefix
        -> geolocation.geolocation_zip_code_prefix

sellers.seller_zip_code_prefix
        -> geolocation.geolocation_zip_code_prefix
```

These relationships will be checked more carefully in later milestones.

## Initial Analytics Themes

The source data can support several reporting themes:

- sales and order trends
- product category performance
- customer and seller geography
- delivery performance
- freight and payment analysis
- review score and customer satisfaction
- pipeline freshness and data quality monitoring

## Initial Modeling Direction

The future dimensional model may include the following tables.

Fact tables:

- `fact_orders`
- `fact_order_items`
- `fact_payments`
- `fact_reviews`

Dimension tables:

- `dim_customers`
- `dim_sellers`
- `dim_products`
- `dim_product_categories`
- `dim_dates`
- `dim_geolocation`

This is only an initial direction. The final model will be refined after source profiling, business question definition, and data quality checks.

## Data Quality Areas to Review Later

Potential data quality areas include:

- missing primary keys
- duplicate records
- invalid date relationships
- missing delivery dates
- missing product category values
- inconsistent customer or seller location fields
- duplicate geolocation zip code prefixes
- orphan records between orders and order items
- payment totals that require order-level reconciliation
- review records without matching orders

These checks are not implemented in M1.

## M1 Scope

In M1, this document only provides an initial overview based on file names, row counts, columns, and expected relationships.

M1 does not include:

- full exploratory data analysis
- loading source files into BigQuery
- creating BigQuery raw tables
- running detailed profiling
- creating dbt sources
- creating data quality tests
- building dimensional models

## Current Status

```text
M1 - Project Setup & Source Understanding: in progress
```








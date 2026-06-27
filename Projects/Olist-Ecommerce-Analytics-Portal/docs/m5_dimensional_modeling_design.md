# M5 - Dimensional Modeling / Analytics Marts Design

## Objective

Build a core analytics mart on top of the dbt staging layer using dimensional modeling principles. The goal is to provide clean, tested, BI-ready fact and dimension tables for order, sales, payment, review, product, seller, customer, and geography analysis.

## Scope

Included:
- Star schema design
- dbt intermediate models
- dbt mart models
- dbt data tests
- dbt documentation
- Validation documentation

Excluded for this milestone:
- Power BI dashboard
- React portal
- Pipeline monitoring UI
- Orchestration
- CI/CD
- AI assistant

## Source Layer

All mart models depend on dbt staging models, not raw BigQuery tables.

Staging inputs:
- stg_customers
- stg_geolocation
- stg_orders
- stg_order_items
- stg_order_payments
- stg_order_reviews
- stg_products
- stg_sellers
- stg_product_category_translation

## Dimensional Model

### Dimensions

| Model | Grain | Primary Key | Upstream Models |
|---|---|---|---|
| dim_customers | One row per customer_id | customer_id | stg_customers |
| dim_sellers | One row per seller_id | seller_id | stg_sellers |
| dim_products | One row per product_id | product_id | stg_products, stg_product_category_translation |
| dim_geolocation_zip_prefix | One row per geolocation_zip_code_prefix | geolocation_zip_code_prefix | stg_geolocation |
| dim_dates | One row per calendar date | date_day | stg_orders |

### Facts

| Model | Grain | Primary Key | Upstream Models |
|---|---|---|---|
| fct_orders | One row per order_id | order_id | stg_orders, int_order_items_agg, int_order_payments_agg, int_order_reviews_agg |
| fct_order_items | One row per order_id + order_item_id | order_item_key | stg_order_items, stg_orders |
| fct_order_payments | One row per order_id + payment_sequential | order_payment_key | stg_order_payments, stg_orders |
| fct_order_reviews | One row per review_id | review_id | stg_order_reviews, stg_orders |

## Intermediate Models

| Model | Grain | Purpose |
|---|---|---|
| int_order_items_agg | One row per order_id | Aggregate item-level sales and freight metrics |
| int_order_payments_agg | One row per order_id | Aggregate payment metrics |
| int_order_reviews_agg | One row per order_id | Aggregate review metrics |

## Model Lineage

The mart layer is built on top of the dbt staging layer. All marts depend on cleaned and standardized staging models rather than raw BigQuery tables.

### Dimension lineage

| Mart model | Upstream models |
|---|---|
| dim_customers | stg_customers |
| dim_sellers | stg_sellers |
| dim_products | stg_products, stg_product_category_translation |
| dim_geolocation_zip_prefix | stg_geolocation |
| dim_dates | stg_orders |

### Intermediate lineage

| Intermediate model | Upstream models | Grain |
|---|---|---|
| int_order_items_agg | stg_order_items | One row per order_id |
| int_order_payments_agg | stg_order_payments | One row per order_id |
| int_order_reviews_agg | stg_order_reviews | One row per order_id |

### Fact lineage

| Fact model | Upstream models | Grain |
|---|---|---|
| fct_orders | stg_orders, int_order_items_agg, int_order_payments_agg, int_order_reviews_agg | One row per order_id |
| fct_order_items | stg_order_items, stg_orders | One row per order_id + order_item_id |
| fct_order_payments | stg_order_payments, stg_orders | One row per order_id + payment_sequential |
| fct_order_reviews | stg_order_reviews, stg_orders | One row per review_id |

### Modeling note

The SQL lineage is intentionally kept clean: mart models are built from staging and intermediate models using dbt `ref()` dependencies.

Fact tables do not directly depend on dimension tables in SQL. Instead, analytical relationships between facts and dimensions are enforced through dbt relationship tests in `schema.yml`.

This keeps the dbt DAG simple while still documenting and validating the star schema relationships.

## Modeling Decisions

### Customer grain

The customer dimension uses customer_id as the primary key because orders reference customer_id. customer_unique_id is retained as an analytical attribute to support repeat-customer analysis later.

### Geolocation grain

The raw geolocation table contains multiple records per zip code prefix.  
The mart creates one row per zip code prefix using aggregated latitude and longitude values.

### Order-level and item-level facts

Order lifecycle analysis and item-level sales analysis are separated:
- fct_orders supports order status, delivery, payment, and review analysis.
- fct_order_items supports product, seller, price, and freight analysis.

### Review modeling

Reviews are stored in a separate review-grain fact table.  
Order-level review metrics are also aggregated into fct_orders through int_order_reviews_agg.

## Validation Plan

The mart layer will include:
- Primary key not_null and unique tests
- Foreign key relationship tests between facts and dimensions
- Accepted values tests for status and payment type
- Not null tests for important business keys
- Basic measure sanity checks where appropriate

## Success Criteria

M5 is complete when:
- All intermediate and mart models run successfully
- dbt tests pass
- dbt docs can show mart lineage
- Design and validation docs are committed
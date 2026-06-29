# M5 - dbt Marts Validation

## Objective

Validate the M5 dimensional marts layer, including intermediate aggregation models, dimension tables, fact tables, and mart-level dbt data tests.

The goal of this validation step is to confirm that the analytics mart layer can be built successfully and that its primary keys, foreign key relationships, accepted values, and important business keys are tested through dbt.

## Validation Command

```bash
cd dbt
dbt build --select intermediate marts
```

## Validation Result

The dbt build completed successfully.

```text
PASS=67 WARN=0 ERROR=0 SKIP=0 NO-OP=0 TOTAL=67
```

## Objects Built

The build included:

- 3 intermediate view models
- 9 mart table models
- 55 data tests

## Intermediate Models

The following intermediate models were built as views in BigQuery:

- `int_order_items_agg`
- `int_order_payments_agg`
- `int_order_reviews_agg`

These models provide reusable order-level aggregations for item, payment, and review metrics.

## Dimension Models

The following dimension models were built as tables in BigQuery:

- `dim_customers`
- `dim_dates`
- `dim_geolocation_zip_prefix`
- `dim_products`
- `dim_sellers`

These dimension tables provide descriptive attributes for customers, dates, geolocation zip prefixes, products, and sellers.

## Fact Models

The following fact models were built as tables in BigQuery:

- `fct_orders`
- `fct_order_items`
- `fct_order_payments`
- `fct_order_reviews`

These fact tables support order lifecycle analysis, item-level sales analysis, payment analysis, and review analysis.

## Test Coverage

The mart layer includes the following dbt data tests:

- Primary key `not_null` and `unique` tests
- Foreign key relationship tests between facts and dimensions
- Accepted values tests for order status, payment type, and review score
- Not null tests for important business keys and measures

## Key Modeling Validation Notes

### Review fact grain

Initial fact-level testing showed that `review_id` is not unique in the source dataset.

To reflect the actual data grain, `fct_order_reviews` was updated to use one row per `review_id` and `order_id`. A generated `review_key` was added as the primary key.

This change was applied consistently across:

- `fct_order_reviews.sql`
- `models/marts/core/schema.yml`
- `docs/m5_dimensional_modeling_design.md`

### Geolocation representative coordinates

The geolocation zip prefix dimension aggregates multiple raw geolocation records into one row per zip code prefix.

Representative latitude and longitude use median coordinates to reduce sensitivity to outliers. Average coordinates are also retained for transparency.

These coordinates are intended for approximate geographic analysis, not precise routing or distance calculation.

### Date dimension coverage

The date dimension is generated from order, shipping, and review dates used across the mart layer.

This improves relationship test coverage for:

- Order purchase dates
- Shipping limit dates
- Review creation dates
- Review answer dates

## BigQuery Output Datasets

The M5 marts validation produced models in the following BigQuery datasets:

- `olist_intermediate`
- `olist_marts`

## Success Criteria

M5 mart validation is considered successful because:

- All intermediate models built successfully
- All dimension models built successfully
- All fact models built successfully
- All mart-level dbt data tests passed
- Fact and dimension relationships were validated
- The final dbt build completed with zero warnings and zero errors

## Final Validation Summary

```text
dbt build --select intermediate marts
PASS=67 WARN=0 ERROR=0 SKIP=0 NO-OP=0 TOTAL=67
```
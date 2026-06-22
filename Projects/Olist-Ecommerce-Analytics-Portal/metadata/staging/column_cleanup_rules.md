# Column Cleanup Rules

## Purpose

This document defines the planned column cleanup rules for the Olist staging layer.

The staging layer should make raw source columns easier to use and validate, while keeping the data source-aligned and traceable.

Column cleanup should be simnple, explicit, and documented before implementation.

## General Naming Rules

Staging columns should follow these naming rules:

- use lower snake case
- keep source identifier columns recognizable
- avoid unnecessary renaming when the raw column name is already clear
- rename columns only when it improves consistency, readability, or downstream usability
- avoid business-level naming that belongs to fact or dimension models

## Identifier Columns

Source identifier columns should usually keep their original names.

Examples:

| Raw column | Planned staging column | Decision |
|---|---|---|
| customer_id | customer_id | Keep |
| customer_unique_id | customer_unique_id | Keep |
| order_id | order_id | Keep |
| order_item_id | order_item_id | Keep |
| product_id | product_id | Keep |
| seller_id | seller_id | Keep |
| review_id | review_id | Keep |

Reason:

Identifier columns are important for traceability between raw and staging layers.

## Location Columns

Location columns should keep their source meaning but use consistent naming when needed.

Examples:

| Raw column | Planned staging column | Decision |
|---|---|---|
| customer_zip_code_prefix | customer_zip_code_prefix | Keep |
| customer_city | customer_city | Keep |
| customer_state | customer_state | Keep |
| seller_zip_code_prefix | seller_zip_code_prefix | Keep |
| seller_city | seller_city | Keep |
| seller_state | seller_state | Keep |
| geolocation_zip_code_prefix | geolocation_zip_code_prefix | Keep |
| geolocation_city | geolocation_city | Keep |
| geolocation_state | geolocation_state | Keep |

Planned cleanup:

- trim leading and trailing whitespace from text fields
- keep city and state values source-aligned
- do not merge customer, seller, and geolocation location logic in staging

## Timestamp and Date Columns

Timestamp and date columns should keep their business meaning clear.

Examples:

| Raw column | Planned staging column | Decision |
|---|---|---|
| order_purchase_timestamp | order_purchase_timestamp | Keep |
| order_approved_at | order_approved_at | Keep |
| order_delivered_carrier_date | order_delivered_carrier_date | Keep |
| order_delivered_customer_date | order_delivered_customer_date | Keep |
| order_estimated_delivery_date | order_estimated_delivery_date | Keep |
| shipping_limit_date | shipping_limit_date | Keep |
| review_creation_date | review_creation_date | Keep |
| review_answer_timestamp | review_answer_timestamp | Keep |

Planned cleanup:

- convert timestamp-like fields to timestamp types when safe
- convert date-like fields to date or timestamp types based on source values
- keep lifecycle timestamps in the staging order table
- do not calculate delivery duration or delay metrics in staging

## Numeric Columns

Numeric columns should be converted only when the expected type is clear.

Examples:

| Raw column | Planned staging column | Decision |
|---|---|---|
| price | price | Keep name, convert to numeric |
| freight_value | freight_value | Keep name, convert to numeric |
| payment_value | payment_value | Keep name, convert to numeric |
| payment_installments | payment_installments | Keep name, convert to integer |
| review_score | review_score | Keep name, convert to integer |
| product_weight_g | product_weight_g | Keep name, convert to numeric or integer |
| product_length_cm | product_length_cm | Keep name, convert to numeric or integer |
| product_height_cm | product_height_cm | Keep name, convert to numeric or integer |
| product_width_cm | product_width_cm | Keep name, convert to numeric or integer |

Planned cleanup:

- use safe type conversion
- avoid rounding monetary values
- do not create revenue, cost, or margin metrics in staging

## Text Columns

Text columns should remain close to source values.

Examples:

| Raw column | Planned staging column | Decision |
|---|---|---|
| order_status | order_status | Keep |
| payment_type | payment_type | Keep |
| product_category_name | product_category_name | Keep |
| product_category_name_english | product_category_name_english | Keep |
| review_comment_title | review_comment_title | Keep |
| review_comment_message | review_comment_message | Keep |

Planned cleanup:

- trim leading and trailing whitespace where appropriate
- keep original category names and translated category names in separate staging tables
- do not translate, group, or simplify business categories in staging
- do not remove review comment fields because null comments are expected source behavior

## Additional Staging Handling Rules

### Timestamp and Date Handling

Timestamp and date fields should be standardized in staging only when the source meaning is clear.

Planned handling:

- convert timestamp-like fields to `TIMESTAMP` when safe
- preserve null timestamp values when they reflect normal source behavior
- keep order lifecycle timestamps in `stg_orders`
- keep `shipping_limit_date` in `stg_order_items`
- keep review timestamps in `stg_order_reviews`
- do not infer missing timestamps
- do not calculate delivery duration, delivery delay, shipping delay, or review response time in staging

The staging layer should make timestamp fields easier to use, but business interpretation should stay in later analytical models.

### Numeric Type Handling

Numeric fields should be converted only when the expected type is clear.

Planned handling:

- convert monetary fields such as `price`, `freight_value`, and `payment_value` to numeric types
- convert count or score fields such as `payment_installments` and `review_score` to integer types
- convert product size and weight fields to numeric or integer types based on source values
- use safe type conversion during implementation
- avoid rounding monetary values
- do not calculate revenue, margin, average order value, or other KPIs in staging

The staging layer should standardize numeric types without creating business metrics.

### Null and Duplicate Handling

Null and duplicate values should be documented before being changed or removed.

Planned handling:

- preserve null values when they are part of normal source behavior
- keep review comment fields even when comments are missing
- preserve missing order lifecycle timestamps when they are related to order status
- document duplicate patterns instead of removing records too early
- treat `geolocation_zip_code_prefix` as a reference column, not a unique key
- do not remove records only because they appear duplicated
- do not hide null values without documentation

The staging layer should make data issues visible and traceable. Business-level filtering or deduplication should be handled later only when the rule is clear.

## Out of Scope

The staging layer should not:

- create fact or dimension columns
- calculate KPIs
- join order, customer, seller, product, and payment entities into analytical models
- remove records only because they look duplicated
- hide null values without documentation
- apply business rules that belong to later modeling layers

## Design Principle

Column cleanup in staging should improve consistency without changing business meaning.

Every staging column should still be easy to trace back to the raw source column.

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

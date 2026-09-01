-- M10 U1.3-D3-A
-- Validate the existing full-history marts before adopting them
-- as the initial M10 window-control baseline.
--
-- DBT_PROJECT_ID is injected at execution time via envsubst.
--
-- Validation goals:
--   1. Determine the existing source history boundary.
--   2. Derive the candidate [start, end) baseline.
--   3. Verify transactional fact coverage against staging.
--   4. Do not modify any control state or mart data.

with order_bounds as (

    select
        min(order_purchase_timestamp) as min_order_purchase_timestamp,
        max(order_purchase_timestamp) as max_order_purchase_timestamp,
        count(*) as staging_order_count
    from `${DBT_PROJECT_ID}.olist_staging.stg_orders`

),

order_validation as (

    select
        'fct_orders' as model_name,

        (
            select count(*)
            from `${DBT_PROJECT_ID}.olist_staging.stg_orders`
        ) as source_row_count,

        (
            select count(*)
            from `${DBT_PROJECT_ID}.olist_marts.fct_orders`
        ) as fact_row_count,

        (
            select count(*)
            from `${DBT_PROJECT_ID}.olist_staging.stg_orders` as source
            left join `${DBT_PROJECT_ID}.olist_marts.fct_orders` as fact
                on source.order_id = fact.order_id
            where fact.order_id is null
        ) as missing_in_fact,

        (
            select count(*)
            from `${DBT_PROJECT_ID}.olist_marts.fct_orders` as fact
            left join `${DBT_PROJECT_ID}.olist_staging.stg_orders` as source
                on fact.order_id = source.order_id
            where source.order_id is null
        ) as unexpected_in_fact

),

item_validation as (

    select
        'fct_order_items' as model_name,

        (
            select count(*)
            from `${DBT_PROJECT_ID}.olist_staging.stg_order_items`
        ) as source_row_count,

        (
            select count(*)
            from `${DBT_PROJECT_ID}.olist_marts.fct_order_items`
        ) as fact_row_count,

        (
            select count(*)
            from `${DBT_PROJECT_ID}.olist_staging.stg_order_items` as source
            left join `${DBT_PROJECT_ID}.olist_marts.fct_order_items` as fact
                on source.order_id = fact.order_id
               and source.order_item_id = fact.order_item_id
            where fact.order_id is null
        ) as missing_in_fact,

        (
            select count(*)
            from `${DBT_PROJECT_ID}.olist_marts.fct_order_items` as fact
            left join `${DBT_PROJECT_ID}.olist_staging.stg_order_items` as source
                on fact.order_id = source.order_id
               and fact.order_item_id = source.order_item_id
            where source.order_id is null
        ) as unexpected_in_fact

),

payment_validation as (

    select
        'fct_order_payments' as model_name,

        (
            select count(*)
            from `${DBT_PROJECT_ID}.olist_staging.stg_order_payments`
        ) as source_row_count,

        (
            select count(*)
            from `${DBT_PROJECT_ID}.olist_marts.fct_order_payments`
        ) as fact_row_count,

        (
            select count(*)
            from `${DBT_PROJECT_ID}.olist_staging.stg_order_payments` as source
            left join `${DBT_PROJECT_ID}.olist_marts.fct_order_payments` as fact
                on source.order_id = fact.order_id
               and source.payment_sequential = fact.payment_sequential
            where fact.order_id is null
        ) as missing_in_fact,

        (
            select count(*)
            from `${DBT_PROJECT_ID}.olist_marts.fct_order_payments` as fact
            left join `${DBT_PROJECT_ID}.olist_staging.stg_order_payments` as source
                on fact.order_id = source.order_id
               and fact.payment_sequential = source.payment_sequential
            where source.order_id is null
        ) as unexpected_in_fact

),

review_validation as (

    select
        'fct_order_reviews' as model_name,

        (
            select count(*)
            from `${DBT_PROJECT_ID}.olist_staging.stg_order_reviews`
        ) as source_row_count,

        (
            select count(*)
            from `${DBT_PROJECT_ID}.olist_marts.fct_order_reviews`
        ) as fact_row_count,

        (
            select count(*)
            from `${DBT_PROJECT_ID}.olist_staging.stg_order_reviews` as source
            left join `${DBT_PROJECT_ID}.olist_marts.fct_order_reviews` as fact
                on source.review_id = fact.review_id
               and source.order_id = fact.order_id
            where fact.order_id is null
        ) as missing_in_fact,

        (
            select count(*)
            from `${DBT_PROJECT_ID}.olist_marts.fct_order_reviews` as fact
            left join `${DBT_PROJECT_ID}.olist_staging.stg_order_reviews` as source
                on fact.review_id = source.review_id
               and fact.order_id = source.order_id
            where source.order_id is null
        ) as unexpected_in_fact

)

select
    'SOURCE_HISTORY_BOUNDARY' as record_type,
    cast(null as string) as model_name,

    staging_order_count as source_row_count,
    cast(null as int64) as fact_row_count,
    cast(null as int64) as missing_in_fact,
    cast(null as int64) as unexpected_in_fact,

    min_order_purchase_timestamp,
    max_order_purchase_timestamp,

    timestamp_trunc(
        min_order_purchase_timestamp,
        day
    ) as candidate_baseline_start,

    timestamp_add(
        timestamp_trunc(
            max_order_purchase_timestamp,
            day
        ),
        interval 1 day
    ) as candidate_baseline_end

from order_bounds

union all

select
    'FACT_VALIDATION' as record_type,
    model_name,
    source_row_count,
    fact_row_count,
    missing_in_fact,
    unexpected_in_fact,

    cast(null as timestamp) as min_order_purchase_timestamp,
    cast(null as timestamp) as max_order_purchase_timestamp,
    cast(null as timestamp) as candidate_baseline_start,
    cast(null as timestamp) as candidate_baseline_end

from (

    select * from order_validation

    union all

    select * from item_validation

    union all

    select * from payment_validation

    union all

    select * from review_validation

)

order by
    record_type desc,
    model_name
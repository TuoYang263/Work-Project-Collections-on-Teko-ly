with order_dates as (

    select date(order_purchase_timestamp) as date_day
    from {{ ref('stg_orders') }}
    where order_purchase_timestamp is not null

    union all

    select date(order_approved_at) as date_day
    from {{ ref('stg_orders') }}
    where order_approved_at is not null

    union all

    select date(order_delivered_carrier_date) as date_day
    from {{ ref('stg_orders') }}
    where order_delivered_carrier_date is not null

    union all

    select date(order_delivered_customer_date) as date_day
    from {{ ref('stg_orders') }}
    where order_delivered_customer_date is not null

    union all

    select date(order_estimated_delivery_date) as date_day
    from {{ ref('stg_orders') }}
    where order_estimated_delivery_date is not null

    union all

    select date(shipping_limit_date) as date_day
    from {{ ref('stg_order_items') }}
    where shipping_limit_date is not null

    union all

    select date(review_creation_date) as date_day
    from {{ ref('stg_order_reviews') }}
    where review_creation_date is not null

    union all

    select date(review_answer_timestamp) as date_day
    from {{ ref('stg_order_reviews') }}
    where review_answer_timestamp is not null

),

date_bounds as (

    select
        min(date_day) as min_date,
        max(date_day) as max_date
    from order_dates

),

date_spine as (

    -- Generate one row per calendar day between the minimum and maximum dates
    -- found in the source data. This creates a continuous date spine for the
    -- date dimension, even if some days have no orders.
    select
        date_day
    from date_bounds,
        unnest(generate_date_array(min_date, max_date, interval 1 day)) as date_day

),

final as (

    select
        date_day,

        extract(year from date_day) as year,
        extract(quarter from date_day) as quarter,
        extract(month from date_day) as month,
        format_date('%B', date_day) as month_name,

        extract(isoweek from date_day) as iso_week,
        extract(day from date_day) as day_of_month,
        extract(dayofweek from date_day) as day_of_week,
        format_date('%A', date_day) as day_name,

        date_trunc(date_day, week) as week_start_date,
        date_trunc(date_day, month) as month_start_date,
        date_trunc(date_day, quarter) as quarter_start_date,
        date_trunc(date_day, year) as year_start_date,

        -- BigQuery DAYOFWEEK: 1 = Sunday, 7 = Saturday.
        case
            when extract(dayofweek from date_day) in (1, 7) then true
            else false
        end as is_weekend

    from date_spine

)

select * from final
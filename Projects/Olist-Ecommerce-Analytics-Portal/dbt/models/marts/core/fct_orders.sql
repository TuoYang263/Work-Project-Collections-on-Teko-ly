{{
    config(
        materialized='incremental',
        incremental_strategy='merge',
        unique_key='order_id'
    )
}}

with orders as (

    select
        order_id,
        customer_id,
        order_status,
        order_purchase_timestamp,
        order_approved_at,
        order_delivered_carrier_date,
        order_delivered_customer_date,
        order_estimated_delivery_date
    from {{ ref('int_orders_windowed') }}

),

items as (

    select
        order_id,
        order_total_items,
        distinct_products_count,
        distinct_sellers_count,
        item_total_value,
        freight_total_value,
        order_gross_value,
        first_shipping_limit_date,
        last_shipping_limit_date
    from {{ ref('int_order_items_agg') }}

),

payments as (

    select
        order_id,
        payment_count,
        payment_method_count,
        max_payment_installments,
        payment_total_value,
        avg_payment_value,
        min_payment_value,
        max_payment_value
    from {{ ref('int_order_payments_agg') }}

),

reviews as (

    select
        order_id,
        review_count,
        avg_review_score,
        min_review_score,
        max_review_score,
        review_with_title_count,
        review_with_message_count,
        first_review_creation_date,
        last_review_creation_date,
        first_review_answer_timestamp,
        last_review_answer_timestamp,
        has_review
    from {{ ref('int_order_reviews_agg') }}

),

final as (

    select
        orders.order_id,
        orders.customer_id,
        orders.order_status,

        orders.order_purchase_timestamp,
        orders.order_approved_at,
        orders.order_delivered_carrier_date,
        orders.order_delivered_customer_date,
        orders.order_estimated_delivery_date,

        date(orders.order_purchase_timestamp) as order_purchase_date,
        date(orders.order_approved_at) as order_approved_date,
        date(orders.order_delivered_carrier_date) as order_delivered_carrier_date_day,
        date(orders.order_delivered_customer_date) as order_delivered_customer_date_day,
        date(orders.order_estimated_delivery_date) as order_estimated_delivery_date_day,

        coalesce(items.order_total_items, 0) as order_total_items,
        coalesce(items.distinct_products_count, 0) as distinct_products_count,
        coalesce(items.distinct_sellers_count, 0) as distinct_sellers_count,
        coalesce(items.item_total_value, 0) as item_total_value,
        coalesce(items.freight_total_value, 0) as freight_total_value,
        coalesce(items.order_gross_value, 0) as order_gross_value,
        items.first_shipping_limit_date,
        items.last_shipping_limit_date,

        coalesce(payments.payment_count, 0) as payment_count,
        coalesce(payments.payment_method_count, 0) as payment_method_count,
        payments.max_payment_installments,
        coalesce(payments.payment_total_value, 0) as payment_total_value,
        payments.avg_payment_value,
        payments.min_payment_value,
        payments.max_payment_value,

        coalesce(reviews.review_count, 0) as review_count,
        reviews.avg_review_score,
        reviews.min_review_score,
        reviews.max_review_score,
        coalesce(reviews.review_with_title_count, 0) as review_with_title_count,
        coalesce(reviews.review_with_message_count, 0) as review_with_message_count,
        reviews.first_review_creation_date,
        reviews.last_review_creation_date,
        reviews.first_review_answer_timestamp,
        reviews.last_review_answer_timestamp,
        coalesce(reviews.has_review, false) as has_review,

        case
            when orders.order_status = 'delivered' then true
            else false
        end as is_delivered,

        case
            when orders.order_delivered_customer_date is not null
                and orders.order_estimated_delivery_date is not null
                and date(orders.order_delivered_customer_date) > date(orders.order_estimated_delivery_date)
                then true
            else false
        end as is_late_delivery,

        date_diff(
            date(orders.order_approved_at),
            date(orders.order_purchase_timestamp),
            day
        ) as days_purchase_to_approval,

        date_diff(
            date(orders.order_delivered_carrier_date),
            date(orders.order_purchase_timestamp),
            day
        ) as days_purchase_to_carrier,

        date_diff(
            date(orders.order_delivered_customer_date),
            date(orders.order_purchase_timestamp),
            day
        ) as days_purchase_to_customer_delivery,

        date_diff(
            date(orders.order_delivered_customer_date),
            date(orders.order_estimated_delivery_date),
            day
        ) as days_estimated_vs_actual_delivery

    from orders
    left join items
        on orders.order_id = items.order_id
    left join payments
        on orders.order_id = payments.order_id
    left join reviews
        on orders.order_id = reviews.order_id

)

select * from final
{{
    config(
        materialized='incremental',
        incremental_strategy='merge',
        unique_key='order_item_key'
    )
}}

with order_items as (

    select
        order_id,
        order_item_id,
        product_id,
        seller_id,
        shipping_limit_date,
        price,
        freight_value
    from {{ ref('stg_order_items') }}

),

orders as (

    select
        order_id,
        customer_id,
        order_status,
        order_purchase_timestamp
    from {{ ref('int_orders_windowed') }}

),

final as (

    select
        concat(
            order_items.order_id,
            '-',
            cast(order_items.order_item_id as string)
        ) as order_item_key,
    
        order_items.order_id,
        order_items.order_item_id,
        orders.customer_id,
        order_items.product_id,
        order_items.seller_id,

        orders.order_status,
        orders.order_purchase_timestamp,
        date(orders.order_purchase_timestamp) as order_purchase_date,

        order_items.shipping_limit_date,
        date(order_items.shipping_limit_date) as shipping_limit_date_day,

        order_items.price,
        order_items.freight_value,
        round(
            coalesce(order_items.price, 0) + coalesce(order_items.freight_value, 0),
            2
        ) as item_gross_value
    
    from orders
    inner join order_items
        on orders.order_id = order_items.order_id

)

select * from final
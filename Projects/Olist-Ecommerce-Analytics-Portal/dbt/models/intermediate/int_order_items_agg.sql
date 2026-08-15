with orders as (

    select
        order_id
    from {{ ref('int_orders_windowed') }}

),

order_items as (

    select
        items.order_id,
        items.order_item_id,
        items.product_id,
        items.seller_id,
        items.shipping_limit_date,
        items.price,
        items.freight_value

    from orders
    inner join {{ ref('stg_order_items') }} as items
        on orders.order_id = items.order_id

),

aggregated as (

    select
        order_id,

        count(*) as order_total_items,
        count(distinct product_id) as distinct_products_count,
        count(distinct seller_id) as distinct_sellers_count,

        round(sum(price), 2) as item_total_value,
        round(sum(freight_value), 2) as freight_total_value,
        round(
            sum(
                coalesce(price, 0)
                + coalesce(freight_value, 0)
            ),
            2
        ) as order_gross_value,

        min(shipping_limit_date) as first_shipping_limit_date,
        max(shipping_limit_date) as last_shipping_limit_date

    from order_items
    group by order_id

)

select * from aggregated
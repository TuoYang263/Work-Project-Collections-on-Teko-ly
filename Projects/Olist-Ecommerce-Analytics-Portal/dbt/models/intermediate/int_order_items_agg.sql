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

aggregated as (

    select
        order_id,

        count(*) as order_total_items,
        count(distinct product_id) as distinct_products_count,
        count(distinct seller_id) as distinct_sellers_count,

        round(sum(price), 2) as item_total_value,
        round(sum(freight_value), 2) as freight_total_value,
        round(sum(coalesce(price, 0) + coalesce(freight_value, 0)), 2) as order_gross_value,

        min(shipping_limit_date) as first_shipping_limit_date,
        max(shipping_limit_date) as last_shipping_limit_date

    from order_items
    group by order_id

)

select * from aggregated
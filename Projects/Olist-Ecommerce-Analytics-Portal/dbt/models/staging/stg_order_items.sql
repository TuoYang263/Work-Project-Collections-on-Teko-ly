with source as (

    select *
    from {{ source('olist_raw', 'raw_order_items') }}

),

renamed as (

    select
        order_id,
        cast(order_item_id as int64) as order_item_id,
        product_id,
        seller_id,
        cast(shipping_limit_date as timestamp) as shipping_limit_date,
        cast(price as numeric) as price,
        cast(freight_value as numeric) as freight_value

    from source

)

select *
from renamed
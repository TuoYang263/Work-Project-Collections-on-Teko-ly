with source as (

    select *
    from {{ source('olist_raw', 'raw_order_payments') }}

),

renamed as (

    select
        order_id,
        cast(payment_sequential as int64) as payment_sequential,
        lower(trim(payment_type)) as payment_type,
        cast(payment_installments as int64) as payment_installments,
        cast(payment_value as numeric) as payment_value

    from source

)

select *
from renamed
{{
    config(
        materialized='incremental',
        incremental_strategy='merge',
        unique_key='order_payment_key'
    )
}}

with order_payments as (

    select
        order_id,
        payment_sequential,
        payment_type,
        payment_installments,
        payment_value
    from {{ ref('stg_order_payments') }}

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
            order_payments.order_id,
            '-',
            cast(order_payments.payment_sequential as string)
        ) as order_payment_key,

        order_payments.order_id,
        order_payments.payment_sequential,
        orders.customer_id,
        
        orders.order_status,
        orders.order_purchase_timestamp,
        date(orders.order_purchase_timestamp) as order_purchase_date,

        order_payments.payment_type,
        order_payments.payment_installments,
        order_payments.payment_value

    from orders
    inner join order_payments
        on orders.order_id = order_payments.order_id

)

select * from final
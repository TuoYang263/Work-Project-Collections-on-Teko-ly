with orders as (

    select
        order_id
    from {{ ref('int_orders_windowed') }}

),

order_payments as (

    select
        payments.order_id,
        payments.payment_sequential,
        payments.payment_type,
        payments.payment_installments,
        payments.payment_value

    from orders
    inner join {{ ref('stg_order_payments') }} as payments
        on orders.order_id = payments.order_id

),

aggregated as (

    select
        order_id,

        count(*) as payment_count,
        count(distinct payment_type) as payment_method_count,

        max(payment_installments) as max_payment_installments,
        round(sum(payment_value), 2) as payment_total_value,

        round(avg(payment_value), 2) as avg_payment_value,
        round(min(payment_value), 2) as min_payment_value,
        round(max(payment_value), 2) as max_payment_value

    from order_payments
    group by order_id

)

select * from aggregated
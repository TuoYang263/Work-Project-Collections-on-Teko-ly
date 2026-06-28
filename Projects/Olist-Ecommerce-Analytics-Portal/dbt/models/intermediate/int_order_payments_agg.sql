with order_payments as (

    select
        order_id,
        payment_sequential,
        payment_type,
        payment_installments,
        payment_value
    from {{ ref('stg_order_payments') }}

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
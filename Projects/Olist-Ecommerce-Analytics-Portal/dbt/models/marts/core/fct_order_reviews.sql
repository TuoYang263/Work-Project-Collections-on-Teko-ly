with order_reviews as (

    select
        review_id,
        order_id,
        review_score,
        review_comment_title,
        review_comment_message,
        review_creation_date,
        review_answer_timestamp
    from {{ ref('stg_order_reviews') }}

),

orders as (

    select
        order_id,
        customer_id,
        order_status,
        order_purchase_timestamp
    from {{ ref('stg_orders') }}

),

final as (

    select
        concat(
            order_reviews.review_id,
            '-',
            order_reviews.order_id
        ) as review_key,

        order_reviews.review_id,
        order_reviews.order_id,
        orders.customer_id,
        
        orders.order_status,
        orders.order_purchase_timestamp,
        date(orders.order_purchase_timestamp) as order_purchase_date,

        order_reviews.review_score,
        order_reviews.review_creation_date,
        date(order_reviews.review_creation_date) as review_creation_date_day,     
        order_reviews.review_answer_timestamp,
        date(order_reviews.review_answer_timestamp) as review_answer_date_day,

        case
            when nullif(trim(order_reviews.review_comment_title), '') is not null then true
            else false
        end as has_review_comment_title,

        case
            when nullif(trim(order_reviews.review_comment_message), '') is not null then true
            else false
        end as has_review_comment_message

    from order_reviews
    left join orders
        on order_reviews.order_id = orders.order_id

)

select * from final
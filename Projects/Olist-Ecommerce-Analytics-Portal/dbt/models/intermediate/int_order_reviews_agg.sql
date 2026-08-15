with orders as (

    select
        order_id
    from {{ ref('int_orders_windowed') }}

),

order_reviews as (

    select
        reviews.review_id,
        reviews.order_id,
        reviews.review_score,
        reviews.review_comment_title,
        reviews.review_comment_message,
        reviews.review_creation_date,
        reviews.review_answer_timestamp

    from orders
    inner join {{ ref('stg_order_reviews') }} as reviews
        on orders.order_id = reviews.order_id

),

aggregated as (

    select
        order_id,

        count(*) as review_count,
        round(avg(review_score), 2) as avg_review_score,
        min(review_score) as min_review_score,
        max(review_score) as max_review_score,

        countif(
            nullif(
                trim(review_comment_title),
                ''
            ) is not null
        ) as review_with_title_count,

        countif(
            nullif(
                trim(review_comment_message),
                ''
            ) is not null
        ) as review_with_message_count,

        min(review_creation_date)
            as first_review_creation_date,

        max(review_creation_date)
            as last_review_creation_date,

        min(review_answer_timestamp)
            as first_review_answer_timestamp,

        max(review_answer_timestamp)
            as last_review_answer_timestamp,

        true as has_review

    from order_reviews
    group by order_id

)

select * from aggregated
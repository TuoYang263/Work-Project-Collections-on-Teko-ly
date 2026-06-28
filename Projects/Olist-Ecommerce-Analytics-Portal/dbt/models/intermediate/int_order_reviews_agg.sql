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

aggregated as (

    select
        order_id,

        count(*) as review_count,
        round(avg(review_score), 2) as avg_review_score,
        min(review_score) as min_review_score,
        max(review_score) as max_review_score,

        countif(nullif(trim(review_comment_title), '') is not null) as review_with_title_count,
        countif(nullif(trim(review_comment_message), '') is not null) as review_with_message_count,

        min(review_creation_date) as first_review_creation_date,
        max(review_creation_date) as last_review_creation_date,
        min(review_answer_timestamp) as first_review_answer_timestamp,
        max(review_answer_timestamp) as last_review_answer_timestamp,

        true as has_review

    from order_reviews
    group by order_id

)

select * from aggregated
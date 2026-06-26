with source as (

    select *
    from {{ source('olist_raw', 'raw_sellers') }}

),

renamed as (

    select
        seller_id,
        cast(seller_zip_code_prefix as string) as seller_zip_code_prefix,
        lower(trim(seller_city)) as seller_city,
        upper(trim(seller_state)) as seller_state

    from source

)

select *
from renamed
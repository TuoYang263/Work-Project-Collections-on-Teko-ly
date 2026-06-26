with source as (

    select *
    from {{ source('olist_raw', 'raw_product_category_translation') }}

),

renamed as (

    select
        lower(trim(product_category_name)) as product_category_name,
        lower(trim(product_category_name_english)) as product_category_name_english

    from source

)

select *
from renamed
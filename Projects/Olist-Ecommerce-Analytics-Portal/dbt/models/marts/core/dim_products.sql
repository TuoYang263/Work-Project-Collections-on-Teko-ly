with products as (

    select
        product_id,
        product_category_name,
        product_name_length,
        product_description_length,
        product_photos_qty,
        product_weight_g,
        product_length_cm,
        product_height_cm,
        product_width_cm
    from {{ ref('stg_products') }}
),

category_translation as (

    select
        product_category_name,
        product_category_name_english
    from {{ ref('stg_product_category_translation') }}

),

joined as (

    select
        products.product_id,
        products.product_category_name,
        coalesce(
            category_translation.product_category_name_english,
            products.product_category_name,
            'unknown'
        ) as product_category_name_english,
        products.product_name_length,
        products.product_description_length,
        products.product_photos_qty,
        products.product_weight_g,
        products.product_length_cm,
        products.product_height_cm,
        products.product_width_cm
    from products
    left join category_translation
        on products.product_category_name = category_translation.product_category_name

)

select * from joined
with source as (

    select *
    from {{ source('olist_raw', 'raw_geolocation') }}

),

renamed as (

    select
        cast(geolocation_zip_code_prefix as string) as geolocation_zip_code_prefix,
        cast(geolocation_lat as float64) as geolocation_lat,
        cast(geolocation_lng as float64) as geolocation_lng,
        lower(trim(geolocation_city)) as geolocation_city,
        upper(trim(geolocation_state)) as geolocation_state

    from source

)

select *
from renamed
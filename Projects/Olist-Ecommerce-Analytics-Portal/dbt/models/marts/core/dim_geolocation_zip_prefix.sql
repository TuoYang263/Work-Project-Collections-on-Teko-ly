with geolocation as (

    select
        geolocation_zip_code_prefix,
        geolocation_lat,
        geolocation_lng,
        geolocation_city,
        geolocation_state
    from {{ ref('stg_geolocation') }}

),

geo_agg as (

    -- Aggregate coordinates per zip code prefix.
    -- Median coordinates are kept as the preferred representative location
    -- because they are less sensitive to outliers than average coordinates.
    select
        geolocation_zip_code_prefix,
        round(avg(geolocation_lat), 6) as avg_geolocation_lat,
        round(avg(geolocation_lng), 6) as avg_geolocation_lng,
        round(approx_quantiles(geolocation_lat, 100)[offset(50)], 6) as median_geolocation_lat,
        round(approx_quantiles(geolocation_lng, 100)[offset(50)], 6) as median_geolocation_lng,

        count(*) as geolocation_record_count

    from geolocation
    group by geolocation_zip_code_prefix

),

city_state_counts as (

    -- Count how often each city/state combination appears for each zip code prefix.
    select
        geolocation_zip_code_prefix,
        geolocation_city,
        geolocation_state,
        count(*) as city_state_record_count

    from geolocation
    group by
        geolocation_zip_code_prefix,
        geolocation_city,
        geolocation_state
        
),

representative_city_state as (

    -- Choose the most frequent city/state combination per zip code prefix.
    -- Alphabetical ordering is used as a deterministic tie-breaker.
    select
        geolocation_zip_code_prefix,
        geolocation_city,
        geolocation_state
    from city_state_counts
    qualify row_number() over (
        partition by geolocation_zip_code_prefix
        order by
            city_state_record_count desc,
            geolocation_city,
            geolocation_state
    ) = 1

),

final as (

    select
        geo_agg.geolocation_zip_code_prefix,

        representative_city_state.geolocation_city,
        representative_city_state.geolocation_state,
        
        geo_agg.median_geolocation_lat as representative_geolocation_lat,
        geo_agg.median_geolocation_lng as representative_geolocation_lng,

        geo_agg.avg_geolocation_lat,
        geo_agg.avg_geolocation_lng,
        geo_agg.geolocation_record_count

    from geo_agg
    left join representative_city_state
        on geo_agg.geolocation_zip_code_prefix
            = representative_city_state.geolocation_zip_code_prefix
        
)

select *
from final
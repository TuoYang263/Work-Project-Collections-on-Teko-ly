{% set control_window_start = var(
    'control_window_start',
    none
) %}

{% set control_window_end = var(
    'control_window_end',
    none
) %}


{% if (
    control_window_start is none
    and control_window_end is not none
) or (
    control_window_start is not none
    and control_window_end is none
) %}

    {{
        exceptions.raise_compiler_error(
            "control_window_start and "
            "control_window_end must either "
            "both be provided or both be omitted"
        )
    }}

{% endif %}


with orders as (

    select
        order_id,
        customer_id,
        order_status,
        order_purchase_timestamp,
        order_approved_at,
        order_delivered_carrier_date,
        order_delivered_customer_date,
        order_estimated_delivery_date

    from {{ ref('stg_orders') }}

    {% if (
        control_window_start is not none
        and control_window_end is not none
    ) %}

    where order_purchase_timestamp
        >= cast('{{ control_window_start }}' as timestamp)

      and order_purchase_timestamp
        < cast('{{ control_window_end }}' as timestamp)

    {% endif %}

)

select *
from orders
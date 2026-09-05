CREATE SCHEMA IF NOT EXISTS `balmy-nuance-468118-g4.olist_analytics`
OPTIONS (
  location = "EU"
);

CREATE OR REPLACE VIEW
  `balmy-nuance-468118-g4.olist_analytics.analytics_kpi_summary`
AS
SELECT
  COUNT(*) AS order_count,

  ROUND(
    SUM(orders.order_gross_value),
    2
  ) AS gmv,

  ROUND(
    SAFE_DIVIDE(
      SUM(orders.order_gross_value),
      COUNT(*)
    ),
    2
  ) AS aov,

  MIN(orders.order_purchase_date) AS first_order_date,
  MAX(orders.order_purchase_date) AS last_order_date

FROM `balmy-nuance-468118-g4.olist_marts.fct_orders` AS orders

WHERE orders.order_purchase_timestamp < (
  SELECT last_successful_window_end
  FROM `balmy-nuance-468118-g4.olist_control.pipeline_control_state`
  WHERE pipeline_name = "olist-dbt-build-job"
    AND environment = "prod"
);


CREATE OR REPLACE VIEW
  `balmy-nuance-468118-g4.olist_analytics.analytics_state_summary`
AS

WITH watermark AS (

  SELECT
    last_successful_window_end

  FROM `balmy-nuance-468118-g4.olist_control.pipeline_control_state`

  WHERE pipeline_name = "olist-dbt-build-job"
    AND environment = "prod"

),

state_codes AS (

  SELECT DISTINCT
    customer_state AS state_code

  FROM `balmy-nuance-468118-g4.olist_marts.dim_customers`

  WHERE customer_state IS NOT NULL

),

eligible_orders AS (

  SELECT
    orders.*,
    customers.customer_state AS state_code

  FROM `balmy-nuance-468118-g4.olist_marts.fct_orders` AS orders

  JOIN `balmy-nuance-468118-g4.olist_marts.dim_customers` AS customers
    ON customers.customer_id = orders.customer_id

  CROSS JOIN watermark

  WHERE
    orders.order_purchase_timestamp
      < watermark.last_successful_window_end

)

SELECT
  states.state_code,

  COUNT(orders.order_id) AS order_count,

  COALESCE(
    ROUND(
      SUM(orders.order_gross_value),
      2
    ),
    0
  ) AS gmv,

  COALESCE(
    ROUND(
      SAFE_DIVIDE(
        SUM(orders.order_gross_value),
        COUNT(orders.order_id)
      ),
      2
    ),
    0
  ) AS aov,

  COUNTIF(
    orders.order_delivered_customer_date IS NOT NULL
    AND orders.order_estimated_delivery_date IS NOT NULL
  ) AS delivery_observation_count,

  ROUND(
    SAFE_DIVIDE(
      COUNTIF(orders.is_late_delivery),
      COUNTIF(
        orders.order_delivered_customer_date IS NOT NULL
        AND orders.order_estimated_delivery_date IS NOT NULL
      )
    ),
    4
  ) AS late_delivery_rate,

  COUNTIF(
    orders.avg_review_score IS NOT NULL
  ) AS reviewed_order_count,

  ROUND(
    AVG(orders.avg_review_score),
    2
  ) AS average_review_score

FROM state_codes AS states

LEFT JOIN eligible_orders AS orders
  ON orders.state_code = states.state_code

GROUP BY states.state_code;

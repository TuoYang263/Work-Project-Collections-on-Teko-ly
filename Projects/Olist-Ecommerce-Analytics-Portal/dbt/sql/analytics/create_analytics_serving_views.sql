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
    SUM(order_gross_value),
    2
  ) AS gmv,

  ROUND(
    SAFE_DIVIDE(
      SUM(order_gross_value),
      COUNT(*)
    ),
    2
  ) AS aov,

  MIN(order_purchase_date) AS first_order_date,
  MAX(order_purchase_date) AS last_order_date

FROM `balmy-nuance-468118-g4.olist_marts.fct_orders`;

CREATE OR REPLACE VIEW
  `balmy-nuance-468118-g4.olist_analytics.analytics_state_summary`
AS
SELECT
  customers.customer_state AS state_code,

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

FROM `balmy-nuance-468118-g4.olist_marts.fct_orders` AS orders

JOIN `balmy-nuance-468118-g4.olist_marts.dim_customers` AS customers
  ON customers.customer_id = orders.customer_id

GROUP BY customers.customer_state;

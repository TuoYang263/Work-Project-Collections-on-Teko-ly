-- Create BigQuery monitoring dataset for M8 metadata refresh.
-- This dataset stores append-only dbt pipeline monitoring history.

CREATE SCHEMA IF NOT EXISTS `olist_monitoring`
OPTIONS (
    location = "EU",
    description = "Append-only dbt artifact metadata and pipeline monitoring tables for the Olist analytics project."
);
# Staging Layer Plan

## Purpose

The staging layer standardizes the raw Olist source tables before analytics modeling.

It keeps the data close to the original source while making it easier, safer, and more consistent to use in later transformation layers.

The staging layer is responsible for:

- standardizing column names
- applying safe data type conversions
- handling dates and timestamps consistently
- documenting null values and duplicate records
- keeping one staging table aligned with one raw source table
- making source data easier to validate and maintain

The staging layer is not responsible for:

- building facts or dimensions
- creating business KPIs
- joining multiple business entities into analytical models
- designing Power BI dashboards
- building the React portal
- implementing pipeline monitoring logic

## Dataset Naming

The BigQuery staging dataset will be:

```text
olist_staging
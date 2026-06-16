# Architecture

## Purpose

This document describes the planned architecture for the Olist E-Commerce Analytics & Pipeline Monitoring Portal.

The project is designed as an analytics engineering and BI-facing data project. The main focus is to show how source e-commerce data can be prepared, modeled, validated, monitored, and served for business reporting.

This document is part of **M1 - Project Setup & Source Understanding**. No pipeline, dbt model, BigQuery table, Power BI report, or portal implementation is created in M1.

## High-Level Flow

The planned data flow is:

```text
Olist CSV source data
        ↓
BigQuery raw layer
        ↓
dbt staging models
        ↓
dimensional model
        ↓
reporting marts
        ↓
data quality and pipeline monitoring tables
        ↓
Power BI dashboard + React/Node TypeScript analytics portal
```

## Main Components

### 1. Source Data

The source data comes from the Olist Brazilian E-Commerce public dataset.

The dataset contains e-commerce related tables such as orders, customers, sellers, products, payments, reviews, geolocation, and order items.

In M1, the source data will only be reviewed and documented. No ingestion pipeline is implemented yet.

### 2. BigQuery

BigQuery is planned as the main analytics warehouse.

The planned warehouse layers are:

```text
raw       - source-aligned data loaded from CSV files
staging   - cleaned and renamed source tables
marts     - dimensional and reporting-ready tables
metadata  - pipeline status, data quality results, and freshness checks
```

In M1, only the dataset naming plan will be documented.

### 3. dbt

dbt is planned for SQL-based transformation, testing, and documentation.

The expected dbt layers are:
```text
sources   - source table definitions
staging   - source cleaning and standardization
marts     - facts, dimensions, and reporting models
tests     - data quality checks
```

dbt implementation is outside the scope of M1.

### 4. Dimensional Model

The project will use dimensional modeling to prepare business-friendly reporting tables.

The expected model will include fact tables and dimension tables around e-commerce entities such as orders, customers, products, sellers, payments, reviews, and dates.

The detailed dimensional model will be designed in a later milestone.

### 5. Data Quality and Pipeline Monitoring

The project will include data quality and monitoring outputs to make pipeline status visible.

Planned monitoring areas include:

- source table freshness
- row count checks
- missing value checks
- duplicate key checks
- referential integrity checks
- dbt test results
- pipeline run status
- reporting table availability

In M1, these are only planned as architecture requirements.

### 6. Power BI

Power BI is planned as the main BI dashboard layer.

It will consume reporting-ready marts from BigQuery and present business metrics such as orders, revenue, products, customers, sellers, delivery performance, and reviews.

Power BI development is outside the scope of M1.

### 7. React/Node TypeScript Analytics Portal

A custom React/Node TypeScript portal is planned as an additional analytics and monitoring layer.

The portal is expected to show selected business metrics and pipeline monitoring information, such as data freshness, quality status, and run summaries.

Portal development is outside the scope of M1.

## Deployment Boundary

Deployment is not part of M1.

The first version of the project will focus on local development and a clear BigQuery/dbt workflow.

A scheduled workflow with GitHub Actions may be added later for running dbt jobs or data quality checks.

Azure-based orchestration, such as Azure Container Apps Jobs, may also be considered later if the project needs a more production-like scheduled runner.

## M1 Architecture Scope

M1 only covers:

- project folder structure
- README skeleton
- architecture documentation
- source data overview
- metadata placeholders
- BigQuery naming plan
- GitHub Projects board planning

M1 does not include:

- loading data into BigQuery
- creating dbt models
- building dashboards
- creating the React/Node portal
- deploying the pipeline
- creating Azure resources

## Current Status

```text
M1 - Project Setup & Source Understanding: in progress
```













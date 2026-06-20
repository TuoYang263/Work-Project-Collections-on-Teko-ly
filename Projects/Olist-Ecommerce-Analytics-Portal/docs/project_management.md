# Project Management

## Purpose

This document describes how the Olist E-Commerce Analytics & Pipeline Monitoring Portal is planned and managed.

The goal is to keep the project structured, realistic, and easy to continue across milestones.

This document is part of **M1 - Project Setup & Source Understanding**.

## Project Approach

This project is developed milestone by milestone.

Each milestone should produce a clear and reviewable output, such as documentation, source inventory, BigQuery planning, dbt models, reporting marts, data quality checks, dashboards, or portal features.

The project should avoid doing too many things at once. Each stage should be completed and committed before moving to the next stage.

## Git Workflow

The current development branch is:

```text
feature/olist-analytics-portal
```

The branch is used for M1 project setup and planning work.

The branch should not be merged into `main` until the milestone has a clean structure and reviewable commits.

## Commit Style

Commit messages should be short and clear.

Examples:

```text
chore: initialize Olist analytics portal structure
docs: add README skeleton for Olist analytics portal
docs: add initial architecture plan
docs: add initial source data overview
docs: add source tables inventory
docs: add BigQuery dataset naming plan
```

General commit prefixes:

- `chore:` for setup and maintenance
- `docs:` for documentation
- `data:` for source metadata or sample data notes
- `feat:` for implemented project features
- `fix:` for corrections

## Milestone Plan

| Milestone | Focus | Output |
|---|---|---|
| M1 | Project setup and source understanding | Folder structure, documentation, source inventory, naming plan |
| M2 | BigQuery raw layer | Raw dataset setup and source-aligned tables |
| M3 | dbt staging layer | Source definitions, staging models, basic tests |
| M4 | Dimensional modeling | Fact and dimension tables |
| M5 | Reporting marts and data quality | BI-ready marts and validation checks |
| M6 | Pipeline monitoring | Pipeline status, freshness, and quality monitoring outputs |
| M7 | Power BI dashboard | Business-facing BI report |
| M8 | React/Node TypeScript portal | Custom analytics and monitoring portal |

Only M1 is currently in scope.

## M1 Scope

M1 includes:

- creating the project folder structure
- creating README skeleton
- documenting the planned architecture
- inspecting source CSV files
- documenting source table overview
- creating source table inventory
- defining BigQuery dataset and table naming plan
- preparing GitHub Projects board planning

M1 does not include:

- creating BigQuery datasets
- loading data into BigQuery
- creating dbt models
- building dimensional models
- creating Power BI dashboards
- building the React/Node TypeScript portal
- deploying scheduled pipelines
- creating Azure resources

## Development Principles

The project should follow these principles:

- keep the scope controlled
- document important design decisions
- use clear folder structure
- avoid committing raw data or credentials
- keep source, staging, marts, and monitoring layers separated
- prefer simple and explainable design over unnecessary complexity
- make outputs useful for BI and analytics consumption

## Current Status

```text
- M1 - Project Setup & Source Understanding: completed
- M2 - BigQuery Raw Layer: in progress
- BigQuery raw dataset olist_raw created in EU location
- Raw table loading plan documented
- Loaded first raw table: `olist_raw.raw_product_category_translation`.
- Loaded raw table: `olist_raw.raw_sellers`.
- Loaded raw table: `olist_raw.raw_customers`.
- Loaded raw table: `olist_raw.raw_products`.
- Loaded raw table: `olist_raw.raw_orders`.
- Loaded raw table: `olist_raw.raw_order_items`.
```
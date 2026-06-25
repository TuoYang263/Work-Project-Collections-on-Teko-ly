# Olist E-Commerce Analytics & Pipeline Monitoring Portal

## Overview

This project uses the Olist Brazilian E-Commerce public dataset to build an analytics engineering project around e-commerce reporting.

The focus is not only on creating dashboards, but also on showing how the data is understood, modeled, checked, and prepared for business use.

The project will include BigQuery, dbt, dimensional modeling, data quality checks, pipeline monitoring, Power BI, and a custom React/Node TypeScript analytics portal.

The current milestone is **M1 - Project Setup & Source Understanding**.

## Goals

The main goals of this project are to:

- understand the source data and business entities
- design a clear dimensional model for e-commerce analytics
- prepare BigQuery datasets for raw, staging, and reporting layers
- use dbt for transformation logic and data quality checks
- build reporting-ready tables for BI use
- track data freshness, pipeline status, and quality issues
- present the final results through Power BI and a custom React/Node TypeScript analytics portal

## Current Milestone

### M1 - Project Setup & Source Understanding

M1 is only about setting up the project foundation.

Scope of M1:

- create the project folder structure
- prepare the first documentation files
- review the Olist source data
- document source tables and key fields
- create metadata placeholder files
- draft the BigQuery dataset naming plan
- prepare the GitHub Projects board structure

Out of scope for M1:

- dbt model development
- BigQuery table creation
- Power BI dashboard development
- React/Node portal development
- pipeline deployment
- Azure orchestration

## Planned Data Flow

The planned flow is:

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

Deployment is not part of M1.

The first version will focus on local development and a clear BigQuery/dbt workflow. A scheduled workflow with GitHub Actions may be added later. Azure-based orchestration can also be considered later if the project needs a more production-like runner.

## Repository Structure

```text
.
├── assets/                     # Images, diagrams, and screenshots
├── bi/                         # Power BI files and notes
├── data/                       # Local reference data only
│   ├── raw/                    # Raw files if used locally
│   └── sample/                 # Small sample files
├── dbt/                        # Future dbt project
├── docs/                       # Project documentation
├── metadata/                   # Source metadata and planning files
│   ├── bigquery/               # BigQuery dataset and table naming plans
│   ├── project_management/     # GitHub Projects planning files
│   └── source/                 # Source table notes
├── portal/                     # Future React/Node TypeScript analytics portal
└── sql/                        # SQL exploration and helper scripts
```

## Planned M1 Documentation

The following files will be created during M1:
- `docs/architecture.md`
- `docs/source_data_overview.md`
- `docs/project_management.md`
- `metadata/bigquery/dataset_naming_plan.md`
- `metadata/source/source_tables_inventory.md`
- `metadata/project_management/github_projects_plan.md`

## Status
- M1 - Project Setup & Source Understanding: Completed
- M2 - BigQuery Raw Layer: Completed
- M3 - Staging Layer Planning: Completed
- Current milestone: M4 - dbt Project Setup and Staging Model Implementation

## Note

This is a portfolio project, but it is structured like a practical analytics engineering workflow. The aim is to show clear thinking around data modeling, data quality, and pipeline visibility.


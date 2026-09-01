# Data Engineering Portfolio — Tuo Yang

Hi, I’m Tuo.

I build **reliable data products and production-oriented data pipelines** — from ingestion and transformation to governed analytics, operational monitoring, and lightweight product interfaces.

My current focus is **data engineering / analytics engineering**, with particular interest in data reliability, cloud execution, deterministic quality controls, and clear system boundaries.

This repository contains a small set of projects that best represent that work.

---

## Flagship Project

### Olist E-Commerce Analytics & Pipeline Monitoring Portal

**Cloud data platform + analytics product combining business decisions, pipeline reliability, window control, and operational monitoring**

**Live portal:** https://olist-analytics-portal.onrender.com

*Free-tier demo — the first request after inactivity may take a short time to wake up.*

![Olist Analytics & Reliability Portal](./Projects/Olist-Ecommerce-Analytics-Portal/assets/screenshots/portal/portal-product-overview.png)

This project started as an e-commerce analytics pipeline and evolved into a broader data product with governed BigQuery serving, deterministic reliability review, explicit processing-window semantics, and a public Next.js portal.

Key engineering work:

- Built a layered BigQuery + dbt warehouse with dimensional marts and governed analytics-serving outputs
- Added append-only monitoring from dbt artifacts, preserving model, test, metadata, and lineage history
- Built a deterministic reliability reviewer with explicit `PASS / TRIGGERED / NOT_EVALUATED` semantics
- Implemented forward-only window and watermark control with same-window retry, transactional audit history, and compare-and-set concurrency protection
- Built a public Next.js analytics portal with Brazil state-level decision analytics and validated statistical diagnostics
- Separated Portal and pipeline CI/CD, using GitHub OIDC + Google Workload Identity Federation for short-lived, least-privilege GCP deployment credentials

**Core stack:**

Python · SQL · dbt · BigQuery · Cloud Run · Docker · Next.js · TypeScript · GitHub Actions

View project details:

[`Olist-Ecommerce-Analytics-Portal`](./Projects/Olist-Ecommerce-Analytics-Portal)

---

## Selected Projects

### Public Transport Telemetry Pipeline with Weather Context

**Production-oriented telemetry pipeline with scheduled Azure refresh, Blob-based serving, data quality validation, and dashboard consumption**

Live demo: https://transport-telemetry-dashboard-vs4l.onrender.com

This project models a scheduled public transport telemetry data product. It combines simulated transit telemetry with FMI weather context and HSL route reference data, runs a containerized refresh workflow through Azure Container Apps Jobs, and serves stable Gold-layer outputs through a Streamlit dashboard.

- Designed a Bronze → Silver → Gold pipeline using Spark and Delta-style data modeling
- Built route-level KPIs, pipeline health metrics, and dashboard-ready parquet outputs
- Added data quality and source compatibility validation reports for scheduled refreshes
- Used Azure Blob Storage as a decoupled serving layer between pipeline execution and dashboard consumption
- Containerized and scheduled the refresh workflow with Docker, Azure Container Registry, and Azure Container Apps Jobs
- Documented cost-aware trade-offs between portfolio deployment and production-grade infrastructure

**Design boundary**

This is a scheduled snapshot dashboard, not a live operations monitoring system. Weather is used as contextual external information only, and the project does not infer causal weather impact.

**Core stack:**

PySpark · Azure Container Apps Jobs · Azure Blob Storage · Azure Databricks · Docker · Streamlit · GitHub Actions

View project details:

[`Public-Transport-Telemetry-Pipeline`](./Projects/Public-Transport-Telemetry-Pipeline)

---

### NYC Yellow Taxi Data Engineering Pipeline

**End-to-end batch pipeline demonstrating core data engineering patterns**

Live demo: https://nyc-taxi-dashboard-render.onrender.com

This project covers the full path from ingestion to consumption:

- Designed a medallion-style data model from raw to analytics-ready outputs
- Implemented batch transformations using PySpark
- Loaded analytics tables into BigQuery for downstream querying
- Built a Streamlit dashboard for exploration and reporting
- Structured the project to be cloud-portable and reproducible

**Core stack:**

PySpark · BigQuery · Airflow · Docker · Streamlit · SQL

View project details:

[`NYC-Taxi-Pipeline`](./Projects/NYC-Taxi-Pipeline)

---

### Terraform on AWS with GitHub OIDC

**Compact infrastructure project focused on security and repeatability**

- Provisioned infrastructure using Terraform
- Implemented CI/CD with GitHub Actions OIDC instead of static cloud credentials
- Kept infrastructure and application concerns clearly separated

**Core stack:**

Terraform · AWS · GitHub Actions · OIDC

View project details:

[`Cloud-IaC-Terraform-AWS-OIDC`](./Projects/Cloud-IaC-Terraform-AWS-OIDC)

---

## Analytics & Data Science Background

Before moving toward data engineering and analytics engineering, I worked on analytics and modeling projects.

That background still influences how I design data systems — especially around data meaning, statistical reasoning, validation, and downstream usability.

Selected examples include:

- Stock Market Analysis — time-series exploration
- Consumer Complaint Resolution — NLP-based analysis
- Used Cars Price Prediction — regression and feature engineering
- Power BI Sales Dashboard — business reporting and KPI tracking

These projects remain available under the `Projects/` directory.

---

## Core Skills & Tools

- **Languages:** Python, SQL, TypeScript
- **Data Engineering:** dbt, PySpark, BigQuery, dimensional modeling, medallion architecture, data quality, lineage, incremental processing
- **Reliability & Operations:** deterministic rule systems, monitoring evidence, retries, watermark control, idempotency, transactional state control, compare-and-set concurrency protection
- **Cloud:** Google Cloud Run Jobs, Cloud Scheduler, Artifact Registry, Azure Container Apps Jobs, Azure Blob Storage, Azure Databricks, Azure Functions
- **DevOps & Infrastructure:** Docker, GitHub Actions, OIDC, Workload Identity Federation, Terraform
- **Analytics & Serving:** Next.js, React, Streamlit, Power BI, geospatial analytics, statistical diagnostics
- **Workflow:** Git, reproducible pipelines, explicit system boundaries, cost-aware deployment trade-offs

---

## Contact

If you are reviewing this repository as part of a job application, I recommend starting with the **Olist E-Commerce Analytics & Pipeline Monitoring Portal**.

LinkedIn:  
https://www.linkedin.com/in/tuo-yang-6b772b207/

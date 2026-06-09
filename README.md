# Data Engineering Portfolio — Tuo Yang

Hi, I’m Tuo.

I build **reliable, end-to-end data pipelines** — from raw ingestion and transformation to analytics-ready tables and lightweight dashboards.

My background is in data science and analytics, which helps with data quality checks, business logic, and downstream usability.

Over time, my work has shifted toward **data engineering and production-oriented systems**.

This repository highlights a small set of projects that best represent my current focus.

---

## Featured Project — Primary Focus

### Public Transport Telemetry Pipeline with Weather Context

> This is the most representative project of my current data engineering work.

**Production-oriented telemetry pipeline with Spark-style layers, scheduled Azure refresh, Blob-based serving, data quality validation, and dashboard consumption**

Live demo: https://transport-telemetry-dashboard-vs4l.onrender.com

This project models a scheduled public transport telemetry data product. It combines simulated transit telemetry with FMI weather context and HSL route reference data, runs a containerized refresh workflow through Azure Container Apps Jobs, and serves stable Gold-layer outputs through a Streamlit dashboard.

- Designed a Bronze → Silver → Gold pipeline using Spark and Delta-style data modeling
- Built route-level KPIs, pipeline health metrics, and dashboard-ready parquet outputs
- Added data quality and source compatibility validation reports for scheduled refreshes
- Used Azure Blob Storage as a decoupled serving layer between pipeline execution and dashboard consumption
- Containerized and scheduled the refresh workflow with Docker, Azure Container Registry, and Azure Container Apps Jobs
- Added lightweight dashboard transparency through Azure Function metadata heartbeat and read-only validation artifacts
- Documented cost-aware trade-offs between portfolio deployment and production-grade infrastructure

The focus is on **clear data responsibilities, stable downstream consumption, and practical engineering trade-offs**, rather than feature complexity.

**Design boundary**

This is a scheduled snapshot dashboard, not a live operations monitoring system. Weather is used as contextual external information only, and the project does not infer causal weather impact.

**Tech stack:**  
PySpark · Delta-style Medallion Architecture · Data Quality Validation · Docker · Azure Container Apps Jobs · Azure Container Registry · Azure Blob Storage · Azure Databricks · Azure Functions · Streamlit · Render · GitHub Actions · OpenAI API

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

**Tech stack:**  
PySpark · BigQuery · Airflow · Docker · Streamlit · SQL

View project details:  
[`NYC-Taxi-Pipeline`](./Projects/NYC-Taxi-Pipeline)

---

## Cloud & Infrastructure — Supporting Project

### Terraform on AWS with GitHub OIDC

A compact infrastructure project focused on security and repeatability:

- Provisioned infrastructure using Terraform
- Implemented CI/CD with GitHub Actions OIDC, avoiding static cloud credentials
- Kept infrastructure and application concerns clearly separated

**Tech stack:**  
Terraform · AWS · GitHub Actions

View project details:  
[`Cloud-IaC-Terraform-AWS-OIDC`](./Projects/Cloud-IaC-Terraform-AWS-OIDC)

---

## Analytics & Data Science Background

Before moving into data engineering, I worked on analytics and modeling projects.

That background still informs how I design pipelines — especially around data meaning, correctness, and downstream use.

Selected examples include:

- Stock Market Analysis — time series exploration
- Consumer Complaint Resolution — NLP-based analysis
- Used Cars Price Prediction — regression and feature engineering
- Power BI Sales Dashboard — business reporting and KPI tracking

These projects are available under the `Projects/` directory.

---

## Archived / Learning Projects

This repository also includes earlier work in machine learning, computer vision, and NLP.

They are kept for reference, but they are not the focus of my current work.

---

## Core Skills & Tools

- **Languages:** Python, SQL
- **Data Engineering:** PySpark, Delta-style Medallion Architecture, BigQuery, Airflow, data modeling, data quality checks
- **Cloud & Serving:** Azure Blob Storage, Azure Container Apps Jobs, Azure Container Registry, Azure Databricks, Azure Functions, Render, Streamlit
- **DevOps & Infrastructure:** Docker, containerized batch jobs, Terraform, GitHub Actions, GitHub OIDC
- **Analytics & BI:** Pandas, Power BI, dashboard-ready data products
- **Workflow:** Git, Jupyter, reproducible pipelines, cost-aware deployment trade-offs

---

## Contact

If you are reviewing this repository as part of a job application, the **Public Transport Telemetry Pipeline with Weather Context** is the best place to start.

LinkedIn:  
https://www.linkedin.com/in/tuo-yang-6b772b207/
# Data Engineering Portfolio — Tuo Yang

Hi, I’m Tuo.

I build **reliable, end-to-end data pipelines** — from raw ingestion and transformation to analytics-ready tables and lightweight dashboards.

My background is in data science and analytics, which helps with data quality checks, business logic, and downstream usability.

Over time, my work has shifted toward **data engineering and production-oriented systems**.

This repository highlights a small set of projects that best represent my current focus.

---

## Featured Project — Primary Focus

### Public Transport Telemetry Pipeline

> This is the most representative project of my current work.

**Production-style data pipeline with observability, decoupled serving, and practical engineering trade-offs**

Live demo: https://transport-telemetry-dashboard-vs4l.onrender.com

This project models a realistic telemetry pipeline where operational signals and external data (weather) are integrated into a unified data flow.

- Implemented a Bronze → Silver → Gold architecture using Spark and Delta-style design
- Unified telemetry and weather data into a shared event model
- Built event-time windowed aggregation for operational metrics
- Designed pipeline observability signals (freshness, lag, duplicates, volume)
- Exported precomputed outputs (parquet) for stable downstream consumption
- Deployed a lightweight dashboard reading from Azure Blob Storage
- Automated pipeline execution and refresh using GitHub Actions

The focus is on **clarity, stability, and realistic trade-offs**, rather than system complexity.

**Design boundary**

This project intentionally avoids platform-level complexity (e.g. streaming infrastructure, orchestration frameworks, API layers) to keep the system focused, explainable, and maintainable. The emphasis is on clear data modeling and reliable downstream consumption rather than full platform coverage.

**Optional extension — explanation layer**

An optional explanation layer can be added on top of the curated KPI outputs to provide lightweight summaries and guided Q&A.

This layer would operate on precomputed data and would not modify the pipeline itself, keeping the core system deterministic and maintainable.

**Tech stack:**  
PySpark · Delta Lake · SQL · GitHub Actions · Azure Blob Storage · Streamlit

View project details:  
[`Public-Transport-Telemetry-Pipeline`](./Projects/Public-Transport-Telemetry-Pipeline)

---

### NYC Yellow Taxi Data Engineering Pipeline  
**End-to-end batch pipeline demonstrating core data engineering patterns**

Live demo: https://nyc-taxi-dashboard-render.onrender.com

This project covers the full path from ingestion to consumption:

- Designed a medallion-style data model (raw → cleaned → aggregated)
- Implemented batch transformations using PySpark
- Loaded analytics tables into BigQuery for downstream querying
- Built dashboards with Streamlit for exploration and reporting
- Structured to be cloud-portable and reproducible

**Tech stack:**  
PySpark · BigQuery · Airflow · Docker · Streamlit · SQL

View project details:  
[`NYC-Taxi-Pipeline`](./Projects/NYC-Taxi-Pipeline)

---

## Cloud & Infrastructure — Supporting Project

### Terraform on AWS with GitHub OIDC

A compact infrastructure project focused on security and repeatability:

- Provisioned infrastructure using Terraform
- Implemented CI/CD with GitHub Actions OIDC (no static credentials)
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

- Stock Market Analysis (time series exploration)
- Consumer Complaint Resolution (NLP-based analysis)
- Used Cars Price Prediction (regression and feature engineering)
- Power BI Sales Dashboard (business reporting and KPI tracking)

These projects are available under the `Projects/` directory.

---

## Archived / Learning Projects

This repository also includes earlier work in machine learning, computer vision, and NLP.

They are kept for reference, but are not the focus of my current work.

---

## Core Skills & Tools

- **Languages:** Python, SQL  
- **Data Engineering:** PySpark, BigQuery, Airflow, Docker  
- **Analytics & BI:** Pandas, Streamlit, Power BI  
- **Cloud & DevOps:** Terraform, GitHub Actions  
- **Workflow:** Git, Jupyter, reproducible pipelines

---

## Contact

If you are reviewing this repository as part of a job application,  
the **Public Transport Telemetry Pipeline** is the best place to start.

LinkedIn:  
https://www.linkedin.com/in/tuo-yang-6b772b207/
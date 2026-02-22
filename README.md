# Data Engineering Portfolio — Tuo Yang

Hi, I’m Tuo.

I build **reliable, end-to-end data pipelines** — from raw ingestion and transformation to analytics-ready tables and dashboards.

My background is in data science and analytics, which helps with data quality checks, business logic, and downstream use cases.
  
Over time, my work has shifted toward **data engineering and production systems**.

This repository highlights the projects that best represent my current focus.

---

## Featured Project — Primary Focus

### NYC Yellow Taxi Data Engineering Pipeline  
**End-to-end batch data pipeline with analytics and visualization**

This project covers the full path from ingestion to consumption:

- Designed a medallion-style data model (raw → cleaned → aggregated)
- Implemented batch transformations using PySpark
- Loaded analytics tables into BigQuery for downstream querying
- Built interactive dashboards with Streamlit for exploration and reporting
- Structured to be cloud-portable and migration-ready

**Tech stack:**  
PySpark · BigQuery · Airflow · Docker · Streamlit · SQL

View pipeline design, code, and usage details:  
[`NYC-Taxi-Pipeline`](./Projects/NYC-Taxi-Pipeline)

---

### Public Transport Telemetry Pipeline  
**Production-style telemetry pipeline with operational metrics and pipeline health monitoring**

This project models how telemetry and external signals can be ingested, aggregated, and monitored in a layered data pipeline:

- Implemented Bronze → Silver → Gold telemetry architecture
- Unified telemetry and weather observations into a shared event model
- Built event-time windowed aggregation for operational metrics
- Modeled pipeline observability metrics (freshness, lag, duplicates, volume)
- Structured for migration to managed environments such as Azure Databricks

**Tech stack:**  
PySpark · Delta Lake · SQL · Databricks-ready design

View pipeline design and implementation:  
[`Public-Transport-Telemetry-Pipeline`](./Projects/Public-Transport-Telemetry-Pipeline)

---

## Cloud & Infrastructure — Supporting Project

### Terraform on AWS with GitHub OIDC

A compact infrastructure project with an emphasis on security and repeatability:

- Provisioned resources using Terraform
- Set up CI/CD with GitHub Actions OIDC (no static credentials)
- Kept infrastructure and application logic separated

**Tech stack:**  
Terraform · AWS · GitHub Actions

View infrastructure setup and CI/CD configuration:  
[`Cloud-IaC-Terraform-AWS-OIDC`](./Projects/Cloud-IaC-Terraform-AWS-OIDC)

---

## Analytics & Data Science Background

Before moving into data engineering, I worked on analytics and modeling projects.  
That background still informs how I design pipelines — especially around data meaning, correctness, and business context.

Selected examples include:

- Stock Market Analysis (time series exploration and trend analysis)
- Consumer Complaint Resolution (NLP-driven analysis)
- Used Cars Price Prediction (regression and feature engineering)
- Power BI Sales Dashboard (business reporting and KPI tracking)

Related analytics and modeling projects are available under the `Projects/` directory.

---

## Archived / Learning Projects

This repository also includes earlier projects in machine learning, computer vision, and NLP from coursework and self-study.

They are kept for reference, but they are not the focus of my current work.

Archived projects are available under the `Projects/` directory for reference.

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
the **NYC Yellow Taxi pipeline** is the best place to start.

LinkedIn:  
https://www.linkedin.com/in/tuo-yang-6b772b207/
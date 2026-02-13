# NYC Yellow Taxi Data Pipeline

This project implements a production-style data pipeline for NYC Yellow Taxi trip data.

It ingests raw trip records, transforms them using PySpark, stores analytics-ready tables in BigQuery, and exposes key metrics through a Streamlit dashboard.

The system is fully containerized and orchestrated with Airflow.  
Data processing, orchestration, and visualization are intentionally separated so that each layer can evolve or be migrated independently.

The pipeline follows a Medallion architecture **(Bronze → Silver → Gold)**.

Raw data is ingested into a Bronze layer, cleaned and standardized in Silver, and aggregated into analytics-ready Gold tables.

This structure makes the pipeline easier to reason about, extend, and migrate to managed platforms if needed.

---

### Tech Stack

- **Docker** – Isolates Airflow and Streamlit as independent services
- **Airflow** – Orchestrates the end-to-end pipeline  
- **PySpark** – Handles data cleaning and transformation beyond notebook scale  
- **Delta Lake** – Supports Bronze / Silver / Gold layering and schema evolution  
- **Google BigQuery** – Serves as the analytics warehouse
- **Streamlit** – Provides a lightweight dashboard for key metrics

---

### What Is Implemented

- End-to-end ELT pipeline from raw data to analytics tables
- Medallion architecture with Bronze, Silver, and Gold layers
- Airflow DAGs for orchestration and scheduling
- BigQuery tables for aggregated analytics
- Streamlit dashboard for trend analysis and zone-level insights

---

### System Architecture

The pipeline is designed as a set of loosely coupled layers.  
Each layer has a clear responsibility, which simplifies debugging, testing, and future migration.

1. **Raw Data Ingestion**  
   Monthly NYC Yellow Taxi trip data is downloaded as Parquet files.  
   The ingestion step is parameterized by year and month.

2. **Transformation with PySpark**  
   Data is cleaned and standardized using PySpark, including filtering invalid records and deriving time-based features.

3. **Analytics Storage in BigQuery**  
   Cleaned and aggregated datasets are written to BigQuery for downstream analytics.

4. **Orchestration with Airflow**  
   The full pipeline is orchestrated using Airflow DAGs, with each step running as an isolated task.

5. **Visualization with Streamlit**  
   A Streamlit dashboard queries BigQuery directly to visualize trends and zone-level activity.

---

### Live Demo (Streamlit Cloud)

The Streamlit dashboard is deployed on Streamlit Cloud and connects directly to BigQuery. The dashboard is backed by preprocessed summary tables rather than raw trip-level data to keep queries fast and costs predictable.

Live demo:
https://nyc-taxi-dashboard.streamlit.app/

The deployment uses a read-only service account configured via Streamlit Secrets.  
No local credentials or Docker setup are required to access the dashboard.

This project is primarily intended as a portfolio and demo system.
The live dashboard represents the main entry point for reviewers.

---

### Operational note – Streamlit Cloud cold start and auto wake

This dashboard is deployed on Streamlit Cloud.

Inactive apps may go to sleep automatically. When that happens, the first request can be slower because the backend container needs to start again.

To keep the public demo reasonably responsive, two lightweight measures are used:

- a small GitHub Actions workflow that pings the app on a schedule
- an external uptime monitor that periodically checks the public URL

What the GitHub Actions workflow does:

- runs on a simple cron schedule
- calls the Streamlit status endpoint (`/api/v2/app/status`)
- uses a few staggered retries to handle cold start cases
- sends one additional homepage request
- treats 200 / 302 / 303 as acceptable responses

Typical behavior:

- if the app is already running -> returns quickly
- if the app is sleeping -> the first ping triggers backend startup
- retries continue until the backend responds OK

This setup is intentionally minimal and only aims to reduce cold-start friction for reviewers, not to guarantee high availability. This behavior is platform-level and not fully controllable from application code.

---

### Local Setup

This project was developed and tested on **Windows 11 using WSL2 (Ubuntu)**, which provides a Linux-compatible environment inside Windows.  
If you're using **macOS** or **Linux**, you can skip WSL and run everything directly in your terminal.

All components—Airflow, PySpark, and Streamlit—are containerized using Docker Compose, so you don’t need to install them manually.

---

#### Prerequisites

Make sure the following are installed:

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)  
- [Python 3.10+](https://www.python.org/) (only if running any standalone scripts)
- [Google Cloud SDK](https://cloud.google.com/sdk) (`gcloud` CLI)
- A Google Cloud project with BigQuery enabled
- A BigQuery dataset (create manually or let the script handle it)

---

#### Setup Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/nyc-taxi-pipeline.git
   cd nyc-taxi-pipeline
   ```

2. **Copy and configure environment files**

   Create your own `.env` file:
   ```bash
   cp .env.example .env
   ```

   Then set your local user ID (optional, improves file permissions in WSL/Linux):
   ```bash
   export AIRFLOW_UID=$(id -u)
   ```

   Copy your GCP service account key (downloaded from Google Cloud Console):
   ```bash
   cp creds/gcp_example_account.example.json creds/gcp_service_account.json
   ```

3. **Start the application stack**
   Build and run all services with Docker Compose:
   ```bash
   docker compose up --build
   ```

4. **Access the local services**
   - **Airflow UI**: [http://localhost:18080](http://localhost:18080)
     *(username/password: `airflow` / `airflow`)
   - **Streamlit Dashboard**: [http://localhost:8700](http://localhost:8700)

---

### Configuration Overview

The project separates configuration into two layers:

- **Environment variables**  
  Used for secrets and runtime-specific values (e.g. GCP credentials).

- **YAML configuration files**  
  Used for pipeline parameters such as time range, Spark settings, and target tables.

This separation keeps sensitive values out of source control while allowing the pipeline behavior to be adjusted without code changes.

---

### Containerization and CI

Docker is used to ensure local reproducibility of the pipeline components.  
Airflow and Streamlit run as isolated services via Docker Compose during development.

Optional CI workflows are included to support image builds, but containerization is primarily used as a development and testing tool.

---

### Local Run (Optional Quick Start Without Docker)

If you want to run parts of the pipeline locally without Docker (for quick testing or debugging), here’s how:

#### Run Airflow DAG Components

You can run and debug individual DAG steps directly using:

```bash
python airflow_pipeline/scripts/main_debug.py
```

Make sure you have:
- Installed all dependencies from `airflow_pipeline/requirements.txt`
- Set up `.env` and `settings.yaml` in the `airflow_pipeline/config/ directory`

#### Run Streamlit Dashboard
To launch the Streamlit dashboard independently:

```bash
cd streamlit_app
streamlit run app.py
```

Make sure:
- Your `.env` file contains the correct BigQuery credentials
- The summary table already exists in BigQuery
- You’ve installed all dependencies from `streamlit_app/requirements.txt`

This is a lightweight way to test the core features without spinning up Docker.

---

### Deployment Pitfalls and Design Decisions

Several design decisions were made during development to reduce operational risk and simplify deployment:

- **Credential handling**  
  A single `GOOGLE_APPLICATION_CREDENTIALS` convention is used across Airflow and Streamlit to avoid container-specific path issues.

- **Configuration strategy**  
  Non-sensitive settings (such as dataset location) are defined in YAML files instead of environment variables to reduce runtime ambiguity.

- **Local vs cloud execution**  
  The pipeline runs fully in Docker for local reproducibility, while the Streamlit Cloud deployment relies on managed secrets and direct BigQuery access.

---

### Project Structure

The repository is organized to reflect a modular and production-like data engineering pipeline. Here's a quick overview of the main components:

```text
Project Structure

.github/workflows/                # CI setup for GitHub Actions (optional)
├── docker.yml                    # Auto-build Docker image and push on commit
├── wake_streamlit.yml            # Scheduled lightweight ping to reduce Streamlit Cloud cold-start delay

.streamlit/                       # Streamlit UI configuration
├── config.toml                   # Custom theme and layout settings

airflow_pipeline/                 # Main directory for the Airflow ETL pipeline
├── config/                       # Pipeline configuration files
│   ├── airflow.cfg               # Airflow runtime settings
│   └── settings.yaml             # Pipeline parameters (e.g., year/month)
├── dags/                         # DAG definitions for Airflow
│   ├── nyc_taxi_etl_dag.py       # Main DAG for NYC Yellow Taxi data pipeline
│   └── nyc_taxi_medallion_dag.py # DAG: Bronze → Silver → Gold → Export
│   
├── data/                        # Local data storage
│   ├── processed/               # Cleaned CSV outputs
│   ├── raw/                     # Original monthly Parquet files
│   └── tmp/                     # Intermediate temp files (optional)
├── logs/                        # Logs generated during Airflow runs
├── scripts/                     # Python helper modules used in DAG
│   ├── delta_medallion_pipeline.py     # Medallion pipeline (Delta Lake)
│   ├── config.py                # Loads .env and settings.yaml
│   ├── get_bigquery_client.py   # Auth wrapper for BigQuery
│   ├── load_nyc_yellow_taxi_data.py  # Downloads and saves monthly data
│   ├── logger.py                # Custom logging utility
│   ├── main_debug.py            # Local dev entry point for pipeline testing
│   ├── sql_utils.py             # Run SQL queries (BigQuery)
│   ├── transform_trip_data_spark.py  # Cleans data using PySpark
│   └── upload_to_big_query.py   # Uploads results to BigQuery
├── sql/                         # SQL scripts used in the pipeline
│   ├── bq/
│   │   └── create_trip_summary_hourly.sql  # Creates hourly summary table in BigQuery
│   └── raw/
│       └── trip_summary_hourly.sql         # DuckDB-based SQL for local test

creds/                           # Local-only folder for GCP credentials
└── gcp_service_account.example.json  # Example for service account key

notebooks/                       # Jupyter notebooks for exploration
├── 01_explore_raw_data.ipynb    # Schema, volume, and column analysis
├── 02_transform_trip_summary.ipynb  # Local test of data cleaning logic
└── 03_visualize_summary.ipynb   # Plots and KPIs (merged from previous 04)

streamlit_app/                   # Interactive dashboard built with Streamlit
├── external/                    # External resources (e.g. GeoJSON files)
│   └── taxi_zones.geojson       # NYC taxi zones for map rendering
├── pages/                       # Modular pages for dashboard
│   ├── Trend_Viewer.py          # Plots hourly trends over time
│   └── Zone_Heatmap.py          # Visualizes pickup/dropoff heatmap by zone
├── utils/                       # Reusable functions for Streamlit app
│   ├── config.py                # Loads config.yaml and env vars
│   ├── get_bigquery_client.py   # BigQuery access for dashboard
│   ├── logger.py                # Logger for diagnostics
│   └── sql_utils.py             # Shared SQL execution tools
├── config.yaml                  # Dashboard config (e.g., date range, metrics)
├── app.py                       # Main app entry point (use: `streamlit run app.py`)
├── .dockerignore                # Prevents cache and temp files in image
├── Dockerfile                   # Dockerfile for dashboard deployment
└── requirements.txt             # Required Python packages for dashboard

.env.example                     # Template for environment variables
.gitignore                       # Git exclusions
docker-compose.yaml              # Main Docker orchestration file
docker-compose.override.yaml     # Local override (ports, mounts, etc.)
Makefile                         # CLI shortcuts (optional)
pyproject.toml                   # Python project metadata
README.md                        # This documentation
```

---
 
### Possible Extensions

- Incremental loading using partitioned BigQuery tables
- Migration of transformation logic to Databricks
- Managed orchestration with Cloud Composer

The current structure was designed to support these extensions without major refactoring.

---

### Key Takeaways

- Separating orchestration from transformation logic improves maintainability
- Medallion architecture simplifies data quality control and extensibility
- Local containerized development and cloud deployments require different assumptions

---

### Acknowledgements

Special thanks to:

- NYC TLC for making the yellow taxi dataset publicly available  
- Google Cloud’s generous free tier (BigQuery & Storage)  
- Airflow & Streamlit communities for excellent documentation  
- [DataTribe](https://github.com/datatribe-collective-labs/data-engineering-learnings/tree/main) for inspiring the capstone structure and workflow

---

### Contact

Feel free to connect or share feedback:

- **GitHub**: [tuo-yang](https://github.com/TuoYang263)  
- **LinkedIn**: [tuo-yang-linkedin-profile](https://www.linkedin.com/in/tuo-yang-6b772b207/)
- **Email**: yangtuomailbox@gmail.com

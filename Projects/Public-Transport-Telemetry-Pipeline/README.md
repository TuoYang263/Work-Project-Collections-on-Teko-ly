# Public Transport Telemetry Pipeline with Weather Context

## Live Demo

**Dashboard (Render):**  
https://transport-telemetry-dashboard-vs4l.onrender.com

This dashboard reads precomputed Gold-layer parquet outputs from Azure Blob Storage and presents the latest exported snapshot.

> Note: This is a scheduled snapshot dashboard, not a live operations monitoring system. 

Render free-tier cold starts may still occur. External uptime checks are used to reduce startup delay, but the dashboard is not designed as an always-on production service.

---

## Summary

A production-oriented public transport telemetry pipeline with weather context, built as a scheduled snapshot data product.

The project uses Spark and Delta-style Bronze/Silver/Gold processing to generate route-level metrics, pipeline health indicators, and dashboard-ready outputs. Gold-layer parquet files are exported to Azure Blob Storage and served by a Streamlit dashboard deployed on Render.

The system is designed around practical engineering trade-offs: scheduled refresh instead of always-on streaming, precomputed outputs instead of live database queries, optional AI explanation over deterministic facts, and lightweight metadata heartbeat instead of expensive continuous refresh.

---

## What this project demonstrates

- Spark / Delta-style Bronze -> Silver -> Gold pipeline design
- Event-time and ingest-time separation for telemetry data
- Route-level KPI modeling and pipeline health metrics
- HSL route geometry and FMI weather station context
- Azure Blob Storage as a decoupled serving layer
- Azure Databricks Job as an optional cloud execution path
- Azure Function metadata heartbeat for lightweight dashboard transparency
- Optional OpenAI explanation layer over precomputed rule-based facts
- Render dashboard deployment with external uptime checks for cold-start reduction
- Cost-aware portfolio deployment decisions

---

## Engineering Signals

This project is designed to demonstrate practical data engineering judgment rather than feature complexity.

It shows how to:

- separate pipeline execution from dashboard serving
- model Bronze/Silver/Gold responsibilities clearly
- expose deterministic Gold-layer outputs for downstream consumption
- keep AI explanation outside the metric calculation path
- use Azure Blob Storage as a lightweight serving boundary
- validate an optional Azure Databricks execution path without keeping compute always on
- make refresh behavior transparent through a lightweight Azure Function metadata heartbeat
- document cost-aware deployment trade-offs for a personal portfolio environment

---

## Project Preview

A lightweight, production-oriented telemetry pipeline simulating public transport operations in the Helsinki region.

The project focuses on data flow clarity, engineering trade-offs, and delivery of stable, query-ready outputs rather than feature complexity.

![Home Overview](docs/dashboard_home.jpg)

> **Preview highlights:** delivered scope, refresh model, serving design, and the lightweight metadata heartbeat displayed directly on the dashboard home page.

---

## Dashboard Preview

### 1. Home / System Overview

The Home page is shown in the project preview above. It summarizes the delivered scope, core system highlights, and the refresh ownership model. It also surfaces the lightweight Azure Function metadata heartbeat used to improve dashboard transparency.

---

### 2. Pipeline Overview

The Pipeline Overview page presents Gold-layer pipeline metrics, freshness information, and a delivery-oriented summary of the latest exported snapshot.

![Pipeline Overview](docs/dashboard_pipeline_overview.jpg)

---

### 3. Route Performance

The Route Performance page displays route-level KPI outputs designed for analysis-ready consumption, including selected-route metrics and recent delay trends.

![Route Performance](docs/dashboard_route_performance.jpg)

---

### 4. Map View

The Map View combines HSL route geometry, sampled vehicle points, and FMI weather station context. Weather is shown as contextual external information, not as causal impact analysis.

![Map View](docs/dashboard_map_combined.jpg)

---

### 5. Architecture Overview

The architecture view summarizes the end-to-end design: Bronze → Silver → Gold processing, parquet export, Azure Blob serving, Streamlit dashboard consumption, optional Databricks execution, optional OpenAI explanation, and lightweight metadata heartbeat.

![Architecture](docs/architecture.png)

---

## Architecture

The project follows a layered data product design:

```text
Data Sources
  ├── Simulated transport telemetry
  ├── FMI weather observations
  └── HSL GTFS route reference data

Pipeline Processing
  └── Spark / Delta-style Bronze -> Silver -> Gold layers

Cloud Execution
  └── Optional Azure Databricks Job for full scheduled refresh

Serving Layer
  └── Exported Gold parquet outputs uploaded to Azure Blob Storage

Dashboard Layer
  └── Streamlit dashboard deployed on Render

Supporting Services
  ├── Azure Function metadata heartbeat
  ├── Optional OpenAI explanation layer
  └── External uptime checks + GitHub Actions best-effort keepalive
```

The dashboard is intentionally separated from the pipeline execution layer. It reads stable exported files instead of querying live processing systems.

---

## Refresh & Serving Model

This project uses a scheduled snapshot refresh model.

The full data refresh is owned by an optional Azure Databricks Job:

1. Run the Bronze -> Silver -> Gold pipeline
2. Export dashboard-ready Gold outputs as parquet files
3. Upload exported parquet files to Azure Blob Storage
4. Let the Render-hosted Streamlit dashboard read the latest exported outputs

GitHub Actions is kept as a manual fallback and best-effort keepalive mechanism. It is not treated as the primary production scheduler.

Azure Function writes a lightweight metadata heartbeat file to Azure Blob Storage. This metadata does not refresh Gold metrics, trigger Databricks, or fetch HSL/FMI data. It only improves dashboard transparency by showing when the lightweight metadata check last ran.

External uptime checks ping the Streamlit endpoint to reduce Render free-tier cold starts. This improves demo availability, but the dashboard is not designed as an always-on production service.

---

## Dashboard Features

### Home

The Home page summarizes the delivered scope, system highlights, refresh model, and dashboard metadata heartbeat.

It explains the difference between:

- full data refresh
- lightweight metadata heartbeat
- dashboard serving
- keepalive behavior

### Pipeline Overview

The Pipeline Overview page shows recent Gold-layer pipeline metrics, including:

- average ingest delay
- processed event count
- data quality status
- rule-based operational snapshot
- optional AI-generated explanation

The AI explanation is generated only from precomputed dashboard facts.

### Route Performance

The Route Performance page provides route-level KPIs and recent historical context, including:

- average delay
- occupancy
- late-rate indicator
- observed event count
- route-level rule-based snapshot
- optional AI-generated explanation

These metrics are descriptive snapshot summaries, not live service alerts.

### Map View

The Map View shows route geometry, recent vehicle points, and FMI weather station context.

It includes:

- HSL route paths from GTFS reference data
- recent vehicle point visualization
- route and transport mode selection
- FMI weather station context
- nearest weather station enrichment based on distance

Weather is used as contextual external information only. The dashboard does not infer causal weather impact.

---

## Optional AI Explanation Layer

The dashboard includes an optional OpenAI explanation layer for the Pipeline Overview and Route Performance pages.

This layer is not part of the metric calculation path. It only rewrites deterministic rule-based dashboard facts into a short plain-English explanation.

Guardrails:

- Uses only precomputed dashboard facts
- Does not inspect raw data
- Does not calculate new metrics
- Does not infer root causes
- Does not claim live monitoring
- Does not claim prediction
- Does not claim weather impact or causal analysis
- Falls back to rule-based insights when the OpenAI API key is not configured

This keeps the dashboard explainable while preserving deterministic metrics as the source of truth.

---

## Azure Function Metadata Heartbeat

A lightweight Azure Function Timer Trigger writes `dashboard/refresh_metadata.json` to Azure Blob Storage.

The metadata heartbeat records:

- latest metadata check time in UTC and Helsinki time
- serving layer type
- full refresh owner
- lightweight check owner
- whether the dashboard should not be treated as a live monitor

This function intentionally does not run the Spark pipeline, trigger Databricks, fetch HSL/FMI data, or refresh Gold-layer dashboard outputs.

It exists only to make the refresh model more transparent.

---

## Cost-aware Deployment Trade-offs

This is a personal portfolio deployment, so the architecture separates production-oriented design from always-on infrastructure cost.

Key trade-offs:

- Azure Databricks is used to validate a managed cloud execution path, but the dashboard can continue serving the latest exported parquet outputs without keeping Databricks compute continuously active.
- Azure Blob Storage acts as a low-cost serving layer between pipeline execution and dashboard consumption.
- Azure Function is used only for lightweight metadata heartbeat, not incremental data refresh.
- Render free-tier hosting may cold start; external uptime checks reduce this but do not provide production SLA.
- GitHub Actions is retained as manual fallback and best-effort automation, not as a strict scheduler.

In a company production environment, this setup would typically be upgraded with managed monitoring, alerting, Key Vault or managed identity, stricter orchestration, and company-owned cloud cost controls.

---

## Data Sources

This project combines simulated transport telemetry with real external reference/context data.

### Simulated transport telemetry

The main telemetry events are synthetic and are used to model route-level operational signals such as delay, occupancy, event volume, and ingestion timing.

Telemetry is simulated intentionally. The project focuses on pipeline design, data modeling, serving architecture, and operational trade-offs rather than building a live transport data collector.

### FMI weather observations

Weather observations are fetched from the Finnish Meteorological Institute API and used as external context for the dashboard.

Weather data is not used to infer causal impact. It is included as contextual information, especially in the Map View.

### HSL GTFS route reference data

HSL GTFS reference files are used to derive route geometry and map context.

The dashboard uses this data to show route paths, sampled vehicle points, and route-level spatial context.

---

## Event Model

Inputs are normalized into a shared event-oriented structure.

Core fields:

- `event_time` — when the event occurred
- `ingest_time` — when the event was processed
- `source` — telemetry or weather
- `metric` — type of measurement
- `value` — numeric value
- `unit` — measurement unit
- `attrs` — flexible metadata such as route id, station id, or context fields

This model keeps telemetry and external context data consistent enough for downstream aggregation, while still allowing source-specific attributes to evolve.

---

## Data Layers

### Bronze

The Bronze layer stores append-only raw events with minimal transformation.

It acts as the system of record for pipeline runs and keeps debugging simple.

### Silver

The Silver layer applies event-time windowing, route-level aggregation, and basic data quality checks.

It produces intermediate metrics such as delay, occupancy, event counts, duplicate checks, and ingestion latency.

### Gold

The Gold layer produces dashboard-ready outputs.

Gold outputs include:

- route-level window KPIs
- route-level daily summaries
- pipeline health metrics
- HSL route/map parquet outputs
- FMI weather station context outputs

These outputs are exported as parquet files and uploaded to Azure Blob Storage for dashboard consumption.

---

## Design Decisions

This project is intentionally designed to stay simple, explainable, and cost-aware.

The focus is not to maximize the number of services, but to show how a data pipeline can be structured around clear responsibilities, stable outputs, and realistic deployment constraints.

### Scheduled snapshot instead of live monitoring

The dashboard is designed as a scheduled snapshot view.

It presents the latest exported Gold-layer metrics and map context, but it does not claim real-time operational monitoring. This keeps the system honest, stable, and affordable for a portfolio deployment.

### Precomputed outputs instead of live queries

The dashboard reads exported parquet files from Azure Blob Storage instead of querying a live database or processing layer.

This reduces runtime dependencies and keeps dashboard performance predictable.

### Decoupled pipeline and dashboard layers

The pipeline produces data. The dashboard consumes data.

Azure Blob Storage acts as the serving boundary between those layers. This makes the dashboard easier to deploy and prevents UI availability from depending on pipeline execution.

### Databricks as an optional cloud execution path

Azure Databricks validates that the pipeline can run in a managed cloud data engineering environment.

The same pipeline logic can run locally or through the Databricks job wrapper. For portfolio cost control, Databricks does not need to stay continuously active after the latest outputs have been exported.

### Lightweight metadata heartbeat

Azure Function is used only to write lightweight dashboard metadata.

It does not perform incremental data refresh. This avoids turning a transparency feature into an expensive or confusing orchestration layer.

### AI explanation outside the metric path

OpenAI is kept outside the data pipeline and outside the metric calculation path.

The model only receives precomputed rule-based facts and rewrites them into a short explanation. Metrics remain deterministic and explainable.

### Weather as context, not causal analysis

FMI weather data is included as external context.

The project does not infer that weather caused route delays or operational changes. This keeps the dashboard descriptive and avoids unsupported conclusions.

--- 

## Repository Structure

```text
.github/workflows/
  telemetry-refresh.yml             # manual fallback refresh workflow
  keepalive_telemetry.yml           # best-effort Render keepalive
  keepalive_nyc.yml                 # legacy NYC dashboard keepalive

Projects/Public-Transport-Telemetry-Pipeline/
  azure_function/
    function_app.py                 # Azure Function Timer Trigger metadata heartbeat
    host.json
    requirements.txt
    local.settings.json.example

  data/
    bronze/
      bronze_events.csv             # append-only raw events
    silver/
      silver_transit_metrics.csv    # aggregated transit metrics
    gold/
      hsl/                          # HSL route/map parquet outputs
      weather/                      # FMI weather station parquet outputs
      output/                       # dashboard-ready Gold parquet outputs
    external/gtfs_hsl/              # HSL GTFS reference files

  src/pipeline/
    bronze.py                       # Bronze ingestion logic
    silver.py                       # Silver aggregation and data quality logic
    gold.py                         # Gold KPI and health metric modeling
    hsl.py                          # HSL route geometry / map data processing
    config.py
    setup.py

  scripts/
    run_pipeline.py                 # local pipeline runner
    run_databricks_refresh.py       # Databricks Job wrapper
    export_gold.py                  # export Gold outputs to parquet
    upload_outputs_to_blob.py       # upload dashboard outputs to Azure Blob

  streamlit_app/
    Home.py
    pages/
      1_Pipeline_Overview.py
      2_Route_Performance.py
      3_Map_View.py
    utils/
      data_access.py
      insights.py                   # deterministic rule-based dashboard insights
      load_data.py
      maps.py
      openai_explainer.py           # optional OpenAI explanation layer
      refresh_metadata.py           # load Azure Function metadata heartbeat

  notebooks/mvp/
    00_pipeline_runner.ipynb
    00_setup_and_schema.ipynb
    10_bronze_transit_ingest.ipynb
    20_silver_transit_metrics.ipynb
    30_bronze_weather_ingest.ipynb
    40_silver_weather_metrics.ipynb
    50_gold_route_kpi_window.ipynb
    60_gold_route_kpi_daily.ipynb
    70_gold_pipeline_metrics_window.ipynb
    80_basic_visual_checks.ipynb
    90_hsl_map_mvp.ipynb

  docs/
    architecture.png
    dashboard_home.jpg
    dashboard_pipeline_overview.jpg
    dashboard_route_performance.jpg
    dashboard_map_combined.jpg
    dashboard_pipeline_overview_ai.jpg
    dashboard_route_performance_ai.jpg
    dashboard_route_ranking_all.jpg
    databricks_successful_run.jpg
    azure_function_metadata_heartbeat.jpg
    uptimerobot_keepalive.jpg

README.md
requirements.txt
.gitignore
```

---

## How to Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full local pipeline:

```bash
python Projects/Public-Transport-Telemetry-Pipeline/scripts/run_pipeline.py --layer full
```

Run individual layers:

```bash
python Projects/Public-Transport-Telemetry-Pipeline/scripts/run_pipeline.py --layer bronze
python Projects/Public-Transport-Telemetry-Pipeline/scripts/run_pipeline.py --layer silver
python Projects/Public-Transport-Telemetry-Pipeline/scripts/run_pipeline.py --layer gold
```

Export Gold outputs:

```bash
python Projects/Public-Transport-Telemetry-Pipeline/scripts/export_gold.py
```

Upload dashboard outputs to Azure Blob Storage:

```bash
python Projects/Public-Transport-Telemetry-Pipeline/scripts/upload_outputs_to_blob.py
```

Run the Streamlit dashboard:

```bash
streamlit run Projects/Public-Transport-Telemetry-Pipeline/streamlit_app/Home.py
```

---

## Optional Azure Databricks Execution Path

The project includes a Databricks-compatible refresh wrapper:

```bash
python Projects/Public-Transport-Telemetry-Pipeline/scripts/run_databricks_refresh.py --layer full
```

In Azure Databricks, this wrapper is used by a Job task to run the full refresh flow:

1. Run the pipeline
2. Export Gold outputs
3. Upload dashboard-ready parquet files to Azure Blob Storage

The Databricks path reuses the same project logic as the local pipeline, while adapting runtime paths for the Databricks environment.

This path is optional for portfolio cost control. Once exported parquet outputs are available in Azure Blob Storage, the dashboard can continue serving the latest snapshot without keeping Databricks compute continuously active.

---

## Deployment Notes

### Dashboard

The dashboard is deployed on Render and reads parquet outputs from Azure Blob Storage.

Required dashboard environment variables:

- `AZURE_STORAGE_CONNECTION_STRING`
- `AZURE_CONTAINER_NAME`
- `OPENAI_API_KEY` optional
- `OPENAI_EXPLANATION_MODEL` optional

If the OpenAI API key is not configured, the dashboard still works with rule-based insights.

### Azure Function

The Azure Function writes lightweight refresh metadata to Azure Blob Storage.

Required Function App settings:

- `AzureWebJobsStorage`
- `FUNCTIONS_WORKER_RUNTIME`
- `AZURE_STORAGE_CONNECTION_STRING`
- `AZURE_CONTAINER_NAME`
- `LIGHTWEIGHT_METADATA_REFRESH_CRON`

### Keepalive

External uptime checks are used to reduce Render free-tier cold starts.

GitHub Actions keepalive workflows are kept as best-effort backup and manual fallback. They are not treated as production-grade uptime guarantees.

---

## Scope and Limitations

This project is a portfolio-scale data engineering system.

It demonstrates production-oriented patterns, but it is not a live transit operations platform.

Current limitations:

- telemetry events are simulated
- dashboard data is refreshed as scheduled snapshots
- weather data is used as context, not causal analysis
- Render free-tier hosting may cold start
- Databricks execution is optional and cost-controlled
- alerting and SLA monitoring are intentionally out of scope

---

## Additional Screenshots

Additional dashboard and operations screenshots are kept in the `docs/` folder. They provide supporting evidence for the optional AI explanation layer, route-level comparison, Databricks execution, and lightweight metadata heartbeat.

<details>
<summary>Show additional screenshots</summary>

### Pipeline Overview with AI explanation

![Pipeline Overview AI](docs/dashboard_pipeline_overview_ai.jpg)

### Route Performance with AI explanation

![Route Performance AI](docs/dashboard_route_performance_ai.jpg)

### All-route ranking view

![All-route Ranking](docs/dashboard_route_ranking_all.jpg)

### Azure Databricks successful run

![Azure Databricks Successful Run](docs/databricks_successful_run.jpg)

### Azure Function metadata heartbeat

![Azure Function Metadata Heartbeat](docs/azure_function_metadata_heartbeat.jpg)

### UptimeRobot keepalive check

![UptimeRobot Keepalive](docs/uptimerobot_keepalive.jpg)

</details>

---

## Possible Extensions

Possible future extensions include:

- structured streaming with lower-latency ingestion
- containerized scheduled execution with Azure Container Apps Jobs as a lower-cost alternative to managed Databricks execution for portfolio-scale refreshes
- stronger backfill and reprocessing support
- automated alerting on pipeline health metrics
- richer data quality checks
- additional external context sources
- production deployment using company-owned cloud infrastructure
- Key Vault or managed identity for secret management

---

## License

This project is covered by the MIT License at the repository root.



# Public Transport Telemetry & Weather Impact Pipeline

## Live Demo

Dashboard (Render):  
https://transport-telemetry-dashboard.onrender.com

The dashboard reads precomputed parquet outputs and reflects recent pipeline runs.

Tip:  
Select a route and toggle weather context in the Map View to explore how external signals relate to transport behavior.

---

## Overview

A lightweight, production-oriented data pipeline focused on:

- reliable data integration  
- simple and consistent aggregation logic  
- stable outputs for downstream use  

This project combines operational telemetry signals with external weather data into a unified event model, producing queryable metrics for monitoring and analysis.

The goal is not complexity, but clarity, maintainability, and realistic engineering trade-offs.

---

## Architecture

The pipeline follows a layered structure:

**Bronze → Silver → Gold → Serving → Dashboard**

Each layer has a clear responsibility:

- **Bronze** — append-only ingestion and raw event storage  
- **Silver** — windowed aggregation and data quality handling  
- **Gold** — KPI modeling and pipeline health metrics  
- **Serving** — exported parquet + Azure Blob Storage  
- **Dashboard** — Streamlit-based visualization layer  

![Pipeline architecture](docs/architecture.png)

---

## Dashboard preview

### Pipeline Overview

Recent operational snapshot showing ingestion delay trends and pipeline health metrics.

![Pipeline Overview](docs/dashboard_pipeline_overview.jpg)

---

### Route Performance

Route-level KPIs and daily summaries derived from aggregated Gold-layer outputs.

![Route Performance](docs/dashboard_route_performance.jpg)

---

### Map View

Route geometry, sampled vehicle points, and optional weather context (FMI).

The map layer is enriched with **HSL GTFS route reference data**, providing realistic route geometry and spatial context.

![Map View](docs/dashboard_map_view.jpg)

---

## Deployment and data flow

The project includes a simple but complete serving pipeline:

1. Pipeline runs via GitHub Actions  
2. Gold outputs are exported as parquet files  
3. Files are uploaded to Azure Blob Storage (via GitHub Secrets)  
4. Streamlit dashboard (Render) reads these parquet files  

Flow: # Public Transport Telemetry & Weather Impact Pipeline

## Live Demo

Dashboard (Render):  
https://transport-telemetry-dashboard.onrender.com

The dashboard reads precomputed parquet outputs and reflects recent pipeline runs.

Tip:  
Select a route and toggle weather context in the Map View to explore how external signals relate to transport behavior.

---

## Overview

A lightweight, production-oriented data pipeline focused on:

- reliable data integration  
- simple and consistent aggregation logic  
- stable outputs for downstream use  

This project combines operational telemetry signals with external weather data into a unified event model, producing queryable metrics for monitoring and analysis.

The goal is not complexity, but clarity, maintainability, and realistic engineering trade-offs.

---

## Architecture

The pipeline follows a layered structure:

**Bronze → Silver → Gold → Serving → Dashboard**

Each layer has a clear responsibility:

- **Bronze** — append-only ingestion and raw event storage  
- **Silver** — windowed aggregation and data quality handling  
- **Gold** — KPI modeling and pipeline health metrics  
- **Serving** — exported parquet + Azure Blob Storage  
- **Dashboard** — Streamlit-based visualization layer  

![Pipeline architecture](docs/architecture.png)

---

## Dashboard preview

### Pipeline Overview

Recent operational snapshot showing ingestion delay trends and pipeline health metrics.

![Pipeline Overview](docs/dashboard_pipeline_overview.jpg)

---

### Route Performance

Route-level KPIs and daily summaries derived from aggregated Gold-layer outputs.

![Route Performance](docs/dashboard_route_performance.jpg)

---

### Map View

Route geometry, sampled vehicle points, and optional weather context (FMI).

The map layer is enriched with **HSL GTFS route reference data**, providing realistic route geometry and spatial context.

![Map View](docs/dashboard_map_view.jpg)

---

## Deployment and data flow

The project includes a simple but complete serving pipeline:

1. Pipeline runs via GitHub Actions  
2. Gold outputs are exported as parquet files  
3. Files are uploaded to Azure Blob Storage (via GitHub Secrets)  
4. Streamlit dashboard (Render) reads these parquet files  

Flow:

GitHub Actions → export_gold → Azure Blob → Streamlit (Render)


This decoupled design avoids direct database dependencies and keeps the system stable and low-maintenance.

---

## Key design decisions

This project intentionally avoids unnecessary complexity and focuses on practical trade-offs.

### Append-only ingestion

Keeps data flow predictable and reproducible.  
Avoids mutation logic and simplifies debugging.

---

### Unified event model

All inputs (telemetry + weather) are normalized into a shared schema.

This simplifies downstream processing, at the cost of stricter schema discipline.

---

### Micro-batch processing

Chosen over streaming to reduce infrastructure complexity.

Keeps the system lightweight while preserving a migration path.

---

### Precomputed outputs (parquet)

Dashboard reads static parquet instead of live queries.

Improves reliability and removes runtime dependencies.

---

### Decoupled serving layer

Data is delivered via parquet + Azure Blob instead of direct database access.

This enables:

- simple deployment  
- low operational overhead  
- clear separation between pipeline and consumption  

---

### Observability as data

Pipeline health (freshness, lag, volume, duplicates) is modeled as tables, not logs.

This allows:

- SQL-based inspection  
- easier debugging  
- consistent monitoring logic  

---

### Lightweight orchestration

GitHub Actions is used instead of Airflow or Azure Data Factory.

This keeps scheduling simple while still supporting automation.

---

## Handling imperfect data

Real-world data sources rarely align perfectly.

Examples:

- telemetry and weather arrive at different times  
- some windows contain only partial data  

Instead of enforcing strict upstream alignment, this project handles mismatches at the presentation layer.

This keeps the pipeline logic simple while still producing usable outputs.

---

## Data sources

- **Simulated telemetry data**  
  Represents transport signals such as delay and occupancy  

- **FMI Weather API**  
  Provides real observational weather data  

- **HSL GTFS reference data**  
  Used for route geometry and map context  

Telemetry is simulated intentionally to focus on pipeline design rather than data collection.

---

## Event model

All inputs are transformed into a shared structure.

Core fields:

- `event_time` — when the event occurred  
- `ingest_time` — when the event was processed  
- `source` — telemetry or weather  
- `metric` — type of measurement  
- `value` — numeric value  
- `unit` — measurement unit  
- `attrs` — flexible metadata (e.g. route_id)  

This enables consistent aggregation and simple schema evolution.

---

## Data layers

### Bronze

- append-only raw events  
- minimal transformation  
- acts as the system of record  

---

### Silver

- event-time windowed aggregation  
- route-level and time-based metrics  
- data quality checks (nulls, duplicates, counts)  
- ingestion latency metrics  

---

### Gold

- final metrics for dashboards  
- route KPIs (window + daily)  
- pipeline health metrics (freshness, lag, volume)  

Outputs are exported as parquet files for stable downstream use.

---

## Dashboard

The Streamlit dashboard provides three views:

- **Pipeline Overview**  
  pipeline health and ingestion delay trends  

- **Route Performance**  
  route-level KPIs and daily summaries  

- **Map View**  
  route geometry, vehicle points, and weather context  

The dashboard is designed for clarity and stability rather than heavy interactivity.

---

## Scheduling

Pipeline execution is handled using GitHub Actions.

Note:  
Execution timing is best-effort and may vary depending on GitHub runner availability.

Workflows:

- `telemetry-refresh.yml` — pipeline execution  
- `keepalive_telemetry.yml` — dashboard keepalive  
- `keepalive_nyc.yml` — legacy project keepalive  

Location:

```bash
.github/workflows/
```

---

## Repository structure

```bash
.github/workflows/
  telemetry-refresh.yml        # scheduled pipeline runs
  keepalive_telemetry.yml     # keep Render service awake

Projects/Public-Transport-Telemetry-Pipeline/

  data/
    bronze/
      bronze_events.csv       # raw append-only events
    silver/
      silver_transit_metrics.csv  # aggregated metrics
    gold/
      hsl/                    # route geometry (GTFS)
      weather/                # weather observations
    output/
      gold_route_daily.parquet
      gold_route_window.parquet
      pipeline_metrics.parquet
    external/gtfs_hsl/        # GTFS reference data

  src/pipeline/
    bronze.py
    silver.py
    gold.py
    hsl.py
    config.py
    setup.py

  scripts/
    run_pipeline.py
    export_gold.py
    upload_outputs_to_blob.py

  streamlit_app/
    Home.py
    pages/
      1_Pipeline_Overview.py
      2_Route_Performance.py
      3_Map_View.py
    utils/
      load_data.py
      data_access.py
      maps.py

  tests/tools/
    inspect_*.py
    test_*.py

  notebooks/mvp/
  docs/
    architecture.png

README.md
requirements.txt
.gitignore
```

---

## How to run

Run full pipeline:

```bash
python scripts/run_pipeline.py --layer full
```

Run individual layers:

```bash
python scripts/run_pipeline.py --layer bronze
python scripts/run_pipeline.py --layer silver
python scripts/run_pipeline.py --layer gold
```

Inspect outputs:

```bash
python tests/tools/inspect_gold.py
```

---

## Scope

In scope:

- data integration
- aggregation logic
- pipeline reliability
- monitoring outputs

Out of scope:

- real-time streaming
- large-scale infrastructure
- predictive modeling

---

## Possible extensions

This project is intentionally scoped to remain simple.

Potential next steps:

- structured streaming for lower latency
- backfill and reprocessing support
- automated alerting and anomaly detection
- additional external signals (e.g. traffic data)

---

## License

This project is covered by the MIT License at the repository root.

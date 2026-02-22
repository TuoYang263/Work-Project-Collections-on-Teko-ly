# Public Transport Telemetry & Weather Impact Pipeline (MVP)

Production-oriented telemetry pipeline using a layered Bronze → Silver → Gold architecture.

It ingests operational telemetry events and weather observations into a unified event model and aggregates them into monitoring-ready operational metrics.

This MVP focuses on data modeling, aggregation, and pipeline observability. Predictive modeling and accuracy optimization are intentionally out of scope.

---

## Problem

Public transport operations are sensitive to external conditions such as weather, but telemetry signals and external observations are often processed separately or analyzed offline.

This project demonstrates how event-based telemetry and weather observations can be ingested into a unified pipeline and transformed into operational monitoring signals.

The goal is to model a realistic telemetry pipeline structure that supports operational visibility, KPI reporting, and pipeline health monitoring.

---

## Architecture

The pipeline follows a layered Bronze → Silver → Gold architecture, separating raw ingestion, operational aggregation, and monitoring-ready outputs.

![Pipeline architecture](docs/architecture.png)

### Layers (high-level)

- **Bronze**  
  Append-only raw events (telemetry and weather) stored with minimal transformation.  
  This layer acts as the immutable system of record.

- **Silver**  
  Windowed aggregations and curated operational signals using event-time processing.  
  Produces consistent, validation-ready operational metrics.

- **Gold**  
  Operational KPI tables and pipeline health metrics designed for dashboards and monitoring.

- **Ops / Health**  
  Observability metrics such as ingestion freshness, event lag, duplicate ratios, and volume trends.

---

## Data sources

- **Simulated telemetry events**  
  Operational signals such as delay, headway, and load metrics.

- **FMI Weather API**  
  Observational weather data ingested into the same unified event model.

Telemetry is simulated in this MVP to focus on pipeline design, aggregation, and observability rather than data collection.

---

## Event model

All sources are normalized into a shared event schema.

Core fields:

- `event_time` — timestamp when the event occurred  
- `ingest_time` — timestamp when the event was ingested  
- `source` — telemetry or weather  
- `metric` — metric name (e.g., temperature, precipitation, delay_seconds)  
- `value` and `unit` — numeric measurement  
- `attrs` — extensible attribute map for schema evolution  

This unified model enables consistent aggregation, flexible schema evolution, and observability metrics such as ingestion lag and freshness.

---

## Silver metrics (15-minute windows)

The Silver layer aggregates events into fixed event-time windows.

Example outputs:

- `silver_weather_metrics`  
  Aggregated weather observations per window

- `silver_transit_metrics` (planned extension)  
  Aggregated operational telemetry metrics per route and window

Validation checks include:

- row count sanity checks  
- duplicate detection  
- null and missing key checks  
- event-time range validation  

---

## Gold outputs

The Gold layer provides monitoring-ready operational metrics.

These tables are designed for:

- dashboard visualization  
- operational monitoring  
- pipeline health analysis  

Outputs include:

- aggregated operational KPIs  
- ingestion freshness metrics  
- ingestion lag metrics  
- duplicate ratios and volume tracking  

These observability signals are stored as structured tables rather than logs.

---

## Pipeline health and observability

Pipeline health is treated as a first-class output.

Tracked signals include:

- ingestion freshness  
- event-time lag  
- event volume trends  
- duplicate ratios  
- missing key indicators  

These metrics enable pipeline monitoring and troubleshooting.

---

## Scope and limitations

In scope:

- append-only Bronze ingestion  
- unified event schema  
- event-time windowed aggregation  
- operational KPI modeling  
- pipeline observability metrics  

Out of scope:

- predictive modeling or machine learning  
- geospatial enrichment  
- real-time streaming infrastructure  

This MVP runs in micro-batch mode but is structured for future streaming migration.

---

## Repo structure

Current structure:

- `public_transport_telemetry_pipeline.ipynb` — complete pipeline implementation
- `public_transport_telemetry_pipeline.py` — script version of the pipeline (for easier inspection if notebook preview fails)  
- `docs/architecture.png` — pipeline architecture diagram  
- `README.md` — project documentation  

The pipeline is intentionally implemented as a single notebook to keep the full data flow transparent and easy to inspect.

In production environments, ingestion, aggregation, and monitoring would typically run as independent jobs.

---

## How to run

1. Configure environment (API access and storage location)

2. Run ingestion cells in the notebook to populate Bronze tables

3. Run aggregation cells to generate Silver metrics

4. Run KPI and health metric cells to generate Gold outputs

The pipeline is designed to remain platform-agnostic and portable.

---

## Roadmap

Possible extensions:

- real telemetry integration  
- streaming ingestion using Structured Streaming  
- geospatial enrichment  
- automated monitoring and alerting  

---

## Portability

The pipeline is structured to be migration-ready to managed environments such as Azure Databricks.

Key design choices supporting portability:

- unified event model  
- append-only ingestion pattern  
- clear layer separation  
- event-time aggregation  

---

## License

This project is covered by the MIT License at the repository root.
# GitHub Projects Plan

## Purpose

This file documents the GitHub Projects board structure for the Olist E-Commerce Analytics & Pipeline Monitoring Portal.

The board supports milestone-based delivery, keeps project work visible, and helps separate completed scope from future scope.

This document has been refreshed during **M6 - README / dbt docs / Project Showcase Cleanup** to reflect the current project status and updated roadmap.

---

## Board Name

```text
Olist Analytics Portal
```

---

## Board Columns

| Column | Purpose |
|---|---|
| Backlog | Ideas and future tasks not ready to start |
| Ready | Tasks with clear scope and acceptance criteria |
| In Progress | Work currently being done |
| Review | Completed work waiting for review or cleanup |
| Done | Completed and committed work |

---

## Milestones

| Milestone | Description | Status |
|---|---|---:|
| M1 - Project Setup & Source Understanding | Repository setup, source overview, metadata planning, project board setup | Completed |
| M2 - BigQuery Raw Layer | BigQuery raw dataset and source-aligned tables | Completed |
| M3 - Staging Layer Planning | Staging design, source-to-staging mapping, cleanup rules | Completed |
| M4 - dbt Staging Layer | dbt sources, staging models, schema docs, staging tests | Completed |
| M5 - Dimensional Modeling / Analytics Marts | Intermediate models, dimensions, facts, mart tests, dbt docs validation | Completed |
| M6 - README / dbt docs / Project Showcase Cleanup | Portfolio-ready README, architecture docs, dbt docs screenshots, roadmap cleanup | In Progress |
| M7 - Google Cloud Scheduler + Cloud Run Job orchestration | Scheduled dbt execution using Google Cloud services | Future |
| M8 - ADE-inspired metadata refresh / monitoring tables | dbt artifact parsing and BigQuery `olist_monitoring` tables | Future |
| M9 - AI-assisted pipeline intelligence layer | Explanation layer on top of metadata, tests, artifacts, and docs | Future |

---

## Labels

| Label | Purpose |
|---|---|
| `area: docs` | Documentation work |
| `area: data` | Source data and metadata work |
| `area: bigquery` | BigQuery setup and warehouse work |
| `area: dbt` | dbt models, tests, and dbt documentation |
| `area: modeling` | Dimensional modeling, facts, dimensions, and marts |
| `area: quality` | Data quality checks and validation |
| `area: monitoring` | Pipeline monitoring, freshness, metadata, and observability |
| `area: orchestration` | Scheduler, Cloud Run Jobs, and pipeline execution |
| `area: ai` | AI-assisted explanation or intelligence layer |
| `area: bi` | Future BI dashboard work |
| `area: portal` | Future portal work |
| `priority: high` | Important milestone task |
| `priority: medium` | Normal task |
| `priority: low` | Optional or later task |
| `status: blocked` | Blocked by another task or decision |

---

## Completed Project Work

### M1 Issues

| Issue | Area | Status |
|---|---|---|
| Create feature branch | `area: docs` | Done |
| Create project folder structure | `area: docs` | Done |
| Add README skeleton | `area: docs` | Done |
| Add architecture documentation | `area: docs` | Done |
| Add `.gitignore` for raw data and credentials | `area: data` | Done |
| Download Olist dataset locally | `area: data` | Done |
| Inspect source CSV structure | `area: data` | Done |
| Add source data overview | `area: data` | Done |
| Add source tables inventory | `area: data` | Done |
| Add BigQuery dataset naming plan | `area: bigquery` | Done |
| Add project management documentation | `area: docs` | Done |
| Add GitHub Projects planning document | `area: docs` | Done |

### M2 Issues

| Issue | Area | Status |
|---|---|---|
| Create BigQuery raw dataset | `area: bigquery` | Done |
| Load Olist CSV files into BigQuery raw tables | `area: bigquery` | Done |
| Validate raw table row counts and schemas | `area: quality` | Done |
| Document raw layer validation | `area: docs` | Done |

### M3 Issues

| Issue | Area | Status |
|---|---|---|
| Document staging layer plan | `area: docs` | Done |
| Document source-to-staging mapping | `area: docs` | Done |
| Document column cleanup rules | `area: docs` | Done |
| Prepare dbt staging implementation scope | `area: dbt` | Done |

### M4 Issues

| Issue | Area | Status |
|---|---|---|
| Initialize dbt project | `area: dbt` | Done |
| Configure BigQuery dbt profile locally | `area: dbt` | Done |
| Register raw tables as dbt sources | `area: dbt` | Done |
| Create 9 staging models | `area: dbt` | Done |
| Add staging model and column documentation | `area: docs` | Done |
| Add staging dbt tests | `area: quality` | Done |
| Validate staging layer with dbt run and dbt test | `area: quality` | Done |
| Document M4 validation results | `area: docs` | Done |

### M5 Issues

| Issue | Area | Status |
|---|---|---|
| Document dimensional modeling design | `area: modeling` | Done |
| Create intermediate aggregation models | `area: dbt` | Done |
| Create dimension models | `area: modeling` | Done |
| Create fact models | `area: modeling` | Done |
| Add marts schema documentation and tests | `area: docs` | Done |
| Correct `fct_order_reviews` grain using `review_key` | `area: modeling` | Done |
| Validate intermediate and marts layers with dbt build | `area: quality` | Done |
| Generate and review dbt docs and lineage | `area: dbt` | Done |
| Document M5 validation results | `area: docs` | Done |

---

## Current M6 Issues

| Issue | Area | Status |
|---|---|---|
| Refresh README as project showcase | `area: docs` | In Progress |
| Update architecture documentation as built | `area: docs` | In Progress |
| Prepare dbt docs and lineage screenshots | `area: dbt` | In Progress |
| Add dbt docs screenshots to README | `area: docs` | In Progress |
| Update project management and roadmap docs | `area: docs` | In Progress |
| Final documentation review and commit | `area: docs` | Ready |

---

## Future Issue Placeholders

| Issue | Milestone | Area |
|---|---|---|
| Create Cloud Run Job for dbt execution | M7 | `area: orchestration` |
| Schedule dbt run with Google Cloud Scheduler | M7 | `area: orchestration` |
| Document orchestration design and runbook | M7 | `area: docs` |
| Parse `manifest.json` into metadata tables | M8 | `area: monitoring` |
| Parse `run_results.json` into run result tables | M8 | `area: monitoring` |
| Parse `catalog.json` into catalog metadata tables | M8 | `area: monitoring` |
| Load dbt artifact metadata into BigQuery `olist_monitoring` | M8 | `area: bigquery` |
| Create monitoring marts for model runs and tests | M8 | `area: monitoring` |
| Add AI-assisted explanation layer design | M9 | `area: ai` |
| Build controlled AI explanation prototype | M9 | `area: ai` |
| Document AI boundaries and validation assumptions | M9 | `area: docs` |

---

## Notes

- The GitHub Projects board should support the project, not replace repository documentation.
- The board should stay simple and practical.
- Issues should be small enough to complete and commit cleanly.
- Each milestone should have clear acceptance criteria.
- Completed scope and future scope should be clearly separated.
- Power BI and React portal work are not part of the current M6 scope.
- Agile Data Engine is not directly integrated; only metadata-driven DataOps thinking is borrowed for M8.
- The future AI layer should explain validated metadata and test results, not replace dbt tests or structured monitoring.

---

## Current Status

### Completed

- M1 - Project Setup & Source Understanding
- M2 - BigQuery Raw Layer
- M3 - Staging Layer Planning
- M4 - dbt Staging Layer
- M5 - Dimensional Modeling / Analytics Marts

### In Progress

- M6 - README / dbt docs / Project Showcase Cleanup

### Next Milestone

- M7 - Google Cloud Scheduler + Cloud Run Job orchestration
# GitHub Projects Plan

## Purpose

This file prepares the planned GitHub Projects board structure for the Olist E-Commerce Analytics & Pipeline Monitoring Portal.

The actual GitHub Projects board may be created later. This document keeps the project planning visible in the repository.

This file is part of **M1 - Project Setup & Source Understanding**.

## Planned Board Name

```text
Olist Analytics Portal
```

## Planned Board Columns

| Column | Purpose |
|---|---|
| Backlog | Ideas and future tasks not ready to start |
| Ready | Tasks with clear scope |
| In Progress | Work currently being done |
| Review | Completed work waiting for review or cleanup |
| Done | Completed and committed work |

## Planned Milestones

| Milestone | Description |
|---|---|
| M1 - Project Setup & Source Understanding | Repository setup, source overview, metadata planning |
| M2 - BigQuery Raw Layer | BigQuery raw dataset and source-aligned tables |
| M3 - dbt Staging Layer | dbt sources, staging models, and basic tests |
| M4 - Dimensional Modeling | Fact and dimension model design and implementation |
| M5 - Reporting Marts & Data Quality | BI-ready marts and data quality checks |
| M6 - Pipeline Monitoring | Pipeline status, freshness, and monitoring outputs |
| M7 - Power BI Dashboard | Business-facing BI dashboard |
| M8 - React/Node TypeScript Portal | Custom analytics and monitoring portal |

## Planned Labels

| Label | Purpose |
|---|---|
| `area: docs` | Documentation work |
| `area: data` | Source data and metadata work |
| `area: bigquery` | BigQuery setup and warehouse work |
| `area: dbt` | dbt models, tests, and documentation |
| `area: modeling` | Dimensional modeling and marts |
| `area: quality` | Data quality checks and validation |
| `area: monitoring` | Pipeline monitoring and freshness |
| `area: bi` | Power BI dashboard work |
| `area: portal` | React/Node TypeScript portal work |
| `priority: high` | Important milestone task |
| `priority: medium` | Normal task |
| `priority: low` | Optional or later task |
| `status: blocked` | Blocked by another task or decision |

## M1 Issues

Planned M1 issues:

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

## Future Issue Placeholders

Future issues may include:

| Issue | Milestone |
|---|---|
| Create BigQuery raw datasets | M2 |
| Load Olist CSV files into BigQuery raw tables | M2 |
| Define dbt sources | M3 |
| Create staging models | M3 |
| Add basic dbt source tests | M3 |
| Design fact and dimension tables | M4 |
| Build reporting marts | M5 |
| Add data quality checks | M5 |
| Add pipeline monitoring tables | M6 |
| Build Power BI dashboard | M7 |
| Build React/Node TypeScript portal | M8 |

## Notes

- The GitHub Projects board should support the project, not replace the repository documentation.
- The board should stay simple and practical.
- Issues should be small enough to complete and commit cleanly.
- M1 focused on setup, source understanding, and planning.
- M2 focused on loading source-aligned CSV files into BigQuery raw tables.

## Current Status

### Completed

- M1 - Project Setup & Source Understanding
- M2 - BigQuery Raw Layer
- M3 - Staging Layer Planning & Source-to-Staging Rules

### Completed GitHub Project Work

- GitHub Project board `Olist Analytics Portal` created
- M1 tasks moved to Done
- M2 raw layer task moved to Done
- M3 staging planning task moved to Done

### M3 Summary

Staging layer planning has been completed.

Completed M3 work:

- staging layer purpose documented
- staging dataset naming documented
- source-to-staging mapping documented
- column cleanup rules documented
- timestamp, numeric, null, and duplicate handling rules included in staging cleanup rules

Main M3 documents:

```text
docs/staging_layer_plan.md
metadata/staging/source_to_staging_mapping.md
metadata/staging/column_cleanup_rules.md
```

### Next Milestone

- M4 - dbt Project Setup and Staging Model Implementation
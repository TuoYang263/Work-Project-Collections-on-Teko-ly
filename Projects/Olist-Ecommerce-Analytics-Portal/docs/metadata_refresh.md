# M8 - dbt Metadata Refresh and Pipeline Monitoring Layer

## Status

```text
Completed: 2026-07-15
```

## Current note

This document records the M8 monitoring milestone. Its original validation counts are kept as historical evidence.

The current repository also includes M9 and M10 U1. M10 added `control_attempt_id` to `pipeline_runs`, and window-controlled runs now resolve the exact `monitoring_run_id` for the current attempt before running M9.

The existing scheduled Cloud Run entry point is still `run_dbt_job.sh`. M10 window control has been validated separately and has not yet replaced that scheduled entry point.

---

M8 implements an ADE-inspired, artifact-based monitoring layer for the Olist E-Commerce Analytics & Pipeline Monitoring Portal.

The implementation converts dbt artifact JSON into append-only BigQuery monitoring records. It preserves pipeline execution history, model and test results, model and column metadata snapshots, and lineage relationships for each dbt pipeline run.

M8 is deterministic metadata ingestion and monitoring infrastructure. The M9 reviewer was added later as a separate milestone.

---

## Purpose

The goal of M8 is to make the scheduled dbt pipeline observable and historically queryable.

The completed flow is:

```text
Cloud Scheduler
    ↓
Cloud Run Job
    ↓
dbt debug
    ↓
dbt build --target prod
    ↓
preserve build manifest.json and run_results.json
    ↓
dbt docs generate --target prod
    ↓
keep catalog.json and restore the build artifacts
    ↓
Python artifact parser
    ↓
BigQuery append loader
    ↓
olist_monitoring
```

This layer turns dbt runtime by-products into a structured analytics layer for pipeline monitoring.

---

## Scope

M8 includes:

- Parsing `manifest.json`
- Parsing `run_results.json`
- Parsing `catalog.json`
- Creating BigQuery dataset `olist_monitoring`
- Creating six append-only monitoring tables
- Preserving historical pipeline run records
- Tracking model status and runtime history
- Tracking dbt test results and failures
- Storing model metadata snapshots
- Storing model/source column metadata snapshots
- Storing lineage dependency edges
- Loading monitoring records locally and from Cloud Run
- Validating the latest monitoring run across all six tables
- Running the completed flow through Cloud Scheduler

M8 supports analysis such as:

- Current run vs previous run
- Current run vs recent average
- Model runtime and status history
- Test pass/fail/warn/error history
- Row-count and table-size change analysis where catalog statistics are available
- Documentation and column-test coverage
- Source-to-model and model-to-model lineage
- Downstream impact analysis foundation

---

## Out of scope

At M8 completion, the milestone did not include:

- the M9 pipeline reviewer
- optional explanation for review findings
- Power BI monitoring pages
- React portal integration
- Airflow, Composer, Dagster, or Prefect
- Full Agile Data Engine integration
- DataHub or OpenLineage integration
- alerting or notification delivery
- automated root-cause analysis
- a governance UI

Some later milestones implemented parts of this list. They are still kept outside the M8 milestone boundary.

---

## Source artifacts

M8 uses three dbt artifacts.

| Artifact | Source command | Main use |
|---|---|---|
| `manifest.json` | `dbt build` | Project structure, models, tests, sources, configs, descriptions, columns, and dependencies |
| `run_results.json` | `dbt build` | Actual model/test execution status, runtime, failures, adapter response, invocation metadata |
| `catalog.json` | `dbt docs generate` | Warehouse relation metadata, data types, column order, row counts, and table size where available |

### Why build artifacts are preserved

`dbt docs generate` is run to create `catalog.json`, but it also regenerates `manifest.json` and `run_results.json`.

For monitoring pipeline health, the useful execution result is the `run_results.json` produced by `dbt build`, because it contains the actual model and test execution results.

The Cloud Run entrypoint therefore performs this sequence:

```text
dbt build
→ copy build manifest.json and run_results.json to a temporary directory
→ dbt docs generate
→ keep the new catalog.json
→ restore the build manifest.json and run_results.json
→ run the monitoring loader
```

The JSON files are not appended or merged. The build artifacts are restored by replacement so that the final artifact set is:

```text
manifest.json      ← dbt build
run_results.json   ← dbt build
catalog.json       ← dbt docs generate
```

---

## Metadata transformation

The most complex part of M8 is the JSON-to-relational transformation.

```text
Nested dbt artifact JSON
    ↓ parse, filter, join, normalize, flatten
Structured monitoring records
    ↓ append
BigQuery monitoring tables
```

The parser joins artifact objects through dbt `unique_id` values and converts nested structures into table-shaped records.

Examples:

```text
run_results.results[]
→ pipeline summary
→ model execution records
→ test execution records
```

```text
manifest.nodes + catalog.nodes
→ model metadata snapshots
```

```text
manifest columns + catalog columns + manifest test nodes
→ column metadata and test-coverage snapshots
```

```text
manifest depends_on.nodes
→ lineage edges
```

---

## Target BigQuery dataset

```text
olist_monitoring
```

All monitoring tables are append-only. Each successful metadata refresh adds a new `monitoring_run_id` and a complete set of related records.

The tables are partitioned by `DATE(ingested_at)` and clustered by frequently queried monitoring dimensions.

---

## Monitoring tables

### `pipeline_runs`

Grain: one row per monitoring load / dbt pipeline execution.

Stores:

- `monitoring_run_id`
- dbt invocation ID
- job and environment identity
- dbt version
- run timestamps and elapsed time
- overall pipeline status
- model and test totals
- artifact paths

### `model_run_results`

Grain: one row per model, seed, or snapshot execution per monitoring run.

Stores:

- model identity and relation metadata
- materialization
- execution status and runtime
- thread and adapter response
- error or informational message

### `test_run_results`

Grain: one row per dbt test execution per monitoring run.

Stores:

- generic/singular test identity
- attached model and column
- normalized status: `pass`, `fail`, `warn`, or `error`
- severity and failures
- runtime, message, and adapter response

### `model_metadata_snapshots`

Grain: one row per model, seed, or snapshot per monitoring run.

Stores:

- model path, relation, schema, alias, and materialization
- descriptions, tags, and meta
- row count and bytes where catalog statistics are available
- catalog relation metadata

### `model_column_snapshots`

Grain: one row per model/source column per monitoring run.

Stores:

- parent model or source
- column name, type, order, and description
- column-level dbt test summaries
- raw catalog column metadata

The validated run included both dbt model columns and raw source columns.

### `model_lineage_edges`

Grain: one row per direct dependency edge per monitoring run.

Stores:

- parent and child `unique_id`
- display names and resource types
- dependency type

The table preserves source-to-model, model-to-model, and model-to-test relationships.

---

## Implementation files

```text
dbt/monitoring/inspect_artifacts.py
dbt/monitoring/artifact_parser.py
dbt/monitoring/load_artifacts_to_bigquery.py

dbt/sql/monitoring/create_olist_monitoring_dataset.sql
dbt/sql/monitoring/create_monitoring_tables.sql
dbt/sql/monitoring/validate_monitoring_tables.sql
dbt/sql/monitoring/validate_latest_monitoring_run.sql

dbt/run_dbt_job.sh
dbt/Dockerfile
```

### Responsibilities

`inspect_artifacts.py`

- Reports artifact paths, node/source counts, dbt version, invocation ID, and elapsed time
- Supports initial artifact structure inspection

`artifact_parser.py`

- Reads and validates artifact files
- Builds six monitoring record collections
- Supports runtime artifact path, job name, and environment configuration

`load_artifacts_to_bigquery.py`

- Creates a BigQuery client
- Builds monitoring records through the parser
- Appends records to the six monitoring tables

`run_dbt_job.sh`

- Runs the production dbt pipeline
- Preserves build artifacts
- Generates the catalog
- Restores build execution artifacts
- Runs the BigQuery monitoring loader

---

## Runtime configuration

| Environment variable | Default | Purpose |
|---|---|---|
| `DBT_ARTIFACT_DIR` | `<dbt project>/target` | Artifact directory used by the parser |
| `GCP_PROJECT_ID` | BigQuery client default project | BigQuery project used by the loader |
| `MONITORING_DATASET_ID` | `olist_monitoring` | Monitoring dataset |
| `MONITORING_JOB_NAME` | `local-dbt-artifact-inspection` locally; set to `olist-dbt-build-job` in Cloud Run | Pipeline job identity |
| `MONITORING_ENVIRONMENT` | `dev` locally; set to `prod` in Cloud Run | Runtime environment identity |

The Cloud Run Job sets:

```text
MONITORING_DATASET_ID=olist_monitoring
MONITORING_JOB_NAME=olist-dbt-build-job
MONITORING_ENVIRONMENT=prod
```

---

## Validated output

### Local parser output

```text
pipeline_runs                  1
model_run_results             21
test_run_results              94
model_metadata_snapshots      21
model_column_snapshots       259
model_lineage_edges          146
```

Column snapshot breakdown:

```text
model columns                 207
source columns                 52
documented columns            176
columns with tests             58
```

Lineage breakdown:

```text
source → model edges            9
model → model edges            21
model → test edges             116
total lineage edges            146
```

### Cloud Run smoke test

Validated on 2026-07-15:

```text
Cloud Run Job: olist-dbt-build-job
Image tag: m8
Environment: prod
Pipeline status: success
Models: 21 / 21 successful
Tests: 94 / 94 passed
Non-passing tests: 0
```

### Cloud Scheduler end-to-end validation

Validated Scheduler-triggered execution:

```text
Cloud Scheduler: olist-dbt-daily-trigger
Cloud Run execution: olist-dbt-build-job-f59xf
Triggered by: olist-scheduler-invoker service account
Tasks: 1 / 1 completed successfully
```

Latest cross-table validation:

```text
pipeline_runs                  1
model_run_results             21
test_run_results              94
model_metadata_snapshots      21
model_column_snapshots       259
model_lineage_edges          146

successful_models             21
passed_tests                  94
non_passing_tests              0
```

---

## Validation approach

The completed implementation uses two levels of validation.

### Table existence validation

```text
dbt/sql/monitoring/validate_monitoring_tables.sql
```

Confirms that all six tables exist and can be queried.

### Latest-run integrity validation

```text
dbt/sql/monitoring/validate_latest_monitoring_run.sql
```

Selects the latest `pipeline_runs` row and checks that the same `monitoring_run_id` has a complete and consistent record set across all monitoring tables.

This verifies:

- pipeline/model/test totals match
- model and test success counts match
- metadata, column, and lineage snapshots exist
- no non-passing tests are present in the validated run

---

## Design principles

1. Keep monitoring history append-only.
2. Preserve build execution results rather than docs-generation execution results.
3. Keep parser logic separate from BigQuery loading logic.
4. Use `unique_id` as the cross-artifact join key.
5. Allow optional artifact fields to be missing without failing unrelated records.
6. Make runtime paths and environment identity configurable.
7. Reuse the existing Cloud Run Job and Cloud Scheduler rather than adding another orchestrator.
8. Keep the implementation deterministic, queryable, and explainable.
9. Keep M8 separate from the later M9 reviewer.

---

## Completion criteria

M8 is complete because:

- `olist_monitoring` exists in BigQuery.
- Six append-only monitoring tables are implemented.
- All six artifact record types are parsed.
- Local BigQuery loading is validated.
- Docker image `m8` contains the monitoring code and required dependencies.
- The Cloud Run Job runs the metadata refresh after dbt build.
- Manual Cloud Run execution is validated.
- Cloud Scheduler-triggered execution is validated.
- The latest monitoring run passes cross-table integrity validation.
- Architecture, commands, README, and M8 documentation are finalized.

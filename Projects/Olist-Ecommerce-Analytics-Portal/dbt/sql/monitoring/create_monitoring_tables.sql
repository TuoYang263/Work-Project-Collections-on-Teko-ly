-- Create BigQuery monitoring tables for M8 metadata refresh.
-- These tables store append-only dbt pipeline monitoring history.

CREATE TABLE IF NOT EXISTS `olist_monitoring.pipeline_runs` (
    monitoring_run_id STRING OPTIONS(description = "Generated unique identifier for each metadata ingestion run."),
    dbt_invocation_id STRING OPTIONS(description = "dbt invocation_id from dbt artifacts."),
    job_name STRING OPTIONS(description = "Cloud Run Job or execution job name."),
    environment STRING OPTIONS(description = "Execution environment, such as prod or dev."),
    dbt_version STRING OPTIONS(description = "dbt version used for the pipeline execution."),

    generated_at TIMESTAMP OPTIONS(description = "Timestamp when dbt artifacts were generated."),
    ingested_at TIMESTAMP OPTIONS(description = "Timestamp when metadata was loaded into BigQuery."),
    run_started_at TIMESTAMP OPTIONS(description = "Timestamp when the pipeline run started."),
    run_completed_at TIMESTAMP OPTIONS(description = "Timestamp when the metadata load completed."),

    total_elapsed_time_seconds FLOAT64 OPTIONS(description = "Total elapsed time reported by dbt run_results.json."),
    status STRING OPTIONS(description = "Overall pipeline status, such as success, error, partial_failure, or unknown."),

    models_total INT64 OPTIONS(description = "Total number of model executions in this pipeline run."),
    models_success INT64 OPTIONS(description = "Number of successful model executions."),
    models_error INT64 OPTIONS(description = "Number of failed model executions."),
    models_skipped INT64 OPTIONS(description = "Number of skipped model executions."),

    tests_total INT64 OPTIONS(description = "Total number of dbt test executions in this pipeline run."),
    tests_passed INT64 OPTIONS(description = "Number of passed dbt tests."),
    tests_failed INT64 OPTIONS(description = "Number of failed dbt tests."),
    tests_warned INT64 OPTIONS(description = "Number of dbt tests with warning status."),
    tests_error INT64 OPTIONS(description = "Number of dbt tests with error status."),

    artifact_manifest_path STRING OPTIONS(description = "Path or reference to manifest.json if available."),
    artifact_run_results_path STRING OPTIONS(description = "Path or reference to run_results.json if available."),
    artifact_catalog_path STRING OPTIONS(description = "Path or reference to catalog.json if available.")
)
-- Partition the table by ingestion date because monitoring records are appended
-- after each pipeline execution. This allows recent run history queries to scan
-- only the relevant daily partitions instead of the full table.
PARTITION BY DATE(ingested_at)

-- Cluster records within each partition by common filtering fields.
-- This helps queries that focus on pipeline status or execution environment,
-- such as finding recent failed production runs.
CLUSTER BY status, environment

OPTIONS (
    description = "Append-only pipeline-level dbt run history generated from dbt artifacts."
);

CREATE TABLE IF NOT EXISTS `olist_monitoring.model_run_results` (
    monitoring_run_id STRING OPTIONS(description = "Generated unique identifier for each metadata ingestion run."),
    dbt_invocation_id STRING OPTIONS(description = "dbt invocation_id from dbt artifacts."),

    unique_id STRING OPTIONS(description = "dbt unique_id for the executed model node."),
    model_name STRING OPTIONS(description = "dbt model name."),
    resource_type STRING OPTIONS(description = "dbt resource type, such as model, seed, or snapshot."),
    package_name STRING OPTIONS(description = "dbt package name."),

    database_name STRING OPTIONS(description = "Target database or BigQuery project name."),
    schema_name STRING OPTIONS(description = "Target BigQuery dataset name."),
    alias STRING OPTIONS(description = "Final relation alias used by dbt."),
    materialized STRING OPTIONS(description = "dbt materialization type, such as view, table, or incremental."),

    status STRING OPTIONS(description = "Execution status of the model, such as success, error, skipped, or unknown."),
    execution_time_seconds FLOAT64 OPTIONS(description = "Model execution time in seconds from run_results.json."),
    thread_id STRING OPTIONS(description = "dbt thread identifier used during model execution."),
    message STRING OPTIONS(description = "dbt execution message if available."),

    adapter_response_json STRING OPTIONS(description = "Raw adapter response serialized as JSON."),
    ingested_at TIMESTAMP OPTIONS(description = "Timestamp when metadata was loaded into BigQuery.")
)
-- Partition model run results by ingestion date because execution records
-- are appended after each pipeline run.
PARTITION BY DATE(ingested_at)

-- Cluster by model name and status to support common monitoring queries,
-- such as finding failed models or checking runtime trends for a specific model.
CLUSTER BY model_name, status

OPTIONS (
    description = "Append-only dbt model execution history generated from run_results.json and manifest.json."
);
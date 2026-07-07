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

CREATE TABLE IF NOT EXISTS `olist_monitoring.test_run_results` (
    monitoring_run_id STRING OPTIONS(description = "Generated unique identifier for each metadata ingestion run."),
    dbt_invocation_id STRING OPTIONS(description = "dbt invocation_id from dbt artifacts."),

    unique_id STRING OPTIONS(description = "dbt unique_id for the executed test node."),
    test_name STRING OPTIONS(description = "dbt test node name."),
    test_type STRING OPTIONS(description = "dbt test type, such as generic or singular."),
    test_metadata_name STRING OPTIONS(description = "dbt generic test name, such as not_null, unique, relationships, or accepted_values."),

    model_unique_id STRING OPTIONS(description = "dbt unique_id of the model or source tested by this test."),
    model_name STRING OPTIONS(description = "Name of the model or source tested by this test."),
    column_name STRING OPTIONS(description = "Column name tested by this test, if applicable."),

    status STRING OPTIONS(description = "Execution status of the test, such as pass, fail, warn, error, skipped, or unknown."),
    severity STRING OPTIONS(description = "Configured dbt test severity, such as warn or error."),
    failures INT64 OPTIONS(description = "Number of failing records reported by dbt, if available."),
    execution_time_seconds FLOAT64 OPTIONS(description = "Test execution time in seconds from run_results.json."),
    thread_id STRING OPTIONS(description = "dbt thread identifier used during test execution."),
    message STRING OPTIONS(description = "dbt test execution or failure message if available."),

    adapter_response_json STRING OPTIONS(description = "Raw adapter response serialized as JSON."),
    ingested_at TIMESTAMP OPTIONS(description = "Timestamp when metadata was loaded into BigQuery.")
)
-- Partition test run results by ingestion date because test execution records
-- are appended after each pipeline run.
PARTITION BY DATE(ingested_at)

-- Cluster by model name, status, and test metadata name to support common
-- monitoring queries, such as finding failed tests for a model or tracking
-- recurring failures by test type.
CLUSTER BY model_name, status, test_metadata_name

OPTIONS (
    description = "Append-only dbt test execution history generated from run_results.json and manifest.json."
);

CREATE TABLE IF NOT EXISTS `olist_monitoring.model_metadata_snapshots` (
    monitoring_run_id STRING OPTIONS(description = "Generated unique identifier for each metadata ingestion run."),
    dbt_invocation_id STRING OPTIONS(description = "dbt invocation_id from dbt artifacts."),

    unique_id STRING OPTIONS(description = "dbt unique_id for the model node."),
    model_name STRING OPTIONS(description = "dbt model name."),
    resource_type STRING OPTIONS(description = "dbt resource type, such as model, seed, snapshot, or source."),
    package_name STRING OPTIONS(description = "dbt package name."),

    database_name STRING OPTIONS(description = "Target database or BigQuery project name."),
    schema_name STRING OPTIONS(description = "Target BigQuery dataset name."),
    alias STRING OPTIONS(description = "Final relation alias used by dbt."),
    relation_name STRING OPTIONS(description = "Fully qualified relation name if available."),
    materialized STRING OPTIONS(description = "dbt materialization type, such as view, table, or incremental."),

    path STRING OPTIONS(description = "Relative dbt model path."),
    original_file_path STRING OPTIONS(description = "Original file path of the dbt model or source definition."),
    description STRING OPTIONS(description = "dbt model description."),

    tags_json STRING OPTIONS(description = "dbt tags serialized as JSON."),
    meta_json STRING OPTIONS(description = "dbt meta configuration serialized as JSON."),

    row_count INT64 OPTIONS(description = "Model row count from catalog.json if available."),
    bytes INT64 OPTIONS(description = "Model table size in bytes from catalog.json if available."),
    catalog_metadata_json STRING OPTIONS(description = "Raw catalog metadata serialized as JSON."),

    ingested_at TIMESTAMP OPTIONS(description = "Timestamp when metadata was loaded into BigQuery.")
)
-- Partition model metadata snapshots by ingestion date because each pipeline
-- run appends a new snapshot of model-level metadata.
PARTITION BY DATE(ingested_at)

-- Cluster by model name and materialization to support model inventory,
-- documentation, and row count change analysis.
CLUSTER BY model_name, materialized

OPTIONS (
    description = "Append-only dbt model metadata snapshots generated from manifest.json and catalog.json."
);

CREATE TABLE IF NOT EXISTS `olist_monitoring.model_column_snapshots` (
    monitoring_run_id STRING OPTIONS(description = "Generated unique identifier for each metadata ingestion run."),
    dbt_invocation_id STRING OPTIONS(description = "dbt invocation_id from dbt artifacts."),

    model_unique_id STRING OPTIONS(description = "dbt unique_id of the parent model or source."),
    model_name STRING OPTIONS(description = "Name of the parent model or source."),
    resource_type STRING OPTIONS(description = "dbt resource type of the parent node, such as model or source."),

    column_name STRING OPTIONS(description = "Column name from manifest.json or catalog.json."),
    data_type STRING OPTIONS(description = "Column data type from catalog.json if available."),
    column_index INT64 OPTIONS(description = "Column ordinal position if available."),

    description STRING OPTIONS(description = "dbt column description."),
    tests_json STRING OPTIONS(description = "Related dbt column tests serialized as JSON if available."),
    catalog_column_metadata_json STRING OPTIONS(description = "Raw catalog column metadata serialized as JSON."),

    ingested_at TIMESTAMP OPTIONS(description = "Timestamp when metadata was loaded into BigQuery.")
)
-- Partition column metadata snapshots by ingestion date because each pipeline
-- run appends a new snapshot of model and source columns.
PARTITION BY DATE(ingested_at)

-- Cluster by model and column name to support documentation coverage,
-- schema inspection, and future column-level impact analysis.
CLUSTER BY model_name, column_name

OPTIONS (
    description = "Append-only dbt model and source column metadata snapshots generated from manifest.json and catalog.json."
);

CREATE TABLE IF NOT EXISTS `olist_monitoring.model_lineage_edges` (
    monitoring_run_id STRING OPTIONS(description = "Generated unique identifier for each metadata ingestion run."),
    dbt_invocation_id STRING OPTIONS(description = "dbt invocation_id from dbt artifacts."),

    parent_unique_id STRING OPTIONS(description = "dbt unique_id of the upstream parent node."),
    parent_name STRING OPTIONS(description = "Name of the upstream parent node."),
    parent_resource_type STRING OPTIONS(description = "Resource type of the upstream parent node, such as source, model, seed, or snapshot."),

    child_unique_id STRING OPTIONS(description = "dbt unique_id of the downstream child node."),
    child_name STRING OPTIONS(description = "Name of the downstream child node."),
    child_resource_type STRING OPTIONS(description = "Resource type of the downstream child node, such as model or test."),

    dependency_type STRING OPTIONS(description = "Type of dependency relationship, such as depends_on_node."),

    ingested_at TIMESTAMP OPTIONS(description = "Timestamp when metadata was loaded into BigQuery.")
)
-- Partition lineage edges by ingestion date because each pipeline run appends
-- a new snapshot of dbt dependency relationships.
PARTITION BY DATE(ingested_at)

-- Cluster by parent and child names to support lineage lookup and downstream
-- impact analysis from failed or changed upstream nodes.
CLUSTER BY parent_name, child_name

OPTIONS (
    description = "Append-only dbt lineage dependency edges generated from manifest.json."
);
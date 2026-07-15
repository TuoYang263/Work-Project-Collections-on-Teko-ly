-- Validate the latest M8 monitoring load across all six tables.

WITH latest_run AS (
    SELECT
        monitoring_run_id,
        dbt_invocation_id,
        status,
        models_total,
        tests_total,
        ingested_at
    FROM `olist_monitoring.pipeline_runs`
    ORDER BY ingested_at DESC
    LIMIT 1
)

SELECT
    latest_run.monitoring_run_id,
    latest_run.dbt_invocation_id,
    latest_run.status AS pipeline_status,
    latest_run.ingested_at,

    latest_run.models_total,
    (
        SELECT COUNT(*)
        FROM `olist_monitoring.model_run_results`
        WHERE monitoring_run_id = latest_run.monitoring_run_id
    ) AS model_run_result_count,

    latest_run.tests_total,
    (
        SELECT COUNT(*)
        FROM `olist_monitoring.test_run_results`
        WHERE monitoring_run_id = latest_run.monitoring_run_id
    ) AS test_run_result_count,

    (
        SELECT COUNT(*)
        FROM `olist_monitoring.model_metadata_snapshots`
        WHERE monitoring_run_id = latest_run.monitoring_run_id
    ) AS model_metadata_snapshot_count,

    (
        SELECT COUNT(*)
        FROM `olist_monitoring.model_column_snapshots`
        WHERE monitoring_run_id = latest_run.monitoring_run_id
    ) AS model_column_snapshot_count,

    (
        SELECT COUNT(*)
        FROM `olist_monitoring.model_lineage_edges`
        WHERE monitoring_run_id = latest_run.monitoring_run_id
    ) AS model_lineage_edge_count,

    (
        SELECT COUNTIF(status = 'success')
        FROM `olist_monitoring.model_run_results`
        WHERE monitoring_run_id = latest_run.monitoring_run_id
    ) AS successful_models,

    (
        SELECT COUNTIF(status = 'pass')
        FROM `olist_monitoring.test_run_results`
        WHERE monitoring_run_id = latest_run.monitoring_run_id
    ) AS passed_tests,

    (
        SELECT COUNTIF(status IN ('fail', 'warn', 'error'))
        FROM `olist_monitoring.test_run_results`
        WHERE monitoring_run_id = latest_run.monitoring_run_id
    ) AS non_passing_tests

FROM latest_run;
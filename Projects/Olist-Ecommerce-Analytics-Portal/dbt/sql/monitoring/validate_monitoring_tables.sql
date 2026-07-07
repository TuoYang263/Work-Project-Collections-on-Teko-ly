-- Validate that M8 monitoring tables exist and are queryable.
-- Expected result before metadata loading: all row_count values should be 0.

SELECT
    'pipeline_runs' AS table_name,
    COUNT(*) AS row_count
FROM `olist_monitoring.pipeline_runs`

UNION ALL

SELECT
    'model_run_results' AS table_name,
    COUNT(*) AS row_count
FROM `olist_monitoring.model_run_results`

UNION ALL

SELECT
    'test_run_results' AS table_name,
    COUNT(*) AS row_count
FROM `olist_monitoring.test_run_results`

UNION ALL

SELECT
    'model_metadata_snapshots' AS table_name,
    COUNT(*) AS row_count
FROM `olist_monitoring.model_metadata_snapshots`

UNION ALL

SELECT
    'model_column_snapshots' AS table_name,
    COUNT(*) AS row_count
FROM `olist_monitoring.model_column_snapshots`

UNION ALL

SELECT
    'model_lineage_edges' AS table_name,
    COUNT(*) AS row_count
FROM `olist_monitoring.model_lineage_edges`

ORDER BY table_name;
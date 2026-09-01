CREATE TABLE IF NOT EXISTS `olist_monitoring.pipeline_review_runs`
(
    review_id STRING NOT NULL,
    monitoring_run_id STRING NOT NULL,

    job_name STRING NOT NULL,
    environment STRING NOT NULL,

    total_evaluations INT64 NOT NULL,
    pass_count INT64 NOT NULL,
    triggered_count INT64 NOT NULL,
    not_evaluated_count INT64 NOT NULL,

    reviewed_at TIMESTAMP NOT NULL
)
PARTITION BY DATE(reviewed_at)
CLUSTER BY
    job_name,
    environment,
    monitoring_run_id;


CREATE TABLE IF NOT EXISTS `olist_monitoring.pipeline_review_evaluations`
(
    review_id STRING NOT NULL,
    monitoring_run_id STRING NOT NULL,

    evaluation_id STRING NOT NULL,
    finding_id STRING,

    rule_id STRING NOT NULL,
    result STRING NOT NULL,
    severity STRING,

    entity_type STRING,
    entity_id STRING,

    evidence_source STRING,
    evidence_json JSON,
    reason STRING,

    reviewed_at TIMESTAMP NOT NULL
)
PARTITION BY DATE(reviewed_at)
CLUSTER BY
    monitoring_run_id,
    result,
    rule_id;

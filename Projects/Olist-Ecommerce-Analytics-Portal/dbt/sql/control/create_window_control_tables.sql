CREATE TABLE IF NOT EXISTS `olist_control.pipeline_control_state`
(
    pipeline_name STRING NOT NULL,
    environment STRING NOT NULL,

    state STRING NOT NULL,

    last_successful_window_start TIMESTAMP,
    last_successful_window_end TIMESTAMP,

    active_window_start TIMESTAMP,
    active_window_end TIMESTAMP,
    active_attempt_id STRING,
    active_attempt_number INT64,
    active_retry_of_attempt_id STRING,

    control_version INT64 NOT NULL,

    last_error_code STRING,
    last_error_message STRING,

    updated_at TIMESTAMP NOT NULL
)
-- Cluster by the main lookup keys used to identify one pipeline environment.
CLUSTER BY
    pipeline_name,
    environment;

CREATE TABLE IF NOT EXISTS `olist_control.pipeline_window_events`
(
    event_id STRING NOT NULL,
    attempt_id STRING NOT NULL,

    pipeline_name STRING NOT NULL,
    environment STRING NOT NULL,

    window_start TIMESTAMP NOT NULL,
    window_end TIMESTAMP NOT NULL,

    attempt_number INT64 NOT NULL,

    from_state STRING,
    to_state STRING NOT NULL,

    from_control_version INT64 NOT NULL,
    to_control_version INT64 NOT NULL,

    event_type STRING NOT NULL,
    event_time TIMESTAMP NOT NULL,

    retry_of_attempt_id STRING,

    error_code STRING,
    error_message STRING,

    metadata_json JSON
)
-- Partition by event date so time-bounded queries can scan less data.
PARTITION BY DATE(event_time)

-- Cluster by common operational filters to reduce scanned data within partitions.
CLUSTER BY
    pipeline_name,
    environment,
    attempt_id,
    to_state;
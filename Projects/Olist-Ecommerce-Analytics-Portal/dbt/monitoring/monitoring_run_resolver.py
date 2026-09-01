from __future__ import annotations

from google.cloud import bigquery

DEFAULT_MONITORING_DATASET_ID = "olist_monitoring"


class MonitoringRunResolutionError(RuntimeError):
    """Base error for monitoring-run resolution failures."""


class MonitoringRunNotFoundError(MonitoringRunResolutionError):
    """Raised when an attempt has no persisted monitoring run."""


class MonitoringRunIntegrityError(MonitoringRunResolutionError):
    """Raised when monitoring-run correlation is ambiguous."""


class BigQueryMonitoringRunResolver:
    def __init__(
        self,
        client: bigquery.Client,
        *,
        dataset_id: str = DEFAULT_MONITORING_DATASET_ID,
        table_id: str = "pipeline_runs",
    ) -> None:
        self._client = client

        self._table_fqn = f"{client.project}.{dataset_id}.{table_id}"

    def resolve(
        self,
        *,
        control_attempt_id: str,
    ) -> str:
        attempt_id = control_attempt_id.strip()

        if not attempt_id:
            raise ValueError("control_attempt_id must be non-empty")

        query = f"""
        SELECT
            monitoring_run_id
        FROM `{self._table_fqn}`
        WHERE control_attempt_id = @control_attempt_id
        LIMIT 2
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter(
                    "control_attempt_id",
                    "STRING",
                    attempt_id,
                )
            ]
        )

        rows = list(
            self._client.query(
                query,
                job_config=job_config,
            ).result()
        )

        if not rows:
            raise MonitoringRunNotFoundError(
                "no monitoring run found for " f"control_attempt_id={attempt_id!r}"
            )

        if len(rows) > 1:
            raise MonitoringRunIntegrityError(
                "multiple monitoring runs found for "
                f"control_attempt_id={attempt_id!r}"
            )

        monitoring_run_id = rows[0]["monitoring_run_id"]

        if monitoring_run_id is None or not str(monitoring_run_id).strip():
            raise MonitoringRunIntegrityError(
                "monitoring run has an empty "
                "monitoring_run_id for "
                f"control_attempt_id={attempt_id!r}"
            )

        return str(monitoring_run_id).strip()

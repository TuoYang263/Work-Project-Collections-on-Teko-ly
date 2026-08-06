from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class BigQueryDependencyError(RuntimeError):
    pass


class BigQueryQueryExecutor:
    """Thin adapter from google-cloud-bigquery to the loader QueryExecutor."""

    def __init__(self, client: Any | None = None, project_id: str | None = None) -> None:
        bigquery = self._import_bigquery()
        self._bigquery = bigquery
        self._client = client or bigquery.Client(project=project_id)

    def execute(
        self,
        sql: str,
        parameters: Mapping[str, Any],
    ) -> Sequence[Mapping[str, Any]]:
        query_parameters = [
            self._bigquery.ScalarQueryParameter(
                name,
                self._parameter_type(value),
                value,
            )
            for name, value in parameters.items()
        ]
        job_config = self._bigquery.QueryJobConfig(
            query_parameters=query_parameters
        )
        query_job = self._client.query(sql, job_config=job_config)
        return [dict(row.items()) for row in query_job.result()]

    @staticmethod
    def _parameter_type(value: Any) -> str:
        if isinstance(value, bool):
            return "BOOL"
        if isinstance(value, int):
            return "INT64"
        if isinstance(value, float):
            return "FLOAT64"
        if isinstance(value, str):
            return "STRING"

        raise TypeError(
            "Unsupported BigQuery scalar parameter type: "
            f"{type(value).__name__}"
        )

    @staticmethod
    def _import_bigquery() -> Any:
        try:
            from google.cloud import bigquery
        except ImportError as exc:
            raise BigQueryDependencyError(
                "google-cloud-bigquery is required for live evidence loading"
            ) from exc

        return bigquery
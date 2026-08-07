from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol


class EvidenceLoadError(RuntimeError):
    pass


class QueryExecutor(Protocol):
    def execute(
        self,
        sql: str,
        parameters: Mapping[str, Any],
    ) -> Sequence[Mapping[str, Any]]:
        """Execute a parameterized query and return mapping-like rows."""


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    monitoring_run_id: str
    evidence: Mapping[str, tuple[Mapping[str, Any], ...]]
    comparable_run_ids: tuple[str, ...] = ()


class BigQueryEvidenceLoader:
    """Load normalized M9 review evidence from M8 monitoring tables."""

    _IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
    HISTORY_LIMIT = 5

    PIPELINE_FIELDS = (
        "monitoring_run_id",
        "dbt_invocation_id",
        "job_name",
        "environment",
        "generated_at",
        "ingested_at",
        "run_started_at",
        "run_completed_at",
        "total_elapsed_time_seconds",
        "status",
        "models_total",
        "models_success",
        "models_error",
        "models_skipped",
        "tests_total",
        "tests_passed",
        "tests_failed",
        "tests_warned",
        "tests_error",
    )

    MODEL_FIELDS = (
        "monitoring_run_id",
        "dbt_invocation_id",
        "unique_id",
        "model_name",
        "resource_type",
        "package_name",
        "database_name",
        "schema_name",
        "alias",
        "materialized",
        "status",
        "execution_time_seconds",
        "thread_id",
        "message",
        "adapter_response_json",
        "ingested_at",
    )

    MODEL_METADATA_FIELDS = (
        "monitoring_run_id",
        "dbt_invocation_id",
        "unique_id",
        "model_name",
        "resource_type",
        "package_name",
        "database_name",
        "schema_name",
        "alias",
        "relation_name",
        "materialized",
        "path",
        "original_file_path",
        "description",
        "tags_json",
        "meta_json",
        "row_count",
        "bytes",
        "catalog_metadata_json",
        "ingested_at",
    )

    TEST_FIELDS = (
        "monitoring_run_id",
        "dbt_invocation_id",
        "unique_id",
        "test_name",
        "test_type",
        "test_metadata_name",
        "model_unique_id",
        "model_name",
        "column_name",
        "status",
        "severity",
        "failures",
        "execution_time_seconds",
        "thread_id",
        "message",
        "adapter_response_json",
        "ingested_at",
    )

    def __init__(
        self,
        executor: QueryExecutor,
        project_id: str,
        dataset_id: str = "olist_monitoring",
    ) -> None:
        self._executor = executor
        self._project_id = self._validate_identifier(
            project_id,
            "project_id",
        )
        self._dataset_id = self._validate_identifier(
            dataset_id,
            "dataset_id",
        )

    def load_latest_status_evidence(
        self,
        job_name: str,
        environment: str,
    ) -> EvidenceBundle:
        job_name = self._require_non_empty(job_name, "job_name")
        environment = self._require_non_empty(
            environment,
            "environment",
        )

        sql = f"""
            SELECT monitoring_run_id
            FROM {self._table_ref("pipeline_runs")}
            WHERE job_name = @job_name
              AND environment = @environment
            ORDER BY generated_at DESC, ingested_at DESC
            LIMIT 1
        """

        rows = self._executor.execute(
            sql,
            {
                "job_name": job_name,
                "environment": environment,
            },
        )

        if not rows:
            raise EvidenceLoadError(
                "No monitoring run was found for "
                f"job_name={job_name!r}, environment={environment!r}"
            )

        monitoring_run_id = rows[0].get("monitoring_run_id")

        if not isinstance(monitoring_run_id, str) or not monitoring_run_id.strip():
            raise EvidenceLoadError("Latest pipeline run is missing monitoring_run_id")

        return self.load_status_evidence(monitoring_run_id)

    def load_status_evidence(
        self,
        monitoring_run_id: str,
    ) -> EvidenceBundle:
        monitoring_run_id = self._require_non_empty(
            monitoring_run_id,
            "monitoring_run_id",
        )

        current_evidence = {
            "pipeline_runs": self._load_table_rows(
                table_name="pipeline_runs",
                fields=self.PIPELINE_FIELDS,
                monitoring_run_id=monitoring_run_id,
            ),
            "model_run_results": self._load_table_rows(
                table_name="model_run_results",
                fields=self.MODEL_FIELDS,
                monitoring_run_id=monitoring_run_id,
            ),
            "test_run_results": self._load_table_rows(
                table_name="test_run_results",
                fields=self.TEST_FIELDS,
                monitoring_run_id=monitoring_run_id,
            ),
            "model_metadata_snapshots": self._load_table_rows(
                table_name="model_metadata_snapshots",
                fields=self.MODEL_METADATA_FIELDS,
                monitoring_run_id=monitoring_run_id,
            ),
        }

        comparable_pipeline_rows = self._load_comparable_pipeline_rows(
            monitoring_run_id=monitoring_run_id,
            history_limit=self.HISTORY_LIMIT,
        )

        comparable_run_ids = tuple(
            self._require_run_id(row) for row in comparable_pipeline_rows
        )

        if comparable_run_ids:
            historical_model_rows = self._load_rows_for_run_ids(
                table_name="model_run_results",
                fields=self.MODEL_FIELDS,
                monitoring_run_ids=comparable_run_ids,
            )

            historical_metadata_rows = self._load_rows_for_run_ids(
                table_name="model_metadata_snapshots",
                fields=self.MODEL_METADATA_FIELDS,
                monitoring_run_ids=comparable_run_ids,
            )
        else:
            historical_model_rows = ()
            historical_metadata_rows = ()

        evidence = {
            "pipeline_runs": (
                *current_evidence["pipeline_runs"],
                *comparable_pipeline_rows,
            ),
            "model_run_results": (
                *current_evidence["model_run_results"],
                *historical_model_rows,
            ),
            "test_run_results": tuple(current_evidence["test_run_results"]),
            "model_metadata_snapshots": (
                *current_evidence["model_metadata_snapshots"],
                *historical_metadata_rows,
            ),
        }

        immutable_evidence = MappingProxyType(
            {
                source: tuple(MappingProxyType(dict(row)) for row in rows)
                for source, rows in evidence.items()
            }
        )

        return EvidenceBundle(
            monitoring_run_id=monitoring_run_id,
            evidence=immutable_evidence,
            comparable_run_ids=comparable_run_ids,
        )

    def _load_comparable_pipeline_rows(
        self,
        monitoring_run_id: str,
        history_limit: int,
    ) -> Sequence[Mapping[str, Any]]:
        selected_fields = ",\n                ".join(
            f"candidate.{field}" for field in self.PIPELINE_FIELDS
        )

        sql = f"""
            WITH selected_run AS (
                SELECT
                    job_name,
                    environment,
                    generated_at
                FROM {self._table_ref("pipeline_runs")}
                WHERE monitoring_run_id = @monitoring_run_id
                ORDER BY ingested_at DESC
                LIMIT 1
            )

            SELECT
                    {selected_fields}
            FROM {self._table_ref("pipeline_runs")} AS candidate
            CROSS JOIN selected_run
            WHERE candidate.job_name = selected_run.job_name
            AND candidate.environment = selected_run.environment
            AND LOWER(TRIM(candidate.status)) = 'success'
            AND candidate.generated_at < selected_run.generated_at
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY candidate.monitoring_run_id
                ORDER BY candidate.ingested_at DESC
            ) = 1
            ORDER BY
                candidate.generated_at DESC,
                candidate.ingested_at DESC
            LIMIT @history_limit
        """

        return self._executor.execute(
            sql,
            {
                "monitoring_run_id": monitoring_run_id,
                "history_limit": history_limit,
            },
        )

    def _load_rows_for_run_ids(
        self,
        table_name: str,
        fields: Sequence[str],
        monitoring_run_ids: Sequence[str],
    ) -> Sequence[Mapping[str, Any]]:
        if not monitoring_run_ids:
            return ()

        selected_fields = ",\n                ".join(fields)

        sql = f"""
            SELECT
                    {selected_fields}
            FROM {self._table_ref(table_name)}
            WHERE monitoring_run_id IN UNNEST(@monitoring_run_ids)
        """

        return self._executor.execute(
            sql,
            {
                "monitoring_run_ids": tuple(monitoring_run_ids),
            },
        )

    @classmethod
    def _require_run_id(
        cls,
        row: Mapping[str, Any],
    ) -> str:
        monitoring_run_id = row.get("monitoring_run_id")

        if not isinstance(monitoring_run_id, str):
            raise EvidenceLoadError(
                "Comparable pipeline run is missing monitoring_run_id"
            )

        return cls._require_non_empty(
            monitoring_run_id,
            "monitoring_run_id",
        )

    def _load_table_rows(
        self,
        table_name: str,
        fields: Sequence[str],
        monitoring_run_id: str,
    ) -> Sequence[Mapping[str, Any]]:
        selected_fields = ",\n                ".join(fields)
        sql = f"""
            SELECT
                {selected_fields}
            FROM {self._table_ref(table_name)}
            WHERE monitoring_run_id = @monitoring_run_id
        """

        return self._executor.execute(
            sql,
            {"monitoring_run_id": monitoring_run_id},
        )

    def _table_ref(self, table_name: str) -> str:
        self._validate_identifier(table_name, "table_name")
        return f"`{self._project_id}.{self._dataset_id}.{table_name}`"

    @classmethod
    def _validate_identifier(cls, value: str, field_name: str) -> str:
        value = cls._require_non_empty(value, field_name)

        if not cls._IDENTIFIER_PATTERN.fullmatch(value):
            raise EvidenceLoadError(
                f"{field_name} contains unsupported characters: {value!r}"
            )

        return value

    @staticmethod
    def _require_non_empty(value: str, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise EvidenceLoadError(f"{field_name} must be non-empty")

        return value.strip()

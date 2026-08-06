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


class BigQueryEvidenceLoader:
    """Load normalized M9 status-rule evidence from M8 monitoring tables."""

    _IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")

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
            raise EvidenceLoadError(
                "Latest pipeline run is missing monitoring_run_id"
            )

        return self.load_status_evidence(monitoring_run_id)

    def load_status_evidence(
        self,
        monitoring_run_id: str,
    ) -> EvidenceBundle:
        monitoring_run_id = self._require_non_empty(
            monitoring_run_id,
            "monitoring_run_id",
        )

        evidence = {
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
        }

        immutable_evidence = MappingProxyType(
            {
                source: tuple(
                    MappingProxyType(dict(row))
                    for row in rows
                )
                for source, rows in evidence.items()
            }
        )

        return EvidenceBundle(
            monitoring_run_id=monitoring_run_id,
            evidence=immutable_evidence,
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
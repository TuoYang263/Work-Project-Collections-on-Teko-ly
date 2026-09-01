from __future__ import annotations

import sys
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REVIEWER_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REVIEWER_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from pipeline_reviewer import (  # noqa: E402
    BigQueryEvidenceLoader,
    DeterministicEvaluator,
    EvidenceLoadError,
    StatusReviewService,
    load_rule_catalog,
)


class FakeQueryExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def execute(
        self,
        sql: str,
        parameters: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        self.calls.append((sql, dict(parameters)))

        # Latest-run lookup.
        if (
            "SELECT monitoring_run_id" in sql
            and "WHERE job_name = @job_name" in sql
            and "WITH selected_run AS" not in sql
        ):
            return [
                {
                    "monitoring_run_id": "run-002",
                }
            ]

        # Comparable historical runs.
        # Base fake intentionally returns no history so the existing
        # R001-R003 tests remain focused on current-run evidence.
        if "WITH selected_run AS" in sql:
            return []

        if "model_metadata_snapshots" in sql:
            return [
                {
                    "monitoring_run_id": parameters["monitoring_run_id"],
                    "unique_id": "model.olist.fct_orders",
                    "model_name": "fct_orders",
                    "row_count": 1000,
                }
            ]

        if "model_run_results" in sql:
            return [
                {
                    "monitoring_run_id": parameters["monitoring_run_id"],
                    "unique_id": "model.olist.fct_orders",
                    "model_name": "fct_orders",
                    "status": "error",
                    "execution_time_seconds": 10.0,
                    "message": "Synthetic model failure",
                }
            ]

        if "test_run_results" in sql:
            return [
                {
                    "monitoring_run_id": parameters["monitoring_run_id"],
                    "unique_id": "test.olist.not_null_orders",
                    "test_name": "not_null_orders",
                    "status": "warn",
                    "severity": "warn",
                }
            ]

        if "pipeline_runs" in sql:
            return [
                {
                    "monitoring_run_id": parameters["monitoring_run_id"],
                    "job_name": "olist-dbt-build-job",
                    "environment": "prod",
                    "status": "success",
                }
            ]

        raise AssertionError(f"Unexpected SQL: {sql}")


class EmptyLatestRunExecutor(FakeQueryExecutor):
    def execute(
        self,
        sql: str,
        parameters: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        if "LIMIT 1" in sql:
            return []
        return super().execute(sql, parameters)


class BigQueryEvidenceLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.executor = FakeQueryExecutor()
        self.loader = BigQueryEvidenceLoader(
            executor=self.executor,
            project_id="balmy-nuance-468118-g4",
            dataset_id="olist_monitoring",
        )

    def test_load_explicit_run_returns_expected_evidence_sources(
        self,
    ) -> None:
        bundle = self.loader.load_status_evidence("run-001")

        self.assertEqual(
            bundle.monitoring_run_id,
            "run-001",
        )

        self.assertEqual(
            bundle.comparable_run_ids,
            (),
        )

        self.assertEqual(
            set(bundle.evidence),
            {
                "pipeline_runs",
                "model_run_results",
                "test_run_results",
                "model_metadata_snapshots",
            },
        )

        self.assertEqual(
            len(bundle.evidence["pipeline_runs"]),
            1,
        )
        self.assertEqual(
            len(bundle.evidence["model_run_results"]),
            1,
        )
        self.assertEqual(
            len(bundle.evidence["test_run_results"]),
            1,
        )
        self.assertEqual(
            len(bundle.evidence["model_metadata_snapshots"]),
            1,
        )

    def test_latest_run_is_selected_by_job_and_environment(self) -> None:
        bundle = self.loader.load_latest_status_evidence(
            job_name="olist-dbt-build-job",
            environment="prod",
        )

        self.assertEqual(bundle.monitoring_run_id, "run-002")
        latest_sql, latest_parameters = self.executor.calls[0]
        self.assertIn("ORDER BY generated_at DESC", latest_sql)
        self.assertEqual(
            latest_parameters,
            {
                "job_name": "olist-dbt-build-job",
                "environment": "prod",
            },
        )

    def test_evidence_rows_and_source_mapping_are_read_only(self) -> None:
        bundle = self.loader.load_status_evidence("run-001")

        with self.assertRaises(TypeError):
            bundle.evidence["new_source"] = ()  # type: ignore[index]

        with self.assertRaises(TypeError):
            bundle.evidence["pipeline_runs"][0]["status"] = "error"  # type: ignore[index]

    def test_missing_latest_run_raises_clear_error(self) -> None:
        loader = BigQueryEvidenceLoader(
            executor=EmptyLatestRunExecutor(),
            project_id="balmy-nuance-468118-g4",
        )

        with self.assertRaisesRegex(
            EvidenceLoadError,
            "No monitoring run was found",
        ):
            loader.load_latest_status_evidence(
                job_name="olist-dbt-build-job",
                environment="prod",
            )

    def test_invalid_project_identifier_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            EvidenceLoadError,
            "unsupported characters",
        ):
            BigQueryEvidenceLoader(
                executor=self.executor,
                project_id="bad.project`",
            )

    def test_service_loads_realistic_rows_and_executes_r001_to_r003(self) -> None:
        catalog = load_rule_catalog(REVIEWER_ROOT / "config" / "rule_catalog.yml")
        service = StatusReviewService(
            loader=self.loader,
            evaluator=DeterministicEvaluator(catalog),
        )

        review = service.review_run("run-001")
        by_rule = {evaluation.rule_id: evaluation for evaluation in review.evaluations}

        self.assertEqual(by_rule["M9-R001"].result, "PASS")
        self.assertEqual(by_rule["M9-R002"].result, "TRIGGERED")
        self.assertEqual(by_rule["M9-R002"].severity, "HIGH")
        self.assertEqual(by_rule["M9-R003"].result, "TRIGGERED")
        self.assertEqual(by_rule["M9-R003"].severity, "MEDIUM")


class HistoricalFakeQueryExecutor(FakeQueryExecutor):
    def execute(
        self,
        sql: str,
        parameters: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        self.calls.append((sql, dict(parameters)))

        if "WITH selected_run AS" in sql:
            return [
                {
                    "monitoring_run_id": "run-004",
                    "job_name": "olist-dbt-build-job",
                    "environment": "prod",
                    "generated_at": "2026-08-04T03:00:00Z",
                    "status": "success",
                },
                {
                    "monitoring_run_id": "run-003",
                    "job_name": "olist-dbt-build-job",
                    "environment": "prod",
                    "generated_at": "2026-08-03T03:00:00Z",
                    "status": "success",
                },
            ]

        if "model_metadata_snapshots" in sql and "monitoring_run_ids" in parameters:
            return [
                {
                    "monitoring_run_id": "run-004",
                    "unique_id": "model.olist.fct_orders",
                    "model_name": "fct_orders",
                    "row_count": 950,
                },
                {
                    "monitoring_run_id": "run-003",
                    "unique_id": "model.olist.fct_orders",
                    "model_name": "fct_orders",
                    "row_count": 900,
                },
            ]

        if "model_run_results" in sql and "monitoring_run_ids" in parameters:
            return [
                {
                    "monitoring_run_id": "run-004",
                    "unique_id": "model.olist.fct_orders",
                    "model_name": "fct_orders",
                    "status": "success",
                    "execution_time_seconds": 8.0,
                },
                {
                    "monitoring_run_id": "run-003",
                    "unique_id": "model.olist.fct_orders",
                    "model_name": "fct_orders",
                    "status": "success",
                    "execution_time_seconds": 7.0,
                },
            ]

        # Avoid recording the same delegated call twice.
        self.calls.pop()

        return super().execute(
            sql,
            parameters,
        )


class HistoricalEvidenceLoaderTests(unittest.TestCase):
    def test_comparable_run_ids_preserve_newest_first_order(
        self,
    ) -> None:
        executor = HistoricalFakeQueryExecutor()

        loader = BigQueryEvidenceLoader(
            executor=executor,
            project_id="balmy-nuance-468118-g4",
            dataset_id="olist_monitoring",
        )

        bundle = loader.load_status_evidence("run-005")

        self.assertEqual(
            bundle.comparable_run_ids,
            (
                "run-004",
                "run-003",
            ),
        )

    def test_historical_metadata_rows_are_loaded(
        self,
    ) -> None:
        executor = HistoricalFakeQueryExecutor()

        loader = BigQueryEvidenceLoader(
            executor=executor,
            project_id="balmy-nuance-468118-g4",
            dataset_id="olist_monitoring",
        )

        bundle = loader.load_status_evidence("run-005")

        metadata_rows = bundle.evidence["model_metadata_snapshots"]

        run_ids = {row["monitoring_run_id"] for row in metadata_rows}

        self.assertEqual(
            run_ids,
            {
                "run-005",
                "run-004",
                "run-003",
            },
        )

    def test_historical_model_execution_rows_are_loaded(
        self,
    ) -> None:
        executor = HistoricalFakeQueryExecutor()

        loader = BigQueryEvidenceLoader(
            executor=executor,
            project_id="balmy-nuance-468118-g4",
            dataset_id="olist_monitoring",
        )

        bundle = loader.load_status_evidence("run-005")

        model_rows = bundle.evidence["model_run_results"]

        run_ids = {row["monitoring_run_id"] for row in model_rows}

        self.assertEqual(
            run_ids,
            {
                "run-005",
                "run-004",
                "run-003",
            },
        )

    def test_comparable_run_query_uses_required_scope(
        self,
    ) -> None:
        executor = HistoricalFakeQueryExecutor()

        loader = BigQueryEvidenceLoader(
            executor=executor,
            project_id="balmy-nuance-468118-g4",
            dataset_id="olist_monitoring",
        )

        loader.load_status_evidence("run-005")

        comparable_calls = [
            (sql, parameters)
            for sql, parameters in executor.calls
            if "WITH selected_run AS" in sql
        ]

        self.assertEqual(
            len(comparable_calls),
            1,
        )

        sql, parameters = comparable_calls[0]

        self.assertIn(
            "candidate.job_name = selected_run.job_name",
            sql,
        )
        self.assertIn(
            "candidate.environment = selected_run.environment",
            sql,
        )
        self.assertIn(
            "LOWER(TRIM(candidate.status)) = 'success'",
            sql,
        )
        self.assertIn(
            "candidate.generated_at < selected_run.generated_at",
            sql,
        )
        self.assertIn(
            "candidate.generated_at DESC",
            sql,
        )

        self.assertEqual(
            parameters,
            {
                "monitoring_run_id": "run-005",
                "history_limit": 5,
            },
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

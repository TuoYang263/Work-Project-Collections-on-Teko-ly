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

        if "LIMIT 1" in sql:
            return [{"monitoring_run_id": "run-002"}]

        if "pipeline_runs" in sql:
            return [
                {
                    "monitoring_run_id": parameters["monitoring_run_id"],
                    "job_name": "olist-dbt-build-job",
                    "environment": "prod",
                    "status": "success",
                }
            ]

        if "model_run_results" in sql:
            return [
                {
                    "monitoring_run_id": parameters["monitoring_run_id"],
                    "unique_id": "model.olist.fct_orders",
                    "model_name": "fct_orders",
                    "status": "error",
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

    def test_load_explicit_run_returns_three_evidence_sources(self) -> None:
        bundle = self.loader.load_status_evidence("run-001")

        self.assertEqual(bundle.monitoring_run_id, "run-001")
        self.assertEqual(
            set(bundle.evidence),
            {
                "pipeline_runs",
                "model_run_results",
                "test_run_results",
            },
        )
        self.assertEqual(len(bundle.evidence["pipeline_runs"]), 1)
        self.assertEqual(len(bundle.evidence["model_run_results"]), 1)
        self.assertEqual(len(bundle.evidence["test_run_results"]), 1)

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
        catalog = load_rule_catalog(
            REVIEWER_ROOT / "config" / "rule_catalog.yml"
        )
        service = StatusReviewService(
            loader=self.loader,
            evaluator=DeterministicEvaluator(catalog),
        )

        review = service.review_run("run-001")
        by_rule = {
            evaluation.rule_id: evaluation
            for evaluation in review.evaluations
        }

        self.assertEqual(by_rule["M9-R001"].result, "PASS")
        self.assertEqual(by_rule["M9-R002"].result, "TRIGGERED")
        self.assertEqual(by_rule["M9-R002"].severity, "HIGH")
        self.assertEqual(by_rule["M9-R003"].result, "TRIGGERED")
        self.assertEqual(by_rule["M9-R003"].severity, "MEDIUM")


if __name__ == "__main__":
    unittest.main(verbosity=2)

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REVIEWER_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REVIEWER_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from pipeline_reviewer import (  # noqa: E402
    DeterministicEvaluator,
    load_rule_catalog,
)


class HistoricalRuleEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        catalog = load_rule_catalog(REVIEWER_ROOT / "config" / "rule_catalog.yml")
        self.evaluator = DeterministicEvaluator(catalog)

    @staticmethod
    def _pipeline_row(
        run_id: str,
        status: str = "success",
    ) -> dict[str, object]:
        return {
            "monitoring_run_id": run_id,
            "job_name": "olist-dbt-build-job",
            "environment": "prod",
            "generated_at": "2026-08-01T03:00:00Z",
            "status": status,
        }

    @staticmethod
    def _model_row(
        run_id: str,
        unique_id: str,
    ) -> dict[str, object]:
        return {
            "monitoring_run_id": run_id,
            "unique_id": unique_id,
            "model_name": unique_id.split(".")[-1],
            "resource_type": "model",
        }

    @staticmethod
    def _r004(evaluations):
        return [
            evaluation for evaluation in evaluations if evaluation.rule_id == "M9-R004"
        ]

    def test_same_model_inventory_passes(self) -> None:
        evidence = {
            "pipeline_runs": [
                self._pipeline_row("run-current"),
                self._pipeline_row("run-baseline"),
            ],
            "model_metadata_snapshots": [
                self._model_row(
                    "run-current",
                    "model.olist.dim_customers",
                ),
                self._model_row(
                    "run-current",
                    "model.olist.fct_orders",
                ),
                self._model_row(
                    "run-baseline",
                    "model.olist.dim_customers",
                ),
                self._model_row(
                    "run-baseline",
                    "model.olist.fct_orders",
                ),
            ],
        }

        evaluations = self.evaluator.evaluate_historical_rules(
            selected_run_id="run-current",
            comparable_run_ids=("run-baseline",),
            evidence=evidence,
        )

        r004_evaluations = self._r004(evaluations)

        self.assertEqual(len(r004_evaluations), 2)

        self.assertTrue(
            all(evaluation.result == "PASS" for evaluation in r004_evaluations)
        )

    def test_missing_model_triggers_medium(self) -> None:
        evidence = {
            "pipeline_runs": [
                self._pipeline_row("run-current"),
                self._pipeline_row("run-baseline"),
            ],
            "model_metadata_snapshots": [
                self._model_row(
                    "run-current",
                    "model.olist.dim_customers",
                ),
                self._model_row(
                    "run-baseline",
                    "model.olist.dim_customers",
                ),
                self._model_row(
                    "run-baseline",
                    "model.olist.fct_orders",
                ),
            ],
        }

        evaluations = self.evaluator.evaluate_historical_rules(
            selected_run_id="run-current",
            comparable_run_ids=("run-baseline",),
            evidence=evidence,
        )

        r004_evaluations = self._r004(evaluations)

        by_entity = {
            evaluation.entity_id: evaluation for evaluation in r004_evaluations
        }

        self.assertEqual(
            by_entity["model.olist.dim_customers"].result,
            "PASS",
        )

        missing = by_entity["model.olist.fct_orders"]

        self.assertEqual(
            missing.result,
            "TRIGGERED",
        )

        self.assertEqual(
            missing.severity,
            "MEDIUM",
        )

    def test_no_comparable_run_is_not_evaluated(
        self,
    ) -> None:
        evaluations = self.evaluator.evaluate_historical_rules(
            selected_run_id="run-current",
            comparable_run_ids=(),
            evidence={},
        )

        r004_evaluations = self._r004(evaluations)

        self.assertEqual(
            len(r004_evaluations),
            1,
        )

        self.assertEqual(
            r004_evaluations[0].result,
            "NOT_EVALUATED",
        )

    def test_unsuccessful_selected_run_is_not_evaluated(
        self,
    ) -> None:
        evidence = {
            "pipeline_runs": [
                self._pipeline_row(
                    "run-current",
                    status="failed",
                ),
                self._pipeline_row("run-baseline"),
            ],
            "model_metadata_snapshots": [
                self._model_row(
                    "run-current",
                    "model.olist.fct_orders",
                ),
                self._model_row(
                    "run-baseline",
                    "model.olist.fct_orders",
                ),
            ],
        }

        evaluations = self.evaluator.evaluate_historical_rules(
            selected_run_id="run-current",
            comparable_run_ids=("run-baseline",),
            evidence=evidence,
        )

        r004_evaluations = self._r004(evaluations)

        self.assertEqual(
            r004_evaluations[0].result,
            "NOT_EVALUATED",
        )

    def test_missing_selected_inventory_is_not_evaluated(
        self,
    ) -> None:
        evidence = {
            "pipeline_runs": [
                self._pipeline_row("run-current"),
                self._pipeline_row("run-baseline"),
            ],
            "model_metadata_snapshots": [
                self._model_row(
                    "run-baseline",
                    "model.olist.fct_orders",
                ),
            ],
        }

        evaluations = self.evaluator.evaluate_historical_rules(
            selected_run_id="run-current",
            comparable_run_ids=("run-baseline",),
            evidence=evidence,
        )

        r004_evaluations = self._r004(evaluations)

        self.assertEqual(
            r004_evaluations[0].result,
            "NOT_EVALUATED",
        )

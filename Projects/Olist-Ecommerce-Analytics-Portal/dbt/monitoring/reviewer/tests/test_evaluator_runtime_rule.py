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


class RuntimeRegressionRuleEvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        catalog_path = REVIEWER_ROOT / "config" / "rule_catalog.yml"
        cls.catalog = load_rule_catalog(catalog_path)
        cls.evaluator = DeterministicEvaluator(cls.catalog)

    @staticmethod
    def _model_execution(
        run_id: str,
        runtime: int | float | None,
        status: str = "success",
    ) -> dict[str, object]:
        return {
            "monitoring_run_id": run_id,
            "unique_id": "model.olist.fct_orders",
            "model_name": "fct_orders",
            "status": status,
            "execution_time_seconds": runtime,
        }

    @classmethod
    def _evidence(
        cls,
        current_runtime: int | float | None,
        historical_runtimes: list[int | float | None],
        current_status: str = "success",
    ) -> tuple[
        dict[str, list[dict[str, object]]],
        tuple[str, ...],
    ]:
        selected_run_id = "run-current"
        comparable_run_ids = tuple(
            f"run-history-{index}" for index in range(1, len(historical_runtimes) + 1)
        )

        pipeline_runs = [
            {
                "monitoring_run_id": selected_run_id,
                "status": "success",
            }
        ]
        pipeline_runs.extend(
            {
                "monitoring_run_id": run_id,
                "status": "success",
            }
            for run_id in comparable_run_ids
        )

        model_rows = [
            cls._model_execution(
                selected_run_id,
                current_runtime,
                status=current_status,
            )
        ]
        model_rows.extend(
            cls._model_execution(run_id, runtime)
            for run_id, runtime in zip(
                comparable_run_ids,
                historical_runtimes,
                strict=True,
            )
        )

        return (
            {
                "pipeline_runs": pipeline_runs,
                "model_run_results": model_rows,
                "model_metadata_snapshots": [],
            },
            comparable_run_ids,
        )

    def _r006(
        self,
        current_runtime: int | float | None,
        historical_runtimes: list[int | float | None],
        current_status: str = "success",
    ):
        evidence, comparable_run_ids = self._evidence(
            current_runtime=current_runtime,
            historical_runtimes=historical_runtimes,
            current_status=current_status,
        )

        evaluations = self.evaluator.evaluate_historical_rules(
            selected_run_id="run-current",
            comparable_run_ids=comparable_run_ids,
            evidence=evidence,
        )

        return next(item for item in evaluations if item.rule_id == "M9-R006")

    def test_stable_runtime_passes(self) -> None:
        r006 = self._r006(
            current_runtime=12,
            historical_runtimes=[10, 11, 12],
        )

        self.assertEqual(r006.result, "PASS")
        self.assertIsNone(r006.severity)
        self.assertEqual(
            r006.evidence["baseline_median_execution_time_seconds"],
            11.0,
        )

    def test_material_runtime_increase_triggers_medium(self) -> None:
        r006 = self._r006(
            current_runtime=16,
            historical_runtimes=[10, 10, 10],
        )

        self.assertEqual(r006.result, "TRIGGERED")
        self.assertEqual(r006.severity, "MEDIUM")
        self.assertEqual(
            r006.evidence["absolute_increase_seconds"],
            6.0,
        )
        self.assertAlmostEqual(
            r006.evidence["relative_increase"],
            0.60,
        )

    def test_large_runtime_regression_triggers_high(self) -> None:
        r006 = self._r006(
            current_runtime=60,
            historical_runtimes=[30, 30, 30],
        )

        self.assertEqual(r006.result, "TRIGGERED")
        self.assertEqual(r006.severity, "HIGH")
        self.assertEqual(
            r006.evidence["absolute_increase_seconds"],
            30.0,
        )
        self.assertAlmostEqual(
            r006.evidence["relative_increase"],
            1.0,
        )

    def test_relative_threshold_without_absolute_threshold_passes(
        self,
    ) -> None:
        r006 = self._r006(
            current_runtime=6,
            historical_runtimes=[4, 4, 4],
        )

        self.assertEqual(r006.result, "PASS")
        self.assertIsNone(r006.severity)
        self.assertAlmostEqual(
            r006.evidence["relative_increase"],
            0.50,
        )
        self.assertEqual(
            r006.evidence["absolute_increase_seconds"],
            2.0,
        )

    def test_absolute_threshold_without_relative_threshold_passes(
        self,
    ) -> None:
        r006 = self._r006(
            current_runtime=106,
            historical_runtimes=[100, 100, 100],
        )

        self.assertEqual(r006.result, "PASS")
        self.assertIsNone(r006.severity)
        self.assertAlmostEqual(
            r006.evidence["relative_increase"],
            0.06,
        )
        self.assertEqual(
            r006.evidence["absolute_increase_seconds"],
            6.0,
        )

    def test_faster_runtime_passes(self) -> None:
        r006 = self._r006(
            current_runtime=8,
            historical_runtimes=[10, 10, 10],
        )

        self.assertEqual(r006.result, "PASS")
        self.assertIsNone(r006.severity)
        self.assertLess(
            r006.evidence["absolute_increase_seconds"],
            0,
        )

    def test_zero_baseline_is_not_evaluated(self) -> None:
        r006 = self._r006(
            current_runtime=10,
            historical_runtimes=[0, 0, 0],
        )

        self.assertEqual(r006.result, "NOT_EVALUATED")
        self.assertIsNone(r006.severity)

    def test_unsuccessful_current_model_is_not_evaluated(self) -> None:
        r006 = self._r006(
            current_runtime=20,
            historical_runtimes=[10, 10, 10],
            current_status="error",
        )

        self.assertEqual(r006.result, "NOT_EVALUATED")
        self.assertIsNone(r006.severity)


if __name__ == "__main__":
    unittest.main(verbosity=2)

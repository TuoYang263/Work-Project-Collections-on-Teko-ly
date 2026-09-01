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


class RowCountRuleEvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        catalog_path = REVIEWER_ROOT / "config" / "rule_catalog.yml"
        cls.catalog = load_rule_catalog(catalog_path)
        cls.evaluator = DeterministicEvaluator(cls.catalog)

    @staticmethod
    def _model_row(
        run_id: str,
        row_count: int | float | None,
    ) -> dict[str, object]:
        return {
            "monitoring_run_id": run_id,
            "unique_id": "model.olist.fct_orders",
            "model_name": "fct_orders",
            "resource_type": "model",
            "row_count": row_count,
        }

    @classmethod
    def _evidence(
        cls,
        current_row_count: int | float | None,
        historical_row_counts: list[int | float | None],
    ) -> tuple[
        dict[str, list[dict[str, object]]],
        tuple[str, ...],
    ]:
        selected_run_id = "run-current"
        comparable_run_ids = tuple(
            f"run-history-{index}" for index in range(1, len(historical_row_counts) + 1)
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

        metadata_rows = [
            cls._model_row(
                selected_run_id,
                current_row_count,
            )
        ]

        metadata_rows.extend(
            cls._model_row(run_id, row_count)
            for run_id, row_count in zip(
                comparable_run_ids,
                historical_row_counts,
                strict=True,
            )
        )

        return (
            {
                "pipeline_runs": pipeline_runs,
                "model_metadata_snapshots": metadata_rows,
            },
            comparable_run_ids,
        )

    def _r005(
        self,
        current_row_count: int | float | None,
        historical_row_counts: list[int | float | None],
    ):
        evidence, comparable_run_ids = self._evidence(
            current_row_count=current_row_count,
            historical_row_counts=historical_row_counts,
        )

        evaluations = self.evaluator.evaluate_historical_rules(
            selected_run_id="run-current",
            comparable_run_ids=comparable_run_ids,
            evidence=evidence,
        )

        return next(item for item in evaluations if item.rule_id == "M9-R005")

    def test_stable_row_count_passes(self) -> None:
        r005 = self._r005(
            current_row_count=1050,
            historical_row_counts=[1000, 1020, 980],
        )

        self.assertEqual(r005.result, "PASS")
        self.assertIsNone(r005.severity)
        self.assertEqual(
            r005.evidence["baseline_median_row_count"],
            1000.0,
        )
        self.assertEqual(
            r005.evidence["baseline_sample_size"],
            3,
        )

    def test_material_change_triggers_medium(self) -> None:
        r005 = self._r005(
            current_row_count=1400,
            historical_row_counts=[1000, 1000, 1000],
        )

        self.assertEqual(r005.result, "TRIGGERED")
        self.assertEqual(r005.severity, "MEDIUM")
        self.assertAlmostEqual(
            r005.evidence["relative_change"],
            0.40,
        )
        self.assertEqual(
            r005.evidence["absolute_change"],
            400.0,
        )

    def test_more_than_double_baseline_triggers_high(self) -> None:
        r005 = self._r005(
            current_row_count=2100,
            historical_row_counts=[1000, 1000, 1000],
        )

        self.assertEqual(r005.result, "TRIGGERED")
        self.assertEqual(r005.severity, "HIGH")
        self.assertAlmostEqual(
            r005.evidence["relative_change"],
            1.10,
        )

    def test_relative_threshold_without_absolute_threshold_passes(
        self,
    ) -> None:
        r005 = self._r005(
            current_row_count=270,
            historical_row_counts=[200, 200, 200],
        )

        self.assertEqual(r005.result, "PASS")
        self.assertIsNone(r005.severity)
        self.assertAlmostEqual(
            r005.evidence["relative_change"],
            0.35,
        )
        self.assertEqual(
            r005.evidence["absolute_change"],
            70.0,
        )

    def test_zero_baseline_uses_absolute_threshold(self) -> None:
        r005 = self._r005(
            current_row_count=150,
            historical_row_counts=[0, 0, 0],
        )

        self.assertEqual(r005.result, "TRIGGERED")
        self.assertEqual(r005.severity, "MEDIUM")
        self.assertEqual(
            r005.evidence["baseline_median_row_count"],
            0.0,
        )
        self.assertIsNone(
            r005.evidence["relative_change"],
        )
        self.assertEqual(
            r005.evidence["absolute_change"],
            150.0,
        )

    def test_missing_current_row_count_is_not_evaluated(self) -> None:
        r005 = self._r005(
            current_row_count=None,
            historical_row_counts=[1000, 1000, 1000],
        )

        self.assertEqual(r005.result, "NOT_EVALUATED")
        self.assertIsNone(r005.severity)
        self.assertIn(
            "row_count",
            r005.reason,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
